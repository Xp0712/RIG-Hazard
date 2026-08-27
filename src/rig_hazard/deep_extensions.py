from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .config import resolve_project_path
from .deep_data import DeepCacheBatchSource
from .deep_models import DeepHazardModel
from .deep_training import (
    WeightedTrajectoryCalibrator,
    _device_from_name,
    collect_trajectory_arrays,
    evaluate_model_nll,
    set_reproducible_seed,
    train_epoch,
    trajectory_probability_rows,
)
from .preprocessing import prepare_output_root, write_json
from .torch_runtime import torch


def extension_variant_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = config["extension_experiment"]["variants"]
    required = {"masked_features", "negative_sampling", "hierarchical_barrier"}
    result: dict[str, dict[str, Any]] = {}
    for name, payload in values.items():
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Extension variant {name} is missing fields: {sorted(missing)}")
        if payload["negative_sampling"] not in {"hard_negative_enriched", "uniform_negative"}:
            raise ValueError(f"Unsupported negative sampling for {name}: {payload['negative_sampling']}")
        result[str(name)] = dict(payload)
    return result


def cache_index_cardinalities(config: dict[str, Any]) -> tuple[int, int]:
    catalog = pd.read_csv(
        resolve_project_path(config["preprocessed_root"]) / "station_catalog.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    stations = int(pd.to_numeric(catalog["station_index"], errors="coerce").max()) + 1
    cities = int(pd.to_numeric(catalog["city_index"], errors="coerce").max()) + 1
    return stations, cities


def build_extension_model(
    variant_name: str,
    spec: dict[str, Any],
    contract: dict[str, Any],
    config: dict[str, Any],
) -> DeepHazardModel:
    feature_names = [str(value) for value in contract["feature_names"]]
    unknown = sorted(set(spec["masked_features"]).difference(feature_names))
    if unknown:
        raise ValueError(f"Unknown masked features for {variant_name}: {unknown}")
    masked_indices = [feature_names.index(str(name)) for name in spec["masked_features"]]
    station_count, city_count = cache_index_cardinalities(config)
    model_config = config["models"]
    return DeepHazardModel(
        encoder_type=str(config["extension_experiment"].get("encoder", "gru")),
        input_size=int(contract["feature_count"]),
        horizon_steps=int(contract["horizon_steps"]),
        hidden_size=int(model_config["gru_hidden_size"]),
        kernel_size=int(model_config["tcn_kernel_size"]),
        dilations=[int(value) for value in model_config["tcn_dilations"]],
        dropout=float(model_config["dropout"]),
        horizon_embedding_dim=int(model_config["horizon_embedding_dim"]),
        initial_log_rate=float(model_config.get("initial_log_rate", -9.0)),
        masked_feature_indices=masked_indices,
        number_stations=station_count if bool(spec["hierarchical_barrier"]) else 0,
        number_cities=city_count if bool(spec["hierarchical_barrier"]) else 0,
    )


def _training_source(
    cache_root: Path,
    spec: dict[str, Any],
    maximum_train_samples: int | None,
    seed: int,
) -> DeepCacheBatchSource:
    if spec["negative_sampling"] == "hard_negative_enriched":
        return DeepCacheBatchSource(cache_root, "train", maximum_train_samples, seed)
    cache_summary = json.loads((cache_root / "cache_summary.json").read_text(encoding="utf-8"))
    target_rows = int(cache_summary["splits"]["train"]["rows"])
    if maximum_train_samples is not None:
        target_rows = min(target_rows, int(maximum_train_samples))
    return DeepCacheBatchSource(
        cache_root,
        "train_full",
        target_rows,
        seed,
        sampling_strategy="uniform_negative",
    )


def _barrier_effect_rows(
    model: DeepHazardModel,
    variant_name: str,
    seed: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not model.has_hierarchical_barrier:
        return []
    catalog = pd.read_csv(
        resolve_project_path(config["preprocessed_root"]) / "station_catalog.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    city_weights, station_weights = model._centered_barrier_weights()
    city_values = city_weights.detach().cpu().numpy()
    station_values = station_weights.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for city_index, city_frame in catalog.groupby("city_index", sort=True):
        effect = float(city_values[int(city_index)])
        rows.append(
            {
                "variant_name": variant_name,
                "seed": seed,
                "level": "city",
                "index": int(city_index),
                "identifier": str(city_frame["city"].iloc[0]),
                "log_susceptibility_effect": effect,
                "barrier_value": -effect,
                "hazard_ratio": float(np.exp(effect)),
            }
        )
    for row in catalog.itertuples(index=False):
        effect = float(station_values[int(row.station_index)])
        rows.append(
            {
                "variant_name": variant_name,
                "seed": seed,
                "level": "station",
                "index": int(row.station_index),
                "identifier": str(row.station_code),
                "station_name": str(row.station_name),
                "city": str(row.city),
                "log_susceptibility_effect": effect,
                "barrier_value": -effect,
                "hazard_ratio": float(np.exp(effect)),
            }
        )
    return rows


def train_extension_variant(
    variant_name: str,
    spec: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    output_root: Path,
    device: torch.device,
    maximum_epochs: int,
    maximum_train_samples: int | None,
    maximum_validation_samples: int | None,
    resume_partial: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cache_root = resolve_project_path(config["cache_root"])
    training = config["training"]
    extension = config["extension_experiment"]
    batch_size = int(training["batch_size"])
    early_limit = int(training["early_stopping_validation_samples"])
    metric_limit = int(training["development_metric_samples"])
    if maximum_validation_samples is not None:
        early_limit = min(early_limit, int(maximum_validation_samples))
        metric_limit = min(metric_limit, int(maximum_validation_samples))
    train_source = _training_source(cache_root, spec, maximum_train_samples, seed)
    early_source = DeepCacheBatchSource(
        cache_root, "validation", early_limit, int(config["sampling"]["seed"]) + 101
    )
    metric_source = DeepCacheBatchSource(
        cache_root, "validation", metric_limit, int(config["sampling"]["seed"]) + 202
    )
    set_reproducible_seed(seed)
    model = build_extension_model(variant_name, spec, train_source.contract, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    city_l2 = float(extension["barrier_regularization"]["city_l2"]) if model.has_hierarchical_barrier else 0.0
    station_l2 = (
        float(extension["barrier_regularization"]["station_l2"])
        if model.has_hierarchical_barrier
        else 0.0
    )
    patience = int(training["early_stopping_patience"])
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    progress_path = output_root / "in_progress" / f"{variant_name}_seed_{seed}.pt"
    start_epoch = 1
    if resume_partial and progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=True)
        model.load_state_dict(progress["state_dict"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        scheduler.load_state_dict(progress["scheduler_state_dict"])
        best_loss = float(progress["best_loss"])
        best_epoch = int(progress["best_epoch"])
        best_state = progress["best_state"]
        epochs_without_improvement = int(progress["epochs_without_improvement"])
        history_rows = list(progress["history_rows"])
        start_epoch = int(progress["completed_epoch"]) + 1
        if "torch_rng_state" in progress:
            torch.set_rng_state(progress["torch_rng_state"].cpu())
        print(
            f"Resumed partial extension run: {variant_name} seed={seed} from epoch={start_epoch}",
            flush=True,
        )
    for epoch in range(start_epoch, maximum_epochs + 1):
        training_loss = train_epoch(
            model,
            train_source,
            optimizer,
            device,
            batch_size,
            seed + epoch * 1009,
            float(training["gradient_clip_norm"]),
            float(training["hard_negative_loss_multiplier"]),
            city_l2,
            station_l2,
        )
        validation_loss = evaluate_model_nll(model, early_source, device, batch_size)
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1e-8
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history_rows.append(
            {
                "variant_name": variant_name,
                "seed": seed,
                "epoch": epoch,
                "training_nll_with_regularization": training_loss,
                "early_stopping_nll": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_so_far": int(improved),
            }
        )
        print(
            f"{variant_name} seed={seed} epoch={epoch}/{maximum_epochs} "
            f"train_nll={training_loss:.8f} development_nll={validation_loss:.8f}",
            flush=True,
        )
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_progress = progress_path.with_suffix(".tmp")
        torch.save(
            {
                "variant_name": variant_name,
                "seed": seed,
                "completed_epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "epochs_without_improvement": epochs_without_improvement,
                "history_rows": history_rows,
                "torch_rng_state": torch.get_rng_state(),
            },
            temporary_progress,
        )
        temporary_progress.replace(progress_path)
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"Extension variant did not produce a checkpoint: {variant_name}")
    model.load_state_dict(best_state)
    arrays = collect_trajectory_arrays(model, metric_source, device, batch_size)
    calibrator = WeightedTrajectoryCalibrator().fit(
        arrays["eta"], arrays["target"], arrays["risk_mask"], arrays["sample_weight"]
    )
    metric_rows = trajectory_probability_rows(
        variant_name,
        arrays["eta"],
        arrays["target"],
        arrays["risk_mask"],
        arrays["sample_weight"],
        int(config["step_minutes"]),
        "raw",
    )
    metric_rows.extend(
        trajectory_probability_rows(
            variant_name,
            calibrator.transform_eta(arrays["eta"]),
            arrays["target"],
            arrays["risk_mask"],
            arrays["sample_weight"],
            int(config["step_minutes"]),
            "calibrated",
        )
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    for row in metric_rows:
        row.update({"seed": seed, "parameters": parameter_count, "best_epoch": best_epoch})
    checkpoint = {
        "model_family": "Deep RIG-Hazard staged extension",
        "model_name": variant_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_early_stopping_nll": best_loss,
        "model_config": model.to_config(),
        "state_dict": best_state,
        "trajectory_calibrator": calibrator.to_dict(),
        "variant_spec": spec,
        "barrier_regularization": {"city_l2": city_l2, "station_l2": station_l2},
        "sample_contract": train_source.contract,
        "training_selection": train_source.selection_summary,
        "early_stopping_selection": early_source.selection_summary,
        "development_metric_selection": metric_source.selection_summary,
    }
    checkpoint_path = output_root / "checkpoints" / f"{variant_name}_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        "variant_name": variant_name,
        "seed": seed,
        "checkpoint": str(checkpoint_path.relative_to(output_root)),
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_early_stopping_nll": best_loss,
        "calibrator": calibrator.to_dict(),
        "variant_spec": spec,
        "barrier_regularization": {"city_l2": city_l2, "station_l2": station_l2},
        "training_selection": train_source.selection_summary,
    }
    progress_path.unlink(missing_ok=True)
    return metric_rows, history_rows, metadata, _barrier_effect_rows(model, variant_name, seed, config)


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = ["pr_auc", "roc_auc", "brier_score", "brier_skill", "log_loss", "ece"]
    summary = (
        metrics.groupby(["model_name", "calibration", "horizon"], as_index=False)[numeric]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(value).rstrip("_") for value in summary.columns.to_flat_index()]
    return summary


def paired_seed_metric_differences(
    metrics: pd.DataFrame,
    control_name: str = "gru_recurrent_control",
) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["calibration"].eq("calibrated") & metrics["horizon"].eq("6h")
    ].copy()
    control = selected.loc[selected["model_name"].eq(control_name)].set_index("seed")
    directions = {
        "pr_auc": "higher",
        "brier_score": "lower",
        "log_loss": "lower",
        "ece": "lower",
    }
    rows: list[dict[str, Any]] = []
    for model_name, model_frame in selected.groupby("model_name", sort=False):
        if model_name == control_name:
            continue
        model = model_frame.set_index("seed")
        shared = control.index.intersection(model.index)
        for metric, favorable in directions.items():
            differences = (model.loc[shared, metric] - control.loc[shared, metric]).to_numpy(dtype=np.float64)
            count = int(differences.size)
            mean = float(differences.mean()) if count else float("nan")
            standard_deviation = float(differences.std(ddof=1)) if count > 1 else float("nan")
            half_width = (
                float(student_t.ppf(0.975, count - 1) * standard_deviation / np.sqrt(count))
                if count > 1
                else float("nan")
            )
            improved = differences > 0 if favorable == "higher" else differences < 0
            rows.append(
                {
                    "variant_name": model_name,
                    "control_name": control_name,
                    "metric": metric,
                    "favorable_direction": favorable,
                    "paired_seeds": count,
                    "mean_difference_variant_minus_control": mean,
                    "standard_deviation": standard_deviation,
                    "ci95_low": mean - half_width if np.isfinite(half_width) else float("nan"),
                    "ci95_high": mean + half_width if np.isfinite(half_width) else float("nan"),
                    "improved_seed_count": int(improved.sum()),
                }
            )
    return pd.DataFrame(rows)


def build_extension_report(
    run_tier: str,
    device: torch.device,
    metrics: pd.DataFrame,
    barrier_effects: pd.DataFrame,
    config: dict[str, Any],
    summary: pd.DataFrame,
    paired_differences: pd.DataFrame,
) -> str:
    selected = metrics.loc[
        metrics["horizon"].eq("6h") & metrics["calibration"].eq("calibrated"),
        ["model_name", "seed", "pr_auc", "brier_score", "log_loss", "ece", "roc_auc"],
    ]
    variant_rows = []
    for name, spec in extension_variant_specs(config).items():
        variant_rows.append(
            {
                "variant_name": name,
                "复发史输入": "移除" if spec["masked_features"] else "保留",
                "负样本策略": spec["negative_sampling"],
                "层次屏障": "是" if spec["hierarchical_barrier"] else "否",
            }
        )
    barrier_note = (
        f"已输出{barrier_effects.shape[0]}条城市/站点屏障效应。"
        if not barrier_effects.empty
        else "本次运行未包含层次屏障变体。"
    )
    return "\n".join(
        [
            "# Deep RIG-Hazard复发史与层次屏障实验",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"运行层级：{run_tier}",
            f"计算设备：{device}",
            "",
            "## 变体口径",
            "",
            pd.DataFrame(variant_rows).to_markdown(index=False),
            "",
            "- 去复发史变体只屏蔽距上次覆冰时间及其缺失标记，并从头重训；风险段编号不作为模型输入。",
            "- 普通负样本变体保留全部临近事件窗口，在其余负样本中等概率抽样，并用逆概率权重恢复总体目标。",
            "- 层次屏障是跨36个未来区间共享的城市与站点加性log-hazard效应，分别向0收缩。",
            "",
            "## 6小时开发集概率结果",
            "",
            selected.to_markdown(index=False) if not selected.empty else "无。",
            "",
            "## 五种子汇总",
            "",
            summary.loc[
                summary["calibration"].eq("calibrated") & summary["horizon"].eq("6h")
            ].to_markdown(index=False),
            "",
            "## 相对严格对照的配对种子差异",
            "",
            paired_differences.to_markdown(index=False),
            "",
            "## 屏障输出",
            "",
            barrier_note,
            "",
            "## 解释边界",
            "",
            "本阶段仍只使用2022训练与2023开发。概率门控通过后还需在完整时序上进行同误报事件比较；2024不参与变体或正则选择。",
        ]
    ) + "\n"


def run_deep_extension_experiment(
    config: dict[str, Any],
    config_path: Path,
    overwrite: bool = False,
    resume: bool = False,
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    maximum_epochs: int | None = None,
    maximum_train_samples: int | None = None,
    maximum_validation_samples: int | None = None,
    output_root_override: str | None = None,
    device_name: str = "auto",
) -> Path:
    output_root = (
        resolve_project_path(output_root_override)
        if output_root_override
        else resolve_project_path(config["extension_output_root"])
    )
    if resume:
        if overwrite:
            raise ValueError("--resume and --overwrite cannot be used together")
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        prepare_output_root(output_root, overwrite)
    all_specs = extension_variant_specs(config)
    selected_variants = variants or list(all_specs)
    unknown = sorted(set(selected_variants).difference(all_specs))
    if unknown:
        raise ValueError(f"Unknown extension variants: {unknown}")
    selected_seeds = seeds or [int(value) for value in config["training"]["seeds"]]
    epochs = int(maximum_epochs or config["training"]["maximum_epochs"])
    torch.set_num_threads(max(int(config["training"].get("torch_num_threads", 1)), 1))
    device = _device_from_name(device_name)
    run_tier = (
        "recurrent-feature and hierarchical-barrier development run"
        if maximum_train_samples is None
        and maximum_validation_samples is None
        and len(selected_seeds) >= 5
        else "staged development check"
    )
    write_json(
        output_root / "resolved_config.json",
        {
            **config,
            "config_path": str(config_path),
            "run_overrides": {
                "variants": selected_variants,
                "seeds": selected_seeds,
                "maximum_epochs": epochs,
                "maximum_train_samples": maximum_train_samples,
                "maximum_validation_samples": maximum_validation_samples,
                "device": str(device),
                "run_tier": run_tier,
            },
        },
    )
    metrics: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    barrier_rows: list[dict[str, Any]] = []
    for variant_name in selected_variants:
        for seed in selected_seeds:
            completed_root = output_root / "completed_runs" / f"{variant_name}_seed_{seed}"
            files = {
                "metrics": completed_root / "metrics.csv",
                "history": completed_root / "training_history.csv",
                "metadata": completed_root / "metadata.json",
                "barriers": completed_root / "barrier_effects.csv",
            }
            if resume and all(files[key].exists() for key in ["metrics", "history", "metadata"]):
                metrics.extend(pd.read_csv(files["metrics"], encoding="utf-8-sig").to_dict("records"))
                histories.extend(pd.read_csv(files["history"], encoding="utf-8-sig").to_dict("records"))
                metadata.append(json.loads(files["metadata"].read_text(encoding="utf-8")))
                if files["barriers"].exists():
                    barrier_rows.extend(pd.read_csv(files["barriers"], encoding="utf-8-sig").to_dict("records"))
                print(f"Reused completed extension run: {variant_name} seed={seed}", flush=True)
                continue
            run_metrics, run_history, run_metadata, run_barriers = train_extension_variant(
                variant_name,
                all_specs[variant_name],
                int(seed),
                config,
                output_root,
                device,
                epochs,
                maximum_train_samples,
                maximum_validation_samples,
                resume,
            )
            metrics.extend(run_metrics)
            histories.extend(run_history)
            metadata.append(run_metadata)
            barrier_rows.extend(run_barriers)
            completed_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(run_metrics).to_csv(files["metrics"], index=False, encoding="utf-8-sig")
            pd.DataFrame(run_history).to_csv(files["history"], index=False, encoding="utf-8-sig")
            write_json(files["metadata"], run_metadata)
            if run_barriers:
                pd.DataFrame(run_barriers).to_csv(files["barriers"], index=False, encoding="utf-8-sig")
    metric_frame = pd.DataFrame(metrics)
    history_frame = pd.DataFrame(histories)
    barrier_frame = pd.DataFrame(barrier_rows)
    metric_frame.to_csv(output_root / "deep_extension_metrics.csv", index=False, encoding="utf-8-sig")
    history_frame.to_csv(output_root / "training_history.csv", index=False, encoding="utf-8-sig")
    summary = _summarize_metrics(metric_frame)
    summary.to_csv(output_root / "deep_extension_summary.csv", index=False, encoding="utf-8-sig")
    paired_differences = paired_seed_metric_differences(metric_frame)
    paired_differences.to_csv(
        output_root / "paired_seed_differences.csv", index=False, encoding="utf-8-sig"
    )
    if not barrier_frame.empty:
        barrier_frame.to_csv(output_root / "barrier_effects.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_root / "run_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_tier": run_tier,
            "device": str(device),
            "torch_version": torch.__version__,
            "variants": metadata,
            "resumed": resume,
        },
    )
    (output_root / "deep_extension_report.md").write_text(
        build_extension_report(
            run_tier,
            device,
            metric_frame,
            barrier_frame,
            config,
            summary,
            paired_differences,
        ),
        encoding="utf-8",
    )
    print(f"Deep extension experiment complete: {output_root}", flush=True)
    return output_root

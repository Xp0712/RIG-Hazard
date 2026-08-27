from __future__ import annotations

import json
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_project_path
from .deep_data import DeepCacheBatchSource
from .deep_training import (
    WeightedTrajectoryCalibrator,
    _device_from_name,
    build_model,
    collect_trajectory_arrays,
    cumulative_probability_at_step,
    load_deep_checkpoint,
    set_reproducible_seed,
    train_epoch,
    train_one_model,
    trajectory_probability_curve_rows,
    trajectory_probability_matrices,
    trajectory_probability_rows,
)
from .torch_runtime import torch


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _concatenate_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("At least one trajectory array is required")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def _prediction_payload(
    arrays: dict[str, np.ndarray],
    calibrator: WeightedTrajectoryCalibrator,
    order: np.ndarray,
    save_full_trajectory: bool,
) -> dict[str, np.ndarray]:
    risk = cumulative_probability_at_step(
        arrays["eta"], arrays["eta"].shape[1],
        eta_shift=calibrator.log_rate_shift, eta_slope=calibrator.slope,
    )
    event = arrays["target"].max(axis=1) > 0.5
    observed = event | (arrays["risk_mask"][:, -1] > 0.5)
    file_id = arrays["file_id"][order]
    row_index = arrays["row_index"][order]
    payload: dict[str, np.ndarray] = {
        "origin_id": (
            file_id.astype(np.int64) * np.int64(1 << 32)
            + row_index.astype(np.int64)
        ),
        "file_id": file_id,
        "row_index": row_index,
        "issue_time_ns": arrays["issue_time_ns"][order],
        "risk_6h": risk[order].astype(np.float32),
        "onset_within_6h": event[order].astype(np.int8),
        "observed_6h": observed[order].astype(np.int8),
    }
    if "sample_weight" in arrays:
        payload["sample_weight"] = arrays["sample_weight"][order].astype(np.float64)
    if "rec_gate" in arrays:
        payload["rec_gate"] = arrays["rec_gate"][order].astype(np.float32)
    if save_full_trajectory:
        hazard, cumulative = trajectory_probability_matrices(
            arrays["eta"], calibrator.log_rate_shift, calibrator.slope
        )
        payload.update(
            {
                "h_trajectory": hazard[order].astype(np.float32),
                "F_trajectory": cumulative[order].astype(np.float32),
                "y_hazard": arrays["target"][order].astype(np.int8),
                "censor_mask": arrays["risk_mask"][order].astype(np.int8),
            }
        )
    return payload


def _variant_source_options(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    variant = dict(config.get("protocol_model_variants", {}).get(model_name, {}))
    contract = json.loads(
        (resolve_project_path(config["cache_root"]) / "sample_contract.json").read_text(encoding="utf-8")
    )
    names = list(contract["feature_names"])
    return {
        "sampling_strategy": str(variant.get("negative_sampling", "stratified")),
        "permuted_feature_indices": [names.index(name) for name in variant.get("permuted_features", [])],
        "permutation_block_days": int(variant.get("permutation_block_days", 7)),
    }


def _fixed_epoch_fit(
    model_name: str,
    seed: int,
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
    maximum_train_samples: int | None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    cache_root = resolve_project_path(config["cache_root"])
    source = DeepCacheBatchSource(
        cache_root, "selection_2022", maximum_train_samples, seed,
        **_variant_source_options(model_name, config),
    )
    feature_count = int(source.contract["feature_count"])
    horizon_steps = int(source.contract["horizon_steps"])
    set_reproducible_seed(seed)
    model = build_model(
        model_name,
        feature_count,
        horizon_steps,
        config,
        feature_names=list(source.contract["feature_names"]),
    ).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        loss = train_epoch(
            model,
            source,
            optimizer,
            device,
            int(training["batch_size"]),
            seed + epoch * 1009,
            float(training["gradient_clip_norm"]),
            float(training["hard_negative_loss_multiplier"]),
        )
        history.append(
            {
                "model_name": model_name,
                "seed": seed,
                "stage": "fixed_2022_refit",
                "epoch": epoch,
                "training_nll": loss,
            }
        )
        print(
            f"Fixed 2022 fit: {model_name} seed={seed} epoch={epoch}/{epochs} nll={loss:.8f}",
            flush=True,
        )
    return model, history, source.contract


def _evaluate_frozen_model(
    model: Any,
    calibrator: WeightedTrajectoryCalibrator,
    model_name: str,
    seed: int,
    split: str,
    display_split: str,
    fixed_epochs: int,
    config: dict[str, Any],
    device: torch.device,
    maximum_samples: int | None,
    prediction_path: Path | None = None,
) -> list[dict[str, Any]]:
    cache_root = resolve_project_path(config["cache_root"])
    source = DeepCacheBatchSource(
        cache_root,
        split,
        maximum_samples,
        int(config["sampling"]["seed"]) + seed + (2023 if "2023" in split else 2024),
        **_variant_source_options(model_name, config),
    )
    arrays = collect_trajectory_arrays(
        model,
        source,
        device,
        int(config["training"]["batch_size"]),
        include_current_features=False,
        include_metadata=True,
    )
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        order = np.lexsort((arrays["row_index"], arrays["file_id"]))
        np.savez_compressed(
            prediction_path,
            **_prediction_payload(
                arrays,
                calibrator,
                order,
                bool(
                    config.get("prediction_contract", {}).get(
                        "save_full_trajectory", False
                    )
                ),
            ),
        )
    rows = trajectory_probability_rows(
        f"{model_name}_hazard",
        arrays["eta"],
        arrays["target"],
        arrays["risk_mask"],
        arrays["sample_weight"],
        int(config["step_minutes"]),
        "raw",
    )
    rows.extend(
        trajectory_probability_rows(
            f"{model_name}_hazard",
            arrays["eta"],
            arrays["target"],
            arrays["risk_mask"],
            arrays["sample_weight"],
            int(config["step_minutes"]),
            "calibrated_2022_oof",
            eta_shift=calibrator.log_rate_shift,
            eta_slope=calibrator.slope,
        )
    )
    if bool(config.get("evaluation", {}).get("save_full_curve_metrics", False)):
        rows.extend(
            trajectory_probability_curve_rows(
                f"{model_name}_hazard",
                arrays["eta"],
                arrays["target"],
                arrays["risk_mask"],
                arrays["sample_weight"],
                int(config["step_minutes"]),
                "calibrated_2022_oof_curve",
                eta_shift=calibrator.log_rate_shift,
                eta_slope=calibrator.slope,
            )
        )
    for row in rows:
        row.update(
            {
                "encoder": model_name,
                "seed": seed,
                "split": display_split,
                "fixed_epochs": fixed_epochs,
            }
        )
    return rows


def run_deep_temporal_protocol(
    config: dict[str, Any],
    config_path: Path,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    number_folds: int = 5,
    maximum_epochs: int | None = None,
    maximum_train_samples: int | None = None,
    maximum_evaluation_samples: int | None = None,
    output_root_override: str | Path | None = None,
    device_name: str = "auto",
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    if bool(config.get("performance", {}).get("allow_tf32", False)):
        os.environ["RIG_HAZARD_FAST_CUDA"] = "1"
    output_root = resolve_project_path(
        output_root_override
        or config.get(
            "deep_protocol_output_root",
            "results/recurrence_analysis/model_comparison/deep_protocol",
        )
    )
    if resume:
        if overwrite:
            raise ValueError("resume and overwrite cannot be used together")
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        _prepare_output(output_root, overwrite)
    selected_models = models or list(config.get("default_local_models", ["gru", "recurrent_dual"]))
    selected_seeds = seeds or [int(value) for value in config["training"]["seeds"]]
    epoch_limit = int(maximum_epochs or config["training"]["maximum_epochs"])
    request_signature = {
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "models": selected_models,
        "seeds": selected_seeds,
        "folds": int(number_folds),
        "maximum_epochs": epoch_limit,
        "maximum_train_samples": maximum_train_samples,
        "maximum_evaluation_samples": maximum_evaluation_samples,
    }
    request_path = output_root / "protocol_request.json"
    if resume:
        if not request_path.exists():
            raise FileNotFoundError(
                f"Cannot resume without the original protocol request: {request_path}"
            )
        previous_request = json.loads(request_path.read_text(encoding="utf-8"))
        if previous_request != request_signature:
            raise ValueError("Resume request does not match the original protocol configuration")
    else:
        request_path.write_text(json.dumps(request_signature, indent=2), encoding="utf-8")
    device = _device_from_name(device_name)
    torch.set_num_threads(max(int(config["training"].get("torch_num_threads", 1)), 1))
    oof_metric_rows: list[dict[str, Any]] = []
    locked_metric_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for model_name in selected_models:
        for seed in selected_seeds:
            fold_arrays: list[dict[str, np.ndarray]] = []
            best_epochs: list[int] = []
            for fold in range(int(number_folds)):
                fold_root = output_root / "folds" / f"fold{fold}" / f"{model_name}_seed_{seed}"
                fold_root.mkdir(parents=True, exist_ok=True)
                fold_checkpoint = fold_root / "checkpoints" / f"{model_name}_seed_{seed}.pt"
                fold_metrics_path = fold_root / "metrics.csv"
                fold_history_path = fold_root / "training_history.csv"
                reusable = resume and all(
                    path.exists() for path in (fold_checkpoint, fold_metrics_path, fold_history_path)
                )
                if reusable:
                    _, saved = load_deep_checkpoint(fold_checkpoint, str(device))
                    expected_train = f"cv_fold{fold}_train"
                    expected_validation = f"cv_fold{fold}_validation"
                    if (
                        str(saved.get("model_name")) != model_name
                        or int(saved.get("seed", -1)) != int(seed)
                        or str(saved.get("train_split")) != expected_train
                        or str(saved.get("validation_split")) != expected_validation
                    ):
                        raise ValueError(f"Incompatible fold checkpoint for resume: {fold_checkpoint}")
                    metrics = pd.read_csv(fold_metrics_path).to_dict(orient="records")
                    fold_history = pd.read_csv(fold_history_path).to_dict(orient="records")
                    print(f"Reused fold checkpoint: {model_name} seed={seed} fold={fold}", flush=True)
                else:
                    metrics, fold_history, result = train_one_model(
                        model_name,
                        seed,
                        config,
                        fold_root,
                        device,
                        epoch_limit,
                        maximum_train_samples,
                        maximum_evaluation_samples,
                        train_split=f"cv_fold{fold}_train",
                        validation_split=f"cv_fold{fold}_validation",
                    )
                    pd.DataFrame(metrics).to_csv(fold_metrics_path, index=False)
                    pd.DataFrame(fold_history).to_csv(fold_history_path, index=False)
                    del result
                for row in metrics:
                    oof_metric_rows.append(
                        {**row, "fold": fold, "encoder": model_name, "stage": "2022_block_oof"}
                    )
                for row in fold_history:
                    history_rows.append({**row, "fold": fold, "stage": "2022_block_fit"})
                fold_model, fold_checkpoint_payload = load_deep_checkpoint(fold_checkpoint, str(device))
                full_validation = DeepCacheBatchSource(
                    resolve_project_path(config["cache_root"]),
                    f"cv_fold{fold}_validation",
                    **_variant_source_options(model_name, config),
                )
                fold_arrays.append(
                    collect_trajectory_arrays(
                        fold_model,
                        full_validation,
                        device,
                        int(config["training"]["batch_size"]),
                        include_current_features=False,
                        include_metadata=True,
                    )
                )
                best_epochs.append(int(fold_checkpoint_payload["best_epoch"]))
            pooled = _concatenate_arrays(fold_arrays)
            calibrator = WeightedTrajectoryCalibrator().fit(
                pooled["eta"], pooled["target"], pooled["risk_mask"], pooled["sample_weight"]
            )
            pooled_rows = trajectory_probability_rows(
                f"{model_name}_hazard",
                pooled["eta"],
                pooled["target"],
                pooled["risk_mask"],
                pooled["sample_weight"],
                int(config["step_minutes"]),
                "calibrated_2022_oof_pooled",
                eta_shift=calibrator.log_rate_shift,
                eta_slope=calibrator.slope,
            )
            for row in pooled_rows:
                row.update({"fold": "pooled", "encoder": model_name, "seed": seed, "stage": "2022_oof_calibration"})
                oof_metric_rows.append(row)
            if bool(config.get("evaluation", {}).get("save_full_curve_metrics", False)):
                curve_rows = trajectory_probability_curve_rows(
                    f"{model_name}_hazard",
                    pooled["eta"],
                    pooled["target"],
                    pooled["risk_mask"],
                    pooled["sample_weight"],
                    int(config["step_minutes"]),
                    "calibrated_2022_oof_curve",
                    eta_shift=calibrator.log_rate_shift,
                    eta_slope=calibrator.slope,
                )
                for row in curve_rows:
                    row.update(
                        {
                            "fold": "pooled",
                            "encoder": model_name,
                            "seed": seed,
                            "stage": "2022_oof_full_curve",
                        }
                    )
                    oof_metric_rows.append(row)
            order = np.lexsort((pooled["row_index"], pooled["file_id"]))
            oof_path = output_root / "oof_predictions" / f"{model_name}_seed_{seed}.npz"
            oof_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                oof_path,
                **_prediction_payload(
                    pooled,
                    calibrator,
                    order,
                    bool(
                        config.get("prediction_contract", {}).get(
                            "save_full_trajectory", False
                        )
                    ),
                ),
            )
            fixed_epochs = max(1, int(np.median(best_epochs)))
            checkpoint_path = output_root / "trained_models" / "checkpoints" / f"{model_name}_seed_{seed}.pt"
            fixed_history_path = (
                output_root / "trained_models" / "training_history" / f"{model_name}_seed_{seed}.csv"
            )
            reusable_final = resume and checkpoint_path.exists() and fixed_history_path.exists()
            if reusable_final:
                trained_model, checkpoint = load_deep_checkpoint(checkpoint_path, str(device))
                if (
                    str(checkpoint.get("model_name")) != model_name
                    or int(checkpoint.get("seed", -1)) != int(seed)
                    or int(checkpoint.get("fixed_epochs", -1)) != fixed_epochs
                ):
                    raise ValueError(f"Incompatible final checkpoint for resume: {checkpoint_path}")
                fixed_history = pd.read_csv(fixed_history_path).to_dict(orient="records")
                sample_contract = checkpoint["sample_contract"]
                print(f"Reused fixed 2022 checkpoint: {model_name} seed={seed}", flush=True)
            else:
                trained_model, fixed_history, sample_contract = _fixed_epoch_fit(
                    model_name,
                    seed,
                    fixed_epochs,
                    config,
                    device,
                    maximum_train_samples,
                )
                fixed_history_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(fixed_history).to_csv(fixed_history_path, index=False)
            history_rows.extend(fixed_history)
            checkpoint = {
                "model_family": "Frozen recurrent-event discrete hazard encoder",
                "model_name": model_name,
                "seed": seed,
                "fixed_epochs": fixed_epochs,
                "fold_best_epochs": best_epochs,
                "model_config": trained_model.to_config(),
                "state_dict": {name: value.detach().cpu() for name, value in trained_model.state_dict().items()},
                "trajectory_calibrator": calibrator.to_dict(),
                "sample_contract": sample_contract,
                "selection_protocol": "2022 block CV only; 2023 and 2024 excluded from fitting and calibration",
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            checkpoint_rows.append(
                {
                    "encoder": model_name,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.relative_to(output_root)),
                    "fixed_epochs": fixed_epochs,
                    "fold_best_epochs": best_epochs,
                    "parameters": int(sum(parameter.numel() for parameter in trained_model.parameters())),
                }
            )
            locked_metric_rows.extend(
                _evaluate_frozen_model(
                    trained_model,
                    calibrator,
                    model_name,
                    seed,
                    "cross_year_2023",
                    "2023_cross_year",
                    fixed_epochs,
                    config,
                    device,
                    maximum_evaluation_samples,
                    output_root / "locked_predictions" / "2023" / f"{model_name}_seed_{seed}.npz",
                )
            )
            locked_metric_rows.extend(
                _evaluate_frozen_model(
                    trained_model,
                    calibrator,
                    model_name,
                    seed,
                    "final_time_2024",
                    "2024_final_time",
                    fixed_epochs,
                    config,
                    device,
                    maximum_evaluation_samples,
                    output_root / "locked_predictions" / "2024" / f"{model_name}_seed_{seed}.npz",
                )
            )
    oof_metrics = pd.DataFrame(oof_metric_rows)
    locked_metrics = pd.DataFrame(locked_metric_rows)
    oof_metrics.to_csv(output_root / "oof_metrics.csv", index=False)
    locked_metrics.to_csv(output_root / "locked_year_metrics.csv", index=False)
    curve_parts = []
    if "horizon_step" in oof_metrics.columns:
        curve_parts.append(oof_metrics.loc[oof_metrics["horizon_step"].notna()].copy())
    if "horizon_step" in locked_metrics.columns:
        curve_parts.append(locked_metrics.loc[locked_metrics["horizon_step"].notna()].copy())
    if curve_parts:
        pd.concat(curve_parts, ignore_index=True).to_csv(
            output_root / "full_curve_probability_metrics.csv", index=False
        )
    pd.DataFrame(history_rows).to_csv(output_root / "training_history.csv", index=False)
    (output_root / "prediction_contract.json").write_text(
        json.dumps(
            {
                "keys": ["origin_id", "file_id", "row_index", "issue_time_ns"],
                "primary_prediction": "risk_6h",
                "full_trajectory_saved": bool(
                    config.get("prediction_contract", {}).get(
                        "save_full_trajectory", False
                    )
                ),
                "trajectory_fields": [
                    "h_trajectory", "F_trajectory", "y_hazard", "censor_mask"
                ],
                "trajectory_shape": ["rows", int(config["horizon_steps"])],
                "origin_id_definition": "int64(file_id * 2^32 + row_index)",
                "calibration": "One 2022-OOF affine log-rate map shared by all 36 steps",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "config_path": str(config_path),
                "models": selected_models,
                "seeds": selected_seeds,
                "folds": int(number_folds),
                "device": str(device),
                "checkpoints": checkpoint_rows,
                "maximum_train_samples": maximum_train_samples,
                "maximum_evaluation_samples": maximum_evaluation_samples,
                "resume": resume,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = [
        "# Deep temporal protocol report",
        "",
        "2022 block folds determine early stopping, the pooled trajectory calibrator, and the fixed refit epoch count.",
        "Final checkpoints are refitted on 2022 only. Neither 2023 nor 2024 is used for gradient updates, early stopping, calibration, or budget choice.",
        "2023 is the cross-year stability evaluation. 2024 is the frozen final time evaluation.",
        "Because 2024 was previously inspected in this project, describe it as retrospectively locked rather than never observed.",
    ]
    (output_root / "deep_temporal_protocol_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Deep temporal protocol complete: {output_root}", flush=True)
    return output_root

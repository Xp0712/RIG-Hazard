from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_compat import read_artifact_csv
from .baseline_experiment import evaluate_warning_model
from .budget_control import CausalBudgetConfig, apply_causal_budget_by_station
from .config import resolve_project_path
from .deep_data import DeepCacheBatchSource
from .deep_models import cloglog_hazard_probability, cumulative_incidence
from .deep_training import load_deep_checkpoint
from .deep_warning import (
    load_frozen_budget,
    monthly_budget_audit,
    monthly_false_alarm_distribution_audit,
    select_matched_false_alarm_thresholds,
)
from .graph_experiment import cluster_bootstrap_event_comparison, event_alarm_records
from .naming import artifact_name_candidates
from .preprocessing import prepare_output_root, write_json
from .torch_runtime import torch


EXTENSION_VARIANTS = [
    "gru_recurrent_control",
    "gru_no_recurrent_history",
    "gru_uniform_negative",
    "gru_hierarchical_barrier",
]
EXTENSION_SCORE_COLUMNS = ["local_weather_hazard_6h", *[f"{name}_6h" for name in EXTENSION_VARIANTS]]
EXTENSION_PAIRS = [
    ("gru_recurrent_control", "local_weather_hazard"),
    ("gru_no_recurrent_history", "gru_recurrent_control"),
    ("gru_uniform_negative", "gru_recurrent_control"),
    ("gru_hierarchical_barrier", "gru_recurrent_control"),
    ("gru_hierarchical_barrier", "gru_no_recurrent_history"),
]


def load_extension_ensembles(
    extension_root: Path,
    seeds: list[int],
    device_name: str,
) -> dict[str, list[tuple[torch.nn.Module, dict[str, Any]]]]:
    result: dict[str, list[tuple[torch.nn.Module, dict[str, Any]]]] = {}
    for variant in EXTENSION_VARIANTS:
        ensemble: list[tuple[torch.nn.Module, dict[str, Any]]] = []
        for seed in seeds:
            checkpoint_candidates = [
                extension_root / "checkpoints" / f"{name}_seed_{seed}.pt"
                for name in artifact_name_candidates(variant)
            ]
            checkpoint_path = next(
                (path for path in checkpoint_candidates if path.exists()),
                checkpoint_candidates[0],
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing extension checkpoint: {checkpoint_path}")
            model, checkpoint = load_deep_checkpoint(checkpoint_path, device_name)
            ensemble.append((model, checkpoint["trajectory_calibrator"]))
        result[variant] = ensemble
    return result


def calibrated_extension_risk(
    model: torch.nn.Module,
    history: torch.Tensor,
    station_index: torch.Tensor,
    city_index: torch.Tensor,
    calibrator: dict[str, Any],
) -> torch.Tensor:
    eta = model(history, station_index, city_index)
    eta = float(calibrator["log_rate_shift"]) + float(calibrator["slope"]) * eta
    return cumulative_incidence(cloglog_hazard_probability(eta))[:, -1]


def _prediction_destination(root: Path, metadata: dict[str, Any]) -> Path:
    return root / str(metadata["year"]) / f"{metadata['station_code']}.csv.gz"


def generate_extension_predictions(
    config: dict[str, Any],
    output_root: Path,
    device_name: str,
    resume: bool,
) -> list[Path]:
    cache_root = resolve_project_path(config["cache_root"])
    extension_root = resolve_project_path(config["extension_output_root"])
    base_prediction_root = resolve_project_path(config["warning_output_root"]) / "ensemble_predictions"
    prediction_root = output_root / "ensemble_predictions"
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    seeds = [int(value) for value in config["training"]["seeds"]]
    ensembles = load_extension_ensembles(extension_root, seeds, str(device))
    batch_size = int(config["warning_evaluation"]["inference_batch_size"])
    compression = int(config["warning_evaluation"]["prediction_compression_level"])
    outputs: list[Path] = []
    for split in ["train_full", "validation"]:
        source = DeepCacheBatchSource(cache_root, split)
        expected_year = int(config["warning_evaluation"]["year"]) - (1 if split == "train_full" else 0)
        pending_file_ids: set[int] = set()
        for file_id in source._positions_by_file:
            if int(source.files[file_id]["year"]) != expected_year:
                continue
            destination = _prediction_destination(prediction_root, source.files[file_id])
            if resume and destination.exists():
                outputs.append(destination)
            else:
                pending_file_ids.add(int(file_id))
        current_file_id: int | None = None
        parts: list[pd.DataFrame] = []

        def flush() -> None:
            nonlocal parts, current_file_id
            if current_file_id is None or not parts:
                return
            metadata = source.files[current_file_id]
            destination = _prediction_destination(prediction_root, metadata)
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = pd.concat(parts, ignore_index=True)
            base_path = _prediction_destination(base_prediction_root, metadata)
            if not base_path.exists():
                raise FileNotFoundError(
                    f"Missing frozen local-weather-hazard prediction source: {base_path}"
                )
            base = read_artifact_csv(
                base_path,
                columns=["row_index", "local_weather_hazard_6h"],
                encoding="utf-8-sig",
                low_memory=False,
            )
            result = result.merge(base, on="row_index", how="left", validate="one_to_one")
            if result["local_weather_hazard_6h"].isna().any():
                raise ValueError(
                    "Local-weather-hazard predictions did not align for "
                    f"{metadata['station_code']} {metadata['year']}"
                )
            result.to_csv(
                destination,
                index=False,
                encoding="utf-8-sig",
                compression={"method": "gzip", "compresslevel": compression},
            )
            outputs.append(destination)
            print(
                f"Generated extension predictions for {metadata['station_code']} {metadata['year']}",
                flush=True,
            )
            parts = []

        with torch.no_grad():
            for batch in source.iter_batches(
                batch_size,
                shuffle=False,
                include_file_ids=pending_file_ids,
            ):
                file_id = int(batch["file_id"][0])
                if current_file_id is not None and file_id != current_file_id:
                    flush()
                current_file_id = file_id
                history = batch["history"].to(device)
                station_index = batch["station_index"].to(device)
                city_index = batch["city_index"].to(device)
                scores: dict[str, np.ndarray] = {}
                for variant, ensemble in ensembles.items():
                    risk = torch.stack(
                        [
                            calibrated_extension_risk(
                                model,
                                history,
                                station_index,
                                city_index,
                                calibrator,
                            )
                            for model, calibrator in ensemble
                        ]
                    ).mean(dim=0)
                    scores[f"{variant}_6h"] = risk.cpu().numpy().astype(np.float32)
                target = batch["hazard_target"].sum(dim=1).gt(0)
                observed = target | batch["risk_mask"][:, -1].gt(0)
                metadata = source.files[file_id]
                parts.append(
                    pd.DataFrame(
                        {
                            "station_code": str(metadata["station_code"]),
                            "issue_time": pd.to_datetime(batch["issue_time_ns"].numpy()),
                            "row_index": batch["row_index"].numpy(),
                            "onset_within_6h": target.numpy().astype(np.int8),
                            "label_observed_6h": observed.numpy().astype(np.int8),
                            "hard_negative_6h": batch["hard_negative_flags"][:, 2].numpy().astype(np.int8),
                            **scores,
                        }
                    )
                )
        flush()
    return sorted(set(path.resolve() for path in outputs))


def apply_extension_budget(
    prediction_paths: list[Path],
    output_root: Path,
    budget_config: CausalBudgetConfig,
    evaluation_year: int,
    compression_level: int,
) -> tuple[list[Path], pd.DataFrame]:
    paths_by_station: dict[str, list[Path]] = {}
    for path in prediction_paths:
        paths_by_station.setdefault(path.stem.split(".", 1)[0], []).append(path)
    output_paths: list[Path] = []
    evaluation_parts: list[pd.DataFrame] = []
    for index, (station_code, paths) in enumerate(sorted(paths_by_station.items()), start=1):
        parts: list[pd.DataFrame] = []
        for path in sorted(paths):
            part = read_artifact_csv(
                path,
                encoding="utf-8-sig",
                low_memory=False,
            )
            part["issue_time"] = pd.to_datetime(part["issue_time"], errors="coerce", format="mixed")
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True).sort_values("issue_time").reset_index(drop=True)
        if frame["issue_time"].isna().any():
            raise ValueError(f"Invalid extension issue_time values for station {station_code}")
        controlled = apply_causal_budget_by_station(frame, EXTENSION_SCORE_COLUMNS, budget_config)
        yearly = controlled.loc[controlled["issue_time"].dt.year.eq(evaluation_year)].copy()
        if not yearly.empty:
            destination = output_root / "budget_predictions" / str(evaluation_year) / f"{station_code}.csv.gz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            yearly.to_csv(
                destination,
                index=False,
                encoding="utf-8-sig",
                compression={"method": "gzip", "compresslevel": compression_level},
            )
            output_paths.append(destination)
            evaluation_parts.append(yearly)
        if index % 10 == 0 or index == len(paths_by_station):
            print(f"Applied extension monthly budget for {index}/{len(paths_by_station)} stations", flush=True)
    return output_paths, pd.concat(evaluation_parts, ignore_index=True)


def _warning_inputs(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int]:
    stability_config = json.loads(
        (resolve_project_path(config["stability_root"]) / "resolved_experiment_config.json").read_text(
            encoding="utf-8"
        )
    )
    warning = stability_config["warning"]
    year = int(config["warning_evaluation"]["year"])
    result = frame.copy()
    result["issue_time"] = pd.to_datetime(result["issue_time"], errors="coerce")
    result["station_month"] = (
        result["station_code"].astype(str) + "|" + result["issue_time"].dt.to_period("M").astype(str)
    )
    event_frame = events.copy()
    event_frame["onset_time"] = pd.to_datetime(event_frame["onset_time"], errors="coerce")
    event_frame["valid_target_event"] = pd.to_numeric(
        event_frame["valid_target_event"], errors="coerce"
    ).fillna(0).astype(int)
    event_frame = event_frame.loc[event_frame["onset_time"].dt.year.eq(year)]
    return result, event_frame, warning, year


def evaluate_extension_alarms(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
    alarm_columns: dict[str, str],
    operating_point: str,
    budget_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, event_frame, warning, year = _warning_inputs(frame, events, config)
    rows: list[dict[str, Any]] = []
    records: list[pd.DataFrame] = []
    for model_name, alarm_column in alarm_columns.items():
        row = evaluate_warning_model(
            frame,
            event_frame,
            alarm_column,
            0.5,
            int(warning["horizon_hours"]),
            int(config["step_minutes"]),
            budget_hours,
            operating_point,
            int(warning["minimum_consecutive_alarm_bins"]),
            int(warning["alarm_merge_gap_minutes"]),
            int(warning["maximum_silence_before_event_minutes"]),
        )
        row.update(
            {
                "model_name": model_name,
                "year": year,
                "lead_utility_hours": (
                    float(row["event_hit_rate"] * row["mean_effective_lead_hours"])
                    if np.isfinite(row["mean_effective_lead_hours"])
                    else float("nan")
                ),
            }
        )
        rows.append(row)
        model_records = event_alarm_records(
            frame,
            event_frame,
            alarm_column,
            0.5,
            int(warning["horizon_hours"]),
            warning,
        )
        model_records["model_name"] = model_name
        records.append(model_records)
    record_frame = pd.concat(records, ignore_index=True)
    comparisons: list[pd.DataFrame] = []
    for index, (model_a, model_b) in enumerate(EXTENSION_PAIRS):
        comparison = cluster_bootstrap_event_comparison(
            record_frame,
            alarm_columns[model_a],
            alarm_columns[model_b],
            int(config["warning_evaluation"]["bootstrap_samples"]),
            int(config["warning_evaluation"]["bootstrap_seed"]) + index,
        )
        comparison["model_a_name"] = model_a
        comparison["model_b_name"] = model_b
        comparisons.append(comparison)
    return pd.DataFrame(rows), record_frame, pd.concat(comparisons, ignore_index=True)


def evaluate_extension_warning(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
    budget_config: CausalBudgetConfig,
) -> dict[str, pd.DataFrame]:
    prepared, _, warning, year = _warning_inputs(frame, events, config)
    frozen_alarm_columns = {
        score.removesuffix("_6h"): f"{score}__budget_alarm" for score in EXTENSION_SCORE_COLUMNS
    }
    metrics, records, comparisons = evaluate_extension_alarms(
        prepared,
        events,
        config,
        frozen_alarm_columns,
        f"{year}::frozen_causal_budget",
        budget_config.monthly_budget_hours,
    )
    metrics["candidate_quantile"] = budget_config.candidate_quantile
    audit = monthly_budget_audit(
        prepared,
        frozen_alarm_columns,
        int(config["step_minutes"]),
        budget_config.monthly_budget_hours,
    )
    diagnostic = config["warning_evaluation"]["matched_false_alarm_diagnostic"]
    target_hours = float(
        diagnostic.get("target_hours_per_station_month", warning["false_alarm_hours_per_station_month"])
    )
    selection = select_matched_false_alarm_thresholds(
        prepared,
        EXTENSION_SCORE_COLUMNS,
        int(warning["horizon_hours"]),
        target_hours,
        int(config["step_minutes"]),
    )
    matched_alarm_columns: dict[str, str] = {}
    matched_frame = prepared.copy()
    for row in selection.itertuples(index=False):
        alarm_column = f"{row.score_column}__matched_alarm"
        score = pd.to_numeric(matched_frame[row.score_column], errors="coerce").fillna(0).to_numpy(np.float64)
        matched_frame[alarm_column] = (score >= float(row.threshold)).astype(np.int8)
        matched_alarm_columns[str(row.model_name)] = alarm_column
    matched_metrics, matched_records, matched_comparisons = evaluate_extension_alarms(
        matched_frame,
        events,
        config,
        matched_alarm_columns,
        f"{year}::matched_false_alarm_diagnostic",
        target_hours,
    )
    matched_metrics = matched_metrics.rename(columns={"threshold": "binary_alarm_threshold"}).merge(
        selection[
            [
                "model_name",
                "threshold",
                "target_false_alarm_hours_per_station_month",
                "absolute_error_hours_per_station_month",
            ]
        ].rename(columns={"threshold": "score_threshold"}),
        on="model_name",
        how="left",
        validate="one_to_one",
    )
    matched_audit = monthly_false_alarm_distribution_audit(
        matched_frame,
        matched_alarm_columns,
        int(warning["horizon_hours"]),
        int(config["step_minutes"]),
        target_hours,
    )
    return {
        "warning_metrics": metrics,
        "event_records": records,
        "paired_event_comparisons": comparisons,
        "monthly_budget_audit": audit,
        "matched_warning_metrics": matched_metrics,
        "matched_event_records": matched_records,
        "matched_paired_event_comparisons": matched_comparisons,
        "matched_monthly_distribution_audit": matched_audit,
        "matched_threshold_selection": selection,
    }


def build_extension_warning_report(
    results: dict[str, pd.DataFrame],
    budget_config: CausalBudgetConfig,
) -> str:
    metric_columns = [
        "model_name",
        "false_alarm_hours_per_station_month",
        "event_hit_rate",
        "mean_effective_lead_hours",
        "median_effective_lead_hours",
        "lead_utility_hours",
        "hard_negative_far",
    ]
    comparison_columns = [
        "model_a_name",
        "model_b_name",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
    ]
    matched_columns = ["model_name", "score_threshold", *metric_columns[1:]]
    return "\n".join(
        [
            "# Deep RIG-Hazard扩展完整时序事件评估",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 冻结因果月预算",
            "",
            f"所有模型共用过去30天滚动分位数{budget_config.candidate_quantile:.2f}，并接受每站每月{budget_config.monthly_budget_hours:.1f}小时总告警硬上限。",
            "",
            results["warning_metrics"][metric_columns].to_markdown(index=False),
            "",
            results["paired_event_comparisons"][comparison_columns].to_markdown(index=False),
            "",
            "### 月预算审计",
            "",
            results["monthly_budget_audit"].to_markdown(index=False),
            "",
            "## 严格同误报时长诊断",
            "",
            "每个模型使用2023开发标签单独选择概率阈值，使总体误报精确匹配10小时/站点月；该口径仅用于能力比较，不是部署阈值。",
            "",
            results["matched_warning_metrics"][matched_columns].to_markdown(index=False),
            "",
            results["matched_paired_event_comparisons"][comparison_columns].to_markdown(index=False),
            "",
            "### 阈值与逐月分布",
            "",
            results["matched_threshold_selection"].to_markdown(index=False),
            "",
            results["matched_monthly_distribution_audit"].to_markdown(index=False),
            "",
            "## 解释边界",
            "",
            "所有扩展选择和比较均止于2023开发年。2024不参与复发史、负样本策略、屏障正则或模型选择；论文级确认仍需新年份或外部区域。",
        ]
    ) + "\n"


def run_deep_extension_warning(
    config: dict[str, Any],
    config_path: Path,
    overwrite: bool = False,
    resume: bool = False,
    device_name: str = "auto",
) -> Path:
    output_root = resolve_project_path(config["extension_warning_output_root"])
    if resume:
        if overwrite:
            raise ValueError("--resume and --overwrite cannot be used together")
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        prepare_output_root(output_root, overwrite)
    torch.set_num_threads(max(int(config["training"].get("torch_num_threads", 1)), 1))
    write_json(
        output_root / "resolved_config.json",
        {**config, "config_path": str(config_path), "device": device_name, "resume": resume},
    )
    prediction_paths = generate_extension_predictions(config, output_root, device_name, resume)
    budget_config = load_frozen_budget(resolve_project_path(config["stability_root"]))
    budget_paths, frame = apply_extension_budget(
        prediction_paths,
        output_root,
        budget_config,
        int(config["warning_evaluation"]["year"]),
        int(config["warning_evaluation"]["prediction_compression_level"]),
    )
    events = pd.read_csv(
        resolve_project_path(config["preprocessed_root"]) / "events_recurrent.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    results = evaluate_extension_warning(frame, events, config, budget_config)
    for name, result in results.items():
        result.to_csv(output_root / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_root / "run_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "prediction_files": len(prediction_paths),
            "budget_prediction_files": len(budget_paths),
            "ensemble_seeds": [int(value) for value in config["training"]["seeds"]],
            "extension_variants": EXTENSION_VARIANTS,
            "budget_config": {
                field.name: getattr(budget_config, field.name) for field in fields(CausalBudgetConfig)
            },
        },
    )
    (output_root / "extension_warning_report.md").write_text(
        build_extension_warning_report(results, budget_config), encoding="utf-8"
    )
    print(f"Deep extension warning evaluation complete: {output_root}", flush=True)
    return output_root

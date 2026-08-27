from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_experiment import (
    evaluate_warning_model,
    observed_warning_rows,
    threshold_for_false_alarm_budget,
)
from .budget_control import CausalBudgetConfig, apply_causal_budget_by_station
from .config import resolve_project_path
from .deep_data import DeepCacheBatchSource
from .deep_models import cloglog_hazard_probability, cumulative_incidence
from .naming import MAIN_MODEL_NAME, artifact_value
from .deep_training import load_deep_checkpoint
from .graph_experiment import cluster_bootstrap_event_comparison, event_alarm_records
from .preprocessing import prepare_output_root, write_json
from .torch_runtime import torch


SCORE_COLUMNS = ["local_weather_hazard_6h", "gru_ensemble_6h", "tcn_ensemble_6h"]
MODEL_PAIRS = [
    ("gru_ensemble", "local_weather_hazard"),
    ("tcn_ensemble", "local_weather_hazard"),
    ("tcn_ensemble", "gru_ensemble"),
]
WARNING_DEFINITION_KEYS = {
    "horizon_hours",
    "false_alarm_hours_per_station_month",
    "minimum_consecutive_alarm_bins",
    "alarm_merge_gap_minutes",
    "maximum_silence_before_event_minutes",
}


def score_columns_for_models(model_names: list[str]) -> list[str]:
    return ["local_weather_hazard_6h", *[f"{name}_ensemble_6h" for name in model_names]]


def comparison_pairs_for_models(model_names: list[str]) -> list[tuple[str, str]]:
    ensembles = [f"{name}_ensemble" for name in model_names]
    pairs = [(name, "local_weather_hazard") for name in ensembles]
    pairs.extend(
        (ensembles[left], ensembles[right])
        for left in range(len(ensembles))
        for right in range(left + 1, len(ensembles))
    )
    return pairs


def load_frozen_budget(stability_root: Path) -> CausalBudgetConfig:
    bundle = json.loads((stability_root / "model_bundle.json").read_text(encoding="utf-8"))
    payload = bundle["causal_budget_config"]
    allowed = {field.name for field in fields(CausalBudgetConfig)}
    return CausalBudgetConfig(**{name: payload[name] for name in allowed if name in payload})


def load_frozen_budgets(
    stability_root: Path,
    score_columns: list[str],
) -> dict[str, CausalBudgetConfig]:
    bundle = json.loads((stability_root / "model_bundle.json").read_text(encoding="utf-8"))
    default_budget = load_frozen_budget(stability_root)
    payloads = dict(bundle.get("causal_budget_configs_by_score", {}))
    allowed = {field.name for field in fields(CausalBudgetConfig)}
    result: dict[str, CausalBudgetConfig] = {}
    for score_column in score_columns:
        payload = payloads.get(score_column)
        result[score_column] = (
            CausalBudgetConfig(
                **{name: payload[name] for name in allowed if name in payload}
            )
            if payload is not None
            else default_budget
        )
    return result


def _budget_map(
    budget: CausalBudgetConfig | dict[str, CausalBudgetConfig],
    score_columns: list[str],
) -> dict[str, CausalBudgetConfig]:
    if isinstance(budget, CausalBudgetConfig):
        return {score_column: budget for score_column in score_columns}
    missing = sorted(set(score_columns) - set(budget))
    if missing:
        raise ValueError(f"Missing frozen budget configuration for score columns: {missing}")
    return {score_column: budget[score_column] for score_column in score_columns}


def load_frozen_warning_definition(config: dict[str, Any]) -> dict[str, Any]:
    stability_root = resolve_project_path(config["stability_root"])
    bundle_path = stability_root / "model_bundle.json"
    warning: dict[str, Any] | None = None
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if "warning_definition" in bundle:
            warning = dict(bundle["warning_definition"])
    if warning is None and "warning" in config:
        warning = dict(config["warning"])
    if warning is None:
        legacy_path = stability_root / "resolved_experiment_config.json"
        if legacy_path.exists():
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            warning = dict(legacy.get("warning", {}))
    if warning is None:
        raise ValueError("No frozen warning definition is available in the budget bundle or experiment config")
    missing = sorted(WARNING_DEFINITION_KEYS - set(warning))
    if missing:
        raise ValueError(f"Frozen warning definition is missing fields: {missing}")
    if int(warning["horizon_hours"]) != 6:
        raise ValueError("Deep warning predictions currently support the frozen 6-hour horizon only")
    return warning


def monthly_budget_audit(
    frame: pd.DataFrame,
    alarm_columns: dict[str, str],
    step_minutes: int,
    budget_hours: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, alarm_column in alarm_columns.items():
        monthly_bins = frame.groupby("station_month")[alarm_column].sum().astype(np.int64)
        monthly_hours = monthly_bins * int(step_minutes) / 60.0
        rows.append(
            {
                "model_name": model_name,
                "station_months": int(monthly_hours.size),
                "mean_alarm_hours": float(monthly_hours.mean()),
                "maximum_alarm_hours": float(monthly_hours.max()),
                "months_above_budget": int((monthly_hours > budget_hours + 1e-9).sum()),
                "budget_hours": budget_hours,
            }
        )
    return pd.DataFrame(rows)


def monthly_false_alarm_distribution_audit(
    frame: pd.DataFrame,
    alarm_columns: dict[str, str],
    horizon_hours: int,
    step_minutes: int,
    target_hours: float,
) -> pd.DataFrame:
    future_event = frame[f"onset_within_{horizon_hours}h"].eq(1)
    rows: list[dict[str, Any]] = []
    for model_name, alarm_column in alarm_columns.items():
        alarm = frame[alarm_column].astype(bool)
        false_alarm = (alarm & ~future_event).astype(np.int8)
        monthly_false_bins = false_alarm.groupby(frame["station_month"]).sum().astype(np.int64)
        monthly_alarm_bins = alarm.astype(np.int8).groupby(frame["station_month"]).sum().astype(np.int64)
        monthly_false_hours = monthly_false_bins * int(step_minutes) / 60.0
        monthly_alarm_hours = monthly_alarm_bins * int(step_minutes) / 60.0
        rows.append(
            {
                "model_name": model_name,
                "station_months": int(monthly_false_hours.size),
                "mean_false_alarm_hours": float(monthly_false_hours.mean()),
                "maximum_false_alarm_hours": float(monthly_false_hours.max()),
                "station_months_above_target": int((monthly_false_hours > target_hours + 1e-9).sum()),
                "target_false_alarm_hours": float(target_hours),
                "mean_total_alarm_hours": float(monthly_alarm_hours.mean()),
                "maximum_total_alarm_hours": float(monthly_alarm_hours.max()),
            }
        )
    return pd.DataFrame(rows)


def select_matched_false_alarm_thresholds(
    frame: pd.DataFrame,
    score_columns: list[str],
    horizon_hours: int,
    target_hours_per_station_month: float,
    step_minutes: int,
) -> pd.DataFrame:
    """Select model-specific development thresholds at one aggregate false-alarm duration."""

    rows: list[dict[str, Any]] = []
    future_event = frame[f"onset_within_{horizon_hours}h"].eq(1).to_numpy()
    station_months = max(int(frame["station_month"].nunique()), 1)
    for score_column in score_columns:
        threshold = threshold_for_false_alarm_budget(
            frame,
            score_column,
            horizon_hours,
            target_hours_per_station_month,
            step_minutes,
        )
        score = pd.to_numeric(frame[score_column], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        false_bins = int(((score >= threshold) & ~future_event).sum())
        actual_hours = false_bins * step_minutes / 60.0 / station_months
        rows.append(
            {
                "model_name": score_column.removesuffix("_6h"),
                "score_column": score_column,
                "threshold": float(threshold),
                "target_false_alarm_hours_per_station_month": float(target_hours_per_station_month),
                "actual_false_alarm_hours_per_station_month": float(actual_hours),
                "absolute_error_hours_per_station_month": float(
                    abs(actual_hours - target_hours_per_station_month)
                ),
                "false_alarm_bins": false_bins,
                "station_months": station_months,
            }
        )
    return pd.DataFrame(rows)


def load_model_ensemble(
    local_root: Path,
    model_name: str,
    expected_seeds: list[int],
    device_name: str,
) -> list[tuple[torch.nn.Module, dict[str, Any]]]:
    ensemble: list[tuple[torch.nn.Module, dict[str, Any]]] = []
    for seed in expected_seeds:
        path = local_root / "checkpoints" / f"{model_name}_seed_{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing formal checkpoint: {path}")
        model, checkpoint = load_deep_checkpoint(path, device_name)
        ensemble.append((model, checkpoint["trajectory_calibrator"]))
    return ensemble


def calibrated_six_hour_risk(
    model: torch.nn.Module,
    history: torch.Tensor,
    calibrator: dict[str, Any],
) -> torch.Tensor:
    eta = model(history)
    eta = float(calibrator["log_rate_shift"]) + float(calibrator["slope"]) * eta
    return cumulative_incidence(cloglog_hazard_probability(eta))[:, -1]


def load_main_model_parameters(
    baseline_root: Path,
    device: torch.device,
    cache_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    recurrence_bundle_path = baseline_root / "statistical_model_bundle.json"
    if recurrence_bundle_path.exists():
        bundle = json.loads(recurrence_bundle_path.read_text(encoding="utf-8"))
        baseline = bundle["models"]["base_cloglog"]
        model = baseline["glm"]
        calibrator = baseline["calibrator"]
        baseline_features = list(baseline["coefficient_names"])
        source = "2022 pooled-OOF calibrated base_cloglog"
    else:
        bundle = json.loads((baseline_root / "model_bundle.json").read_text(encoding="utf-8"))
        model = artifact_value(bundle["models"], MAIN_MODEL_NAME)
        calibrator = artifact_value(bundle["calibrators"], MAIN_MODEL_NAME)
        transformer = bundle.get("feature_transformer", {})
        baseline_features = list(
            transformer.get(
                "feature_names",
                [*transformer.get("continuous_features", []), *transformer.get("binary_features", [])],
            )
        )
        source = "configured frozen local-weather-hazard bundle"
    feature_indices = None
    if cache_feature_names is not None and baseline_features:
        missing = sorted(set(baseline_features) - set(cache_feature_names))
        if missing:
            raise ValueError(
                f"Deep cache is missing local-weather-hazard features: {missing}"
            )
        feature_indices = torch.tensor(
            [cache_feature_names.index(name) for name in baseline_features],
            dtype=torch.long,
            device=device,
        )
    return {
        "coefficients": torch.tensor(model["coefficients"], dtype=torch.float32, device=device),
        "intercept": float(model["intercept"]),
        "shift": float(calibrator["log_rate_shift"]),
        "slope": float(calibrator["slope"]),
        "feature_indices": feature_indices,
        "source": source,
    }


def main_model_six_hour_risk(current_features: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    if parameters.get("feature_indices") is not None:
        current_features = current_features.index_select(1, parameters["feature_indices"])
    if current_features.shape[1] != parameters["coefficients"].numel():
        raise ValueError(
            "Local-weather-hazard feature count does not match its frozen "
            "coefficient vector"
        )
    eta = parameters["intercept"] + current_features @ parameters["coefficients"]
    eta = parameters["shift"] + parameters["slope"] * eta
    rate = torch.exp(torch.clamp(eta, max=15.0))
    return -torch.expm1(-36.0 * rate)


def _prediction_destination(root: Path, metadata: dict[str, Any]) -> Path:
    return root / str(metadata["year"]) / f"{metadata['station_code']}.csv.gz"


def generate_ensemble_predictions(
    config: dict[str, Any],
    output_root: Path,
    device_name: str,
    resume: bool,
    model_names: list[str] | None = None,
    local_root_override: str | None = None,
) -> list[Path]:
    cache_root = resolve_project_path(config["cache_root"])
    local_root = resolve_project_path(local_root_override or config["local_output_root"])
    baseline_root = resolve_project_path(config.get("warning_baseline_root", config["baseline_root"]))
    prediction_root = output_root / "ensemble_predictions"
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    seeds = [int(value) for value in config["training"]["seeds"]]
    selected_models = model_names or [
        str(value) for value in config.get("default_local_models", ["gru", "tcn"])
    ]
    ensembles = {
        model_name: load_model_ensemble(local_root, model_name, seeds, str(device))
        for model_name in selected_models
    }
    required_score_columns = set(score_columns_for_models(selected_models))
    cache_contract = json.loads((cache_root / "sample_contract.json").read_text(encoding="utf-8"))
    main_model_parameters = load_main_model_parameters(
        baseline_root, device, list(cache_contract["feature_names"])
    )
    batch_size = int(config["warning_evaluation"]["inference_batch_size"])
    compression = int(config["warning_evaluation"]["prediction_compression_level"])
    outputs: list[Path] = []

    warning_config = config["warning_evaluation"]
    history_split = str(warning_config.get("history_split", "train_full"))
    evaluation_split = str(warning_config.get("evaluation_split", "validation"))
    evaluation_year = int(warning_config["year"])
    for split, expected_year in [
        (history_split, evaluation_year - 1),
        (evaluation_split, evaluation_year),
    ]:
        source = DeepCacheBatchSource(cache_root, split)
        pending_file_ids: set[int] = set()
        for file_id in source._positions_by_file:
            if int(source.files[file_id]["year"]) != expected_year:
                continue
            destination = _prediction_destination(prediction_root, source.files[file_id])
            can_reuse = False
            if resume and destination.exists():
                existing_columns = set(pd.read_csv(destination, nrows=0).columns)
                can_reuse = required_score_columns.issubset(existing_columns)
            if can_reuse:
                outputs.append(destination)
                continue
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
            pd.concat(parts, ignore_index=True).to_csv(
                destination,
                index=False,
                encoding="utf-8-sig",
                compression={"method": "gzip", "compresslevel": compression},
            )
            outputs.append(destination)
            print(
                f"Generated ensemble predictions for {metadata['station_code']} {metadata['year']}",
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
                ensemble_risks = {
                    f"{model_name}_ensemble_6h": torch.stack(
                        [
                            calibrated_six_hour_risk(model, history, calibrator)
                            for model, calibrator in ensemble
                        ]
                    ).mean(dim=0)
                    for model_name, ensemble in ensembles.items()
                }
                main_model_risk = main_model_six_hour_risk(
                    history[:, -1, :], main_model_parameters
                )
                target = batch["hazard_target"].sum(dim=1).gt(0)
                observed = target | batch["risk_mask"][:, -1].gt(0)
                metadata = source.files[file_id]
                rows = batch["row_index"].numpy()
                data: dict[str, Any] = {
                    "station_code": str(metadata["station_code"]),
                    "issue_time": pd.to_datetime(batch["issue_time_ns"].numpy()),
                    "row_index": rows,
                    "onset_within_6h": target.numpy().astype(np.int8),
                    "label_observed_6h": observed.numpy().astype(np.int8),
                    "hard_negative_6h": batch["hard_negative_flags"][:, 2].numpy().astype(np.int8),
                    "local_weather_hazard_6h": main_model_risk.cpu()
                    .numpy()
                    .astype(np.float32),
                }
                data.update(
                    {
                        score_column: risk.cpu().numpy().astype(np.float32)
                        for score_column, risk in ensemble_risks.items()
                    }
                )
                parts.append(pd.DataFrame(data))
        flush()
    return sorted(set(path.resolve() for path in outputs))


def apply_frozen_budget(
    prediction_paths: list[Path],
    output_root: Path,
    budget_config: CausalBudgetConfig | dict[str, CausalBudgetConfig],
    evaluation_year: int,
    compression_level: int,
    score_columns: list[str] | None = None,
) -> tuple[list[Path], pd.DataFrame]:
    selected_score_columns = score_columns or SCORE_COLUMNS
    budget_by_score = _budget_map(budget_config, selected_score_columns)
    paths_by_station: dict[str, list[Path]] = {}
    for path in prediction_paths:
        paths_by_station.setdefault(path.stem.split(".", 1)[0], []).append(path)
    output_paths: list[Path] = []
    evaluation_parts: list[pd.DataFrame] = []
    for index, (station_code, paths) in enumerate(sorted(paths_by_station.items()), start=1):
        parts: list[pd.DataFrame] = []
        for path in sorted(paths):
            part = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            part["issue_time"] = pd.to_datetime(part["issue_time"], errors="coerce", format="mixed")
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True).sort_values("issue_time").reset_index(drop=True)
        if frame["issue_time"].isna().any():
            raise ValueError(f"Invalid issue_time values in ensemble predictions for station {station_code}")
        controlled = frame
        for score_column in selected_score_columns:
            controlled = apply_causal_budget_by_station(
                controlled,
                [score_column],
                budget_by_score[score_column],
            )
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
            print(f"Applied frozen monthly budget for {index}/{len(paths_by_station)} stations", flush=True)
    return output_paths, pd.concat(evaluation_parts, ignore_index=True)


def _prepare_warning_inputs(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int]:
    warning = load_frozen_warning_definition(config)
    year = int(config["warning_evaluation"]["year"])
    frame = frame.copy()
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], errors="coerce")
    frame["station_month"] = frame["station_code"].astype(str) + "|" + frame["issue_time"].dt.to_period("M").astype(str)
    event_frame = events.copy()
    event_frame["onset_time"] = pd.to_datetime(event_frame["onset_time"], errors="coerce")
    event_frame["valid_target_event"] = pd.to_numeric(
        event_frame["valid_target_event"], errors="coerce"
    ).fillna(0).astype(int)
    event_frame = event_frame.loc[event_frame["onset_time"].dt.year.eq(year)]
    return frame, event_frame, warning, year


def _evaluate_alarm_columns(
    frame: pd.DataFrame,
    event_frame: pd.DataFrame,
    config: dict[str, Any],
    warning: dict[str, Any],
    year: int,
    alarm_columns: dict[str, str],
    operating_point: str,
    monthly_budget_hours: float,
    candidate_quantiles: dict[str, float] | None = None,
    model_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluation_frame = observed_warning_rows(frame, "label_observed_6h")
    total_time_bins = int(frame.shape[0])
    observed_time_bins = int(evaluation_frame.shape[0])
    rows: list[dict[str, Any]] = []
    records: list[pd.DataFrame] = []
    for model_name, alarm_column in alarm_columns.items():
        row = evaluate_warning_model(
            evaluation_frame,
            event_frame,
            alarm_column,
            0.5,
            int(warning["horizon_hours"]),
            int(config["step_minutes"]),
            float(warning["false_alarm_hours_per_station_month"]),
            operating_point,
            int(warning["minimum_consecutive_alarm_bins"]),
            int(warning["alarm_merge_gap_minutes"]),
            int(warning["maximum_silence_before_event_minutes"]),
        )
        row.update(
            {
                "model_name": model_name,
                "year": year,
                "candidate_quantile": (
                    candidate_quantiles.get(model_name, float("nan"))
                    if candidate_quantiles is not None
                    else float("nan")
                ),
                "lead_utility_hours": float(row["event_hit_rate"] * row["mean_effective_lead_hours"])
                if np.isfinite(row["mean_effective_lead_hours"])
                else float("nan"),
                "total_time_bins": total_time_bins,
                "observed_time_bins": observed_time_bins,
                "censored_time_bins": total_time_bins - observed_time_bins,
            }
        )
        rows.append(row)
        model_records = event_alarm_records(
            evaluation_frame,
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
    selected_pairs = model_pairs or MODEL_PAIRS
    for index, (model_a, model_b) in enumerate(selected_pairs):
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
    audit = monthly_budget_audit(
        frame,
        alarm_columns,
        int(config["step_minutes"]),
        monthly_budget_hours,
    )
    return pd.DataFrame(rows), record_frame, pd.concat(comparisons, ignore_index=True), audit


def evaluate_deep_warning(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
    budget_config: CausalBudgetConfig | dict[str, CausalBudgetConfig],
    score_columns: list[str] | None = None,
    model_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, event_frame, warning, year = _prepare_warning_inputs(frame, events, config)
    selected_score_columns = score_columns or SCORE_COLUMNS
    budget_by_score = _budget_map(budget_config, selected_score_columns)
    monthly_budget_values = {
        value.monthly_budget_hours for value in budget_by_score.values()
    }
    if len(monthly_budget_values) != 1:
        raise ValueError("All compared models must use the same frozen monthly alarm budget")
    alarm_columns = {
        score_column.removesuffix("_6h"): f"{score_column}__budget_alarm"
        for score_column in selected_score_columns
    }
    quantiles = {
        score_column.removesuffix("_6h"): budget_by_score[score_column].candidate_quantile
        for score_column in selected_score_columns
    }
    return _evaluate_alarm_columns(
        frame,
        event_frame,
        config,
        warning,
        year,
        alarm_columns,
        f"{year}::frozen_causal_budget",
        next(iter(monthly_budget_values)),
        quantiles,
        model_pairs,
    )


def evaluate_matched_false_alarm_diagnostic(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
    budget_config: CausalBudgetConfig | dict[str, CausalBudgetConfig],
    score_columns: list[str] | None = None,
    model_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, event_frame, warning, year = _prepare_warning_inputs(frame, events, config)
    evaluation_frame = observed_warning_rows(frame, "label_observed_6h")
    diagnostic = config["warning_evaluation"]["matched_false_alarm_diagnostic"]
    target_hours = float(
        diagnostic.get("target_hours_per_station_month", warning["false_alarm_hours_per_station_month"])
    )
    selected_score_columns = score_columns or SCORE_COLUMNS
    budget_by_score = _budget_map(budget_config, selected_score_columns)
    reference_budget = budget_by_score[selected_score_columns[0]]
    selection = select_matched_false_alarm_thresholds(
        evaluation_frame,
        selected_score_columns,
        int(warning["horizon_hours"]),
        target_hours,
        int(config["step_minutes"]),
    )
    alarm_columns: dict[str, str] = {}
    for row in selection.itertuples(index=False):
        alarm_column = f"{row.score_column}__matched_alarm"
        score = pd.to_numeric(frame[row.score_column], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        frame[alarm_column] = (score >= float(row.threshold)).astype(np.int8)
        alarm_columns[str(row.model_name)] = alarm_column
    metrics, records, comparisons, audit = _evaluate_alarm_columns(
        frame,
        event_frame,
        config,
        warning,
        year,
        alarm_columns,
        f"{year}::matched_false_alarm_diagnostic",
        reference_budget.monthly_budget_hours,
        model_pairs=model_pairs,
    )
    metrics = metrics.rename(columns={"threshold": "binary_alarm_threshold"})
    metrics = metrics.merge(
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
    audit = monthly_false_alarm_distribution_audit(
        observed_warning_rows(frame, "label_observed_6h"),
        alarm_columns,
        int(warning["horizon_hours"]),
        int(config["step_minutes"]),
        target_hours,
    )
    return metrics, records, comparisons, audit, selection


def build_warning_report(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    audit: pd.DataFrame,
    budget_config: CausalBudgetConfig,
    matched_metrics: pd.DataFrame | None = None,
    matched_comparisons: pd.DataFrame | None = None,
    matched_audit: pd.DataFrame | None = None,
    matched_selection: pd.DataFrame | None = None,
    ensemble_model_names: list[str] | None = None,
    evaluation_year: int = 2023,
    ensemble_seed_count: int = 5,
    baseline_source: str | None = None,
    budget_configs_by_score: dict[str, CausalBudgetConfig] | None = None,
) -> str:
    columns = [
        "model_name",
        "false_alarm_hours_per_station_month",
        "event_hit_rate",
        "mean_effective_lead_hours",
        "median_effective_lead_hours",
        "lead_utility_hours",
        "hard_negative_far",
    ]
    evaluation_role = {
        2023: "跨年份验证",
        2024: "冻结时间确认",
    }.get(int(evaluation_year), "冻结时序评价")
    model_labels = "、".join(ensemble_model_names or ["GRU", "TCN"])
    sections = [
        "# Deep RIG-Hazard完整时序预警评估",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"评价年份：{evaluation_year}（{evaluation_role}）",
        "",
        "## 冻结口径",
        "",
        "- 模型选择、概率校准与告警预算选择仅使用2022区块交叉验证的折外预测。",
        f"- 统计基线来源：{baseline_source or '配置中冻结的local weather hazard危害率模型'}。",
        f"- {model_labels}均为{ensemble_seed_count}个随机种子校准概率的等权集成。",
        f"- 每站每月总告警硬上限固定为{budget_config.monthly_budget_hours:.1f}小时。",
        f"- {evaluation_year}年的模型参数、校准器和预算参数保持冻结，不读取该年标签进行调整。",
        "- 预算控制在完整时间线上执行；误报、硬负样本和事件指标只在6小时结局完整可判定的窗口上评分。",
    ]
    if budget_configs_by_score:
        quantile_summary = "；".join(
            f"{score.removesuffix('_6h')}={value.candidate_quantile:.4f}"
            for score, value in budget_configs_by_score.items()
        )
        sections.append(f"- 2022折外冻结的模型候选分位数：{quantile_summary}。")
    coverage_columns = ["total_time_bins", "observed_time_bins", "censored_time_bins"]
    if set(coverage_columns).issubset(metrics.columns) and not metrics.empty:
        coverage = metrics.iloc[0]
        sections.extend(
            [
                f"- 时间窗审计：总计{int(coverage['total_time_bins'])}，可判定{int(coverage['observed_time_bins'])}，删失{int(coverage['censored_time_bins'])}。",
            ]
        )
    sections.extend(
        [
            "",
            "## 事件结果",
            "",
            metrics[columns].to_markdown(index=False),
            "",
            "## 配对站点聚类Bootstrap",
            "",
            comparisons[["model_a_name", "model_b_name", "metric", "estimate", "ci95_low", "ci95_high"]].to_markdown(index=False),
            "",
            "## 月预算审计",
            "",
            audit.to_markdown(index=False),
        ]
    )
    if (
        matched_metrics is not None
        and matched_comparisons is not None
        and matched_audit is not None
        and matched_selection is not None
    ):
        matched_columns = [
            "model_name",
            "score_threshold",
            "false_alarm_hours_per_station_month",
            "event_hit_rate",
            "mean_effective_lead_hours",
            "median_effective_lead_hours",
            "lead_utility_hours",
            "hard_negative_far",
        ]
        target_hours = float(
            matched_selection["target_false_alarm_hours_per_station_month"].iloc[0]
        )
        sections.extend(
            [
                "",
                "## 严格同误报时长诊断",
                "",
                f"- 每个模型在{evaluation_year}年单独选择6小时风险阈值，使总体误报时长尽可能贴近{target_hours:.1f}小时/站点月。",
                f"- 该阈值使用了完整{evaluation_year}年标签，只用于判断模型在同一误报代价下的相对能力，不是可部署阈值，也不进入后续年份。",
                "- 该口径匹配总体误报时长，不施加逐站逐月硬上限；逐月偏离情况见后续审计。",
                "",
                matched_metrics[matched_columns].to_markdown(index=False),
                "",
                "### 阈值匹配误差",
                "",
                matched_selection.to_markdown(index=False),
                "",
                "### 配对站点聚类Bootstrap",
                "",
                matched_comparisons[
                    ["model_a_name", "model_b_name", "metric", "estimate", "ci95_low", "ci95_high"]
                ].to_markdown(index=False),
                "",
                "### 逐月分布审计",
                "",
                matched_audit.to_markdown(index=False),
            ]
        )
    boundary = (
        "2023结果只用于跨年份稳定性验证，不得据此重新拟合模型、校准概率或选择预算。"
        if int(evaluation_year) == 2023
        else "2024结果按回顾性锁定时间确认解释，不得用于重新调参；若要宣称从未查看的独立测试，仍需新增年份或外部区域数据。"
        if int(evaluation_year) == 2024
        else "该年份结果仅按预先指定的冻结时序评价解释，不得反向调整模型。"
    )
    sections.extend(["", "## 解释边界", "", boundary])
    return "\n".join(sections) + "\n"


def run_deep_warning(
    config: dict[str, Any],
    config_path: Path,
    overwrite: bool = False,
    resume: bool = False,
    device_name: str = "auto",
    models: list[str] | None = None,
    local_root_override: str | None = None,
    output_root_override: str | None = None,
    evaluation_year_override: int | None = None,
    history_split_override: str | None = None,
    evaluation_split_override: str | None = None,
    disable_matched_diagnostic: bool = False,
) -> Path:
    config = json.loads(json.dumps(config))
    if evaluation_year_override is not None:
        config["warning_evaluation"]["year"] = int(evaluation_year_override)
    if history_split_override is not None:
        config["warning_evaluation"]["history_split"] = str(history_split_override)
    if evaluation_split_override is not None:
        config["warning_evaluation"]["evaluation_split"] = str(evaluation_split_override)
    if disable_matched_diagnostic:
        config["warning_evaluation"].setdefault("matched_false_alarm_diagnostic", {})["enabled"] = False
    evaluation_year = int(config["warning_evaluation"]["year"])
    matched_config = config["warning_evaluation"].get("matched_false_alarm_diagnostic", {})
    if evaluation_year >= 2024 and bool(matched_config.get("enabled", True)):
        raise ValueError(
            "Matched false-alarm diagnostics read evaluation-year labels and must be disabled "
            "for frozen 2024 confirmation with --disable-matched-diagnostic"
        )
    selected_models = models or [
        str(value) for value in config.get("default_local_models", ["gru", "tcn"])
    ]
    score_columns = score_columns_for_models(selected_models)
    model_pairs = comparison_pairs_for_models(selected_models)
    output_root = resolve_project_path(output_root_override or config["warning_output_root"])
    if resume:
        if overwrite:
            raise ValueError("--resume and --overwrite cannot be used together")
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        prepare_output_root(output_root, overwrite)
    torch.set_num_threads(max(int(config["training"].get("torch_num_threads", 1)), 1))
    write_json(
        output_root / "resolved_config.json",
        {
            **config,
            "config_path": str(config_path),
            "device": device_name,
            "resume": resume,
            "warning_models": selected_models,
            "local_root_override": local_root_override,
            "output_root_override": output_root_override,
        },
    )
    prediction_paths = generate_ensemble_predictions(
        config,
        output_root,
        device_name,
        resume,
        selected_models,
        local_root_override,
    )
    stability_root = resolve_project_path(config["stability_root"])
    budget_configs = load_frozen_budgets(stability_root, score_columns)
    budget_config = budget_configs[score_columns[0]]
    warning_definition = load_frozen_warning_definition(config)
    warning_budget_hours = float(warning_definition["false_alarm_hours_per_station_month"])
    inconsistent_budgets = [
        score
        for score, value in budget_configs.items()
        if not np.isclose(warning_budget_hours, value.monthly_budget_hours)
    ]
    if inconsistent_budgets:
        raise ValueError(
            f"Frozen warning and causal-budget monthly hours do not match: {inconsistent_budgets}"
        )
    budget_paths, frame = apply_frozen_budget(
        prediction_paths,
        output_root,
        budget_configs,
        int(config["warning_evaluation"]["year"]),
        int(config["warning_evaluation"]["prediction_compression_level"]),
        score_columns,
    )
    events = pd.read_csv(
        resolve_project_path(config["preprocessed_root"]) / "events_recurrent.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    metrics, records, comparisons, audit = evaluate_deep_warning(
        frame, events, config, budget_configs, score_columns, model_pairs
    )
    warning_baseline_root = resolve_project_path(
        config.get("warning_baseline_root", config["baseline_root"])
    )
    warning_baseline_source = (
        "2022 pooled-OOF calibrated base_cloglog"
        if (warning_baseline_root / "statistical_model_bundle.json").exists()
        else "configured frozen local weather hazard hazard bundle"
    )
    metrics.to_csv(output_root / "warning_metrics.csv", index=False, encoding="utf-8-sig")
    records.to_csv(output_root / "event_records.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output_root / "paired_event_comparisons.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output_root / "monthly_budget_audit.csv", index=False, encoding="utf-8-sig")
    matched_metrics: pd.DataFrame | None = None
    matched_records: pd.DataFrame | None = None
    matched_comparisons: pd.DataFrame | None = None
    matched_audit: pd.DataFrame | None = None
    matched_selection: pd.DataFrame | None = None
    if bool(matched_config.get("enabled", True)):
        matched_metrics, matched_records, matched_comparisons, matched_audit, matched_selection = (
            evaluate_matched_false_alarm_diagnostic(
                frame, events, config, budget_config, score_columns, model_pairs
            )
        )
        matched_metrics.to_csv(
            output_root / "matched_warning_metrics.csv", index=False, encoding="utf-8-sig"
        )
        matched_records.to_csv(
            output_root / "matched_event_records.csv", index=False, encoding="utf-8-sig"
        )
        matched_comparisons.to_csv(
            output_root / "matched_paired_event_comparisons.csv", index=False, encoding="utf-8-sig"
        )
        matched_audit.to_csv(
            output_root / "matched_monthly_distribution_audit.csv", index=False, encoding="utf-8-sig"
        )
        matched_selection.to_csv(
            output_root / "matched_threshold_selection.csv", index=False, encoding="utf-8-sig"
        )
    write_json(
        output_root / "run_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "prediction_files": len(prediction_paths),
            "budget_prediction_files": len(budget_paths),
            "ensemble_seeds": [int(value) for value in config["training"]["seeds"]],
            "ensemble_models": selected_models,
            "warning_baseline_root": str(warning_baseline_root),
            "warning_baseline_source": warning_baseline_source,
            "evaluation_year": evaluation_year,
            "history_split": config["warning_evaluation"].get("history_split", "train_full"),
            "evaluation_split": config["warning_evaluation"].get("evaluation_split", "validation"),
            "budget_config": {field.name: getattr(budget_config, field.name) for field in fields(CausalBudgetConfig)},
            "budget_configs_by_score": {
                score: {field.name: getattr(value, field.name) for field in fields(CausalBudgetConfig)}
                for score, value in budget_configs.items()
            },
            "matched_false_alarm_diagnostic": matched_config,
        },
    )
    (output_root / "warning_report.md").write_text(
        build_warning_report(
            metrics,
            comparisons,
            audit,
            budget_config,
            matched_metrics,
            matched_comparisons,
            matched_audit,
            matched_selection,
            selected_models,
            evaluation_year,
            len(config["training"]["seeds"]),
            warning_baseline_source,
            budget_configs,
        ),
        encoding="utf-8",
    )
    print(f"Deep event-warning evaluation complete: {output_root}", flush=True)
    return output_root

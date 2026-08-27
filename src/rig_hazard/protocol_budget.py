from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_experiment import evaluate_warning_model, observed_warning_rows
from .budget_control import CausalBudgetConfig, apply_causal_budget_by_station
from .config import resolve_project_path


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_oof_ensemble(
    prediction_root: Path,
    model_name: str,
    seeds: list[int],
) -> dict[str, np.ndarray]:
    parts = []
    for seed in seeds:
        path = prediction_root / f"{model_name}_seed_{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as values:
            parts.append({key: values[key].copy() for key in values.files})
    identity_fields = (
        "file_id",
        "row_index",
        "issue_time_ns",
        "onset_within_6h",
        "observed_6h",
    )
    for part in parts[1:]:
        for field in identity_fields:
            if not np.array_equal(parts[0][field], part[field]):
                raise ValueError(
                    f"OOF field {field} is not aligned across seeds for {model_name}"
                )
    return {
        "file_id": parts[0]["file_id"],
        "row_index": parts[0]["row_index"],
        "issue_time_ns": parts[0]["issue_time_ns"],
        "risk_6h": np.mean(np.stack([part["risk_6h"] for part in parts]), axis=0),
        "onset_within_6h": parts[0]["onset_within_6h"],
        "observed_6h": parts[0]["observed_6h"],
    }


def _oof_warning_frame(cache_root: Path, ensemble: dict[str, np.ndarray]) -> pd.DataFrame:
    manifest = json.loads((cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
    files = {int(row["file_id"]): row for row in manifest["files"]}
    count = ensemble["row_index"].size
    station_code = np.empty(count, dtype=object)
    hard_negative = np.zeros(count, dtype=np.int8)
    for file_id in np.unique(ensemble["file_id"]):
        positions = np.flatnonzero(ensemble["file_id"] == file_id)
        metadata = files[int(file_id)]
        rows = ensemble["row_index"][positions].astype(np.int64)
        with np.load(cache_root / metadata["target_path"], allow_pickle=False) as targets:
            hard_negative[positions] = targets["hard_negative_6h"][rows].astype(np.int8)
        station_code[positions] = str(metadata["station_code"])
    frame = pd.DataFrame(
        {
            "station_code": station_code.astype(str),
            "issue_time": pd.to_datetime(ensemble["issue_time_ns"], unit="ns"),
            "risk_6h": ensemble["risk_6h"],
            "onset_within_6h": ensemble["onset_within_6h"],
            "hard_negative_6h": hard_negative,
            "observed_6h": ensemble["observed_6h"],
        }
    )
    frame = frame.sort_values(["station_code", "issue_time"]).reset_index(drop=True)
    frame["station_month"] = frame["station_code"] + "|" + frame["issue_time"].dt.to_period("M").astype(str)
    return frame


def _evaluate_candidate(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    budget: CausalBudgetConfig,
    model_name: str,
    warning_definition: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    controlled = apply_causal_budget_by_station(frame, ["risk_6h"], budget)
    evaluation_frame = observed_warning_rows(controlled, "observed_6h")
    alarm_column = "risk_6h__budget_alarm"
    result = evaluate_warning_model(
        evaluation_frame,
        events,
        alarm_column,
        0.5,
        horizon=int(warning_definition["horizon_hours"]),
        step_minutes=budget.step_minutes,
        false_alarm_budget=budget.monthly_budget_hours,
        operating_point=f"2022_oof_quantile_{budget.candidate_quantile:.4f}",
        minimum_consecutive_alarm_bins=int(warning_definition["minimum_consecutive_alarm_bins"]),
        alarm_merge_gap_minutes=int(warning_definition["alarm_merge_gap_minutes"]),
        maximum_silence_before_event_minutes=int(
            warning_definition["maximum_silence_before_event_minutes"]
        ),
    )
    mean_lead = float(result["mean_effective_lead_hours"])
    hit_rate = float(result["event_hit_rate"])
    result.update(
        {
            "selection_model": model_name,
            "candidate_quantile": budget.candidate_quantile,
            "lead_utility_hours": hit_rate * mean_lead if np.isfinite(mean_lead) else 0.0,
            "total_time_bins": int(controlled.shape[0]),
            "observed_time_bins": int(evaluation_frame.shape[0]),
            "censored_time_bins": int(controlled.shape[0] - evaluation_frame.shape[0]),
        }
    )
    return result, controlled


def select_matched_budget_candidate(
    selection: pd.DataFrame,
    model_name: str,
    target_false_alarm_hours: float,
) -> pd.Series:
    model_rows = selection.loc[selection["selection_model"].eq(model_name)].copy()
    if model_rows.empty:
        raise ValueError(f"No budget candidates are available for {model_name}")
    model_rows["false_alarm_duration_error"] = (
        model_rows["false_alarm_hours_per_station_month"] - float(target_false_alarm_hours)
    ).abs()
    eligible = model_rows.loc[model_rows["budget_met"].eq(1)].copy()
    if eligible.empty:
        eligible = model_rows
    return eligible.sort_values(
        [
            "false_alarm_duration_error",
            "lead_utility_hours",
            "event_hit_rate",
            "mean_effective_lead_hours",
            "candidate_quantile",
        ],
        ascending=[True, False, False, False, False],
    ).iloc[0]


def select_2022_oof_budget(
    config: dict[str, Any],
    config_path: Path,
    deep_protocol_root: str | Path | None = None,
    output_root_override: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    cache_root = resolve_project_path(config["cache_root"])
    protocol_root = resolve_project_path(
        deep_protocol_root
        or config.get(
            "deep_protocol_output_root",
            "results/recurrence_analysis/model_comparison/deep_protocol",
        )
    )
    output_root = resolve_project_path(output_root_override or config["stability_root"])
    _prepare_output(output_root, overwrite)
    settings = dict(config.get("budget_selection", {}))
    if "warning" not in config:
        raise ValueError("The recurrence experiment config must define the frozen warning settings")
    warning_definition = dict(config["warning"])
    required_warning = {
        "horizon_hours",
        "false_alarm_hours_per_station_month",
        "minimum_consecutive_alarm_bins",
        "alarm_merge_gap_minutes",
        "maximum_silence_before_event_minutes",
    }
    missing_warning = sorted(required_warning - set(warning_definition))
    if missing_warning:
        raise ValueError(f"Frozen warning settings are missing fields: {missing_warning}")
    if int(warning_definition["horizon_hours"]) != 6:
        raise ValueError("The OOF warning-budget selector currently requires a 6-hour horizon")
    selection_models = [
        str(value)
        for value in settings.get(
            "selection_models",
            [settings.get("selection_model", "gru")],
        )
    ]
    if not selection_models or len(selection_models) != len(set(selection_models)):
        raise ValueError("budget_selection.selection_models must contain unique model names")
    reference_model = str(settings.get("reference_model", selection_models[0]))
    if reference_model not in selection_models:
        raise ValueError("budget_selection.reference_model must be one of selection_models")
    seeds = [int(value) for value in config["training"]["seeds"]]
    events = pd.read_csv(resolve_project_path(config["preprocessed_root"]) / "events_recurrent.csv", low_memory=False)
    events["onset_time"] = pd.to_datetime(events["onset_time"], errors="coerce")
    events["valid_target_event"] = pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).astype(int)
    events = events.loc[events["onset_time"].dt.year.eq(2022)].copy()
    base = CausalBudgetConfig(
        step_minutes=int(config["step_minutes"]),
        monthly_budget_hours=float(settings.get("monthly_budget_hours", 10.0)),
        trailing_history_days=int(settings.get("trailing_history_days", 30)),
        candidate_quantile=0.98,
        minimum_history_rows=int(settings.get("minimum_history_rows", 144)),
        burst_allowance_hours=float(settings.get("burst_allowance_hours", 1.0)),
        minimum_candidate_run_bins=int(settings.get("minimum_candidate_run_bins", 2)),
        minimum_alarm_run_bins=int(settings.get("minimum_alarm_run_bins", 2)),
    )
    rows: list[dict[str, Any]] = []
    controlled_by_candidate: dict[tuple[str, float], pd.DataFrame] = {}
    candidate_quantiles = [
        float(value)
        for value in settings.get(
            "candidate_quantiles",
            [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995],
        )
    ]
    if (
        not candidate_quantiles
        or len(candidate_quantiles) != len(set(candidate_quantiles))
        or any(not 0.0 < value < 1.0 for value in candidate_quantiles)
    ):
        raise ValueError("candidate_quantiles must contain unique values strictly between zero and one")
    for selection_model in selection_models:
        ensemble = _load_oof_ensemble(protocol_root / "oof_predictions", selection_model, seeds)
        frame = _oof_warning_frame(cache_root, ensemble)
        if set(frame["issue_time"].dt.year.unique()) != {2022}:
            raise ValueError("Budget selection predictions must contain only 2022 OOF rows")
        for quantile in candidate_quantiles:
            budget = CausalBudgetConfig(**{**asdict(base), "candidate_quantile": quantile})
            result, controlled = _evaluate_candidate(
                frame,
                events,
                budget,
                selection_model,
                warning_definition,
            )
            rows.append(result)
            controlled_by_candidate[(selection_model, quantile)] = controlled
    selection = pd.DataFrame(rows)
    configured_warning_budget = float(warning_definition["false_alarm_hours_per_station_month"])
    if not np.isclose(configured_warning_budget, base.monthly_budget_hours):
        raise ValueError(
            "warning.false_alarm_hours_per_station_month must equal "
            "budget_selection.monthly_budget_hours"
        )
    selection["false_alarm_duration_error"] = (
        selection["false_alarm_hours_per_station_month"] - configured_warning_budget
    ).abs()
    selection["selected"] = np.int8(0)
    selected_budgets: dict[str, CausalBudgetConfig] = {}
    selected_metrics: dict[str, dict[str, Any]] = {}
    for selection_model in selection_models:
        chosen = select_matched_budget_candidate(
            selection,
            selection_model,
            configured_warning_budget,
        )
        selected_quantile = float(chosen["candidate_quantile"])
        selected_budget = CausalBudgetConfig(
            **{**asdict(base), "candidate_quantile": selected_quantile}
        )
        selected_budgets[selection_model] = selected_budget
        selected_metrics[selection_model] = json.loads(chosen.to_json())
        selected_mask = selection["selection_model"].eq(selection_model) & selection[
            "candidate_quantile"
        ].eq(selected_quantile)
        selection.loc[selected_mask, "selected"] = np.int8(1)
        controlled_by_candidate[(selection_model, selected_quantile)].to_csv(
            output_root / f"selected_2022_oof_budget_predictions_{selection_model}.csv.gz",
            index=False,
            compression="gzip",
        )
    selection.to_csv(output_root / "causal_budget_quantile_selection.csv", index=False)
    reference_budget = selected_budgets[reference_model]
    score_budget_configs = {
        f"{model_name}_ensemble_6h": asdict(budget)
        for model_name, budget in selected_budgets.items()
    }
    bundle = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "selection_year": 2022,
        "selection_source": "pooled block-OOF five-seed ensemble",
        "selection_models": selection_models,
        "reference_model": reference_model,
        "selection_model": reference_model,
        "causal_budget_config": asdict(reference_budget),
        "causal_budget_configs_by_score": score_budget_configs,
        "warning_definition": warning_definition,
        "selected_metrics": selected_metrics,
    }
    (output_root / "model_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    report = [
        "# 2022 OOF causal warning-budget selection",
        "",
        f"Selection models: {', '.join(selection_models)}.",
        f"Reference model for unmapped comparators: {reference_model}.",
        f"Monthly hard alarm cap: {reference_budget.monthly_budget_hours:.1f} hours per station.",
        "Each deep model receives a model-specific causal candidate quantile. Selection first minimizes the 2022 OOF false-alarm-duration error, then maximizes warning utility.",
        "All candidate selection uses complete 2022 out-of-fold probabilities. Every selected configuration remains unchanged in 2023 and 2024.",
        "",
        selection.to_markdown(index=False),
    ]
    (output_root / "budget_selection_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"2022 OOF budget selection complete: {output_root}", flush=True)
    return output_root

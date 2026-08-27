from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.alert_governance import (
    CandidatePolicy,
    SafeBudgetPolicy,
    STRICT_EVENT_CONTRACT_VERSION,
    apply_candidate_policy_by_entity,
    station_month_budget_distribution,
    strict_event_alert_evaluation,
)
from rig_hazard.baseline_experiment import observed_warning_rows
from rig_hazard.budget_control import CausalBudgetConfig, apply_causal_budget_by_station
from rig_hazard.cluster_bootstrap import station_cluster_bootstrap_frame
from rig_hazard.config import resolve_project_path
from rig_hazard.evaluation_provenance import file_bundle_manifest, sha256_file
from rig_hazard.event_eligibility import (
    add_event_interval_metadata,
    annotate_timeline_exclusion_reasons,
    eligibility_bias_summary,
    eligibility_funnel,
    eligibility_station_season_distribution,
)
from rig_hazard.nested_alert import evaluate_event_alerts
from rig_hazard.utility_warning import _ensemble_frame, _events


STRATEGIES = (
    "fixed_threshold",
    "rolling_quantile",
    "hysteresis",
    "original_simple",
    "budget_safe",
)


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_policy(parameters: dict[str, Any]) -> CandidatePolicy:
    allowed = CandidatePolicy.__dataclass_fields__
    return CandidatePolicy(**{key: value for key, value in parameters.items() if key in allowed})


def _safe_policy(parameters: dict[str, Any], step_minutes: int, budget: float) -> SafeBudgetPolicy:
    return SafeBudgetPolicy(
        step_minutes=step_minutes,
        budget_hours=budget,
        burst_allowance_hours=float(parameters.get("burst_allowance_hours", 0.5)),
        pace_multiplier=float(parameters.get("pace_multiplier", 1.0)),
    )


def _apply(
    frame: pd.DataFrame,
    strategy: str,
    parameters: dict[str, Any],
    budget: float,
    step_minutes: int,
) -> pd.DataFrame:
    if strategy == "original_simple":
        controlled = apply_causal_budget_by_station(
            frame,
            ["risk_6h"],
            CausalBudgetConfig(
                step_minutes=step_minutes,
                monthly_budget_hours=budget,
                trailing_history_days=int(parameters.get("trailing_history_days", 30)),
                candidate_quantile=float(parameters["rolling_quantile"]),
                minimum_history_rows=int(parameters.get("minimum_history_rows", 144)),
                burst_allowance_hours=float(parameters.get("burst_allowance_hours", 1.0)),
                minimum_candidate_run_bins=int(parameters.get("minimum_candidate_run_bins", 2)),
                minimum_alarm_run_bins=int(parameters.get("minimum_alarm_run_bins", 2)),
            ),
        )
        return controlled.rename(
            columns={
                "risk_6h__candidate_alarm": "candidate_alarm",
                "risk_6h__causal_threshold": "causal_threshold",
                "risk_6h__budget_alarm": "budget_alarm",
            }
        )
    safe = (
        _safe_policy(parameters, step_minutes, budget)
        if strategy == "budget_safe"
        else None
    )
    return apply_candidate_policy_by_entity(
        frame,
        "risk_6h",
        _candidate_policy(parameters),
        safe_budget=safe,
    )


def _evaluate(
    controlled: pd.DataFrame,
    events: pd.DataFrame,
    budget: float,
    step_minutes: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    strict, records, strict_frame = strict_event_alert_evaluation(
        controlled,
        events,
        alarm_column="budget_alarm",
        step_minutes=step_minutes,
        horizon_hours=6,
    )
    monthly, distribution = station_month_budget_distribution(
        strict_frame,
        "budget_alarm",
        budget,
        step_minutes,
        strict_false_column="_strict_false_alarm",
        unsettled_column="_strict_unsettled_alarm",
        reserved_column="_strict_reserved_alarm",
    )
    legacy_controlled = observed_warning_rows(controlled, "observed_6h")
    legacy, _ = evaluate_event_alerts(
        legacy_controlled,
        events,
        "budget_alarm",
        budget,
        step_minutes=step_minutes,
    )
    station_months = max(int(distribution["station_months"]), 1)
    strict_fah = strict["strict_false_alarm_bins"] * step_minutes / 60.0 / station_months
    unsettled = (
        strict["strict_unsettled_alarm_bins"] * step_minutes / 60.0 / station_months
    )
    reserved = strict["strict_reserved_alarm_bins"] * step_minutes / 60.0 / station_months
    result = {
        "evaluation_contract_version": STRICT_EVENT_CONTRACT_VERSION,
        **{f"strict_{key}": value for key, value in strict.items()},
        **{f"legacy_{key}": value for key, value in legacy.items()},
        **distribution,
        "strict_false_alarm_hours_per_station_month": float(strict_fah),
        "strict_unsettled_alarm_hours_per_station_month": float(unsettled),
        "strict_reserved_alarm_hours_per_station_month": float(reserved),
        "budget_met_mean_fah": int(strict_fah <= budget + 1e-9),
        "budget_met_mean_reserved_capacity": int(reserved <= budget + 1e-9),
        "budget_met_every_station_month": int(
            distribution["maximum_false_alarm_hours"] <= budget + 1e-9
        ),
        "budget_met_every_station_month_reserved": int(
            distribution["maximum_reserved_alarm_hours"] <= budget + 1e-9
        ),
    }
    return result, records, monthly


def _parameter_grid(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    budget: float,
) -> list[tuple[str, dict[str, Any]]]:
    score = pd.to_numeric(frame["risk_6h"], errors="coerce").to_numpy(dtype=np.float64)
    finite = score[np.isfinite(score)]
    absolute_quantiles = [float(value) for value in settings["absolute_quantiles"]]
    absolute = {
        quantile: float(np.quantile(finite, quantile)) for quantile in absolute_quantiles
    }
    rows: list[tuple[str, dict[str, Any]]] = []
    for quantile, threshold in absolute.items():
        rows.append(
            (
                "fixed_threshold",
                asdict(CandidatePolicy("fixed_threshold", threshold=threshold)),
            )
        )
    for quantile in settings["rolling_quantiles"]:
        rows.append(
            (
                "rolling_quantile",
                asdict(
                    CandidatePolicy(
                        "rolling_quantile",
                        rolling_quantile=float(quantile),
                        trailing_history_days=int(settings["trailing_history_days"]),
                        minimum_history_rows=int(settings["minimum_history_rows"]),
                    )
                ),
            )
        )
    for quantile, threshold in absolute.items():
        for ratio in settings["hysteresis_off_ratios"]:
            rows.append(
                (
                    "hysteresis",
                    asdict(
                        CandidatePolicy(
                            "hysteresis",
                            threshold=threshold,
                            ema_alpha=float(settings["hysteresis_ema_alpha"]),
                            off_threshold_ratio=float(ratio),
                            minimum_consecutive_bins=int(settings["minimum_consecutive_bins"]),
                            hold_bins=int(settings["hold_bins"]),
                        )
                    ),
                )
            )
    for quantile in settings["rolling_quantiles"]:
        rows.append(
            (
                "original_simple",
                {
                    "rolling_quantile": float(quantile),
                    "trailing_history_days": int(settings["trailing_history_days"]),
                    "minimum_history_rows": int(settings["minimum_history_rows"]),
                    "burst_allowance_hours": float(settings["original_burst_hours"]),
                    "minimum_candidate_run_bins": int(settings["minimum_consecutive_bins"]),
                    "minimum_alarm_run_bins": int(settings["hold_bins"]),
                },
            )
        )
    for quantile, threshold in absolute.items():
        for ratio in settings["hysteresis_off_ratios"]:
            parameters = asdict(
                CandidatePolicy(
                    "hysteresis",
                    threshold=threshold,
                    ema_alpha=float(settings["hysteresis_ema_alpha"]),
                    off_threshold_ratio=float(ratio),
                    minimum_consecutive_bins=int(settings["minimum_consecutive_bins"]),
                    hold_bins=int(settings["hold_bins"]),
                )
            )
            parameters.update(
                {
                    "burst_allowance_hours": float(settings["safe_burst_hours"]),
                    "pace_multiplier": float(settings["safe_pace_multiplier"]),
                }
            )
            rows.append(("budget_safe", parameters))
    unique: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for strategy, parameters in rows:
        unique[(strategy, _json(parameters))] = (strategy, parameters)
    return list(unique.values())


def _select_utility(candidates: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for (model, budget, strategy), rows in candidates.groupby(
        ["model", "budget_hours", "strategy"], sort=True
    ):
        eligible = rows.loc[rows["budget_met_mean_fah"].eq(1)].copy()
        if strategy in {"original_simple", "budget_safe"}:
            eligible = eligible.loc[eligible["budget_met_every_station_month"].eq(1)]
        if eligible.empty:
            eligible = rows.copy()
            selection_feasible = 0
        else:
            selection_feasible = 1
        row = eligible.sort_values(
            [
                "strict_lead_utility_hours",
                "strict_event_hit_rate",
                "strict_mean_effective_lead_hours",
                "strict_false_alarm_hours_per_station_month",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]
        row = row.copy()
        row["selection_feasible_under_2022_budget_rule"] = selection_feasible
        selected.append(row)
    result = pd.DataFrame(selected)
    result["selection_mode"] = "max_utility_subject_to_budget"
    return result


def _select_matched_fah(candidates: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for (model, budget), all_rows in candidates.groupby(["model", "budget_hours"], sort=True):
        maxima: list[float] = []
        for strategy in STRATEGIES:
            rows = all_rows.loc[
                all_rows["strategy"].eq(strategy)
                & all_rows["budget_met_mean_fah"].eq(1)
            ]
            if not rows.empty:
                maxima.append(float(rows["strict_false_alarm_hours_per_station_month"].max()))
        if len(maxima) != len(STRATEGIES):
            continue
        target = min(maxima)
        for strategy in STRATEGIES:
            rows = all_rows.loc[
                all_rows["strategy"].eq(strategy)
                & all_rows["budget_met_mean_fah"].eq(1)
            ].copy()
            rows["fah_match_absolute_error"] = (
                rows["strict_false_alarm_hours_per_station_month"] - target
            ).abs()
            row = rows.sort_values(
                ["fah_match_absolute_error", "strict_lead_utility_hours"],
                ascending=[True, False],
            ).iloc[0]
            row = row.copy()
            row["matched_fah_target"] = target
            selected.append(row)
    result = pd.DataFrame(selected)
    result["selection_feasible_under_2022_budget_rule"] = pd.NA
    result["selection_mode"] = "matched_2022_oof_actual_fah"
    return result


def _optional_int(value: Any) -> int | None:
    """Preserve non-applicable diagnostic fields instead of coercing NaN to int."""
    return None if pd.isna(value) else int(value)


def _load_model_frame(
    cache_root: Path,
    model: dict[str, Any],
    year: int | None,
) -> pd.DataFrame:
    root = resolve_project_path(model["root"])
    prediction_root = (
        root / "oof_predictions"
        if year is None
        else root / "locked_predictions" / str(year)
    )
    return _ensemble_frame(
        cache_root,
        prediction_root,
        str(model["prediction_name"]),
        [int(value) for value in model["seeds"]],
    )


def _prediction_paths(model: dict[str, Any], year: int | None) -> list[Path]:
    root = resolve_project_path(model["root"])
    prediction_root = (
        root / "oof_predictions"
        if year is None
        else root / "locked_predictions" / str(year)
    )
    return [
        prediction_root / f"{model['prediction_name']}_seed_{int(seed)}.npz"
        for seed in model["seeds"]
    ]


def _event_eligibility_records(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    step_minutes: int,
) -> pd.DataFrame:
    frame["_eligibility_alarm"] = 0
    try:
        _, records, _ = strict_event_alert_evaluation(
            frame,
            events,
            alarm_column="_eligibility_alarm",
            step_minutes=step_minutes,
            horizon_hours=6,
        )
    finally:
        frame.drop(columns=["_eligibility_alarm"], inplace=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen alert controllers at fair FAH.")
    parser.add_argument("--config", default="configs/rig_hazard_alert_governance.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one candidate per strategy, one budget, and the main model for contract QA.",
    )
    args = parser.parse_args()

    config = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    settings = dict(config["strategy_comparison"])
    if args.smoke:
        settings["models"] = [
            model for model in settings["models"] if str(model["name"]) == "rec_none"
        ]
        settings["budgets_hours"] = settings["budgets_hours"][:1]
        settings["absolute_quantiles"] = settings["absolute_quantiles"][:1]
        settings["rolling_quantiles"] = settings["rolling_quantiles"][:1]
        settings["hysteresis_off_ratios"] = settings["hysteresis_off_ratios"][:1]
        settings["bootstrap_samples"] = 20
    output_root = resolve_project_path(args.output_root or settings["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / ".complete_alert_strategy_comparison"
    if args.resume and marker.exists():
        print(f"Alert strategy comparison already complete: {output_root}", flush=True)
        return

    cache_root = resolve_project_path(config["cache_root"])
    event_table_path = resolve_project_path(config["seasonal_event_mapping"])
    events = _events(config, event_table_path)
    event_table_sha256 = sha256_file(event_table_path)
    cache_config = json.loads((cache_root / "resolved_config.json").read_text(encoding="utf-8"))
    timeline_root = resolve_project_path(cache_config["preprocessed_root"]) / "timelines"
    step_minutes = int(config["step_minutes"])
    budgets = [float(value) for value in settings["budgets_hours"]]
    model_lock = {
        "main_model": "rec_none",
        "baselines": ["fair_gru", "fair_timesnet", "xgboost"],
        "stopped_extensions": ["rec_full", "dual_time_scale", "station_graph"],
        "selection_year": 2022,
        "locked_evaluation_years": [2023, 2024],
    }
    (output_root / "model_lock.json").write_text(
        json.dumps(model_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    candidate_parts: list[pd.DataFrame] = []
    selection_frames: dict[str, pd.DataFrame] = {}
    prediction_manifests: dict[str, dict[str, Any]] = {}
    for model in settings["models"]:
        model_name = str(model["name"])
        selection_bundle_key = f"{model_name}|2022_oof"
        prediction_manifests[selection_bundle_key] = file_bundle_manifest(
            _prediction_paths(model, None), PROJECT_ROOT
        )
        selection_prediction_sha256 = prediction_manifests[selection_bundle_key][
            "bundle_sha256"
        ]
        selection = _load_model_frame(cache_root, model, None)
        selection_frames[model_name] = selection
        selection_events = events.loc[events["onset_time"].dt.year.eq(2022)]
        cache_path = output_root / "selection_candidates" / f"{model_name}.csv.gz"
        if args.resume and cache_path.exists():
            candidate_parts.append(pd.read_csv(cache_path))
            print(f"Selection candidates reused: {model_name}", flush=True)
            continue
        rows: list[dict[str, Any]] = []
        for budget in budgets:
            grid = _parameter_grid(selection, settings, budget)
            for index, (strategy, parameters) in enumerate(grid, start=1):
                controlled = _apply(selection, strategy, parameters, budget, step_minutes)
                metrics, _, _ = _evaluate(
                    controlled, selection_events, budget, step_minutes
                )
                rows.append(
                    {
                        "model": model_name,
                        "budget_hours": budget,
                        "strategy": strategy,
                        "parameters": _json(parameters),
                        "event_table_sha256": event_table_sha256,
                        "prediction_bundle_sha256": selection_prediction_sha256,
                        **metrics,
                    }
                )
                if index % 10 == 0 or index == len(grid):
                    print(
                        f"Selection grid {model_name} B={budget:g}: {index}/{len(grid)}",
                        flush=True,
                    )
        part = pd.DataFrame(rows)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(cache_path, index=False, compression="gzip")
        candidate_parts.append(part)
    candidates = pd.concat(candidate_parts, ignore_index=True)
    candidates.to_csv(
        output_root / "selection_candidate_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    utility_selected = _select_utility(candidates)
    matched_selected = _select_matched_fah(candidates)
    selected = pd.concat([utility_selected, matched_selected], ignore_index=True)
    selected.to_csv(output_root / "selected_policies.csv", index=False)
    selected_policy_file_sha256 = sha256_file(output_root / "selected_policies.csv")

    frozen_rows: list[dict[str, Any]] = []
    monthly_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    eligibility_parts: list[pd.DataFrame] = []
    for model in settings["models"]:
        model_name = str(model["name"])
        for year in (2023, 2024):
            bundle_key = f"{model_name}|{year}_frozen"
            prediction_manifests[bundle_key] = file_bundle_manifest(
                _prediction_paths(model, year), PROJECT_ROOT
            )
        year_frames = {
            year: _load_model_frame(cache_root, model, year) for year in (2023, 2024)
        }
        if model_name == model_lock["main_model"]:
            for audit_year, audit_frame in (
                (2022, selection_frames[model_name]),
                (2023, year_frames[2023]),
                (2024, year_frames[2024]),
            ):
                year_events = events.loc[events["onset_time"].dt.year.eq(audit_year)].copy()
                audit = _event_eligibility_records(audit_frame, year_events, step_minutes)
                audit["year"] = audit_year
                audit["prediction_bundle_sha256"] = prediction_manifests[
                    f"{model_name}|{'2022_oof' if audit_year == 2022 else f'{audit_year}_frozen'}"
                ]["bundle_sha256"]
                eligibility_parts.append(audit)
        for year in (2023, 2024):
            history = selection_frames[model_name] if year == 2023 else year_frames[2023]
            combined = pd.concat([history, year_frames[year]], ignore_index=True).sort_values(
                ["station_code", "issue_time"]
            )
            year_events = events.loc[events["onset_time"].dt.year.eq(year)]
            policies = selected.loc[selected["model"].eq(model_name)]
            prediction_bundle_sha256 = prediction_manifests[
                f"{model_name}|{year}_frozen"
            ]["bundle_sha256"]
            for row in policies.itertuples(index=False):
                budget = float(row.budget_hours)
                parameters = json.loads(row.parameters)
                controlled_all = _apply(
                    combined, str(row.strategy), parameters, budget, step_minutes
                )
                controlled = controlled_all.loc[
                    pd.to_datetime(controlled_all["issue_time"]).dt.year.eq(year)
                ].copy()
                metrics, records, monthly = _evaluate(
                    controlled, year_events, budget, step_minutes
                )
                policy_id = (
                    f"{model_name}|{row.selection_mode}|{row.strategy}|B{budget:g}"
                )
                frozen_rows.append(
                    {
                        "policy_id": policy_id,
                        "model": model_name,
                        "model_role": str(model["role"]),
                        "year": year,
                        "budget_hours": budget,
                        "strategy": str(row.strategy),
                        "selection_mode": str(row.selection_mode),
                        "selection_feasible_under_2022_budget_rule": _optional_int(
                            row.selection_feasible_under_2022_budget_rule
                        ),
                        "parameters": row.parameters,
                        "event_table_sha256": event_table_sha256,
                        "prediction_bundle_sha256": prediction_bundle_sha256,
                        "selection_policy_file_sha256": selected_policy_file_sha256,
                        **metrics,
                    }
                )
                monthly["policy_id"] = policy_id
                monthly["model"] = model_name
                monthly["year"] = year
                monthly["strategy"] = str(row.strategy)
                monthly["selection_mode"] = str(row.selection_mode)
                monthly["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
                monthly["event_table_sha256"] = event_table_sha256
                monthly["prediction_bundle_sha256"] = prediction_bundle_sha256
                monthly["selection_policy_file_sha256"] = selected_policy_file_sha256
                monthly_parts.append(monthly)
                if str(row.selection_mode) == "max_utility_subject_to_budget":
                    records["policy_id"] = policy_id
                    records["model"] = model_name
                    records["year"] = year
                    records["strategy"] = str(row.strategy)
                    records["budget_hours"] = budget
                    records["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
                    records["event_table_sha256"] = event_table_sha256
                    records["prediction_bundle_sha256"] = prediction_bundle_sha256
                    records["selection_policy_file_sha256"] = selected_policy_file_sha256
                    event_parts.append(records)
            print(f"Frozen controller evaluation complete: {model_name} {year}", flush=True)
    frozen = pd.DataFrame(frozen_rows)
    monthly_all = pd.concat(monthly_parts, ignore_index=True)
    event_all = pd.concat(event_parts, ignore_index=True)
    main_frozen = frozen.loc[
        frozen["model"].eq(model_lock["main_model"])
        & frozen["selection_mode"].eq("max_utility_subject_to_budget")
    ].sort_values(["year", "budget_hours", "strategy"]).copy()
    bootstrap_parts: list[pd.DataFrame] = []
    bootstrap_samples = int(settings.get("bootstrap_samples", 5000))
    bootstrap_seed = int(settings.get("bootstrap_seed", 20260824))
    for row in main_frozen.itertuples(index=False):
        event_subset = event_all.loc[
            event_all["policy_id"].eq(row.policy_id) & event_all["year"].eq(row.year)
        ]
        monthly_subset = monthly_all.loc[
            monthly_all["policy_id"].eq(row.policy_id)
            & monthly_all["year"].eq(row.year)
        ]
        bootstrap = station_cluster_bootstrap_frame(
            event_subset,
            monthly_subset,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        bootstrap["policy_id"] = row.policy_id
        bootstrap["model"] = row.model
        bootstrap["year"] = row.year
        bootstrap["budget_hours"] = row.budget_hours
        bootstrap["strategy"] = row.strategy
        bootstrap["estimate"] = [
            getattr(row, metric) if hasattr(row, metric) else np.nan
            for metric in bootstrap["metric"]
        ]
        bootstrap["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
        bootstrap["event_table_sha256"] = event_table_sha256
        bootstrap["prediction_bundle_sha256"] = row.prediction_bundle_sha256
        bootstrap["selection_policy_file_sha256"] = selected_policy_file_sha256
        bootstrap_parts.append(bootstrap)
    bootstrap_all = pd.concat(bootstrap_parts, ignore_index=True)
    ci_wide = bootstrap_all.pivot(
        index=["policy_id", "year"],
        columns="metric",
        values=["ci_lower_2_5", "ci_upper_97_5"],
    )
    ci_wide.columns = [f"{metric}_{bound}" for bound, metric in ci_wide.columns]
    ci_wide = ci_wide.reset_index()
    main_frozen = main_frozen.merge(
        ci_wide, on=["policy_id", "year"], how="left", validate="one_to_one"
    )
    frozen.to_csv(output_root / "frozen_strategy_metrics.csv", index=False)
    main_frozen.to_csv(output_root / "frozen_main_baseline_table.csv", index=False)
    bootstrap_all.to_csv(
        output_root / "frozen_main_station_cluster_bootstrap.csv", index=False
    )
    monthly_all.to_csv(
        output_root / "station_month_metrics.csv.gz", index=False, compression="gzip"
    )
    event_all.to_csv(
        output_root / "strict_event_records.csv.gz", index=False, compression="gzip"
    )
    eligibility = pd.concat(eligibility_parts, ignore_index=True)
    eligibility = add_event_interval_metadata(eligibility, events)
    eligibility = annotate_timeline_exclusion_reasons(
        eligibility, timeline_root, step_minutes
    )
    eligibility["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    eligibility["event_table_sha256"] = event_table_sha256
    eligibility["selection_policy_file_sha256"] = selected_policy_file_sha256
    if eligibility.duplicated("event_id").any():
        raise RuntimeError("event_eligibility_audit.csv must contain exactly one row per event")
    eligibility.to_csv(output_root / "event_eligibility_audit.csv", index=False)
    funnel = eligibility_funnel(eligibility)
    funnel["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    funnel["event_table_sha256"] = event_table_sha256
    funnel["selection_policy_file_sha256"] = selected_policy_file_sha256
    funnel.to_csv(output_root / "event_eligibility_funnel.csv", index=False)
    bias = eligibility_bias_summary(eligibility)
    bias["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    bias["event_table_sha256"] = event_table_sha256
    bias["selection_policy_file_sha256"] = selected_policy_file_sha256
    bias.to_csv(output_root / "event_eligibility_bias_summary.csv", index=False)
    station_season = eligibility_station_season_distribution(eligibility)
    station_season["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    station_season["event_table_sha256"] = event_table_sha256
    station_season["selection_policy_file_sha256"] = selected_policy_file_sha256
    station_season.to_csv(
        output_root / "event_eligibility_station_season.csv", index=False
    )
    (output_root / "prediction_bundle_manifest.json").write_text(
        json.dumps(prediction_manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evaluation_contract = {
        "version": STRICT_EVENT_CONTRACT_VERSION,
        "event_window": (
            "36 strictly increasing 10-minute predictions ending at the last grid point "
            "strictly before raw onset_time"
        ),
        "standardized_queue": "all 36 legal risk-set prediction bins are present",
        "operational_queue": "at least one legal risk-set prediction exists within 6 hours",
        "known_event_censoring_rule": (
            "event-table matches remain true alarms even when cached future labels are censored"
        ),
        "unmatched_alarm_rule": (
            "unmatched alarms are false only after complete follow-up; otherwise unsettled and reserved"
        ),
        "event_table_sha256": event_table_sha256,
        "selected_policy_file_sha256": selected_policy_file_sha256,
    }
    (output_root / "evaluation_contract.json").write_text(
        json.dumps(evaluation_contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_lock": model_lock,
        "strategies": list(STRATEGIES),
        "budgets_hours": budgets,
        "primary_selection": "maximize strict event utility subject to 2022 OOF mean FAH <= B",
        "safety_requirement": (
            "original_simple and budget_safe additionally require every 2022 station-month <= B; "
            "budget_safe enforces a causal total-alarm hard cap on every locked station-month"
        ),
        "fair_comparison": (
            "secondary policies are matched to a common achieved 2022 OOF FAH; 2023/2024 "
            "remain frozen and are not re-tuned"
        ),
        "evaluation_contract_version": STRICT_EVENT_CONTRACT_VERSION,
        "event_table_sha256": event_table_sha256,
        "selected_policy_file_sha256": selected_policy_file_sha256,
        "main_frozen_table": (
            "frozen_main_baseline_table.csv contains five strategies x two frozen years x four budgets; "
            "no frozen-year winner is selected"
        ),
        "station_cluster_bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "cluster_unit": "station",
            "paired_draws_across_frozen_policies": True,
        },
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker.touch()
    print(f"Alert strategy comparison complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()

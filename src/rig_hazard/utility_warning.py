from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .baseline_experiment import evaluate_warning_model, observed_warning_rows
from .budget_control import CausalBudgetConfig, apply_causal_budget_by_station
from .config import resolve_project_path
from .graph_experiment import event_alarm_records
from .protocol_budget import _load_oof_ensemble, _oof_warning_frame


DEFAULT_MODELS = (
    "rec_none",
    "rec_load",
    "rec_previous",
    "rec_full",
    "rec_full_uniform",
)
DEFAULT_COMPARISONS = (
    ("rec_load", "rec_none"),
    ("rec_previous", "rec_none"),
    ("rec_full", "rec_none"),
    ("rec_full", "rec_full_uniform"),
)
SEASONAL_STRATA = ("all", "first", "second", "third_plus", "recurrent")
SEASON_COHORTS = ("all_seasons", "complete_seasons", "censored_seasons")


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def select_utility_candidate(
    selection: pd.DataFrame,
    model_name: str,
    monthly_budget_hours: float,
) -> pd.Series:
    """Select the highest event utility among candidates that meet the FAR budget."""

    rows = selection.loc[
        selection["selection_model"].eq(model_name)
        & np.isclose(selection["monthly_budget_hours"], float(monthly_budget_hours))
    ].copy()
    if rows.empty:
        raise ValueError(
            f"No candidate rows for model={model_name}, budget={monthly_budget_hours}"
        )
    eligible = rows.loc[rows["budget_met"].eq(1)].copy()
    if eligible.empty:
        raise ValueError(
            f"No candidate meets the FAR budget for model={model_name}, "
            f"budget={monthly_budget_hours}"
        )
    return eligible.sort_values(
        [
            "lead_utility_hours",
            "event_hit_rate",
            "mean_effective_lead_hours",
            "false_alarm_hours_per_station_month",
            "candidate_quantile",
        ],
        ascending=[False, False, False, True, False],
    ).iloc[0]


def _events(config: dict[str, Any], seasonal_mapping_path: Path) -> pd.DataFrame:
    events = pd.read_csv(seasonal_mapping_path, low_memory=False)
    events["event_id"] = events["event_id"].astype(str)
    events["station_code"] = events["station_code"].astype(str)
    events["onset_time"] = pd.to_datetime(events["onset_time"], errors="coerce")
    events["valid_target_event"] = (
        pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).astype(int)
    )
    for column in (
        "within_season_event_order",
        "complete_icing_season",
        "left_censored_season",
        "right_censored_season",
    ):
        events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0).astype(int)
    valid = events.loc[events["valid_target_event"].eq(1) & events["onset_time"].notna()].copy()
    if valid.empty:
        raise ValueError("The seasonal recurrence mapping contains no valid target events")
    return valid


def _warning_settings(config: dict[str, Any]) -> dict[str, Any]:
    warning = dict(config["warning"])
    if int(warning["horizon_hours"]) != 6:
        raise ValueError("Utility warning experiments require a 6-hour horizon")
    return warning


def _base_budget(
    config: dict[str, Any], monthly_budget_hours: float, candidate_quantile: float
) -> CausalBudgetConfig:
    settings = dict(config.get("budget_selection", {}))
    return CausalBudgetConfig(
        step_minutes=int(config["step_minutes"]),
        monthly_budget_hours=float(monthly_budget_hours),
        trailing_history_days=int(settings.get("trailing_history_days", 30)),
        candidate_quantile=float(candidate_quantile),
        minimum_history_rows=int(settings.get("minimum_history_rows", 144)),
        burst_allowance_hours=float(settings.get("burst_allowance_hours", 1.0)),
        minimum_candidate_run_bins=int(settings.get("minimum_candidate_run_bins", 2)),
        minimum_alarm_run_bins=int(settings.get("minimum_alarm_run_bins", 2)),
    )


def _ensemble_frame(
    cache_root: Path,
    prediction_root: Path,
    model_name: str,
    seeds: list[int],
) -> pd.DataFrame:
    ensemble = _load_oof_ensemble(prediction_root, model_name, seeds)
    return _oof_warning_frame(cache_root, ensemble)


def _evaluate_controlled(
    controlled: pd.DataFrame,
    events: pd.DataFrame,
    model_name: str,
    budget: CausalBudgetConfig,
    warning: dict[str, Any],
    operating_point: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    evaluation_frame = observed_warning_rows(controlled, "observed_6h")
    alarm_column = "risk_6h__budget_alarm"
    result = evaluate_warning_model(
        evaluation_frame,
        events,
        alarm_column,
        0.5,
        horizon=6,
        step_minutes=budget.step_minutes,
        false_alarm_budget=budget.monthly_budget_hours,
        operating_point=operating_point,
        minimum_consecutive_alarm_bins=int(warning["minimum_consecutive_alarm_bins"]),
        alarm_merge_gap_minutes=int(warning["alarm_merge_gap_minutes"]),
        maximum_silence_before_event_minutes=int(
            warning["maximum_silence_before_event_minutes"]
        ),
    )
    records = event_alarm_records(
        evaluation_frame,
        events,
        alarm_column,
        0.5,
        horizon=6,
        warning=warning,
    )
    records["model"] = model_name
    hit_rate = float(result["event_hit_rate"])
    mean_lead = float(result["mean_effective_lead_hours"])
    result["lead_utility_hours"] = (
        hit_rate * mean_lead if np.isfinite(hit_rate) and np.isfinite(mean_lead) else 0.0
    )
    result["selection_model"] = model_name
    return result, records, evaluation_frame


def _station_alarm_summary(
    frame: pd.DataFrame,
    model_name: str,
    year: int,
    monthly_budget_hours: float,
    step_minutes: int,
) -> pd.DataFrame:
    working = frame.copy()
    working["_alarm"] = pd.to_numeric(
        working["risk_6h__budget_alarm"], errors="coerce"
    ).fillna(0).ge(0.5)
    working["_future"] = working["onset_within_6h"].eq(1)
    working["_hard"] = working["hard_negative_6h"].eq(1)
    rows: list[dict[str, Any]] = []
    for station, group in working.groupby("station_code", sort=True):
        rows.append(
            {
                "model": model_name,
                "year": int(year),
                "monthly_budget_hours": float(monthly_budget_hours),
                "station_code": str(station),
                "station_months": int(group["station_month"].nunique()),
                "false_alarm_bins": int((group["_alarm"] & ~group["_future"]).sum()),
                "alarm_bins": int(group["_alarm"].sum()),
                "hard_negative_bins": int(group["_hard"].sum()),
                "hard_negative_alarm_bins": int((group["_alarm"] & group["_hard"]).sum()),
                "step_minutes": int(step_minutes),
            }
        )
    return pd.DataFrame(rows)


def _season_mask(records: pd.DataFrame, cohort: str, stratum: str) -> pd.Series:
    mask = pd.Series(True, index=records.index)
    if cohort == "complete_seasons":
        mask &= records["complete_icing_season"].eq(1)
    elif cohort == "censored_seasons":
        mask &= records["complete_icing_season"].eq(0)
    elif cohort != "all_seasons":
        raise ValueError(f"Unknown season cohort: {cohort}")
    order = records["within_season_event_order"]
    if stratum == "first":
        mask &= order.eq(1)
    elif stratum == "second":
        mask &= order.eq(2)
    elif stratum == "third_plus":
        mask &= order.ge(3)
    elif stratum == "recurrent":
        mask &= order.ge(2)
    elif stratum != "all":
        raise ValueError(f"Unknown seasonal stratum: {stratum}")
    return mask


def _seasonal_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["year", "monthly_budget_hours", "model", "candidate_quantile"]
    for key, group in records.groupby(keys, sort=True):
        for cohort in SEASON_COHORTS:
            for stratum in SEASONAL_STRATA:
                selected = group.loc[_season_mask(group, cohort, stratum)]
                evaluable = selected.loc[selected["evaluable"].eq(1)]
                hits = evaluable.loc[evaluable["hit"].eq(1)]
                rows.append(
                    {
                        **dict(zip(keys, key)),
                        "season_cohort": cohort,
                        "seasonal_event_stratum": stratum,
                        "target_events": int(selected.shape[0]),
                        "evaluable_events": int(evaluable.shape[0]),
                        "hit_events": int(hits.shape[0]),
                        "event_hit_rate": (
                            float(evaluable["hit"].mean()) if not evaluable.empty else float("nan")
                        ),
                        "mean_effective_lead_hours": (
                            float(hits["effective_lead_hours"].mean())
                            if not hits.empty
                            else float("nan")
                        ),
                        "median_effective_lead_hours": (
                            float(hits["effective_lead_hours"].median())
                            if not hits.empty
                            else float("nan")
                        ),
                        "lead_utility_hours": (
                            float(evaluable["lead_utility_hours"].mean())
                            if not evaluable.empty
                            else float("nan")
                        ),
                        "stations": int(evaluable["station_code"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap_counts(station_count: int, samples: int, seed: int) -> np.ndarray:
    if station_count < 1:
        return np.empty((0, 0), dtype=np.int16)
    rng = np.random.default_rng(seed)
    probabilities = np.full(station_count, 1.0 / station_count)
    return rng.multinomial(station_count, probabilities, size=samples).astype(np.int16)


def paired_event_station_bootstrap(
    records: pd.DataFrame,
    model_a: str,
    model_b: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    left = records.loc[records["model"].eq(model_a)].copy()
    right = records.loc[records["model"].eq(model_b)].copy()
    merged = left.merge(right, on=["event_id", "station_code"], suffixes=("_a", "_b"))
    merged = merged.loc[
        merged["evaluable_a"].eq(1) & merged["evaluable_b"].eq(1)
    ].copy()
    if merged.empty:
        return pd.DataFrame()
    stations = np.sort(merged["station_code"].unique())
    station_lookup = {station: index for index, station in enumerate(stations)}
    merged["_station"] = merged["station_code"].map(station_lookup).astype(int)
    common = merged["hit_a"].eq(1) & merged["hit_b"].eq(1)
    aggregates = {
        "event_count": np.bincount(merged["_station"], minlength=stations.size),
        "hit_difference": np.bincount(
            merged["_station"],
            weights=(merged["hit_a"] - merged["hit_b"]).to_numpy(dtype=float),
            minlength=stations.size,
        ),
        "utility_difference": np.bincount(
            merged["_station"],
            weights=(
                merged["lead_utility_hours_a"] - merged["lead_utility_hours_b"]
            ).to_numpy(dtype=float),
            minlength=stations.size,
        ),
        "common_count": np.bincount(
            merged.loc[common, "_station"], minlength=stations.size
        ),
        "common_lead_difference": np.bincount(
            merged.loc[common, "_station"],
            weights=(
                merged.loc[common, "effective_lead_hours_a"]
                - merged.loc[common, "effective_lead_hours_b"]
            ).to_numpy(dtype=float),
            minlength=stations.size,
        ),
    }
    counts = _bootstrap_counts(stations.size, bootstrap_samples, seed)
    denominator = counts @ aggregates["event_count"]
    common_denominator = counts @ aggregates["common_count"]
    draws = {
        "delta_hit_rate": np.divide(
            counts @ aggregates["hit_difference"], denominator,
            out=np.full(bootstrap_samples, np.nan), where=denominator > 0,
        ),
        "delta_lead_utility_hours": np.divide(
            counts @ aggregates["utility_difference"], denominator,
            out=np.full(bootstrap_samples, np.nan), where=denominator > 0,
        ),
        "delta_common_hit_lead_hours": np.divide(
            counts @ aggregates["common_lead_difference"], common_denominator,
            out=np.full(bootstrap_samples, np.nan), where=common_denominator > 0,
        ),
    }
    point_denominator = float(aggregates["event_count"].sum())
    point_common = float(aggregates["common_count"].sum())
    points = {
        "delta_hit_rate": float(aggregates["hit_difference"].sum() / point_denominator),
        "delta_lead_utility_hours": float(
            aggregates["utility_difference"].sum() / point_denominator
        ),
        "delta_common_hit_lead_hours": (
            float(aggregates["common_lead_difference"].sum() / point_common)
            if point_common > 0
            else float("nan")
        ),
    }
    rows: list[dict[str, Any]] = []
    for metric, values in draws.items():
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "estimate": points[metric],
                "ci95_low": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
                "ci95_high": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
                "evaluable_events": int(point_denominator),
                "stations": int(stations.size),
                "bootstrap_samples": int(finite.size),
            }
        )
    return pd.DataFrame(rows)


def paired_alarm_station_bootstrap(
    station_rows: pd.DataFrame,
    model_a: str,
    model_b: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    left = station_rows.loc[station_rows["model"].eq(model_a)]
    right = station_rows.loc[station_rows["model"].eq(model_b)]
    columns = [
        "station_code", "station_months", "false_alarm_bins",
        "hard_negative_bins", "hard_negative_alarm_bins", "step_minutes",
    ]
    merged = left[columns].merge(right[columns], on="station_code", suffixes=("_a", "_b"))
    if merged.empty:
        return pd.DataFrame()
    counts = _bootstrap_counts(merged.shape[0], bootstrap_samples, seed)

    def ratio(numerator: str, denominator: str, side: str) -> np.ndarray:
        num = counts @ merged[f"{numerator}_{side}"].to_numpy(dtype=float)
        den = counts @ merged[f"{denominator}_{side}"].to_numpy(dtype=float)
        return np.divide(num, den, out=np.full(bootstrap_samples, np.nan), where=den > 0)

    hard_delta = ratio("hard_negative_alarm_bins", "hard_negative_bins", "a") - ratio(
        "hard_negative_alarm_bins", "hard_negative_bins", "b"
    )
    false_a = ratio("false_alarm_bins", "station_months", "a")
    false_b = ratio("false_alarm_bins", "station_months", "b")
    step_hours = float(merged["step_minutes_a"].iloc[0]) / 60.0
    draws = {
        "delta_hard_negative_far": hard_delta,
        "delta_false_alarm_hours_per_station_month": (false_a - false_b) * step_hours,
    }

    def point_ratio(numerator: str, denominator: str, side: str) -> float:
        den = float(merged[f"{denominator}_{side}"].sum())
        return float(merged[f"{numerator}_{side}"].sum() / den) if den > 0 else float("nan")

    points = {
        "delta_hard_negative_far": point_ratio(
            "hard_negative_alarm_bins", "hard_negative_bins", "a"
        )
        - point_ratio("hard_negative_alarm_bins", "hard_negative_bins", "b"),
        "delta_false_alarm_hours_per_station_month": (
            point_ratio("false_alarm_bins", "station_months", "a")
            - point_ratio("false_alarm_bins", "station_months", "b")
        )
        * step_hours,
    }
    rows: list[dict[str, Any]] = []
    for metric, values in draws.items():
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "estimate": points[metric],
                "ci95_low": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
                "ci95_high": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
                "stations": int(merged.shape[0]),
                "bootstrap_samples": int(finite.size),
            }
        )
    return pd.DataFrame(rows)


def _selection_experiment(
    config: dict[str, Any],
    protocol_root: Path,
    cache_root: Path,
    events: pd.DataFrame,
    models: list[str],
    seeds: list[int],
    budgets: list[float],
    quantiles: list[float],
    warning: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_2022 = events.loc[events["onset_time"].dt.year.eq(2022)].copy()
    rows: list[dict[str, Any]] = []
    for model_name in models:
        print(f"Loading 2022 OOF ensemble: {model_name}", flush=True)
        frame = _ensemble_frame(
            cache_root, protocol_root / "oof_predictions", model_name, seeds
        )
        for monthly_budget in budgets:
            for quantile in quantiles:
                budget = _base_budget(config, monthly_budget, quantile)
                controlled = apply_causal_budget_by_station(frame, ["risk_6h"], budget)
                result, _, _ = _evaluate_controlled(
                    controlled,
                    events_2022,
                    model_name,
                    budget,
                    warning,
                    f"2022_oof_utility_q{quantile:.4f}_budget_{monthly_budget:g}h",
                )
                result.update(
                    {
                        "selection_year": 2022,
                        "monthly_budget_hours": float(monthly_budget),
                        "candidate_quantile": float(quantile),
                        "selected": 0,
                    }
                )
                rows.append(result)
                print(
                    f"  {model_name} budget={monthly_budget:g}h q={quantile:.4f} "
                    f"U={result['lead_utility_hours']:.6f} "
                    f"hit={result['event_hit_rate']:.4f} "
                    f"FARh={result['false_alarm_hours_per_station_month']:.4f}",
                    flush=True,
                )
        del frame
    selection = pd.DataFrame(rows)
    selected_rows: list[pd.Series] = []
    for model_name in models:
        for monthly_budget in budgets:
            chosen = select_utility_candidate(selection, model_name, monthly_budget)
            selected_rows.append(chosen)
            mask = (
                selection["selection_model"].eq(model_name)
                & np.isclose(selection["monthly_budget_hours"], monthly_budget)
                & np.isclose(selection["candidate_quantile"], chosen["candidate_quantile"])
            )
            selection.loc[mask, "selected"] = 1
    return selection, pd.DataFrame(selected_rows).reset_index(drop=True)


def _frozen_evaluation(
    config: dict[str, Any],
    protocol_root: Path,
    cache_root: Path,
    events: pd.DataFrame,
    selected: pd.DataFrame,
    models: list[str],
    seeds: list[int],
    warning: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_parts: list[pd.DataFrame] = []
    station_parts: list[pd.DataFrame] = []
    seasonal_columns = [
        "event_id", "within_season_event_order", "seasonal_event_class",
        "icing_season_start_year", "complete_icing_season",
        "left_censored_season", "right_censored_season",
    ]
    seasonal = events[seasonal_columns].drop_duplicates("event_id")
    for model_name in models:
        print(f"Frozen warning evaluation: {model_name}", flush=True)
        history = _ensemble_frame(
            cache_root, protocol_root / "oof_predictions", model_name, seeds
        )
        for year in (2023, 2024):
            current = _ensemble_frame(
                cache_root,
                protocol_root / "locked_predictions" / str(year),
                model_name,
                seeds,
            )
            combined = pd.concat([history, current], ignore_index=True)
            combined = combined.sort_values(["station_code", "issue_time"]).reset_index(drop=True)
            year_events = events.loc[events["onset_time"].dt.year.eq(year)].copy()
            model_selection = selected.loc[selected["selection_model"].eq(model_name)]
            for choice in model_selection.itertuples(index=False):
                budget = _base_budget(
                    config,
                    float(choice.monthly_budget_hours),
                    float(choice.candidate_quantile),
                )
                controlled = apply_causal_budget_by_station(combined, ["risk_6h"], budget)
                controlled_year = controlled.loc[controlled["issue_time"].dt.year.eq(year)].copy()
                result, records, evaluation_frame = _evaluate_controlled(
                    controlled_year,
                    year_events,
                    model_name,
                    budget,
                    warning,
                    f"frozen_2022_utility_budget_{budget.monthly_budget_hours:g}h",
                )
                result.update(
                    {
                        "year": year,
                        "monthly_budget_hours": budget.monthly_budget_hours,
                        "candidate_quantile": budget.candidate_quantile,
                        "selection_source": "2022 pooled block OOF only",
                    }
                )
                metric_rows.append(result)
                records = records.merge(seasonal, on="event_id", how="left", validate="one_to_one")
                records["year"] = year
                records["monthly_budget_hours"] = budget.monthly_budget_hours
                records["candidate_quantile"] = budget.candidate_quantile
                event_parts.append(records)
                station_parts.append(
                    _station_alarm_summary(
                        evaluation_frame,
                        model_name,
                        year,
                        budget.monthly_budget_hours,
                        budget.step_minutes,
                    )
                )
                print(
                    f"  year={year} budget={budget.monthly_budget_hours:g}h "
                    f"hit={result['event_hit_rate']:.4f} "
                    f"lead={result['mean_effective_lead_hours']:.3f} "
                    f"U={result['lead_utility_hours']:.6f} "
                    f"hardFAR={result['hard_negative_far']:.6f}",
                    flush=True,
                )
                del controlled, controlled_year, evaluation_frame
            history = current
            del combined
    return (
        pd.DataFrame(metric_rows),
        pd.concat(event_parts, ignore_index=True),
        pd.concat(station_parts, ignore_index=True),
    )


def _bootstrap_experiments(
    events: pd.DataFrame,
    station_rows: pd.DataFrame,
    comparisons: Iterable[tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    budgets = sorted(events["monthly_budget_hours"].unique())
    for year in (2023, 2024):
        for budget_index, monthly_budget in enumerate(budgets):
            base = events.loc[
                events["year"].eq(year)
                & np.isclose(events["monthly_budget_hours"], monthly_budget)
            ]
            station_base = station_rows.loc[
                station_rows["year"].eq(year)
                & np.isclose(station_rows["monthly_budget_hours"], monthly_budget)
            ]
            for comparison_index, (model_a, model_b) in enumerate(comparisons):
                comparison_seed = seed + year * 1009 + budget_index * 101 + comparison_index * 17
                alarm_result = paired_alarm_station_bootstrap(
                    station_base, model_a, model_b, bootstrap_samples, comparison_seed
                )
                if not alarm_result.empty:
                    alarm_result["year"] = year
                    alarm_result["monthly_budget_hours"] = monthly_budget
                    alarm_result["season_cohort"] = "all_seasons"
                    alarm_result["seasonal_event_stratum"] = "time_bin_metrics"
                    rows.append(alarm_result)
                for cohort_index, cohort in enumerate(SEASON_COHORTS):
                    for stratum_index, stratum in enumerate(SEASONAL_STRATA):
                        subset = base.loc[_season_mask(base, cohort, stratum)]
                        result = paired_event_station_bootstrap(
                            subset,
                            model_a,
                            model_b,
                            bootstrap_samples,
                            comparison_seed + cohort_index * 31 + stratum_index,
                        )
                        if result.empty:
                            continue
                        result["year"] = year
                        result["monthly_budget_hours"] = monthly_budget
                        result["season_cohort"] = cohort
                        result["seasonal_event_stratum"] = stratum
                        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_utility_warning_experiments(
    config_path: str | Path,
    output_root: str | Path,
    models: Iterable[str] = DEFAULT_MODELS,
    budgets: Iterable[float] = (2.0, 5.0, 10.0, 20.0),
    quantiles: Iterable[float] = (0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995),
    bootstrap_samples: int = 5000,
    overwrite: bool = False,
) -> Path:
    config_path = resolve_project_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve_project_path(output_root)
    _prepare_output(output_root, overwrite)
    model_names = [str(value) for value in models]
    budget_values = sorted({float(value) for value in budgets})
    quantile_values = sorted({float(value) for value in quantiles})
    if not model_names or len(model_names) != len(set(model_names)):
        raise ValueError("models must contain unique names")
    if not budget_values or any(value <= 0 for value in budget_values):
        raise ValueError("budgets must be positive")
    if not quantile_values or any(not 0 < value < 1 for value in quantile_values):
        raise ValueError("quantiles must be between zero and one")

    protocol_root = resolve_project_path(config["deep_protocol_output_root"])
    cache_root = resolve_project_path(config["cache_root"])
    seeds = [int(value) for value in config["training"]["seeds"]]
    warning = _warning_settings(config)
    seasonal_path = resolve_project_path(
        "results/recurrence_modeling/seasonal_recurrence/"
        "event_global_seasonal_mapping.csv"
    )
    events = _events(config, seasonal_path)
    available_comparisons = [
        pair for pair in DEFAULT_COMPARISONS if pair[0] in model_names and pair[1] in model_names
    ]

    selection, selected = _selection_experiment(
        config,
        protocol_root,
        cache_root,
        events,
        model_names,
        seeds,
        budget_values,
        quantile_values,
        warning,
    )
    selection.to_csv(output_root / "utility_first_2022_candidate_table.csv", index=False)
    selected.to_csv(output_root / "utility_first_2022_selected_budgets.csv", index=False)

    metrics, event_records, station_rows = _frozen_evaluation(
        config,
        protocol_root,
        cache_root,
        events,
        selected,
        model_names,
        seeds,
        warning,
    )
    metrics.to_csv(output_root / "frozen_2023_2024_warning_metrics.csv", index=False)
    event_records.to_csv(
        output_root / "frozen_event_records.csv.gz", index=False, compression="gzip"
    )
    station_rows.to_csv(output_root / "frozen_station_alarm_counts.csv", index=False)
    seasonal_summary = _seasonal_summary(event_records)
    seasonal_summary.to_csv(output_root / "seasonal_first_recurrent_metrics.csv", index=False)

    bootstrap = _bootstrap_experiments(
        event_records,
        station_rows,
        available_comparisons,
        int(bootstrap_samples),
        int(config.get("recurrence_evaluation", {}).get("bootstrap_seed", 20260807)),
    )
    bootstrap.to_csv(output_root / "paired_station_bootstrap_summary.csv", index=False)

    frozen_configs = []
    for row in selected.itertuples(index=False):
        frozen_configs.append(
            {
                "model": row.selection_model,
                "monthly_budget_hours": float(row.monthly_budget_hours),
                "candidate_quantile": float(row.candidate_quantile),
                "controller": asdict(
                    _base_budget(
                        config,
                        float(row.monthly_budget_hours),
                        float(row.candidate_quantile),
                    )
                ),
            }
        )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "models": model_names,
        "seeds": seeds,
        "selection_year": 2022,
        "frozen_evaluation_years": [2023, 2024],
        "budgets_hours_per_station_month": budget_values,
        "candidate_quantiles": quantile_values,
        "selection_rule": (
            "Among candidates satisfying FAR <= budget, maximize HitRate * mean effective lead; "
            "tie-break by hit rate, lead, lower FAR, then higher quantile."
        ),
        "season_definition": "November through April, indexed by November calendar year",
        "primary_season_cohort": "complete_seasons",
        "censoring_sensitivity": "all_seasons and censored_seasons are reported separately",
        "bootstrap_unit": "paired station cluster retaining all within-station events/time bins",
        "bootstrap_samples": int(bootstrap_samples),
        "comparisons": [f"{left}_minus_{right}" for left, right in available_comparisons],
        "frozen_controllers": frozen_configs,
        "warning_definition": warning,
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Utility-first warning experiments complete: {output_root}", flush=True)
    return output_root

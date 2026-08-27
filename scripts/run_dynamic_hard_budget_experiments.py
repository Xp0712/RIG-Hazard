from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
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
    apply_candidate_policy_by_entity,
    station_month_budget_distribution,
    strict_event_alert_evaluation,
)
from rig_hazard.budget_control import CausalBudgetConfig, apply_causal_budget_by_station
from rig_hazard.config import resolve_project_path
from rig_hazard.dynamic_budget_statistics import paired_station_cluster_bootstrap
from rig_hazard.dynamic_hard_budget import (
    DYNAMIC_BUDGET_CONTRACT_VERSION,
    DynamicBudgetConfig,
    apply_dynamic_hard_budget,
    controller_ablation,
    nesting_audit,
    offline_score_oracle,
)
from rig_hazard.evaluation_provenance import sha256_file
from rig_hazard.naming import FROZEN_MAIN_MODEL_ARTIFACT_ID, MAIN_MODEL_NAME
from rig_hazard.risk_trajectory import (
    TRAJECTORY_CONTRACT_VERSION,
    audit_trajectory_arrays,
    load_trajectory_npz,
    multihorizon_probability_metrics,
    trajectory_value,
)


SPLIT_FILES = {
    2022: "selection_2022_oof.npz",
    2023: "2023_cross_year.npz",
    2024: "2024_final_time.npz",
}
EXISTING_BASELINES = (
    "fixed_threshold",
    "hysteresis",
    "rolling_quantile",
    "original_simple",
    "budget_safe",
)
ONLINE_METHODS = (
    "uadhbac",
    "uniform_pacing_hard_guard",
    "risk_greedy_hard_guard",
    "utility_static_hard_guard",
    "primal_dual_hard_guard",
)
STANDARD_ALGORITHM_METHODS = (
    "dual_mirror_descent_hard_guard",
    "switch_over_knapsack_hard_guard",
)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, low_memory=False)
    events["station_code"] = events["station_code"].astype(str)
    events["event_id"] = events["event_id"].astype(str)
    events["onset_time"] = pd.to_datetime(events["onset_time"], errors="coerce")
    if "valid_target_event" in events:
        events = events.loc[
            pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).eq(1)
        ]
    return events.dropna(subset=["onset_time"]).copy()


def _trajectory_frame(
    payload: dict[str, np.ndarray], cache_root: Path
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    manifest = json.loads((cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
    files = {int(row["file_id"]): row for row in manifest["files"]}
    file_id = np.asarray(payload["file_id"], dtype=np.int64)
    row_index = np.asarray(payload["row_index"], dtype=np.int64)
    station = np.empty(file_id.size, dtype=object)
    hard_negative = np.zeros(file_id.size, dtype=np.int8)
    for value in np.unique(file_id):
        positions = np.flatnonzero(file_id == value)
        metadata = files[int(value)]
        station[positions] = str(metadata["station_code"])
        target_path = cache_root / metadata["target_path"]
        with np.load(target_path, allow_pickle=False) as target:
            if "hard_negative_6h" in target:
                hard_negative[positions] = target["hard_negative_6h"][row_index[positions]]
    frame = pd.DataFrame(
        {
            "station_code": station.astype(str),
            "issue_time": pd.to_datetime(payload["issue_time_ns"], unit="ns"),
            "risk_6h": np.asarray(payload["F_trajectory"])[:, -1],
            "onset_within_6h": payload["y_hazard"].max(axis=1).astype(np.int8),
            "observed_6h": (
                (payload["y_hazard"].max(axis=1) > 0)
                | (payload["censor_mask"][:, -1] > 0)
            ).astype(np.int8),
            "hard_negative_6h": hard_negative,
            "_source_position": np.arange(file_id.size, dtype=np.int64),
        }
    )
    order = frame.sort_values(["station_code", "issue_time", "_source_position"]).index.to_numpy()
    sorted_frame = frame.iloc[order].drop(columns="_source_position").reset_index(drop=True)
    sorted_payload = {
        key: (value[order] if np.asarray(value).shape[0] == order.size else value)
        for key, value in payload.items()
    }
    audit = audit_trajectory_arrays(
        sorted_payload,
        step_minutes=10,
        entity=sorted_frame["station_code"],
        expected_issue_time_ns=sorted_frame["issue_time"].astype("int64").to_numpy(),
    )
    if not audit.passed:
        raise RuntimeError(f"Trajectory identity audit failed: {audit.to_dict()}")
    return sorted_frame, np.asarray(sorted_payload["h_trajectory"]), sorted_payload


def _load_year(
    year: int, config: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], Path]:
    path = resolve_project_path(config["trajectory_root"]) / SPLIT_FILES[year]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing full trajectory {path}. Run scripts/export_selected_full_trajectory.py first."
        )
    payload = load_trajectory_npz(path)
    frame, hazard, sorted_payload = _trajectory_frame(
        payload, resolve_project_path(config["cache_root"])
    )
    return frame, hazard, sorted_payload, path


def _evaluate_alarm(
    controlled: pd.DataFrame,
    events: pd.DataFrame,
    alarm_column: str,
    budget: float,
    step_minutes: int,
    online_monthly: pd.DataFrame | None = None,
    horizon_hours: float = 6.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    strict, event_records, strict_frame = strict_event_alert_evaluation(
        controlled,
        events,
        alarm_column=alarm_column,
        step_minutes=step_minutes,
        horizon_hours=horizon_hours,
    )
    monthly, distribution = station_month_budget_distribution(
        strict_frame,
        alarm_column,
        budget,
        step_minutes,
        strict_false_column="_strict_false_alarm",
        unsettled_column="_strict_unsettled_alarm",
        reserved_column="_strict_reserved_alarm",
    )
    online_summary: dict[str, Any] = {}
    if online_monthly is not None and not online_monthly.empty:
        selected = online_monthly.loc[online_monthly["budget_hours"].eq(float(budget))]
        if not selected.empty:
            online_summary = {
                "maximum_instantaneous_reserved_hours": float(
                    selected["maximum_reserved_hours"].max()
                ),
                "online_hard_budget_violations": int(selected["hard_budget_met"].eq(0).sum()),
                "mean_online_budget_utilization": float(selected["budget_utilization"].mean()),
            }
    metrics = {
        **strict,
        **distribution,
        **online_summary,
        "budget_hours": float(budget),
        "strict_hard_budget_met": int(
            distribution["maximum_reserved_alarm_hours"] <= budget + 1e-9
        ),
    }
    if "hard_negative_6h" in strict_frame:
        alarm = pd.to_numeric(strict_frame[alarm_column], errors="coerce").fillna(0).ge(0.5)
        hard = pd.to_numeric(strict_frame["hard_negative_6h"], errors="coerce").fillna(0).ge(0.5)
        metrics.update(
            {
                "hard_negative_bins": int(hard.sum()),
                "hard_negative_alarm_bins": int((alarm & hard).sum()),
                "hard_negative_alarm_rate": float((alarm & hard).sum() / max(int(hard.sum()), 1)),
            }
        )
    return metrics, event_records, monthly


def _controller_config(
    method: str,
    selected: dict[str, Any],
    experiment: dict[str, Any],
) -> DynamicBudgetConfig:
    base = DynamicBudgetConfig(
        step_minutes=int(experiment["step_minutes"]),
        horizon_steps=int(experiment["horizon_steps"]),
        method="uadhbac",
        utility=str(experiment["controller"]["utility"]),
        score_threshold=float(selected["utility_score_threshold"]),
        price_initial=float(experiment["controller"].get("price_initial", 0.0)),
        price_learning_rate=float(selected["price_learning_rate"]),
        pending_pressure=float(selected["pending_pressure"]),
        pacing_slack_bins=int(experiment["controller"].get("pacing_slack_bins", 1)),
        deduplication_bins=int(experiment["controller"].get("deduplication_bins", 2)),
        couple_budgets=bool(experiment["controller"].get("couple_budgets", True)),
    )
    if method == "uadhbac":
        return base
    if method == "uniform_pacing_hard_guard":
        return DynamicBudgetConfig(
            **{
                **asdict(base),
                "method": "uniform_pacing",
                "score_threshold": float(selected["risk_score_threshold"]),
                "use_lead_utility": False,
                "use_dynamic_price": False,
                "couple_budgets": False,
            }
        )
    if method == "risk_greedy_hard_guard":
        return DynamicBudgetConfig(
            **{
                **asdict(base),
                "method": "risk_greedy",
                "score_threshold": float(selected["risk_score_threshold"]),
                "use_lead_utility": False,
                "use_dynamic_price": False,
                "couple_budgets": False,
            }
        )
    if method == "utility_static_hard_guard":
        return DynamicBudgetConfig(
            **{
                **asdict(base),
                "method": "utility_static",
                "use_dynamic_price": False,
                "couple_budgets": False,
            }
        )
    if method == "primal_dual_hard_guard":
        return DynamicBudgetConfig(
            **{
                **asdict(base),
                "method": "primal_dual",
                "deduplicate_events": False,
                "couple_budgets": False,
            }
        )
    if method == "dual_mirror_descent_hard_guard":
        return DynamicBudgetConfig(
            **{
                **asdict(base),
                "method": "dual_mirror_descent",
                "use_dynamic_price": False,
                "deduplicate_events": False,
                "couple_budgets": False,
                "dual_step_scale": float(selected["dual_step_scale"]),
                "dual_initial_price": 0.0,
            }
        )
    if method == "switch_over_knapsack_hard_guard":
        return DynamicBudgetConfig(
            **{
                **asdict(base),
                "method": "switch_over_knapsack",
                "use_dynamic_price": False,
                "deduplicate_events": False,
                "couple_budgets": False,
                "switch_high_threshold": float(selected["switch_high_threshold"]),
                "switch_low_threshold": float(selected["switch_low_threshold"]),
                "switch_fraction": float(selected["switch_fraction"]),
            }
        )
    if method.startswith("dynamic_without_") or method == "dynamic_with_F6_only":
        return controller_ablation(method, base)
    raise ValueError(f"Unsupported controller method: {method}")


def _safe_budget_policy(parameters: dict[str, Any], step: int, budget: float) -> SafeBudgetPolicy:
    return SafeBudgetPolicy(
        step_minutes=step,
        budget_hours=budget,
        burst_allowance_hours=float(parameters.get("burst_allowance_hours", 0.0)),
        pace_multiplier=float(parameters.get("pace_multiplier", 1.0)),
    )


def _apply_existing_baseline(
    frame: pd.DataFrame,
    strategy: str,
    parameters: dict[str, Any],
    budget: float,
    step: int,
    candidate_only: bool = False,
) -> pd.DataFrame:
    if strategy == "original_simple":
        controlled = apply_causal_budget_by_station(
            frame,
            ["risk_6h"],
            CausalBudgetConfig(
                step_minutes=step,
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
                "risk_6h__budget_alarm": "budget_alarm",
            }
        )
    allowed = CandidatePolicy.__dataclass_fields__
    policy = CandidatePolicy(**{key: value for key, value in parameters.items() if key in allowed})
    safe = None if candidate_only or strategy != "budget_safe" else _safe_budget_policy(parameters, step, budget)
    return apply_candidate_policy_by_entity(frame, "risk_6h", policy, safe_budget=safe)


def _load_baseline_policies(config: dict[str, Any], budgets: list[float]) -> dict[tuple[float, str], dict]:
    path = resolve_project_path(config["baseline_selection_file"])
    selected = pd.read_csv(path)
    selected = selected.loc[
        selected["model"].eq(FROZEN_MAIN_MODEL_ARTIFACT_ID)
        & selected["selection_mode"].eq("max_utility_subject_to_budget")
    ]
    result = {}
    for budget in budgets:
        for strategy in EXISTING_BASELINES:
            row = selected.loc[
                selected["budget_hours"].eq(budget) & selected["strategy"].eq(strategy)
            ]
            if row.shape[0] != 1:
                raise ValueError(f"Expected one frozen 2022 policy for {strategy}, B={budget}")
            result[(budget, strategy)] = json.loads(row.iloc[0]["parameters"])
    return result


def _event_subgroup_summary(
    event_records: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    metadata_columns = [
        "event_id",
        "station_code",
        "onset_time",
        "within_season_event_order",
        "complete_icing_season",
        "left_censored_season",
        "right_censored_season",
    ]
    metadata = events[[column for column in metadata_columns if column in events]].copy()
    metadata["onset_time"] = pd.to_datetime(metadata["onset_time"], errors="coerce")
    metadata = metadata.sort_values(["station_code", "onset_time"])
    metadata["recurrence_interval_hours"] = (
        metadata.groupby("station_code")["onset_time"].diff().dt.total_seconds() / 3600.0
    )
    station_counts = metadata.groupby("station_code")["event_id"].size()
    station_levels = pd.qcut(
        station_counts.rank(method="first"),
        3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    ).astype(str)
    metadata["station_risk_level"] = metadata["station_code"].map(station_levels)
    merged = event_records.merge(
        metadata,
        on=["event_id", "station_code"],
        how="left",
        suffixes=("", "_metadata"),
    )
    order = pd.to_numeric(
        merged.get("within_season_event_order_metadata", merged.get("within_season_event_order")),
        errors="coerce",
    )
    merged["event_order_stratum"] = np.select(
        [order.eq(1), order.eq(2), order.ge(3)], ["first", "second", "third_plus"], default="unknown"
    )
    interval = pd.to_numeric(merged["recurrence_interval_hours"], errors="coerce")
    merged["recurrence_interval_stratum"] = pd.cut(
        interval,
        [-np.inf, 6, 24, 72, np.inf],
        labels=["lt_6h", "6_to_24h", "24_to_72h", "gt_72h"],
        right=False,
    ).astype(str)
    onset = pd.to_datetime(merged.get("onset_time_metadata", merged.get("onset_time")), errors="coerce")
    merged["grid_alignment_stratum"] = np.where(
        onset.dt.minute.mod(10).eq(0) & onset.dt.second.eq(0), "aligned_10min", "unaligned"
    )
    complete = pd.to_numeric(merged.get("complete_icing_season", 0), errors="coerce").fillna(0)
    merged["season_completeness_stratum"] = np.where(complete.eq(1), "complete", "censored")
    rows = []
    groupings = {
        "event_order": "event_order_stratum",
        "recurrence_interval": "recurrence_interval_stratum",
        "grid_alignment": "grid_alignment_stratum",
        "season_completeness": "season_completeness_stratum",
        "station_risk": "station_risk_level",
    }
    keys = ["year", "budget_hours", "method"]
    for group_type, column in groupings.items():
        for values, group in merged.groupby([*keys, column], dropna=False, sort=True):
            evaluable = group.loc[group["operational_evaluable"].eq(1)]
            hits = evaluable.loc[evaluable["hit"].eq(1)]
            hit_rate = float(evaluable["hit"].mean()) if not evaluable.empty else float("nan")
            lead = (
                float(pd.to_numeric(hits["effective_lead_hours"], errors="coerce").mean())
                if not hits.empty else float("nan")
            )
            rows.append(
                {
                    **dict(zip(keys, values[:3])),
                    "subgroup_type": group_type,
                    "subgroup": str(values[3]),
                    "operational_events": int(evaluable.shape[0]),
                    "hit_events": int(hits.shape[0]),
                    "event_hit_rate": hit_rate,
                    "mean_effective_lead_hours": lead,
                    "lead_utility_hours": hit_rate * lead if np.isfinite(lead) else 0.0,
                    "mean_available_warning_window_hours": float(
                        pd.to_numeric(evaluable["available_warning_window_hours"], errors="coerce").mean()
                    ) if not evaluable.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def select_controller(
    config: dict[str, Any], output: Path, smoke: bool = False
) -> Path:
    frame, hazard, _, trajectory_path = _load_year(2022, config)
    events = _load_events(resolve_project_path(config["event_table"]))
    events = events.loc[events["onset_time"].dt.year.eq(2022)]
    budgets = [float(value) for value in config["budgets_hours"]]
    utility_scores = trajectory_value(hazard, int(config["step_minutes"]), "linear")
    risk_scores = 1.0 - np.prod(1.0 - hazard, axis=1)
    grid = config["selection_grid"]
    candidates = [
        (float(quantile), float(rate), float(pending))
        for quantile in grid["utility_score_quantiles"]
        for rate in grid["price_learning_rates"]
        for pending in grid["pending_pressures"]
    ]
    if smoke:
        candidates = candidates[:1]
        budgets = budgets[:2]
    rows = []
    candidate_root = output / "selection_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    for index, (quantile, rate, pending) in enumerate(candidates, start=1):
        parameters = {
            "utility_score_quantile": quantile,
            "utility_score_threshold": float(np.quantile(utility_scores, quantile)),
            "risk_score_threshold": float(np.quantile(risk_scores, quantile)),
            "price_learning_rate": rate,
            "pending_pressure": pending,
        }
        cache = candidate_root / f"candidate_{index:03d}.csv"
        if cache.exists():
            rows.extend(pd.read_csv(cache).to_dict(orient="records"))
            continue
        controller = _controller_config("uadhbac", parameters, config)
        controlled, online_monthly, _ = apply_dynamic_hard_budget(
            frame, hazard, budgets, controller, events=events
        )
        nested = nesting_audit(controlled, budgets)
        candidate_rows = []
        for budget in budgets:
            metrics, _, _ = _evaluate_alarm(
                controlled, events, f"budget_alarm_{budget:g}h", budget,
                int(config["step_minutes"]), online_monthly,
            )
            candidate_rows.append(
                {
                    "candidate_id": index,
                    **parameters,
                    "budget_hours": budget,
                    "nesting_violations": int(nested["nesting_violations"].sum()),
                    **metrics,
                }
            )
        pd.DataFrame(candidate_rows).to_csv(cache, index=False)
        rows.extend(candidate_rows)
        print(f"2022 controller selection: {index}/{len(candidates)}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "selection_search.csv", index=False)
    primary = set(float(value) for value in config["primary_budgets_hours"])
    summary = (
        results.assign(
            feasible=lambda x: x["strict_hard_budget_met"].eq(1)
            & x["online_hard_budget_violations"].eq(0)
            & x["nesting_violations"].eq(0),
            primary_utility=lambda x: np.where(
                x["budget_hours"].isin(primary), x["operational_lead_utility_hours"], np.nan
            ),
        )
        .groupby("candidate_id", as_index=False)
        .agg(feasible=("feasible", "all"), primary_utility=("primary_utility", "mean"))
    )
    eligible = summary.loc[summary["feasible"]]
    if eligible.empty:
        raise RuntimeError("No 2022 controller candidate satisfies all hard constraints")
    winner_id = int(eligible.sort_values("primary_utility", ascending=False).iloc[0]["candidate_id"])
    winner = results.loc[results["candidate_id"].eq(winner_id)].iloc[0]
    selected = {
        "method": "uadhbac",
        "selection_year": 2022,
        "selection_source": f"{MAIN_MODEL_NAME} pooled OOF only",
        "utility_score_quantile": float(winner["utility_score_quantile"]),
        "utility_score_threshold": float(winner["utility_score_threshold"]),
        "risk_score_threshold": float(winner["risk_score_threshold"]),
        "price_learning_rate": float(winner["price_learning_rate"]),
        "pending_pressure": float(winner["pending_pressure"]),
        "trajectory_sha256": sha256_file(trajectory_path),
        "test_years_used_for_selection": [],
    }
    selected["parameter_sha256"] = _canonical_hash(selected)
    path = output / "selected_controller_2022.json"
    path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_frozen_years(
    config: dict[str, Any], output: Path, include_ablations: bool = True
) -> None:
    selected_path = output / "selected_controller_2022.json"
    if not selected_path.exists():
        raise FileNotFoundError("Run the 2022 selection phase before frozen evaluation")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_hash = sha256_file(selected_path)
    budgets = [float(value) for value in config["budgets_hours"]]
    policies = _load_baseline_policies(config, budgets)
    methods = list(ONLINE_METHODS)
    if include_ablations:
        methods.extend(str(value) for value in config["ablations"])
    metric_rows, event_parts, month_parts, nesting_parts = [], [], [], []
    trajectory_hashes = {}
    start_time = time.perf_counter()
    for year in [int(value) for value in config["locked_years"]]:
        frame, hazard, payload, trajectory_path = _load_year(year, config)
        trajectory_hashes[str(year)] = sha256_file(trajectory_path)
        events = _load_events(resolve_project_path(config["event_table"]))
        year_events = events.loc[events["onset_time"].dt.year.eq(year)]
        multihorizon_probability_metrics(
            payload, str(year), int(config["step_minutes"])
        ).to_csv(output / f"probability_metrics_{year}.csv", index=False)

        for method in methods:
            controller = _controller_config(method, selected, config)
            method_start = time.perf_counter()
            controlled, online_monthly, _ = apply_dynamic_hard_budget(
                frame, hazard, budgets, controller, events=year_events
            )
            method_elapsed = time.perf_counter() - method_start
            nested = nesting_audit(controlled, budgets)
            nested["year"] = year
            nested["method"] = method
            nesting_parts.append(nested)
            for budget in budgets:
                metrics, records, monthly = _evaluate_alarm(
                    controlled, year_events, f"budget_alarm_{budget:g}h", budget,
                    int(config["step_minutes"]), online_monthly,
                )
                metrics.update(
                    {
                        "year": year,
                        "method": method,
                        "parameter_file_sha256": selected_hash,
                        "nesting_violations": int(nested["nesting_violations"].sum()),
                        "decision_elapsed_seconds": method_elapsed,
                        "decision_rows_per_second": frame.shape[0] / max(method_elapsed, 1e-9),
                        "deployable": 1,
                    }
                )
                metric_rows.append(metrics)
                records["year"], records["method"], records["budget_hours"] = year, method, budget
                event_parts.append(records)
                monthly["year"], monthly["method"] = year, method
                month_parts.append(monthly)
            print(f"Frozen evaluation: {year} {method}", flush=True)

        oracle_score = trajectory_value(
            hazard, int(config["step_minutes"]), str(config["controller"]["utility"])
        )
        oracle = offline_score_oracle(
            frame, oracle_score, budgets, int(config["step_minutes"])
        )
        oracle_nesting = []
        for lower, upper in zip(budgets[:-1], budgets[1:]):
            conflict = (
                oracle[f"offline_score_oracle_{lower:g}h"].eq(1)
                & oracle[f"offline_score_oracle_{upper:g}h"].eq(0)
            )
            oracle_nesting.append(int(conflict.sum()))
        for budget in budgets:
            metrics, records, monthly = _evaluate_alarm(
                oracle, year_events, f"offline_score_oracle_{budget:g}h", budget,
                int(config["step_minutes"]),
            )
            metrics.update(
                {
                    "year": year,
                    "method": "offline_score_oracle_diagnostic",
                    "parameter_file_sha256": selected_hash,
                    "nesting_violations": int(sum(oracle_nesting)),
                    "deployable": 0,
                    "uses_event_labels_for_ranking": 0,
                }
            )
            metric_rows.append(metrics)
            records["year"] = year
            records["method"] = "offline_score_oracle_diagnostic"
            records["budget_hours"] = budget
            event_parts.append(records)
            monthly["year"] = year
            monthly["method"] = "offline_score_oracle_diagnostic"
            month_parts.append(monthly)

        # Existing five baselines and their common pending-reserve hard guards.
        for budget in budgets:
            for strategy in EXISTING_BASELINES:
                parameters = policies[(budget, strategy)]
                controlled = _apply_existing_baseline(
                    frame, strategy, parameters, budget, int(config["step_minutes"])
                )
                metrics, records, monthly = _evaluate_alarm(
                    controlled, year_events, "budget_alarm", budget, int(config["step_minutes"])
                )
                metrics.update(
                    {
                        "year": year,
                        "method": strategy,
                        "parameter_file_sha256": selected_hash,
                        "nesting_violations": np.nan,
                        "deployable": 1,
                    }
                )
                metric_rows.append(metrics)
                records["year"], records["method"], records["budget_hours"] = year, strategy, budget
                event_parts.append(records)
                monthly["year"], monthly["method"] = year, strategy
                month_parts.append(monthly)
                if strategy in config["hard_guard_baselines"]:
                    candidate = _apply_existing_baseline(
                        frame, strategy, parameters, budget, int(config["step_minutes"]),
                        candidate_only=True,
                    )
                    guard_config = DynamicBudgetConfig(
                        step_minutes=int(config["step_minutes"]),
                        horizon_steps=int(config["horizon_steps"]),
                        method=f"{strategy}_hard_guard",
                        score_threshold=0.0,
                        use_lead_utility=False,
                        use_dynamic_price=False,
                        couple_budgets=False,
                    )
                    guarded, online_monthly, _ = apply_dynamic_hard_budget(
                        frame, hazard, [budget], guard_config, events=year_events,
                        candidate_masks={budget: candidate["candidate_alarm"].to_numpy(dtype=bool)},
                    )
                    guard_name = f"{strategy}_hard_guard"
                    metrics, records, monthly = _evaluate_alarm(
                        guarded, year_events, f"budget_alarm_{budget:g}h", budget,
                        int(config["step_minutes"]), online_monthly,
                    )
                    metrics.update(
                        {
                            "year": year,
                            "method": guard_name,
                            "parameter_file_sha256": selected_hash,
                            "nesting_violations": np.nan,
                            "deployable": 1,
                        }
                    )
                    metric_rows.append(metrics)
                    records["year"], records["method"], records["budget_hours"] = year, guard_name, budget
                    event_parts.append(records)
                    monthly["year"], monthly["method"] = year, guard_name
                    month_parts.append(monthly)

    metrics = pd.DataFrame(metric_rows)
    event_records = pd.concat(event_parts, ignore_index=True)
    months = pd.concat(month_parts, ignore_index=True)
    nesting = pd.concat(nesting_parts, ignore_index=True)
    metrics.to_csv(output / "frozen_main_results.csv", index=False)
    event_records.to_csv(output / "frozen_event_records.csv.gz", index=False, compression="gzip")
    months.to_csv(output / "frozen_station_month_audit.csv.gz", index=False, compression="gzip")
    nesting.to_csv(output / "frozen_nesting_audit.csv", index=False)
    all_events = _load_events(resolve_project_path(config["event_table"]))
    _event_subgroup_summary(event_records, all_events).to_csv(
        output / "frozen_event_subgroups.csv", index=False
    )
    queue_rows = []
    for year in [int(value) for value in config["locked_years"]]:
        representative = metrics.loc[
            metrics["year"].eq(year) & metrics["method"].eq("uadhbac")
        ].iloc[0]
        expected = config.get("expected_queue_sizes", {}).get(str(year), {})
        queue_rows.append(
            {
                "year": year,
                "operational_events": int(representative["operational_evaluable_events"]),
                "expected_operational_events": expected.get("operational"),
                "complete_window_events": int(representative["evaluable_events"]),
                "expected_complete_window_events": expected.get("complete"),
                "operational_count_matches_contract": int(
                    not expected or int(representative["operational_evaluable_events"])
                    == int(expected["operational"])
                ),
                "complete_count_matches_contract": int(
                    not expected or int(representative["evaluable_events"])
                    == int(expected["complete"])
                ),
            }
        )
    pd.DataFrame(queue_rows).to_csv(output / "queue_contract_audit.csv", index=False)

    bootstrap_parts = []
    bootstrap_config = config["bootstrap"]
    for year in [int(value) for value in config["locked_years"]]:
        for budget in [float(value) for value in config["primary_budgets_hours"]]:
            selected_records = event_records.loc[
                event_records["year"].eq(year) & event_records["budget_hours"].eq(budget)
            ]
            for comparator in bootstrap_config["comparators"]:
                if comparator not in set(selected_records["method"]):
                    continue
                part = paired_station_cluster_bootstrap(
                    selected_records,
                    "uadhbac",
                    comparator,
                    int(bootstrap_config["samples"]),
                    int(bootstrap_config["seed"]) + year + int(budget * 10),
                    queue="operational",
                )
                part["year"], part["budget_hours"] = year, budget
                bootstrap_parts.append(part)
    if bootstrap_parts:
        pd.concat(bootstrap_parts, ignore_index=True).to_csv(
            output / "paired_station_cluster_bootstrap.csv", index=False
        )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - start_time,
        "model": MAIN_MODEL_NAME,
        "selection_year": 2022,
        "locked_years": config["locked_years"],
        "selected_parameter_file_sha256": selected_hash,
        "event_table_sha256": sha256_file(resolve_project_path(config["event_table"])),
        "baseline_2022_policy_file_sha256": sha256_file(
            resolve_project_path(config["baseline_selection_file"])
        ),
        "trajectory_sha256": trajectory_hashes,
        "trajectory_contract_version": TRAJECTORY_CONTRACT_VERSION,
        "controller_contract_version": DYNAMIC_BUDGET_CONTRACT_VERSION,
        "test_year_model_or_parameter_selection": False,
    }
    (output / "frozen_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / ".complete_frozen_dynamic_hard_budget").touch()


def audit_trajectories(config: dict[str, Any], output: Path) -> None:
    rows, metrics = [], []
    for year, filename in SPLIT_FILES.items():
        path = resolve_project_path(config["trajectory_root"]) / filename
        payload = load_trajectory_npz(path)
        audit = audit_trajectory_arrays(payload, int(config["step_minutes"]))
        rows.append({"year": year, "file": str(path), "sha256": sha256_file(path), **audit.to_dict()})
        metrics.append(multihorizon_probability_metrics(payload, str(year), int(config["step_minutes"])))
    pd.DataFrame(rows).to_csv(output / "trajectory_integrity_audit.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(output / "multihorizon_probability_metrics.csv", index=False)


def _baseline_taxonomy() -> pd.DataFrame:
    """Describe what the original six hard-budget baselines actually cover."""

    dmd_reference = "https://proceedings.mlr.press/v119/balseiro20a.html"
    knapsack_reference = "https://doi.org/10.1287/opre.1080.0555"
    pacing_reference = "https://doi.org/10.1287/opre.2020.2073"
    rows = [
        ("budget_safe", 1, "custom budget pacing", "partial", "Project-specific quantile policy with a hard reserve guard", pacing_reference),
        ("original_simple", 1, "rolling-threshold budget control", "partial", "Project-specific rolling-quantile controller", pacing_reference),
        ("uniform_pacing_hard_guard", 1, "deterministic budget pacing", "covered", "Linear target-spend path plus the common hard reserve guard", pacing_reference),
        ("risk_greedy_hard_guard", 1, "online knapsack", "partial", "Myopic greedy acceptance by F6 risk; no switch-over value-to-go approximation", knapsack_reference),
        ("utility_static_hard_guard", 1, "online knapsack", "partial", "Myopic greedy acceptance by lead utility; no switch-over value-to-go approximation", knapsack_reference),
        ("primal_dual_hard_guard", 1, "online primal-dual", "partial", "Project-specific exponential occupancy price, not canonical dual mirror descent", dmd_reference),
        ("dual_mirror_descent_hard_guard", 0, "online primal-dual", "standard_added", "Euclidean dual mirror descent with adaptive target calendar-bin pacing", dmd_reference),
        ("switch_over_knapsack_hard_guard", 0, "online stochastic knapsack", "standard_added", "Two-stage switch-over threshold policy frozen from 2022 OOF", knapsack_reference),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "counted_in_original_six",
            "algorithm_family",
            "coverage_status",
            "implementation_scope",
            "primary_reference",
        ],
    )


def select_standard_algorithm_baselines(
    config: dict[str, Any], output: Path, smoke: bool = False
) -> Path:
    """Select canonical baseline parameters using 2022 pooled OOF only."""

    source = resolve_project_path(config.get("theory_source_root", config["output_root"]))
    base_selected_path = source / "selected_controller_2022.json"
    if not base_selected_path.exists():
        raise FileNotFoundError(f"Missing frozen 2022 controller parameters: {base_selected_path}")
    base_selected = json.loads(base_selected_path.read_text(encoding="utf-8"))
    frame, hazard, _, trajectory_path = _load_year(2022, config)
    events = _load_events(resolve_project_path(config["event_table"]))
    events = events.loc[events["onset_time"].dt.year.eq(2022)]
    budgets = [float(value) for value in config["budgets_hours"]]
    primary = set(float(value) for value in config["primary_budgets_hours"])
    utility_scores = trajectory_value(
        hazard,
        int(config["step_minutes"]),
        str(config["controller"]["utility"]),
    )
    standard = config["standard_algorithm_baselines"]
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for index, scale in enumerate(standard["dual_mirror_descent"]["eta_scales"], start=1):
        candidates.append(
            (
                "dual_mirror_descent_hard_guard",
                f"dmd_{index:02d}",
                {"dual_step_scale": float(scale)},
            )
        )
    switch_index = 0
    for high_quantile in standard["switch_over_knapsack"]["high_quantiles"]:
        for low_quantile in standard["switch_over_knapsack"]["low_quantiles"]:
            if float(low_quantile) > float(high_quantile):
                continue
            for switch_fraction in standard["switch_over_knapsack"]["switch_fractions"]:
                switch_index += 1
                candidates.append(
                    (
                        "switch_over_knapsack_hard_guard",
                        f"switch_{switch_index:02d}",
                        {
                            "switch_high_quantile": float(high_quantile),
                            "switch_low_quantile": float(low_quantile),
                            "switch_high_threshold": float(np.quantile(utility_scores, high_quantile)),
                            "switch_low_threshold": float(np.quantile(utility_scores, low_quantile)),
                            "switch_fraction": float(switch_fraction),
                        },
                    )
                )
    if smoke:
        candidates = [
            next(row for row in candidates if row[0] == method)
            for method in STANDARD_ALGORITHM_METHODS
        ]
        budgets = budgets[:2]
        primary = set(budgets)

    candidate_root = output / "standard_baseline_selection_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for method, candidate_id, parameters in candidates:
        cache = candidate_root / f"{candidate_id}.csv"
        if cache.exists():
            rows.extend(pd.read_csv(cache).to_dict(orient="records"))
            continue
        controller = _controller_config(method, {**base_selected, **parameters}, config)
        controlled, online_monthly, _ = apply_dynamic_hard_budget(
            frame, hazard, budgets, controller, events=events
        )
        nested = nesting_audit(controlled, budgets)
        candidate_rows = []
        for budget in budgets:
            metrics, _, _ = _evaluate_alarm(
                controlled,
                events,
                f"budget_alarm_{budget:g}h",
                budget,
                int(config["step_minutes"]),
                online_monthly,
            )
            candidate_rows.append(
                {
                    "method": method,
                    "candidate_id": candidate_id,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "budget_hours": budget,
                    "nesting_violations": int(nested["nesting_violations"].sum()),
                    **metrics,
                }
            )
        pd.DataFrame(candidate_rows).to_csv(cache, index=False)
        rows.extend(candidate_rows)
        print(f"2022 standard baseline selection: {method} {candidate_id}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(output / "standard_baseline_selection_2022.csv", index=False)
    selected_methods: dict[str, dict[str, Any]] = {}
    for method in STANDARD_ALGORITHM_METHODS:
        method_rows = results.loc[results["method"].eq(method)].copy()
        summary = (
            method_rows.assign(
                feasible=lambda x: x["strict_hard_budget_met"].eq(1)
                & x["online_hard_budget_violations"].fillna(0).eq(0),
                primary_utility=lambda x: np.where(
                    x["budget_hours"].isin(primary),
                    x["operational_lead_utility_hours"],
                    np.nan,
                ),
            )
            .groupby(["candidate_id", "parameters"], as_index=False)
            .agg(feasible=("feasible", "all"), primary_utility=("primary_utility", "mean"))
        )
        eligible = summary.loc[summary["feasible"]]
        if eligible.empty:
            raise RuntimeError(f"No hard-feasible 2022 candidate for {method}")
        winner = eligible.sort_values(
            ["primary_utility", "candidate_id"], ascending=[False, True]
        ).iloc[0]
        selected_methods[method] = {
            "candidate_id": str(winner["candidate_id"]),
            **json.loads(winner["parameters"]),
            "selection_primary_utility": float(winner["primary_utility"]),
        }
    selected = {
        "selection_year": 2022,
        "selection_source": f"{MAIN_MODEL_NAME} pooled OOF only",
        "test_years_used_for_selection": [],
        "methods": selected_methods,
        "trajectory_sha256": sha256_file(trajectory_path),
        "base_controller_parameter_sha256": sha256_file(base_selected_path),
    }
    selected["parameter_sha256"] = _canonical_hash(selected)
    path = output / "selected_standard_baselines_2022.json"
    path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _nesting_cost_outputs(
    config: dict[str, Any], output: Path, smoke: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = resolve_project_path(config.get("theory_source_root", config["output_root"]))
    metrics = pd.read_csv(source / "frozen_main_results.csv")
    records = pd.read_csv(source / "frozen_event_records.csv.gz", low_memory=False)
    audit = pd.read_csv(source / "frozen_nesting_audit.csv")
    years = [int(value) for value in config["locked_years"]]
    budgets = [float(value) for value in config["budgets_hours"]]
    if smoke:
        years = years[:1]
        budgets = budgets[:2]
    coupled = "uadhbac"
    uncoupled = "dynamic_without_budget_coupling"
    rows = []
    for year in years:
        violations = int(
            audit.loc[
                audit["year"].eq(year) & audit["method"].eq(uncoupled),
                "nesting_violations",
            ].sum()
        )
        for budget in budgets:
            pair = metrics.loc[
                metrics["year"].eq(year)
                & metrics["budget_hours"].eq(budget)
                & metrics["method"].isin([coupled, uncoupled])
            ].set_index("method")
            if pair.shape[0] != 2:
                raise ValueError(f"Missing coupled/uncoupled pair for {year}, B={budget}")
            nested_utility = float(pair.loc[coupled, "operational_lead_utility_hours"])
            free_utility = float(pair.loc[uncoupled, "operational_lead_utility_hours"])
            utility_cost = free_utility - nested_utility
            rows.append(
                {
                    "year": year,
                    "budget_hours": budget,
                    "nested_method": coupled,
                    "non_nested_method": uncoupled,
                    "nested_operational_utility_hours": nested_utility,
                    "non_nested_operational_utility_hours": free_utility,
                    "nesting_utility_cost_hours": utility_cost,
                    "relative_nesting_utility_cost": utility_cost / max(abs(free_utility), 1e-12),
                    "nesting_hit_rate_cost": float(
                        pair.loc[uncoupled, "operational_event_hit_rate"]
                        - pair.loc[coupled, "operational_event_hit_rate"]
                    ),
                    "nested_reserved_budget_utilization": float(
                        pair.loc[coupled, "mean_reserved_budget_utilization"]
                    ),
                    "non_nested_reserved_budget_utilization": float(
                        pair.loc[uncoupled, "mean_reserved_budget_utilization"]
                    ),
                    "non_nested_adjacent_pair_violations": violations,
                    "nested_adjacent_pair_violations": 0,
                    "positive_cost_means_nesting_reduced_utility": 1,
                }
            )
    cost = pd.DataFrame(rows)
    cost.to_csv(output / "nesting_utility_cost.csv", index=False)

    bootstrap_parts = []
    samples = int(config["bootstrap"]["samples"])
    seed = int(config["bootstrap"]["seed"])
    for year in years:
        for budget in budgets:
            selected_records = records.loc[
                records["year"].eq(year)
                & records["budget_hours"].eq(budget)
                & records["method"].isin([coupled, uncoupled])
            ]
            for queue in ("complete", "operational"):
                part = paired_station_cluster_bootstrap(
                    selected_records,
                    uncoupled,
                    coupled,
                    samples,
                    seed + year + int(budget * 10) + (0 if queue == "complete" else 10000),
                    queue=queue,
                )
                part["year"], part["budget_hours"] = year, budget
                part["positive_difference_means_nesting_cost"] = 1
                bootstrap_parts.append(part)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    bootstrap.to_csv(output / "nesting_cost_station_cluster_bootstrap.csv", index=False)
    return cost, bootstrap


def run_theory_supplement(
    config: dict[str, Any], output: Path, smoke: bool = False
) -> None:
    """Run canonical online baselines, nesting-price inference, and complexity audit."""

    source = resolve_project_path(config.get("theory_source_root", config["output_root"]))
    selected_path = select_standard_algorithm_baselines(config, output, smoke=smoke)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    base_selected_path = source / "selected_controller_2022.json"
    base_selected = json.loads(base_selected_path.read_text(encoding="utf-8"))
    budgets = [float(value) for value in config["budgets_hours"]]
    years = [int(value) for value in config["locked_years"]]
    if smoke:
        budgets = budgets[:2]
        years = years[:1]

    metric_rows: list[dict[str, Any]] = []
    event_parts: list[pd.DataFrame] = []
    month_parts: list[pd.DataFrame] = []
    nesting_parts: list[pd.DataFrame] = []
    complexity_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    taxonomy = _baseline_taxonomy()
    taxonomy.to_csv(output / "hard_guard_algorithm_taxonomy.csv", index=False)

    for year in years:
        frame, hazard, _, trajectory_path = _load_year(year, config)
        all_events = _load_events(resolve_project_path(config["event_table"]))
        year_events = all_events.loc[all_events["onset_time"].dt.year.eq(year)]
        for method in STANDARD_ALGORITHM_METHODS:
            parameters = selected["methods"][method]
            controller = _controller_config(method, {**base_selected, **parameters}, config)
            method_start = time.perf_counter()
            controlled, online_monthly, _ = apply_dynamic_hard_budget(
                frame, hazard, budgets, controller, events=year_events
            )
            elapsed = time.perf_counter() - method_start
            nested = nesting_audit(controlled, budgets)
            nested["year"], nested["method"] = year, method
            nesting_parts.append(nested)
            family = taxonomy.set_index("method").loc[method, "algorithm_family"]
            for budget in budgets:
                metrics, records, monthly = _evaluate_alarm(
                    controlled,
                    year_events,
                    f"budget_alarm_{budget:g}h",
                    budget,
                    int(config["step_minutes"]),
                    online_monthly,
                )
                metrics.update(
                    {
                        "year": year,
                        "method": method,
                        "algorithm_family": family,
                        "selected_parameter_file_sha256": sha256_file(selected_path),
                        "nesting_violations": int(nested["nesting_violations"].sum()),
                        "decision_elapsed_seconds": elapsed,
                        "decision_rows_per_second": frame.shape[0] / max(elapsed, 1e-9),
                        "deployable": 1,
                    }
                )
                metric_rows.append(metrics)
                records["year"], records["method"], records["budget_hours"] = year, method, budget
                event_parts.append(records)
                monthly["year"], monthly["method"] = year, method
                month_parts.append(monthly)
            rows = int(frame.shape[0])
            horizon = int(config["horizon_steps"])
            budget_count = len(budgets)
            complexity_rows.append(
                {
                    "year": year,
                    "method": method,
                    "decision_rows_n": rows,
                    "events_e": int(year_events.shape[0]),
                    "budgets_k": budget_count,
                    "horizon_bins_h": horizon,
                    "elapsed_seconds": elapsed,
                    "rows_per_second": rows / max(elapsed, 1e-9),
                    "hazard_array_bytes_float64": rows * horizon * 8,
                    "decision_mask_bytes_int8": rows * budget_count,
                    "general_time_complexity": "O(N log N + NH + KN H + KEH)",
                    "fixed_horizon_time_complexity": "O(N log N + KN)",
                    "materialized_space_complexity": "O(N(H+K) + MK)",
                    "streaming_controller_space_complexity": "O(KH) per active entity",
                    "trajectory_sha256": sha256_file(trajectory_path),
                }
            )
            print(f"Theory baseline frozen evaluation: {year} {method}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    event_records = pd.concat(event_parts, ignore_index=True)
    months = pd.concat(month_parts, ignore_index=True)
    nesting = pd.concat(nesting_parts, ignore_index=True)
    metrics.to_csv(output / "standard_algorithm_baselines.csv", index=False)
    event_records.to_csv(
        output / "standard_algorithm_event_records.csv.gz", index=False, compression="gzip"
    )
    months.to_csv(
        output / "standard_algorithm_station_month_audit.csv.gz", index=False, compression="gzip"
    )
    nesting.to_csv(output / "standard_algorithm_nesting_audit.csv", index=False)
    pd.DataFrame(complexity_rows).to_csv(output / "controller_complexity_audit.csv", index=False)

    main_records = pd.read_csv(source / "frozen_event_records.csv.gz", low_memory=False)
    bootstrap_parts = []
    for year in years:
        for budget in [value for value in budgets if value in set(config["primary_budgets_hours"])]:
            for method in STANDARD_ALGORITHM_METHODS:
                selected_records = pd.concat(
                    [
                        main_records.loc[
                            main_records["year"].eq(year)
                            & main_records["budget_hours"].eq(budget)
                            & main_records["method"].eq("uadhbac")
                        ],
                        event_records.loc[
                            event_records["year"].eq(year)
                            & event_records["budget_hours"].eq(budget)
                            & event_records["method"].eq(method)
                        ],
                    ],
                    ignore_index=True,
                )
                for queue in ("complete", "operational"):
                    part = paired_station_cluster_bootstrap(
                        selected_records,
                        "uadhbac",
                        method,
                        int(config["bootstrap"]["samples"]),
                        int(config["bootstrap"]["seed"])
                        + year
                        + int(budget * 10)
                        + (0 if queue == "complete" else 10000),
                        queue=queue,
                    )
                    part["year"], part["budget_hours"] = year, budget
                    bootstrap_parts.append(part)
    if bootstrap_parts:
        pd.concat(bootstrap_parts, ignore_index=True).to_csv(
            output / "standard_baseline_station_cluster_bootstrap.csv", index=False
        )

    _nesting_cost_outputs(config, output, smoke=smoke)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - start_time,
        "selection_year": 2022,
        "locked_years": years,
        "budgets_hours": budgets,
        "test_year_model_or_parameter_selection": False,
        "standard_methods": list(STANDARD_ALGORITHM_METHODS),
        "selected_standard_baselines_sha256": sha256_file(selected_path),
        "base_controller_parameters_sha256": sha256_file(base_selected_path),
        "main_results_sha256": sha256_file(source / "frozen_main_results.csv"),
        "main_event_records_sha256": sha256_file(source / "frozen_event_records.csv.gz"),
        "event_table_sha256": sha256_file(resolve_project_path(config["event_table"])),
        "controller_contract_version": DYNAMIC_BUDGET_CONTRACT_VERSION,
        "theory_claim_scope": "deterministic feasibility and nesting; no regret claim for UADHBAC",
    }
    (output / "theory_supplement_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / ".complete_hard_budget_theory_supplement").touch()


def run_sensitivity(config: dict[str, Any], output: Path, smoke: bool = False) -> None:
    """Run frozen budget, utility, and prediction-window sensitivity grids."""

    selected_path = output / "selected_controller_2022.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selection_frame, selection_hazard, _, _ = _load_year(2022, config)
    del selection_frame
    quantile = float(selected["utility_score_quantile"])
    utilities = [str(value) for value in config["utility_sensitivity"]]
    utility_thresholds = {
        utility: float(
            np.quantile(
                trajectory_value(
                    selection_hazard,
                    int(config["step_minutes"]),
                    utility,
                ),
                quantile,
            )
        )
        for utility in utilities
    }
    window_steps = [int(value) for value in config["window_sensitivity_steps"]]
    window_thresholds = {
        steps: float(
            np.quantile(
                trajectory_value(
                    selection_hazard[:, :steps],
                    int(config["step_minutes"]),
                    "linear",
                ),
                quantile,
            )
        )
        for steps in window_steps
    }
    dense_budgets = [float(value) for value in config["dense_budgets_hours"]]
    years = [int(value) for value in config["locked_years"]]
    if smoke:
        dense_budgets = dense_budgets[:3]
        utilities = utilities[:2]
        window_steps = window_steps[:2]
        years = years[:1]
    budget_rows, utility_rows, window_rows = [], [], []
    for year in years:
        frame, hazard, _, _ = _load_year(year, config)
        events = _load_events(resolve_project_path(config["event_table"]))
        year_events = events.loc[events["onset_time"].dt.year.eq(year)]
        base = _controller_config("uadhbac", selected, config)
        controlled, online_monthly, _ = apply_dynamic_hard_budget(
            frame, hazard, dense_budgets, base, events=year_events
        )
        nested = nesting_audit(controlled, dense_budgets)
        for budget in dense_budgets:
            metrics, _, _ = _evaluate_alarm(
                controlled,
                year_events,
                f"budget_alarm_{budget:g}h",
                budget,
                int(config["step_minutes"]),
                online_monthly,
            )
            budget_rows.append(
                {
                    "year": year,
                    "method": "uadhbac",
                    "nesting_violations": int(nested["nesting_violations"].sum()),
                    **metrics,
                }
            )
        for utility in utilities:
            utility_config = DynamicBudgetConfig(
                **{
                    **asdict(base),
                    "utility": utility,
                    "score_threshold": utility_thresholds[utility],
                }
            )
            controlled, online_monthly, _ = apply_dynamic_hard_budget(
                frame,
                hazard,
                [float(value) for value in config["primary_budgets_hours"]],
                utility_config,
                events=year_events,
            )
            for budget in [float(value) for value in config["primary_budgets_hours"]]:
                metrics, _, _ = _evaluate_alarm(
                    controlled, year_events, f"budget_alarm_{budget:g}h", budget,
                    int(config["step_minutes"]), online_monthly,
                )
                utility_rows.append({"year": year, "utility": utility, **metrics})
        for steps in window_steps:
            horizon_hours = steps * int(config["step_minutes"]) / 60.0
            window_config = DynamicBudgetConfig(
                **{
                    **asdict(base),
                    "horizon_steps": steps,
                    "score_threshold": window_thresholds[steps],
                }
            )
            controlled, online_monthly, _ = apply_dynamic_hard_budget(
                frame,
                hazard[:, :steps],
                [float(value) for value in config["primary_budgets_hours"]],
                window_config,
                events=year_events,
            )
            for budget in [float(value) for value in config["primary_budgets_hours"]]:
                metrics, _, _ = _evaluate_alarm(
                    controlled, year_events, f"budget_alarm_{budget:g}h", budget,
                    int(config["step_minutes"]), online_monthly,
                    horizon_hours=horizon_hours,
                )
                window_rows.append(
                    {
                        "year": year,
                        "window_steps": steps,
                        "window_hours": horizon_hours,
                        **metrics,
                    }
                )
        print(f"Sensitivity grids complete: {year}", flush=True)
    pd.DataFrame(budget_rows).to_csv(output / "dense_budget_sensitivity.csv", index=False)
    pd.DataFrame(utility_rows).to_csv(output / "utility_function_sensitivity.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output / "prediction_window_sensitivity.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 2022-only selection and frozen UADHBAC controller experiments."
    )
    parser.add_argument("--config", default="configs/rig_hazard_dynamic_hard_budget.json")
    parser.add_argument(
        "--phase",
        choices=("audit", "select", "evaluate", "sensitivity", "theory", "all"),
        default="all",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--trajectory-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.trajectory_root is not None:
        config["trajectory_root"] = args.trajectory_root
    if args.smoke:
        config["budgets_hours"] = list(config["budgets_hours"][:2])
        config["primary_budgets_hours"] = list(config["budgets_hours"])
        config["locked_years"] = list(config["locked_years"][:1])
        config["bootstrap"]["samples"] = 20
    output = resolve_project_path(args.output_root or config["output_root"])
    if args.overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_experiment_contract.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.phase in ("audit", "all"):
        audit_trajectories(config, output)
    if args.phase in ("select", "all"):
        select_controller(config, output, smoke=args.smoke)
    if args.phase in ("evaluate", "all"):
        run_frozen_years(config, output, include_ablations=not args.skip_ablations)
    if args.phase == "sensitivity" or (args.phase == "all" and not args.skip_sensitivity):
        run_sensitivity(config, output, smoke=args.smoke)
    if args.phase in ("theory", "all"):
        run_theory_supplement(config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()

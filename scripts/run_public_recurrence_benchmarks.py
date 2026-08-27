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
    apply_candidate_policy_by_entity,
    station_month_budget_distribution,
    strict_event_alert_evaluation,
)
from rig_hazard.budget_control import CausalBudgetConfig, apply_causal_budget_by_station
from rig_hazard.config import resolve_project_path
from rig_hazard.dynamic_hard_budget import DynamicBudgetConfig, apply_dynamic_hard_budget
from rig_hazard.public_recurrence import (
    PublicDatasetContract,
    add_causal_recurrence_features,
    aggregate_ecommerce,
    aggregate_us_accidents,
    build_complete_panel,
    chronological_masks,
    event_table,
    fit_public_xgboost,
    read_parquet_with_duckdb,
    score_public_panel,
)


def _evaluate(
    controlled: pd.DataFrame,
    events: pd.DataFrame,
    budget: float,
    contract: PublicDatasetContract,
    *,
    return_monthly: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    controlled = controlled.copy()
    if controlled.empty:
        raise ValueError("No public-benchmark rows to evaluate")
    strict, _, strict_frame = strict_event_alert_evaluation(
        controlled.rename(columns={"entity_id": "station_code"}),
        events,
        alarm_column="budget_alarm",
        step_minutes=contract.step_minutes,
        horizon_hours=contract.horizon_hours,
        observability_column="observed_horizon",
    )
    strict_frame["onset_within_horizon"] = pd.to_numeric(
        strict_frame["onset_within_horizon"], errors="coerce"
    ).fillna(0)
    monthly, distribution = station_month_budget_distribution(
        strict_frame,
        "budget_alarm",
        budget,
        contract.step_minutes,
        future_column="onset_within_horizon",
        strict_false_column="_strict_false_alarm",
        unsettled_column="_strict_unsettled_alarm",
        reserved_column="_strict_reserved_alarm",
    )
    station_months = max(int(distribution["station_months"]), 1)
    fah = strict["strict_false_alarm_bins"] * contract.step_minutes / 60.0 / station_months
    metrics = {
        **strict,
        **distribution,
        "strict_false_alarm_hours_per_entity_month": float(fah),
        "budget_met_mean_fah": int(fah <= budget + 1e-9),
        "budget_met_every_entity_month": int(
            distribution["maximum_false_alarm_hours"] <= budget + 1e-9
        ),
        "budget_met_every_entity_month_reserved": int(
            distribution["maximum_reserved_alarm_hours"] <= budget + 1e-9
        ),
    }
    if return_monthly:
        return metrics, monthly
    return metrics


def _apply_strategy(
    frame: pd.DataFrame,
    strategy: str,
    parameters: dict[str, Any],
    budget: float,
    contract: PublicDatasetContract,
) -> pd.DataFrame:
    renamed = frame.rename(columns={"entity_id": "station_code"})
    if strategy == "original_simple":
        controlled = apply_causal_budget_by_station(
            renamed,
            ["risk_score"],
            CausalBudgetConfig(
                step_minutes=contract.step_minutes,
                monthly_budget_hours=budget,
                trailing_history_days=int(parameters["trailing_history_days"]),
                candidate_quantile=float(parameters["rolling_quantile"]),
                minimum_history_rows=int(parameters["minimum_history_rows"]),
                burst_allowance_hours=float(parameters["burst_allowance_hours"]),
                minimum_candidate_run_bins=1,
                minimum_alarm_run_bins=1,
            ),
        ).rename(
            columns={
                "risk_score__budget_alarm": "budget_alarm",
                "risk_score__candidate_alarm": "candidate_alarm",
                "risk_score__causal_threshold": "causal_threshold",
            }
        )
    else:
        safe = None
        if strategy == "budget_safe":
            safe = SafeBudgetPolicy(
                contract.step_minutes,
                budget,
                burst_allowance_hours=float(parameters["burst_allowance_hours"]),
                pace_multiplier=float(parameters["pace_multiplier"]),
            )
        candidate_fields = CandidatePolicy.__dataclass_fields__
        candidate = CandidatePolicy(
            **{key: value for key, value in parameters.items() if key in candidate_fields}
        )
        controlled = apply_candidate_policy_by_entity(
            renamed,
            "risk_score",
            candidate,
            safe_budget=safe,
        )
    return controlled.rename(columns={"station_code": "entity_id"})


def _grid(selection: pd.DataFrame, settings: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    score = selection["risk_score"].to_numpy(dtype=np.float64)
    legacy_quantiles = settings.get("quantiles")
    absolute_values = settings.get("absolute_quantiles", legacy_quantiles)
    rolling_values = settings.get("rolling_quantiles", legacy_quantiles)
    if not absolute_values or not rolling_values:
        raise KeyError(
            "Public benchmark settings require absolute_quantiles and "
            "rolling_quantiles (or legacy quantiles)."
        )
    absolute_quantiles = [float(value) for value in absolute_values]
    rolling_quantiles = [float(value) for value in rolling_values]
    finite_score = score[np.isfinite(score)]
    if finite_score.size == 0:
        raise ValueError("Public benchmark selection split has no finite risk scores")
    thresholds = {
        value: float(np.quantile(finite_score, value)) for value in absolute_quantiles
    }
    rows: list[tuple[str, dict[str, Any]]] = []
    for quantile, threshold in thresholds.items():
        rows.append(("fixed_threshold", asdict(CandidatePolicy("fixed_threshold", threshold=threshold))))
        rows.append(
            (
                "hysteresis",
                asdict(
                    CandidatePolicy(
                        "hysteresis",
                        threshold=threshold,
                        ema_alpha=0.5,
                        off_threshold_ratio=0.7,
                        minimum_consecutive_bins=1,
                        hold_bins=1,
                    )
                ),
            )
        )
        safe = asdict(
            CandidatePolicy(
                "hysteresis",
                threshold=threshold,
                ema_alpha=0.5,
                off_threshold_ratio=0.7,
                minimum_consecutive_bins=1,
                hold_bins=1,
            )
        )
        safe.update({"burst_allowance_hours": 0.0, "pace_multiplier": 1.0})
        rows.append(("budget_safe", safe))
    for quantile in rolling_quantiles:
        rows.append(
            (
                "rolling_quantile",
                asdict(
                    CandidatePolicy(
                        "rolling_quantile",
                        rolling_quantile=quantile,
                        trailing_history_days=int(settings["trailing_history_days"]),
                        minimum_history_rows=int(settings["minimum_history_rows"]),
                    )
                ),
            )
        )
        rows.append(
            (
                "original_simple",
                {
                    "rolling_quantile": quantile,
                    "trailing_history_days": int(settings["trailing_history_days"]),
                    "minimum_history_rows": int(settings["minimum_history_rows"]),
                    "burst_allowance_hours": 0.0,
                },
            )
        )
    unique = {(strategy, json.dumps(parameters, sort_keys=True)): (strategy, parameters) for strategy, parameters in rows}
    return list(unique.values())


def _constant_hazard_from_risk(score: np.ndarray, horizon_bins: int) -> np.ndarray:
    risk = np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0 - 1e-7)
    hazard = 1.0 - np.power(1.0 - risk, 1.0 / int(horizon_bins))
    return np.repeat(hazard[:, None], int(horizon_bins), axis=1).astype(np.float32)


def _apply_public_dynamic(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    budgets: list[float],
    contract: PublicDatasetContract,
    parameters: dict[str, float],
    method: str,
    candidate_masks: dict[float, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    renamed = frame.rename(columns={"entity_id": "station_code"}).copy()
    trajectory = _constant_hazard_from_risk(renamed["risk_score"], contract.horizon_bins)
    config = DynamicBudgetConfig(
        step_minutes=contract.step_minutes,
        horizon_steps=contract.horizon_bins,
        method=method,
        utility="hit_only",
        score_threshold=float(parameters["risk_score_threshold"]),
        price_learning_rate=float(parameters.get("price_learning_rate", 2.0)),
        pending_pressure=float(parameters.get("pending_pressure", 1.0)),
        use_lead_utility=False,
        use_dynamic_price=method not in {"utility_static", "fixed_threshold_hard_guard"},
        couple_budgets=method == "uadhbac_public_Fh_only",
    )
    controlled, monthly, _ = apply_dynamic_hard_budget(
        renamed,
        trajectory,
        budgets,
        config,
        events=events,
        observability_column="observed_horizon",
        candidate_masks=candidate_masks,
    )
    return controlled.rename(columns={"station_code": "entity_id"}), monthly


def _fixed_threshold_parameters_for_guard(
    candidates: pd.DataFrame,
    budget: float,
) -> dict[str, Any]:
    """Select a development-only fixed-threshold candidate for the hard guard.

    The unguarded candidate is not required to satisfy the budget itself: the
    online hard guard is the component responsible for enforcing feasibility.
    Requiring pre-guard feasibility can leave low-budget levels without a
    candidate even though the guarded policy is well-defined and feasible.
    """

    rows = candidates.loc[
        candidates["budget_hours_per_entity_month"].eq(float(budget))
        & candidates["strategy"].eq("fixed_threshold")
    ].copy()
    if rows.empty:
        raise ValueError(f"No development fixed-threshold candidate for B={budget}")
    winner = rows.sort_values(
        ["lead_utility_hours", "event_hit_rate", "strict_false_alarm_hours_per_entity_month"],
        ascending=[False, False, True],
    ).iloc[0]
    return json.loads(str(winner["parameters"]))


def _run_dataset(
    name: str,
    panel: pd.DataFrame,
    contract: PublicDatasetContract,
    settings: dict[str, Any],
    output_root: Path,
    *,
    resume: bool = False,
) -> None:
    panel, feature_names = add_causal_recurrence_features(panel, contract.horizon_bins)
    masks = chronological_masks(panel, contract)
    model_path = output_root / "xgboost.json"
    training_path = output_root / "xgboost_training.json"
    if resume and model_path.exists():
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(model_path)
        training = (
            json.loads(training_path.read_text(encoding="utf-8"))
            if training_path.exists()
            else {
                "resumed_from_model_checkpoint": True,
                "model_checkpoint": str(model_path),
            }
        )
        print(f"Resumed public model checkpoint: {model_path}", flush=True)
    else:
        model, training = fit_public_xgboost(
            panel,
            feature_names,
            masks["train"],
            int(settings["seed"]),
            int(settings["maximum_training_rows"]),
            settings["xgboost"],
        )
        model.save_model(model_path)
        training_path.write_text(
            json.dumps(training, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    panel["risk_score"] = score_public_panel(model, panel, feature_names)
    events = event_table(panel)
    issue_time = pd.to_datetime(panel["issue_time"], errors="coerce")
    selection = panel.loc[masks["selection"]].copy()
    test = panel.loc[masks["test"]].copy()
    context_days = max(int(settings["trailing_history_days"]) + 2, 35)
    selection_context = panel.loc[
        issue_time.between(
            pd.Timestamp(contract.train_end) - pd.Timedelta(days=context_days),
            pd.Timestamp(contract.selection_end),
            inclusive="left",
        )
    ].copy()
    test_context = panel.loc[
        issue_time.between(
            pd.Timestamp(contract.selection_end) - pd.Timedelta(days=context_days),
            pd.Timestamp(contract.test_end),
            inclusive="left",
        )
    ].copy()
    selection_events = events.loc[
        pd.to_datetime(events["onset_time"]).between(
            pd.Timestamp(contract.train_end), pd.Timestamp(contract.selection_end), inclusive="left"
        )
    ]
    test_events = events.loc[
        pd.to_datetime(events["onset_time"]).between(
            pd.Timestamp(contract.selection_end), pd.Timestamp(contract.test_end), inclusive="left"
        )
    ]
    candidates_path = output_root / "selection_candidates.csv.gz"
    selected_path = output_root / "selected_policies.csv"
    if resume and candidates_path.exists() and selected_path.exists():
        candidates = pd.read_csv(candidates_path)
        selected = pd.read_csv(selected_path)
        print(f"Resumed public policy checkpoints: {output_root}", flush=True)
    else:
        candidate_rows: list[dict[str, Any]] = []
        for budget in contract.budgets_hours:
            for strategy, parameters in _grid(selection, settings):
                controlled = _apply_strategy(
                    selection_context, strategy, parameters, budget, contract
                )
                controlled = controlled.loc[
                    pd.to_datetime(controlled["issue_time"], errors="coerce").between(
                        pd.Timestamp(contract.train_end),
                        pd.Timestamp(contract.selection_end),
                        inclusive="left",
                    )
                ].copy()
                metrics = _evaluate(controlled, selection_events, budget, contract)
                candidate_rows.append(
                    {
                        "dataset": name,
                        "budget_hours_per_entity_month": budget,
                        "strategy": strategy,
                        "parameters": json.dumps(parameters, sort_keys=True),
                        **metrics,
                    }
                )
        candidates = pd.DataFrame(candidate_rows)
        candidates.to_csv(candidates_path, index=False, compression="gzip")
        selected_rows: list[pd.Series] = []
        for (budget, strategy), rows in candidates.groupby(
            ["budget_hours_per_entity_month", "strategy"], sort=True
        ):
            eligible = rows.loc[rows["budget_met_mean_fah"].eq(1)].copy()
            if strategy in {"original_simple", "budget_safe"}:
                eligible = eligible.loc[eligible["budget_met_every_entity_month"].eq(1)]
            if eligible.empty:
                continue
            selected_rows.append(
                eligible.sort_values(
                    ["lead_utility_hours", "event_hit_rate", "strict_false_alarm_hours_per_entity_month"],
                    ascending=[False, False, True],
                ).iloc[0]
            )
        selected = pd.DataFrame(selected_rows)
        selected.to_csv(selected_path, index=False)

    # Select the public F_h-only dynamic controller on the public development
    # split.  No frozen-test row or label participates in this search.
    selection_score = selection["risk_score"].to_numpy(dtype=np.float64)
    budgets = [float(value) for value in contract.budgets_hours]
    dynamic_parameters_path = output_root / "selected_dynamic_controller.json"
    if resume and dynamic_parameters_path.exists():
        dynamic_parameters = json.loads(dynamic_parameters_path.read_text(encoding="utf-8"))
        print(f"Resumed public dynamic-controller checkpoint: {output_root}", flush=True)
    else:
        dynamic_candidates: list[dict[str, Any]] = []
        dynamic_grid = [
            (float(quantile), float(rate), float(pending))
            for quantile in settings.get("dynamic_score_quantiles", [0.95, 0.98])
            for rate in settings.get("dynamic_price_learning_rates", [1.0, 2.0])
            for pending in settings.get("dynamic_pending_pressures", [1.0])
        ]
        for candidate_id, (quantile, rate, pending) in enumerate(dynamic_grid, start=1):
            parameters = {
                "risk_score_quantile": quantile,
                "risk_score_threshold": float(np.quantile(selection_score, quantile)),
                "price_learning_rate": rate,
                "pending_pressure": pending,
            }
            controlled, _ = _apply_public_dynamic(
                selection_context,
                events,
                budgets,
                contract,
                parameters,
                "uadhbac_public_Fh_only",
            )
            controlled = controlled.loc[
                pd.to_datetime(controlled["issue_time"], errors="coerce").between(
                    pd.Timestamp(contract.train_end),
                    pd.Timestamp(contract.selection_end),
                    inclusive="left",
                )
            ].copy()
            for budget in budgets:
                evaluated = controlled.rename(
                    columns={f"budget_alarm_{budget:g}h": "budget_alarm"}
                )
                metrics = _evaluate(evaluated, selection_events, budget, contract)
                dynamic_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        **parameters,
                        "budget_hours_per_entity_month": budget,
                        **metrics,
                    }
                )
        dynamic_search = pd.DataFrame(dynamic_candidates)
        dynamic_search.to_csv(
            output_root / "dynamic_selection_candidates.csv.gz", index=False, compression="gzip"
        )
        dynamic_summary = (
            dynamic_search.assign(
                feasible=lambda x: x["budget_met_every_entity_month_reserved"].eq(1)
            )
            .groupby("candidate_id", as_index=False)
            .agg(feasible=("feasible", "all"), lead_utility_hours=("lead_utility_hours", "mean"))
        )
        eligible_dynamic = dynamic_summary.loc[dynamic_summary["feasible"]]
        if eligible_dynamic.empty:
            raise RuntimeError("No public dynamic controller candidate satisfies reserved hard budgets")
        dynamic_winner_id = int(
            eligible_dynamic.sort_values("lead_utility_hours", ascending=False).iloc[0]["candidate_id"]
        )
        dynamic_winner = dynamic_search.loc[
            dynamic_search["candidate_id"].eq(dynamic_winner_id)
        ].iloc[0]
        dynamic_parameters = {
            "risk_score_quantile": float(dynamic_winner["risk_score_quantile"]),
            "risk_score_threshold": float(dynamic_winner["risk_score_threshold"]),
            "price_learning_rate": float(dynamic_winner["price_learning_rate"]),
            "pending_pressure": float(dynamic_winner["pending_pressure"]),
            "selection_split": "development_only",
            "frozen_test_used_for_selection": False,
        }
        dynamic_parameters_path.write_text(
            json.dumps(dynamic_parameters, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    result_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        budget = float(row.budget_hours_per_entity_month)
        parameters = json.loads(row.parameters)
        controlled = _apply_strategy(
            test_context, str(row.strategy), parameters, budget, contract
        )
        controlled = controlled.loc[
            pd.to_datetime(controlled["issue_time"], errors="coerce").between(
                pd.Timestamp(contract.selection_end),
                pd.Timestamp(contract.test_end),
                inclusive="left",
            )
        ].copy()
        metrics, monthly = _evaluate(
            controlled,
            test_events,
            budget,
            contract,
            return_monthly=True,
        )
        monthly = monthly.assign(
            dataset=name,
            split="frozen_test",
            budget_hours_per_entity_month=budget,
            strategy=str(row.strategy),
        )
        monthly_rows.append(monthly)
        result_rows.append(
            {
                "dataset": name,
                "split": "frozen_test",
                "budget_hours_per_entity_month": budget,
                "budget_bins_per_entity_month": budget * 60.0 / contract.step_minutes,
                "strategy": str(row.strategy),
                **metrics,
            }
        )

    dynamic_test, dynamic_online_monthly = _apply_public_dynamic(
        test_context,
        events,
        budgets,
        contract,
        dynamic_parameters,
        "uadhbac_public_Fh_only",
    )
    dynamic_test = dynamic_test.loc[
        pd.to_datetime(dynamic_test["issue_time"], errors="coerce").between(
            pd.Timestamp(contract.selection_end), pd.Timestamp(contract.test_end), inclusive="left"
        )
    ].copy()
    for method in ("uadhbac_public_Fh_only", "utility_static_hard_guard"):
        method_test, method_monthly = (
            (dynamic_test, dynamic_online_monthly)
            if method == "uadhbac_public_Fh_only"
            else _apply_public_dynamic(
                test_context,
                events,
                budgets,
                contract,
                dynamic_parameters,
                "utility_static",
            )
        )
        method_test = method_test.loc[
            pd.to_datetime(method_test["issue_time"], errors="coerce").between(
                pd.Timestamp(contract.selection_end), pd.Timestamp(contract.test_end), inclusive="left"
            )
        ].copy()
        for budget in budgets:
            evaluated = method_test.rename(
                columns={f"budget_alarm_{budget:g}h": "budget_alarm"}
            )
            metrics, monthly = _evaluate(
                evaluated, test_events, budget, contract, return_monthly=True
            )
            online_selected = method_monthly.loc[
                method_monthly["budget_hours"].eq(budget)
            ]
            metrics["maximum_instantaneous_reserved_hours"] = float(
                online_selected["maximum_reserved_hours"].max()
            )
            metrics["online_hard_budget_violations"] = int(
                online_selected["hard_budget_met"].eq(0).sum()
            )
            monthly_rows.append(
                monthly.assign(
                    dataset=name,
                    split="frozen_test",
                    budget_hours_per_entity_month=budget,
                    strategy=method,
                )
            )
            result_rows.append(
                {
                    "dataset": name,
                    "split": "frozen_test",
                    "budget_hours_per_entity_month": budget,
                    "budget_bins_per_entity_month": budget * 60.0 / contract.step_minutes,
                    "strategy": method,
                    **metrics,
                }
            )

    fixed_masks: dict[float, np.ndarray] = {}
    for budget in budgets:
        candidate = _apply_strategy(
            test_context,
            "fixed_threshold",
            _fixed_threshold_parameters_for_guard(candidates, budget),
            budget,
            contract,
        )
        fixed_masks[budget] = candidate["candidate_alarm"].to_numpy(dtype=bool)
    fixed_guard, fixed_online_monthly = _apply_public_dynamic(
        test_context,
        events,
        budgets,
        contract,
        dynamic_parameters,
        "fixed_threshold_hard_guard",
        candidate_masks=fixed_masks,
    )
    fixed_guard = fixed_guard.loc[
        pd.to_datetime(fixed_guard["issue_time"], errors="coerce").between(
            pd.Timestamp(contract.selection_end), pd.Timestamp(contract.test_end), inclusive="left"
        )
    ].copy()
    for budget in budgets:
        evaluated = fixed_guard.rename(
            columns={f"budget_alarm_{budget:g}h": "budget_alarm"}
        )
        metrics, monthly = _evaluate(
            evaluated, test_events, budget, contract, return_monthly=True
        )
        online_selected = fixed_online_monthly.loc[
            fixed_online_monthly["budget_hours"].eq(budget)
        ]
        metrics["maximum_instantaneous_reserved_hours"] = float(
            online_selected["maximum_reserved_hours"].max()
        )
        metrics["online_hard_budget_violations"] = int(
            online_selected["hard_budget_met"].eq(0).sum()
        )
        monthly_rows.append(
            monthly.assign(
                dataset=name,
                split="frozen_test",
                budget_hours_per_entity_month=budget,
                strategy="fixed_threshold_hard_guard",
            )
        )
        result_rows.append(
            {
                "dataset": name,
                "split": "frozen_test",
                "budget_hours_per_entity_month": budget,
                "budget_bins_per_entity_month": budget * 60.0 / contract.step_minutes,
                "strategy": "fixed_threshold_hard_guard",
                **metrics,
            }
        )
    pd.DataFrame(result_rows).to_csv(output_root / "frozen_test_metrics.csv", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_csv(
        output_root / "frozen_station_month_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    manifest = {
        "dataset": name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "contract": asdict(contract),
        "entities": int(panel["entity_id"].nunique()),
        "panel_rows": int(panel.shape[0]),
        "events": int((panel["event_count"] > 0).sum()),
        "feature_names": feature_names,
        "training": training,
        "scientific_role": (
            "cross-domain recurrent-event and alert-controller benchmark; not external icing validation"
        ),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / ".complete_public_benchmark").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two public recurrent-event benchmarks.")
    parser.add_argument("--config", default="configs/rig_hazard_alert_governance.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("ecommerce", "us_accidents"),
        default=["ecommerce"],
        help="US accidents is opt-in diagnostic only until its zero-event panel is rebuilt.",
    )
    args = parser.parse_args()
    config = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    settings = config["public_benchmarks"]
    output_root = resolve_project_path(args.output_root or settings["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.resume and (output_root / ".complete_public_benchmarks").exists():
        print(f"Public benchmarks already complete: {output_root}", flush=True)
        return
    cache_root = output_root / "prepared"
    cache_root.mkdir(parents=True, exist_ok=True)

    if "ecommerce" in args.datasets:
        ecommerce = settings["ecommerce"]
        ecommerce_aggregate = aggregate_ecommerce(
            resolve_project_path(ecommerce["source"]),
            cache_root / "ecommerce_hourly.parquet",
            int(ecommerce["maximum_entities"]),
        )
        ecommerce_panel = build_complete_panel(
            read_parquet_with_duckdb(ecommerce_aggregate),
            int(ecommerce["step_minutes"]),
            ecommerce["start"],
            ecommerce["test_end"],
        )
        ecommerce_root = output_root / "ecommerce"
        ecommerce_root.mkdir(parents=True, exist_ok=True)
        if not (ecommerce_root / ".complete_public_benchmark").exists():
            _run_dataset(
                "multi_category_ecommerce_2019_nov",
                ecommerce_panel,
                PublicDatasetContract(
                    "multi_category_ecommerce_2019_nov",
                    int(ecommerce["step_minutes"]),
                    int(ecommerce["horizon_bins"]),
                    ecommerce["train_end"],
                    ecommerce["selection_end"],
                    ecommerce["test_end"],
                    tuple(float(value) for value in ecommerce["budgets_hours"]),
                ),
                settings,
                ecommerce_root,
                resume=args.resume,
            )
        print("Public ecommerce benchmark complete", flush=True)

    if "us_accidents" in args.datasets:
        accidents = settings["us_accidents"]
        accident_aggregate = aggregate_us_accidents(
            resolve_project_path(accidents["source"]),
            cache_root / "us_accidents_daily.parquet",
            int(accidents["maximum_entities"]),
            float(accidents["grid_degrees"]),
        )
        accident_panel = build_complete_panel(
            read_parquet_with_duckdb(accident_aggregate),
            int(accidents["step_minutes"]),
            accidents["start"],
            accidents["test_end"],
        )
        accident_root = output_root / "us_accidents"
        accident_root.mkdir(parents=True, exist_ok=True)
        if not (accident_root / ".complete_public_benchmark").exists():
            _run_dataset(
                "us_accidents_spatial_recurrence_diagnostic_only",
                accident_panel,
                PublicDatasetContract(
                    "us_accidents_spatial_recurrence_diagnostic_only",
                    int(accidents["step_minutes"]),
                    int(accidents["horizon_bins"]),
                    accidents["train_end"],
                    accidents["selection_end"],
                    accidents["test_end"],
                    tuple(float(value) for value in accidents["budgets_hours"]),
                ),
                settings,
                accident_root,
                resume=args.resume,
            )
        print("Public US accidents diagnostic complete", flush=True)
    (output_root / ".complete_public_benchmarks").touch()
    print(f"All public recurrent-event benchmarks complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()

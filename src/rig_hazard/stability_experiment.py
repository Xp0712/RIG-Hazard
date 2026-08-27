from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .artifact_compat import read_artifact_csv
from .baseline_experiment import evaluate_warning_model, markdown_table, probability_metric_row, read_timeline, timeline_files
from .baseline_models import HazardRateCalibrator
from .budget_control import CausalBudgetConfig, apply_causal_budget_by_station
from .config import resolve_project_path
from .graph_data import (
    GraphSchema,
    LocalHazardBundle,
    SignalBank,
    build_graph_schema,
    build_structured_design,
    build_structured_training_sample,
    load_local_hazard_bundle,
)
from .graph_experiment import (
    barrier_effect_table,
    cluster_bootstrap_event_comparison,
    event_alarm_records,
    model_penalties_and_bounds,
)
from .graph_models import StructuredCloglogHazard
from .preprocessing import prepare_output_root, write_json


STABILITY_ETA_COLUMNS = {
    "local_weather_hazard": "local_weather_hazard_eta",
    "unfiltered_graph": "unfiltered_graph_eta",
    "unfiltered_graph_ablation": "unfiltered_graph_ablation_eta",
    "stable_graph": "stable_graph_eta",
    "stable_graph_ablation": "stable_graph_ablation_eta",
}


def station_month_subsample_mask(
    metadata: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 0 < fraction < 1:
        raise ValueError("block subsample fraction must be between zero and one")
    selected_blocks: set[str] = set()
    for _, station_frame in metadata.groupby("station_code", sort=False):
        blocks = station_frame["station_month_block"].drop_duplicates().astype(str).to_numpy()
        count = max(1, int(np.ceil(fraction * blocks.size)))
        selected_blocks.update(rng.choice(blocks, size=count, replace=False).tolist())
    return metadata["station_month_block"].astype(str).isin(selected_blocks).to_numpy()


def run_block_stability_selection(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    sample_weight: np.ndarray,
    metadata: pd.DataFrame,
    schema: GraphSchema,
    original_model: StructuredCloglogHazard,
    graph_l1: float,
    graph_config: dict[str, Any],
    stability_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    l2, l1, bounds = model_penalties_and_bounds(schema, graph_config["hierarchy"], graph_l1)
    replicates = int(stability_config["replicates"])
    coefficient_threshold = float(stability_config["coefficient_threshold"])
    fraction = float(stability_config["block_fraction"])
    rng = np.random.default_rng(int(stability_config["seed"]))
    graph_coefficients: list[np.ndarray] = []
    summary_rows: list[dict[str, Any]] = []

    for replicate in range(replicates):
        mask = station_month_subsample_mask(metadata, fraction, rng)
        replicate_weight = sample_weight * mask.astype(np.float64) / fraction
        model = StructuredCloglogHazard(
            feature_names=schema.feature_names,
            l2_penalties=l2,
            l1_penalties=l1,
            coefficient_bounds=bounds,
            max_iter=int(stability_config["max_iter"]),
            tolerance=float(stability_config["tolerance"]),
        )
        model.fit(
            matrix,
            y,
            sample_weight=replicate_weight,
            initial_intercept=original_model.intercept,
            initial_coefficients=original_model.coefficients,
        )
        coefficients = model.coefficients[schema.base_feature_count :].copy()
        graph_coefficients.append(coefficients)
        summary_rows.append(
            {
                "replicate": replicate,
                "converged": int(model.converged),
                "iterations": model.iterations,
                "objective": model.objective,
                "selected_blocks": int(metadata.loc[mask, "station_month_block"].nunique()),
                "represented_positive_rows": int(y[mask].sum()),
                "active_edge_lags": int((coefficients > coefficient_threshold).sum()),
            }
        )
        print(
            f"Stability replicate {replicate + 1}/{replicates}: converged={model.converged} "
            f"iterations={model.iterations} active={(coefficients > coefficient_threshold).sum()}",
            flush=True,
        )

    coefficient_matrix = np.vstack(graph_coefficients)
    frequency = (coefficient_matrix > coefficient_threshold).mean(axis=0)
    edge_table = schema.edge_features.copy()
    edge_table["original_coefficient"] = original_model.coefficients[schema.base_feature_count :]
    edge_table["selection_frequency"] = frequency
    edge_table["coefficient_median"] = np.median(coefficient_matrix, axis=0)
    edge_table["coefficient_q25"] = np.quantile(coefficient_matrix, 0.25, axis=0)
    edge_table["coefficient_q75"] = np.quantile(coefficient_matrix, 0.75, axis=0)
    stable = frequency >= float(stability_config["selection_frequency"])
    stable &= edge_table["original_coefficient"].to_numpy(dtype=np.float64) > coefficient_threshold
    edge_table["stable"] = stable.astype(int)
    edge_table = edge_table.sort_values(
        ["stable", "selection_frequency", "coefficient_median"], ascending=[False, False, False]
    ).reset_index(drop=True)
    return edge_table, pd.DataFrame(summary_rows), stable


def fit_stable_graph_model(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    sample_weight: np.ndarray,
    schema: GraphSchema,
    original_model: StructuredCloglogHazard,
    stable_graph_mask: np.ndarray,
    graph_l1: float,
    graph_config: dict[str, Any],
    stability_config: dict[str, Any],
) -> StructuredCloglogHazard:
    l2, l1, bounds = model_penalties_and_bounds(schema, graph_config["hierarchy"], graph_l1)
    for graph_index, is_stable in enumerate(stable_graph_mask):
        if not is_stable:
            absolute_index = schema.base_feature_count + graph_index
            bounds[absolute_index] = (0.0, 0.0)
            l1[absolute_index] = 0.0
    initial = original_model.coefficients.copy()
    initial[schema.base_feature_count :][~stable_graph_mask] = 0.0
    model = StructuredCloglogHazard(
        feature_names=schema.feature_names,
        l2_penalties=l2,
        l1_penalties=l1,
        coefficient_bounds=bounds,
        max_iter=int(stability_config["refit_max_iter"]),
        tolerance=float(stability_config["tolerance"]),
    )
    model.fit(
        matrix,
        y,
        sample_weight=sample_weight,
        initial_intercept=original_model.intercept,
        initial_coefficients=initial,
    )
    print(
        f"Stable graph refit: converged={model.converged} iterations={model.iterations} "
        f"stable_edges={stable_graph_mask.sum()}",
        flush=True,
    )
    return model


def prediction_risk_frame(path: Path, bundle: LocalHazardBundle, horizons: list[int]) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [
                *bundle.transformer.feature_names,
                "station_code",
                "station_name",
                "city",
                "seen_in_development",
                "issue_time",
                "risk_set",
                "hazard_label",
                *[f"onset_within_{horizon}h" for horizon in horizons],
                *[f"hard_negative_{horizon}h" for horizon in horizons],
            ]
        )
    )
    frame = read_timeline(path, columns)
    frame = frame.loc[frame["risk_set"].eq(1)].reset_index(drop=True)
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], errors="coerce")
    return frame


def generate_stability_predictions(
    paths: list[Path],
    destination_root: Path,
    schema: GraphSchema,
    signal_cache_root: Path,
    bundle: LocalHazardBundle,
    original_model: StructuredCloglogHazard,
    stable_model: StructuredCloglogHazard,
    horizons: list[int],
    step_minutes: int,
    rolling_window_minutes: int,
    compression_level: int,
) -> list[Path]:
    banks: dict[int, SignalBank] = {}
    outputs: list[Path] = []
    for index, path in enumerate(paths, start=1):
        frame = prediction_risk_frame(path, bundle, horizons)
        if frame.empty:
            continue
        year = int(frame["issue_time"].dt.year.mode().iloc[0])
        bank = banks.setdefault(year, SignalBank(signal_cache_root, year, step_minutes, rolling_window_minutes))
        matrix, local_eta = build_structured_design(frame, schema, bank, bundle)
        base = matrix[:, : schema.base_feature_count]
        original_eta = original_model.decision_function(matrix)
        original_ablation = original_model.intercept + np.asarray(
            base @ original_model.coefficients[: schema.base_feature_count]
        ).ravel()
        stable_eta = stable_model.decision_function(matrix)
        stable_ablation = stable_model.intercept + np.asarray(
            base @ stable_model.coefficients[: schema.base_feature_count]
        ).ravel()
        output_columns = [
            "station_code",
            "station_name",
            "city",
            "seen_in_development",
            "issue_time",
            "hazard_label",
            *[f"onset_within_{horizon}h" for horizon in horizons],
            *[f"hard_negative_{horizon}h" for horizon in horizons],
        ]
        output = frame[output_columns].copy()
        output["local_weather_hazard_eta"] = local_eta.astype(np.float32)
        output["unfiltered_graph_eta"] = original_eta.astype(np.float32)
        output["unfiltered_graph_ablation_eta"] = original_ablation.astype(np.float32)
        output["stable_graph_eta"] = stable_eta.astype(np.float32)
        output["stable_graph_ablation_eta"] = stable_ablation.astype(np.float32)
        output["stable_graph_contribution"] = (stable_eta - stable_ablation).astype(np.float32)
        destination = destination_root / str(year) / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(
            destination,
            index=False,
            encoding="utf-8-sig",
            compression={"method": "gzip", "compresslevel": compression_level},
        )
        outputs.append(destination)
        if index % 10 == 0 or index == len(paths):
            print(f"Generated stability predictions for {index}/{len(paths)} files", flush=True)
    return outputs


def fit_stability_calibrators(paths: list[Path], calibration_year: int) -> dict[str, HazardRateCalibrator]:
    labels: list[np.ndarray] = []
    scores: dict[str, list[np.ndarray]] = {name: [] for name in STABILITY_ETA_COLUMNS}
    columns = ["issue_time", "hazard_label", *STABILITY_ETA_COLUMNS.values()]
    for path in paths:
        frame = read_artifact_csv(
            path, columns=columns, encoding="utf-8-sig", low_memory=False
        )
        issue_time = pd.to_datetime(frame["issue_time"], errors="coerce")
        mask = issue_time.dt.year.eq(calibration_year)
        if not mask.any():
            continue
        labels.append(frame.loc[mask, "hazard_label"].to_numpy(dtype=np.int8))
        for name, column in STABILITY_ETA_COLUMNS.items():
            scores[name].append(frame.loc[mask, column].to_numpy(dtype=np.float64))
    y = np.concatenate(labels)
    calibrators: dict[str, HazardRateCalibrator] = {}
    for name in STABILITY_ETA_COLUMNS:
        calibrator = HazardRateCalibrator().fit(np.concatenate(scores[name]), y)
        calibrators[name] = calibrator
        print(
            f"Stability calibration {name}: converged={calibrator.converged} "
            f"shift={calibrator.log_rate_shift:.5f} slope={calibrator.slope:.5f}",
            flush=True,
        )
    return calibrators


def apply_stability_calibrators(
    paths: list[Path],
    calibrators: dict[str, HazardRateCalibrator],
    horizons: list[int],
    step_minutes: int,
    compression_level: int,
) -> None:
    for index, path in enumerate(paths, start=1):
        frame = read_artifact_csv(path, encoding="utf-8-sig", low_memory=False)
        for name, eta_column in STABILITY_ETA_COLUMNS.items():
            eta = frame[eta_column].to_numpy(dtype=np.float64)
            frame[f"{name}_step"] = calibrators[name].predict(eta).astype(np.float32)
            for horizon in horizons:
                frame[f"{name}_{horizon}h"] = calibrators[name].predict(
                    eta, steps=horizon * 60 / step_minutes
                ).astype(np.float32)
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
            compression={"method": "gzip", "compresslevel": compression_level},
        )
        if index % 10 == 0 or index == len(paths):
            print(f"Applied stability calibration to {index}/{len(paths)} files", flush=True)


def evaluate_stability_probability(paths: list[Path], horizons: list[int], year: int) -> pd.DataFrame:
    columns = [
        "seen_in_development",
        "issue_time",
        "hazard_label",
        *[f"onset_within_{horizon}h" for horizon in horizons],
    ]
    for model in STABILITY_ETA_COLUMNS:
        columns.append(f"{model}_step")
        columns.extend(f"{model}_{horizon}h" for horizon in horizons)
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = read_artifact_csv(
            path, columns=columns, encoding="utf-8-sig", low_memory=False
        )
        issue_time = pd.to_datetime(frame["issue_time"], errors="coerce")
        frame = frame.loc[issue_time.dt.year.eq(year)]
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    seen = pd.to_numeric(combined["seen_in_development"], errors="coerce").fillna(0).eq(1).to_numpy()
    cohorts = {"all": np.ones(combined.shape[0], dtype=bool), "seen": seen, "unseen": ~seen}
    rows: list[dict[str, Any]] = []
    for cohort, mask in cohorts.items():
        if not mask.any():
            continue
        subset = combined.loc[mask]
        for model in STABILITY_ETA_COLUMNS:
            row = probability_metric_row(
                f"{model}_step",
                "hazard_label",
                subset["hazard_label"].to_numpy(dtype=np.int8),
                subset[f"{model}_step"].to_numpy(dtype=np.float64),
            )
            row.update({"cohort": cohort, "year": year})
            rows.append(row)
            for horizon in horizons:
                label = f"onset_within_{horizon}h"
                row = probability_metric_row(
                    f"{model}_{horizon}h",
                    label,
                    subset[label].to_numpy(dtype=np.int8),
                    subset[f"{model}_{horizon}h"].to_numpy(dtype=np.float64),
                )
                row.update({"cohort": cohort, "year": year})
                rows.append(row)
    return pd.DataFrame(rows)


def generate_causal_budget_predictions(
    prediction_paths: list[Path],
    destination_root: Path,
    evaluation_years: list[int],
    budget_config: CausalBudgetConfig,
    compression_level: int,
) -> list[Path]:
    paths_by_station: dict[str, list[Path]] = {}
    for path in prediction_paths:
        paths_by_station.setdefault(path.name.split("_", 1)[0], []).append(path)
    outputs: list[Path] = []
    eta_columns = list(STABILITY_ETA_COLUMNS.values())
    label_columns = [
        "station_code",
        "station_name",
        "seen_in_development",
        "issue_time",
        "hazard_label",
        "onset_within_6h",
        "hard_negative_6h",
    ]
    for station_index, (station_code, station_paths) in enumerate(sorted(paths_by_station.items()), start=1):
        frames = [
            read_artifact_csv(
                path,
                columns=[*label_columns, *eta_columns],
                encoding="utf-8-sig",
                low_memory=False,
            )
            for path in sorted(station_paths)
        ]
        station_frame = pd.concat(frames, ignore_index=True)
        station_frame["issue_time"] = pd.to_datetime(station_frame["issue_time"], errors="coerce")
        controlled = apply_causal_budget_by_station(station_frame, eta_columns, budget_config)
        for year in evaluation_years:
            yearly = controlled.loc[controlled["issue_time"].dt.year.eq(year)].copy()
            if yearly.empty:
                continue
            destination = destination_root / str(year) / f"{station_code}.csv.gz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            yearly.to_csv(
                destination,
                index=False,
                encoding="utf-8-sig",
                compression={"method": "gzip", "compresslevel": compression_level},
            )
            outputs.append(destination)
        if station_index % 10 == 0 or station_index == len(paths_by_station):
            print(f"Generated causal-budget alarms for {station_index}/{len(paths_by_station)} stations", flush=True)
    return outputs


def select_causal_budget_quantile(
    prediction_paths: list[Path],
    events: pd.DataFrame,
    selection_year: int,
    candidate_quantiles: list[float],
    base_config: CausalBudgetConfig,
    warning: dict[str, Any],
    step_minutes: int,
    utility_tie_hours: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """Select one model-agnostic budget quantile on a development year only."""

    quantiles = sorted({float(value) for value in candidate_quantiles})
    if not quantiles or any(not 0 < value < 1 for value in quantiles):
        raise ValueError("candidate_quantiles must contain values strictly between zero and one")
    if utility_tie_hours < 0:
        raise ValueError("utility_tie_hours must be non-negative")

    horizon = int(warning["horizon_hours"])
    eta_column = STABILITY_ETA_COLUMNS["local_weather_hazard"]
    alarm_column = f"{eta_column}__budget_alarm"
    columns = [
        "station_code",
        "station_name",
        "issue_time",
        f"onset_within_{horizon}h",
        f"hard_negative_{horizon}h",
        eta_column,
    ]
    paths_by_station: dict[str, list[Path]] = {}
    for path in prediction_paths:
        try:
            path_year = int(path.parent.name)
        except ValueError:
            continue
        if path_year <= selection_year:
            paths_by_station.setdefault(path.name.split("_", 1)[0], []).append(path)

    selection_parts: dict[float, list[pd.DataFrame]] = {quantile: [] for quantile in quantiles}
    for station_index, station_paths in enumerate(paths_by_station.values(), start=1):
        station_frame = pd.concat(
            [
                read_artifact_csv(
                    path,
                    columns=columns,
                    encoding="utf-8-sig",
                    low_memory=False,
                )
                for path in sorted(station_paths)
            ],
            ignore_index=True,
        )
        station_frame["issue_time"] = pd.to_datetime(station_frame["issue_time"], errors="coerce")
        station_frame = station_frame.sort_values("issue_time").reset_index(drop=True)
        for quantile in quantiles:
            controlled = apply_causal_budget_by_station(
                station_frame,
                [eta_column],
                replace(base_config, candidate_quantile=quantile),
            )
            selected = controlled.loc[controlled["issue_time"].dt.year.eq(selection_year)].copy()
            if not selected.empty:
                selection_parts[quantile].append(selected)
        if station_index % 10 == 0 or station_index == len(paths_by_station):
            print(
                f"Evaluated budget-quantile grid for {station_index}/{len(paths_by_station)} stations",
                flush=True,
            )

    selection_events = events.copy()
    selection_events["onset_time"] = pd.to_datetime(selection_events["onset_time"], errors="coerce")
    selection_events["valid_target_event"] = (
        pd.to_numeric(selection_events["valid_target_event"], errors="coerce").fillna(0).astype(int)
    )
    selection_events = selection_events.loc[selection_events["onset_time"].dt.year.eq(selection_year)]
    rows: list[dict[str, Any]] = []
    for quantile in quantiles:
        if not selection_parts[quantile]:
            raise ValueError(f"No prediction rows available for budget selection year {selection_year}")
        frame = pd.concat(selection_parts[quantile], ignore_index=True)
        frame["station_month"] = (
            frame["station_code"].astype(str) + "|" + frame["issue_time"].dt.to_period("M").astype(str)
        )
        metric = evaluate_warning_model(
            frame,
            selection_events,
            alarm_column,
            0.5,
            horizon,
            step_minutes,
            float(warning["false_alarm_hours_per_station_month"]),
            f"{selection_year}::budget_quantile_selection",
            int(warning["minimum_consecutive_alarm_bins"]),
            int(warning["alarm_merge_gap_minutes"]),
            int(warning["maximum_silence_before_event_minutes"]),
        )
        hit_rate = float(metric["event_hit_rate"])
        mean_lead = float(metric["mean_effective_lead_hours"])
        utility = hit_rate * mean_lead if np.isfinite(hit_rate) and np.isfinite(mean_lead) else float("nan")
        monthly_alarm_hours = (
            frame.groupby("station_month")[alarm_column].sum().astype(np.int64) * step_minutes / 60.0
        )
        rows.append(
            {
                "selection_year": selection_year,
                "selection_model": "local_weather_hazard",
                "candidate_quantile": quantile,
                "false_alarm_hours_per_station_month": metric["false_alarm_hours_per_station_month"],
                "alarm_hours_per_station_month": metric["alarm_hours_per_station_month"],
                "maximum_monthly_alarm_hours": float(monthly_alarm_hours.max()),
                "months_above_budget": int(
                    (
                        monthly_alarm_hours
                        > float(warning["false_alarm_hours_per_station_month"]) + 1e-9
                    ).sum()
                ),
                "evaluable_events": metric["evaluable_events"],
                "hit_events": metric["hit_events"],
                "event_hit_rate": hit_rate,
                "mean_effective_lead_hours": mean_lead,
                "event_weighted_lead_utility_hours": utility,
            }
        )

    selection = pd.DataFrame(rows).sort_values("candidate_quantile").reset_index(drop=True)
    finite = selection["event_weighted_lead_utility_hours"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        selected_quantile = float(selection["candidate_quantile"].max())
    else:
        best_utility = float(finite.max())
        tied = selection.loc[
            selection["event_weighted_lead_utility_hours"].ge(best_utility - utility_tie_hours)
        ]
        selected_quantile = float(tied["candidate_quantile"].max())
    selection["selected"] = selection["candidate_quantile"].eq(selected_quantile).astype(int)
    print(
        f"Selected causal-budget quantile {selected_quantile:.3f} on {selection_year} "
        f"using the local_weather_hazard event-weighted lead utility",
        flush=True,
    )
    return selected_quantile, selection


def load_budget_frame(paths: list[Path], year: int) -> pd.DataFrame:
    eta_columns = list(STABILITY_ETA_COLUMNS.values())
    columns = [
        "station_code",
        "station_name",
        "issue_time",
        "onset_within_6h",
        "hard_negative_6h",
        *[f"{column}__budget_alarm" for column in eta_columns],
    ]
    frames = []
    for path in paths:
        if path.parent.name != str(year):
            continue
        frames.append(
            read_artifact_csv(
                path,
                columns=columns,
                encoding="utf-8-sig",
                low_memory=False,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], errors="coerce")
    frame["station_month"] = frame["station_code"].astype(str) + "|" + frame["issue_time"].dt.to_period("M").astype(str)
    return frame


def evaluate_causal_budget_year(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    year: int,
    warning: dict[str, Any],
    step_minutes: int,
    bootstrap: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_frame = events.copy()
    event_frame["onset_time"] = pd.to_datetime(event_frame["onset_time"], errors="coerce")
    event_frame["valid_target_event"] = pd.to_numeric(event_frame["valid_target_event"], errors="coerce").fillna(0).astype(int)
    event_frame = event_frame.loc[event_frame["onset_time"].dt.year.eq(year)]
    horizon = int(warning["horizon_hours"])
    budget = float(warning["false_alarm_hours_per_station_month"])
    rows: list[dict[str, Any]] = []
    records: list[pd.DataFrame] = []
    alarm_columns: dict[str, str] = {}
    for model, eta_column in STABILITY_ETA_COLUMNS.items():
        alarm_column = f"{eta_column}__budget_alarm"
        alarm_columns[model] = alarm_column
        row = evaluate_warning_model(
            frame,
            event_frame,
            alarm_column,
            0.5,
            horizon,
            step_minutes,
            budget,
            f"{year}::causal_budget",
            int(warning["minimum_consecutive_alarm_bins"]),
            int(warning["alarm_merge_gap_minutes"]),
            int(warning["maximum_silence_before_event_minutes"]),
        )
        row["year"] = year
        row["model_name"] = model
        rows.append(row)
        model_records = event_alarm_records(frame, event_frame, alarm_column, 0.5, horizon, warning)
        model_records["model_name"] = model
        model_records["year"] = year
        records.append(model_records)
    record_frame = pd.concat(records, ignore_index=True)
    comparisons: list[pd.DataFrame] = []
    for comparison_index, model_b in enumerate(["stable_graph_ablation", "unfiltered_graph", "local_weather_hazard"]):
        comparison = cluster_bootstrap_event_comparison(
            record_frame,
            alarm_columns["stable_graph"],
            alarm_columns[model_b],
            int(bootstrap["samples"]),
            int(bootstrap["seed"]) + year * 10 + comparison_index,
        )
        comparison["year"] = year
        comparison["model_a_name"] = "stable_graph"
        comparison["model_b_name"] = model_b
        comparisons.append(comparison)
    return pd.DataFrame(rows), record_frame, pd.concat(comparisons, ignore_index=True)


def causal_budget_guarantee_audit(
    frames: dict[int, pd.DataFrame],
    step_minutes: int,
    budget_hours: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, frame in frames.items():
        for model, eta_column in STABILITY_ETA_COLUMNS.items():
            alarm_column = f"{eta_column}__budget_alarm"
            monthly_hours = (
                frame.groupby("station_month")[alarm_column].sum().astype(np.int64) * step_minutes / 60.0
            )
            rows.append(
                {
                    "year": year,
                    "model": model,
                    "station_months": int(monthly_hours.size),
                    "mean_total_alarm_hours": float(monthly_hours.mean()),
                    "maximum_total_alarm_hours": float(monthly_hours.max()),
                    "months_above_budget": int((monthly_hours > budget_hours + 1e-9).sum()),
                    "budget_hours": budget_hours,
                }
            )
    return pd.DataFrame(rows)


def build_stability_report(
    config: dict[str, Any],
    training_summary: dict[str, Any],
    stability_edges: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    stable_model: StructuredCloglogHazard,
    probability_metrics: pd.DataFrame,
    budget_selection: pd.DataFrame,
    selected_budget_quantile: float,
    warning_metrics: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    budget_audit: pd.DataFrame,
) -> str:
    stable = stability_edges.loc[stability_edges["stable"].eq(1)]
    unique_edges = stable[["source_station_code", "target_station_code"]].drop_duplicates().shape[0]
    probability_2024 = probability_metrics.loc[
        probability_metrics["year"].eq(2024)
        & probability_metrics["cohort"].eq("all")
        & probability_metrics["label"].eq("onset_within_6h")
    ]
    warning_display = warning_metrics.loc[
        warning_metrics["year"].isin([2023, 2024])
    ]
    stable_2024 = paired_comparisons.loc[
        paired_comparisons["year"].eq(2024)
        & paired_comparisons["model_b_name"].eq("stable_graph_ablation")
        & paired_comparisons["metric"].eq("common_hit_lead_difference_hours")
    ]
    descriptive_early = bool(
        not stable_2024.empty
        and float(stable_2024.iloc[0]["estimate"]) >= 0.1
        and float(stable_2024.iloc[0]["ci95_low"]) > 0
    )
    guarantee_passed = bool(budget_audit["months_above_budget"].eq(0).all())
    top_edges = stable.head(25)
    lines = [
        "# RIG-Hazard稳定图与因果预算预警报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 方法定位",
        "",
        "- 图结构不再依据单次全样本拟合直接解释，而是对2022年站点×月份块反复子采样；只有重复出现的方向边-滞后项进入稳定图。",
        "- 稳定图保留原unfiltered graph的非负cloglog强度结构，不增加新的深度学习模块。",
        "- 因果预算控制器只使用当前及过去分数的滚动分位数，并按月逐步释放告警令牌。每站每月总告警时长不超过预算，因此误报时长也不可能超过预算。",
        f"- 控制分位数只用{config['budget_control']['selection_year']}年local weather hazard危害模型选择，最终冻结为{selected_budget_quantile:.3f}；所有模型共用该值，2024年不参与选择。",
        "",
        "## 模型体系与选用依据",
        "",
        "### 模型语义定义",
        "",
        "以下模型共享同一复发起冰事件口径，但使用的信息逐步增加；名称直接表达模型职责，不再使用顺序编号。",
        "",
        "| 标识 | 完整名称 | 数学或算法类型 | 使用信息 | 在实验中的角色 |",
        "| --- | --- | --- | --- | --- |",
        "| `cold_humid_rule` / `strict_condensation_rule` | 冷湿规则与严格凝结规则 | 非学习型物理经验规则 | 本站温度、湿度和凝结条件 | 最低基线；本报告不重复计算 |",
        "| `local_weather_hazard` | 本站复发起冰离散时间危害率模型 | cloglog广义线性危害率模型 | 本站当前及历史气象特征 | 核心统计基线；用于选择统一预算分位数 |",
        "| `hierarchical_barrier` | 城市-站点层次屏障危害率模型 | 带层次正则的离散时间危害率模型 | 局地危害率加城市和站点风险偏移 | 校正站点异质性 |",
        "| `unfiltered_graph` | 非负稀疏滞后站点图危害率模型 | 层次屏障加非负图系数和L1稀疏约束 | 本站、层次屏障及邻站30/60/120/180分钟滞后信号 | 检验邻站是否提供更早的预测信息 |",
        "| `stable_graph` | 块稳定性筛选的稀疏滞后站点图危害率模型 | 对未筛选图做站点×月份块重采样、频率筛选和受限重拟合 | 仅保留重复出现的边-滞后项 | 本报告评估的稳定空间模型 |",
        f"| 预算控制器 | 因果滚动分位数月预算控制器 | 在线决策策略，不是预测模型 | 当前及过去{config['budget_control']['trailing_history_days']}天模型分数 | 将任一模型分数转换为逐站逐月不超过{config['warning']['false_alarm_hours_per_station_month']}小时的告警 |",
        "",
        "普通单步Logit和1/3/6小时直接Logit是与`local_weather_hazard`同口径的分类对照；其作用是检验离散时间hazard是否确实优于普通分类器。",
        "",
        "### 核心数学结构",
        "",
        "站点 `i` 在10分钟时刻 `t` 的条件起冰危害率定义为：",
        "",
        "```text",
        "h(i,t) = P(站点i在时刻t起冰 | t之前仍处于无冰风险集, 截至t的历史信息)",
        "```",
        "",
        "各层线性预测量为：",
        "",
        "```text",
        "local_weather_hazard: log[-log(1-h(i,t))] = alpha + beta^T x(i,t)",
        "hierarchical_barrier: eta_barrier(i,t) = eta_local(i,t) + u_city(i) + v_station(i)",
        "unfiltered_graph: eta_graph(i,t) = eta_barrier(i,t) + sum[j,l] w(j->i,l) s(j,t-l)",
        "    其中 w(j->i,l) >= 0，并对图系数施加L1稀疏约束",
        f"stable_graph: 仅保留“重采样选择频率 >= {config['stability']['selection_frequency']:.2f}且未筛选图系数 > {config['stability']['coefficient_threshold']}”的图项，再在完整2022训练集上重拟合",
        "```",
        "",
        "每个相互分离的起冰过程记为一次事件；事件结束并经过60分钟冷却后，站点重新进入风险集，因此同一站点可以贡献多次复发起冰事件。1/3/6小时风险由当前发报时刻的危害率在冻结协变量假设下累积，不读取未来气象观测。",
        "",
        "### 为什么采用这一模型路线",
        "",
        "| 任务特征 | 对应选择 | 原因 |",
        "| --- | --- | --- |",
        "| 目标是起冰开始而不是覆冰厚度 | 离散时间hazard | 直接建模仍处于无冰状态时的条件起始概率 |",
        "| 同一站点可多次起冰 | 复发事件风险集 | 每次独立事件结束和冷却后重新进入风险集 |",
        "| 一步正例仅447个且类别极不平衡 | 正则化统计模型、逆概率负样本权重 | 比高容量深度网络更适合当前正例规模，也便于校准与解释 |",
        "| 31个站点的事件数量差异明显 | hierarchical barrier层次城市-站点屏障 | 跨站共享信息，同时允许站点基线风险不同 |",
        "| 需要检验邻站是否更早出现信号 | unfiltered graph稀疏滞后图 | 用方向边和滞后显式表达预测性空间关联 |",
        "| 单次图拟合可能不稳定 | stable graph块稳定性选择 | 只保留在不同站点月份子样本中重复出现的图项 |",
        "| 业务要求固定误报时长 | 因果月预算控制器 | 使用过去分数并施加逐站逐月硬上限，避免未来信息和季节漂移造成预算超额 |",
        "",
        "### 本报告字段映射",
        "",
        "| 报告字段 | 对应含义 |",
        "| --- | --- |",
        "| `local_weather_hazard` | local weather hazard本站复发起冰离散时间危害率模型 |",
        "| `unfiltered_graph` | 上一阶段冻结的原始unfiltered graph完整图模型 |",
        "| `unfiltered_graph_ablation` | 保留已训练unfiltered graph的局地与层次参数，但将全部图贡献置零；不是重新训练的hierarchical barrier |",
        "| `stable_graph` | 经过块稳定性筛选并重拟合的stable graph完整模型 |",
        "| `stable_graph_ablation` | 保留已训练stable graph的局地与层次参数，但将全部稳定图贡献置零 |",
        "",
        "### 当前模型选择结论",
        "",
        "- local weather hazard是当前最稳妥的核心预测模型，因为它符合复发事件机制，概率指标不差于图模型，并且复杂度最低。",
        "- stable graph用于检验稳定邻站信息及图结构稳健性；它是空间扩展候选，而不是已经证明优于local weather hazard的最终模型。",
        "- rule baseline、普通Logit、hierarchical barrier、unfiltered graph和图消融共同构成必要的基线与消融链，不能省略其比较角色。",
        "- 图边表示控制本站历史后的预测性滞后关联，不等同于物理传播因果关系。",
        "- 当前可确认的组合是“local weather hazard复发起冰hazard + 凝结暴露hard negatives + 因果月预算控制”；邻站图显著延长提前量仍未得到验证。",
        "",
        "## 稳定性设置",
        "",
        f"- 重采样次数：{config['stability']['replicates']}；每站抽取月份比例：{config['stability']['block_fraction']}；稳定频率阈值：{config['stability']['selection_frequency']}。",
        f"- 系数激活阈值：{config['stability']['coefficient_threshold']}；原图L1保持冻结，不根据2024重新选择。",
        f"- 原候选图项：{training_summary['graph_feature_count']}；稳定图项：{stable.shape[0]}；稳定唯一有向边：{unique_edges}。",
        "",
        "## 训练样本",
        "",
        "```json",
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 重采样收敛",
        "",
        markdown_table(
            [
                {
                    "replicates": int(replicate_summary.shape[0]),
                    "converged": int(replicate_summary["converged"].sum()),
                    "mean_active_edge_lags": float(replicate_summary["active_edge_lags"].mean()),
                    "min_positive_rows": int(replicate_summary["represented_positive_rows"].min()),
                    "max_positive_rows": int(replicate_summary["represented_positive_rows"].max()),
                    "stable_refit_converged": stable_model.converged,
                    "stable_refit_iterations": stable_model.iterations,
                }
            ],
            [
                "replicates",
                "converged",
                "mean_active_edge_lags",
                "min_positive_rows",
                "max_positive_rows",
                "stable_refit_converged",
                "stable_refit_iterations",
            ],
        ),
        "",
        "## 主要稳定边",
        "",
        markdown_table(
            top_edges.to_dict("records"),
            [
                "source_station_code",
                "target_station_code",
                "lag_minutes",
                "distance_km",
                "selection_frequency",
                "original_coefficient",
                "coefficient_median",
            ],
        )
        if not top_edges.empty
        else "没有边达到预先设定的稳定频率阈值。",
        "",
        "## 因果预算参数选择（开发集）",
        "",
        markdown_table(
            budget_selection.to_dict("records"),
            [
                "selection_year",
                "selection_model",
                "candidate_quantile",
                "selected",
                "false_alarm_hours_per_station_month",
                "alarm_hours_per_station_month",
                "maximum_monthly_alarm_hours",
                "event_hit_rate",
                "mean_effective_lead_hours",
                "event_weighted_lead_utility_hours",
            ],
        ),
        "",
        "## 2024概率指标（描述性，6小时）",
        "",
        markdown_table(
            probability_2024.to_dict("records"),
            ["model", "positives", "pr_auc", "brier_score", "brier_skill", "log_loss", "ece"],
        ),
        "",
        "## 因果预算告警",
        "",
        markdown_table(
            warning_display.to_dict("records"),
            [
                "year",
                "model_name",
                "false_alarm_hours_per_station_month",
                "alarm_hours_per_station_month",
                "event_hit_rate",
                "mean_effective_lead_hours",
                "median_effective_lead_hours",
            ],
        ),
        "",
        "## 月度预算硬约束审计",
        "",
        markdown_table(
            budget_audit.to_dict("records"),
            ["year", "model", "station_months", "mean_total_alarm_hours", "maximum_total_alarm_hours", "months_above_budget"],
        ),
        "",
        "## 事件级配对检验",
        "",
        markdown_table(
            paired_comparisons.to_dict("records"),
            [
                "year",
                "model_a_name",
                "model_b_name",
                "metric",
                "estimate",
                "ci95_low",
                "ci95_high",
                "evaluable_events",
                "stations",
            ],
        ),
        "",
        "## 当前判断",
        "",
        f"1. 稳定性选择将{training_summary['graph_feature_count']}个候选图项收缩为{stable.shape[0]}个，避免直接把单次拟合的全部非零系数解释为空间传播。",
        (
            "2. 因果预算控制在所有站点月份都满足硬上界，季节漂移不再造成预算超额。"
            if guarantee_passed
            else "2. 存在站点月份超过硬预算，控制器实现需要修正。"
        ),
        (
            "3. 2024描述性结果中，稳定图相对同参数图消融获得至少0.1小时且聚类置信区间下界大于0的共同命中提前量。"
            if descriptive_early
            else "3. 2024描述性结果仍未达到至少0.1小时且聚类置信区间下界大于0的共同命中提前量标准。"
        ),
        "4. 2024已被前一阶段查看，本报告不把它重新包装成未触碰测试。可确认的贡献是稳定性诊断和预算保证；泛化提前量仍需新年份或外部区域验证。",
        "",
        f"完整产物位于：`{resolve_project_path(config['output_root'])}`",
    ]
    return "\n".join(lines) + "\n"


def run_stability_experiment(config: dict[str, Any], config_path: Path, overwrite: bool = False) -> Path:
    graph_root = resolve_project_path(config["graph_experiment_root"])
    output_root = resolve_project_path(config["output_root"])
    prepare_output_root(output_root, overwrite)
    write_json(output_root / "resolved_experiment_config.json", config)

    graph_config = json.loads((graph_root / "resolved_experiment_config.json").read_text(encoding="utf-8"))
    preprocessed_root = resolve_project_path(graph_config["preprocessed_root"])
    baseline_root = resolve_project_path(graph_config["baseline_root"])
    preprocessed_config = json.loads((preprocessed_root / "resolved_config.json").read_text(encoding="utf-8"))
    horizons = [int(value) for value in preprocessed_config["risk_set"]["forecast_horizons_hours"]]
    step_minutes = int(preprocessed_config["time_step_minutes"])
    all_years = sorted(
        {
            int(year)
            for split in preprocessed_config["splits"]["development"].values()
            if isinstance(split, list)
            for year in split
        }
    )
    compression_level = int(config.get("prediction_compression_level", 1))

    graph_schema_value = json.loads((graph_root / "graph_schema.json").read_text(encoding="utf-8"))
    seen_stations = set(str(value) for value in graph_schema_value["seen_stations"])
    station_catalog = pd.read_csv(preprocessed_root / "station_catalog.csv", encoding="utf-8-sig")
    station_catalog["seen_in_development"] = station_catalog["station_code"].astype(str).isin(seen_stations).astype(int)
    candidate_edges = pd.read_csv(preprocessed_root / "spatial_candidate_edges.csv", encoding="utf-8-sig")
    schema = build_graph_schema(
        station_catalog,
        candidate_edges,
        [int(value) for value in graph_config["graph"]["lags_minutes"]],
        int(graph_config["graph"]["candidate_neighbors"]),
        float(graph_config["graph"]["max_distance_km"]),
    )

    original_bundle = json.loads((graph_root / "model_bundle.json").read_text(encoding="utf-8"))
    original_model = StructuredCloglogHazard.from_dict(original_bundle["selected_graph_model"])
    if schema.feature_names != original_model.feature_names:
        raise RuntimeError("Reconstructed graph schema does not match the frozen unfiltered graph model")
    graph_l1 = float(original_bundle["selected_graph_l1"])
    bundle = load_local_hazard_bundle(baseline_root)
    signal_cache_root = graph_root / "source_signals"

    def station_code(path: Path) -> str:
        return path.name.split("_", 1)[0]

    train_paths = [
        path
        for path in timeline_files(preprocessed_root, [2022])
        if station_code(path) in seen_stations
    ]
    include_unseen = bool(config.get("include_unseen_test_stations", True))
    allowed_stations = set(station_catalog["station_code"].astype(str)) if include_unseen else seen_stations
    prediction_source_paths = [
        path
        for path in timeline_files(preprocessed_root, all_years)
        if station_code(path) in allowed_stations and (signal_cache_root / str(path.parent.name) / f"{station_code(path)}.csv.gz").exists()
    ]

    matrix, y, sample_weight, training_summary, metadata = build_structured_training_sample(
        train_paths,
        schema,
        signal_cache_root,
        bundle,
        split_column="split_development",
        split_name="train",
        max_horizon=max(horizons),
        sampling=graph_config["sampling"],
        step_minutes=step_minutes,
        rolling_window_minutes=int(graph_config["source_signal"]["rolling_window_minutes"]),
        return_metadata=True,
    )
    write_json(output_root / "training_sample_summary.json", training_summary)
    stability_edges, replicate_summary, stable_mask = run_block_stability_selection(
        matrix,
        y,
        sample_weight,
        metadata,
        schema,
        original_model,
        graph_l1,
        graph_config,
        config["stability"],
    )
    stability_edges.to_csv(output_root / "graph_edge_stability.csv", index=False, encoding="utf-8-sig")
    replicate_summary.to_csv(output_root / "stability_replicates.csv", index=False, encoding="utf-8-sig")
    stable_model = fit_stable_graph_model(
        matrix,
        y,
        sample_weight,
        schema,
        original_model,
        stable_mask,
        graph_l1,
        graph_config,
        config["stability"],
    )
    barrier_effect_table(schema, stable_model).to_csv(
        output_root / "stable_hierarchical_barriers.csv", index=False, encoding="utf-8-sig"
    )

    prediction_paths = generate_stability_predictions(
        prediction_source_paths,
        output_root / "predictions",
        schema,
        signal_cache_root,
        bundle,
        original_model,
        stable_model,
        horizons,
        step_minutes,
        int(graph_config["source_signal"]["rolling_window_minutes"]),
        compression_level,
    )
    calibration_year = int(config["calibration_year"])
    calibrators = fit_stability_calibrators(prediction_paths, calibration_year)
    apply_stability_calibrators(prediction_paths, calibrators, horizons, step_minutes, compression_level)
    probability_parts = [evaluate_stability_probability(prediction_paths, horizons, year) for year in config["evaluation_years"]]
    probability_metrics = pd.concat(probability_parts, ignore_index=True)
    probability_metrics.to_csv(output_root / "probability_metrics.csv", index=False, encoding="utf-8-sig")

    budget_control = config["budget_control"]
    candidate_quantiles = [
        float(value)
        for value in budget_control.get(
            "candidate_quantile_grid",
            [budget_control.get("candidate_quantile", 0.98)],
        )
    ]
    base_budget_config = CausalBudgetConfig(
        step_minutes=step_minutes,
        monthly_budget_hours=float(config["warning"]["false_alarm_hours_per_station_month"]),
        trailing_history_days=int(budget_control["trailing_history_days"]),
        candidate_quantile=candidate_quantiles[0],
        minimum_history_rows=int(budget_control["minimum_history_rows"]),
        burst_allowance_hours=float(budget_control["burst_allowance_hours"]),
        minimum_candidate_run_bins=int(budget_control["minimum_candidate_run_bins"]),
        minimum_alarm_run_bins=int(budget_control["minimum_alarm_run_bins"]),
    )
    events = pd.read_csv(preprocessed_root / "events_recurrent.csv", encoding="utf-8-sig")
    selected_budget_quantile, budget_selection = select_causal_budget_quantile(
        prediction_paths,
        events,
        int(budget_control.get("selection_year", calibration_year)),
        candidate_quantiles,
        base_budget_config,
        config["warning"],
        step_minutes,
        float(budget_control.get("utility_tie_hours", 0.05)),
    )
    budget_selection.to_csv(
        output_root / "causal_budget_quantile_selection.csv", index=False, encoding="utf-8-sig"
    )
    budget_config = replace(base_budget_config, candidate_quantile=selected_budget_quantile)
    budget_paths = generate_causal_budget_predictions(
        prediction_paths,
        output_root / "budget_predictions",
        [int(year) for year in config["evaluation_years"]],
        budget_config,
        compression_level,
    )
    warning_parts: list[pd.DataFrame] = []
    record_parts: list[pd.DataFrame] = []
    comparison_parts: list[pd.DataFrame] = []
    budget_frames: dict[int, pd.DataFrame] = {}
    for year in [int(value) for value in config["evaluation_years"]]:
        budget_frame = load_budget_frame(budget_paths, year)
        budget_frames[year] = budget_frame
        warning_rows, records, comparisons = evaluate_causal_budget_year(
            budget_frame,
            events,
            year,
            config["warning"],
            step_minutes,
            config["bootstrap"],
        )
        warning_parts.append(warning_rows)
        record_parts.append(records)
        comparison_parts.append(comparisons)
    warning_metrics = pd.concat(warning_parts, ignore_index=True)
    event_records = pd.concat(record_parts, ignore_index=True)
    paired_comparisons = pd.concat(comparison_parts, ignore_index=True)
    warning_metrics.to_csv(output_root / "causal_budget_warning_metrics.csv", index=False, encoding="utf-8-sig")
    event_records.to_csv(output_root / "causal_budget_event_records.csv", index=False, encoding="utf-8-sig")
    paired_comparisons.to_csv(output_root / "causal_budget_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    budget_audit = causal_budget_guarantee_audit(
        budget_frames,
        step_minutes,
        float(config["warning"]["false_alarm_hours_per_station_month"]),
    )
    budget_audit.to_csv(output_root / "causal_budget_guarantee_audit.csv", index=False, encoding="utf-8-sig")

    model_bundle = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_family": "block_stability_selected_sparse_lag_hazard",
        "config_path": str(config_path),
        "frozen_original_graph_l1": graph_l1,
        "stable_edge_lag_count": int(stable_mask.sum()),
        "stable_unique_edge_count": int(
            stability_edges.loc[stability_edges["stable"].eq(1), ["source_station_code", "target_station_code"]]
            .drop_duplicates()
            .shape[0]
        ),
        "stable_model": stable_model.to_dict(),
        "calibrators": {name: value.to_dict() for name, value in calibrators.items()},
        "causal_budget_config": budget_config.__dict__,
        "causal_budget_selection_year": int(budget_control.get("selection_year", calibration_year)),
        "causal_budget_selection_model": "local_weather_hazard",
    }
    write_json(output_root / "model_bundle.json", model_bundle)
    report = build_stability_report(
        config,
        training_summary,
        stability_edges,
        replicate_summary,
        stable_model,
        probability_metrics,
        budget_selection,
        selected_budget_quantile,
        warning_metrics,
        paired_comparisons,
        budget_audit,
    )
    (output_root / "stability_budget_report.md").write_text(report, encoding="utf-8")
    print(f"Stability and causal-budget experiment complete: {output_root}", flush=True)
    return output_root

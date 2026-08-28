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
        "# RIG-Hazard Stable-Graph and Causal-Budget Warning Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Method scope",
        "",
        "- Graph structure is not interpreted from a single full-sample fit. Station-month blocks from 2022 are repeatedly subsampled, and only recurring directed edge-lag terms enter the stable graph.",
        "- The stable graph retains the nonnegative cloglog intensity structure of the original unfiltered graph and adds no deep-learning module.",
        "- The causal-budget controller uses rolling quantiles of current and past scores and releases alert tokens progressively each month. Total alert time cannot exceed the station-month budget, so false-alarm time cannot exceed it either.",
        f"- The control quantile is selected using only the {config['budget_control']['selection_year']} local weather hazard model and frozen at {selected_budget_quantile:.3f}; every model shares this value, and 2024 does not influence selection.",
        "",
        "## Model system and rationale",
        "",
        "### Semantic model definitions",
        "",
        "The following models share one recurrent-icing event contract and progressively add information. Names describe model roles directly rather than using sequence numbers.",
        "",
        "| Identifier | Full name | Mathematical or algorithmic type | Information used | Experimental role |",
        "| --- | --- | --- | --- | --- |",
        "| `cold_humid_rule` / `strict_condensation_rule` | Cold-humid and strict-condensation rules | Non-learning physical heuristics | Local temperature, humidity, and condensation conditions | Minimum baselines, not recomputed in this report |",
        "| `local_weather_hazard` | Local recurrent-icing discrete-time hazard | Cloglog generalized linear hazard model | Current and historical local weather | Core statistical baseline and source for the common budget quantile |",
        "| `hierarchical_barrier` | City-station hierarchical-barrier hazard | Hierarchically regularized discrete-time hazard | Local hazard plus city and station risk offsets | Adjustment for station heterogeneity |",
        "| `unfiltered_graph` | Nonnegative sparse-lag station-graph hazard | Hierarchical barrier plus nonnegative graph coefficients and L1 sparsity | Local and hierarchical terms plus neighbor lags at 30/60/120/180 minutes | Test whether neighbors provide earlier predictive information |",
        "| `stable_graph` | Block-stability-selected sparse-lag station-graph hazard | Station-month block resampling, frequency selection, and constrained refit | Repeatedly selected edge-lag terms only | Stable spatial model evaluated here |",
        f"| Budget controller | Causal rolling-quantile monthly-budget controller | Online decision policy, not a prediction model | Current and previous {config['budget_control']['trailing_history_days']} days of model scores | Converts any model score to alerts capped at {config['warning']['false_alarm_hours_per_station_month']} hours per station-month |",
        "",
        "Ordinary one-step logistic regression and direct 1-hour, 3-hour, and 6-hour logistic models are classifier controls under the same contract as `local_weather_hazard`; they test whether a discrete-time hazard is genuinely better than ordinary classification.",
        "",
        "### Core mathematical structure",
        "",
        "The conditional icing-onset hazard for station `i` at 10-minute time `t` is",
        "",
        "```text",
        "h(i,t) = P(station i starts icing at t | ice-free risk set before t, history through t)",
        "```",
        "",
        "The linear predictor at each layer is",
        "",
        "```text",
        "local_weather_hazard: log[-log(1-h(i,t))] = alpha + beta^T x(i,t)",
        "hierarchical_barrier: eta_barrier(i,t) = eta_local(i,t) + u_city(i) + v_station(i)",
        "unfiltered_graph: eta_graph(i,t) = eta_barrier(i,t) + sum[j,l] w(j->i,l) s(j,t-l)",
        "    where w(j->i,l) >= 0 and graph coefficients have an L1 sparsity penalty",
        f"stable_graph: retain terms with resampling frequency >= {config['stability']['selection_frequency']:.2f} and unfiltered-graph coefficient > {config['stability']['coefficient_threshold']}, then refit on all 2022 training data",
        "```",
        "",
        "Each separated icing episode is an event. A station re-enters the risk set after the event ends and a 60-minute cooldown, so one station may contribute multiple recurrent events. Risks at 1, 3, and 6 hours accumulate the issue-time hazard under frozen covariates and do not read future weather observations.",
        "",
        "### Modelling rationale",
        "",
        "| Task property | Design choice | Rationale |",
        "| --- | --- | --- |",
        "| Target is icing onset rather than thickness | Discrete-time hazard | Directly model conditional onset while the station remains ice-free |",
        "| A station may ice repeatedly | Recurrent-event risk set | Re-enter the risk set after each independent event and cooldown |",
        "| Only 447 one-step positives with extreme imbalance | Regularized statistical model and inverse-probability negative weights | Better matched to the positive count than a high-capacity network and easier to calibrate and interpret |",
        "| Event counts differ substantially across 31 stations | Hierarchical city-station barrier | Share information while allowing different baseline station risks |",
        "| Need to test whether neighbor signals arrive earlier | Unfiltered sparse-lag graph | Represent predictive spatial associations explicitly with directed edges and lags |",
        "| A single graph fit may be unstable | Stable-graph block selection | Retain only terms recurring across station-month subsamples |",
        "| Operations require fixed false-alarm time | Causal monthly-budget controller | Use past scores with a station-month hard cap to prevent future leakage and seasonal overspend |",
        "",
        "### Report field mapping",
        "",
        "| Report field | Meaning |",
        "| --- | --- |",
        "| `local_weather_hazard` | Local recurrent-icing discrete-time hazard |",
        "| `unfiltered_graph` | Full unfiltered graph frozen in the previous stage |",
        "| `unfiltered_graph_ablation` | Retains the fitted local and hierarchical parameters but zeros all graph contribution; it is not a retrained hierarchical barrier |",
        "| `stable_graph` | Full stable graph after block-stability selection and refitting |",
        "| `stable_graph_ablation` | Retains the fitted stable-graph local and hierarchical parameters but zeros all stable graph contribution |",
        "",
        "### Model-selection conclusion",
        "",
        "- Local weather hazard is the most defensible primary model because it matches the recurrent-event mechanism, performs no worse than graph models on probability metrics, and has the lowest complexity.",
        "- Stable graph tests robust neighbor information and graph structure. It is a candidate spatial extension, not a final model already shown to outperform local weather hazard.",
        "- Rule baselines, ordinary logistic models, hierarchical barrier, unfiltered graph, and graph ablations form the necessary baseline and ablation chain.",
        "- Graph edges are predictive lagged associations after controlling for local history, not evidence of physical propagation or causality.",
        "- The currently supported combination is local-weather recurrent-icing hazard, condensation-exposure hard negatives, and causal monthly-budget control. A material neighbor-graph lead-time gain remains unverified.",
        "",
        "## Stability settings",
        "",
        f"- Resamples: {config['stability']['replicates']}; station-month sampling fraction: {config['stability']['block_fraction']}; stability-frequency threshold: {config['stability']['selection_frequency']}.",
        f"- Coefficient activation threshold: {config['stability']['coefficient_threshold']}; the original graph L1 remains frozen and is not reselected on 2024.",
        f"- Original candidate graph terms: {training_summary['graph_feature_count']}; stable terms: {stable.shape[0]}; unique stable directed edges: {unique_edges}.",
        "",
        "## Training samples",
        "",
        "```json",
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Resampling convergence",
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
        "## Leading stable edges",
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
        else "No edge reaches the prespecified stability-frequency threshold.",
        "",
        "## Causal-budget parameter selection (development set)",
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
        "## 2024 probability metrics (descriptive, 6 hours)",
        "",
        markdown_table(
            probability_2024.to_dict("records"),
            ["model", "positives", "pr_auc", "brier_score", "brier_skill", "log_loss", "ece"],
        ),
        "",
        "## Causal-budget warnings",
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
        "## Monthly hard-budget audit",
        "",
        markdown_table(
            budget_audit.to_dict("records"),
            ["year", "model", "station_months", "mean_total_alarm_hours", "maximum_total_alarm_hours", "months_above_budget"],
        ),
        "",
        "## Paired event-level tests",
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
        "## Conclusions",
        "",
        f"1. Stability selection reduces {training_summary['graph_feature_count']} candidate graph terms to {stable.shape[0]}, preventing every nonzero coefficient from one fit from being interpreted as spatial propagation.",
        (
            "2. Causal-budget control satisfies the hard limit in every station-month, so seasonal drift no longer causes budget overspend."
            if guarantee_passed
            else "2. At least one station-month exceeds the hard budget; the controller implementation requires correction."
        ),
        (
            "3. In descriptive 2024 results, the stable graph improves common-hit lead time over its parameter-matched graph ablation by at least 0.1 hours with a positive cluster-confidence lower bound."
            if descriptive_early
            else "3. Descriptive 2024 results do not meet the criterion of at least 0.1 hours of common-hit lead-time improvement with a positive cluster-confidence lower bound."
        ),
        "4. The previous stage already inspected 2024, so this report does not repackage it as an untouched test. The confirmed contributions are stability diagnostics and the budget guarantee; generalized lead time still requires a new year or external region.",
        "",
        f"Complete artifacts: `{resolve_project_path(config['output_root'])}`",
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

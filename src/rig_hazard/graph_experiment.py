from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .artifact_compat import read_artifact_csv
from .baseline_experiment import (
    evaluate_warning_model,
    markdown_table,
    probability_metric_row,
    read_timeline,
    split_years,
    threshold_for_false_alarm_budget,
    timeline_files,
)
from .baseline_models import HazardRateCalibrator
from .config import resolve_project_path
from .graph_data import (
    GraphSchema,
    LocalHazardBundle,
    SignalBank,
    build_graph_schema,
    build_structured_design,
    build_structured_training_sample,
    load_local_hazard_bundle,
    prepare_source_signals,
)
from .graph_models import StructuredCloglogHazard
from .preprocessing import prepare_output_root, write_json


MODEL_ETA_COLUMNS = {
    "local_weather_hazard": "local_weather_hazard_eta",
    "hierarchical_barrier": "hierarchical_barrier_eta",
    "sparse_lag_graph_ablation": "sparse_lag_graph_ablation_eta",
    "sparse_lag_graph": "sparse_lag_graph_eta",
}


@dataclass
class FittedGraphModels:
    barrier: StructuredCloglogHazard
    graph_candidates: dict[float, StructuredCloglogHazard]


def model_penalties_and_bounds(
    schema: GraphSchema,
    hierarchy: dict[str, Any],
    graph_l1: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float | None, float | None]]]:
    l2 = np.zeros(len(schema.feature_names), dtype=np.float64)
    l1 = np.zeros(len(schema.feature_names), dtype=np.float64)
    bounds: list[tuple[float | None, float | None]] = []
    effect_bound = float(hierarchy["maximum_effect"])
    graph_bound = float(hierarchy["maximum_graph_coefficient"])
    for index, name in enumerate(schema.feature_names):
        if name == "local::eta":
            l2[index] = float(hierarchy["local_slope_l2"])
            bounds.append((0.0, float(hierarchy["maximum_local_slope"])))
        elif name.startswith("barrier::city::"):
            l2[index] = float(hierarchy["city_effect_l2"])
            bounds.append((-effect_bound, effect_bound))
        elif name.startswith("barrier::station::"):
            l2[index] = float(hierarchy["station_effect_l2"])
            bounds.append((-effect_bound, effect_bound))
        elif name.startswith("graph::"):
            l2[index] = float(hierarchy["graph_l2"])
            l1[index] = float(graph_l1)
            bounds.append((0.0, graph_bound))
        else:
            raise ValueError(f"Unknown structured feature: {name}")
    return l2, l1, bounds


def fit_structured_models(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    sample_weight: np.ndarray,
    schema: GraphSchema,
    config: dict[str, Any],
) -> FittedGraphModels:
    hierarchy = config["hierarchy"]
    optimization = config["optimization"]
    base_names = schema.feature_names[: schema.base_feature_count]
    base_l2, base_l1, base_bounds = model_penalties_and_bounds(schema, hierarchy, 0.0)
    barrier = StructuredCloglogHazard(
        feature_names=base_names,
        l2_penalties=base_l2[: schema.base_feature_count],
        l1_penalties=base_l1[: schema.base_feature_count],
        coefficient_bounds=base_bounds[: schema.base_feature_count],
        max_iter=int(optimization["max_iter"]),
        tolerance=float(optimization["tolerance"]),
    )
    initial = np.zeros(schema.base_feature_count, dtype=np.float64)
    initial[0] = 1.0
    print(f"Fitting hierarchical barrier on {matrix.shape[0]:,} rows", flush=True)
    barrier.fit(
        matrix[:, : schema.base_feature_count],
        y,
        sample_weight=sample_weight,
        initial_intercept=0.0,
        initial_coefficients=initial,
    )
    print(
        f"  hierarchical barrier converged={barrier.converged} iterations={barrier.iterations} objective={barrier.objective:.8f}",
        flush=True,
    )

    candidates: dict[float, StructuredCloglogHazard] = {}
    for graph_l1 in sorted({float(value) for value in config["graph"]["l1_grid"]}, reverse=True):
        l2, l1, bounds = model_penalties_and_bounds(schema, hierarchy, graph_l1)
        model = StructuredCloglogHazard(
            feature_names=schema.feature_names,
            l2_penalties=l2,
            l1_penalties=l1,
            coefficient_bounds=bounds,
            max_iter=int(optimization["max_iter"]),
            tolerance=float(optimization["tolerance"]),
        )
        initial = np.zeros(len(schema.feature_names), dtype=np.float64)
        initial[: schema.base_feature_count] = barrier.coefficients
        print(f"Fitting sparse-lag graph candidate with L1={graph_l1:.3g}", flush=True)
        model.fit(
            matrix,
            y,
            sample_weight=sample_weight,
            initial_intercept=barrier.intercept,
            initial_coefficients=initial,
        )
        active = int((model.coefficients[schema.base_feature_count :] > float(config["graph"]["nonzero_tolerance"])).sum())
        print(
            f"  unfiltered graph converged={model.converged} iterations={model.iterations} objective={model.objective:.8f} active={active}",
            flush=True,
        )
        candidates[graph_l1] = model
    return FittedGraphModels(barrier=barrier, graph_candidates=candidates)


def risk_frame(path: Path, bundle: LocalHazardBundle, horizons: list[int]) -> pd.DataFrame:
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


def evaluate_regularization_candidates(
    paths: list[Path],
    schema: GraphSchema,
    signal_cache_root: Path,
    bundle: LocalHazardBundle,
    fitted: FittedGraphModels,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
    step_minutes: int,
    rolling_window_minutes: int,
    nonzero_tolerance: float,
    events: pd.DataFrame,
    warning: dict[str, Any],
) -> pd.DataFrame:
    y_parts: list[np.ndarray] = []
    eta_parts: dict[float, list[np.ndarray]] = {value: [] for value in fitted.graph_candidates}
    barrier_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    banks: dict[int, SignalBank] = {}
    for index, path in enumerate(paths, start=1):
        frame = risk_frame(path, bundle, [6])
        if frame.empty:
            continue
        year = int(frame["issue_time"].dt.year.mode().iloc[0])
        bank = banks.setdefault(year, SignalBank(signal_cache_root, year, step_minutes, rolling_window_minutes))
        matrix, _ = build_structured_design(frame, schema, bank, bundle)
        mask = frame["issue_time"].ge(selection_start) & frame["issue_time"].lt(selection_end)
        if not mask.any():
            continue
        selected_matrix = matrix[mask.to_numpy()]
        y_parts.append(frame.loc[mask, "hazard_label"].to_numpy(dtype=np.int8))
        metadata_parts.append(
            frame.loc[
                mask,
                ["station_code", "station_name", "issue_time", "onset_within_6h", "hard_negative_6h"],
            ].copy()
        )
        barrier_parts.append(fitted.barrier.decision_function(selected_matrix[:, : schema.base_feature_count]))
        for value, model in fitted.graph_candidates.items():
            eta_parts[value].append(model.decision_function(selected_matrix))
        if index % 10 == 0 or index == len(paths):
            print(f"Scored regularization selection data from {index}/{len(paths)} files", flush=True)

    y = np.concatenate(y_parts)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    metadata["station_month"] = metadata["station_code"].astype(str) + "|" + metadata["issue_time"].dt.to_period("M").astype(str)
    selection_events = events.copy()
    selection_events["onset_time"] = pd.to_datetime(selection_events["onset_time"], errors="coerce")
    selection_events["valid_target_event"] = pd.to_numeric(
        selection_events["valid_target_event"], errors="coerce"
    ).fillna(0).astype(int)
    selection_events = selection_events.loc[
        selection_events["onset_time"].ge(selection_start) & selection_events["onset_time"].lt(selection_end)
    ]
    horizon = int(warning["horizon_hours"])
    budget = float(warning["false_alarm_hours_per_station_month"])

    def warning_values(probability: np.ndarray, name: str) -> dict[str, Any]:
        score_column = "_selection_probability"
        evaluation_frame = metadata.assign(**{score_column: probability})
        threshold = threshold_for_false_alarm_budget(
            evaluation_frame, score_column, horizon, budget, step_minutes
        )
        values = evaluate_warning_model(
            evaluation_frame,
            selection_events,
            score_column,
            threshold,
            horizon,
            step_minutes,
            budget,
            "selection::matched_budget",
            int(warning["minimum_consecutive_alarm_bins"]),
            int(warning["alarm_merge_gap_minutes"]),
            int(warning["maximum_silence_before_event_minutes"]),
        )
        return {
            "selection_threshold": threshold,
            "false_alarm_hours_per_station_month": values["false_alarm_hours_per_station_month"],
            "event_hit_rate": values["event_hit_rate"],
            "mean_effective_lead_hours": values["mean_effective_lead_hours"],
            "lead_utility_hours": values["event_hit_rate"] * values["mean_effective_lead_hours"],
            "evaluable_events": values["evaluable_events"],
        }

    rows: list[dict[str, Any]] = []
    barrier_eta = np.concatenate(barrier_parts)
    barrier_probability = -np.expm1(-np.exp(np.clip(barrier_eta, -30.0, 15.0)))
    horizon_steps = horizon * 60 / step_minutes
    barrier_probability_6h = -np.expm1(-horizon_steps * np.exp(np.clip(barrier_eta, -30.0, 15.0)))
    barrier_row = probability_metric_row("hierarchical_barrier", "hazard_label", y, barrier_probability)
    barrier_row.update({"graph_l1": float("nan"), "active_edge_lags": 0, "unique_directed_edges": 0})
    barrier_row.update(warning_values(barrier_probability_6h, "hierarchical_barrier"))
    rows.append(barrier_row)
    for value, model in fitted.graph_candidates.items():
        eta = np.concatenate(eta_parts[value])
        probability = -np.expm1(-np.exp(np.clip(eta, -30.0, 15.0)))
        probability_6h = -np.expm1(-horizon_steps * np.exp(np.clip(eta, -30.0, 15.0)))
        row = probability_metric_row("sparse_lag_graph", "hazard_label", y, probability)
        graph_coefficients = model.coefficients[schema.base_feature_count :]
        active = graph_coefficients > nonzero_tolerance
        active_metadata = schema.edge_features.loc[active]
        row.update(
            {
                "graph_l1": value,
                "active_edge_lags": int(active.sum()),
                "unique_directed_edges": int(
                    active_metadata[["source_station_code", "target_station_code"]].drop_duplicates().shape[0]
                ),
                "converged": int(model.converged),
                "iterations": model.iterations,
                "training_objective": model.objective,
            }
        )
        row.update(warning_values(probability_6h, f"sparse_lag_graph::{value}"))
        rows.append(row)
    return pd.DataFrame(rows)


def select_graph_model(
    selection: pd.DataFrame,
    lead_utility_tie_hours: float,
    maximum_pr_auc_relative_degradation: float,
) -> float:
    candidates = selection.loc[selection["model"].eq("sparse_lag_graph") & selection["converged"].eq(1)].copy()
    if candidates.empty:
        raise RuntimeError("No converged unfiltered graph graph candidate")
    barrier_pr = float(selection.loc[selection["model"].eq("hierarchical_barrier"), "pr_auc"].iloc[0])
    minimum_pr = barrier_pr * (1.0 - maximum_pr_auc_relative_degradation)
    eligible = candidates.loc[candidates["pr_auc"].ge(minimum_pr)].copy()
    if eligible.empty:
        eligible = candidates.loc[candidates["pr_auc"].eq(candidates["pr_auc"].max())].copy()
    best_utility = float(eligible["lead_utility_hours"].max())
    eligible = eligible.loc[eligible["lead_utility_hours"].ge(best_utility - lead_utility_tie_hours)].copy()
    eligible = eligible.sort_values(["active_edge_lags", "graph_l1"], ascending=[True, False])
    return float(eligible.iloc[0]["graph_l1"])


def generate_raw_predictions(
    paths: list[Path],
    destination_root: Path,
    schema: GraphSchema,
    signal_cache_root: Path,
    bundle: LocalHazardBundle,
    barrier: StructuredCloglogHazard,
    graph: StructuredCloglogHazard,
    horizons: list[int],
    step_minutes: int,
    rolling_window_minutes: int,
    compression_level: int,
) -> list[Path]:
    outputs: list[Path] = []
    banks: dict[int, SignalBank] = {}
    for index, path in enumerate(paths, start=1):
        frame = risk_frame(path, bundle, horizons)
        if frame.empty:
            continue
        year = int(frame["issue_time"].dt.year.mode().iloc[0])
        bank = banks.setdefault(year, SignalBank(signal_cache_root, year, step_minutes, rolling_window_minutes))
        matrix, local_eta = build_structured_design(frame, schema, bank, bundle)
        base = matrix[:, : schema.base_feature_count]
        graph_part = matrix[:, schema.base_feature_count :]
        hierarchical_barrier_eta = barrier.decision_function(base)
        sparse_lag_graph_eta = graph.decision_function(matrix)
        sparse_lag_graph_ablation_eta = graph.intercept + np.asarray(
            base @ graph.coefficients[: schema.base_feature_count]
        ).ravel()
        graph_contribution = np.asarray(graph_part @ graph.coefficients[schema.base_feature_count :]).ravel()
        barrier_contribution = np.asarray(base[:, 1:] @ graph.coefficients[1 : schema.base_feature_count]).ravel()

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
        output["hierarchical_barrier_eta"] = hierarchical_barrier_eta.astype(np.float32)
        output["sparse_lag_graph_ablation_eta"] = (
            sparse_lag_graph_ablation_eta.astype(np.float32)
        )
        output["sparse_lag_graph_eta"] = sparse_lag_graph_eta.astype(np.float32)
        output["hierarchical_barrier_contribution"] = barrier_contribution.astype(
            np.float32
        )
        output["sparse_lag_graph_contribution"] = graph_contribution.astype(np.float32)
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
            print(
                "Generated raw hierarchical/sparse-graph predictions for "
                f"{index}/{len(paths)} files",
                flush=True,
            )
    return outputs


def fit_graph_calibrators(
    paths: list[Path],
    calibration_start: pd.Timestamp,
    calibration_end: pd.Timestamp,
) -> dict[str, HazardRateCalibrator]:
    labels: list[np.ndarray] = []
    scores: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_ETA_COLUMNS}
    columns = ["issue_time", "hazard_label", *MODEL_ETA_COLUMNS.values()]
    for path in paths:
        frame = read_artifact_csv(
            path, columns=columns, encoding="utf-8-sig", low_memory=False
        )
        issue_time = pd.to_datetime(frame["issue_time"], errors="coerce")
        mask = issue_time.ge(calibration_start) & issue_time.lt(calibration_end)
        if not mask.any():
            continue
        labels.append(pd.to_numeric(frame.loc[mask, "hazard_label"], errors="coerce").fillna(0).to_numpy(dtype=np.int8))
        for name, column in MODEL_ETA_COLUMNS.items():
            scores[name].append(pd.to_numeric(frame.loc[mask, column], errors="coerce").to_numpy(dtype=np.float64))
    y = np.concatenate(labels)
    calibrators: dict[str, HazardRateCalibrator] = {}
    for name in MODEL_ETA_COLUMNS:
        calibrator = HazardRateCalibrator().fit(np.concatenate(scores[name]), y)
        calibrators[name] = calibrator
        print(
            f"Calibrated {name}: converged={calibrator.converged} shift={calibrator.log_rate_shift:.5f} slope={calibrator.slope:.5f}",
            flush=True,
        )
    return calibrators


def apply_graph_calibrators(
    paths: list[Path],
    calibrators: dict[str, HazardRateCalibrator],
    horizons: list[int],
    step_minutes: int,
    compression_level: int,
) -> None:
    for index, path in enumerate(paths, start=1):
        frame = read_artifact_csv(path, encoding="utf-8-sig", low_memory=False)
        for name, eta_column in MODEL_ETA_COLUMNS.items():
            eta = pd.to_numeric(frame[eta_column], errors="coerce").to_numpy(dtype=np.float64)
            frame[f"{name}_step"] = calibrators[name].predict(eta).astype(np.float32)
            for horizon in horizons:
                steps = horizon * 60 / step_minutes
                frame[f"{name}_{horizon}h"] = calibrators[name].predict(eta, steps=steps).astype(np.float32)
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
            compression={"method": "gzip", "compresslevel": compression_level},
        )
        if index % 10 == 0 or index == len(paths):
            print(f"Applied hierarchical/sparse-graph calibration to {index}/{len(paths)} files", flush=True)


def evaluate_graph_probability(
    paths: list[Path],
    horizons: list[int],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    model_names = list(MODEL_ETA_COLUMNS)
    columns = [
        "seen_in_development",
        "issue_time",
        "hazard_label",
        *[f"onset_within_{horizon}h" for horizon in horizons],
    ]
    for model in model_names:
        columns.append(f"{model}_step")
        columns.extend(f"{model}_{horizon}h" for horizon in horizons)
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(paths, start=1):
        frame = read_artifact_csv(
            path, columns=columns, encoding="utf-8-sig", low_memory=False
        )
        frame["issue_time"] = pd.to_datetime(frame["issue_time"], errors="coerce")
        if start is not None:
            frame = frame.loc[frame["issue_time"].ge(start)]
        if end is not None:
            frame = frame.loc[frame["issue_time"].lt(end)]
        if not frame.empty:
            frames.append(frame)
        if index % 10 == 0 or index == len(paths):
            print(f"Loaded graph probability data from {index}/{len(paths)} files", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    seen = pd.to_numeric(combined["seen_in_development"], errors="coerce").fillna(0).eq(1)
    cohorts = {"all": np.ones(combined.shape[0], dtype=bool), "seen": seen.to_numpy(), "unseen": (~seen).to_numpy()}
    rows: list[dict[str, Any]] = []
    for cohort, mask in cohorts.items():
        if not mask.any():
            continue
        subset = combined.loc[mask]
        for model in model_names:
            row = probability_metric_row(
                f"{model}_step",
                "hazard_label",
                subset["hazard_label"].to_numpy(dtype=np.int8),
                subset[f"{model}_step"].to_numpy(dtype=np.float64),
            )
            row["cohort"] = cohort
            rows.append(row)
            for horizon in horizons:
                label = f"onset_within_{horizon}h"
                row = probability_metric_row(
                    f"{model}_{horizon}h",
                    label,
                    subset[label].to_numpy(dtype=np.int8),
                    subset[f"{model}_{horizon}h"].to_numpy(dtype=np.float64),
                )
                row["cohort"] = cohort
                rows.append(row)
    return pd.DataFrame(rows)


def load_graph_warning_frame(paths: list[Path], horizon: int, model_columns: list[str]) -> pd.DataFrame:
    columns = [
        "station_code",
        "station_name",
        "seen_in_development",
        "issue_time",
        f"onset_within_{horizon}h",
        f"hard_negative_{horizon}h",
        *model_columns,
    ]
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(paths, start=1):
        frame = read_artifact_csv(
            path, columns=columns, encoding="utf-8-sig", low_memory=False
        )
        frames.append(frame)
        if index % 10 == 0 or index == len(paths):
            print(f"Loaded graph warning data from {index}/{len(paths)} files", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result["issue_time"] = pd.to_datetime(result["issue_time"], errors="coerce")
    result["station_month"] = result["station_code"].astype(str) + "|" + result["issue_time"].dt.to_period("M").astype(str)
    result["seen_in_development"] = pd.to_numeric(result["seen_in_development"], errors="coerce").fillna(0).astype(int)
    return result


def evaluate_graph_warning_points(
    validation_paths: list[Path],
    test_paths: list[Path],
    events: pd.DataFrame,
    calibration_start: pd.Timestamp,
    calibration_end: pd.Timestamp,
    test_years: list[int],
    warning: dict[str, Any],
    step_minutes: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float], dict[str, float], pd.DataFrame, pd.DataFrame]:
    horizon = int(warning["horizon_hours"])
    model_columns = [f"{name}_{horizon}h" for name in MODEL_ETA_COLUMNS]
    validation = load_graph_warning_frame(validation_paths, horizon, model_columns)
    validation = validation.loc[
        validation["issue_time"].ge(calibration_start) & validation["issue_time"].lt(calibration_end)
    ].reset_index(drop=True)
    test = load_graph_warning_frame(test_paths, horizon, model_columns)
    events = events.copy()
    events["onset_time"] = pd.to_datetime(events["onset_time"], errors="coerce")
    events["valid_target_event"] = pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).astype(int)
    validation_events = events.loc[
        events["onset_time"].ge(calibration_start) & events["onset_time"].lt(calibration_end)
    ]
    test_events = events.loc[events["onset_time"].dt.year.isin(test_years)]

    budget = float(warning["false_alarm_hours_per_station_month"])
    def evaluate(frame: pd.DataFrame, event_frame: pd.DataFrame, model: str, threshold: float, point: str) -> dict[str, Any]:
        return evaluate_warning_model(
            frame,
            event_frame,
            model,
            threshold,
            horizon,
            step_minutes,
            budget,
            point,
            int(warning["minimum_consecutive_alarm_bins"]),
            int(warning["alarm_merge_gap_minutes"]),
            int(warning["maximum_silence_before_event_minutes"]),
        )
    locked_thresholds: dict[str, float] = {}
    matched_thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for model in model_columns:
        locked = threshold_for_false_alarm_budget(validation, model, horizon, budget, step_minutes)
        matched = threshold_for_false_alarm_budget(test, model, horizon, budget, step_minutes)
        locked_thresholds[model] = locked
        matched_thresholds[model] = matched
        rows.append(evaluate(validation, validation_events, model, locked, "calibration::locked_budget"))
        rows.append(evaluate(test, test_events, model, locked, "test::locked_calibration_budget"))
        rows.append(evaluate(test, test_events, model, matched, "test::matched_budget_diagnostic"))

        seen_test = test.loc[test["seen_in_development"].eq(1)].reset_index(drop=True)
        seen_events = test_events.loc[test_events["station_code"].astype(str).isin(set(seen_test["station_code"].astype(str)))]
        seen_matched = threshold_for_false_alarm_budget(seen_test, model, horizon, budget, step_minutes)
        rows.append(evaluate(seen_test, seen_events, model, seen_matched, "test_seen::matched_budget_diagnostic"))
    threshold_payload = {
        "calibration_period": [str(calibration_start), str(calibration_end)],
        "locked_calibration_thresholds": locked_thresholds,
        "test_matched_thresholds_diagnostic_only": matched_thresholds,
        "false_alarm_hours_per_station_month": budget,
    }
    return pd.DataFrame(rows), threshold_payload, locked_thresholds, matched_thresholds, test, test_events


def event_alarm_records(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    model_column: str,
    threshold: float,
    horizon: int,
    warning: dict[str, Any],
) -> pd.DataFrame:
    score = pd.to_numeric(frame[model_column], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    station_frames: dict[str, pd.DataFrame] = {}
    for station_code, station_frame in frame.assign(_alarm=score >= threshold).groupby("station_code", sort=False):
        station_frames[str(station_code)] = station_frame.sort_values("issue_time")
    horizon_delta = pd.Timedelta(hours=horizon)
    rows: list[dict[str, Any]] = []
    for event in events.loc[events["valid_target_event"].eq(1)].itertuples(index=False):
        station_frame = station_frames.get(str(event.station_code))
        base = {
            "model": model_column,
            "event_id": str(event.event_id),
            "station_code": str(event.station_code),
            "onset_time": pd.Timestamp(event.onset_time),
            "evaluable": 0,
            "hit": 0,
            "effective_lead_hours": float("nan"),
            "lead_utility_hours": 0.0,
        }
        if station_frame is None:
            rows.append(base)
            continue
        onset = pd.Timestamp(event.onset_time)
        window = station_frame.loc[
            station_frame["issue_time"].ge(onset - horizon_delta) & station_frame["issue_time"].lt(onset)
        ]
        if window.empty:
            rows.append(base)
            continue
        base["evaluable"] = 1
        alarm_times = window.loc[window["_alarm"], "issue_time"].sort_values()
        if alarm_times.empty:
            rows.append(base)
            continue
        silence = (onset - alarm_times.max()).total_seconds() / 60.0
        if silence > int(warning["maximum_silence_before_event_minutes"]):
            rows.append(base)
            continue
        gaps = alarm_times.diff().dt.total_seconds().div(60).fillna(0).to_numpy()
        breaks = np.flatnonzero(gaps > int(warning["alarm_merge_gap_minutes"]))
        episode_start = int(breaks[-1]) if breaks.size else 0
        episode = alarm_times.iloc[episode_start:]
        if episode.shape[0] < int(warning["minimum_consecutive_alarm_bins"]):
            rows.append(base)
            continue
        lead = float((onset - episode.min()).total_seconds() / 3600.0)
        base.update({"hit": 1, "effective_lead_hours": lead, "lead_utility_hours": lead})
        rows.append(base)
    return pd.DataFrame(rows)


def cluster_bootstrap_event_comparison(
    records: pd.DataFrame,
    model_a: str,
    model_b: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    left = records.loc[records["model"].eq(model_a)].copy()
    right = records.loc[records["model"].eq(model_b)].copy()
    merged = left.merge(right, on=["event_id", "station_code"], suffixes=("_a", "_b"))
    merged = merged.loc[merged["evaluable_a"].eq(1) & merged["evaluable_b"].eq(1)].reset_index(drop=True)
    stations = merged["station_code"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)

    def statistics(frame: pd.DataFrame) -> dict[str, float]:
        common = frame["hit_a"].eq(1) & frame["hit_b"].eq(1)
        return {
            "hit_rate_difference": float((frame["hit_a"] - frame["hit_b"]).mean()),
            "lead_utility_difference_hours": float(
                (frame["lead_utility_hours_a"] - frame["lead_utility_hours_b"]).mean()
            ),
            "common_hit_lead_difference_hours": float(
                (frame.loc[common, "effective_lead_hours_a"] - frame.loc[common, "effective_lead_hours_b"]).mean()
            )
            if common.any()
            else float("nan"),
        }

    point = statistics(merged)
    draws: dict[str, list[float]] = {name: [] for name in point}
    if stations.size:
        grouped = {station: merged.loc[merged["station_code"].eq(station)] for station in stations}
        for _ in range(bootstrap_samples):
            sampled = rng.choice(stations, size=stations.size, replace=True)
            draw = pd.concat([grouped[station] for station in sampled], ignore_index=True)
            values = statistics(draw)
            for name, value in values.items():
                if np.isfinite(value):
                    draws[name].append(value)
    rows: list[dict[str, Any]] = []
    for metric, estimate in point.items():
        values = np.asarray(draws[metric], dtype=np.float64)
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "estimate": estimate,
                "ci95_low": float(np.quantile(values, 0.025)) if values.size else float("nan"),
                "ci95_high": float(np.quantile(values, 0.975)) if values.size else float("nan"),
                "evaluable_events": int(merged.shape[0]),
                "stations": int(stations.size),
                "bootstrap_samples": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def selected_edge_table(
    schema: GraphSchema,
    model: StructuredCloglogHazard,
    nonzero_tolerance: float,
) -> pd.DataFrame:
    edges = schema.edge_features.copy()
    edges["coefficient"] = model.coefficients[schema.base_feature_count :]
    edges["selected"] = edges["coefficient"].gt(nonzero_tolerance).astype(int)
    return edges.sort_values(["selected", "coefficient"], ascending=[False, False]).reset_index(drop=True)


def barrier_effect_table(schema: GraphSchema, model: StructuredCloglogHazard) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(schema.feature_names[: schema.base_feature_count]):
        if name == "local::eta":
            continue
        kind, identifier = ("city", name.split("::", 2)[2]) if name.startswith("barrier::city::") else (
            "station",
            name.split("::", 2)[2],
        )
        effect = float(model.coefficients[index])
        rows.append(
            {
                "level": kind,
                "identifier": identifier,
                "log_susceptibility_effect": effect,
                "barrier_value": -effect,
                "hazard_ratio": float(np.exp(effect)),
            }
        )
    return pd.DataFrame(rows).sort_values(["level", "barrier_value"], ascending=[True, False])


def graph_signal_audit(paths: list[Path]) -> dict[str, Any]:
    columns = ["seen_in_development", "onset_within_6h", "hard_negative_6h", "sparse_lag_graph_contribution"]
    frames = [
        read_artifact_csv(
            path, columns=columns, encoding="utf-8-sig", low_memory=False
        )
        for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    contribution = pd.to_numeric(frame["sparse_lag_graph_contribution"], errors="coerce").fillna(0)
    near = frame["onset_within_6h"].eq(1)
    hard = frame["hard_negative_6h"].eq(1)
    unseen = pd.to_numeric(frame["seen_in_development"], errors="coerce").fillna(0).eq(0)
    return {
        "rows": int(frame.shape[0]),
        "positive_graph_contribution_fraction": float(contribution.gt(0).mean()),
        "near_event_mean_graph_contribution": float(contribution.loc[near].mean()) if near.any() else float("nan"),
        "hard_negative_mean_graph_contribution": float(contribution.loc[hard].mean()) if hard.any() else float("nan"),
        "near_event_positive_graph_fraction": float(contribution.loc[near].gt(0).mean()) if near.any() else float("nan"),
        "hard_negative_positive_graph_fraction": float(contribution.loc[hard].gt(0).mean()) if hard.any() else float("nan"),
        "unseen_station_rows": int(unseen.sum()),
        "unseen_station_max_abs_graph_contribution": float(contribution.loc[unseen].abs().max()) if unseen.any() else 0.0,
    }


def build_graph_report(
    config: dict[str, Any],
    training_summary: dict[str, Any],
    selection: pd.DataFrame,
    selected_l1: float,
    calibrators: dict[str, HazardRateCalibrator],
    probability_test: pd.DataFrame,
    warning_metrics: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    edges: pd.DataFrame,
    signal_audit: dict[str, Any],
) -> str:
    selected_edges = edges.loc[edges["selected"].eq(1)]
    unique_edges = selected_edges[["source_station_code", "target_station_code"]].drop_duplicates().shape[0]
    test_probability = probability_test.loc[
        probability_test["cohort"].eq("all")
        & probability_test["label"].eq("onset_within_6h")
    ]
    warning_subset = warning_metrics.loc[
        warning_metrics["operating_point"].isin(
            ["test::locked_calibration_budget", "test::matched_budget_diagnostic", "test_seen::matched_budget_diagnostic"]
        )
    ]
    direct = paired_comparisons.loc[
        paired_comparisons["operating_point"].eq("locked_calibration_budget")
        &
        paired_comparisons["model_a"].eq("sparse_lag_graph_6h")
        & paired_comparisons["model_b"].eq("sparse_lag_graph_ablation_6h")
    ]
    common_lead = direct.loc[direct["metric"].eq("common_hit_lead_difference_hours")]
    neighbor_supported = bool(
        not common_lead.empty
        and float(common_lead.iloc[0]["estimate"]) >= 0.1
        and float(common_lead.iloc[0]["ci95_low"]) > 0
    )
    unseen_safe = float(signal_audit["unseen_station_max_abs_graph_contribution"]) <= 1e-12

    calibration_rows = []
    for name, calibrator in calibrators.items():
        calibration_rows.append(
            {
                "model": name,
                "shift": calibrator.log_rate_shift,
                "slope": calibrator.slope,
                "converged": calibrator.converged,
            }
        )
    top_edges = selected_edges.head(20)
    selection_display = selection.copy()
    selection_display["graph_l1_display"] = selection_display["graph_l1"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.3g}"
    )
    lines = [
        "# RIG-Hazard Hierarchical-Barrier and Sparse-Lag-Graph Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model definitions",
        "",
        "- `local_weather_hazard`: local cloglog discrete-time hazard without station structure.",
        "- `hierarchical_barrier`: jointly estimates city and station effects on the local linear predictor; L2 partial pooling shrinks small-sample stations toward their city level.",
        "- `sparse_lag_graph`: adds nonnegative, L1-sparse directed neighbor-lag excitation to the hierarchical barrier. Graph features read only source-station signals before issue time.",
        "- `sparse_lag_graph_ablation`: retains the same local and barrier parameters while setting graph contribution to zero to identify the neighbor increment.",
        "",
        "## Leakage prevention and validation contract",
        "",
        f"- Training: 2022; graph-regularization selection: {config['validation']['selection_start']} through {config['validation']['selection_end']}; calibration and threshold locking: {config['validation']['calibration_start']} through {config['validation']['calibration_end']}; test: 2024.",
        "- Within-source-station anomaly quantiles, graph coefficients, and city and station effects are all learned from 2022 only.",
        "- Graph regularization is fixed in the selection segment; no model selection occurs in the calibration segment; 2024 only tests locked thresholds. Matched-budget test results are diagnostic.",
        "",
        "## Training design",
        "",
        "```json",
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Graph-regularization selection",
        "",
        markdown_table(
            selection_display.to_dict("records"),
            [
                "model",
                "graph_l1_display",
                "active_edge_lags",
                "unique_directed_edges",
                "pr_auc",
                "brier_score",
                "event_hit_rate",
                "mean_effective_lead_hours",
                "lead_utility_hours",
            ],
        ),
        "",
        f"Selected graph L1: `{selected_l1:.8g}`; active edge-lag terms: {selected_edges.shape[0]}; unique directed edges: {unique_edges}.",
        "",
        "## Calibration parameters",
        "",
        markdown_table(calibration_rows, ["model", "shift", "slope", "converged"]),
        "",
        "## 2024 probability metrics (6 hours)",
        "",
        markdown_table(
            test_probability.to_dict("records"),
            ["model", "positives", "pr_auc", "brier_score", "brier_skill", "log_loss", "ece"],
        ),
        "",
        "## Event-warning metrics",
        "",
        markdown_table(
            warning_subset.to_dict("records"),
            [
                "model",
                "operating_point",
                "threshold",
                "false_alarm_hours_per_station_month",
                "event_hit_rate",
                "mean_effective_lead_hours",
                "median_effective_lead_hours",
                "hard_negative_far",
            ],
        ),
        "",
        "## Paired event-level tests",
        "",
        markdown_table(
            paired_comparisons.to_dict("records"),
            [
                "operating_point",
                "model_a",
                "model_b",
                "metric",
                "estimate",
                "ci95_low",
                "ci95_high",
                "evaluable_events",
                "stations",
            ],
        ),
        "",
        "`common_hit_lead_difference_hours` compares only events hit by both models; `lead_utility_difference_hours` assigns zero hours to misses and combines hit rate with lead time. Confidence intervals use station-cluster bootstrap. Confirmatory interpretation uses thresholds locked in 2023; matched-budget test results are diagnostic only.",
        "",
        "## Leading sparse edges",
        "",
        markdown_table(
            top_edges.to_dict("records"),
            ["source_station_code", "target_station_code", "lag_minutes", "distance_km", "coefficient"],
        )
        if not top_edges.empty
        else "No graph coefficient exceeds the nonzero threshold.",
        "",
        "## Conclusions",
        "",
        f"1. Training retains {selected_edges.shape[0]} edge-lag terms and {unique_edges} unique directed edges.",
        (
            "2. Under the 2023 matched-false-alarm threshold and paired station-cluster test, the neighbor graph improves common-hit lead time by at least 0.1 hours with a positive 95% lower bound; the current data support an earlier neighbor signal."
            if neighbor_supported
            else "2. The neighbor graph does not simultaneously achieve at least 0.1 hours of common-hit lead-time improvement and a positive 95% lower bound; the current data do not support a significantly earlier neighbor signal than local history."
        ),
        (
            "3. Graph contribution is exactly zero for unseen stations in 2024, so the model falls back to the city-level barrier and local hazard as designed."
            if unseen_safe
            else "3. Unseen stations have a nonzero graph contribution; inspect graph indexing or data leakage."
        ),
        "4. Regularization selection, calibration, and testing are separated in this run, but the 2024 results have now been inspected. Any subsequent structural change motivated by them requires a new year, external region, or nested rolling validation; 2024 can no longer be called an untouched final test set.",
        "",
        f"Complete artifacts: `{resolve_project_path(config['output_root'])}`",
    ]
    return "\n".join(lines) + "\n"


def run_graph_experiment(config: dict[str, Any], config_path: Path, overwrite: bool = False) -> Path:
    preprocessed_root = resolve_project_path(config["preprocessed_root"])
    baseline_root = resolve_project_path(config["baseline_root"])
    output_root = resolve_project_path(config["output_root"])
    prepare_output_root(output_root, overwrite)
    write_json(output_root / "resolved_experiment_config.json", config)

    preprocessed_config = json.loads((preprocessed_root / "resolved_config.json").read_text(encoding="utf-8"))
    scheme = str(config.get("scheme", "development"))
    train_years = split_years(preprocessed_config, scheme, "train")
    validation_years = split_years(preprocessed_config, scheme, "validation")
    test_years = split_years(preprocessed_config, scheme, "test")
    all_years = sorted(set(train_years + validation_years + test_years))
    horizons = [int(value) for value in preprocessed_config["risk_set"]["forecast_horizons_hours"]]
    step_minutes = int(preprocessed_config["time_step_minutes"])
    compression_level = int(config.get("prediction_compression_level", 1))

    station_catalog = pd.read_csv(preprocessed_root / "station_catalog.csv", encoding="utf-8-sig")
    original_seen = station_catalog.loc[
        pd.to_numeric(station_catalog["seen_in_development"], errors="coerce").fillna(0).eq(1), "station_code"
    ].astype(str)
    station_limit = config.get("station_limit")
    if station_limit:
        allowed_stations = set(sorted(original_seen)[: int(station_limit)])
        station_catalog.loc[
            ~station_catalog["station_code"].astype(str).isin(allowed_stations), "seen_in_development"
        ] = 0
    else:
        allowed_stations = set(station_catalog["station_code"].astype(str))

    def selected_paths(years: list[int]) -> list[Path]:
        return [
            path
            for path in timeline_files(preprocessed_root, years)
            if path.name.split("_", 1)[0] in allowed_stations
        ]

    train_paths = selected_paths(train_years)
    validation_paths = selected_paths(validation_years)
    test_paths = selected_paths(test_years)
    bundle = load_local_hazard_bundle(baseline_root)
    signal_cache_root, normalization, signal_summary = prepare_source_signals(
        preprocessed_root,
        output_root,
        bundle,
        train_years,
        all_years,
        config["source_signal"],
        compression_level,
        allowed_station_codes=allowed_stations,
    )

    candidate_edges = pd.read_csv(preprocessed_root / "spatial_candidate_edges.csv", encoding="utf-8-sig")
    schema = build_graph_schema(
        station_catalog,
        candidate_edges,
        [int(value) for value in config["graph"]["lags_minutes"]],
        int(config["graph"]["candidate_neighbors"]),
        float(config["graph"]["max_distance_km"]),
    )
    schema.edge_features.to_csv(output_root / "candidate_edge_lag_features.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_root / "graph_schema.json",
        {
            "feature_names": schema.feature_names,
            "base_feature_count": schema.base_feature_count,
            "graph_feature_count": schema.graph_feature_count,
            "seen_stations": sorted(schema.seen_stations),
            "source_signal_normalized_stations": sorted(normalization),
            "source_signal_station_years": int(signal_summary.shape[0]),
        },
    )

    matrix, y, sample_weight, training_summary = build_structured_training_sample(
        train_paths,
        schema,
        signal_cache_root,
        bundle,
        split_column=f"split_{scheme}",
        split_name=str(config.get("train_split", "train")),
        max_horizon=max(horizons),
        sampling=config["sampling"],
        step_minutes=step_minutes,
        rolling_window_minutes=int(config["source_signal"]["rolling_window_minutes"]),
    )
    write_json(output_root / "training_sample_summary.json", training_summary)
    fitted = fit_structured_models(matrix, y, sample_weight, schema, config)

    selection_start = pd.Timestamp(config["validation"]["selection_start"])
    selection_end = pd.Timestamp(config["validation"]["selection_end"])
    calibration_start = pd.Timestamp(config["validation"]["calibration_start"])
    calibration_end = pd.Timestamp(config["validation"]["calibration_end"])
    events = pd.read_csv(preprocessed_root / "events_recurrent.csv", encoding="utf-8-sig")
    selection = evaluate_regularization_candidates(
        validation_paths,
        schema,
        signal_cache_root,
        bundle,
        fitted,
        selection_start,
        selection_end,
        step_minutes,
        int(config["source_signal"]["rolling_window_minutes"]),
        float(config["graph"]["nonzero_tolerance"]),
        events,
        config["warning"],
    )
    selection.to_csv(output_root / "graph_regularization_selection.csv", index=False, encoding="utf-8-sig")
    selected_l1 = select_graph_model(
        selection,
        float(config["graph"]["selection_lead_utility_tie_hours"]),
        float(config["graph"]["maximum_pr_auc_relative_degradation"]),
    )
    graph_model = fitted.graph_candidates[selected_l1]
    print(f"Selected graph L1={selected_l1:.3g}", flush=True)

    edges = selected_edge_table(schema, graph_model, float(config["graph"]["nonzero_tolerance"]))
    edges.to_csv(output_root / "learned_sparse_lag_graph.csv", index=False, encoding="utf-8-sig")
    barrier_table = barrier_effect_table(schema, graph_model)
    barrier_table.to_csv(output_root / "hierarchical_station_barriers.csv", index=False, encoding="utf-8-sig")

    prediction_root = output_root / "predictions"
    validation_prediction_paths = generate_raw_predictions(
        validation_paths,
        prediction_root,
        schema,
        signal_cache_root,
        bundle,
        fitted.barrier,
        graph_model,
        horizons,
        step_minutes,
        int(config["source_signal"]["rolling_window_minutes"]),
        compression_level,
    )
    test_prediction_paths = generate_raw_predictions(
        test_paths,
        prediction_root,
        schema,
        signal_cache_root,
        bundle,
        fitted.barrier,
        graph_model,
        horizons,
        step_minutes,
        int(config["source_signal"]["rolling_window_minutes"]),
        compression_level,
    )
    calibrators = fit_graph_calibrators(validation_prediction_paths, calibration_start, calibration_end)
    apply_graph_calibrators(validation_prediction_paths + test_prediction_paths, calibrators, horizons, step_minutes, compression_level)

    probability_validation = evaluate_graph_probability(
        validation_prediction_paths,
        horizons,
        calibration_start,
        calibration_end,
    )
    probability_test = evaluate_graph_probability(test_prediction_paths, horizons)
    probability_validation.to_csv(output_root / "probability_metrics_calibration.csv", index=False, encoding="utf-8-sig")
    probability_test.to_csv(output_root / "probability_metrics_test.csv", index=False, encoding="utf-8-sig")

    warning_metrics, threshold_payload, locked_thresholds, matched_thresholds, test_warning_frame, test_events = evaluate_graph_warning_points(
        validation_prediction_paths,
        test_prediction_paths,
        events,
        calibration_start,
        calibration_end,
        test_years,
        config["warning"],
        step_minutes,
    )
    warning_metrics.to_csv(output_root / "warning_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output_root / "warning_thresholds.json", threshold_payload)

    horizon = int(config["warning"]["horizon_hours"])
    record_parts = []
    for operating_point, thresholds in [
        ("locked_calibration_budget", locked_thresholds),
        ("matched_budget_diagnostic", matched_thresholds),
    ]:
        for model, threshold in thresholds.items():
            records = event_alarm_records(test_warning_frame, test_events, model, threshold, horizon, config["warning"])
            records["operating_point"] = operating_point
            record_parts.append(records)
    event_records = pd.concat(record_parts, ignore_index=True)
    event_records.to_csv(output_root / "event_warning_records_test.csv", index=False, encoding="utf-8-sig")
    comparison_parts = []
    comparison_index = 0
    for operating_point in ["locked_calibration_budget", "matched_budget_diagnostic"]:
        operating_records = event_records.loc[event_records["operating_point"].eq(operating_point)]
        for model_b in [f"sparse_lag_graph_ablation_{horizon}h", f"hierarchical_barrier_{horizon}h", f"local_weather_hazard_{horizon}h"]:
            comparison = cluster_bootstrap_event_comparison(
                operating_records,
                f"sparse_lag_graph_{horizon}h",
                model_b,
                int(config["bootstrap"]["samples"]),
                int(config["bootstrap"]["seed"]) + comparison_index,
            )
            comparison["operating_point"] = operating_point
            comparison_parts.append(comparison)
            comparison_index += 1
    paired_comparisons = pd.concat(comparison_parts, ignore_index=True)
    paired_comparisons.to_csv(output_root / "paired_event_comparisons.csv", index=False, encoding="utf-8-sig")
    signal_audit = graph_signal_audit(test_prediction_paths)
    write_json(output_root / "graph_signal_test_audit.json", signal_audit)

    model_bundle = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_family": "hierarchical_sparse_lag_cloglog_hazard",
        "config_path": str(config_path),
        "selected_graph_l1": selected_l1,
        "barrier_model": fitted.barrier.to_dict(),
        "selected_graph_model": graph_model.to_dict(),
        "graph_candidate_models": {str(value): model.to_dict() for value, model in fitted.graph_candidates.items()},
        "calibrators": {name: calibrator.to_dict() for name, calibrator in calibrators.items()},
        "time_step_minutes": step_minutes,
        "horizons_hours": horizons,
    }
    write_json(output_root / "model_bundle.json", model_bundle)
    report = build_graph_report(
        config,
        training_summary,
        selection,
        selected_l1,
        calibrators,
        probability_test,
        warning_metrics,
        paired_comparisons,
        edges,
        signal_audit,
    )
    (output_root / "graph_experiment_report.md").write_text(report, encoding="utf-8")
    print(f"hierarchical/sparse-graph graph experiment complete: {output_root}", flush=True)
    return output_root

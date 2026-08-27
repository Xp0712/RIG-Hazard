from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_models import HazardRateCalibrator, WeightedBinaryGLM, cloglog_probability
from .config import resolve_project_path
from .deep_data import (
    RECURRENCE_BINARY_FEATURES,
    RECURRENCE_CONTINUOUS_FEATURES,
    batched_multistep_targets,
    stratified_subsample_positions,
)
from .deep_training import weighted_probability_metric_row
from .temporal_protocol import load_index, subset_index


HORIZON_STEPS = {"step": 1, "1h": 6, "3h": 18, "6h": 36}
MODEL_NAMES = ("base_cloglog", "pwp_gap_cloglog", "station_frailty_cloglog")


@dataclass
class CurrentRiskData:
    features: np.ndarray
    labels: dict[str, np.ndarray]
    observed: dict[str, np.ndarray]
    sample_weight: np.ndarray
    issue_time_ns: np.ndarray
    station_code: np.ndarray
    next_event_order: np.ndarray
    feature_names: list[str]


def compact_evaluation_data(data: CurrentRiskData) -> CurrentRiskData:
    """Drop design features once predictions have been produced for a fold."""

    return CurrentRiskData(
        features=np.empty((data.sample_weight.size, 0), dtype=np.float32),
        labels=data.labels,
        observed=data.observed,
        sample_weight=data.sample_weight,
        issue_time_ns=data.issue_time_ns,
        station_code=data.station_code,
        next_event_order=data.next_event_order,
        feature_names=[],
    )


def _subsample_index(
    index: dict[str, np.ndarray],
    maximum_rows: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    if maximum_rows is None or index["row_index"].size <= int(maximum_rows):
        return index
    positions, factors, _ = stratified_subsample_positions(index["stratum"], int(maximum_rows), seed)
    selected = subset_index(index, np.isin(np.arange(index["row_index"].size), positions))
    position_lookup = {int(position): float(factor) for position, factor in zip(positions, factors)}
    ordered_positions = np.flatnonzero(np.isin(np.arange(index["row_index"].size), positions))
    selected["sample_weight"] = selected["sample_weight"] * np.asarray(
        [position_lookup[int(position)] for position in ordered_positions], dtype=np.float32
    )
    return selected


def load_current_risk_data(
    cache_root: Path,
    split: str,
    maximum_rows: int | None = None,
    seed: int = 20260807,
) -> CurrentRiskData:
    contract = json.loads((cache_root / "sample_contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
    files = {int(row["file_id"]): row for row in manifest["files"]}
    index = _subsample_index(load_index(cache_root, split), maximum_rows, seed)
    count = int(index["row_index"].size)
    feature_count = int(contract["feature_count"])
    horizon_steps = int(contract["horizon_steps"])
    features = np.empty((count, feature_count), dtype=np.float32)
    trajectory_labels = np.zeros((count, horizon_steps), dtype=np.int8)
    trajectory_mask = np.zeros((count, horizon_steps), dtype=np.int8)
    issue_time_ns = np.empty(count, dtype=np.int64)
    station_code = np.empty(count, dtype=object)
    next_event_order = np.empty(count, dtype=np.int32)
    for file_id in np.unique(index["file_id"]):
        positions = np.flatnonzero(index["file_id"] == file_id)
        metadata = files[int(file_id)]
        rows = index["row_index"][positions].astype(np.int64)
        feature_values = np.load(cache_root / metadata["feature_path"], mmap_mode="r", allow_pickle=False)
        features[positions] = np.asarray(feature_values[rows], dtype=np.float32)
        with np.load(cache_root / metadata["target_path"], allow_pickle=False) as targets:
            labels, mask, _ = batched_multistep_targets(
                targets["risk_set"],
                targets["hazard_label"],
                targets["issue_time_ns"],
                rows,
                horizon_steps,
                int(contract["step_minutes"]),
            )
            trajectory_labels[positions] = labels
            trajectory_mask[positions] = mask
            issue_time_ns[positions] = targets["issue_time_ns"][rows]
            next_event_order[positions] = targets["next_recurrent_event_index"][rows].astype(np.int32)
        station_code[positions] = str(metadata["station_code"])
    labels_by_horizon: dict[str, np.ndarray] = {}
    observed_by_horizon: dict[str, np.ndarray] = {}
    for horizon, steps in HORIZON_STEPS.items():
        event = trajectory_labels[:, :steps].max(axis=1) > 0.5
        observed = event | (trajectory_mask[:, steps - 1] > 0.5)
        labels_by_horizon[horizon] = event.astype(np.int8)
        observed_by_horizon[horizon] = observed
    return CurrentRiskData(
        features=features,
        labels=labels_by_horizon,
        observed=observed_by_horizon,
        sample_weight=index["sample_weight"].astype(np.float64),
        issue_time_ns=issue_time_ns,
        station_code=station_code.astype(str),
        next_event_order=next_event_order,
        feature_names=list(contract["feature_names"]),
    )


class RecurrenceDesign:
    def __init__(
        self,
        model_name: str,
        feature_names: list[str],
        transformer: dict[str, Any],
    ):
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unsupported recurrence statistical model: {model_name}")
        self.model_name = model_name
        self.feature_names = list(feature_names)
        self.continuous_names = list(transformer["continuous_features"])
        self.means = np.asarray(transformer["means"], dtype=float)
        self.stds = np.asarray(transformer["stds"], dtype=float)
        recurrence_names = set(RECURRENCE_CONTINUOUS_FEATURES + RECURRENCE_BINARY_FEATURES)
        self.base_indices = [index for index, name in enumerate(feature_names) if name not in recurrence_names]
        self.base_names = [feature_names[index] for index in self.base_indices]
        self.gap_knots: list[float] = []
        self.added_means = np.empty(0, dtype=float)
        self.added_stds = np.empty(0, dtype=float)
        self.added_names: list[str] = []
        self.station_levels: list[str] = []

    def _raw_feature(self, features: np.ndarray, name: str) -> np.ndarray:
        index = self.feature_names.index(name)
        values = np.asarray(features[:, index], dtype=float)
        if name in self.continuous_names:
            continuous_index = self.continuous_names.index(name)
            values = values * self.stds[continuous_index] + self.means[continuous_index]
        return values

    def _raw_added(self, features: np.ndarray) -> tuple[np.ndarray, list[str]]:
        order = np.maximum(np.rint(self._raw_feature(features, "current_event_order")), 1)
        gap = np.maximum(self._raw_feature(features, "time_since_last_recurrent_event_hours"), 0.0)
        missing = self._raw_feature(features, "previous_recurrent_event_missing") > 0.5
        log_gap = np.where(missing, 0.0, np.log1p(gap))
        columns = [
            ("event_order_2", order == 2),
            ("event_order_3plus", order >= 3),
            ("previous_event_missing", missing),
            ("log_gap_hours", log_gap),
        ]
        for index, knot in enumerate(self.gap_knots, start=1):
            columns.append((f"log_gap_hinge_{index}", np.maximum(log_gap - float(knot), 0.0)))
        events_7d = np.log1p(np.maximum(self._raw_feature(features, "events_past_7d"), 0.0))
        events_30d = np.log1p(np.maximum(self._raw_feature(features, "events_past_30d"), 0.0))
        previous_duration = np.log1p(np.maximum(self._raw_feature(features, "previous_event_duration_hours"), 0.0))
        previous_thickness = np.log1p(np.maximum(self._raw_feature(features, "previous_event_max_thickness"), 0.0))
        previous_severity = np.maximum(self._raw_feature(features, "previous_event_severity"), 0.0)
        previous_duration = np.where(missing, 0.0, previous_duration)
        previous_thickness = np.where(missing, 0.0, previous_thickness)
        previous_severity = np.where(missing, 0.0, previous_severity)
        columns.extend(
            [
                ("events_past_7d_log", events_7d),
                ("events_past_30d_log", events_30d),
                ("previous_duration_log", previous_duration),
                ("previous_thickness_log", previous_thickness),
                ("previous_severity", previous_severity),
                ("order_2_by_log_gap", (order == 2) * log_gap),
                ("order_3plus_by_log_gap", (order >= 3) * log_gap),
            ]
        )
        return np.column_stack([np.asarray(values, dtype=float) for _, values in columns]), [
            name for name, _ in columns
        ]

    def fit(self, data: CurrentRiskData) -> "RecurrenceDesign":
        if self.model_name == "base_cloglog":
            return self
        gap = np.maximum(self._raw_feature(data.features, "time_since_last_recurrent_event_hours"), 0.0)
        missing = self._raw_feature(data.features, "previous_recurrent_event_missing") > 0.5
        observed_gap = np.log1p(gap[~missing & np.isfinite(gap)])
        self.gap_knots = (
            np.quantile(observed_gap, [0.25, 0.5, 0.75]).astype(float).tolist()
            if observed_gap.size
            else [0.0, 0.0, 0.0]
        )
        added, self.added_names = self._raw_added(data.features)
        self.added_means = np.nanmean(added, axis=0)
        self.added_stds = np.nanstd(added, axis=0)
        self.added_means = np.where(np.isfinite(self.added_means), self.added_means, 0.0)
        self.added_stds = np.where(np.isfinite(self.added_stds) & (self.added_stds > 1e-8), self.added_stds, 1.0)
        if self.model_name == "station_frailty_cloglog":
            self.station_levels = sorted(set(data.station_code.tolist()))
        return self

    def transform(self, data: CurrentRiskData) -> tuple[np.ndarray, list[str], np.ndarray]:
        base = np.asarray(data.features[:, self.base_indices], dtype=np.float64)
        parts = [base]
        names = list(self.base_names)
        penalty = [np.ones(base.shape[1], dtype=float)]
        if self.model_name != "base_cloglog":
            added, names_now = self._raw_added(data.features)
            added = np.where(np.isfinite(added), added, self.added_means)
            added = (added - self.added_means) / self.added_stds
            parts.append(added)
            names.extend(names_now)
            penalty.append(np.ones(added.shape[1], dtype=float))
        if self.model_name == "station_frailty_cloglog":
            station_matrix = np.column_stack(
                [data.station_code == station for station in self.station_levels]
            ).astype(float)
            parts.append(station_matrix)
            names.extend([f"station_frailty[{station}]" for station in self.station_levels])
            penalty.append(np.full(station_matrix.shape[1], 10.0, dtype=float))
        return np.concatenate(parts, axis=1), names, np.concatenate(penalty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "continuous_names": self.continuous_names,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "base_indices": self.base_indices,
            "base_names": self.base_names,
            "gap_knots": self.gap_knots,
            "added_means": self.added_means.tolist(),
            "added_stds": self.added_stds.tolist(),
            "added_names": self.added_names,
            "station_levels": self.station_levels,
        }


def _fit_model(
    model_name: str,
    train: CurrentRiskData,
    transformer: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[RecurrenceDesign, WeightedBinaryGLM, list[str]]:
    design = RecurrenceDesign(model_name, train.feature_names, transformer).fit(train)
    x, names, penalty = design.transform(train)
    l2_key = {
        "base_cloglog": "base_l2",
        "pwp_gap_cloglog": "pwp_l2",
        "station_frailty_cloglog": "frailty_l2",
    }[model_name]
    model = WeightedBinaryGLM(
        "cloglog",
        l2=float(settings.get(l2_key, 1e-4)),
        max_iter=int(settings.get("maximum_iterations", 250)),
        penalty_weights=penalty,
    ).fit(x, train.labels["step"], train.sample_weight)
    return design, model, names


def _event_order_masks(data: CurrentRiskData) -> dict[str, np.ndarray]:
    order = data.next_event_order
    return {
        "all": np.ones(order.size, dtype=bool),
        "first_risk": order == 1,
        "second_risk": order == 2,
        "third_plus_risk": order >= 3,
        "recurrent_risk": order >= 2,
    }


def probability_rows(
    model_name: str,
    split: str,
    eta: np.ndarray,
    data: CurrentRiskData,
    calibrator: HazardRateCalibrator | None,
    fold: int | str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    calibration_modes = [("raw", None)]
    if calibrator is not None:
        calibration_modes.append(("calibrated_2022_oof", calibrator))
    for calibration, calibration_model in calibration_modes:
        for horizon, steps in HORIZON_STEPS.items():
            probability = (
                cloglog_probability(eta, steps=steps)
                if calibration_model is None
                else calibration_model.predict(eta, steps=steps)
            )
            for event_group, group_mask in _event_order_masks(data).items():
                selected = data.observed[horizon] & group_mask
                if not selected.any():
                    continue
                row = weighted_probability_metric_row(
                    model_name,
                    calibration,
                    horizon,
                    data.labels[horizon][selected],
                    probability[selected],
                    data.sample_weight[selected],
                )
                row.update({"split": split, "fold": fold, "risk_order_group": event_group})
                rows.append(row)
    return rows


def _prediction_frame(
    data: CurrentRiskData,
    predictions: dict[str, tuple[np.ndarray, HazardRateCalibrator]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "issue_time": pd.to_datetime(data.issue_time_ns, unit="ns"),
            "station_code": data.station_code,
            "next_event_order_evaluation_only": data.next_event_order,
            "sample_weight": data.sample_weight,
        }
    )
    for horizon in HORIZON_STEPS:
        frame[f"label_{horizon}"] = data.labels[horizon]
        frame[f"observed_{horizon}"] = data.observed[horizon].astype(np.int8)
    for model_name, (eta, calibrator) in predictions.items():
        for horizon, steps in HORIZON_STEPS.items():
            frame[f"{model_name}_risk_{horizon}"] = calibrator.predict(eta, steps=steps)
    return frame


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_recurrence_statistical_models(
    config: dict[str, Any],
    config_path: Path,
    output_root_override: str | Path | None = None,
    number_folds: int = 5,
    maximum_train_rows: int | None = None,
    maximum_evaluation_rows: int | None = None,
    save_predictions: bool = True,
    overwrite: bool = False,
) -> Path:
    cache_root = resolve_project_path(config["cache_root"])
    output_root = resolve_project_path(
        output_root_override
        or config.get(
            "statistical_output_root",
            "results/recurrence_analysis/model_comparison/statistical_models",
        )
    )
    _prepare_output(output_root, overwrite)
    transformer = json.loads((cache_root / "feature_transformer.json").read_text(encoding="utf-8"))
    settings = dict(config.get("statistical_models", {}))
    seed = int(config["sampling"]["seed"])
    fold_metric_rows: list[dict[str, Any]] = []
    oof: dict[str, dict[str, list[Any]]] = {
        name: {"eta": [], "data": []} for name in MODEL_NAMES
    }
    for fold in range(int(number_folds)):
        train = load_current_risk_data(
            cache_root, f"cv_fold{fold}_train", maximum_train_rows, seed + fold * 101
        )
        validation = load_current_risk_data(
            cache_root,
            f"cv_fold{fold}_validation",
            maximum_evaluation_rows,
            seed + fold * 101 + 1,
        )
        compact_validation = compact_evaluation_data(validation)
        for model_name in MODEL_NAMES:
            design, model, _ = _fit_model(model_name, train, transformer, settings)
            x_validation, _, _ = design.transform(validation)
            eta = model.decision_function(x_validation)
            fold_metric_rows.extend(
                probability_rows(model_name, "2022_block_validation", eta, validation, None, fold)
            )
            oof[model_name]["eta"].append(eta)
            oof[model_name]["data"].append(compact_validation)
        print(f"Recurrence statistical cross-validation fold: {fold + 1}/{number_folds}", flush=True)

    calibrators: dict[str, HazardRateCalibrator] = {}
    oof_metric_rows: list[dict[str, Any]] = []
    pooled_oof_data = concatenate_risk_data(oof[MODEL_NAMES[0]]["data"])
    for model_name in MODEL_NAMES:
        eta = np.concatenate(oof[model_name]["eta"])
        calibrator = HazardRateCalibrator().fit(
            eta,
            pooled_oof_data.labels["step"],
            pooled_oof_data.sample_weight,
        )
        calibrators[model_name] = calibrator
        oof_metric_rows.extend(
            probability_rows(
                model_name,
                "2022_pooled_oof",
                eta,
                pooled_oof_data,
                calibrator,
                "pooled",
            )
        )

    training = load_current_risk_data(cache_root, "selection_2022", maximum_train_rows, seed + 7001)
    trained_models: dict[str, tuple[RecurrenceDesign, WeightedBinaryGLM, list[str]]] = {}
    bundles: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        design, model, names = _fit_model(model_name, training, transformer, settings)
        trained_models[model_name] = (design, model, names)
        bundles[model_name] = {
            "design": design.to_dict(),
            "glm": model.to_dict(),
            "coefficient_names": names,
            "coefficients": model.coefficients.tolist() if model.coefficients is not None else [],
            "calibrator": calibrators[model_name].to_dict(),
        }

    evaluation_metric_rows: list[dict[str, Any]] = []
    evaluation_splits = (("cross_year_2023", "2023_cross_year"), ("final_time_2024", "2024_final_time"))
    for split_name, display_name in evaluation_splits:
        data = load_current_risk_data(
            cache_root,
            split_name,
            maximum_evaluation_rows,
            seed + (2023 if "2023" in split_name else 2024),
        )
        predictions: dict[str, tuple[np.ndarray, HazardRateCalibrator]] = {}
        for model_name, (design, model, _) in trained_models.items():
            x, _, _ = design.transform(data)
            eta = model.decision_function(x)
            predictions[model_name] = (eta, calibrators[model_name])
            evaluation_metric_rows.extend(
                probability_rows(model_name, display_name, eta, data, calibrators[model_name], "frozen")
            )
        if save_predictions:
            _prediction_frame(data, predictions).to_csv(
                output_root / f"predictions_{display_name}.csv.gz",
                index=False,
                compression="gzip",
            )
        print(f"Recurrence statistical evaluation complete: {display_name}", flush=True)

    fold_metrics = pd.DataFrame(fold_metric_rows)
    oof_metrics = pd.DataFrame(oof_metric_rows)
    evaluation_metrics = pd.DataFrame(evaluation_metric_rows)
    fold_metrics.to_csv(output_root / "fold_metrics.csv", index=False)
    oof_metrics.to_csv(output_root / "pooled_oof_metrics.csv", index=False)
    evaluation_metrics.to_csv(output_root / "locked_year_metrics.csv", index=False)
    (output_root / "statistical_model_bundle.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "config_path": str(config_path),
                "models": bundles,
                "input_contract": "All recurrence predictors are causal at issue_time; next_event_order is evaluation-only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selected = oof_metrics.loc[
        oof_metrics["calibration"].eq("calibrated_2022_oof")
        & oof_metrics["horizon"].eq("6h")
        & oof_metrics["risk_order_group"].eq("all")
    ].sort_values("log_loss")
    selected_name = str(selected.iloc[0]["model_name"]) if not selected.empty else "unavailable"
    base_loss = selected.loc[selected["model_name"].eq("base_cloglog"), "log_loss"]
    recurrence_loss = selected.loc[selected["model_name"].ne("base_cloglog"), "log_loss"]
    recurrence_gain = (
        float(base_loss.iloc[0] - recurrence_loss.min())
        if not base_loss.empty and not recurrence_loss.empty
        else None
    )
    report = [
        "# Recurrence-specific statistical model comparison",
        "",
        "Models: base cloglog, event-order-stratified PWP gap-time cloglog, and a station-penalized frailty approximation.",
        "All selection and calibration use pooled 2022 out-of-fold predictions. The 2023 and 2024 calibrators are frozen.",
        "",
        f"Selected by pooled 2022 calibrated 6-hour log loss: {selected_name}.",
        f"Best recurrence-model log-loss gain over base: {recurrence_gain if recurrence_gain is not None else 'unavailable'}.",
        "",
        "If both recurrence-specific models fail to improve locked-year calibration and warning utility, recurrence should be retained as a risk-set structure rather than claimed as a predictive feature.",
        "The 2024 set is a retrospectively locked temporal confirmation because it was previously inspected in this project.",
    ]
    (output_root / "recurrence_statistical_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Recurrence statistical models complete: {output_root}", flush=True)
    return output_root


def concatenate_risk_data(parts: list[CurrentRiskData]) -> CurrentRiskData:
    if not parts:
        raise ValueError("At least one risk-data part is required")
    return CurrentRiskData(
        features=np.concatenate([part.features for part in parts]),
        labels={key: np.concatenate([part.labels[key] for part in parts]) for key in HORIZON_STEPS},
        observed={key: np.concatenate([part.observed[key] for part in parts]) for key in HORIZON_STEPS},
        sample_weight=np.concatenate([part.sample_weight for part in parts]),
        issue_time_ns=np.concatenate([part.issue_time_ns for part in parts]),
        station_code=np.concatenate([part.station_code for part in parts]),
        next_event_order=np.concatenate([part.next_event_order for part in parts]),
        feature_names=parts[0].feature_names,
    )

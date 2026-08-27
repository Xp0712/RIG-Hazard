from __future__ import annotations

import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .config import resolve_project_path
from .deep_data import DeepCacheBatchSource
from .deep_models import (
    DeepHazardModel,
    discrete_hazard_nll,
    discrete_hazard_nll_components,
)
from .naming import MAIN_MODEL_NAME, artifact_value
from .preprocessing import prepare_output_root, write_json
from .torch_runtime import torch


DEFAULT_SLOW_RECURRENCE_FEATURES = [
    "time_since_last_recurrent_event_hours",
    "current_event_order",
    "events_past_7d",
    "events_past_30d",
    "previous_event_duration_hours",
    "previous_event_max_thickness",
    "previous_event_severity",
    "previous_recurrent_event_missing",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

SPEC_RECURRENCE_FEATURES = [
    "time_since_last_recurrent_event_hours",
    "current_event_order",
    "events_past_7d",
    "events_past_30d",
    "previous_event_duration_hours",
    "previous_event_max_thickness",
    "previous_event_severity",
    "previous_recurrent_event_missing",
]


@dataclass
class WeightedTrajectoryCalibrator:
    log_rate_shift: float = 0.0
    slope: float = 1.0
    objective: float | None = None
    converged: bool = False

    def fit(
        self,
        eta: np.ndarray,
        target: np.ndarray,
        risk_mask: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "WeightedTrajectoryCalibrator":
        eta_values = np.asarray(eta)
        target_values = np.asarray(target)
        mask_values = np.asarray(risk_mask)
        row_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if eta_values.shape != target_values.shape or eta_values.shape != mask_values.shape:
            raise ValueError("eta, target, and risk_mask must have identical shapes")
        if eta_values.ndim != 2 or row_weight.size != eta_values.shape[0]:
            raise ValueError("Trajectory calibration requires one row weight per two-dimensional trajectory")
        if not np.isfinite(row_weight).all() or np.any(row_weight <= 0):
            raise ValueError("Trajectory calibration weights must be finite and positive")
        chunk_rows = 32768
        denominator = 0.0
        for start in range(0, eta_values.shape[0], chunk_rows):
            stop = min(start + chunk_rows, eta_values.shape[0])
            valid_counts = (mask_values[start:stop] > 0.5).sum(axis=1)
            denominator += float(np.dot(row_weight[start:stop], valid_counts))
        if denominator <= 0:
            raise ValueError("Trajectory calibration has no observable hazard bins")

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            shift, slope = parameters
            loss_total = 0.0
            shift_gradient = 0.0
            slope_gradient = 0.0
            for start in range(0, eta_values.shape[0], chunk_rows):
                stop = min(start + chunk_rows, eta_values.shape[0])
                valid = mask_values[start:stop] > 0.5
                if not valid.any():
                    continue
                score = np.asarray(eta_values[start:stop][valid], dtype=np.float64)
                label = target_values[start:stop][valid] > 0.5
                weight = np.broadcast_to(
                    row_weight[start:stop, None], valid.shape
                )[valid]
                calibrated = np.minimum(shift + slope * score, 15.0)
                rate = np.exp(calibrated)
                probability = -np.expm1(-rate)
                positive_loss = np.where(
                    calibrated < -10.0,
                    -calibrated,
                    -np.log(np.clip(probability, 1e-300, 1.0)),
                )
                losses = np.where(label, positive_loss, rate)
                positive_gradient = np.zeros_like(rate)
                small = calibrated < -10.0
                positive_gradient[small] = -1.0
                stable = ~small & (rate < 50.0)
                positive_gradient[stable] = -rate[stable] / np.expm1(rate[stable])
                gradient_eta = np.where(label, positive_gradient, rate)
                weighted_gradient = weight * gradient_eta
                loss_total += float(np.dot(weight, losses))
                shift_gradient += float(weighted_gradient.sum())
                slope_gradient += float(np.dot(weighted_gradient, score))
            gradient = np.asarray(
                [shift_gradient / denominator, slope_gradient / denominator], dtype=np.float64
            )
            return loss_total / denominator, gradient

        result = minimize(
            objective,
            np.asarray([0.0, 1.0], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-20.0, 20.0), (0.0, 10.0)],
            options={"maxiter": 200, "ftol": 1e-10, "maxls": 40},
        )
        self.log_rate_shift = float(result.x[0])
        self.slope = float(result.x[1])
        self.objective = float(result.fun)
        self.converged = bool(result.success)
        return self

    def transform_eta(self, eta: np.ndarray) -> np.ndarray:
        return self.log_rate_shift + self.slope * np.asarray(eta, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_rate_shift": self.log_rate_shift,
            "slope": self.slope,
            "objective": self.objective,
            "converged": self.converged,
            "semantics": "One affine map is fitted to all observed future hazard bins; cumulative risk remains coherent.",
        }


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = os.environ.get("RIG_HAZARD_FAST_CUDA") == "1"
        torch.backends.cudnn.allow_tf32 = os.environ.get("RIG_HAZARD_FAST_CUDA") == "1"
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = os.environ.get("RIG_HAZARD_FAST_CUDA") == "1"


def build_model(
    model_name: str,
    feature_count: int,
    horizon_steps: int,
    config: dict[str, Any],
    feature_names: list[str] | None = None,
) -> DeepHazardModel:
    protocol_variant = dict(config.get("protocol_model_variants", {}).get(model_name, {}))
    encoder_name = str(protocol_variant.get("encoder", model_name))
    model_config = config["models"]
    history_steps = int(round(float(config["history_hours"]) * 60 / int(config["step_minutes"])))
    encoder_config: dict[str, Any] = {}
    if encoder_name == "gru":
        hidden_size = int(protocol_variant.get("hidden_size", model_config["gru_hidden_size"]))
    elif encoder_name == "gru_subset":
        if feature_names is None or len(feature_names) != feature_count:
            raise ValueError("gru_subset requires ordered feature_names")
        included = list(protocol_variant.get("included_features", []))
        missing = sorted(set(included) - set(feature_names))
        if not included or missing:
            raise ValueError(f"GRU subset is empty or missing cache features: {missing}")
        encoder_config = {
            "feature_indices": [feature_names.index(name) for name in included],
            "feature_names": included,
        }
        hidden_size = int(protocol_variant.get("hidden_size", model_config["gru_hidden_size"]))
    elif encoder_name == "tcn":
        hidden_size = int(model_config["tcn_channels"])
    elif encoder_name == "recurrent_dual":
        if feature_names is None or len(feature_names) != feature_count:
            raise ValueError("recurrent_dual requires the ordered feature_names from the cache contract")
        settings = dict(config.get("recurrence_model", {}))
        slow_names = list(settings.get("slow_feature_names", DEFAULT_SLOW_RECURRENCE_FEATURES))
        missing = sorted(set(slow_names) - set(feature_names))
        if missing:
            raise ValueError(f"Recurrence cache is missing slow-branch features: {missing}")
        excluded_fast = set(settings.get("fast_excluded_feature_names", slow_names))
        encoder_config = {
            **settings,
            "fast_feature_indices": [index for index, name in enumerate(feature_names) if name not in excluded_fast],
            "slow_feature_indices": [feature_names.index(name) for name in slow_names],
            "slow_feature_names": slow_names,
        }
        hidden_size = int(encoder_config.get("output_size", 24))
    elif encoder_name in {
        "weather_fast",
        "weather_slow",
        "weather_dual",
        "weather_dual_rec",
        "weather_dual_rec_gate",
    }:
        if feature_names is None or len(feature_names) != feature_count:
            raise ValueError(
                "Dual-resolution weather encoders require ordered feature_names"
            )
        settings = {
            **dict(config.get("dual_weather_model", {})),
            **dict(protocol_variant.get("encoder_config", {})),
        }
        recurrence_names = list(
            settings.get("recurrence_feature_names", SPEC_RECURRENCE_FEATURES)
        )
        missing = sorted(set(recurrence_names) - set(feature_names))
        if missing:
            raise ValueError(
                f"Dual-resolution cache is missing recurrence features: {missing}"
            )
        weather_names = list(
            settings.get(
                "weather_feature_names",
                [name for name in feature_names if name not in set(recurrence_names)],
            )
        )
        missing_weather = sorted(set(weather_names) - set(feature_names))
        if missing_weather:
            raise ValueError(
                f"Dual-resolution cache is missing weather features: {missing_weather}"
            )
        encoder_config = {
            **settings,
            "weather_feature_indices": [feature_names.index(name) for name in weather_names],
            "weather_feature_names": weather_names,
            "recurrence_feature_indices": [
                feature_names.index(name) for name in recurrence_names
            ],
            "recurrence_feature_names": recurrence_names,
        }
        hidden_size = int(encoder_config.get("output_size", 16))
    else:
        modern_models = config.get("modern_baseline_models", {})
        if encoder_name not in modern_models:
            raise ValueError(f"Missing modern baseline configuration for {encoder_name}")
        encoder_config = dict(modern_models[encoder_name])
        hidden_size = int(encoder_config.get("output_size", 32))
    included_features = protocol_variant.get("included_features")
    if included_features is not None and feature_names is None:
        raise ValueError("included_features requires ordered feature_names")
    if included_features is not None and protocol_variant.get("masked_features"):
        raise ValueError("Use either included_features or masked_features, not both")
    if included_features is not None and encoder_name != "gru_subset":
        unknown_included = sorted(set(included_features) - set(feature_names or []))
        if unknown_included:
            raise ValueError(f"Included model features are missing from the cache: {unknown_included}")
        masked_features = [
            name for name in (feature_names or []) if name not in set(included_features)
        ]
    elif encoder_name == "gru_subset":
        masked_features = []
    else:
        masked_features = list(protocol_variant.get("masked_features", []))
    unknown_masked = sorted(set(masked_features) - set(feature_names or []))
    if unknown_masked:
        raise ValueError(f"Masked model features are missing from the cache: {unknown_masked}")
    model = DeepHazardModel(
        encoder_type=encoder_name,
        input_size=feature_count,
        horizon_steps=horizon_steps,
        hidden_size=hidden_size,
        kernel_size=int(model_config["tcn_kernel_size"]),
        dilations=[int(value) for value in model_config["tcn_dilations"]],
        dropout=float(model_config["dropout"]),
        horizon_embedding_dim=int(model_config["horizon_embedding_dim"]),
        initial_log_rate=float(model_config.get("initial_log_rate", -9.0)),
        history_steps=history_steps,
        encoder_config=encoder_config,
        masked_feature_indices=[
            feature_names.index(name)
            for name in masked_features
        ] if feature_names is not None else [],
    )
    if encoder_name == "tcn":
        if model.encoder.receptive_field_steps < history_steps:
            raise ValueError(
                f"TCN receptive field ({model.encoder.receptive_field_steps}) is shorter than history ({history_steps})"
            )
    parameter_target = protocol_variant.get("parameter_target")
    if parameter_target is not None:
        target = int(parameter_target)
        tolerance = float(protocol_variant.get("parameter_tolerance", 0.10))
        actual = int(sum(parameter.numel() for parameter in model.parameters()))
        lower, upper = target * (1.0 - tolerance), target * (1.0 + tolerance)
        model.architecture_config["parameter_audit"] = {
            "target": target,
            "tolerance": tolerance,
            "actual": actual,
            "within_tolerance": bool(lower <= actual <= upper),
        }
        if not lower <= actual <= upper:
            warnings.warn(
                f"{model_name} has {actual} parameters; target is {target} ± {tolerance:.0%}",
                RuntimeWarning,
            )
    return model


def _hard_negative_multiplier(stratum: torch.Tensor, multiplier: float) -> torch.Tensor:
    if multiplier == 1.0:
        return torch.ones_like(stratum, dtype=torch.float32)
    return torch.where(
        stratum.eq(1),
        torch.full_like(stratum, multiplier, dtype=torch.float32),
        torch.ones_like(stratum, dtype=torch.float32),
    )


def _training_microbatch_size(
    model: DeepHazardModel,
    device: torch.device,
    effective_batch_size: int,
) -> int:
    if model.encoder_type != "patchtst":
        return effective_batch_size
    raw_value = os.environ.get("RIG_HAZARD_PATCHTST_MICROBATCH_SIZE", "").strip()
    if not raw_value:
        return effective_batch_size
    try:
        requested = int(raw_value)
    except ValueError as error:
        raise ValueError("RIG_HAZARD_PATCHTST_MICROBATCH_SIZE must be an integer") from error
    if requested < 1:
        raise ValueError("RIG_HAZARD_PATCHTST_MICROBATCH_SIZE must be positive")
    return min(requested, effective_batch_size)


def train_epoch(
    model: DeepHazardModel,
    source: DeepCacheBatchSource,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    seed: int,
    gradient_clip_norm: float,
    hard_negative_loss_multiplier: float,
    city_barrier_l2: float = 0.0,
    station_barrier_l2: float = 0.0,
    return_details: bool = False,
) -> float | dict[str, float]:
    model.train()
    weighted_loss = 0.0
    represented_batches = 0.0
    positive_bins = 0.0
    trajectory_weight = 0.0
    weighted_objective = 0.0
    for batch in source.iter_batches(batch_size, shuffle=True, seed=seed):
        history = batch["history"].to(device)
        target = batch["hazard_target"].to(device)
        risk_mask = batch["risk_mask"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        hard_weight = _hard_negative_multiplier(
            batch["stratum"].to(device), hard_negative_loss_multiplier
        )
        station_index = batch["station_index"].to(device)
        city_index = batch["city_index"].to(device)
        optimizer.zero_grad(set_to_none=True)
        microbatch_size = _training_microbatch_size(model, device, int(history.shape[0]))
        if microbatch_size < int(history.shape[0]):
            represented = risk_mask * sample_weight[:, None] * hard_weight[:, None]
            normalizer = max(float(represented.sum().detach().cpu()), 1.0)
            detached_numerator = history.new_zeros(())
            for start in range(0, int(history.shape[0]), microbatch_size):
                selected = slice(start, start + microbatch_size)
                eta = model(history[selected], station_index[selected], city_index[selected])
                numerator, _ = discrete_hazard_nll_components(
                    eta,
                    target[selected],
                    risk_mask[selected],
                    sample_weight[selected],
                    hard_weight[selected],
                )
                (numerator / normalizer).backward()
                detached_numerator = detached_numerator + numerator.detach()
                del eta, numerator
            data_loss = detached_numerator / normalizer
            penalty = model.barrier_penalty(city_barrier_l2, station_barrier_l2)
            if penalty.requires_grad:
                penalty.backward()
            objective = data_loss + penalty.detach()
        else:
            eta = model(history, station_index, city_index)
            data_loss = discrete_hazard_nll(eta, target, risk_mask, sample_weight, hard_weight)
            objective = data_loss + model.barrier_penalty(city_barrier_l2, station_barrier_l2)
            objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        represented = float((risk_mask * sample_weight[:, None] * hard_weight[:, None]).sum().detach().cpu())
        weighted_loss += float(data_loss.detach().cpu()) * represented
        weighted_objective += float(objective.detach().cpu()) * represented
        represented_batches += represented
        row_weight = sample_weight * hard_weight
        positive_bins += float((target * risk_mask * row_weight[:, None]).sum().detach().cpu())
        trajectory_weight += float(row_weight.sum().detach().cpu())
    details = {
        "nll": weighted_loss / max(represented_batches, 1.0),
        "objective": weighted_objective / max(represented_batches, 1.0),
        "effective_risk_bins": represented_batches,
        "positive_risk_bins": positive_bins,
        "trajectory_weight": trajectory_weight,
        "positive_bin_rate": positive_bins / max(represented_batches, 1.0),
    }
    return details if return_details else details["nll"]


def evaluate_model_nll(
    model: DeepHazardModel,
    source: DeepCacheBatchSource,
    device: torch.device,
    batch_size: int,
    return_details: bool = False,
) -> float | dict[str, float]:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    positive_bins = 0.0
    trajectory_weight = 0.0
    with torch.no_grad():
        for batch in source.iter_batches(batch_size, shuffle=False):
            history = batch["history"].to(device)
            target = batch["hazard_target"].to(device)
            risk_mask = batch["risk_mask"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            eta = model(
                history,
                batch["station_index"].to(device),
                batch["city_index"].to(device),
            )
            batch_numerator, batch_denominator = discrete_hazard_nll_components(
                eta, target, risk_mask, sample_weight
            )
            numerator += float(batch_numerator.detach().cpu())
            denominator += float(batch_denominator.detach().cpu())
            positive_bins += float((target * risk_mask * sample_weight[:, None]).sum().detach().cpu())
            trajectory_weight += float(sample_weight.sum().detach().cpu())
    details = {
        "nll": numerator / max(denominator, 1.0),
        "nll_numerator": numerator,
        "effective_risk_bins": denominator,
        "positive_risk_bins": positive_bins,
        "trajectory_weight": trajectory_weight,
        "positive_bin_rate": positive_bins / max(denominator, 1.0),
    }
    return details if return_details else details["nll"]


def collect_trajectory_arrays(
    model: DeepHazardModel,
    source: DeepCacheBatchSource,
    device: torch.device,
    batch_size: int,
    include_current_features: bool = True,
    include_metadata: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ["eta", "target", "risk_mask", "sample_weight"]
    }
    if include_current_features:
        parts["current_features"] = []
    if include_metadata:
        parts.update({"issue_time_ns": [], "file_id": [], "row_index": []})
    rec_gate_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in source.iter_batches(batch_size, shuffle=False):
            history = batch["history"].to(device)
            eta = model(
                history,
                batch["station_index"].to(device),
                batch["city_index"].to(device),
            )
            diagnostics = model.diagnostics()
            parts["eta"].append(eta.detach().cpu().numpy().astype(np.float32))
            parts["target"].append(batch["hazard_target"].numpy().astype(np.int8, copy=False))
            parts["risk_mask"].append(batch["risk_mask"].numpy().astype(np.int8, copy=False))
            parts["sample_weight"].append(batch["sample_weight"].numpy().astype(np.float64, copy=False))
            if include_current_features:
                parts["current_features"].append(
                    batch["history"][:, -1, :].numpy().astype(np.float32, copy=False)
                )
            if include_metadata:
                parts["issue_time_ns"].append(batch["issue_time_ns"].numpy().astype(np.int64, copy=False))
                parts["file_id"].append(batch["file_id"].numpy().astype(np.int16, copy=False))
                parts["row_index"].append(batch["row_index"].numpy().astype(np.int32, copy=False))
            if "rec_gate" in diagnostics:
                rec_gate_parts.append(
                    diagnostics["rec_gate"].detach().cpu().numpy().astype(np.float32)
                )
    arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    if rec_gate_parts:
        arrays["rec_gate"] = np.concatenate(rec_gate_parts, axis=0)
    return arrays


def weighted_probability_metric_row(
    model_name: str,
    calibration: str,
    horizon: str,
    y: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, Any]:
    label = np.asarray(y, dtype=np.int8)
    score = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    weight = np.asarray(sample_weight, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(weight) & (weight > 0)
    label, score, weight = label[valid], score[valid], weight[valid]
    weight_sum = max(float(weight.sum()), 1e-12)
    prevalence = float(np.dot(weight, label) / weight_sum)
    brier = float(np.dot(weight, np.square(score - label)) / weight_sum)
    reference_brier = prevalence * (1.0 - prevalence)
    edges = np.asarray(
        [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0 + 1e-9]
    )
    assignments = np.clip(np.digitize(score, edges) - 1, 0, edges.size - 2)
    ece = 0.0
    maximum_gap = 0.0
    populated_bins = 0
    for bin_index in range(edges.size - 1):
        mask = assignments == bin_index
        if not mask.any():
            continue
        populated_bins += 1
        bin_weight = weight[mask]
        bin_total = float(bin_weight.sum())
        observed = float(np.dot(bin_weight, label[mask]) / bin_total)
        predicted = float(np.dot(bin_weight, score[mask]) / bin_total)
        gap = abs(observed - predicted)
        ece += bin_total / weight_sum * gap
        maximum_gap = max(maximum_gap, gap)
    has_two_classes = 0 < label.sum() < label.size
    return {
        "model_name": model_name,
        "calibration": calibration,
        "horizon": horizon,
        "sample_rows": int(label.size),
        "represented_rows": weight_sum,
        "positive_rows": int(label.sum()),
        "represented_positives": float(np.dot(weight, label)),
        "prevalence": prevalence,
        "mean_probability": float(np.dot(weight, score) / weight_sum),
        "pr_auc": float(average_precision_score(label, score, sample_weight=weight)) if has_two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(label, score, sample_weight=weight)) if has_two_classes else float("nan"),
        "brier_score": brier,
        "brier_skill": 1.0 - brier / reference_brier if reference_brier > 0 else float("nan"),
        "log_loss": float(log_loss(label, score, sample_weight=weight, labels=[0, 1])),
        "ece": ece,
        "maximum_calibration_gap": maximum_gap,
        "populated_calibration_bins": populated_bins,
    }


def trajectory_probability_rows(
    model_name: str,
    eta: np.ndarray,
    target: np.ndarray,
    risk_mask: np.ndarray,
    sample_weight: np.ndarray,
    step_minutes: int,
    calibration: str,
    eta_shift: float = 0.0,
    eta_slope: float = 1.0,
) -> list[dict[str, Any]]:
    eta_values = np.asarray(eta)
    if eta_values.ndim != 2:
        raise ValueError("eta must be a two-dimensional trajectory matrix")
    rows: list[dict[str, Any]] = []
    specifications = [("step", 1), ("1h", int(60 / step_minutes)), ("3h", int(180 / step_minutes)), ("6h", int(360 / step_minutes))]
    requested_steps = {steps for _, steps in specifications}
    if max(requested_steps) > eta_values.shape[1]:
        raise ValueError("Requested probability horizon exceeds the trajectory width")
    survival = np.ones(eta_values.shape[0], dtype=np.float64)
    probability_by_step: dict[int, np.ndarray] = {}
    for step in range(1, max(requested_steps) + 1):
        column = eta_shift + eta_slope * np.asarray(
            eta_values[:, step - 1], dtype=np.float64
        )
        rate = np.exp(np.minimum(column, 15.0))
        hazard = np.clip(-np.expm1(-rate), 0.0, 1.0 - 1e-7)
        survival *= 1.0 - hazard
        if step in requested_steps:
            probability_by_step[step] = 1.0 - survival.copy()
    for horizon, steps in specifications:
        if horizon == "step":
            label = target[:, 0].astype(np.int8)
            observed = risk_mask[:, 0] > 0.5
            probability = probability_by_step[1]
        else:
            event = target[:, :steps].max(axis=1) > 0.5
            observed = event | (risk_mask[:, steps - 1] > 0.5)
            label = event.astype(np.int8)
            probability = probability_by_step[steps]
        rows.append(
            weighted_probability_metric_row(
                model_name,
                calibration,
                horizon,
                label[observed],
                probability[observed],
                sample_weight[observed],
            )
        )
    return rows


def trajectory_probability_matrices(
    eta: np.ndarray,
    eta_shift: float = 0.0,
    eta_slope: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coherent per-step hazard and cumulative-risk matrices."""

    eta_values = np.asarray(eta, dtype=np.float64)
    if eta_values.ndim != 2:
        raise ValueError("eta must be a two-dimensional trajectory matrix")
    calibrated = np.minimum(eta_shift + eta_slope * eta_values, 15.0)
    hazard = np.clip(-np.expm1(-np.exp(calibrated)), 0.0, 1.0 - 1e-7)
    cumulative = 1.0 - np.cumprod(1.0 - hazard, axis=1)
    if not np.isfinite(hazard).all() or not np.isfinite(cumulative).all():
        raise ValueError("Hazard trajectory contains NaN or infinity")
    if np.any(np.diff(cumulative, axis=1) < -1e-12):
        raise ValueError("Cumulative risk must be monotonically non-decreasing")
    return hazard, cumulative


def trajectory_probability_curve_rows(
    model_name: str,
    eta: np.ndarray,
    target: np.ndarray,
    risk_mask: np.ndarray,
    sample_weight: np.ndarray,
    step_minutes: int,
    calibration: str,
    eta_shift: float = 0.0,
    eta_slope: float = 1.0,
) -> list[dict[str, Any]]:
    """Evaluate every cumulative horizon while respecting event/censor masking."""

    _, cumulative = trajectory_probability_matrices(eta, eta_shift, eta_slope)
    rows: list[dict[str, Any]] = []
    for step in range(1, cumulative.shape[1] + 1):
        event = np.asarray(target[:, :step]).max(axis=1) > 0.5
        observed = event | (np.asarray(risk_mask[:, step - 1]) > 0.5)
        row = weighted_probability_metric_row(
            model_name,
            calibration,
            f"k{step:02d}",
            event[observed].astype(np.int8),
            cumulative[observed, step - 1],
            np.asarray(sample_weight)[observed],
        )
        row["horizon_step"] = step
        row["horizon_minutes"] = step * int(step_minutes)
        rows.append(row)
    return rows


def cumulative_probability_at_step(
    eta: np.ndarray,
    steps: int,
    eta_shift: float = 0.0,
    eta_slope: float = 1.0,
) -> np.ndarray:
    eta_values = np.asarray(eta)
    if eta_values.ndim != 2 or not 1 <= int(steps) <= eta_values.shape[1]:
        raise ValueError("steps must address a valid two-dimensional trajectory horizon")
    survival = np.ones(eta_values.shape[0], dtype=np.float64)
    for step in range(int(steps)):
        column = eta_shift + eta_slope * np.asarray(eta_values[:, step], dtype=np.float64)
        rate = np.exp(np.minimum(column, 15.0))
        hazard = np.clip(-np.expm1(-rate), 0.0, 1.0 - 1e-7)
        survival *= 1.0 - hazard
    return 1.0 - survival


def main_model_comparison_rows(
    baseline_root: Path,
    arrays: dict[str, np.ndarray],
    step_minutes: int,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    bundle = json.loads((baseline_root / "model_bundle.json").read_text(encoding="utf-8"))
    model = artifact_value(bundle["models"], MAIN_MODEL_NAME)
    calibrator = artifact_value(bundle["calibrators"], MAIN_MODEL_NAME)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    current = np.asarray(arrays["current_features"], dtype=np.float64)
    baseline_transformer = bundle.get("feature_transformer", {})
    baseline_features = list(
        baseline_transformer.get(
            "feature_names",
            [
                *baseline_transformer.get("continuous_features", []),
                *baseline_transformer.get("binary_features", []),
            ],
        )
    )
    if feature_names is not None and baseline_features:
        missing = sorted(set(baseline_features) - set(feature_names))
        if missing:
            raise ValueError(f"Deep cache is missing local weather hazard baseline features: {missing}")
        current = current[:, [feature_names.index(name) for name in baseline_features]]
    if current.shape[1] != coefficients.size:
        raise ValueError(
            f"local weather hazard baseline expects {coefficients.size} features but received {current.shape[1]}"
        )
    one_step_eta = float(model["intercept"]) + current @ coefficients
    raw_eta = np.repeat(one_step_eta[:, None], arrays["target"].shape[1], axis=1)
    rows = trajectory_probability_rows(
        "local_weather_hazard", raw_eta, arrays["target"], arrays["risk_mask"], arrays["sample_weight"], step_minutes, "raw"
    )
    rows.extend(
        trajectory_probability_rows(
            "local_weather_hazard",
            raw_eta,
            arrays["target"],
            arrays["risk_mask"],
            arrays["sample_weight"],
            step_minutes,
            "calibrated",
            eta_shift=float(calibrator["log_rate_shift"]),
            eta_slope=float(calibrator["slope"]),
        )
    )
    for row in rows:
        row.update({"seed": "locked", "parameters": int(coefficients.size + 1), "best_epoch": "locked"})
    return rows


def _device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_deep_checkpoint(path: str | Path, device_name: str = "cpu") -> tuple[DeepHazardModel, dict[str, Any]]:
    device = _device_from_name(device_name)
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model = DeepHazardModel.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train_one_model(
    model_name: str,
    seed: int,
    config: dict[str, Any],
    output_root: Path,
    device: torch.device,
    maximum_epochs: int,
    maximum_train_samples: int | None,
    maximum_validation_samples: int | None,
    train_split: str = "train",
    validation_split: str = "validation",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache_root = resolve_project_path(config["cache_root"])
    training = config["training"]
    batch_size = int(training["batch_size"])
    early_limit = int(training["early_stopping_validation_samples"])
    metric_limit = int(training["development_metric_samples"])
    if maximum_validation_samples is not None:
        early_limit = min(early_limit, maximum_validation_samples)
        metric_limit = min(metric_limit, maximum_validation_samples)
    variant = dict(config.get("protocol_model_variants", {}).get(model_name, {}))
    feature_names = json.loads((cache_root / "sample_contract.json").read_text(encoding="utf-8"))["feature_names"]
    permuted_indices = [feature_names.index(name) for name in variant.get("permuted_features", [])]
    source_options = {
        "sampling_strategy": str(variant.get("negative_sampling", "stratified")),
        "permuted_feature_indices": permuted_indices,
        "permutation_block_days": int(variant.get("permutation_block_days", 7)),
    }
    train_source = DeepCacheBatchSource(
        cache_root, train_split, maximum_train_samples, seed, **source_options
    )
    early_source = DeepCacheBatchSource(
        cache_root, validation_split, early_limit, int(config["sampling"]["seed"]) + 101,
        **source_options,
    )
    metric_source = DeepCacheBatchSource(
        cache_root, validation_split, metric_limit, int(config["sampling"]["seed"]) + 202,
        **source_options,
    )
    feature_count = int(train_source.contract["feature_count"])
    horizon_steps = int(train_source.contract["horizon_steps"])
    set_reproducible_seed(seed)
    model = build_model(
        model_name,
        feature_count,
        horizon_steps,
        config,
        feature_names=list(train_source.contract["feature_names"]),
    ).to(device)
    microbatch_size = _training_microbatch_size(model, device, batch_size)
    if microbatch_size < batch_size:
        print(
            f"{model_name} gradient accumulation: microbatch_size={microbatch_size} "
            f"effective_batch_size={batch_size}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    patience = int(training["early_stopping_patience"])
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        training_summary = train_epoch(
            model,
            train_source,
            optimizer,
            device,
            batch_size,
            seed + epoch * 1009,
            float(training["gradient_clip_norm"]),
            float(training["hard_negative_loss_multiplier"]),
            return_details=True,
        )
        validation_summary = evaluate_model_nll(
            model, early_source, device, batch_size, return_details=True
        )
        training_loss = float(training_summary["nll"])
        validation_loss = float(validation_summary["nll"])
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1e-8
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history_rows.append(
            {
                "model_name": model_name,
                "seed": seed,
                "epoch": epoch,
                "training_nll": training_loss,
                "early_stopping_nll": validation_loss,
                "training_effective_risk_bins": training_summary["effective_risk_bins"],
                "training_positive_risk_bins": training_summary["positive_risk_bins"],
                "training_positive_bin_rate": training_summary["positive_bin_rate"],
                "development_nll_numerator": validation_summary["nll_numerator"],
                "development_effective_risk_bins": validation_summary["effective_risk_bins"],
                "development_positive_risk_bins": validation_summary["positive_risk_bins"],
                "development_positive_bin_rate": validation_summary["positive_bin_rate"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_so_far": int(improved),
            }
        )
        print(
            f"{model_name} seed={seed} epoch={epoch}/{maximum_epochs} "
            f"train_nll={training_loss:.8f} development_nll={validation_loss:.8f} "
            f"train_pos_rate={training_summary['positive_bin_rate']:.8g} "
            f"development_pos_rate={validation_summary['positive_bin_rate']:.8g} "
            f"development_risk_bins={validation_summary['effective_risk_bins']:.0f}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Deep model did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    arrays = collect_trajectory_arrays(model, metric_source, device, batch_size)
    calibrator = WeightedTrajectoryCalibrator().fit(
        arrays["eta"], arrays["target"], arrays["risk_mask"], arrays["sample_weight"]
    )
    deep_rows = trajectory_probability_rows(
        f"{model_name}_hazard",
        arrays["eta"],
        arrays["target"],
        arrays["risk_mask"],
        arrays["sample_weight"],
        int(config["step_minutes"]),
        "raw",
    )
    deep_rows.extend(
        trajectory_probability_rows(
            f"{model_name}_hazard",
            arrays["eta"],
            arrays["target"],
            arrays["risk_mask"],
            arrays["sample_weight"],
            int(config["step_minutes"]),
            "calibrated",
            eta_shift=calibrator.log_rate_shift,
            eta_slope=calibrator.slope,
        )
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    for row in deep_rows:
        row.update({"seed": seed, "parameters": parameter_count, "best_epoch": best_epoch})
    checkpoint = {
        "model_family": "Deep RIG-Hazard local temporal encoder",
        "model_name": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_early_stopping_nll": best_loss,
        "model_config": model.to_config(),
        "state_dict": best_state,
        "trajectory_calibrator": calibrator.to_dict(),
        "sample_contract": train_source.contract,
        "training_selection": train_source.selection_summary,
        "early_stopping_selection": early_source.selection_summary,
        "development_metric_selection": metric_source.selection_summary,
        "train_split": train_split,
        "validation_split": validation_split,
    }
    checkpoint_path = output_root / "checkpoints" / f"{model_name}_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        "model_name": model_name,
        "seed": seed,
        "checkpoint": str(checkpoint_path.relative_to(output_root)),
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_early_stopping_nll": best_loss,
        "calibrator": calibrator.to_dict(),
        "training_selection": train_source.selection_summary,
        "early_stopping_selection": early_source.selection_summary,
        "development_metric_selection": metric_source.selection_summary,
        "train_split": train_split,
        "validation_split": validation_split,
    }
    return deep_rows, history_rows, {"metadata": metadata, "arrays": arrays}


def build_deep_local_report(
    run_tier: str,
    device: torch.device,
    metrics: pd.DataFrame,
    metadata: list[dict[str, Any]],
) -> str:
    model_names = sorted({str(item["model_name"]) for item in metadata})
    display_names = {
        "gru": "GRU",
        "tcn": "因果TCN",
        "patchtst": "PatchTST",
        "timesnet": "TimesNet",
        "itransformer": "iTransformer",
    }
    model_description = "、".join(display_names.get(name, name) for name in model_names)
    selected = metrics.loc[
        metrics["horizon"].eq("6h") & metrics["calibration"].eq("calibrated"),
        ["model_name", "seed", "pr_auc", "brier_score", "log_loss", "ece", "roc_auc"],
    ].copy()
    return "\n".join(
        [
            "# Deep RIG-Hazard非图时序基线报告",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"运行层级：{run_tier}",
            f"计算设备：{device}",
            "",
            "## 模型口径",
            "",
            f"- 本次编码器为{model_description}，均只读取本站过去24小时动态特征，不读取邻站信息。",
            "- 输出未来36个10分钟区间的条件离散hazard；事件发生或风险集退出后停止贡献似然。",
            "- 所有区间hazard经cloglog映射后累计，因此1/3/6小时累计风险天然单调。",
            "- 校准只使用一个共享的log-rate平移与斜率，不破坏多步hazard的一致性。",
            "- local weather hazard在完全相同的开发样本上重算，避免24小时历史完整性筛选造成不公平比较。",
            "",
            "## 6小时开发集结果",
            "",
            selected.to_markdown(index=False) if not selected.empty else "无。",
            "",
            "## 解释边界",
            "",
            "该阶段只用于2022训练、2023开发选择。2024已被查看，不用于结构或超参数选择；固定月误报预算下的事件提前量需要在后续完整时序预测阶段评估。",
            "",
            f"已保存深度检查点：{len(metadata)}个。",
        ]
    ) + "\n"


def run_deep_local_experiment(
    config: dict[str, Any],
    config_path: Path,
    overwrite: bool = False,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    maximum_epochs: int | None = None,
    maximum_train_samples: int | None = None,
    maximum_validation_samples: int | None = None,
    output_root_override: str | None = None,
    device_name: str = "auto",
    resume: bool = False,
    train_split: str = "train",
    validation_split: str = "validation",
) -> Path:
    output_root = resolve_project_path(output_root_override or config["local_output_root"])
    if resume:
        if overwrite:
            raise ValueError("--resume and --overwrite cannot be used together")
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        prepare_output_root(output_root, overwrite)
    selected_models = models or [
        str(value) for value in config.get("default_local_models", ["gru", "tcn"])
    ]
    supported_models = {
        "gru", "gru_subset", "tcn", "patchtst", "timesnet", "itransformer", "recurrent_dual",
        "weather_fast", "weather_slow", "weather_dual", "weather_dual_rec",
        "weather_dual_rec_gate",
    }
    unsupported = sorted(
        name
        for name in set(selected_models)
        if name not in supported_models
        and str(config.get("protocol_model_variants", {}).get(name, {}).get("encoder", ""))
        not in supported_models
    )
    if unsupported:
        raise ValueError(f"Unsupported deep local models: {unsupported}")
    selected_seeds = seeds or [int(value) for value in config["training"]["seeds"]]
    epochs = int(maximum_epochs or config["training"]["maximum_epochs"])
    torch.set_num_threads(max(int(config["training"].get("torch_num_threads", 1)), 1))
    device = _device_from_name(device_name)
    run_tier = (
        "local deep-model development run"
        if maximum_train_samples is None and maximum_validation_samples is None and len(selected_seeds) >= 5
        else "implementation check"
    )
    write_json(
        output_root / "resolved_config.json",
        {
            **config,
            "config_path": str(config_path),
            "run_overrides": {
                "models": selected_models,
                "seeds": selected_seeds,
                "maximum_epochs": epochs,
                "maximum_train_samples": maximum_train_samples,
                "maximum_validation_samples": maximum_validation_samples,
                "output_root": str(output_root),
                "device": str(device),
                "run_tier": run_tier,
                "train_split": train_split,
                "validation_split": validation_split,
            },
        },
    )
    metric_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    baseline_added = False
    baseline_root = resolve_project_path(config["baseline_root"])
    for model_name in selected_models:
        for seed in selected_seeds:
            completed_root = output_root / "completed_runs" / f"{model_name}_seed_{seed}"
            completed_metrics_path = completed_root / "metrics.csv"
            completed_history_path = completed_root / "training_history.csv"
            completed_metadata_path = completed_root / "metadata.json"
            if resume and all(
                path.exists()
                for path in [completed_metrics_path, completed_history_path, completed_metadata_path]
            ):
                completed_metrics = pd.read_csv(completed_metrics_path, encoding="utf-8-sig", low_memory=False)
                completed_history = pd.read_csv(completed_history_path, encoding="utf-8-sig", low_memory=False)
                completed_metadata = json.loads(completed_metadata_path.read_text(encoding="utf-8"))
                deep_completed = completed_metrics.loc[completed_metrics["model_name"].ne("local_weather_hazard")]
                metric_rows.extend(deep_completed.to_dict(orient="records"))
                history_rows.extend(completed_history.to_dict(orient="records"))
                metadata.append(completed_metadata)
                if not baseline_added:
                    baseline_completed = completed_metrics.loc[completed_metrics["model_name"].eq("local_weather_hazard")]
                    metric_rows.extend(baseline_completed.to_dict(orient="records"))
                    baseline_added = not baseline_completed.empty
                print(f"Reused completed run: {model_name} seed={seed}", flush=True)
                continue
            deep_rows, model_history, result = train_one_model(
                model_name,
                int(seed),
                config,
                output_root,
                device,
                epochs,
                maximum_train_samples,
                maximum_validation_samples,
                train_split,
                validation_split,
            )
            metric_rows.extend(deep_rows)
            history_rows.extend(model_history)
            metadata.append(result["metadata"])
            baseline_rows = main_model_comparison_rows(
                baseline_root,
                result["arrays"],
                int(config["step_minutes"]),
                feature_names=list(
                    json.loads((resolve_project_path(config["cache_root"]) / "sample_contract.json").read_text(encoding="utf-8"))[
                        "feature_names"
                    ]
                ),
            )
            if not baseline_added:
                metric_rows.extend(baseline_rows)
                baseline_added = True
            completed_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([*deep_rows, *baseline_rows]).to_csv(
                completed_metrics_path, index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(model_history).to_csv(completed_history_path, index=False, encoding="utf-8-sig")
            write_json(completed_metadata_path, result["metadata"])
            del result
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_root / "deep_local_baselines.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(output_root / "training_history.csv", index=False, encoding="utf-8-sig")
    numeric_metrics = ["pr_auc", "roc_auc", "brier_score", "brier_skill", "log_loss", "ece"]
    deep_only = metrics.loc[metrics["seed"].ne("locked")].copy()
    if not deep_only.empty:
        summary = (
            deep_only.groupby(["model_name", "calibration", "horizon"], as_index=False)[numeric_metrics]
            .agg(["mean", "std"])
        )
        summary.columns = ["_".join(value).rstrip("_") for value in summary.columns.to_flat_index()]
        summary.to_csv(output_root / "deep_local_summary.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_root / "run_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_tier": run_tier,
            "device": str(device),
            "torch_version": torch.__version__,
            "models": metadata,
            "resumed": resume,
        },
    )
    report = build_deep_local_report(run_tier, device, metrics, metadata)
    (output_root / "deep_local_report.md").write_text(report, encoding="utf-8")
    print(f"Deep local hazard experiment complete: {output_root}", flush=True)
    return output_root

"""Validation and scoring utilities for discrete recurrent-event trajectories.

The functions in this module deliberately keep the probability model separate
from the alert controller.  They operate only on frozen model outputs and do
not fit or recalibrate a model on a locked evaluation year.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .deep_training import weighted_probability_metric_row
from .naming import MAIN_MODEL_NAME


TRAJECTORY_CONTRACT_VERSION = "local-weather-hazard-36-step"


@dataclass(frozen=True)
class TrajectoryAudit:
    rows: int
    horizon_steps: int
    step_minutes: int
    missing_values: int
    out_of_range_hazards: int
    out_of_range_cumulative: int
    nonmonotone_cumulative_rows: int
    incoherent_cumulative_rows: int
    duplicate_origin_ids: int
    duplicate_entity_times: int
    issue_time_mismatches: int

    @property
    def passed(self) -> bool:
        return all(
            value == 0
            for value in (
                self.missing_values,
                self.out_of_range_hazards,
                self.out_of_range_cumulative,
                self.nonmonotone_cumulative_rows,
                self.incoherent_cumulative_rows,
                self.duplicate_origin_ids,
                self.duplicate_entity_times,
                self.issue_time_mismatches,
            )
        ) and self.horizon_steps == 36 and self.step_minutes == 10

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "passed": int(self.passed),
            "trajectory_contract_version": TRAJECTORY_CONTRACT_VERSION,
        }


def first_event_probability(hazard: np.ndarray) -> np.ndarray:
    """Convert conditional hazards to first-event probabilities.

    For step k this returns h[k] * product(1-h[j], j<k).  The row sum is the
    cumulative incidence over the available prediction horizon.
    """

    values = np.asarray(hazard, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("hazard must be a two-dimensional matrix")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("hazard values must be finite and lie in [0, 1]")
    survival_before = np.concatenate(
        [
            np.ones((values.shape[0], 1), dtype=np.float64),
            np.cumprod(1.0 - values[:, :-1], axis=1),
        ],
        axis=1,
    )
    probability = values * survival_before
    if np.any(probability < -1e-12) or np.any(probability.sum(axis=1) > 1.0 + 1e-9):
        raise ValueError("invalid first-event probability trajectory")
    return probability


def cumulative_from_hazard(hazard: np.ndarray) -> np.ndarray:
    values = np.asarray(hazard, dtype=np.float64)
    return 1.0 - np.cumprod(1.0 - values, axis=1)


def lead_weights(
    horizon_steps: int,
    step_minutes: int = 10,
    kind: str = "linear",
) -> np.ndarray:
    leads = np.arange(1, int(horizon_steps) + 1, dtype=np.float64) * (
        float(step_minutes) / 60.0
    )
    if kind == "linear":
        return leads
    if kind == "hit_only":
        return np.ones_like(leads)
    if kind == "saturating_3h":
        return np.minimum(leads, 3.0)
    if kind == "business_piecewise":
        # Continuous, monotone utility with diminishing returns after 4 h.
        return np.piecewise(
            leads,
            [leads < 0.5, (leads >= 0.5) & (leads < 2.0), (leads >= 2.0) & (leads < 4.0), leads >= 4.0],
            [lambda x: 0.25 * x / 0.5, lambda x: 0.25 + 0.5 * (x - 0.5) / 1.5,
             lambda x: 0.75 + 0.25 * (x - 2.0) / 2.0, 1.0],
        )
    raise ValueError(f"Unsupported lead utility: {kind}")


def trajectory_value(
    hazard: np.ndarray,
    step_minutes: int = 10,
    utility: str = "linear",
    horizon_steps: int | None = None,
) -> np.ndarray:
    values = np.asarray(hazard)
    width = values.shape[1] if horizon_steps is None else int(horizon_steps)
    if width < 1 or width > values.shape[1]:
        raise ValueError("horizon_steps is outside the available trajectory")
    probability = first_event_probability(values[:, :width])
    return probability @ lead_weights(width, step_minutes, utility)


def audit_trajectory_arrays(
    payload: dict[str, np.ndarray],
    step_minutes: int = 10,
    entity: Iterable[str] | None = None,
    expected_issue_time_ns: np.ndarray | None = None,
    tolerance: float = 2e-6,
) -> TrajectoryAudit:
    required = {"h_trajectory", "F_trajectory", "issue_time_ns"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Trajectory payload is missing fields: {sorted(missing)}")
    hazard = np.asarray(payload["h_trajectory"], dtype=np.float64)
    cumulative = np.asarray(payload["F_trajectory"], dtype=np.float64)
    if hazard.ndim != 2 or cumulative.shape != hazard.shape:
        raise ValueError("h_trajectory and F_trajectory must have the same 2-D shape")
    rows, width = hazard.shape
    if np.asarray(payload["issue_time_ns"]).shape != (rows,):
        raise ValueError("issue_time_ns must contain one value per trajectory")
    nonfinite = int((~np.isfinite(hazard)).sum() + (~np.isfinite(cumulative)).sum())
    computed = cumulative_from_hazard(np.clip(hazard, 0.0, 1.0))
    incoherent = int(np.any(np.abs(computed - cumulative) > tolerance, axis=1).sum())
    duplicate_origin = 0
    if "origin_id" in payload:
        duplicate_origin = int(pd.Series(payload["origin_id"]).duplicated().sum())
    duplicate_entity_times = 0
    if entity is not None:
        keys = pd.DataFrame(
            {
                "entity": np.asarray(list(entity), dtype=str),
                "issue_time_ns": np.asarray(payload["issue_time_ns"], dtype=np.int64),
            }
        )
        if keys.shape[0] != rows:
            raise ValueError("entity must contain one value per trajectory")
        duplicate_entity_times = int(keys.duplicated().sum())
    issue_mismatch = 0
    if expected_issue_time_ns is not None:
        expected = np.asarray(expected_issue_time_ns, dtype=np.int64)
        if expected.shape != (rows,):
            raise ValueError("expected_issue_time_ns must contain one value per trajectory")
        issue_mismatch = int(
            (np.asarray(payload["issue_time_ns"], dtype=np.int64) != expected).sum()
        )
    return TrajectoryAudit(
        rows=rows,
        horizon_steps=width,
        step_minutes=int(step_minutes),
        missing_values=nonfinite,
        out_of_range_hazards=int(((hazard < 0.0) | (hazard > 1.0)).sum()),
        out_of_range_cumulative=int(((cumulative < 0.0) | (cumulative > 1.0)).sum()),
        nonmonotone_cumulative_rows=int((np.diff(cumulative, axis=1) < -tolerance).any(axis=1).sum()),
        incoherent_cumulative_rows=incoherent,
        duplicate_origin_ids=duplicate_origin,
        duplicate_entity_times=duplicate_entity_times,
        issue_time_mismatches=issue_mismatch,
    )


def load_trajectory_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def multihorizon_probability_metrics(
    payload: dict[str, np.ndarray],
    split: str,
    step_minutes: int = 10,
    horizons_minutes: Iterable[int] = (30, 60, 180, 360),
) -> pd.DataFrame:
    required = {"F_trajectory", "y_hazard", "censor_mask"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Cannot evaluate trajectories; missing {sorted(missing)}")
    cumulative = np.asarray(payload["F_trajectory"], dtype=np.float64)
    target = np.asarray(payload["y_hazard"], dtype=np.int8)
    mask = np.asarray(payload["censor_mask"], dtype=np.int8)
    weight = np.asarray(
        payload.get("sample_weight", np.ones(cumulative.shape[0])), dtype=np.float64
    )
    if target.shape != cumulative.shape or mask.shape != cumulative.shape:
        raise ValueError("Probability, target, and censor matrices must align")
    rows: list[dict[str, Any]] = []
    for minutes in horizons_minutes:
        if int(minutes) % int(step_minutes):
            raise ValueError("Every horizon must be divisible by step_minutes")
        steps = int(minutes) // int(step_minutes)
        if steps < 1 or steps > cumulative.shape[1]:
            raise ValueError(f"Horizon {minutes} minutes is unavailable")
        event = target[:, :steps].max(axis=1) > 0
        observed = event | (mask[:, steps - 1] > 0)
        row = weighted_probability_metric_row(
            MAIN_MODEL_NAME,
            "frozen_2022_oof_calibration",
            f"{minutes}m",
            event[observed].astype(np.int8),
            cumulative[observed, steps - 1],
            weight[observed],
        )
        row.update(
            {
                "split": split,
                "horizon_minutes": int(minutes),
                "horizon_steps": steps,
                "trajectory_contract_version": TRAJECTORY_CONTRACT_VERSION,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

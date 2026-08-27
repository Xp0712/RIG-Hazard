from __future__ import annotations

import unittest

import numpy as np

from rig_hazard.risk_trajectory import (
    audit_trajectory_arrays,
    cumulative_from_hazard,
    first_event_probability,
    multihorizon_probability_metrics,
    trajectory_value,
)


def _payload(rows: int = 8) -> dict[str, np.ndarray]:
    hazard = np.full((rows, 36), 0.01, dtype=np.float32)
    cumulative = cumulative_from_hazard(hazard).astype(np.float32)
    target = np.zeros((rows, 36), dtype=np.int8)
    target[::2, 2] = 1
    mask = np.ones((rows, 36), dtype=np.int8)
    return {
        "origin_id": np.arange(rows, dtype=np.int64),
        "issue_time_ns": np.arange(rows, dtype=np.int64) * 600_000_000_000,
        "h_trajectory": hazard,
        "F_trajectory": cumulative,
        "y_hazard": target,
        "censor_mask": mask,
        "sample_weight": np.ones(rows),
    }


class RiskTrajectoryTests(unittest.TestCase):
    def test_first_event_probability_matches_cumulative_incidence(self) -> None:
        payload = _payload()
        probability = first_event_probability(payload["h_trajectory"])
        self.assertTrue(
            np.allclose(probability.sum(axis=1), payload["F_trajectory"][:, -1])
        )
        linear = trajectory_value(payload["h_trajectory"], utility="linear")
        hit_only = trajectory_value(payload["h_trajectory"], utility="hit_only")
        self.assertTrue(np.all(linear > 0))
        self.assertTrue(np.allclose(hit_only, payload["F_trajectory"][:, -1]))

    def test_trajectory_audit_and_multihorizon_metrics(self) -> None:
        payload = _payload()
        audit = audit_trajectory_arrays(payload, step_minutes=10)
        self.assertTrue(audit.passed)
        metrics = multihorizon_probability_metrics(payload, "2022_oof")
        self.assertEqual(metrics["horizon_minutes"].tolist(), [30, 60, 180, 360])
        self.assertTrue(
            {"pr_auc", "log_loss", "brier_score", "ece", "prevalence"}.issubset(
                metrics.columns
            )
        )

    def test_trajectory_audit_detects_nonmonotone_and_duplicate_rows(self) -> None:
        payload = _payload(2)
        payload["origin_id"][1] = payload["origin_id"][0]
        payload["F_trajectory"][0, 5] = 0.0
        audit = audit_trajectory_arrays(payload)
        self.assertEqual(audit.duplicate_origin_ids, 1)
        self.assertEqual(audit.nonmonotone_cumulative_rows, 1)
        self.assertFalse(audit.passed)

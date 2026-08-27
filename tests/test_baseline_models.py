from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rig_hazard.baseline_models import HazardRateCalibrator, PlattCalibrator, WeightedBinaryGLM, cloglog_probability
from rig_hazard.baseline_experiment import evaluate_warning_model, observed_warning_rows


class BaselineModelTests(unittest.TestCase):
    def test_observed_warning_rows_excludes_censored_horizons(self) -> None:
        frame = pd.DataFrame(
            {
                "value": [1, 2, 3, 4],
                "observed_6h": [1, 0, np.nan, 1],
            }
        )

        observed = observed_warning_rows(frame, "observed_6h")

        self.assertEqual(observed["value"].tolist(), [1, 4])
        with self.assertRaises(KeyError):
            observed_warning_rows(frame, "missing")

    def test_cloglog_horizon_accumulation_is_coherent(self) -> None:
        eta = np.array([-8.0, -4.0, -1.0])
        one_step = cloglog_probability(eta, steps=1)
        six_steps = cloglog_probability(eta, steps=6)
        np.testing.assert_allclose(six_steps, 1.0 - np.power(1.0 - one_step, 6), rtol=1e-10, atol=1e-12)
        self.assertTrue(np.all(six_steps >= one_step))

    def test_weighted_glms_learn_ordered_risk(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(800, 2))
        latent = -3.0 + 1.5 * x[:, 0] - 0.8 * x[:, 1]
        probability = cloglog_probability(latent)
        y = (rng.random(x.shape[0]) < probability).astype(np.int8)
        model = WeightedBinaryGLM("cloglog", l2=1e-4, max_iter=150).fit(x, y)
        self.assertTrue(model.converged)
        learned_score = model.decision_function(x)
        self.assertGreater(np.corrcoef(latent, learned_score)[0, 1], 0.9)

    def test_feature_specific_penalty_round_trip(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=(300, 2))
        y = (x[:, 0] + rng.normal(scale=0.2, size=300) > 1.0).astype(np.int8)
        model = WeightedBinaryGLM(
            "cloglog",
            l2=0.1,
            max_iter=100,
            penalty_weights=np.asarray([1.0, 10.0]),
        ).fit(x, y)
        restored = WeightedBinaryGLM.from_dict(model.to_dict())

        np.testing.assert_allclose(model.predict_probability(x), restored.predict_probability(x))
        np.testing.assert_array_equal(restored.penalty_weights, [1.0, 10.0])

    def test_calibrators_return_finite_probabilities(self) -> None:
        score = np.linspace(-10, 2, 500)
        y = np.zeros(500, dtype=np.int8)
        y[-20:] = 1
        hazard = HazardRateCalibrator().fit(score, y)
        platt = PlattCalibrator().fit(score, y)
        self.assertTrue(np.isfinite(hazard.predict(score)).all())
        self.assertTrue(np.isfinite(platt.predict(score)).all())

    def test_effective_lead_uses_final_persistent_alarm_episode(self) -> None:
        issue_time = pd.date_range("2024-01-01 00:00:00", periods=6, freq="10min")
        frame = pd.DataFrame(
            {
                "station_code": ["F0001"] * 6,
                "issue_time": issue_time,
                "station_month": ["F0001|2024-01"] * 6,
                "onset_within_1h": [1] * 6,
                "hard_negative_1h": [0] * 6,
                "score": [0.0, 0.0, 0.8, 0.8, 0.8, 0.8],
            }
        )
        events = pd.DataFrame(
            {
                "station_code": ["F0001"],
                "onset_time": [pd.Timestamp("2024-01-01 01:00:00")],
                "valid_target_event": [1],
            }
        )
        result = evaluate_warning_model(
            frame,
            events,
            "score",
            0.5,
            horizon=1,
            step_minutes=10,
            false_alarm_budget=10.0,
            operating_point="test",
            minimum_consecutive_alarm_bins=2,
            alarm_merge_gap_minutes=20,
            maximum_silence_before_event_minutes=30,
        )
        self.assertEqual(result["hit_events"], 1)
        self.assertAlmostEqual(result["mean_effective_lead_hours"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()

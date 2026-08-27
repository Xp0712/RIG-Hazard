from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_public_recurrence_benchmarks import (
    _constant_hazard_from_risk,
    _fixed_threshold_parameters_for_guard,
    _grid,
)


class PublicRecurrenceBenchmarkGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.selection = pd.DataFrame({"risk_score": [0.1, 0.2, 0.4, 0.8]})
        self.common = {
            "trailing_history_days": 30,
            "minimum_history_rows": 24,
        }

    def test_split_absolute_and_rolling_quantiles(self) -> None:
        settings = {
            **self.common,
            "absolute_quantiles": [0.8, 0.9],
            "rolling_quantiles": [0.94],
        }

        grid = _grid(self.selection, settings)
        counts: dict[str, int] = {}
        for strategy, _ in grid:
            counts[strategy] = counts.get(strategy, 0) + 1

        self.assertEqual(counts["fixed_threshold"], 2)
        self.assertEqual(counts["hysteresis"], 2)
        self.assertEqual(counts["budget_safe"], 2)
        self.assertEqual(counts["rolling_quantile"], 1)
        self.assertEqual(counts["original_simple"], 1)

    def test_legacy_quantiles_remain_supported(self) -> None:
        grid = _grid(self.selection, {**self.common, "quantiles": [0.9]})
        self.assertEqual({strategy for strategy, _ in grid}, {
            "fixed_threshold",
            "hysteresis",
            "budget_safe",
            "rolling_quantile",
            "original_simple",
        })

    def test_constant_hazard_reconstructs_public_horizon_risk(self) -> None:
        risk = np.asarray([0.0, 0.1, 0.8])
        hazard = _constant_hazard_from_risk(risk, 6)
        reconstructed = 1.0 - np.prod(1.0 - hazard, axis=1)
        self.assertTrue(np.allclose(reconstructed, risk, atol=1e-6))

    def test_hard_guard_can_use_an_unguarded_infeasible_fixed_candidate(self) -> None:
        candidates = pd.DataFrame(
            {
                "budget_hours_per_entity_month": [24.0, 24.0, 48.0],
                "strategy": ["fixed_threshold", "fixed_threshold", "fixed_threshold"],
                "lead_utility_hours": [2.0, 3.0, 4.0],
                "event_hit_rate": [0.4, 0.5, 0.6],
                "strict_false_alarm_hours_per_entity_month": [30.0, 36.0, 20.0],
                "budget_met_mean_fah": [0, 0, 1],
                "parameters": [
                    '{"threshold": 0.8}',
                    '{"threshold": 0.7}',
                    '{"threshold": 0.6}',
                ],
            }
        )

        parameters = _fixed_threshold_parameters_for_guard(candidates, 24.0)

        self.assertEqual(parameters, {"threshold": 0.7})


if __name__ == "__main__":
    unittest.main()

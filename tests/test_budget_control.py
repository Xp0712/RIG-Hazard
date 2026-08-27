import unittest

import numpy as np
import pandas as pd

from rig_hazard.budget_control import CausalBudgetConfig, causal_budget_alarms
from rig_hazard.stability_experiment import STABILITY_ETA_COLUMNS, causal_budget_guarantee_audit


class CausalBudgetControllerTests(unittest.TestCase):
    def config(self) -> CausalBudgetConfig:
        return CausalBudgetConfig(
            step_minutes=10,
            monthly_budget_hours=2.0,
            trailing_history_days=7,
            candidate_quantile=0.5,
            minimum_history_rows=3,
            burst_allowance_hours=0.5,
            minimum_candidate_run_bins=2,
            minimum_alarm_run_bins=2,
        )

    def test_monthly_alarm_duration_never_exceeds_budget(self) -> None:
        times = pd.date_range("2023-12-20", "2024-02-01", freq="10min", inclusive="left")
        scores = np.linspace(0, 1, times.size)
        result = causal_budget_alarms(times, scores, self.config())
        monthly = result.assign(month=result["issue_time"].dt.to_period("M")).groupby("month")["budget_alarm"].sum()
        self.assertTrue((monthly <= self.config().monthly_budget_bins).all())

    def test_prefix_predictions_do_not_change_when_future_is_appended(self) -> None:
        times = pd.date_range("2024-01-01", periods=300, freq="10min")
        scores = np.sin(np.arange(times.size) / 11.0) + np.arange(times.size) / 500.0
        prefix = causal_budget_alarms(times[:180], scores[:180], self.config())
        full = causal_budget_alarms(times, scores, self.config())
        np.testing.assert_array_equal(prefix["budget_alarm"], full.loc[:179, "budget_alarm"])
        np.testing.assert_allclose(
            prefix["causal_threshold"], full.loc[:179, "causal_threshold"], equal_nan=True
        )

    def test_monthly_budget_resets(self) -> None:
        times = pd.date_range("2024-01-20", "2024-03-01", freq="10min", inclusive="left")
        scores = np.arange(times.size, dtype=float)
        result = causal_budget_alarms(times, scores, self.config())
        result["month"] = result["issue_time"].dt.to_period("M")
        totals = result.groupby("month")["budget_alarm"].sum()
        self.assertGreater(totals.loc[pd.Period("2024-01")], 0)
        self.assertGreater(totals.loc[pd.Period("2024-02")], 0)

    def test_isolated_candidate_spikes_do_not_spend_budget(self) -> None:
        times = pd.date_range("2024-01-01", periods=200, freq="10min")
        scores = np.zeros(times.size)
        scores[20::20] = 10.0
        result = causal_budget_alarms(times, scores, self.config())
        self.assertEqual(int(result["budget_alarm"].sum()), 0)

    def test_hour_audit_does_not_overflow_int8_counts(self) -> None:
        frame = pd.DataFrame({"station_month": ["S1|2024-01"] * 60})
        for eta_column in STABILITY_ETA_COLUMNS.values():
            frame[f"{eta_column}__budget_alarm"] = np.ones(60, dtype=np.int8)
        audit = causal_budget_guarantee_audit({2024: frame}, step_minutes=10, budget_hours=10.0)
        np.testing.assert_allclose(audit["maximum_total_alarm_hours"], 10.0)
        self.assertTrue(audit["months_above_budget"].eq(0).all())


if __name__ == "__main__":
    unittest.main()

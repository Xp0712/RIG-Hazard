from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rig_hazard.alert_governance import (
    station_month_budget_distribution,
    strict_event_alert_evaluation,
)
from rig_hazard.dynamic_hard_budget import (
    DynamicBudgetConfig,
    apply_dynamic_hard_budget,
    budget_capacity_bins,
    nesting_audit,
    offline_score_oracle,
)


def _frame(periods: int, start: str = "2024-01-01 00:00") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_code": ["S1"] * periods,
            "issue_time": pd.date_range(start, periods=periods, freq="10min"),
            "observed_6h": np.ones(periods, dtype=np.int8),
        }
    )


def _hazard(rows: int, value: float = 0.05) -> np.ndarray:
    return np.full((rows, 36), value, dtype=np.float64)


def _always_alarm_config(**updates: object) -> DynamicBudgetConfig:
    values = dict(
        score_threshold=0.0,
        use_dynamic_price=False,
        deduplication_bins=1,
        couple_budgets=True,
    )
    values.update(updates)
    return DynamicBudgetConfig(**values)


class DynamicHardBudgetTests(unittest.TestCase):
    def test_zero_tiny_and_exact_one_bin_budgets(self) -> None:
        frame = _frame(4)
        controlled, monthly, _ = apply_dynamic_hard_budget(
            frame, _hazard(4), [0.0, 0.1, 1 / 6], _always_alarm_config()
        )
        self.assertEqual(int(controlled["budget_alarm_0h"].sum()), 0)
        self.assertEqual(int(controlled["budget_alarm_0.1h"].sum()), 0)
        self.assertEqual(int(controlled["budget_alarm_0.166667h"].sum()), 1)
        self.assertTrue(monthly["hard_budget_met"].eq(1).all())

    def test_coupled_alarm_sets_are_strictly_nested_and_hard_feasible(self) -> None:
        frame = _frame(200)
        budgets = [2, 5, 10, 20]
        controlled, monthly, _ = apply_dynamic_hard_budget(
            frame, _hazard(200), budgets, _always_alarm_config()
        )
        audit = nesting_audit(controlled, budgets)
        self.assertTrue(audit["nesting_violations"].eq(0).all())
        self.assertTrue(monthly["hard_budget_met"].eq(1).all())
        self.assertTrue((monthly["peak_occupied_bins"] <= monthly["capacity_bins"]).all())

    def test_event_releases_pending_capacity_only_at_onset(self) -> None:
        frame = _frame(2)
        events = pd.DataFrame(
            {"station_code": ["S1"], "onset_time": [frame.loc[1, "issue_time"]]}
        )
        controlled, monthly, trace = apply_dynamic_hard_budget(
            frame, _hazard(2), [1 / 6], _always_alarm_config(), events=events,
            trace_all_steps=True,
        )
        self.assertEqual(controlled["budget_alarm_0.166667h"].tolist(), [1, 1])
        self.assertEqual(
            int(trace.loc[trace["issue_time"].eq(frame.loc[0, "issue_time"]), "pending_bins"].iloc[0]),
            1,
        )
        self.assertEqual(int(monthly["released_true_bins"].iloc[0]), 1)
        self.assertEqual(int(monthly["peak_occupied_bins"].iloc[0]), 1)

    def test_one_event_releases_only_one_alarm_segment(self) -> None:
        frame = pd.DataFrame(
            {
                "station_code": ["S1", "S1", "S1"],
                "issue_time": pd.to_datetime(
                    ["2024-01-01 00:00", "2024-01-01 00:20", "2024-01-01 00:30"]
                ),
                "observed_6h": [1, 1, 1],
            }
        )
        events = pd.DataFrame(
            {"station_code": ["S1"], "onset_time": pd.to_datetime(["2024-01-01 00:30"])}
        )
        _, monthly, _ = apply_dynamic_hard_budget(
            frame,
            _hazard(3),
            [0.5],
            _always_alarm_config(deduplication_bins=2),
            events=events,
        )
        self.assertEqual(int(monthly["released_true_bins"].iloc[0]), 1)

    def test_online_release_and_strict_offline_reserve_share_event_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "station_code": "S1",
                "issue_time": pd.date_range("2024-01-01", periods=6, freq="h"),
                "observed_6h": np.ones(6, dtype=np.int8),
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "station_code": ["S1", "S1"],
                "onset_time": [frame.loc[2, "issue_time"], frame.loc[5, "issue_time"]],
                "valid_target_event": [1, 1],
            }
        )
        config = DynamicBudgetConfig(
            step_minutes=60,
            horizon_steps=2,
            score_threshold=0.0,
            use_dynamic_price=False,
            deduplication_bins=1,
            couple_budgets=False,
        )
        controlled, online_monthly, _ = apply_dynamic_hard_budget(
            frame,
            np.full((6, 2), 0.1, dtype=np.float64),
            [2.0],
            config,
            events=events,
        )
        _, _, strict_frame = strict_event_alert_evaluation(
            controlled,
            events,
            alarm_column="budget_alarm_2h",
            step_minutes=60,
            horizon_hours=2,
            observability_column="observed_6h",
        )
        offline_monthly, _ = station_month_budget_distribution(
            strict_frame,
            "budget_alarm_2h",
            budget_hours=2.0,
            step_minutes=60,
            strict_false_column="_strict_false_alarm",
            unsettled_column="_strict_unsettled_alarm",
            reserved_column="_strict_reserved_alarm",
        )

        self.assertTrue(online_monthly["hard_budget_met"].eq(1).all())
        self.assertLessEqual(float(offline_monthly["reserved_alarm_hours"].max()), 2.0)
        self.assertEqual(int(strict_frame["_strict_true_alarm"].sum()), 3)

    def test_prefix_decisions_do_not_depend_on_future_predictions_or_events(self) -> None:
        frame = _frame(20)
        hazard = _hazard(20, 0.02)
        config = _always_alarm_config(score_threshold=0.1, use_dynamic_price=True)
        first, _, _ = apply_dynamic_hard_budget(frame, hazard, [2, 5], config)
        changed = hazard.copy()
        changed[10:] = 0.9
        future_events = pd.DataFrame(
            {"station_code": ["S1"], "onset_time": [frame.loc[15, "issue_time"]]}
        )
        second, _, _ = apply_dynamic_hard_budget(
            frame, changed, [2, 5], config, events=future_events
        )
        for budget in [2, 5]:
            column = f"budget_alarm_{budget:g}h"
            self.assertEqual(first.loc[:9, column].tolist(), second.loc[:9, column].tolist())

    def test_cross_month_reservations_stay_charged_to_issue_month(self) -> None:
        frame = pd.DataFrame(
            {
                "station_code": ["S1", "S1"],
                "issue_time": pd.to_datetime(["2024-01-31 23:50", "2024-02-01 00:00"]),
                "observed_6h": [0, 0],
            }
        )
        _, monthly, _ = apply_dynamic_hard_budget(
            frame, _hazard(2), [1 / 6], _always_alarm_config()
        )
        self.assertEqual(set(monthly["station_month"]), {"S1|2024-01", "S1|2024-02"})
        self.assertEqual(
            monthly.set_index("station_month")["unsettled_bins"].to_dict(),
            {"S1|2024-01": 1, "S1|2024-02": 1},
        )
        self.assertTrue(monthly["hard_budget_met"].eq(1).all())

    def test_integer_budget_accounting_has_no_int8_overflow(self) -> None:
        self.assertEqual(budget_capacity_bins(2.5, 10), 15)
        frame = _frame(1600)
        _, monthly, _ = apply_dynamic_hard_budget(
            frame, _hazard(1600), [300], _always_alarm_config()
        )
        self.assertGreaterEqual(int(monthly["action_bins"].sum()), 255)
        self.assertTrue(monthly["maximum_reserved_hours"].ge(0).all())
        self.assertGreaterEqual(monthly["peak_occupied_bins"].dtype.itemsize, 8)

    def test_euclidean_dual_mirror_descent_uses_the_reference_update(self) -> None:
        frame = pd.DataFrame(
            {
                "station_code": ["S1", "S1"],
                "issue_time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
                "observed_6h": [1, 1],
            }
        )
        config = DynamicBudgetConfig(
            step_minutes=60,
            horizon_steps=1,
            method="dual_mirror_descent",
            dual_step_scale=1.0,
            dual_initial_price=0.0,
            deduplicate_events=False,
            couple_budgets=False,
        )
        controlled, monthly, trace = apply_dynamic_hard_budget(
            frame,
            np.full((2, 1), 0.5, dtype=np.float64),
            [2.0],
            config,
            trace_all_steps=True,
        )
        calendar_bins = 31 * 24
        expected_after_first = (1.0 - 2.0 / calendar_bins) / np.sqrt(calendar_bins)
        self.assertAlmostEqual(float(trace.loc[0, "decision_threshold"]), 0.0)
        self.assertAlmostEqual(float(trace.loc[0, "dual_price"]), expected_after_first)
        self.assertAlmostEqual(float(trace.loc[1, "decision_threshold"]), expected_after_first)
        self.assertEqual(controlled["budget_alarm_2h"].tolist(), [1, 1])
        self.assertTrue(monthly["hard_budget_met"].eq(1).all())

    def test_switch_over_knapsack_relaxes_only_after_frozen_switch_time(self) -> None:
        frame = pd.DataFrame(
            {
                "station_code": ["S1", "S1"],
                "issue_time": pd.to_datetime(["2024-01-05 00:00", "2024-01-20 00:00"]),
                "observed_6h": [1, 1],
            }
        )
        config = DynamicBudgetConfig(
            step_minutes=60,
            horizon_steps=1,
            method="switch_over_knapsack",
            switch_high_threshold=0.75,
            switch_low_threshold=0.25,
            switch_fraction=0.5,
            deduplicate_events=False,
            couple_budgets=False,
        )
        controlled, monthly, trace = apply_dynamic_hard_budget(
            frame,
            np.full((2, 1), 0.5, dtype=np.float64),
            [1.0],
            config,
            trace_all_steps=True,
        )
        self.assertEqual(controlled["budget_alarm_1h"].tolist(), [0, 1])
        self.assertEqual(trace["decision_threshold"].tolist(), [0.75, 0.25])
        self.assertTrue(monthly["hard_budget_met"].eq(1).all())

    def test_invalid_standard_baseline_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DynamicBudgetConfig(dual_step_scale=0.0)
        with self.assertRaises(ValueError):
            DynamicBudgetConfig(switch_low_threshold=0.2, switch_high_threshold=0.1)
        with self.assertRaises(ValueError):
            DynamicBudgetConfig(switch_fraction=1.1)

    def test_offline_score_oracle_is_nested_and_never_reads_labels(self) -> None:
        frame = _frame(40)
        oracle = offline_score_oracle(frame, np.arange(40), [1 / 6, 0.5])
        low = oracle["offline_score_oracle_0.166667h"].astype(bool)
        high = oracle["offline_score_oracle_0.5h"].astype(bool)
        self.assertTrue((~low | high).all())
        self.assertEqual(int(low.sum()), 1)
        self.assertEqual(int(high.sum()), 3)
        self.assertTrue(oracle["oracle_uses_event_labels"].eq(0).all())

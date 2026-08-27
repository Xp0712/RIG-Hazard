import unittest

import numpy as np
import pandas as pd

from rig_hazard.alert_governance import (
    CandidatePolicy,
    SafeBudgetPolicy,
    apply_candidate_policy_by_entity,
    assert_prefix_invariance,
    station_month_budget_distribution,
    strict_event_alert_evaluation,
    strict_pre_event_prediction_grid,
)


class AlertGovernanceTests(unittest.TestCase):
    def test_safe_cap_guarantees_every_entity_month(self) -> None:
        times = pd.date_range("2024-01-01", "2024-03-01", freq="10min", inclusive="left")
        frame = pd.concat(
            [
                pd.DataFrame({"station_code": station, "issue_time": times, "score": 1.0})
                for station in ("A", "B")
            ],
            ignore_index=True,
        )
        controlled = apply_candidate_policy_by_entity(
            frame,
            "score",
            CandidatePolicy("fixed_threshold", threshold=0.5),
            safe_budget=SafeBudgetPolicy(10, 2.0, burst_allowance_hours=0.5),
        )
        controlled["onset_within_6h"] = 0
        monthly, summary = station_month_budget_distribution(
            controlled, "budget_alarm", 2.0, 10
        )
        self.assertTrue((monthly["alarm_hours"] <= 2.0 + 1e-9).all())
        self.assertEqual(summary["station_month_exceedance_rate"], 0.0)

    def test_candidate_and_safe_cap_are_prefix_invariant(self) -> None:
        times = pd.date_range("2024-01-01", periods=800, freq="10min")
        scores = np.sin(np.arange(times.size) / 13.0) + np.arange(times.size) / 1000.0
        assert_prefix_invariance(
            times,
            scores,
            CandidatePolicy(
                "rolling_quantile",
                rolling_quantile=0.8,
                trailing_history_days=2,
                minimum_history_rows=12,
            ),
            SafeBudgetPolicy(10, 2.0),
            cut_points=(250, 500),
        )

    def test_daily_budget_distribution_does_not_overflow_int8(self) -> None:
        frame = pd.DataFrame(
            {
                "station_code": "A",
                "issue_time": pd.date_range("2024-01-01", periods=31, freq="D"),
                "alarm": np.ones(31, dtype=np.int8),
                "onset_within_6h": np.zeros(31, dtype=np.int8),
            }
        )

        monthly, summary = station_month_budget_distribution(
            frame,
            "alarm",
            budget_hours=744.0,
            step_minutes=1440,
        )

        self.assertEqual(int(monthly.loc[0, "alarm_bins"]), 31)
        self.assertEqual(float(monthly.loc[0, "alarm_hours"]), 31.0 * 24.0)
        self.assertEqual(float(monthly.loc[0, "false_alarm_hours"]), 31.0 * 24.0)
        self.assertEqual(summary["station_month_exceedance_rate"], 0.0)

    def test_strict_matching_never_reuses_event_or_segment(self) -> None:
        times = pd.date_range("2024-01-01", periods=80, freq="10min")
        onset_a = times[40]
        onset_b = times[70]
        alarm = np.zeros(times.size, dtype=np.int8)
        alarm[10:15] = 1
        alarm[45:49] = 1
        frame = pd.DataFrame(
            {
                "station_code": "A",
                "issue_time": times,
                "alarm": alarm,
                "onset_within_6h": 0,
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "station_code": ["A", "A"],
                "onset_time": [onset_a, onset_b],
                "valid_target_event": [1, 1],
            }
        )
        metrics, records, _ = strict_event_alert_evaluation(
            frame, events, alarm_column="alarm", horizon_hours=6
        )
        self.assertEqual(metrics["duplicate_event_matches"], 0)
        self.assertEqual(metrics["duplicate_segment_matches"], 0)
        self.assertLessEqual(int(records["hit"].sum()), 2)

    def test_observed_event_resets_a_contiguous_alarm_segment(self) -> None:
        times = pd.date_range("2024-01-01 00:00", periods=5, freq="h")
        frame = pd.DataFrame(
            {
                "station_code": "A",
                "issue_time": times,
                "alarm": np.ones(times.size, dtype=np.int8),
                "observed_6h": np.ones(times.size, dtype=np.int8),
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "station_code": ["A", "A"],
                "onset_time": [times[2], times[-1] + pd.Timedelta(hours=1)],
                "valid_target_event": [1, 1],
            }
        )

        metrics, records, strict_frame = strict_event_alert_evaluation(
            frame,
            events,
            alarm_column="alarm",
            step_minutes=60,
            horizon_hours=2,
            observability_column="observed_6h",
        )

        self.assertEqual(metrics["alert_segments"], 2)
        self.assertEqual(metrics["operational_hit_events"], 2)
        self.assertEqual(int(records["hit"].sum()), 2)
        self.assertEqual(int(strict_frame["_strict_true_alarm"].sum()), 4)
        self.assertEqual(int(strict_frame["_strict_false_alarm"].sum()), 1)

    def test_incomplete_event_window_is_not_evaluable(self) -> None:
        times = pd.date_range("2024-01-01 02:00", periods=30, freq="10min")
        onset = pd.Timestamp("2024-01-01 06:00")
        frame = pd.DataFrame(
            {"station_code": "A", "issue_time": times, "alarm": 1}
        )
        events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "station_code": ["A"],
                "onset_time": [onset],
                "valid_target_event": [1],
            }
        )
        metrics, _, _ = strict_event_alert_evaluation(frame, events, alarm_column="alarm")
        self.assertEqual(metrics["evaluable_events"], 0)

    def test_grid_anchor_invariants_for_aligned_and_unaligned_events(self) -> None:
        for onset_text in ("2024-01-01 22:30:00", "2024-01-01 22:28:00"):
            onset = pd.Timestamp(onset_text)
            grid = strict_pre_event_prediction_grid(onset, step_minutes=10, horizon_hours=6)
            self.assertEqual(grid.size, 36)
            self.assertEqual(grid[-1], pd.Timestamp("2024-01-01 22:20:00"))
            self.assertLess(grid[-1], onset)
            self.assertLessEqual(onset - grid[-1], pd.Timedelta(minutes=10))
            self.assertEqual(grid[0], grid[-1] - pd.Timedelta(minutes=350))
            np.testing.assert_array_equal(
                np.diff(grid.asi8),
                np.full(35, pd.Timedelta(minutes=10).value, dtype=np.int64),
            )
            leads = onset - grid
            self.assertTrue(bool(np.all(leads > pd.Timedelta(0))))
            self.assertTrue(bool(np.all(leads <= pd.Timedelta(hours=6))))

    def test_unaligned_event_with_complete_grid_is_evaluable(self) -> None:
        onset = pd.Timestamp("2024-01-01 22:28:00")
        times = strict_pre_event_prediction_grid(onset)
        frame = pd.DataFrame(
            {
                "station_code": "A",
                "issue_time": times,
                "alarm": np.ones(times.size, dtype=np.int8),
                "observed_6h": np.ones(times.size, dtype=np.int8),
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "station_code": ["A"],
                "onset_time": [onset],
                "valid_target_event": [1],
            }
        )
        metrics, records, _ = strict_event_alert_evaluation(
            frame, events, alarm_column="alarm"
        )
        self.assertEqual(metrics["evaluable_events"], 1)
        self.assertEqual(metrics["hit_events"], 1)
        self.assertEqual(int(records.loc[0, "available_prediction_bins"]), 36)
        self.assertGreater(float(records.loc[0, "effective_lead_hours"]), 0.0)
        self.assertLessEqual(float(records.loc[0, "effective_lead_hours"]), 6.0)

    def test_operational_queue_and_unsettled_alarm_contract(self) -> None:
        onset = pd.Timestamp("2024-01-01 22:28:00")
        full_grid = strict_pre_event_prediction_grid(onset)
        times = full_grid[-3:].append(
            pd.DatetimeIndex([pd.Timestamp("2024-01-02 08:00:00")])
        )
        frame = pd.DataFrame(
            {
                "station_code": "A",
                "issue_time": times,
                "alarm": [1, 1, 1, 1],
                "observed_6h": [0, 0, 0, 0],
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "station_code": ["A"],
                "onset_time": [onset],
                "valid_target_event": [1],
            }
        )
        metrics, records, strict_frame = strict_event_alert_evaluation(
            frame, events, alarm_column="alarm"
        )
        self.assertEqual(metrics["evaluable_events"], 0)
        self.assertEqual(metrics["operational_evaluable_events"], 1)
        self.assertEqual(metrics["operational_hit_events"], 1)
        self.assertEqual(int(records.loc[0, "known_event_overrides_horizon_censoring"]), 1)
        self.assertEqual(int(strict_frame["_strict_true_alarm"].sum()), 3)
        self.assertEqual(int(strict_frame["_strict_false_alarm"].sum()), 0)
        self.assertEqual(int(strict_frame["_strict_unsettled_alarm"].sum()), 1)
        monthly, summary = station_month_budget_distribution(
            strict_frame,
            "alarm",
            budget_hours=1.0,
            step_minutes=10,
            strict_false_column="_strict_false_alarm",
            unsettled_column="_strict_unsettled_alarm",
            reserved_column="_strict_reserved_alarm",
        )
        self.assertEqual(int(monthly.loc[0, "false_alarm_bins"]), 0)
        self.assertEqual(int(monthly.loc[0, "unsettled_alarm_bins"]), 1)
        self.assertEqual(int(monthly.loc[0, "reserved_alarm_bins"]), 1)
        self.assertAlmostEqual(summary["maximum_reserved_alarm_hours"], 1.0 / 6.0)

    def test_event_records_include_unmatched_station_once(self) -> None:
        frame = pd.DataFrame(
            {
                "station_code": ["A"],
                "issue_time": [pd.Timestamp("2024-01-01 00:00:00")],
                "alarm": [0],
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "station_code": ["B"],
                "onset_time": [pd.Timestamp("2024-01-01 06:00:00")],
                "valid_target_event": [1],
            }
        )
        _, records, _ = strict_event_alert_evaluation(frame, events, alarm_column="alarm")
        self.assertEqual(records.shape[0], 1)
        self.assertEqual(records.loc[0, "primary_exclusion_reason"], "station_id_not_matched")


if __name__ == "__main__":
    unittest.main()

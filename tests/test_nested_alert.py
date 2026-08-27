import unittest

import numpy as np
import pandas as pd

from rig_hazard.nested_alert import (
    AlertShape,
    apply_alert_policy,
    assert_nested_alert_masks,
    causal_alert_mask,
    evaluate_event_alerts,
    select_nested_thresholds,
)


class NestedAlertTests(unittest.TestCase):
    def test_causal_alert_does_not_change_past_when_future_score_changes(self) -> None:
        shape = AlertShape(ema_alpha=0.5, minimum_consecutive_bins=2, hold_bins=3, merge_gap_bins=1)
        first = np.asarray([0.0, 0.8, 0.9, 0.1, 0.2, 0.3])
        second = first.copy()
        second[4:] = 1.0
        alarm_a, smooth_a = causal_alert_mask(first, 0.5, shape)
        alarm_b, smooth_b = causal_alert_mask(second, 0.5, shape)
        np.testing.assert_array_equal(alarm_a[:4], alarm_b[:4])
        np.testing.assert_allclose(smooth_a[:4], smooth_b[:4])

    def test_vectorized_state_machine_matches_online_reference(self) -> None:
        rng = np.random.default_rng(31)
        values = rng.uniform(size=200)
        values[[17, 118]] = np.nan
        for shape in (
            AlertShape(1.0, 1, 3, 2),
            AlertShape(0.5, 2, 6, 4),
            AlertShape(0.25, 3, 12, 6),
        ):
            expected = np.zeros(values.size, dtype=np.int8)
            previous = None
            run = hold = grace = 0
            for index, value in enumerate(values):
                if not np.isfinite(value):
                    previous = None
                    run = hold = grace = 0
                    continue
                previous = value if previous is None else shape.ema_alpha * value + (1 - shape.ema_alpha) * previous
                run = run + 1 if previous >= 0.7 else 0
                if run >= shape.minimum_consecutive_bins:
                    expected[index] = 1
                    hold = max(shape.hold_bins - 1, 0)
                    grace = shape.merge_gap_bins
                elif hold > 0:
                    expected[index] = 1
                    hold -= 1
                    grace = shape.merge_gap_bins
                elif grace > 0:
                    expected[index] = 1
                    grace -= 1
            actual, _ = causal_alert_mask(values, 0.7, shape)
            np.testing.assert_array_equal(actual, expected)

    def test_threshold_order_produces_nested_masks(self) -> None:
        times = pd.date_range("2022-01-01", periods=12, freq="10min")
        frame = pd.DataFrame(
            {"station_code": "A", "issue_time": times, "risk_6h": np.linspace(0, 1, 12)}
        )
        shape = AlertShape(1.0, 1, 2, 1)
        masks = {}
        for budget, threshold in zip((2.0, 5.0, 10.0, 20.0), (0.9, 0.7, 0.5, 0.3)):
            masks[budget] = apply_alert_policy(frame, "risk_6h", threshold, shape)["budget_alarm"]
        assert_nested_alert_masks(masks)

    def test_dynamic_programming_selects_monotone_thresholds(self) -> None:
        rows = []
        for budget in (2.0, 5.0, 10.0, 20.0):
            for threshold in (0.9, 0.7, 0.5, 0.3):
                rows.append(
                    {
                        "budget_hours": budget,
                        "threshold": threshold,
                        "budget_met": int(threshold >= {2.0: 0.9, 5.0: 0.7, 10.0: 0.5, 20.0: 0.3}[budget]),
                        "lead_utility_hours": 1.0 - threshold,
                        "event_hit_rate": 1.0 - threshold,
                        "median_effective_lead_hours": 2.0,
                        "alert_segments": int(100 * (1.0 - threshold)),
                    }
                )
        selected = select_nested_thresholds(pd.DataFrame(rows))
        thresholds = selected.sort_values("budget_hours")["threshold"].to_numpy()
        self.assertTrue(np.all(thresholds[:-1] >= thresholds[1:]))

    def test_event_evaluation_uses_first_alarm_and_one_segment_per_event(self) -> None:
        times = pd.date_range("2022-01-01", periods=18, freq="10min")
        frame = pd.DataFrame(
            {
                "station_code": "A",
                "issue_time": times,
                "station_month": "A_2022-01",
                "observed_6h": 1,
                "onset_within_6h": 0,
                "hard_negative_6h": 0,
                "alarm": [0, 0, 1, 1, 1, *([0] * 13)],
            }
        )
        onset = times[8]
        frame.loc[frame["issue_time"].between(onset - pd.Timedelta(hours=6), onset, inclusive="left"), "onset_within_6h"] = 1
        events = pd.DataFrame(
            {"event_id": ["E1"], "station_code": ["A"], "onset_time": [onset], "valid_target_event": [1]}
        )
        metrics, records = evaluate_event_alerts(frame, events, "alarm", 20.0)
        self.assertEqual(metrics["hit_events"], 1)
        self.assertEqual(records.loc[0, "effective_lead_hours"], 1.0)


if __name__ == "__main__":
    unittest.main()

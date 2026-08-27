from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rig_hazard.preprocessing import (
    causal_recurrence_features,
    extract_events,
    forward_window_sum,
    interval_event_counts,
)


def test_config() -> dict:
    return {
        "time_step_minutes": 10,
        "event": {
            "ice_threshold": 0.0,
            "merge_gap_minutes": 10,
            "cooldown_minutes": 60,
            "cold_plausible_temperature_c": 2.0,
            "minimum_positive_minutes": 1,
            "use_only_cold_plausible_events_as_targets": True,
        },
    }


class EventExtractionTests(unittest.TestCase):
    def test_gap_merge_and_cold_filter(self) -> None:
        positives = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2022-01-01 00:00:00",
                        "2022-01-01 00:11:00",
                        "2022-01-01 00:30:00",
                    ]
                ),
                "ice_thickness": [0.1, 0.3, 0.2],
                "air_temperature": [-1.0, -0.5, 8.0],
                "relative_humidity": [100.0, 99.0, 90.0],
                "visibility": [500.0, 400.0, 5000.0],
                "fog_flag": [1, 1, 0],
            }
        )
        events = extract_events(positives, "F0001", "测试站", "测试", test_config())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["ice_positive_minutes"], 2)
        self.assertEqual(events[0]["valid_target_event"], 1)
        self.assertEqual(events[1]["valid_target_event"], 0)

    def test_cooldown_is_checked_at_prediction_grid_time(self) -> None:
        positives = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2022-01-01 00:00:00", "2022-01-01 01:03:00"]),
                "ice_thickness": [0.1, 0.2],
                "air_temperature": [-1.0, -1.0],
                "relative_humidity": [100.0, 100.0],
                "visibility": [500.0, 500.0],
                "fog_flag": [1, 1],
            }
        )
        events = extract_events(positives, "F0001", "测试站", "测试", test_config())
        self.assertEqual(events[1]["reference_issue_time"], pd.Timestamp("2022-01-01 01:00:00"))
        self.assertEqual(events[1]["valid_target_event"], 1)

        positives.loc[1, "timestamp"] = pd.Timestamp("2022-01-01 00:59:00")
        events = extract_events(positives, "F0001", "测试站", "测试", test_config())
        self.assertEqual(events[1]["reference_issue_time"], pd.Timestamp("2022-01-01 00:50:00"))
        self.assertEqual(events[1]["valid_target_event"], 0)
        self.assertEqual(events[1]["target_exclusion_reason"], "prediction_time_within_cooldown")

    def test_event_interval_is_left_closed(self) -> None:
        issue = pd.Series(pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20"]))
        counts = interval_event_counts(issue, [pd.Timestamp("2022-01-01 00:10")], 10)
        np.testing.assert_array_equal(counts, np.array([0, 1, 0], dtype=np.int16))

    def test_forward_window_excludes_current_bin(self) -> None:
        sums, complete = forward_window_sum(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        self.assertEqual(sums[0], 5.0)
        self.assertEqual(sums[1], 7.0)
        self.assertTrue(complete[1])
        self.assertFalse(complete[2])

    def test_recurrence_features_use_only_completed_events(self) -> None:
        events = [
            {
                "onset_time": pd.Timestamp("2022-01-01 00:00"),
                "end_time": pd.Timestamp("2022-01-01 01:00"),
                "elapsed_minutes": 61,
                "max_thickness": 0.4,
            },
            {
                "onset_time": pd.Timestamp("2022-01-03 00:00"),
                "end_time": pd.Timestamp("2022-01-03 02:00"),
                "elapsed_minutes": 121,
                "max_thickness": 0.8,
            },
        ]
        issues = pd.Series(
            pd.to_datetime(
                [
                    "2021-12-31 23:00",
                    "2022-01-01 00:30",
                    "2022-01-01 02:00",
                    "2022-01-03 01:00",
                    "2022-01-04 02:00",
                ]
            )
        )

        features = causal_recurrence_features(issues, events)

        np.testing.assert_array_equal(features["current_event_order"], [1, 1, 2, 2, 3])
        np.testing.assert_array_equal(features["previous_recurrent_event_missing"], [1, 1, 0, 0, 0])
        self.assertAlmostEqual(features["time_since_last_recurrent_event_hours"][2], 1.0)
        self.assertAlmostEqual(features["time_since_last_recurrent_event_hours"][4], 24.0)
        self.assertAlmostEqual(features["previous_event_duration_hours"][2], 61.0 / 60.0)
        self.assertAlmostEqual(features["previous_event_max_thickness"][4], 0.8)
        np.testing.assert_array_equal(features["events_past_7d"], [0, 1, 1, 2, 2])
        np.testing.assert_array_equal(features["events_past_30d"], [0, 1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()

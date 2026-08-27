from __future__ import annotations

import unittest

import pandas as pd

from rig_hazard.recurrence_analysis import prepare_valid_events, station_event_summary


class RecurrenceAnalysisTests(unittest.TestCase):
    def test_valid_order_is_recomputed_after_filtering(self) -> None:
        events = pd.DataFrame(
            {
                "event_id": ["A-1", "A-2", "A-3", "B-1"],
                "station_code": ["A", "A", "A", "B"],
                "station_name": ["alpha", "alpha", "alpha", "beta"],
                "onset_time": pd.to_datetime(
                    ["2022-01-01", "2022-01-02", "2022-01-03", "2022-02-01"]
                ),
                "end_time": pd.to_datetime(
                    ["2022-01-01 01:00", "2022-01-02 01:00", "2022-01-03 02:00", "2022-02-01 01:00"]
                ),
                "valid_target_event": [1, 0, 1, 1],
                "elapsed_minutes": [61, 61, 121, 61],
                "max_thickness": [0.1, 0.2, 0.3, 0.1],
            }
        )

        valid = prepare_valid_events(events)
        station_a = valid.loc[valid["station_code"].eq("A")]

        self.assertEqual(station_a["valid_event_order"].tolist(), [1, 2])
        self.assertEqual(station_a["event_order_group"].tolist(), ["first", "second"])
        self.assertAlmostEqual(station_a["gap_from_previous_end_hours"].iloc[1], 47.0)

        summary = station_event_summary(valid).set_index("station_code")
        self.assertEqual(summary.loc["A", "first_event_count"], 1)
        self.assertEqual(summary.loc["A", "second_event_count"], 1)
        self.assertEqual(summary.loc["A", "third_plus_event_count"], 0)
        self.assertEqual(summary.loc["B", "has_recurrence"], 0)


if __name__ == "__main__":
    unittest.main()

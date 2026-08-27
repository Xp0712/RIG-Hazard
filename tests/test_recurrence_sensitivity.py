from __future__ import annotations

import unittest

import pandas as pd

from rig_hazard.recurrence_sensitivity import (
    _valid_events,
    estimate_instrument_resolution,
    resolution_persistent_records,
    sensitivity_grid,
)


class RecurrenceSensitivityTests(unittest.TestCase):
    def test_grid_contains_all_requested_definitions(self) -> None:
        grid = sensitivity_grid(persistence_records=3)

        self.assertEqual(len(grid), 36)
        self.assertEqual(sum(int(row["is_reference"]) for row in grid), 1)
        self.assertEqual({row["merge_gap_minutes"] for row in grid}, {10, 30, 60})
        self.assertEqual({row["cooldown_hours"] for row in grid}, {1, 3, 6})
        self.assertEqual({row["cold_temperature_c"] for row in grid}, {0, 2})

    def test_resolution_and_persistence_filter(self) -> None:
        values = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.05])
        resolution = estimate_instrument_resolution(values)
        self.assertAlmostEqual(resolution, 0.01)

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2022-01-01 00:00",
                        "2022-01-01 00:01",
                        "2022-01-01 00:02",
                        "2022-01-01 01:00",
                        "2022-01-01 01:01",
                    ]
                ),
                "ice_thickness": [0.02, 0.03, 0.04, 0.02, 0.03],
            }
        )
        filtered = resolution_persistent_records(frame, resolution=resolution, minimum_records=3)

        self.assertEqual(filtered.shape[0], 3)
        self.assertEqual(filtered["timestamp"].max(), pd.Timestamp("2022-01-01 00:02"))

    def test_event_order_is_separate_for_each_definition(self) -> None:
        events = pd.DataFrame(
            {
                "definition_id": ["a", "a", "b", "b"],
                "station_code": ["S", "S", "S", "S"],
                "valid_target_event": [1, 1, 1, 1],
                "onset_time": pd.to_datetime(
                    ["2022-01-01 00:00", "2022-01-02 00:00", "2022-01-01 01:00", "2022-01-03 00:00"]
                ),
                "end_time": pd.to_datetime(
                    ["2022-01-01 01:00", "2022-01-02 01:00", "2022-01-01 02:00", "2022-01-03 01:00"]
                ),
            }
        )

        valid = _valid_events(events)

        self.assertEqual(valid.groupby("definition_id")["sensitivity_event_order"].apply(list).to_dict(), {"a": [1, 2], "b": [1, 2]})
        self.assertTrue(valid.loc[valid["sensitivity_event_order"].eq(1), "gap_from_previous_end_hours"].isna().all())


if __name__ == "__main__":
    unittest.main()

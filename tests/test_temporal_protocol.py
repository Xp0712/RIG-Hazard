from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rig_hazard.temporal_protocol import (
    assign_event_balanced_blocks,
    calendar_block_start,
    expanded_interval_mask,
    map_block_folds,
)


class TemporalProtocolTests(unittest.TestCase):
    def test_nanosecond_block_keys_are_never_cast_to_platform_int(self) -> None:
        keys = np.asarray(
            [1640995200000000000, 1641600000000000000], dtype=np.int64
        )
        values = np.asarray([keys[1], keys[0], keys[1]], dtype=np.int64)

        mapped = map_block_folds(values, keys, np.asarray([4, 2], dtype=np.int8))

        np.testing.assert_array_equal(mapped, [2, 4, 2])

    def test_all_rows_in_week_share_fold_and_events_are_represented(self) -> None:
        times = pd.date_range("2022-01-01", periods=12, freq="7D")
        metadata = pd.DataFrame(
            {
                "issue_time_ns": np.repeat(times.view("int64"), 2),
                "hazard_label": np.tile([1, 0], 12),
            }
        )

        blocks = assign_event_balanced_blocks(metadata, number_folds=3, block_days=7)

        self.assertEqual(blocks.shape[0], 12)
        self.assertEqual(set(blocks["fold"]), {0, 1, 2})
        self.assertTrue((blocks.groupby("fold")["event_rows"].sum() > 0).all())

    def test_quiet_blocks_do_not_accumulate_in_the_lowest_event_fold(self) -> None:
        times = pd.date_range("2022-01-01", periods=15, freq="7D")
        event_counts = [20, 10, 5] + [0] * 12
        rows = []
        for timestamp, events in zip(times, event_counts):
            rows.extend(
                {"issue_time_ns": timestamp.value, "hazard_label": int(row < events)}
                for row in range(100)
            )
        blocks = assign_event_balanced_blocks(pd.DataFrame(rows), number_folds=3, block_days=7)

        block_counts = blocks.groupby("fold").size()
        row_counts = blocks.groupby("fold")["risk_rows"].sum()
        self.assertLessEqual(int(block_counts.max() - block_counts.min()), 1)
        self.assertLessEqual(int(row_counts.max() - row_counts.min()), 100)

    def test_purge_covers_history_and_horizon_around_block(self) -> None:
        start = pd.Timestamp("2022-02-01").value
        times = pd.to_datetime(
            ["2022-01-30 17:00", "2022-01-30 18:00", "2022-02-09 05:59", "2022-02-09 06:00"]
        ).view("int64")

        mask = expanded_interval_mask(times, np.asarray([start]), block_days=7, purge_hours=30)

        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_calendar_blocks_are_deterministic(self) -> None:
        times = pd.to_datetime(["2022-01-01", "2022-01-07", "2022-01-08"]).view("int64")
        blocks = calendar_block_start(times, block_days=7)

        self.assertEqual(blocks[0], blocks[1])
        self.assertNotEqual(blocks[1], blocks[2])


if __name__ == "__main__":
    unittest.main()

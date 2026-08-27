import unittest

import numpy as np
import pandas as pd

from rig_hazard.deep_graph_gate import aligned_graph_contribution


class DeepGraphGateTests(unittest.TestCase):
    def test_past_shift_never_reads_current_or_future_contribution(self) -> None:
        frame = pd.DataFrame(
            {
                "station_code": ["A"] * 4,
                "issue_time": pd.date_range("2023-01-01", periods=4, freq="h"),
                "stable_graph_contribution": np.asarray([1.0, 2.0, 3.0, 4.0]),
            }
        )
        shifted = aligned_graph_contribution(frame, -pd.Timedelta(hours=1))
        np.testing.assert_array_equal(shifted, [0.0, 1.0, 2.0, 3.0])

    def test_station_permutation_uses_other_station_at_same_time(self) -> None:
        times = pd.date_range("2023-01-01", periods=2, freq="h")
        frame = pd.DataFrame(
            {
                "station_code": ["A", "A", "B", "B"],
                "issue_time": [*times, *times],
                "stable_graph_contribution": [1.0, 2.0, 10.0, 20.0],
            }
        )
        shifted = aligned_graph_contribution(
            frame,
            pd.Timedelta(0),
            {"A": "B", "B": "A"},
        )
        np.testing.assert_array_equal(shifted, [10.0, 20.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()

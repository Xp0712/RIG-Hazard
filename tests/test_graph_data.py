import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rig_hazard.graph_data import SignalBank, build_graph_schema


class SignalBankTests(unittest.TestCase):
    def test_lagged_rolling_signal_never_reads_after_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "2022" / "A.csv.gz"
            destination.parent.mkdir(parents=True)
            times = pd.date_range("2022-01-01 00:10", periods=6, freq="10min")
            pd.DataFrame({"issue_time": times, "source_signal": [0, 1, 2, 3, 4, 5]}).to_csv(
                destination, index=False, compression="gzip"
            )
            bank = SignalBank(root, 2022, step_minutes=10, rolling_window_minutes=20)
            values = bank.lagged_values("A", lag_minutes=20, issue_times=pd.DatetimeIndex(times))
            np.testing.assert_allclose(values, [0, 0, 0, 1, 2, 3])


class GraphSchemaTests(unittest.TestCase):
    def test_schema_excludes_unseen_sources_and_targets(self) -> None:
        catalog = pd.DataFrame(
            {
                "station_code": ["A", "B", "C"],
                "city": ["X", "X", "X"],
                "seen_in_development": [1, 1, 0],
            }
        )
        edges = pd.DataFrame(
            {
                "source_station_code": ["A", "C"],
                "source_station_name": ["A", "C"],
                "target_station_code": ["B", "B"],
                "target_station_name": ["B", "B"],
                "distance_km": [10, 5],
                "distance_rank": [1, 2],
                "source_minus_target_elevation_m": [0, 0],
                "same_city": [1, 1],
            }
        )
        schema = build_graph_schema(catalog, edges, [30, 60], candidate_neighbors=4, max_distance_km=100)
        self.assertEqual(schema.graph_feature_count, 2)
        self.assertTrue(schema.edge_features["source_station_code"].eq("A").all())
        self.assertNotIn("C", schema.station_columns)


if __name__ == "__main__":
    unittest.main()

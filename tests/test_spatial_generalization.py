import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rig_hazard.spatial_generalization import (
    _paired_station_gap_bootstrap,
    build_spatial_holdouts,
)


class SpatialGeneralizationTests(unittest.TestCase):
    def test_paired_bootstrap_reports_probability_and_calibration_metrics(self) -> None:
        identity = {
            "file_id": np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
            "row_index": np.asarray([0, 1, 2, 3, 0, 1, 2, 3]),
            "issue_time_ns": np.arange(8, dtype=np.int64),
            "onset_within_6h": np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8),
            "observed_6h": np.ones(8, dtype=np.int8),
        }
        first = {
            **identity,
            "risk_6h": np.asarray([0.1, 0.8, 0.2, 0.7, 0.1, 0.9, 0.3, 0.8]),
            "sample_weight": np.ones(8),
        }
        second = {
            **identity,
            "risk_6h": np.asarray([0.4, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5, 0.5]),
        }
        with tempfile.TemporaryDirectory() as directory:
            result = _paired_station_gap_bootstrap(
                first,
                second,
                {0: "A", 1: "B"},
                "spatial",
                "reference",
                "2023_spatial_temporal",
                20,
                7,
                Path(directory),
                4,
            )
        self.assertEqual(
            set(result["metric"]),
            {"delta_pr_auc", "delta_log_loss", "delta_brier", "delta_ece"},
        )
        pr_auc = result.set_index("metric").loc["delta_pr_auc"]
        self.assertGreater(float(pr_auc["estimate"]), 0.0)

    def test_holdouts_are_deterministic_disjoint_and_complete(self) -> None:
        catalog = pd.DataFrame(
            {
                "station_code": [f"S{index}" for index in range(8)],
                "city": ["A", "A", "B", "B", "C", "C", "D", "D"],
                "city_index": [0, 0, 1, 1, 2, 2, 3, 3],
            }
        )
        events = pd.DataFrame(
            {
                "station_code": ["S0", "S0", "S1", "S2", "S4", "S5", "S7"],
                "onset_time": pd.to_datetime(
                    [
                        "2022-01-01",
                        "2022-02-01",
                        "2022-01-02",
                        "2022-01-03",
                        "2022-01-04",
                        "2022-01-05",
                        "2021-01-01",
                    ]
                ),
                "valid_target_event": [1, 1, 1, 1, 1, 1, 1],
            }
        )

        holdouts, assignments = build_spatial_holdouts(catalog, events, station_folds=2)
        repeated, repeated_assignments = build_spatial_holdouts(catalog, events, station_folds=2)
        self.assertEqual(holdouts, repeated)
        pd.testing.assert_frame_equal(assignments, repeated_assignments)

        station_groups = [value for value in holdouts if value.protocol == "station_group_cv"]
        station_union = set().union(*(set(value.heldout_stations) for value in station_groups))
        station_total = sum(len(value.heldout_stations) for value in station_groups)
        self.assertEqual(station_union, set(catalog["station_code"]))
        self.assertEqual(station_total, len(station_union))

        region_groups = [value for value in holdouts if value.protocol == "region_loco"]
        self.assertEqual(len(region_groups), catalog["city"].nunique())
        for holdout in region_groups:
            expected = set(catalog.loc[catalog["city"].eq(holdout.heldout_region), "station_code"])
            self.assertEqual(set(holdout.heldout_stations), expected)


if __name__ == "__main__":
    unittest.main()

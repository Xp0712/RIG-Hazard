from __future__ import annotations

import unittest

import numpy as np

from rig_hazard.deep_data import RECURRENCE_BINARY_FEATURES, RECURRENCE_CONTINUOUS_FEATURES
from rig_hazard.recurrence_models import (
    CurrentRiskData,
    HORIZON_STEPS,
    RecurrenceDesign,
    compact_evaluation_data,
)


class RecurrenceModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_names = ["temperature_mean", *RECURRENCE_CONTINUOUS_FEATURES, *RECURRENCE_BINARY_FEATURES]
        continuous = ["temperature_mean", *RECURRENCE_CONTINUOUS_FEATURES]
        self.transformer = {
            "continuous_features": continuous,
            "binary_features": RECURRENCE_BINARY_FEATURES,
            "means": [0.0] * len(continuous),
            "stds": [1.0] * len(continuous),
        }
        features = np.asarray(
            [
                [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [-0.5, 24.0, 2.0, 1.0, 2.0, 3.0, 0.2, 0.1, 0.0],
                [0.0, 48.0, 3.0, 2.0, 4.0, 2.0, 0.3, 0.2, 0.0],
                [0.5, 72.0, 4.0, 3.0, 6.0, 1.0, 0.4, 0.3, 0.0],
            ],
            dtype=np.float32,
        )
        labels = {key: np.asarray([0, 1, 0, 1], dtype=np.int8) for key in HORIZON_STEPS}
        observed = {key: np.ones(4, dtype=bool) for key in HORIZON_STEPS}
        self.data = CurrentRiskData(
            features=features,
            labels=labels,
            observed=observed,
            sample_weight=np.ones(4),
            issue_time_ns=np.arange(4, dtype=np.int64),
            station_code=np.asarray(["A", "A", "B", "B"]),
            next_event_order=np.asarray([1, 2, 3, 4]),
            feature_names=self.feature_names,
        )

    def test_pwp_design_adds_gap_and_event_order_terms(self) -> None:
        design = RecurrenceDesign("pwp_gap_cloglog", self.feature_names, self.transformer).fit(self.data)
        matrix, names, penalty = design.transform(self.data)

        self.assertEqual(matrix.shape[0], 4)
        self.assertIn("log_gap_hours", names)
        self.assertIn("event_order_3plus", names)
        self.assertEqual(matrix.shape[1], len(names))
        self.assertEqual(matrix.shape[1], penalty.size)
        self.assertTrue(np.isfinite(matrix).all())

    def test_compact_oof_data_drops_only_design_features(self) -> None:
        compact = compact_evaluation_data(self.data)

        self.assertEqual(compact.features.shape, (4, 0))
        np.testing.assert_array_equal(compact.labels["6h"], self.data.labels["6h"])
        np.testing.assert_array_equal(compact.next_event_order, self.data.next_event_order)

    def test_frailty_design_has_shrunk_station_indicators(self) -> None:
        design = RecurrenceDesign("station_frailty_cloglog", self.feature_names, self.transformer).fit(self.data)
        matrix, names, penalty = design.transform(self.data)

        station_columns = [index for index, name in enumerate(names) if name.startswith("station_frailty[")]
        self.assertEqual(len(station_columns), 2)
        np.testing.assert_array_equal(penalty[station_columns], [10.0, 10.0])
        np.testing.assert_array_equal(matrix[:, station_columns].sum(axis=1), np.ones(4))


if __name__ == "__main__":
    unittest.main()

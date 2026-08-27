from __future__ import annotations

import unittest
from unittest.mock import patch
import sys
import types

import numpy as np
import pandas as pd

from rig_hazard.public_recurrence import (
    PublicDatasetContract,
    add_causal_recurrence_features,
    chronological_masks,
    score_public_panel,
)


class PublicRecurrenceTests(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        times = pd.date_range("2020-01-01", periods=72, freq="h")
        return pd.DataFrame(
            {
                "entity_id": np.repeat(["a", "b"], times.size),
                "issue_time": np.tile(times, 2),
                "event_count": np.r_[
                    np.eye(1, times.size, 10).ravel(),
                    np.eye(1, times.size, 20).ravel(),
                ],
                "exposure_count": 1.0,
            }
        )

    def test_horizon_label_uses_only_future_bins(self) -> None:
        panel, _ = add_causal_recurrence_features(self._panel(), horizon_bins=3)
        station = panel.loc[panel["entity_id"].eq("a")].reset_index(drop=True)
        self.assertEqual(int(station.loc[9, "onset_within_horizon"]), 1)
        self.assertEqual(int(station.loc[10, "onset_within_horizon"]), 0)
        self.assertEqual(int(station.loc[7, "onset_within_horizon"]), 1)

    def test_history_features_are_prefix_invariant(self) -> None:
        panel = self._panel()
        full, features = add_causal_recurrence_features(panel, horizon_bins=3)
        prefix, _ = add_causal_recurrence_features(
            panel.groupby("entity_id", sort=False).head(40).copy(), horizon_bins=3
        )
        for entity in ("a", "b"):
            expected = full.loc[full["entity_id"].eq(entity), features].head(40)
            actual = prefix.loc[prefix["entity_id"].eq(entity), features]
            np.testing.assert_allclose(
                expected.to_numpy(dtype=float),
                actual.to_numpy(dtype=float),
                equal_nan=True,
            )

    def test_chronological_masks_do_not_overlap(self) -> None:
        panel, _ = add_causal_recurrence_features(self._panel(), horizon_bins=3)
        contract = PublicDatasetContract(
            name="toy",
            step_minutes=60,
            horizon_bins=3,
            train_end="2020-01-02",
            selection_end="2020-01-03",
            test_end="2020-01-04",
            budgets_hours=(1.0,),
        )
        masks = chronological_masks(panel, contract)
        self.assertFalse(np.any(masks["train"] & masks["selection"]))
        self.assertFalse(np.any(masks["selection"] & masks["test"]))
        self.assertGreater(int(masks["train"].sum()), 0)
        self.assertGreater(int(masks["selection"].sum()), 0)

    def test_cuda_scoring_uses_device_matched_inplace_prediction(self) -> None:
        class FakeBooster:
            def __init__(self) -> None:
                self.calls = 0

            def inplace_predict(self, values, validate_features=True):
                self.calls += 1
                self.asserted_validate_features = validate_features
                return np.asarray(values[:, 0], dtype=np.float32)

        class FakeModel:
            def __init__(self) -> None:
                self.booster = FakeBooster()

            def get_params(self):
                return {"device": "cuda:0"}

            def get_booster(self):
                return self.booster

            def predict_proba(self, values):
                raise AssertionError("CPU prediction fallback should not be used")

        fake_cupy = types.ModuleType("cupy")
        fake_cupy.asarray = np.asarray
        fake_cupy.asnumpy = np.asarray
        model = FakeModel()
        frame = pd.DataFrame({"feature": [0.1, 0.2, 0.3]})

        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            prediction = score_public_panel(
                model,
                frame,
                ["feature"],
                batch_size=2,
            )

        np.testing.assert_allclose(prediction, [0.1, 0.2, 0.3])
        self.assertEqual(model.booster.calls, 2)
        self.assertFalse(model.booster.asserted_validate_features)


if __name__ == "__main__":
    unittest.main()

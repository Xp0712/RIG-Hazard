import copy
import os
import unittest
from unittest import mock

import numpy as np

from rig_hazard.deep_models import DeepHazardModel, cloglog_hazard_probability, cumulative_incidence
from rig_hazard.deep_training import (
    WeightedTrajectoryCalibrator,
    cumulative_probability_at_step,
    trajectory_probability_curve_rows,
    trajectory_probability_matrices,
    train_epoch,
    weighted_probability_metric_row,
)
from rig_hazard.torch_runtime import torch


class DeepCalibrationTests(unittest.TestCase):
    def test_memory_efficient_cumulative_probability_matches_torch_definition(self) -> None:
        rng = np.random.default_rng(19)
        eta = rng.normal(loc=-4.0, scale=2.0, size=(8, 36)).astype(np.float32)

        actual = cumulative_probability_at_step(eta, 36, eta_shift=0.3, eta_slope=0.8)
        calibrated = torch.from_numpy(0.3 + 0.8 * eta)
        expected = cumulative_incidence(cloglog_hazard_probability(calibrated))[:, -1].numpy()

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_full_trajectory_probability_contract_is_monotone(self) -> None:
        eta = np.asarray([[-8.0, -7.0, -6.0], [-5.0, -5.0, -5.0]], dtype=np.float32)
        hazard, cumulative = trajectory_probability_matrices(eta, eta_shift=0.2, eta_slope=0.9)
        self.assertEqual(hazard.shape, eta.shape)
        self.assertEqual(cumulative.shape, eta.shape)
        self.assertTrue(np.all((hazard >= 0) & (hazard <= 1)))
        self.assertTrue(np.all(np.diff(cumulative, axis=1) >= 0))

    def test_full_curve_metrics_respect_censoring(self) -> None:
        eta = np.full((3, 3), -5.0, dtype=np.float32)
        target = np.asarray([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        mask = np.asarray([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=np.int8)
        rows = trajectory_probability_curve_rows(
            "model", eta, target, mask, np.ones(3), 10, "calibrated"
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["sample_rows"], 3)
        self.assertEqual(rows[1]["sample_rows"], 2)
        self.assertEqual(rows[2]["sample_rows"], 2)

    def test_weighted_trajectory_calibrator_is_finite(self) -> None:
        eta = np.asarray([[-5.0, -4.0], [-2.0, -1.0], [-6.0, -5.0]], dtype=np.float64)
        target = np.asarray([[0, 0], [0, 1], [0, 0]], dtype=np.float64)
        mask = np.ones_like(target)
        calibrator = WeightedTrajectoryCalibrator().fit(eta, target, mask, np.asarray([1.0, 2.0, 1.0]))
        self.assertTrue(np.isfinite(calibrator.log_rate_shift))
        self.assertTrue(np.isfinite(calibrator.slope))
        self.assertGreaterEqual(calibrator.slope, 0.0)

    def test_calibrator_handles_extreme_low_rates(self) -> None:
        eta = np.asarray([[-500.0], [-400.0], [-300.0]], dtype=np.float64)
        target = np.asarray([[0.0], [1.0], [0.0]], dtype=np.float64)
        mask = np.ones_like(target)
        calibrator = WeightedTrajectoryCalibrator().fit(eta, target, mask, np.ones(3))
        self.assertTrue(np.isfinite(calibrator.objective))
        self.assertTrue(np.isfinite(calibrator.log_rate_shift))

    def test_weighted_metrics_represent_subsampled_population(self) -> None:
        row = weighted_probability_metric_row(
            "test",
            "raw",
            "1h",
            np.asarray([0, 1]),
            np.asarray([0.1, 0.8]),
            np.asarray([9.0, 1.0]),
        )
        self.assertAlmostEqual(row["prevalence"], 0.1)
        self.assertAlmostEqual(row["represented_rows"], 10.0)
        self.assertAlmostEqual(row["mean_probability"], 0.17)

    def test_patchtst_microbatch_matches_full_batch_update(self) -> None:
        class SingleBatchSource:
            def __init__(self, batch):
                self.batch = batch

            def iter_batches(self, batch_size, shuffle, seed):
                del batch_size, shuffle, seed
                yield self.batch

        torch.manual_seed(23)
        model = DeepHazardModel(
            encoder_type="patchtst",
            input_size=4,
            horizon_steps=3,
            hidden_size=8,
            dropout=0.0,
            history_steps=12,
            encoder_config={
                "output_size": 8,
                "patch_length": 4,
                "patch_stride": 2,
                "embedding_dim": 8,
                "attention_heads": 2,
                "layers": 1,
                "feedforward_dim": 16,
                "dropout": 0.0,
                "revin": True,
            },
        )
        microbatched_model = copy.deepcopy(model)
        batch = {
            "history": torch.randn(6, 12, 4),
            "hazard_target": torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]],
                dtype=torch.float32,
            ),
            "risk_mask": torch.tensor(
                [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1]],
                dtype=torch.float32,
            ),
            "sample_weight": torch.tensor([1.0, 2.0, 1.5, 1.0, 0.5, 3.0]),
            "stratum": torch.tensor([0, 1, 0, 1, 0, 0]),
            "station_index": torch.zeros(6, dtype=torch.long),
            "city_index": torch.zeros(6, dtype=torch.long),
        }
        full_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        micro_optimizer = torch.optim.SGD(microbatched_model.parameters(), lr=0.01)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RIG_HAZARD_PATCHTST_MICROBATCH_SIZE", None)
            full = train_epoch(
                model,
                SingleBatchSource(batch),
                full_optimizer,
                torch.device("cpu"),
                batch_size=6,
                seed=1,
                gradient_clip_norm=100.0,
                hard_negative_loss_multiplier=1.5,
                return_details=True,
            )
        with mock.patch.dict(os.environ, {"RIG_HAZARD_PATCHTST_MICROBATCH_SIZE": "2"}):
            micro = train_epoch(
                microbatched_model,
                SingleBatchSource(batch),
                micro_optimizer,
                torch.device("cpu"),
                batch_size=6,
                seed=1,
                gradient_clip_norm=100.0,
                hard_negative_loss_multiplier=1.5,
                return_details=True,
            )
        self.assertAlmostEqual(full["nll"], micro["nll"], places=7)
        for full_parameter, micro_parameter in zip(model.parameters(), microbatched_model.parameters()):
            torch.testing.assert_close(full_parameter, micro_parameter, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

import unittest

from rig_hazard.deep_models import (
    CausalConv1d,
    DeepHazardModel,
    cloglog_hazard_probability,
    cumulative_incidence,
    discrete_hazard_nll,
    standard_horizon_risks,
)
from rig_hazard.torch_runtime import torch


class DeepModelTests(unittest.TestCase):
    def test_causal_convolution_does_not_read_future_positions(self) -> None:
        torch.manual_seed(7)
        layer = CausalConv1d(2, 3, kernel_size=3, dilation=2).eval()
        original = torch.randn(1, 2, 10)
        changed = original.clone()
        changed[:, :, 6:] = 1000.0
        with torch.no_grad():
            first = layer(original)
            second = layer(changed)
        torch.testing.assert_close(first[:, :, :6], second[:, :, :6])

    def test_normalized_tcn_blocks_remain_causal(self) -> None:
        torch.manual_seed(9)
        model = DeepHazardModel(
            "tcn", input_size=2, horizon_steps=6, hidden_size=4, dilations=[1, 2], dropout=0.0
        ).eval()
        original = torch.randn(1, 10, 2)
        changed = original.clone()
        changed[:, 6:, :] = 1000.0
        with torch.no_grad():
            first = model.encoder.network(original.transpose(1, 2))
            second = model.encoder.network(changed.transpose(1, 2))
        torch.testing.assert_close(first[:, :, :6], second[:, :, :6])

    def test_gru_and_tcn_emit_full_hazard_trajectory(self) -> None:
        history = torch.randn(4, 24, 5)
        for encoder_type in ["gru", "tcn"]:
            model = DeepHazardModel(
                encoder_type,
                input_size=5,
                horizon_steps=36,
                hidden_size=8,
                dilations=[1, 2, 4],
                dropout=0.0,
            )
            self.assertEqual(tuple(model(history).shape), (4, 36))
            self.assertEqual(model.to_config()["input_size"], 5)
            self.assertEqual(model.to_config()["horizon_steps"], 36)

    def test_modern_baselines_emit_finite_hazard_trajectories(self) -> None:
        history = torch.randn(3, 24, 5)
        configurations = {
            "patchtst": {
                "output_size": 8,
                "patch_length": 6,
                "patch_stride": 3,
                "embedding_dim": 8,
                "attention_heads": 2,
                "layers": 1,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
            "timesnet": {
                "output_size": 8,
                "embedding_dim": 8,
                "feedforward_dim": 12,
                "layers": 1,
                "top_k": 2,
                "inception_kernels": [1, 3],
                "dropout": 0.0,
            },
            "itransformer": {
                "output_size": 8,
                "embedding_dim": 8,
                "attention_heads": 2,
                "layers": 1,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
        }
        for encoder_type, encoder_config in configurations.items():
            with self.subTest(encoder_type=encoder_type):
                model = DeepHazardModel(
                    encoder_type,
                    input_size=5,
                    horizon_steps=36,
                    hidden_size=8,
                    dropout=0.0,
                    history_steps=24,
                    encoder_config=encoder_config,
                )
                eta = model(history)
                self.assertEqual(tuple(eta.shape), (3, 36))
                self.assertTrue(bool(torch.isfinite(eta).all()))
                eta.mean().backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))

    def test_patchtst_production_dimensions_support_forward_and_backward(self) -> None:
        model = DeepHazardModel(
            "patchtst",
            input_size=64,
            horizon_steps=36,
            hidden_size=32,
            dropout=0.0,
            history_steps=144,
            encoder_config={
                "output_size": 32,
                "patch_length": 6,
                "patch_stride": 3,
                "embedding_dim": 32,
                "attention_heads": 4,
                "layers": 2,
                "feedforward_dim": 64,
                "dropout": 0.0,
                "revin": True,
            },
        )
        history = torch.randn(2, 144, 64)
        eta = model(history)
        self.assertEqual(tuple(eta.shape), (2, 36))
        self.assertTrue(bool(torch.isfinite(eta).all()))
        eta.mean().backward()
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))

    def test_dual_resolution_weather_encoder_and_gate_diagnostics(self) -> None:
        torch.manual_seed(31)
        history = torch.randn(4, 12, 7)
        for encoder_type in [
            "weather_fast",
            "weather_slow",
            "weather_dual",
            "weather_dual_rec",
            "weather_dual_rec_gate",
        ]:
            with self.subTest(encoder_type=encoder_type):
                model = DeepHazardModel(
                    encoder_type,
                    input_size=7,
                    horizon_steps=6,
                    hidden_size=8,
                    dropout=0.0,
                    history_steps=12,
                    encoder_config={
                        "weather_feature_indices": [0, 1, 2, 3, 4],
                        "recurrence_feature_indices": [5, 6],
                        "fast_steps": 4,
                        "slow_bucket_steps": 3,
                        "fast_hidden_size": 4,
                        "slow_hidden_size": 4,
                        "output_size": 8,
                        "recurrence_hidden_size": 4,
                        "gate_penalty_weight": 0.1,
                    },
                )
                eta = model(history)
                self.assertEqual(tuple(eta.shape), (4, 6))
                self.assertTrue(bool(torch.isfinite(eta).all()))
                diagnostics = model.diagnostics()
                if encoder_type.endswith("rec_gate"):
                    self.assertEqual(tuple(diagnostics["rec_gate"].shape), (4, 1))
                    self.assertTrue(bool(((diagnostics["rec_gate"] >= 0) & (diagnostics["rec_gate"] <= 1)).all()))
                    self.assertGreater(float(model.barrier_penalty(0.0, 0.0)), 0.0)

    def test_dual_resolution_slow_branch_uses_causal_nonoverlapping_buckets(self) -> None:
        model = DeepHazardModel(
            "weather_slow",
            input_size=2,
            horizon_steps=2,
            hidden_size=4,
            dropout=0.0,
            history_steps=6,
            encoder_config={
                "weather_feature_indices": [0, 1],
                "recurrence_feature_indices": [],
                "slow_bucket_steps": 3,
                "slow_hidden_size": 3,
                "output_size": 4,
            },
        )
        history = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        pooled = model.encoder._slow_history(history)
        expected = torch.stack([history[:, :3].mean(dim=1), history[:, 3:].mean(dim=1)], dim=1)
        torch.testing.assert_close(pooled, expected)

    def test_modern_baseline_config_round_trip_preserves_predictions(self) -> None:
        torch.manual_seed(19)
        configurations = {
            "patchtst": {
                "output_size": 8,
                "patch_length": 4,
                "patch_stride": 2,
                "embedding_dim": 8,
                "attention_heads": 2,
                "layers": 1,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
            "timesnet": {
                "output_size": 8,
                "embedding_dim": 8,
                "feedforward_dim": 12,
                "layers": 1,
                "top_k": 2,
                "inception_kernels": [1, 3],
                "dropout": 0.0,
            },
            "itransformer": {
                "output_size": 8,
                "embedding_dim": 8,
                "attention_heads": 2,
                "layers": 1,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
        }
        history = torch.randn(2, 16, 4)
        for encoder_type, encoder_config in configurations.items():
            with self.subTest(encoder_type=encoder_type):
                model = DeepHazardModel(
                    encoder_type,
                    input_size=4,
                    horizon_steps=6,
                    hidden_size=8,
                    dropout=0.0,
                    history_steps=16,
                    encoder_config=encoder_config,
                ).eval()
                restored = DeepHazardModel.from_config(model.to_config()).eval()
                restored.load_state_dict(model.state_dict())
                with torch.no_grad():
                    torch.testing.assert_close(model(history), restored(history))

    def test_timesnet_constant_history_remains_finite(self) -> None:
        model = DeepHazardModel(
            "timesnet",
            input_size=3,
            horizon_steps=6,
            hidden_size=8,
            history_steps=24,
            encoder_config={
                "output_size": 8,
                "embedding_dim": 8,
                "feedforward_dim": 12,
                "layers": 1,
                "top_k": 3,
                "inception_kernels": [1, 3],
                "dropout": 0.0,
            },
        )
        history = torch.zeros(2, 24, 3)
        eta = model(history)
        self.assertTrue(bool(torch.isfinite(eta).all()))

    def test_recurrent_dual_uses_fast_history_and_current_slow_state(self) -> None:
        torch.manual_seed(23)
        model = DeepHazardModel(
            "recurrent_dual",
            input_size=6,
            horizon_steps=6,
            hidden_size=10,
            dropout=0.0,
            history_steps=24,
            encoder_config={
                "fast_feature_indices": [0, 1, 2],
                "slow_feature_indices": [3, 4, 5],
                "fast_steps": 6,
                "fast_hidden_size": 5,
                "slow_hidden_size": 4,
                "output_size": 10,
                "dropout": 0.0,
            },
        ).eval()
        history = torch.randn(2, 24, 6)
        changed_old_fast = history.clone()
        changed_old_fast[:, :-6, :3] = 1000.0
        changed_old_slow = history.clone()
        changed_old_slow[:, :-1, 3:] = 1000.0
        with torch.no_grad():
            baseline = model(history)
            torch.testing.assert_close(baseline, model(changed_old_fast))
            torch.testing.assert_close(baseline, model(changed_old_slow))
        self.assertEqual(tuple(baseline.shape), (2, 6))

        restored = DeepHazardModel.from_config(model.to_config()).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(baseline, restored(history))

    def test_cumulative_risk_is_monotonic_and_horizons_are_aligned(self) -> None:
        eta = torch.linspace(-5.0, -2.0, 36).reshape(1, -1)
        cumulative = cumulative_incidence(cloglog_hazard_probability(eta))
        self.assertTrue(bool(torch.all(cumulative[:, 1:] >= cumulative[:, :-1])))
        risks = standard_horizon_risks(cumulative, step_minutes=10)
        torch.testing.assert_close(risks.risk_1h, cumulative[:, 5])
        torch.testing.assert_close(risks.risk_3h, cumulative[:, 17])
        torch.testing.assert_close(risks.risk_6h, cumulative[:, 35])

    def test_masked_future_does_not_change_loss(self) -> None:
        eta = torch.tensor([[0.0, -1.0, 2.0, 3.0]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        first = discrete_hazard_nll(eta, target, mask)
        changed = eta.detach().clone()
        changed[:, 2:] = -20.0
        second = discrete_hazard_nll(changed, target, mask)
        torch.testing.assert_close(first.detach(), second)
        first.backward()
        self.assertTrue(bool(torch.isfinite(eta.grad).all()))
        torch.testing.assert_close(eta.grad[:, 2:], torch.zeros_like(eta.grad[:, 2:]))

    def test_positive_loss_keeps_gradient_at_extreme_low_rate(self) -> None:
        eta = torch.tensor([[-100.0]], requires_grad=True)
        loss = discrete_hazard_nll(eta, torch.ones_like(eta), torch.ones_like(eta))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertAlmostEqual(float(loss), 100.0, places=4)
        self.assertAlmostEqual(float(eta.grad), -1.0, places=4)

    def test_model_config_round_trip_preserves_predictions(self) -> None:
        torch.manual_seed(11)
        model = DeepHazardModel("gru", input_size=3, horizon_steps=6, hidden_size=5, dropout=0.0).eval()
        restored = DeepHazardModel.from_config(model.to_config()).eval()
        restored.load_state_dict(model.state_dict())
        history = torch.randn(2, 8, 3)
        with torch.no_grad():
            torch.testing.assert_close(model(history), restored(history))

    def test_masked_history_features_cannot_change_predictions(self) -> None:
        torch.manual_seed(13)
        model = DeepHazardModel(
            "gru", input_size=3, horizon_steps=6, hidden_size=5, dropout=0.0, masked_feature_indices=[1]
        ).eval()
        original = torch.randn(2, 8, 3)
        changed = original.clone()
        changed[:, :, 1] = 1000.0
        with torch.no_grad():
            torch.testing.assert_close(model(original), model(changed))

    def test_feature_subset_gru_has_no_dependency_on_excluded_fields(self) -> None:
        torch.manual_seed(29)
        model = DeepHazardModel(
            "gru_subset",
            input_size=5,
            horizon_steps=6,
            hidden_size=4,
            dropout=0.0,
            encoder_config={"feature_indices": [0, 2, 4]},
        ).eval()
        original = torch.randn(2, 8, 5)
        changed = original.clone()
        changed[:, :, [1, 3]] = 1000.0
        with torch.no_grad():
            torch.testing.assert_close(model(original), model(changed))
        restored = DeepHazardModel.from_config(model.to_config()).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(model(original), restored(original))

    def test_hierarchical_barrier_is_shared_across_future_hazard_steps(self) -> None:
        torch.manual_seed(17)
        model = DeepHazardModel(
            "gru",
            input_size=3,
            horizon_steps=6,
            hidden_size=5,
            dropout=0.0,
            number_stations=3,
            number_cities=2,
        ).eval()
        with torch.no_grad():
            model.city_log_susceptibility.weight[:2, 0] = torch.tensor([-0.5, 0.5])
            model.station_log_susceptibility.weight[:3, 0] = torch.tensor([-1.0, 0.0, 1.0])
        history = torch.randn(1, 8, 3).expand(2, -1, -1).clone()
        with torch.no_grad():
            eta = model(history, torch.tensor([0, 2]), torch.tensor([0, 1]))
        difference = eta[1] - eta[0]
        torch.testing.assert_close(difference, torch.full_like(difference, 3.0))

        restored = DeepHazardModel.from_config(model.to_config()).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(
                eta,
                restored(history, torch.tensor([0, 2]), torch.tensor([0, 1])),
            )


if __name__ == "__main__":
    unittest.main()

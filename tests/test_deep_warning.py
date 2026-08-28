import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rig_hazard.deep_warning import (
    build_warning_report,
    calibrated_six_hour_risk,
    comparison_pairs_for_models,
    main_model_six_hour_risk,
    monthly_budget_audit,
    monthly_false_alarm_distribution_audit,
    load_frozen_warning_definition,
    load_frozen_budgets,
    load_main_model_parameters,
    run_deep_warning,
    select_matched_false_alarm_thresholds,
    score_columns_for_models,
)
from rig_hazard.budget_control import CausalBudgetConfig
from rig_hazard.torch_runtime import torch


class ConstantTrajectory(torch.nn.Module):
    def __init__(self, eta: float):
        super().__init__()
        self.eta = eta

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return torch.full((history.shape[0], 36), self.eta, dtype=history.dtype)


class DeepWarningTests(unittest.TestCase):
    def test_model_specific_budget_loader_uses_reference_fallback(self) -> None:
        reference = CausalBudgetConfig(candidate_quantile=0.91)
        model_specific = CausalBudgetConfig(candidate_quantile=0.97)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model_bundle.json").write_text(
                json.dumps(
                    {
                        "causal_budget_config": reference.__dict__,
                        "causal_budget_configs_by_score": {
                            "recurrent_dual_ensemble_6h": model_specific.__dict__,
                        },
                    }
                ),
                encoding="utf-8",
            )
            budgets = load_frozen_budgets(
                root,
                ["local_weather_hazard_6h", "recurrent_dual_ensemble_6h"],
            )

        self.assertAlmostEqual(budgets["local_weather_hazard_6h"].candidate_quantile, 0.91)
        self.assertAlmostEqual(
            budgets["recurrent_dual_ensemble_6h"].candidate_quantile,
            0.97,
        )

    def test_main_model_loader_accepts_selection_year_oof_baseline(self) -> None:
        payload = {
            "models": {
                "base_cloglog": {
                    "glm": {"coefficients": [2.0, -1.0], "intercept": -4.0},
                    "calibrator": {"log_rate_shift": 0.2, "slope": 0.8},
                    "coefficient_names": ["feature_b", "feature_a"],
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "statistical_model_bundle.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            parameters = load_main_model_parameters(
                root,
                torch.device("cpu"),
                ["feature_a", "feature_b", "unused"],
            )

        np.testing.assert_array_equal(parameters["feature_indices"].numpy(), [1, 0])
        self.assertEqual(parameters["source"], "2022 pooled-OOF calibrated base_cloglog")

    def test_budget_bundle_warning_definition_takes_precedence(self) -> None:
        frozen = {
            "horizon_hours": 6,
            "false_alarm_hours_per_station_month": 10.0,
            "minimum_consecutive_alarm_bins": 2,
            "alarm_merge_gap_minutes": 20,
            "maximum_silence_before_event_minutes": 30,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model_bundle.json").write_text(
                json.dumps({"warning_definition": frozen}), encoding="utf-8"
            )
            loaded = load_frozen_warning_definition(
                {
                    "stability_root": str(root),
                    "warning": {**frozen, "horizon_hours": 3},
                }
            )

        self.assertEqual(loaded, frozen)

    def test_frozen_2024_rejects_label_matched_diagnostic(self) -> None:
        config = {
            "warning_evaluation": {
                "year": 2024,
                "matched_false_alarm_diagnostic": {"enabled": True},
            }
        }
        with self.assertRaisesRegex(ValueError, "must be disabled"):
            run_deep_warning(config, Path("unused.json"))

    def test_warning_report_uses_dynamic_frozen_year_language(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "model_name": "gru_ensemble",
                    "false_alarm_hours_per_station_month": 1.0,
                    "event_hit_rate": 0.5,
                    "mean_effective_lead_hours": 2.0,
                    "median_effective_lead_hours": 2.0,
                    "lead_utility_hours": 1.0,
                    "hard_negative_far": 0.1,
                    "total_time_bins": 100,
                    "observed_time_bins": 90,
                    "censored_time_bins": 10,
                }
            ]
        )
        comparisons = pd.DataFrame(
            columns=["model_a_name", "model_b_name", "metric", "estimate", "ci95_low", "ci95_high"]
        )
        audit = pd.DataFrame([{"model_name": "gru_ensemble", "station_months": 1}])

        report = build_warning_report(
            metrics,
            comparisons,
            audit,
            CausalBudgetConfig(),
            ensemble_model_names=["gru"],
            evaluation_year=2024,
            ensemble_seed_count=3,
        )

        self.assertIn("2024 (frozen temporal confirmation)", report)
        self.assertIn("3 random seeds", report)
        self.assertIn("90 observable, and 10 censored", report)
        self.assertIn("retrospective frozen temporal confirmation", report)

    def test_dynamic_model_columns_and_pairs_cover_all_baselines(self) -> None:
        models = ["patchtst", "timesnet", "itransformer"]
        self.assertEqual(
            score_columns_for_models(models),
            [
                "local_weather_hazard_6h",
                "patchtst_ensemble_6h",
                "timesnet_ensemble_6h",
                "itransformer_ensemble_6h",
            ],
        )
        pairs = comparison_pairs_for_models(models)
        self.assertEqual(len(pairs), 6)
        self.assertIn(("itransformer_ensemble", "local_weather_hazard"), pairs)
        self.assertIn(("patchtst_ensemble", "timesnet_ensemble"), pairs)

    def test_calibrated_trajectory_accumulates_exactly_36_steps(self) -> None:
        history = torch.zeros(2, 4, 3)
        risk = calibrated_six_hour_risk(
            ConstantTrajectory(-9.0), history, {"log_rate_shift": 0.0, "slope": 1.0}
        )
        expected = -torch.expm1(-36.0 * torch.exp(torch.tensor(-9.0)))
        torch.testing.assert_close(risk, expected.expand_as(risk))

    def test_main_model_accumulation_matches_constant_hazard_formula(self) -> None:
        features = torch.tensor([[1.0, 2.0]])
        parameters = {
            "coefficients": torch.tensor([0.5, -0.25]),
            "intercept": -8.0,
            "shift": 0.0,
            "slope": 1.0,
        }
        risk = main_model_six_hour_risk(features, parameters)
        expected = -torch.expm1(-36.0 * torch.exp(torch.tensor(-8.0)))
        torch.testing.assert_close(risk, expected.reshape(1))

    def test_monthly_audit_does_not_overflow_int8_durations(self) -> None:
        frame = pd.DataFrame(
            {
                "station_month": ["A|2023-01"] * 60,
                "alarm": np.ones(60, dtype=np.int8),
            }
        )
        audit = monthly_budget_audit(frame, {"model": "alarm"}, step_minutes=10, budget_hours=10.0)
        self.assertAlmostEqual(float(audit.loc[0, "mean_alarm_hours"]), 10.0)
        self.assertAlmostEqual(float(audit.loc[0, "maximum_alarm_hours"]), 10.0)
        self.assertEqual(int(audit.loc[0, "months_above_budget"]), 0)

    def test_matched_thresholds_share_the_requested_false_alarm_duration(self) -> None:
        rows = 120
        frame = pd.DataFrame(
            {
                "station_month": ["A|2023-01"] * 60 + ["A|2023-02"] * 60,
                "onset_within_6h": np.zeros(rows, dtype=np.int8),
                "model_a_6h": np.linspace(0.001, 0.999, rows),
                "model_b_6h": np.linspace(0.999, 0.001, rows),
            }
        )
        selection = select_matched_false_alarm_thresholds(
            frame,
            ["model_a_6h", "model_b_6h"],
            horizon_hours=6,
            target_hours_per_station_month=0.5,
            step_minutes=10,
        )

        self.assertEqual(selection.shape[0], 2)
        np.testing.assert_allclose(
            selection["actual_false_alarm_hours_per_station_month"].to_numpy(),
            np.full(2, 0.5),
        )
        self.assertTrue(selection["absolute_error_hours_per_station_month"].eq(0).all())

    def test_false_alarm_distribution_audit_uses_false_bins(self) -> None:
        frame = pd.DataFrame(
            {
                "station_month": ["A|2023-01"] * 6,
                "onset_within_6h": np.array([0, 0, 1, 1, 0, 0], dtype=np.int8),
                "alarm": np.ones(6, dtype=np.int8),
            }
        )
        audit = monthly_false_alarm_distribution_audit(
            frame,
            {"model": "alarm"},
            horizon_hours=6,
            step_minutes=10,
            target_hours=1.0,
        )

        self.assertAlmostEqual(float(audit.loc[0, "mean_false_alarm_hours"]), 4 / 6)
        self.assertAlmostEqual(float(audit.loc[0, "mean_total_alarm_hours"]), 1.0)


if __name__ == "__main__":
    unittest.main()

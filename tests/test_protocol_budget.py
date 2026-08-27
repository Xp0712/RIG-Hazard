from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rig_hazard.protocol_budget import _load_oof_ensemble, select_matched_budget_candidate


class ProtocolBudgetTests(unittest.TestCase):
    def test_false_alarm_matching_precedes_lead_utility(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "selection_model": "gru",
                    "candidate_quantile": 0.90,
                    "false_alarm_hours_per_station_month": 9.8,
                    "budget_met": 1,
                    "lead_utility_hours": 1.0,
                    "event_hit_rate": 0.5,
                    "mean_effective_lead_hours": 2.0,
                },
                {
                    "selection_model": "gru",
                    "candidate_quantile": 0.98,
                    "false_alarm_hours_per_station_month": 7.0,
                    "budget_met": 1,
                    "lead_utility_hours": 4.0,
                    "event_hit_rate": 0.8,
                    "mean_effective_lead_hours": 5.0,
                },
            ]
        )

        chosen = select_matched_budget_candidate(candidates, "gru", 10.0)

        self.assertAlmostEqual(float(chosen["candidate_quantile"]), 0.90)

    def test_oof_seed_ensemble_requires_identical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "file_id": np.asarray([0, 1], dtype=np.int16),
                "row_index": np.asarray([10, 20], dtype=np.int32),
                "issue_time_ns": np.asarray([100, 200], dtype=np.int64),
                "onset_within_6h": np.asarray([0, 1], dtype=np.int8),
                "observed_6h": np.asarray([1, 1], dtype=np.int8),
            }
            np.savez(root / "gru_seed_1.npz", **common, risk_6h=np.asarray([0.1, 0.3]))
            np.savez(root / "gru_seed_2.npz", **common, risk_6h=np.asarray([0.3, 0.5]))

            ensemble = _load_oof_ensemble(root, "gru", [1, 2])

            np.testing.assert_allclose(ensemble["risk_6h"], [0.2, 0.4])

            changed = {**common, "row_index": np.asarray([10, 21], dtype=np.int32)}
            np.savez(root / "gru_seed_2.npz", **changed, risk_6h=np.asarray([0.3, 0.5]))
            with self.assertRaises(ValueError):
                _load_oof_ensemble(root, "gru", [1, 2])

            changed = {**common, "observed_6h": np.asarray([1, 0], dtype=np.int8)}
            np.savez(root / "gru_seed_2.npz", **changed, risk_6h=np.asarray([0.3, 0.5]))
            with self.assertRaisesRegex(ValueError, "observed_6h"):
                _load_oof_ensemble(root, "gru", [1, 2])


if __name__ == "__main__":
    unittest.main()

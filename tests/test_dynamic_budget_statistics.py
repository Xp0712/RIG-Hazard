from __future__ import annotations

import unittest

import pandas as pd

from rig_hazard.dynamic_budget_statistics import (
    holm_adjust,
    paired_station_cluster_bootstrap,
)


class DynamicBudgetStatisticsTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted.tolist(), [0.03, 0.06, 0.06])

    def test_paired_bootstrap_keeps_event_pairing(self) -> None:
        rows = []
        for method, hits in (("new", [1, 1, 1, 0]), ("base", [0, 1, 0, 0])):
            for index, hit in enumerate(hits):
                rows.append(
                    {
                        "method": method,
                        "event_id": f"E{index}",
                        "station_code": f"S{index // 2}",
                        "operational_evaluable": 1,
                        "evaluable": 1,
                        "hit": hit,
                        "effective_lead_hours": 2.0 if hit else float("nan"),
                    }
                )
        result = paired_station_cluster_bootstrap(
            pd.DataFrame(rows), "new", "base", samples=100, seed=7
        )
        utility = result.loc[result["metric"].eq("lead_utility")].iloc[0]
        self.assertGreater(float(utility["difference"]), 0)
        self.assertEqual(int(utility["stations"]), 2)
        self.assertEqual(int(utility["paired_events"]), 4)

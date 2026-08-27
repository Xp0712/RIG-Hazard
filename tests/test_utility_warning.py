from __future__ import annotations

import pandas as pd

from rig_hazard.utility_warning import (
    paired_event_station_bootstrap,
    select_utility_candidate,
)


def test_select_utility_candidate_maximizes_utility_within_budget() -> None:
    rows = pd.DataFrame(
        {
            "selection_model": ["model", "model", "model"],
            "monthly_budget_hours": [10.0, 10.0, 10.0],
            "budget_met": [1, 1, 0],
            "lead_utility_hours": [0.2, 0.8, 1.0],
            "event_hit_rate": [0.4, 0.5, 0.9],
            "mean_effective_lead_hours": [0.5, 1.6, 2.0],
            "false_alarm_hours_per_station_month": [9.9, 5.0, 11.0],
            "candidate_quantile": [0.8, 0.94, 0.7],
        }
    )
    selected = select_utility_candidate(rows, "model", 10.0)
    assert selected["candidate_quantile"] == 0.94


def test_paired_event_bootstrap_preserves_direction() -> None:
    records = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "event_id": ["e1", "e2", "e1", "e2"],
            "station_code": ["s1", "s2", "s1", "s2"],
            "evaluable": [1, 1, 1, 1],
            "hit": [1, 1, 0, 1],
            "effective_lead_hours": [2.0, 1.0, float("nan"), 0.5],
            "lead_utility_hours": [2.0, 1.0, 0.0, 0.5],
        }
    )
    result = paired_event_station_bootstrap(records, "a", "b", 200, 7)
    estimates = result.set_index("metric")["estimate"]
    assert estimates["delta_hit_rate"] == 0.5
    assert estimates["delta_lead_utility_hours"] == 1.25

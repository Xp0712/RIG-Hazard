from __future__ import annotations

import numpy as np
import pandas as pd

from rig_hazard.recurrence_models import CurrentRiskData
from rig_hazard.seasonal_risk_structure import (
    HISTORY_FEATURES,
    INTERACTION_FEATURES,
    MODEL_NAMES,
    WEATHER_FEATURES,
    SeasonalHazardDesign,
    SeasonalRiskData,
    add_seasonal_state,
    icing_season_start_year,
)


def test_icing_season_start_year_excludes_warm_season() -> None:
    times = pd.to_datetime(["2023-01-01", "2023-06-01", "2023-11-01"]).astype("int64")
    assert icing_season_start_year(times.to_numpy()).tolist() == [2022, -1, 2023]


def test_seasonal_state_changes_only_after_prior_event() -> None:
    times = pd.to_datetime(
        ["2023-11-01 00:00", "2023-11-02 00:00", "2023-11-03 00:00"]
    ).astype("int64").to_numpy()
    data = CurrentRiskData(
        features=np.zeros((3, 1), dtype=np.float32),
        labels={name: np.zeros(3, dtype=np.int8) for name in ("step", "1h", "3h", "6h")},
        observed={name: np.ones(3, dtype=bool) for name in ("step", "1h", "3h", "6h")},
        sample_weight=np.ones(3),
        issue_time_ns=times,
        station_code=np.asarray(["S1", "S1", "S1"]),
        next_event_order=np.ones(3, dtype=np.int32),
        feature_names=["temperature_mean"],
    )
    events = pd.DataFrame(
        {
            "station_code": ["S1"],
            "onset_time": ["2023-11-02 00:00"],
            "valid_target_event": [1],
        }
    )
    result = add_seasonal_state(
        data, events, pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31 23:50")
    )
    assert result.recurrence_state.tolist() == [0, 1, 1]
    assert result.next_seasonal_order.tolist() == [1, 2, 2]


def test_nested_design_adds_only_prespecified_terms() -> None:
    feature_names = list(dict.fromkeys((*WEATHER_FEATURES, *HISTORY_FEATURES)))
    data = SeasonalRiskData(
        features=np.ones((4, len(feature_names)), dtype=np.float32),
        labels={name: np.zeros(4, dtype=np.int8) for name in ("step", "1h", "3h", "6h")},
        observed={name: np.ones(4, dtype=bool) for name in ("step", "1h", "3h", "6h")},
        sample_weight=np.ones(4),
        issue_time_ns=np.arange(4, dtype=np.int64),
        station_code=np.asarray(["S1", "S1", "S2", "S2"]),
        season_start_year=np.full(4, 2023, dtype=np.int16),
        recurrence_state=np.asarray([0, 1, 0, 1], dtype=np.int8),
        next_seasonal_order=np.asarray([1, 2, 1, 3], dtype=np.int16),
        complete_season=np.ones(4, dtype=bool),
        feature_names=feature_names,
    )
    widths = {
        model: SeasonalHazardDesign.fit(model, data).transform(data).shape[1]
        for model in MODEL_NAMES
    }
    assert widths["recurrence_state"] - widths["weather_only"] == 1
    assert widths["weather_recurrence_interaction"] - widths["recurrence_state"] == len(INTERACTION_FEATURES)
    assert widths["event_history"] - widths["weather_recurrence_interaction"] == len(HISTORY_FEATURES)

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_project_path


def icing_season_year(values: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(values, errors="coerce")
    return np.where(timestamps.dt.month.ge(11), timestamps.dt.year, timestamps.dt.year - 1)


def run_seasonal_recurrence(
    events_path: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> Path:
    source = resolve_project_path(events_path)
    destination = resolve_project_path(output_root)
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(source, low_memory=False)
    events["onset_time"] = pd.to_datetime(events["onset_time"], errors="coerce")
    events["end_time"] = pd.to_datetime(events["end_time"], errors="coerce")
    events["valid_target_event"] = pd.to_numeric(
        events["valid_target_event"], errors="coerce"
    ).fillna(0).astype(int)
    valid = events.loc[events["valid_target_event"].eq(1) & events["onset_time"].notna()].copy()
    valid["icing_season_start_year"] = icing_season_year(valid["onset_time"])
    valid = valid.sort_values(["station_code", "icing_season_start_year", "onset_time"])
    valid["global_event_order"] = valid.groupby("station_code").cumcount() + 1
    valid["within_season_event_order"] = (
        valid.groupby(["station_code", "icing_season_start_year"]).cumcount() + 1
    )
    valid["global_event_class"] = np.where(valid["global_event_order"].eq(1), "first", "recurrent")
    valid["seasonal_event_class"] = np.where(
        valid["within_season_event_order"].eq(1), "first", "recurrent"
    )
    data_start = events["onset_time"].min()
    data_end = events["end_time"].max()
    season_start = pd.to_datetime(valid["icing_season_start_year"].astype(str) + "-11-01")
    season_end = pd.to_datetime((valid["icing_season_start_year"] + 1).astype(str) + "-04-30 23:59:59")
    valid["left_censored_season"] = (season_start < data_start).astype(np.int8)
    valid["right_censored_season"] = (season_end > data_end).astype(np.int8)
    valid["complete_icing_season"] = (
        valid["left_censored_season"].eq(0) & valid["right_censored_season"].eq(0)
    ).astype(np.int8)
    groups = valid.groupby(["station_code", "icing_season_start_year"], as_index=False)
    station_season = groups.agg(
        event_count=("event_id", "size"),
        first_onset=("onset_time", "min"),
        last_end=("end_time", "max"),
        left_censored=("left_censored_season", "max"),
        right_censored=("right_censored_season", "max"),
    )
    station_season["first_event_count"] = 1
    station_season["recurrent_event_count"] = station_season["event_count"].sub(1).clip(lower=0)
    summary = pd.DataFrame(
        [
            {
                "definition": "global",
                "station_or_station_season_units": int(valid["station_code"].nunique()),
                "first_events": int(valid["global_event_order"].eq(1).sum()),
                "recurrent_events": int(valid["global_event_order"].ge(2).sum()),
                "second_events": int(valid["global_event_order"].eq(2).sum()),
                "third_or_later_events": int(valid["global_event_order"].ge(3).sum()),
            },
            {
                "definition": "november_to_april_season_reset",
                "station_or_station_season_units": int(station_season.shape[0]),
                "first_events": int(valid["within_season_event_order"].eq(1).sum()),
                "recurrent_events": int(valid["within_season_event_order"].ge(2).sum()),
                "second_events": int(valid["within_season_event_order"].eq(2).sum()),
                "third_or_later_events": int(valid["within_season_event_order"].ge(3).sum()),
            },
        ]
    )
    valid.to_csv(destination / "event_global_seasonal_mapping.csv", index=False, encoding="utf-8-sig")
    station_season.to_csv(destination / "station_icing_season_counts.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(destination / "global_vs_seasonal_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "icing_season": "November 1 through April 30, labelled by November year",
        "data_start": str(data_start),
        "data_end": str(data_end),
        "left_censor_rule": "season start precedes observed data start",
        "right_censor_rule": "season end follows observed data end",
        "selection_policy": "seasonal definition is sensitivity analysis; both global and seasonal orders are retained",
    }
    (destination / "seasonal_recurrence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Seasonal recurrence analysis complete: {destination}", flush=True)
    return destination

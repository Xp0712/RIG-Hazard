from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIMELINE_REASON_COLUMNS = (
    "issue_time",
    "risk_set",
    "physical_at_risk",
    "history_valid",
    "future_step_label_valid",
)


def _timeline_file(timeline_root: Path, year: int, station_code: str) -> Path | None:
    candidates = sorted((timeline_root / str(year)).glob(f"{station_code}_*.csv.gz"))
    return candidates[0] if len(candidates) == 1 else None


def add_event_interval_metadata(
    audit: pd.DataFrame,
    events: pd.DataFrame,
    entity_column: str = "station_code",
) -> pd.DataFrame:
    result = audit.copy()
    source = events.copy()
    source["onset_time"] = pd.to_datetime(source["onset_time"], errors="coerce")
    source["end_time"] = pd.to_datetime(source.get("end_time"), errors="coerce")
    source = source.sort_values([entity_column, "onset_time"])
    source["previous_event_end"] = source.groupby(entity_column)["end_time"].shift(1)
    source["previous_event_gap_hours"] = (
        source["onset_time"] - source["previous_event_end"]
    ).dt.total_seconds() / 3600.0
    metadata = source[
        ["event_id", "previous_event_end", "previous_event_gap_hours"]
    ].drop_duplicates("event_id")
    return result.merge(metadata, on="event_id", how="left", validate="one_to_one")


def annotate_timeline_exclusion_reasons(
    audit: pd.DataFrame,
    timeline_root: str | Path,
    step_minutes: int,
) -> pd.DataFrame:
    """Add per-event risk/history/follow-up reasons without changing eligibility."""

    result = audit.copy()
    result["onset_time"] = pd.to_datetime(result["onset_time"], errors="coerce")
    result["window_start"] = pd.to_datetime(result["window_start"], errors="coerce")
    result["window_end"] = pd.to_datetime(result["window_end"], errors="coerce")
    result["year"] = result["onset_time"].dt.year.astype("Int64")
    timeline_base = Path(timeline_root)
    annotated: list[dict[str, Any]] = []
    for (year, station), group in result.groupby(["year", "station_code"], dropna=False):
        path = None if pd.isna(year) else _timeline_file(timeline_base, int(year), str(station))
        timeline: pd.DataFrame | None = None
        if path is not None:
            timeline = pd.read_csv(path, usecols=list(TIMELINE_REASON_COLUMNS), low_memory=False)
            timeline["issue_time"] = pd.to_datetime(timeline["issue_time"], errors="coerce")
            timeline = (
                timeline.dropna(subset=["issue_time"])
                .drop_duplicates("issue_time")
                .set_index("issue_time")
                .sort_index()
            )
        for row in group.to_dict(orient="records"):
            expected_bins = int(row["expected_prediction_bins"])
            if timeline is None:
                counts = {
                    "prediction_grid_missing_bins": expected_bins,
                    "physical_risk_excluded_bins": 0,
                    "history_excluded_bins": 0,
                    "next_step_label_unobservable_bins": 0,
                    "risk_set_valid_bins": 0,
                }
            else:
                expected = pd.date_range(
                    start=pd.Timestamp(row["window_start"]),
                    end=pd.Timestamp(row["window_end"]),
                    freq=f"{int(step_minutes)}min",
                )
                window = timeline.reindex(expected)
                risk = pd.to_numeric(window["risk_set"], errors="coerce")
                physical = pd.to_numeric(window["physical_at_risk"], errors="coerce")
                history = pd.to_numeric(window["history_valid"], errors="coerce")
                follow_up = pd.to_numeric(
                    window["future_step_label_valid"], errors="coerce"
                )
                counts = {
                    "prediction_grid_missing_bins": int(risk.isna().sum()),
                    "physical_risk_excluded_bins": int(physical.fillna(1).eq(0).sum()),
                    "history_excluded_bins": int(history.fillna(1).eq(0).sum()),
                    "next_step_label_unobservable_bins": int(
                        follow_up.fillna(1).eq(0).sum()
                    ),
                    "risk_set_valid_bins": int(risk.fillna(0).eq(1).sum()),
                }
            row.update(counts)
            reasons: list[str] = []
            if row.get("primary_exclusion_reason") == "station_id_not_matched":
                reasons.append("station_id_not_matched")
            if counts["prediction_grid_missing_bins"]:
                reasons.append("prediction_grid_missing")
            if counts["history_excluded_bins"]:
                reasons.append("incomplete_24h_history")
            if counts["next_step_label_unobservable_bins"]:
                reasons.append("incomplete_next_step_label")
            if counts["physical_risk_excluded_bins"]:
                reasons.append("outside_physical_risk_or_cooldown")
            if not int(row["standardized_queue"]) and not reasons:
                reasons.append("prediction_row_missing_or_cache_contract")
            row["primary_exclusion_reason"] = reasons[0] if reasons else ""
            row["all_exclusion_reasons"] = ";".join(reasons)
            row["operational_primary_exclusion_reason"] = (
                row["primary_exclusion_reason"]
                if not int(row["operational_queue"])
                else ""
            )
            annotated.append(row)
    enriched = pd.DataFrame(annotated)
    if enriched.duplicated("event_id").any():
        duplicates = enriched.loc[enriched.duplicated("event_id", keep=False), "event_id"]
        raise ValueError(f"event_eligibility_audit must contain one row per event: {duplicates.tolist()}")
    return enriched.sort_values(["year", "station_code", "onset_time"]).reset_index(drop=True)


def eligibility_funnel(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in audit.groupby("year", dropna=False):
        rows.append(
            {
                "year": year,
                "target_events": int(group.shape[0]),
                "station_matched_events": int(
                    group["primary_exclusion_reason"].ne("station_id_not_matched").sum()
                ),
                "operational_queue_events": int(group["operational_queue"].sum()),
                "standardized_queue_events": int(group["standardized_queue"].sum()),
                "excluded_from_standardized_queue": int(group["standardized_queue"].eq(0).sum()),
                "excluded_from_operational_queue": int(group["operational_queue"].eq(0).sum()),
                "events_with_physical_risk_or_cooldown_conflict": int(
                    group["physical_risk_excluded_bins"].gt(0).sum()
                ),
                "events_with_incomplete_24h_history": int(
                    group["history_excluded_bins"].gt(0).sum()
                ),
                "events_with_incomplete_next_step_label": int(
                    group["next_step_label_unobservable_bins"].gt(0).sum()
                ),
                "events_with_prediction_grid_missing": int(
                    group["prediction_grid_missing_bins"].gt(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def eligibility_bias_summary(audit: pd.DataFrame) -> pd.DataFrame:
    parts: list[dict[str, Any]] = []
    for queue_column, queue_name in (
        ("standardized_queue", "complete_36_bin"),
        ("operational_queue", "at_least_one_legal_prediction"),
    ):
        for (year, included), group in audit.groupby(["year", queue_column], dropna=False):
            order = pd.to_numeric(group.get("within_season_event_order"), errors="coerce")
            gap = pd.to_numeric(group.get("previous_event_gap_hours"), errors="coerce")
            parts.append(
                {
                    "year": year,
                    "queue": queue_name,
                    "included": int(included),
                    "events": int(group.shape[0]),
                    "stations": int(group["station_code"].nunique()),
                    "recurrent_event_rate": float(order.gt(1).mean()),
                    "mean_within_season_event_order": float(order.mean()),
                    "median_previous_event_gap_hours": float(gap.median()),
                    "mean_previous_event_gap_hours": float(gap.mean()),
                    "mean_available_warning_window_hours": float(
                        pd.to_numeric(
                            group["available_warning_window_hours"], errors="coerce"
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(parts)


def eligibility_station_season_distribution(audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "year",
        "station_code",
        "icing_season_start_year",
        "standardized_queue",
        "operational_queue",
    ]
    available = [column for column in columns if column in audit]
    return (
        audit.groupby(available, dropna=False, as_index=False)
        .agg(events=("event_id", "size"))
        .sort_values(available)
        .reset_index(drop=True)
    )

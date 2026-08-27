from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import resolve_project_path


SEASON_BY_MONTH = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _quantiles(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "minimum": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "maximum": float(clean.max()),
    }


def _event_order_group(order: int) -> str:
    if order <= 1:
        return "first"
    if order == 2:
        return "second"
    return "third_plus"


def prepare_valid_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id",
        "station_code",
        "onset_time",
        "end_time",
        "valid_target_event",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Event table is missing columns: {', '.join(missing)}")
    frame = events.loc[pd.to_numeric(events["valid_target_event"], errors="coerce").eq(1)].copy()
    frame["onset_time"] = pd.to_datetime(frame["onset_time"], errors="coerce")
    frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce")
    frame = frame.dropna(subset=["onset_time", "end_time", "station_code"])
    frame["station_code"] = frame["station_code"].astype(str)
    frame = frame.sort_values(["station_code", "onset_time", "end_time", "event_id"]).reset_index(drop=True)
    frame["valid_event_order"] = frame.groupby("station_code").cumcount().add(1).astype(int)
    frame["event_order_group"] = frame["valid_event_order"].map(_event_order_group)
    frame["recurrence_group"] = np.where(frame["valid_event_order"].eq(1), "first", "recurrent")
    frame["onset_year"] = frame["onset_time"].dt.year.astype(int)
    frame["onset_month"] = frame["onset_time"].dt.month.astype(int)
    frame["season"] = frame["onset_month"].map(SEASON_BY_MONTH)
    previous_onset = frame.groupby("station_code")["onset_time"].shift(1)
    previous_end = frame.groupby("station_code")["end_time"].shift(1)
    frame["onset_to_onset_hours"] = (frame["onset_time"] - previous_onset).dt.total_seconds().div(3600)
    frame["gap_from_previous_end_hours"] = (frame["onset_time"] - previous_end).dt.total_seconds().div(3600)
    frame.loc[frame["valid_event_order"].eq(1), ["onset_to_onset_hours", "gap_from_previous_end_hours"]] = np.nan
    return frame


def station_event_summary(valid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station_code, group in valid.groupby("station_code", sort=True):
        count = int(group.shape[0])
        row: dict[str, Any] = {
            "station_code": station_code,
            "station_name": group["station_name"].iloc[0] if "station_name" in group else "",
            "event_count": count,
            "first_event_count": int(count >= 1),
            "second_event_count": int(count >= 2),
            "third_plus_event_count": max(count - 2, 0),
            "recurrent_event_count": max(count - 1, 0),
            "has_recurrence": int(count >= 2),
            "first_onset": group["onset_time"].min(),
            "last_onset": group["onset_time"].max(),
        }
        row.update({f"gap_{key}_hours": value for key, value in _quantiles(group["gap_from_previous_end_hours"]).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["event_count", "station_code"], ascending=[False, True])


def event_order_summary(valid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        ("first", valid.loc[valid["valid_event_order"].eq(1)]),
        ("second", valid.loc[valid["valid_event_order"].eq(2)]),
        ("third_plus", valid.loc[valid["valid_event_order"].ge(3)]),
        ("recurrent", valid.loc[valid["valid_event_order"].ge(2)]),
        ("all", valid),
    ]
    for label, group in groups:
        rows.append(
            {
                "event_group": label,
                "event_count": int(group.shape[0]),
                "station_count": int(group["station_code"].nunique()),
                "year_count": int(group["onset_year"].nunique()),
                "duration_median_minutes": float(pd.to_numeric(group.get("elapsed_minutes"), errors="coerce").median())
                if not group.empty and "elapsed_minutes" in group
                else None,
                "max_thickness_median": float(pd.to_numeric(group.get("max_thickness"), errors="coerce").median())
                if not group.empty and "max_thickness" in group
                else None,
            }
        )
    return pd.DataFrame(rows)


def recurrence_gap_tables(valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gaps = valid.loc[valid["valid_event_order"].ge(2)].copy()
    detail_columns = [
        "event_id",
        "station_code",
        "station_name",
        "valid_event_order",
        "onset_time",
        "end_time",
        "onset_to_onset_hours",
        "gap_from_previous_end_hours",
        "onset_year",
        "onset_month",
        "season",
    ]
    detail = gaps[[column for column in detail_columns if column in gaps]].copy()
    rows: list[dict[str, Any]] = []
    for label, group in [
        ("second", gaps.loc[gaps["valid_event_order"].eq(2)]),
        ("third_plus", gaps.loc[gaps["valid_event_order"].ge(3)]),
        ("all_recurrent", gaps),
    ]:
        for metric in ("gap_from_previous_end_hours", "onset_to_onset_hours"):
            rows.append({"event_group": label, "metric": metric, **_quantiles(group[metric])})
    return detail, pd.DataFrame(rows)


def seasonality_tables(valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month = (
        valid.groupby(["onset_year", "onset_month", "recurrence_group"], observed=True)
        .size()
        .rename("event_count")
        .reset_index()
    )
    season = (
        valid.groupby(["onset_year", "season", "recurrence_group"], observed=True)
        .size()
        .rename("event_count")
        .reset_index()
    )
    station_rows: list[dict[str, Any]] = []
    for station_code, group in valid.groupby("station_code", sort=True):
        counts = group["onset_month"].value_counts().reindex(range(1, 13), fill_value=0).to_numpy(dtype=float)
        total = float(counts.sum())
        shares = counts / total if total else counts
        positive = shares[shares > 0]
        entropy = float(-np.sum(positive * np.log(positive)) / math.log(12)) if positive.size else 0.0
        peak_month = int(np.argmax(counts) + 1) if total else 0
        winter_share = float(group["season"].eq("winter").mean()) if total else 0.0
        station_rows.append(
            {
                "station_code": station_code,
                "station_name": group["station_name"].iloc[0] if "station_name" in group else "",
                "event_count": int(total),
                "peak_month": peak_month,
                "peak_month_share": float(shares.max()) if total else 0.0,
                "monthly_hhi": float(np.square(shares).sum()) if total else 0.0,
                "normalized_monthly_entropy": entropy,
                "winter_event_share": winter_share,
            }
        )
    return month, season, pd.DataFrame(station_rows)


def _performance_row(keys: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    evaluable = group.loc[pd.to_numeric(group["evaluable"], errors="coerce").eq(1)].copy()
    hits = evaluable.loc[pd.to_numeric(evaluable["hit"], errors="coerce").eq(1)]
    lead = pd.to_numeric(hits.get("effective_lead_hours"), errors="coerce").dropna()
    utility = pd.to_numeric(evaluable.get("lead_utility_hours"), errors="coerce").fillna(0.0)
    return {
        **keys,
        "event_records": int(group.shape[0]),
        "evaluable_events": int(evaluable.shape[0]),
        "hit_events": int(hits.shape[0]),
        "event_sensitivity": float(hits.shape[0] / evaluable.shape[0]) if not evaluable.empty else None,
        "mean_effective_lead_hours_among_hits": float(lead.mean()) if not lead.empty else None,
        "median_effective_lead_hours_among_hits": float(lead.median()) if not lead.empty else None,
        "mean_lead_utility_hours": float(utility.mean()) if not utility.empty else None,
    }


def stratified_event_performance(valid: pd.DataFrame, record_paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    for path in record_paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["record_source"] = str(path)
        records.append(frame)
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    record = pd.concat(records, ignore_index=True)
    required = {"model", "event_id", "station_code", "evaluable", "hit"}
    missing = sorted(required - set(record.columns))
    if missing:
        raise ValueError(f"Event record table is missing columns: {', '.join(missing)}")
    event_lookup = valid[
        ["event_id", "station_code", "onset_year", "valid_event_order", "event_order_group", "recurrence_group"]
    ].drop_duplicates(["event_id", "station_code"])
    joined = record.merge(event_lookup, on=["event_id", "station_code"], how="inner", validate="many_to_one")
    if joined.empty:
        return pd.DataFrame(), joined
    rows: list[dict[str, Any]] = []
    group_keys = ["record_source", "model", "onset_year"]
    for keys, group in joined.groupby(group_keys, dropna=False, sort=True):
        base = dict(zip(group_keys, keys))
        for label, subset in [
            ("first", group.loc[group["valid_event_order"].eq(1)]),
            ("second", group.loc[group["valid_event_order"].eq(2)]),
            ("third_plus", group.loc[group["valid_event_order"].ge(3)]),
            ("recurrent", group.loc[group["valid_event_order"].ge(2)]),
            ("all", group),
        ]:
            rows.append(_performance_row({**base, "event_group": label}, subset))
    return pd.DataFrame(rows), joined


def clustered_first_recurrent_bootstrap(joined: pd.DataFrame, replicates: int = 1000, seed: int = 20260807) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group_keys = ["record_source", "model", "onset_year"]
    for keys, frame in joined.groupby(group_keys, dropna=False, sort=True):
        data = frame.loc[pd.to_numeric(frame["evaluable"], errors="coerce").eq(1)].copy()
        stations = data["station_code"].dropna().unique()
        first = data.loc[data["valid_event_order"].eq(1)]
        recurrent = data.loc[data["valid_event_order"].ge(2)]
        base = dict(zip(group_keys, keys))
        if stations.size < 2 or first.empty or recurrent.empty:
            rows.append({**base, "replicates": 0, "station_count": int(stations.size)})
            continue
        estimates = []
        for _ in range(int(replicates)):
            sampled = rng.choice(stations, size=stations.size, replace=True)
            pieces = [data.loc[data["station_code"].eq(station)] for station in sampled]
            draw = pd.concat(pieces, ignore_index=True)
            draw_first = draw.loc[draw["valid_event_order"].eq(1)]
            draw_recurrent = draw.loc[draw["valid_event_order"].ge(2)]
            if draw_first.empty or draw_recurrent.empty:
                continue
            first_hit = pd.to_numeric(draw_first["hit"], errors="coerce").mean()
            recurrent_hit = pd.to_numeric(draw_recurrent["hit"], errors="coerce").mean()
            first_utility = pd.to_numeric(draw_first.get("lead_utility_hours"), errors="coerce").fillna(0).mean()
            recurrent_utility = pd.to_numeric(draw_recurrent.get("lead_utility_hours"), errors="coerce").fillna(0).mean()
            estimates.append((recurrent_hit - first_hit, recurrent_utility - first_utility))
        array = np.asarray(estimates, dtype=float)
        row = {**base, "replicates": int(array.shape[0]), "station_count": int(stations.size)}
        if array.size:
            row.update(
                {
                    "sensitivity_difference_recurrent_minus_first": float(array[:, 0].mean()),
                    "sensitivity_difference_ci_low": float(np.quantile(array[:, 0], 0.025)),
                    "sensitivity_difference_ci_high": float(np.quantile(array[:, 0], 0.975)),
                    "utility_difference_recurrent_minus_first": float(array[:, 1].mean()),
                    "utility_difference_ci_low": float(np.quantile(array[:, 1], 0.025)),
                    "utility_difference_ci_high": float(np.quantile(array[:, 1], 0.975)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    output_root: Path,
    valid: pd.DataFrame,
    stations: pd.DataFrame,
    order: pd.DataFrame,
    gap_summary: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    recurrent_stations = int(stations["has_recurrence"].sum()) if not stations.empty else 0
    recurrent_events = int(valid["valid_event_order"].ge(2).sum())
    lines = [
        "# Recurrence structure report",
        "",
        "## Cohort",
        "",
        f"- Valid target events: {valid.shape[0]}",
        f"- Stations with at least one event: {stations.shape[0]}",
        f"- Stations with at least two events: {recurrent_stations}",
        f"- Recurrent events (order >= 2): {recurrent_events}",
        "",
        "The event order is recomputed after filtering valid target events and is global across all available years.",
        "Gap time is measured from the previous event end to the next onset; onset-to-onset time is retained separately.",
        "",
        "## Event order",
        "",
        order.to_markdown(index=False) if not order.empty else "No valid events.",
        "",
        "## Recurrence gap distribution",
        "",
        gap_summary.to_markdown(index=False) if not gap_summary.empty else "No recurrent events.",
        "",
        "## First versus recurrent warning performance",
        "",
        performance.to_markdown(index=False) if not performance.empty else "No compatible event-level warning records were supplied.",
        "",
        "## Interpretation boundary",
        "",
        "This report describes recurrence under the current event definition. It does not establish that every separated event is physically independent; use the definition-sensitivity experiment for that claim.",
    ]
    (output_root / "recurrence_structure_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_recurrence_structure(
    events_path: str | Path = "results/preprocessed_data_10min/events_recurrent.csv",
    output_root: str | Path = "results/recurrence_analysis/structure",
    event_record_paths: Iterable[str | Path] | None = None,
    bootstrap_replicates: int = 1000,
    overwrite: bool = False,
) -> Path:
    events_file = resolve_project_path(events_path)
    destination = resolve_project_path(output_root)
    _prepare_output(destination, overwrite=overwrite)
    if not events_file.exists():
        raise FileNotFoundError(events_file)

    valid = prepare_valid_events(pd.read_csv(events_file, low_memory=False))
    stations = station_event_summary(valid)
    order = event_order_summary(valid)
    gap_detail, gap_summary = recurrence_gap_tables(valid)
    month, season, station_seasonality = seasonality_tables(valid)
    supplied_records = list(event_record_paths or [])
    record_files = [resolve_project_path(path) for path in supplied_records]
    performance, joined = stratified_event_performance(valid, record_files)
    bootstrap = clustered_first_recurrent_bootstrap(joined, replicates=bootstrap_replicates)

    valid.to_csv(destination / "valid_events_with_order.csv", index=False)
    stations.to_csv(destination / "station_event_counts.csv", index=False)
    order.to_csv(destination / "event_order_summary.csv", index=False)
    gap_detail.to_csv(destination / "recurrence_gap_records.csv", index=False)
    gap_summary.to_csv(destination / "recurrence_gap_summary.csv", index=False)
    month.to_csv(destination / "monthly_event_counts.csv", index=False)
    season.to_csv(destination / "seasonal_event_counts.csv", index=False)
    station_seasonality.to_csv(destination / "station_seasonality.csv", index=False)
    performance.to_csv(destination / "first_recurrent_performance.csv", index=False)
    joined.to_csv(destination / "event_records_with_recurrence_order.csv", index=False)
    bootstrap.to_csv(destination / "first_recurrent_cluster_bootstrap.csv", index=False)
    _write_report(destination, valid, stations, order, gap_summary, performance)
    manifest = {
        "analysis": "recurrence_structure",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "events_path": str(events_file),
        "event_record_paths": [str(path) for path in record_files],
        "bootstrap_replicates": int(bootstrap_replicates),
        "valid_event_count": int(valid.shape[0]),
        "station_count": int(valid["station_code"].nunique()),
        "artifacts": sorted(path.name for path in destination.iterdir() if path.is_file()),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Recurrence structure analysis complete: {destination}", flush=True)
    return destination

from __future__ import annotations

import copy
import itertools
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, resolve_project_path
from .preprocessing import (
    RAW_COLUMN_MAP,
    canonicalize_chunk,
    discover_sources,
    extract_events,
    infer_city,
)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def estimate_instrument_resolution(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean.loc[clean.gt(0)]
    unique = np.unique(np.round(clean.to_numpy(dtype=float), 4))
    if unique.size < 2:
        return float(unique[0]) if unique.size == 1 else 0.01
    differences = np.round(np.diff(unique), 4)
    differences = differences[differences >= 0.0001]
    if differences.size == 0:
        return 0.01
    counts = pd.Series(differences).value_counts().sort_index()
    repeated = counts.loc[counts.ge(max(3, int(np.ceil(unique.size * 0.01))))]
    if not repeated.empty:
        return float(repeated.index[0])
    return float(np.quantile(differences, 0.1))


def resolution_persistent_records(
    positives: pd.DataFrame,
    resolution: float,
    minimum_records: int = 3,
    maximum_record_gap_minutes: float = 2.0,
) -> pd.DataFrame:
    if positives.empty:
        return positives.copy()
    frame = positives.dropna(subset=["timestamp", "ice_thickness"]).copy()
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    frame = frame.loc[pd.to_numeric(frame["ice_thickness"], errors="coerce").gt(float(resolution) + 1e-12)].copy()
    if frame.empty:
        return frame
    gap = frame["timestamp"].diff().dt.total_seconds().div(60)
    frame["persistence_group"] = gap.gt(float(maximum_record_gap_minutes)).fillna(True).cumsum()
    sizes = frame.groupby("persistence_group")["timestamp"].transform("size")
    return frame.loc[sizes.ge(int(minimum_records))].drop(columns="persistence_group")


def sensitivity_grid(persistence_records: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for merge, cooldown_hours, temperature, thickness_rule in itertools.product(
        (10, 30, 60),
        (1, 3, 6),
        (2, 0),
        ("positive", "resolution_persistent"),
    ):
        rule_suffix = "positive" if thickness_rule == "positive" else f"resolution_persistent_{persistence_records}"
        rows.append(
            {
                "definition_id": f"merge{merge}_cool{cooldown_hours}h_temp{temperature}_{rule_suffix}",
                "merge_gap_minutes": merge,
                "cooldown_hours": cooldown_hours,
                "cooldown_minutes": cooldown_hours * 60,
                "cold_temperature_c": temperature,
                "thickness_rule": thickness_rule,
                "persistence_records": 1 if thickness_rule == "positive" else int(persistence_records),
                "is_reference": int(
                    merge == 10 and cooldown_hours == 1 and temperature == 2 and thickness_rule == "positive"
                ),
            }
        )
    return rows


def _read_sensitivity_chunks(source: Any, chunksize: int):
    needed = {
        "timestamp",
        "ice_thickness",
        "air_temperature",
        "relative_humidity",
        "visibility",
        "visibility_alias",
        "visibility_obstacle",
    }
    usecols = {raw for raw, canonical in RAW_COLUMN_MAP.items() if canonical in needed}
    yield from pd.read_csv(
        source.path,
        encoding=source.encoding,
        dtype=str,
        usecols=lambda column: column in usecols,
        chunksize=chunksize,
        keep_default_na=False,
        low_memory=False,
    )


def collect_positive_records(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = resolve_project_path(config["data_root"])
    sources = discover_sources(data_root)
    anomaly_counts: dict[str, int] = defaultdict(int)
    parts: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        source_positive = 0
        source_raw = 0
        if not source.empty:
            for chunk in _read_sensitivity_chunks(source, int(config["chunksize"])):
                source_raw += int(chunk.shape[0])
                canonical = canonicalize_chunk(chunk, source.year, config["valid_ranges"], anomaly_counts)
                if canonical.empty:
                    continue
                positive = canonical.loc[
                    canonical["ice_thickness"].gt(0),
                    ["timestamp", "ice_thickness", "air_temperature", "relative_humidity", "visibility", "fog_flag"],
                ].copy()
                if positive.empty:
                    continue
                positive["station_code"] = str(source.station_code)
                positive["station_name"] = str(source.station_name)
                positive["city"] = infer_city(str(source.station_name))
                positive["source_year"] = int(source.year)
                source_positive += int(positive.shape[0])
                parts.append(positive)
        source_rows.append(
            {
                "relative_path": source.relative_path,
                "station_code": source.station_code,
                "station_name": source.station_name,
                "year": source.year,
                "raw_rows": source_raw,
                "positive_rows": source_positive,
                "empty": int(source.empty),
            }
        )
        if index % 10 == 0 or index == len(sources):
            print(f"Positive-record scan: {index}/{len(sources)} files", flush=True)
    positives = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return positives, pd.DataFrame(source_rows)


def _valid_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    valid = frame.loc[pd.to_numeric(frame["valid_target_event"], errors="coerce").eq(1)].copy()
    valid["onset_time"] = pd.to_datetime(valid["onset_time"], errors="coerce")
    valid["end_time"] = pd.to_datetime(valid["end_time"], errors="coerce")
    grouping = ["definition_id", "station_code"] if "definition_id" in valid else ["station_code"]
    valid = valid.sort_values([*grouping, "onset_time", "end_time"])
    valid["sensitivity_event_order"] = valid.groupby(grouping).cumcount().add(1)
    valid["gap_from_previous_end_hours"] = (
        valid["onset_time"] - valid.groupby(grouping)["end_time"].shift(1)
    ).dt.total_seconds().div(3600)
    return valid


def evaluate_sensitivity_grid(
    positives: pd.DataFrame,
    base_config: dict[str, Any],
    persistence_records: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "timestamp",
        "ice_thickness",
        "air_temperature",
        "relative_humidity",
        "visibility",
        "fog_flag",
        "station_code",
        "station_name",
        "city",
    }
    missing = sorted(required - set(positives.columns))
    if missing:
        raise ValueError(f"Positive record table is missing columns: {', '.join(missing)}")
    positives = positives.copy()
    positives["timestamp"] = pd.to_datetime(positives["timestamp"], errors="coerce")
    positives = positives.dropna(subset=["timestamp"])
    resolution_rows = []
    station_groups: dict[str, pd.DataFrame] = {}
    station_metadata: dict[str, tuple[str, str]] = {}
    for station_code, group in positives.groupby("station_code", sort=True):
        code = str(station_code)
        station_groups[code] = group.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        station_metadata[code] = (str(group["station_name"].iloc[0]), str(group["city"].iloc[0]))
        resolution = estimate_instrument_resolution(group["ice_thickness"])
        resolution_rows.append(
            {
                "station_code": code,
                "station_name": station_metadata[code][0],
                "positive_record_count": int(group.shape[0]),
                "estimated_resolution": resolution,
                "estimation_method": "smallest_repeated_positive_level_increment",
            }
        )
    resolutions = pd.DataFrame(resolution_rows)
    resolution_map = resolutions.set_index("station_code")["estimated_resolution"].to_dict()
    filtered_cache: dict[tuple[str, str], pd.DataFrame] = {}
    all_event_parts: list[pd.DataFrame] = []
    station_summary_rows: list[dict[str, Any]] = []
    definitions = sensitivity_grid(persistence_records)
    for definition_index, definition in enumerate(definitions, start=1):
        local_config = copy.deepcopy(base_config)
        local_config["event"].update(
            {
                "ice_threshold": 0.0,
                "merge_gap_minutes": int(definition["merge_gap_minutes"]),
                "cooldown_minutes": int(definition["cooldown_minutes"]),
                "cold_plausible_temperature_c": float(definition["cold_temperature_c"]),
                "minimum_positive_minutes": 1,
                "use_only_cold_plausible_events_as_targets": True,
            }
        )
        definition_parts = []
        for station_code, group in station_groups.items():
            cache_key = (station_code, str(definition["thickness_rule"]))
            if cache_key not in filtered_cache:
                if definition["thickness_rule"] == "positive":
                    filtered_cache[cache_key] = group.copy()
                else:
                    filtered_cache[cache_key] = resolution_persistent_records(
                        group,
                        resolution=float(resolution_map[station_code]),
                        minimum_records=int(persistence_records),
                    )
            station_name, city = station_metadata[station_code]
            events = extract_events(
                filtered_cache[cache_key].copy(),
                station_code,
                station_name,
                city,
                local_config,
            )
            if not events:
                station_summary_rows.append(
                    {
                        **definition,
                        "station_code": station_code,
                        "station_name": station_name,
                        "all_event_count": 0,
                        "valid_event_count": 0,
                        "recurrent_event_count": 0,
                    }
                )
                continue
            event_frame = pd.DataFrame(events)
            event_frame.insert(0, "definition_id", definition["definition_id"])
            for key in (
                "merge_gap_minutes",
                "cooldown_hours",
                "cold_temperature_c",
                "thickness_rule",
                "persistence_records",
                "is_reference",
            ):
                event_frame[key] = definition[key]
            definition_parts.append(event_frame)
            valid_count = int(pd.to_numeric(event_frame["valid_target_event"], errors="coerce").eq(1).sum())
            station_summary_rows.append(
                {
                    **definition,
                    "station_code": station_code,
                    "station_name": station_name,
                    "all_event_count": int(event_frame.shape[0]),
                    "valid_event_count": valid_count,
                    "recurrent_event_count": max(valid_count - 1, 0),
                }
            )
        definition_frame = pd.concat(definition_parts, ignore_index=True) if definition_parts else pd.DataFrame()
        all_event_parts.append(definition_frame)
        print(f"Sensitivity definition: {definition_index}/{len(definitions)}", flush=True)
    all_events = pd.concat(all_event_parts, ignore_index=True) if all_event_parts else pd.DataFrame()
    station_summary = pd.DataFrame(station_summary_rows)
    return all_events, station_summary, resolutions, pd.DataFrame(definitions)


def _greedy_onset_match(reference: pd.DataFrame, candidate: pd.DataFrame, tolerance_minutes: float = 60.0) -> dict[str, Any]:
    matched_shifts: list[float] = []
    matched = 0
    for station_code in sorted(set(reference.get("station_code", [])) | set(candidate.get("station_code", []))):
        ref = reference.loc[reference["station_code"].eq(station_code), "onset_time"].sort_values().tolist()
        cand = candidate.loc[candidate["station_code"].eq(station_code), "onset_time"].sort_values().tolist()
        pairs = []
        for i, ref_time in enumerate(ref):
            for j, candidate_time in enumerate(cand):
                shift = abs((pd.Timestamp(candidate_time) - pd.Timestamp(ref_time)).total_seconds() / 60.0)
                if shift <= tolerance_minutes:
                    pairs.append((shift, i, j))
        used_ref: set[int] = set()
        used_cand: set[int] = set()
        for shift, ref_index, candidate_index in sorted(pairs):
            if ref_index in used_ref or candidate_index in used_cand:
                continue
            used_ref.add(ref_index)
            used_cand.add(candidate_index)
            matched += 1
            matched_shifts.append(float(shift))
    reference_count = int(reference.shape[0])
    candidate_count = int(candidate.shape[0])
    return {
        "reference_event_count": reference_count,
        "candidate_event_count": candidate_count,
        "matched_event_count": matched,
        "reference_recall": float(matched / reference_count) if reference_count else None,
        "candidate_precision": float(matched / candidate_count) if candidate_count else None,
        "median_absolute_onset_shift_minutes": float(np.median(matched_shifts)) if matched_shifts else None,
        "maximum_absolute_onset_shift_minutes": float(np.max(matched_shifts)) if matched_shifts else None,
        "matching_tolerance_minutes": float(tolerance_minutes),
    }


def summarize_sensitivity(
    all_events: pd.DataFrame,
    definitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = _valid_events(all_events)
    summary_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    reference_id = str(definitions.loc[definitions["is_reference"].eq(1), "definition_id"].iloc[0])
    reference = valid.loc[valid["definition_id"].eq(reference_id)]
    for _, definition in definitions.iterrows():
        definition_id = str(definition["definition_id"])
        group_all = all_events.loc[all_events["definition_id"].eq(definition_id)]
        group = valid.loc[valid["definition_id"].eq(definition_id)]
        event_count = int(group.shape[0])
        station_counts = group.groupby("station_code").size() if not group.empty else pd.Series(dtype=int)
        gap = pd.to_numeric(group["gap_from_previous_end_hours"], errors="coerce").dropna()
        summary_rows.append(
            {
                **definition.to_dict(),
                "all_candidate_event_count": int(group_all.shape[0]),
                "valid_event_count": event_count,
                "station_count": int(group["station_code"].nunique()),
                "stations_with_recurrence": int(station_counts.ge(2).sum()),
                "first_event_count": int(station_counts.ge(1).sum()),
                "second_event_count": int(station_counts.ge(2).sum()),
                "third_plus_event_count": int(np.maximum(station_counts.to_numpy() - 2, 0).sum()),
                "recurrent_event_count": int(np.maximum(station_counts.to_numpy() - 1, 0).sum()),
                "median_gap_hours": float(gap.median()) if not gap.empty else None,
                "p25_gap_hours": float(gap.quantile(0.25)) if not gap.empty else None,
                "p75_gap_hours": float(gap.quantile(0.75)) if not gap.empty else None,
            }
        )
        for year, year_group in group.groupby(group["onset_time"].dt.year, sort=True):
            year_counts = year_group.groupby("station_code").size()
            year_rows.append(
                {
                    "definition_id": definition_id,
                    "year": int(year),
                    "event_count": int(year_group.shape[0]),
                    "station_count": int(year_group["station_code"].nunique()),
                    "within_year_recurrent_event_count": int(np.maximum(year_counts.to_numpy() - 1, 0).sum()),
                }
            )
        match_rows.append({"definition_id": definition_id, **_greedy_onset_match(reference, group)})
    return pd.DataFrame(summary_rows), pd.DataFrame(year_rows), pd.DataFrame(match_rows)


def _write_report(output_root: Path, summary: pd.DataFrame, resolutions: pd.DataFrame) -> None:
    reference = summary.loc[summary["is_reference"].eq(1)].iloc[0]
    range_min = int(summary["valid_event_count"].min())
    range_max = int(summary["valid_event_count"].max())
    lines = [
        "# Recurrence definition sensitivity",
        "",
        "## Design",
        "",
        "- Merge gap: 10, 30, or 60 minutes.",
        "- Cooldown without active ice: 1, 3, or 6 hours.",
        "- Cold plausibility threshold: <= 2 C or <= 0 C.",
        "- Thickness: > 0, or > station-specific estimated resolution for at least three consecutive records.",
        "",
        f"The full factorial grid contains {summary.shape[0]} definitions across {resolutions.shape[0]} stations.",
        f"The reference definition contains {int(reference['valid_event_count'])} valid events; across the grid the count ranges from {range_min} to {range_max}.",
        "",
        "## Definition summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Stability across this grid supports robustness to record interruption and cooldown choices. It does not by itself prove physical independence; event-level traces should be reviewed for the most influential definition changes.",
    ]
    (output_root / "recurrence_definition_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_recurrence_sensitivity(
    config_path: str | Path = "configs/rig_hazard_preprocessing.json",
    output_root: str | Path = "results/recurrence_analysis/definition_sensitivity",
    persistence_records: int = 3,
    overwrite: bool = False,
) -> Path:
    config, resolved_config_path, config_hash = load_config(config_path)
    destination = resolve_project_path(output_root)
    _prepare_output(destination, overwrite=overwrite)
    positives, source_scan = collect_positive_records(config)
    positives.to_csv(destination / "positive_record_cache.csv.gz", index=False, compression="gzip")
    source_scan.to_csv(destination / "positive_record_source_scan.csv", index=False)
    all_events, station_summary, resolutions, definitions = evaluate_sensitivity_grid(
        positives,
        config,
        persistence_records=persistence_records,
    )
    summary, year_summary, matches = summarize_sensitivity(all_events, definitions)
    all_events.to_csv(destination / "sensitivity_event_records.csv.gz", index=False, compression="gzip")
    station_summary.to_csv(destination / "sensitivity_station_summary.csv", index=False)
    resolutions.to_csv(destination / "instrument_resolution_by_station.csv", index=False)
    definitions.to_csv(destination / "sensitivity_definitions.csv", index=False)
    summary.to_csv(destination / "sensitivity_summary.csv", index=False)
    year_summary.to_csv(destination / "sensitivity_year_summary.csv", index=False)
    matches.to_csv(destination / "reference_onset_matching.csv", index=False)
    _write_report(destination, summary, resolutions)
    manifest = {
        "analysis": "recurrence_definition_sensitivity",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(resolved_config_path),
        "config_sha256": config_hash,
        "positive_record_count": int(positives.shape[0]),
        "definition_count": int(definitions.shape[0]),
        "persistence_records": int(persistence_records),
        "artifacts": sorted(path.name for path in destination.iterdir() if path.is_file()),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Recurrence sensitivity analysis complete: {destination}", flush=True)
    return destination

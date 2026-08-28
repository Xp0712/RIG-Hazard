from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, resolve_project_path


MISSING_MARKERS = {"", "--", "nan", "NaN", "NAN", "null", "NULL", "None", "none"}
CITY_PREFIXES = (
    "\u6b66\u5937\u5c71",
    "\u9f99\u5ca9",
    "\u6cc9\u5dde",
    "\u5b81\u5fb7",
    "\u5357\u5e73",
    "\u4e09\u660e",
)

RAW_COLUMN_MAP = {
    "\u89c2\u6d4b\u65f6\u95f4": "timestamp",
    "\u7ad9\u70b9\u540d\u79f0": "station_name",
    "\u7ad9\u70b9\u7f16\u53f7": "station_code",
    "\u7ad9\u70b9ID": "station_id",
    "\u8bbe\u5907\u7535\u538b": "device_voltage",
    "\u8986\u51b0\u539a\u5ea6": "ice_thickness",
    "\u7ed3\u51b0\u4f20\u611f\u5668\u9891\u7387": "ice_sensor_frequency",
    "\u6c14\u6e29": "air_temperature",
    "\u76f8\u5bf9\u6e7f\u5ea6": "relative_humidity",
    "\u6c14\u538b": "station_pressure",
    "\u96e8": "rain",
    "\u5c0f\u65f6\u964d\u96e8": "hourly_rain",
    "\u5341\u5206\u949f\u5e73\u5747\u98ce\u5411": "wind_direction",
    "\u5341\u5206\u949f\u5e73\u5747\u98ce\u901f": "wind_speed",
    "\u80fd\u89c1\u5ea6": "visibility",
    "\u4e00\u5206\u949f\u80fd\u89c1\u5ea6": "visibility_alias",
    "\u5341\u5206\u949f\u80fd\u89c1\u5ea6": "visibility_10min",
    "\u964d\u6c34\u5929\u6c14\u73b0\u8c61": "precipitation_phenomenon",
    "\u89c6\u7a0b\u969c\u788d": "visibility_obstacle",
}
RAW_COLUMN_BY_CANONICAL = {canonical: raw for raw, canonical in RAW_COLUMN_MAP.items()}
RAW_FOG_TOKEN = "\u96fe"
RAW_SNOW_TOKEN = "\u96ea"
RAW_FREEZING_RAIN_TOKEN = "\u51bb\u96e8"

NUMERIC_RAW_COLUMNS = (
    "device_voltage",
    "ice_thickness",
    "ice_sensor_frequency",
    "air_temperature",
    "relative_humidity",
    "station_pressure",
    "rain",
    "hourly_rain",
    "wind_direction",
    "wind_speed",
    "visibility",
    "visibility_10min",
)

SUM_COLUMNS = (
    "raw_rows",
    "observed_minutes",
    "ice_label_count",
    "ice_positive_minutes",
    "device_voltage_sum",
    "device_voltage_count",
    "ice_frequency_sum",
    "ice_frequency_count",
    "temperature_sum",
    "temperature_count",
    "rh_sum",
    "rh_count",
    "pressure_sum",
    "pressure_count",
    "rain_sum",
    "rain_count",
    "hourly_rain_sum",
    "hourly_rain_count",
    "wind_speed_sum",
    "wind_speed_count",
    "wind_direction_sin_sum",
    "wind_direction_cos_sum",
    "wind_direction_count",
    "visibility_sum",
    "visibility_count",
    "visibility_10min_sum",
    "visibility_10min_count",
    "fog_minutes",
    "precipitation_minutes",
    "snow_minutes",
    "freezing_rain_minutes",
)

MIN_COLUMNS = ("temperature_min", "rh_min", "visibility_min", "visibility_10min_min")
MAX_COLUMNS = (
    "ice_thickness_max",
    "ice_frequency_max",
    "temperature_max",
    "rh_max",
    "pressure_max",
    "rain_max",
    "hourly_rain_max",
    "wind_speed_max",
)

TIMELINE_AUDIT_AGGREGATES = {"raw_rows", "observed_minutes", "ice_label_count", "ice_positive_minutes"}
INTERNAL_AGGREGATION_COLUMNS = [column for column in SUM_COLUMNS if column not in TIMELINE_AUDIT_AGGREGATES]

CONTINUOUS_MODEL_FEATURES = [
    "device_voltage_mean",
    "ice_frequency_mean",
    "ice_frequency_max",
    "ice_frequency_change_30m",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "temperature_change_30m",
    "rh_mean",
    "rh_min",
    "rh_max",
    "rh_change_30m",
    "pressure_mean",
    "pressure_change_1h",
    "rain_mean",
    "rain_max",
    "hourly_rain_mean",
    "hourly_rain_max",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_direction_sin",
    "wind_direction_cos",
    "visibility_mean",
    "visibility_min",
    "visibility_change_30m",
    "visibility_10min_mean",
    "visibility_10min_min",
    "fog_fraction",
    "precipitation_fraction",
    "snow_fraction",
    "freezing_rain_fraction",
    "exposure_fraction_1h",
    "coverage_fraction",
    "history_bin_fraction",
    "history_observation_fraction",
    "time_since_risk_entry_hours",
    "time_since_last_icing_hours",
    "time_since_last_recurrent_event_hours",
    "current_event_order",
    "events_past_7d",
    "events_past_30d",
    "previous_event_duration_hours",
    "previous_event_max_thickness",
    "previous_event_severity",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

BINARY_MODEL_FEATURES = [
    "exposure_e1_cold_humid",
    "exposure_e2_fog_low_visibility",
    "exposure_e3_any",
    "exposure_recent_1h",
    "device_voltage_missing",
    "ice_frequency_missing",
    "temperature_missing",
    "rh_missing",
    "pressure_missing",
    "rain_missing",
    "hourly_rain_missing",
    "wind_speed_missing",
    "wind_direction_missing",
    "visibility_missing",
    "time_since_last_icing_missing",
    "previous_recurrent_event_missing",
]


@dataclass
class SourceInfo:
    path: str
    relative_path: str
    year: int
    station_name: str
    station_code: str
    station_id: str
    encoding: str
    size_bytes: int
    empty: bool


class RunningMoments:
    def __init__(self, columns: Iterable[str]) -> None:
        self.stats = {column: {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": None, "max": None} for column in columns}

    def update(self, frame: pd.DataFrame, mask: pd.Series) -> None:
        selected = frame.loc[mask]
        if selected.empty:
            return
        for column, stat in self.stats.items():
            if column not in selected.columns:
                continue
            values = pd.to_numeric(selected[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            stat["count"] += int(values.size)
            stat["sum"] += float(values.sum())
            stat["sum_sq"] += float(np.square(values).sum())
            current_min = float(values.min())
            current_max = float(values.max())
            stat["min"] = current_min if stat["min"] is None else min(float(stat["min"]), current_min)
            stat["max"] = current_max if stat["max"] is None else max(float(stat["max"]), current_max)

    def finalize(self) -> dict[str, dict[str, float | int | None]]:
        result: dict[str, dict[str, float | int | None]] = {}
        for column, stat in self.stats.items():
            count = int(stat["count"])
            if count == 0:
                result[column] = {"count": 0, "mean": None, "std": None, "min": None, "max": None}
                continue
            mean = float(stat["sum"]) / count
            variance = max(float(stat["sum_sq"]) / count - mean * mean, 0.0)
            result[column] = {
                "count": count,
                "mean": mean,
                "std": math.sqrt(variance),
                "min": stat["min"],
                "max": stat["max"],
            }
        return result


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def numeric(series: pd.Series) -> pd.Series:
    text = clean_text(series)
    return pd.to_numeric(text.mask(text.isin(MISSING_MARKERS)), errors="coerce")


def safe_mean(frame: pd.DataFrame, total: str, count: str) -> pd.Series:
    denominator = frame[count].replace(0, np.nan)
    return frame[total] / denominator


def infer_city(station_name: str) -> str:
    for prefix in CITY_PREFIXES:
        if station_name.startswith(prefix):
            return prefix
    return "Unknown"


def infer_year(path: Path) -> int | None:
    match = re.search(r"(20\d{2})", path.parent.name)
    return int(match.group(1)) if match else None


def infer_station_name_from_path(path: Path) -> str:
    return re.sub(r"20\d{6}_20\d{6}$", "", path.stem)


def detect_encoding(path: Path) -> str:
    sample = path.read_bytes()[:65536]
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "gb18030"


def scan_source(path: Path, data_root: Path) -> SourceInfo:
    encoding = detect_encoding(path)
    year = infer_year(path)
    if year is None:
        raise ValueError(f"Cannot infer year from {path}")
    preview = pd.read_csv(path, encoding=encoding, dtype=str, nrows=20, keep_default_na=False, low_memory=False)
    station_name = infer_station_name_from_path(path)
    station_code = ""
    station_id = ""
    if not preview.empty:
        if RAW_COLUMN_BY_CANONICAL["station_name"] in preview:
            values = clean_text(preview[RAW_COLUMN_BY_CANONICAL["station_name"]])
            valid = values[~values.isin(MISSING_MARKERS)]
            if not valid.empty:
                station_name = str(valid.iloc[0])
        if RAW_COLUMN_BY_CANONICAL["station_code"] in preview:
            values = clean_text(preview[RAW_COLUMN_BY_CANONICAL["station_code"]])
            valid = values[~values.isin(MISSING_MARKERS)]
            if not valid.empty:
                station_code = str(valid.iloc[0])
        if RAW_COLUMN_BY_CANONICAL["station_id"] in preview:
            values = clean_text(preview[RAW_COLUMN_BY_CANONICAL["station_id"]])
            valid = values[~values.isin(MISSING_MARKERS)]
            if not valid.empty:
                station_id = str(valid.iloc[0])
    return SourceInfo(
        path=str(path),
        relative_path=str(path.relative_to(data_root.parent)),
        year=year,
        station_name=station_name,
        station_code=station_code,
        station_id=station_id,
        encoding=encoding,
        size_bytes=path.stat().st_size,
        empty=preview.empty,
    )


def discover_sources(data_root: Path) -> list[SourceInfo]:
    paths = sorted(data_root.glob("**/*.csv"))
    return [scan_source(path, data_root) for path in paths]


def read_chunks(source: SourceInfo, chunksize: int) -> Iterable[pd.DataFrame]:
    usecols = set(RAW_COLUMN_MAP)
    yield from pd.read_csv(
        source.path,
        encoding=source.encoding,
        dtype=str,
        usecols=lambda column: column in usecols,
        chunksize=chunksize,
        keep_default_na=False,
        low_memory=False,
    )


def canonicalize_chunk(chunk: pd.DataFrame, expected_year: int, valid_ranges: dict[str, list[float | None]], anomaly_counts: dict[str, int]) -> pd.DataFrame:
    renamed = chunk.rename(columns=RAW_COLUMN_MAP)
    if "visibility" not in renamed and "visibility_alias" in renamed:
        renamed["visibility"] = renamed["visibility_alias"]
    elif "visibility" in renamed and "visibility_alias" in renamed:
        canonical = clean_text(renamed["visibility"])
        alias = clean_text(renamed["visibility_alias"])
        renamed["visibility"] = canonical.where(~canonical.isin(MISSING_MARKERS), alias)

    for column in RAW_COLUMN_MAP.values():
        if column not in renamed:
            renamed[column] = ""

    result = pd.DataFrame(index=renamed.index)
    result["timestamp"] = pd.to_datetime(clean_text(renamed["timestamp"]), format="mixed", errors="coerce")
    invalid_time = int(result["timestamp"].isna().sum())
    anomaly_counts["invalid_timestamp"] += invalid_time
    year_mask = result["timestamp"].dt.year.eq(expected_year)
    anomaly_counts["timestamp_outside_directory_year"] += int((result["timestamp"].notna() & ~year_mask).sum())
    result = result.loc[year_mask].copy()
    if result.empty:
        return result
    renamed = renamed.loc[result.index]

    for column in NUMERIC_RAW_COLUMNS:
        values = numeric(renamed[column])
        range_key = column
        if column == "visibility_10min":
            range_key = "visibility"
        bounds = valid_ranges.get(range_key)
        if bounds:
            low, high = bounds
            invalid = pd.Series(False, index=values.index)
            if low is not None:
                invalid |= values < float(low)
            if high is not None:
                invalid |= values > float(high)
            anomaly_counts[f"out_of_range_{column}"] += int(invalid.sum())
            values = values.mask(invalid)
        result[column] = values

    obstacle = clean_text(renamed["visibility_obstacle"])
    phenomenon = clean_text(renamed["precipitation_phenomenon"])
    result["fog_flag"] = obstacle.str.contains(RAW_FOG_TOKEN, regex=False, na=False).astype(np.int8)
    result["precipitation_flag"] = (~phenomenon.isin(MISSING_MARKERS)).astype(np.int8)
    result["snow_flag"] = phenomenon.str.contains(RAW_SNOW_TOKEN, regex=False, na=False).astype(np.int8)
    result["freezing_rain_flag"] = phenomenon.str.contains(RAW_FREEZING_RAIN_TOKEN, regex=False, na=False).astype(np.int8)
    return result


def aggregate_chunk(frame: pd.DataFrame, step_minutes: int, ice_threshold: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = pd.DataFrame(index=frame.index)
    work["bin_start"] = frame["timestamp"].dt.floor(f"{step_minutes}min")
    work["minute"] = frame["timestamp"].dt.floor("min")
    work["raw_rows"] = 1
    work["ice_label_valid"] = frame["ice_thickness"].notna().astype(np.int16)
    work["ice_positive"] = (frame["ice_thickness"] > ice_threshold).astype(np.int16)

    numeric_mappings = {
        "device_voltage": "device_voltage",
        "ice_frequency": "ice_sensor_frequency",
        "temperature": "air_temperature",
        "rh": "relative_humidity",
        "pressure": "station_pressure",
        "rain": "rain",
        "hourly_rain": "hourly_rain",
        "wind_speed": "wind_speed",
        "visibility": "visibility",
        "visibility_10min": "visibility_10min",
    }
    for output, source in numeric_mappings.items():
        work[output] = frame[source]

    direction_radians = np.deg2rad(frame["wind_direction"])
    work["wind_direction_sin"] = np.sin(direction_radians)
    work["wind_direction_cos"] = np.cos(direction_radians)
    work["wind_direction_valid"] = frame["wind_direction"].notna().astype(np.int16)
    work["ice_thickness"] = frame["ice_thickness"]
    work["fog_flag"] = frame["fog_flag"]
    work["precipitation_flag"] = frame["precipitation_flag"]
    work["snow_flag"] = frame["snow_flag"]
    work["freezing_rain_flag"] = frame["freezing_rain_flag"]

    grouped = work.groupby("bin_start", sort=True).agg(
        raw_rows=("raw_rows", "sum"),
        observed_minutes=("minute", "nunique"),
        ice_label_count=("ice_label_valid", "sum"),
        ice_positive_minutes=("ice_positive", "sum"),
        ice_thickness_max=("ice_thickness", "max"),
        device_voltage_sum=("device_voltage", "sum"),
        device_voltage_count=("device_voltage", "count"),
        ice_frequency_sum=("ice_frequency", "sum"),
        ice_frequency_count=("ice_frequency", "count"),
        ice_frequency_max=("ice_frequency", "max"),
        temperature_sum=("temperature", "sum"),
        temperature_count=("temperature", "count"),
        temperature_min=("temperature", "min"),
        temperature_max=("temperature", "max"),
        rh_sum=("rh", "sum"),
        rh_count=("rh", "count"),
        rh_min=("rh", "min"),
        rh_max=("rh", "max"),
        pressure_sum=("pressure", "sum"),
        pressure_count=("pressure", "count"),
        pressure_max=("pressure", "max"),
        rain_sum=("rain", "sum"),
        rain_count=("rain", "count"),
        rain_max=("rain", "max"),
        hourly_rain_sum=("hourly_rain", "sum"),
        hourly_rain_count=("hourly_rain", "count"),
        hourly_rain_max=("hourly_rain", "max"),
        wind_speed_sum=("wind_speed", "sum"),
        wind_speed_count=("wind_speed", "count"),
        wind_speed_max=("wind_speed", "max"),
        wind_direction_sin_sum=("wind_direction_sin", "sum"),
        wind_direction_cos_sum=("wind_direction_cos", "sum"),
        wind_direction_count=("wind_direction_valid", "sum"),
        visibility_sum=("visibility", "sum"),
        visibility_count=("visibility", "count"),
        visibility_min=("visibility", "min"),
        visibility_10min_sum=("visibility_10min", "sum"),
        visibility_10min_count=("visibility_10min", "count"),
        visibility_10min_min=("visibility_10min", "min"),
        fog_minutes=("fog_flag", "sum"),
        precipitation_minutes=("precipitation_flag", "sum"),
        snow_minutes=("snow_flag", "sum"),
        freezing_rain_minutes=("freezing_rain_flag", "sum"),
    )
    return grouped


def reduce_chunk_aggregates(parts: list[pd.DataFrame], year: int, step_minutes: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    calendar = pd.date_range(start, end, freq=f"{step_minutes}min", inclusive="left", name="bin_start")
    if not parts:
        reduced = pd.DataFrame(index=calendar)
    else:
        combined = pd.concat(parts).sort_index()
        aggregation = {column: "sum" for column in SUM_COLUMNS}
        aggregation.update({column: "min" for column in MIN_COLUMNS})
        aggregation.update({column: "max" for column in MAX_COLUMNS})
        reduced = combined.groupby(level=0).agg(aggregation).reindex(calendar)

    for column in SUM_COLUMNS:
        if column not in reduced:
            reduced[column] = 0
        reduced[column] = reduced[column].fillna(0)
    for column in (*MIN_COLUMNS, *MAX_COLUMNS):
        if column not in reduced:
            reduced[column] = np.nan

    reduced["observed_minutes"] = reduced["observed_minutes"].clip(upper=step_minutes)
    reduced["coverage_fraction"] = reduced["observed_minutes"] / step_minutes
    reduced["ice_label_coverage_fraction"] = reduced["ice_label_count"].clip(upper=step_minutes) / step_minutes
    reduced["device_voltage_mean"] = safe_mean(reduced, "device_voltage_sum", "device_voltage_count")
    reduced["ice_frequency_mean"] = safe_mean(reduced, "ice_frequency_sum", "ice_frequency_count")
    reduced["temperature_mean"] = safe_mean(reduced, "temperature_sum", "temperature_count")
    reduced["rh_mean"] = safe_mean(reduced, "rh_sum", "rh_count")
    reduced["pressure_mean"] = safe_mean(reduced, "pressure_sum", "pressure_count")
    reduced["rain_mean"] = safe_mean(reduced, "rain_sum", "rain_count")
    reduced["hourly_rain_mean"] = safe_mean(reduced, "hourly_rain_sum", "hourly_rain_count")
    reduced["wind_speed_mean"] = safe_mean(reduced, "wind_speed_sum", "wind_speed_count")
    reduced["wind_direction_sin"] = safe_mean(reduced, "wind_direction_sin_sum", "wind_direction_count")
    reduced["wind_direction_cos"] = safe_mean(reduced, "wind_direction_cos_sum", "wind_direction_count")
    reduced["visibility_mean"] = safe_mean(reduced, "visibility_sum", "visibility_count")
    reduced["visibility_10min_mean"] = safe_mean(reduced, "visibility_10min_sum", "visibility_10min_count")
    denominator = reduced["observed_minutes"].replace(0, np.nan)
    reduced["fog_fraction"] = reduced["fog_minutes"] / denominator
    reduced["precipitation_fraction"] = reduced["precipitation_minutes"] / denominator
    reduced["snow_fraction"] = reduced["snow_minutes"] / denominator
    reduced["freezing_rain_fraction"] = reduced["freezing_rain_minutes"] / denominator
    reduced["bin_start"] = reduced.index
    reduced["issue_time"] = reduced.index + pd.Timedelta(minutes=step_minutes)
    reduced["year"] = year
    return reduced.reset_index(drop=True)


def process_source(source: SourceInfo, config: dict[str, Any], anomaly_counts: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if source.empty:
        return reduce_chunk_aggregates([], source.year, int(config["time_step_minutes"])), pd.DataFrame(), 0
    parts: list[pd.DataFrame] = []
    positive_parts: list[pd.DataFrame] = []
    raw_rows = 0
    step = int(config["time_step_minutes"])
    threshold = float(config["event"]["ice_threshold"])
    for chunk in read_chunks(source, int(config["chunksize"])):
        raw_rows += int(chunk.shape[0])
        canonical = canonicalize_chunk(chunk, source.year, config["valid_ranges"], anomaly_counts)
        if canonical.empty:
            continue
        part = aggregate_chunk(canonical, step, threshold)
        if not part.empty:
            parts.append(part)
        positive = canonical.loc[
            canonical["ice_thickness"].gt(threshold),
            ["timestamp", "ice_thickness", "air_temperature", "relative_humidity", "visibility", "fog_flag"],
        ].copy()
        if not positive.empty:
            positive_parts.append(positive)
    aggregated = reduce_chunk_aggregates(parts, source.year, step)
    positives = pd.concat(positive_parts, ignore_index=True) if positive_parts else pd.DataFrame()
    return aggregated, positives, raw_rows


def extract_events(positives: pd.DataFrame, station_code: str, station_name: str, city: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    if positives.empty:
        return []
    positives = positives.dropna(subset=["timestamp", "ice_thickness"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if positives.empty:
        return []
    event_config = config["event"]
    max_separation = float(event_config["merge_gap_minutes"]) + 1.0
    minimum_minutes = int(event_config["minimum_positive_minutes"])
    cold_limit = float(event_config["cold_plausible_temperature_c"])
    cooldown = pd.Timedelta(minutes=int(event_config["cooldown_minutes"]))
    event_break = positives["timestamp"].diff().dt.total_seconds().div(60).gt(max_separation).fillna(True)
    positives["event_group"] = event_break.cumsum()
    events: list[dict[str, Any]] = []
    recurrent_index = 0
    previous_event_end: pd.Timestamp | None = None
    for sequence, (_, group) in enumerate(positives.groupby("event_group", sort=True), start=1):
        onset = pd.Timestamp(group["timestamp"].iloc[0])
        end = pd.Timestamp(group["timestamp"].iloc[-1])
        reference_issue_time = onset.floor(f"{int(config['time_step_minutes'])}min")
        min_temp = group["air_temperature"].min(skipna=True)
        cold_plausible = bool(pd.notna(min_temp) and float(min_temp) <= cold_limit)
        enough_positive_minutes = int(group.shape[0]) >= minimum_minutes
        cold_candidate = cold_plausible and enough_positive_minutes
        if not bool(event_config["use_only_cold_plausible_events_as_targets"]):
            cold_candidate = enough_positive_minutes
        cooldown_eligible = previous_event_end is None or reference_issue_time >= previous_event_end + cooldown
        valid_target = cold_candidate and cooldown_eligible
        if valid_target:
            recurrent_index += 1
        visibility = group["visibility"].min(skipna=True)
        max_rh = group["relative_humidity"].max(skipna=True)
        event_id = f"{station_code}-E{sequence:04d}"
        events.append(
            {
                "event_id": event_id,
                "station_code": station_code,
                "station_name": station_name,
                "city": city,
                "event_sequence_all": sequence,
                "recurrent_event_index": recurrent_index if valid_target else "",
                "onset_time": onset,
                "end_time": end,
                "onset_year": int(onset.year),
                "elapsed_minutes": int((end - onset).total_seconds() // 60) + 1,
                "ice_positive_minutes": int(group.shape[0]),
                "max_thickness": float(group["ice_thickness"].max()),
                "min_temperature": None if pd.isna(min_temp) else float(min_temp),
                "max_relative_humidity": None if pd.isna(max_rh) else float(max_rh),
                "min_visibility": None if pd.isna(visibility) else float(visibility),
                "fog_supported": int(group["fog_flag"].max()) if "fog_flag" in group else 0,
                "cold_plausible": int(cold_plausible),
                "minimum_duration_pass": int(enough_positive_minutes),
                "cold_candidate_event": int(cold_candidate),
                "cooldown_eligible_onset": int(cooldown_eligible),
                "valid_target_event": int(valid_target),
                "target_exclusion_reason": "" if valid_target else (
                    "warm_or_missing_temperature"
                    if not cold_plausible
                    else "too_few_positive_minutes"
                    if not enough_positive_minutes
                    else "prediction_time_within_cooldown"
                ),
                "reference_issue_time": reference_issue_time,
                "eligible_hazard_label": 0,
                "model_eligibility_exclusion_reason": "not_a_target_event" if not valid_target else "",
            }
        )
        previous_event_end = end
    return events


def forward_window_sum(values: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    result = np.full(n, np.nan, dtype=np.float64)
    complete = np.zeros(n, dtype=bool)
    if steps <= 0 or n == 0:
        return result, complete
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    last_start = n - steps - 1
    if last_start >= 0:
        indexes = np.arange(last_start + 1)
        result[indexes] = cumulative[indexes + steps + 1] - cumulative[indexes + 1]
        complete[indexes] = True
    return result, complete


def interval_event_counts(issue_times: pd.Series, onset_times: list[pd.Timestamp], horizon_minutes: int) -> np.ndarray:
    issues = issue_times.to_numpy(dtype="datetime64[ns]")
    if not onset_times:
        return np.zeros(issues.size, dtype=np.int16)
    onsets = np.sort(np.asarray(onset_times, dtype="datetime64[ns]"))
    ends = issues + np.timedelta64(horizon_minutes, "m")
    left = np.searchsorted(onsets, issues, side="left")
    right = np.searchsorted(onsets, ends, side="left")
    return (right - left).astype(np.int16)


def causal_recurrence_features(
    issue_times: pd.Series,
    valid_events: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Build recurrence covariates using only valid events completed by each issue time."""

    issues = pd.to_datetime(issue_times).to_numpy(dtype="datetime64[ns]")
    size = issues.size
    empty = {
        "time_since_last_recurrent_event_hours": np.full(size, np.nan, dtype=np.float64),
        "current_event_order": np.ones(size, dtype=np.float64),
        "events_past_7d": np.zeros(size, dtype=np.float64),
        "events_past_30d": np.zeros(size, dtype=np.float64),
        "previous_event_duration_hours": np.full(size, np.nan, dtype=np.float64),
        "previous_event_max_thickness": np.full(size, np.nan, dtype=np.float64),
        "previous_event_severity": np.full(size, np.nan, dtype=np.float64),
        "previous_recurrent_event_missing": np.ones(size, dtype=np.int8),
    }
    if not valid_events:
        return empty
    ordered = sorted(valid_events, key=lambda event: pd.Timestamp(event["end_time"]))
    ends = np.asarray([pd.Timestamp(event["end_time"]) for event in ordered], dtype="datetime64[ns]")
    onsets = np.sort(
        np.asarray([pd.Timestamp(event["onset_time"]) for event in ordered], dtype="datetime64[ns]")
    )
    duration_hours = np.asarray(
        [float(event["elapsed_minutes"]) / 60.0 for event in ordered], dtype=np.float64
    )
    maximum_thickness = np.asarray(
        [float(event["max_thickness"]) for event in ordered], dtype=np.float64
    )
    previous_count = np.searchsorted(ends, issues, side="right")
    previous_index = previous_count - 1
    has_previous = previous_index >= 0
    safe_index = np.clip(previous_index, 0, len(ordered) - 1)
    gap_hours = np.full(size, np.nan, dtype=np.float64)
    gap_hours[has_previous] = (
        issues[has_previous] - ends[safe_index[has_previous]]
    ) / np.timedelta64(1, "h")
    previous_duration = np.full(size, np.nan, dtype=np.float64)
    previous_duration[has_previous] = duration_hours[safe_index[has_previous]]
    previous_thickness = np.full(size, np.nan, dtype=np.float64)
    previous_thickness[has_previous] = maximum_thickness[safe_index[has_previous]]
    severity = np.full(size, np.nan, dtype=np.float64)
    severity[has_previous] = np.log1p(previous_duration[has_previous]) * np.log1p(
        previous_thickness[has_previous]
    )
    prior_onsets = np.searchsorted(onsets, issues, side="left")
    seven_day_start = np.searchsorted(onsets, issues - np.timedelta64(7, "D"), side="left")
    thirty_day_start = np.searchsorted(onsets, issues - np.timedelta64(30, "D"), side="left")
    return {
        "time_since_last_recurrent_event_hours": gap_hours,
        "current_event_order": previous_count.astype(np.float64) + 1.0,
        "events_past_7d": (prior_onsets - seven_day_start).astype(np.float64),
        "events_past_30d": (prior_onsets - thirty_day_start).astype(np.float64),
        "previous_event_duration_hours": previous_duration,
        "previous_event_max_thickness": previous_thickness,
        "previous_event_severity": severity,
        "previous_recurrent_event_missing": (~has_previous).astype(np.int8),
    }


def split_label(years: pd.Series, split: dict[str, list[int]]) -> pd.Series:
    mapping: dict[int, str] = {}
    for year in split.get("train_years", []):
        mapping[int(year)] = "train"
    for year in split.get("validation_years", []):
        mapping[int(year)] = "validation"
    for year in split.get("test_years", []):
        mapping[int(year)] = "test"
    return years.map(mapping).fillna("excluded")


def add_temporal_features_and_labels(
    timeline: pd.DataFrame,
    events: list[dict[str, Any]],
    station_row: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = timeline.sort_values("issue_time").reset_index(drop=True).copy()
    step = int(config["time_step_minutes"])
    risk = config["risk_set"]
    exposure = config["exposure"]
    history_steps = int(round(float(risk["history_hours"]) * 60 / step))
    lookback_steps = int(round(float(exposure["lookback_hours"]) * 60 / step))

    issue = pd.to_datetime(frame["issue_time"])
    frame["temperature_change_30m"] = frame["temperature_mean"] - frame["temperature_mean"].shift(max(1, 30 // step))
    frame["rh_change_30m"] = frame["rh_mean"] - frame["rh_mean"].shift(max(1, 30 // step))
    frame["visibility_change_30m"] = frame["visibility_mean"] - frame["visibility_mean"].shift(max(1, 30 // step))
    frame["ice_frequency_change_30m"] = frame["ice_frequency_mean"] - frame["ice_frequency_mean"].shift(max(1, 30 // step))
    frame["pressure_change_1h"] = frame["pressure_mean"] - frame["pressure_mean"].shift(max(1, 60 // step))

    frame["exposure_e1_cold_humid"] = (
        frame["temperature_min"].le(float(exposure["temperature_c"]))
        & frame["rh_max"].ge(float(exposure["relative_humidity_percent"]))
    ).astype(np.int8)
    frame["exposure_e2_fog_low_visibility"] = (
        frame["visibility_min"].lt(float(exposure["visibility_m"])) | frame["fog_fraction"].gt(0)
    ).astype(np.int8)
    frame["exposure_e3_any"] = (
        frame["exposure_e1_cold_humid"].eq(1) | frame["exposure_e2_fog_low_visibility"].eq(1)
    ).astype(np.int8)
    frame["exposure_recent_1h"] = frame["exposure_e3_any"].rolling(lookback_steps, min_periods=1).max().astype(np.int8)
    frame["exposure_fraction_1h"] = frame["exposure_e3_any"].rolling(lookback_steps, min_periods=1).mean()

    frame["bin_observed"] = frame["observed_minutes"].ge(int(risk["minimum_observed_minutes_per_bin"])).astype(np.int8)
    frame["history_bin_fraction"] = frame["bin_observed"].rolling(history_steps, min_periods=history_steps).mean()
    frame["history_observation_fraction"] = frame["coverage_fraction"].rolling(history_steps, min_periods=history_steps).mean()
    frame["history_valid"] = (
        frame["history_bin_fraction"].ge(float(risk["minimum_history_bin_fraction"]))
        & frame["history_observation_fraction"].ge(float(risk["minimum_history_observation_fraction"]))
    ).astype(np.int8)

    valid_events = [event for event in events if int(event["valid_target_event"]) == 1]
    valid_onsets = [pd.Timestamp(event["onset_time"]) for event in valid_events]
    all_events = events
    physical_at_risk = np.ones(frame.shape[0], dtype=bool)
    issue_array = issue.to_numpy(dtype="datetime64[ns]")
    time_since_last = np.full(frame.shape[0], np.nan, dtype=np.float64)
    last_end: pd.Timestamp | None = None
    cooldown = pd.Timedelta(minutes=int(config["event"]["cooldown_minutes"]))
    for event in all_events:
        onset = pd.Timestamp(event["onset_time"])
        end = pd.Timestamp(event["end_time"])
        in_event = (issue_array > np.datetime64(onset)) & (issue_array <= np.datetime64(end))
        in_cooldown = (issue_array > np.datetime64(end)) & (issue_array < np.datetime64(end + cooldown))
        physical_at_risk &= ~(in_event | in_cooldown)
        if last_end is None or end > last_end:
            after = issue_array >= np.datetime64(end)
            time_since_last[after] = (issue_array[after] - np.datetime64(end)) / np.timedelta64(1, "m")
            last_end = end
    physical_at_risk &= frame["ice_positive_minutes"].eq(0).to_numpy()
    frame["physical_at_risk"] = physical_at_risk.astype(np.int8)
    frame["time_since_last_icing_hours"] = time_since_last / 60.0

    for name, values in causal_recurrence_features(issue, valid_events).items():
        frame[name] = values

    hazard_counts = interval_event_counts(issue, valid_onsets, step)
    frame["hazard_label"] = (hazard_counts > 0).astype(np.int8)
    next_label_fraction = frame["ice_label_coverage_fraction"].shift(-1)
    frame["future_step_label_valid"] = next_label_fraction.ge(float(risk["minimum_future_label_bin_fraction"])).astype(np.int8)
    frame["risk_set"] = (
        frame["physical_at_risk"].eq(1)
        & frame["history_valid"].eq(1)
        & frame["future_step_label_valid"].eq(1)
    ).astype(np.int8)
    frame.loc[frame["risk_set"].eq(0), "hazard_label"] = 0

    for horizon in [int(value) for value in risk["forecast_horizons_hours"]]:
        steps = int(round(horizon * 60 / step))
        event_counts = interval_event_counts(issue, valid_onsets, horizon * 60)
        frame[f"onset_within_{horizon}h"] = (event_counts > 0).astype(np.int8)
        future_sum, complete_length = forward_window_sum(frame["ice_label_coverage_fraction"].to_numpy(), steps)
        future_fraction = future_sum / steps
        valid_future = complete_length & (future_fraction >= float(risk["minimum_future_label_bin_fraction"]))
        frame[f"future_label_fraction_{horizon}h"] = future_fraction
        frame[f"future_label_valid_{horizon}h"] = valid_future.astype(np.int8)
        frame[f"hard_negative_{horizon}h"] = (
            frame["risk_set"].eq(1)
            & frame["exposure_recent_1h"].eq(1)
            & frame[f"future_label_valid_{horizon}h"].eq(1)
            & frame[f"onset_within_{horizon}h"].eq(0)
        ).astype(np.int8)

    eligible = frame["risk_set"].eq(1)
    new_spell = eligible & (~eligible.shift(fill_value=False))
    frame["risk_spell_index"] = new_spell.cumsum().where(eligible, 0).astype(np.int32)
    frame["risk_spell_step"] = frame.groupby("risk_spell_index").cumcount().add(1).where(eligible, 0).astype(np.int32)
    frame["time_since_risk_entry_hours"] = frame["risk_spell_step"] * step / 60.0
    valid_onset_array = np.sort(np.asarray(valid_onsets, dtype="datetime64[ns]")) if valid_onsets else np.array([], dtype="datetime64[ns]")
    frame["next_recurrent_event_index"] = np.searchsorted(valid_onset_array, issue_array, side="left") + 1

    frame["hour_sin"] = np.sin(2 * np.pi * (issue.dt.hour + issue.dt.minute / 60.0) / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * (issue.dt.hour + issue.dt.minute / 60.0) / 24.0)
    days_in_year = np.where(issue.dt.is_leap_year, 366.0, 365.0)
    frame["day_of_year_sin"] = np.sin(2 * np.pi * (issue.dt.dayofyear - 1) / days_in_year)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * (issue.dt.dayofyear - 1) / days_in_year)

    missing_sources = {
        "device_voltage_missing": "device_voltage_mean",
        "ice_frequency_missing": "ice_frequency_mean",
        "temperature_missing": "temperature_mean",
        "rh_missing": "rh_mean",
        "pressure_missing": "pressure_mean",
        "rain_missing": "rain_mean",
        "hourly_rain_missing": "hourly_rain_mean",
        "wind_speed_missing": "wind_speed_mean",
        "wind_direction_missing": "wind_direction_sin",
        "visibility_missing": "visibility_mean",
    }
    for output, source in missing_sources.items():
        frame[output] = frame[source].isna().astype(np.int8)
    frame["time_since_last_icing_missing"] = frame["time_since_last_icing_hours"].isna().astype(np.int8)

    frame["station_code"] = station_row["station_code"]
    frame["station_name"] = station_row["station_name"]
    frame["city"] = station_row["city"]
    frame["station_index"] = int(station_row["station_index"])
    frame["city_index"] = int(station_row["city_index"])
    frame["seen_in_development"] = int(station_row["seen_in_development"])
    frame["issue_year"] = issue.dt.year.astype(np.int16)
    frame["split_development"] = split_label(frame["issue_year"], config["splits"]["development"])
    frame["split_final"] = split_label(frame["issue_year"], config["splits"]["final"])

    issue_lookup = {pd.Timestamp(value): index for index, value in enumerate(pd.to_datetime(frame["issue_time"]))}
    for event in valid_events:
        reference = pd.Timestamp(event["reference_issue_time"])
        row_index = issue_lookup.get(reference)
        if row_index is None:
            event["model_eligibility_exclusion_reason"] = "reference_time_outside_timeline"
            continue
        reasons: list[str] = []
        if int(frame.at[row_index, "physical_at_risk"]) != 1:
            reasons.append("not_physically_at_risk")
        if int(frame.at[row_index, "history_valid"]) != 1:
            reasons.append("insufficient_history_coverage")
        if int(frame.at[row_index, "future_step_label_valid"]) != 1:
            reasons.append("incomplete_next_step_label")
        eligible = int(frame.at[row_index, "risk_set"]) == 1 and int(frame.at[row_index, "hazard_label"]) == 1
        event["eligible_hazard_label"] = int(eligible)
        event["model_eligibility_exclusion_reason"] = "" if eligible else ";".join(reasons) or "risk_set_exclusion"
    return frame


def load_station_metadata(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path, dtype={"station_code": str})
    metadata: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        code = str(row["station_code"]).strip()
        metadata[code] = row
    return metadata


def build_station_catalog(sources: list[SourceInfo], metadata: dict[str, dict[str, Any]], development_years: set[int]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SourceInfo]] = defaultdict(list)
    for source in sources:
        if source.station_code and not source.empty:
            grouped[source.station_code].append(source)
    city_names = sorted({infer_city(items[0].station_name) for items in grouped.values()})
    city_index = {name: index for index, name in enumerate(city_names)}
    catalog: list[dict[str, Any]] = []
    for station_index, code in enumerate(sorted(grouped)):
        items = sorted(grouped[code], key=lambda item: item.year)
        names = [item.station_name for item in items if item.station_name]
        station_name = max(set(names), key=names.count)
        station_ids = [item.station_id for item in items if item.station_id]
        station_id = max(set(station_ids), key=station_ids.count) if station_ids else ""
        city = infer_city(station_name)
        meta = metadata.get(code, {})
        years = sorted({item.year for item in items})
        catalog.append(
            {
                "station_index": station_index,
                "station_code": code,
                "station_name": station_name,
                "metadata_station_name": meta.get("metadata_station_name", ""),
                "station_id": station_id,
                "city": city,
                "city_index": city_index[city],
                "longitude": meta.get("longitude", ""),
                "latitude": meta.get("latitude", ""),
                "elevation_m": meta.get("elevation_m", ""),
                "observed_years": ",".join(str(year) for year in years),
                "seen_in_development": int(bool(set(years) & development_years)),
                "metadata_name_differs": int(bool(meta.get("metadata_station_name")) and str(meta.get("metadata_station_name")) != station_name),
            }
        )
    return catalog


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    lon1r, lat1r, lon2r, lat2r = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def build_spatial_candidates(catalog: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    valid = [row for row in catalog if pd.notna(row["longitude"]) and pd.notna(row["latitude"]) and row["longitude"] != "" and row["latitude"] != ""]
    edges: list[dict[str, Any]] = []
    for target in valid:
        candidates: list[dict[str, Any]] = []
        for source in valid:
            if source["station_code"] == target["station_code"]:
                continue
            distance = haversine_km(float(source["longitude"]), float(source["latitude"]), float(target["longitude"]), float(target["latitude"]))
            source_elevation = pd.to_numeric(pd.Series([source["elevation_m"]]), errors="coerce").iloc[0]
            target_elevation = pd.to_numeric(pd.Series([target["elevation_m"]]), errors="coerce").iloc[0]
            elevation_difference = None if pd.isna(source_elevation) or pd.isna(target_elevation) else float(source_elevation - target_elevation)
            candidates.append(
                {
                    "source_station_code": source["station_code"],
                    "source_station_name": source["station_name"],
                    "target_station_code": target["station_code"],
                    "target_station_name": target["station_name"],
                    "distance_km": round(distance, 4),
                    "source_minus_target_elevation_m": elevation_difference,
                    "same_city": int(source["city"] == target["city"]),
                }
            )
        candidates.sort(key=lambda row: row["distance_km"])
        for rank, row in enumerate(candidates[:top_k], start=1):
            row["distance_rank"] = rank
            edges.append(row)
    return edges


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_scalar(value) for key, value in row.items()})


def format_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_timeline(frame: pd.DataFrame, path: Path, output_format: str, compression_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv.gz":
        frame.to_csv(path, index=False, encoding="utf-8-sig", compression={"method": "gzip", "compresslevel": compression_level})
    elif output_format == "csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    elif output_format == "parquet":
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            raise RuntimeError("Parquet output requires pyarrow or fastparquet. Use output_format='csv.gz' or install pyarrow.") from exc
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")


def timeline_suffix(output_format: str) -> str:
    return {"csv.gz": ".csv.gz", "csv": ".csv", "parquet": ".parquet"}[output_format]


def prepare_output_root(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    project = PROJECT_ROOT.resolve()
    if resolved == project or project not in resolved.parents:
        raise ValueError(f"Output root must be a child of the project directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {resolved}. Re-run with --overwrite to replace it.")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def config_feature_contract(config: dict[str, Any]) -> dict[str, Any]:
    horizons = [int(value) for value in config["risk_set"]["forecast_horizons_hours"]]
    return {
        "identifiers": ["station_code", "station_name", "city", "station_index", "city_index", "year", "issue_year", "bin_start", "issue_time", "risk_spell_index", "risk_spell_step"],
        "continuous_dynamic_features": CONTINUOUS_MODEL_FEATURES,
        "binary_dynamic_features": BINARY_MODEL_FEATURES,
        "static_features_join_from_station_catalog": ["longitude", "latitude", "elevation_m", "seen_in_development"],
        "main_label": "hazard_label",
        "risk_filter": "risk_set == 1",
        "multi_horizon_labels": [f"onset_within_{horizon}h" for horizon in horizons],
        "hard_negative_flags": [f"hard_negative_{horizon}h" for horizon in horizons],
        "split_columns": ["split_development", "split_final"],
        "forbidden_as_model_inputs": [
            "ice_thickness_max",
            "ice_positive_minutes",
            "ice_label_count",
            "ice_label_coverage_fraction",
            "physical_at_risk",
            "history_valid",
            "future_step_label_valid",
            "risk_set",
            "hazard_label",
            *[f"onset_within_{horizon}h" for horizon in horizons],
            *[f"future_label_fraction_{horizon}h" for horizon in horizons],
            *[f"future_label_valid_{horizon}h" for horizon in horizons],
            *[f"hard_negative_{horizon}h" for horizon in horizons],
        ],
        "time_semantics": "Features summarize [issue_time - time_step, issue_time); hazard_label indicates a valid onset in [issue_time, issue_time + time_step).",
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 100) -> str:
    if not rows:
        return "None."
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(format_scalar(row.get(column, ""))) for column in columns) + " |")
    return "\n".join(lines)


def build_quality_report(
    config: dict[str, Any],
    sources: list[SourceInfo],
    catalog: list[dict[str, Any]],
    events: list[dict[str, Any]],
    station_summaries: list[dict[str, Any]],
    split_summaries: list[dict[str, Any]],
    anomaly_counts: dict[str, int],
    output_root: Path,
) -> str:
    cold_events = [event for event in events if int(event["cold_candidate_event"]) == 1]
    valid_events = [event for event in events if int(event["valid_target_event"]) == 1]
    captured_events = [event for event in valid_events if int(event["eligible_hazard_label"]) == 1]
    source_rows = sum(int(row["raw_rows"]) for row in station_summaries)
    timeline_rows = sum(int(row["timeline_rows"]) for row in station_summaries)
    risk_rows = sum(int(row["risk_rows"]) for row in station_summaries)
    positives = sum(int(row["hazard_positive_rows"]) for row in station_summaries)
    hard_columns = [f"hard_negative_{int(value)}h" for value in config["risk_set"]["forecast_horizons_hours"]]
    hard_totals = {column: sum(int(row.get(column, 0)) for row in station_summaries) for column in hard_columns}
    coordinate_count = sum(1 for row in catalog if row["longitude"] != "" and pd.notna(row["longitude"]))
    summary_columns = ["station_code", "station_name", "years", "raw_rows", "cold_candidate_events", "valid_events", "risk_rows", "hazard_positive_rows", *hard_columns]
    hard_lines = "\n".join(f"- `{column}`: {value:,} rows." for column, value in hard_totals.items())
    anomaly_rows = [{"item": key, "count": value} for key, value in sorted(anomaly_counts.items()) if value]
    return f"""# RIG-Hazard Preprocessing Quality Report

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contract

- Raw minute records are aggregated to `{int(config['time_step_minutes'])}`-minute intervals.
- Features in each row use observations only from `[issue_time-Δt, issue_time)`. The primary label indicates a valid icing onset in `[issue_time, issue_time+Δt)`, preventing post-onset observations from leaking into the model.
- Every positive-icing episode is excluded from the physical risk set. Primary labels use only events whose minimum temperature is no greater than `{float(config['event']['cold_plausible_temperature_c'])}` °C.
- A `{int(config['event']['cooldown_minutes'])}`-minute cooldown is excluded after each event.
- Ice thickness is used only to construct events and labels and is explicitly prohibited as a model input.

## Overall results

- CSV files scanned: {len(sources):,}, including {sum(int(source.empty) for source in sources):,} empty files.
- Valid stations: {len(catalog):,}; stations with coordinates: {coordinate_count:,}.
- Raw records scanned: {source_rows:,}.
- Full 10-minute timeline: {timeline_rows:,} rows.
- Positive-icing episodes: {len(events):,}; cold-condition candidates: {len(cold_events):,}; modelled recurrent events outside cooldown: {len(valid_events):,}.
- Valid onset labels captured in the leakage-free risk set: {len(captured_events):,}; event retention: {(100 * len(captured_events) / len(valid_events) if valid_events else 0):.2f}%.
- Risk-set samples: {risk_rows:,}; one-step positive hazard samples: {positives:,}.
{hard_lines}

## Station details

{markdown_table(station_summaries, summary_columns, 100)}

## Data splits

{markdown_table(split_summaries, ['scheme', 'split', 'timeline_rows', 'risk_rows', 'hazard_positive_rows', *hard_columns], 20)}

## Cleaning log

{markdown_table(anomaly_rows, ['item', 'count'], 100)}

## Output contract

- `timelines/`: full station-year 10-minute timelines; training filters to `risk_set == 1`.
- `events_recurrent.csv`: all events and valid-target indicators.
- `station_catalog.csv`: station, city, coordinates, elevation, and indices.
- `spatial_candidate_edges.csv`: candidates for later lag-graph screening, not the final graph.
- `feature_contract.json`: model inputs, labels, splits, and prohibited leakage fields.
- `normalization_development.json`: computed only from the development split's training segment.
- `normalization_final.json`: computed only from the final split's training segment.
- `manifest.json`: configuration hash, source-file inventory, and artifact inventory.

Output directory: `{output_root}`
"""


def preprocess(
    config: dict[str, Any],
    config_path: Path,
    config_hash: str,
    output_root_override: str | None = None,
    max_stations: int | None = None,
    selected_stations: set[str] | None = None,
    selected_years: set[int] | None = None,
    overwrite: bool = False,
) -> Path:
    data_root = resolve_project_path(config["data_root"])
    metadata_path = resolve_project_path(config["station_metadata"])
    output_root = resolve_project_path(output_root_override or config["output_root"])
    prepare_output_root(output_root, overwrite)

    print(f"Scanning source files under {data_root}", flush=True)
    sources = discover_sources(data_root)
    if selected_years:
        sources = [source for source in sources if source.year in selected_years]
    if selected_stations:
        sources = [source for source in sources if source.station_code in selected_stations or source.station_name in selected_stations]

    development_years = set(int(year) for year in config["splits"]["development"]["train_years"] + config["splits"]["development"]["validation_years"])
    metadata = load_station_metadata(metadata_path)
    catalog = build_station_catalog(sources, metadata, development_years)
    if max_stations is not None:
        keep_codes = {row["station_code"] for row in catalog[:max_stations]}
        catalog = [row for row in catalog if row["station_code"] in keep_codes]
        sources = [source for source in sources if source.station_code in keep_codes or source.empty]
    catalog_by_code = {row["station_code"]: row for row in catalog}

    sources_by_station: dict[str, list[SourceInfo]] = defaultdict(list)
    for source in sources:
        if source.station_code in catalog_by_code and not source.empty:
            sources_by_station[source.station_code].append(source)

    output_format = str(config["output_format"])
    suffix = timeline_suffix(output_format)
    compression_level = int(config.get("compression_level", 1))
    anomaly_counts: dict[str, int] = defaultdict(int)
    all_events: list[dict[str, Any]] = []
    station_summaries: list[dict[str, Any]] = []
    station_year_summaries: list[dict[str, Any]] = []
    split_totals: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dev_moments = RunningMoments(CONTINUOUS_MODEL_FEATURES)
    final_moments = RunningMoments(CONTINUOUS_MODEL_FEATURES)
    source_runtime: dict[str, dict[str, Any]] = {}

    for station_number, station_row in enumerate(catalog, start=1):
        code = station_row["station_code"]
        station_sources = sorted(sources_by_station.get(code, []), key=lambda source: source.year)
        if not station_sources:
            continue
        print(f"[{station_number}/{len(catalog)}] {code} {station_row['station_name']}", flush=True)
        annual_frames: list[pd.DataFrame] = []
        positive_frames: list[pd.DataFrame] = []
        raw_row_count = 0
        years: list[int] = []
        for source in station_sources:
            print(f"  - {source.year}: {source.relative_path}", flush=True)
            annual, positives, raw_rows = process_source(source, config, anomaly_counts)
            annual_frames.append(annual)
            if not positives.empty:
                positive_frames.append(positives)
            raw_row_count += raw_rows
            years.append(source.year)
            source_runtime[source.relative_path] = {"raw_rows": raw_rows, "timeline_rows": int(annual.shape[0])}

        positives = pd.concat(positive_frames, ignore_index=True) if positive_frames else pd.DataFrame()
        events = extract_events(positives, code, station_row["station_name"], station_row["city"], config)
        timeline = pd.concat(annual_frames, ignore_index=True)
        timeline = add_temporal_features_and_labels(timeline, events, station_row, config)
        timeline = timeline.drop(columns=[column for column in INTERNAL_AGGREGATION_COLUMNS if column in timeline.columns])

        dev_moments.update(timeline, timeline["risk_set"].eq(1) & timeline["split_development"].eq("train"))
        final_moments.update(timeline, timeline["risk_set"].eq(1) & timeline["split_final"].eq("train"))

        for year, year_frame in timeline.groupby("year", sort=True):
            timeline_path = output_root / "timelines" / str(int(year)) / f"{code}_{station_row['station_name']}{suffix}"
            write_timeline(year_frame, timeline_path, output_format, compression_level)

        hard_counts = {
            f"hard_negative_{int(horizon)}h": int(timeline[f"hard_negative_{int(horizon)}h"].sum())
            for horizon in config["risk_set"]["forecast_horizons_hours"]
        }
        cold_candidate_count = sum(int(event["cold_candidate_event"]) for event in events)
        valid_event_count = sum(int(event["valid_target_event"]) for event in events)
        station_summaries.append(
            {
                "station_code": code,
                "station_name": station_row["station_name"],
                "years": ",".join(str(year) for year in sorted(years)),
                "raw_rows": raw_row_count,
                "timeline_rows": int(timeline.shape[0]),
                "all_events": len(events),
                "cold_candidate_events": cold_candidate_count,
                "valid_events": valid_event_count,
                "eligible_events": sum(int(event["eligible_hazard_label"]) for event in events),
                "risk_rows": int(timeline["risk_set"].sum()),
                "hazard_positive_rows": int(timeline.loc[timeline["risk_set"].eq(1), "hazard_label"].sum()),
                **hard_counts,
            }
        )
        for year, year_frame in timeline.groupby("year", sort=True):
            year_events = [event for event in events if int(event["onset_year"]) == int(year)]
            station_year_summaries.append(
                {
                    "station_code": code,
                    "station_name": station_row["station_name"],
                    "year": int(year),
                    "timeline_rows": int(year_frame.shape[0]),
                    "cold_candidate_events": sum(int(event["cold_candidate_event"]) for event in year_events),
                    "valid_events": sum(int(event["valid_target_event"]) for event in year_events),
                    "eligible_events": sum(int(event["eligible_hazard_label"]) for event in year_events),
                    "risk_rows": int(year_frame["risk_set"].sum()),
                    "hazard_positive_rows": int(year_frame.loc[year_frame["risk_set"].eq(1), "hazard_label"].sum()),
                    **{
                        f"hard_negative_{int(horizon)}h": int(year_frame[f"hard_negative_{int(horizon)}h"].sum())
                        for horizon in config["risk_set"]["forecast_horizons_hours"]
                    },
                }
            )
        for scheme, split_column in (("development", "split_development"), ("final", "split_final")):
            for split_name, split_frame in timeline.groupby(split_column, sort=True):
                totals = split_totals[(scheme, str(split_name))]
                totals["timeline_rows"] += int(split_frame.shape[0])
                totals["risk_rows"] += int(split_frame["risk_set"].sum())
                totals["hazard_positive_rows"] += int(split_frame.loc[split_frame["risk_set"].eq(1), "hazard_label"].sum())
                for horizon in config["risk_set"]["forecast_horizons_hours"]:
                    column = f"hard_negative_{int(horizon)}h"
                    totals[column] += int(split_frame[column].sum())
        all_events.extend(events)

    station_summaries.sort(key=lambda row: row["valid_events"], reverse=True)
    station_year_summaries.sort(key=lambda row: (row["year"], row["station_code"]))
    split_order = {"train": 0, "validation": 1, "test": 2, "excluded": 3}
    split_summaries = [
        {"scheme": scheme, "split": split_name, **dict(values)}
        for (scheme, split_name), values in sorted(split_totals.items(), key=lambda item: (item[0][0], split_order.get(item[0][1], 99)))
    ]
    write_csv_rows(output_root / "events_recurrent.csv", all_events)
    write_csv_rows(output_root / "station_catalog.csv", catalog)
    write_csv_rows(output_root / "source_files.csv", [asdict(source) | source_runtime.get(source.relative_path, {}) for source in sources])
    write_csv_rows(output_root / "station_preprocessing_summary.csv", station_summaries)
    write_csv_rows(output_root / "station_year_preprocessing_summary.csv", station_year_summaries)
    write_csv_rows(output_root / "split_summary.csv", split_summaries)
    candidates = build_spatial_candidates(catalog, int(config.get("graph", {}).get("spatial_candidate_neighbors", 8)))
    write_csv_rows(output_root / "spatial_candidate_edges.csv", candidates)
    write_json(output_root / "feature_contract.json", config_feature_contract(config))
    write_json(output_root / "normalization_development.json", dev_moments.finalize())
    write_json(output_root / "normalization_final.json", final_moments.finalize())
    write_json(output_root / "resolved_config.json", config)
    report = build_quality_report(config, sources, catalog, all_events, station_summaries, split_summaries, anomaly_counts, output_root)
    (output_root / "quality_report.md").write_text(report, encoding="utf-8")

    source_manifest = [
        {
            "relative_path": source.relative_path,
            "size_bytes": source.size_bytes,
            "year": source.year,
            "station_code": source.station_code,
            "empty": source.empty,
        }
        for source in sources
    ]
    artifact_paths = sorted(path for path in output_root.rglob("*") if path.is_file())
    manifest = {
        "pipeline": "RIG-Hazard preprocessing",
        "pipeline_version": "0.1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "source_files": source_manifest,
        "artifacts": [str(path.relative_to(output_root)) for path in artifact_paths] + ["manifest.json"],
        "feature_contract_sha256": hashlib.sha256((output_root / "feature_contract.json").read_bytes()).hexdigest(),
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"Preprocessing complete: {output_root}", flush=True)
    return output_root

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "meteorology_raw"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "legacy_exploration" / "analysis_outputs"

TIME_COL = "\u89c2\u6d4b\u65f6\u95f4"
STATION_COL = "\u7ad9\u70b9\u540d\u79f0"
STATION_CODE_COL = "\u7ad9\u70b9\u7f16\u53f7"
STATION_ID_COL = "\u7ad9\u70b9ID"
VOLTAGE_COL = "\u8bbe\u5907\u7535\u538b"
ICE_THICKNESS_COL = "\u8986\u51b0\u539a\u5ea6"
ICE_FREQ_COL = "\u7ed3\u51b0\u4f20\u611f\u5668\u9891\u7387"
ICE_TYPE_COL = "\u8986\u51b0\u7c7b\u578b"
TEMP_COL = "\u6c14\u6e29"
RH_COL = "\u76f8\u5bf9\u6e7f\u5ea6"
PRESSURE_COL = "\u6c14\u538b"
RAIN_COL = "\u96e8"
HOURLY_RAIN_COL = "\u5c0f\u65f6\u964d\u96e8"
WIND10_COL = "\u5341\u5206\u949f\u5e73\u5747\u98ce\u901f"
VIS_COL = "\u80fd\u89c1\u5ea6"
VIS10_COL = "\u5341\u5206\u949f\u80fd\u89c1\u5ea6"
PRECIP_PHENOM_COL = "\u964d\u6c34\u5929\u6c14\u73b0\u8c61"
VIS_OBSTACLE_COL = "\u89c6\u7a0b\u969c\u788d"

ALIAS_COLS = {
    "\u4e00\u5206\u949f\u80fd\u89c1\u5ea6": VIS_COL,
}

SELECT_COLS = [
    TIME_COL,
    STATION_COL,
    STATION_CODE_COL,
    STATION_ID_COL,
    VOLTAGE_COL,
    ICE_THICKNESS_COL,
    ICE_FREQ_COL,
    ICE_TYPE_COL,
    TEMP_COL,
    RH_COL,
    PRESSURE_COL,
    RAIN_COL,
    HOURLY_RAIN_COL,
    WIND10_COL,
    VIS_COL,
    VIS10_COL,
    PRECIP_PHENOM_COL,
    VIS_OBSTACLE_COL,
]

READ_COLS = SELECT_COLS + list(ALIAS_COLS.keys())

NUMERIC_COLS = [
    VOLTAGE_COL,
    ICE_THICKNESS_COL,
    ICE_FREQ_COL,
    TEMP_COL,
    RH_COL,
    PRESSURE_COL,
    RAIN_COL,
    HOURLY_RAIN_COL,
    WIND10_COL,
    VIS_COL,
    VIS10_COL,
]

MISSING_MARKERS = {"", "--", "nan", "NaN", "NAN", "null", "NULL", "None", "none"}
CITY_PREFIXES = [
    "\u6b66\u5937\u5c71",
    "\u9f99\u5ca9",
    "\u6cc9\u5dde",
    "\u5b81\u5fb7",
    "\u5357\u5e73",
    "\u4e09\u660e",
]
RAW_FOG_TOKEN = "\u96fe"
MERGE_GAP_MINUTES = 10
CHUNKSIZE = 200_000


@dataclass
class NumericStats:
    count: int = 0
    missing: int = 0
    total: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add_series(self, series: pd.Series) -> None:
        valid = series.dropna()
        self.count += int(valid.shape[0])
        self.missing += int(series.shape[0] - valid.shape[0])
        if valid.empty:
            return
        self.total += float(valid.sum())
        current_min = float(valid.min())
        current_max = float(valid.max())
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class Aggregate:
    rows: int = 0
    time_parse_errors: int = 0
    duplicate_adjacent_minutes: int = 0
    out_of_order_rows: int = 0
    missing_sequence_minutes: int = 0
    first_time: pd.Timestamp | None = None
    last_time: pd.Timestamp | None = None
    prev_time: pd.Timestamp | None = None
    station: str = ""
    station_code: str = ""
    station_id: str = ""
    folder_year: str = ""
    city: str = ""
    source_file: str = ""
    stats: dict[str, NumericStats] = field(default_factory=lambda: {c: NumericStats() for c in NUMERIC_COLS})
    missing_text: Counter = field(default_factory=Counter)
    anomaly_counts: Counter = field(default_factory=Counter)
    condition_counts: Counter = field(default_factory=Counter)
    precip_counter: Counter = field(default_factory=Counter)
    obstacle_counter: Counter = field(default_factory=Counter)


@dataclass
class EventState:
    active: bool = False
    station: str = ""
    city: str = ""
    year: str = ""
    source_file: str = ""
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    obs_minutes: int = 0
    max_thickness: float = 0.0
    peak_time: pd.Timestamp | None = None

    def start_event(self, station: str, city: str, year: str, source_file: str, ts: pd.Timestamp, thickness: float) -> None:
        self.active = True
        self.station = station
        self.city = city
        self.year = year
        self.source_file = source_file
        self.start = ts
        self.end = ts
        self.obs_minutes = 1
        self.max_thickness = float(thickness)
        self.peak_time = ts

    def add_observation(self, ts: pd.Timestamp, thickness: float) -> None:
        self.end = ts
        self.obs_minutes += 1
        if float(thickness) > self.max_thickness:
            self.max_thickness = float(thickness)
            self.peak_time = ts

    def close(self) -> dict[str, Any] | None:
        if not self.active or self.start is None or self.end is None:
            return None
        elapsed_minutes = int((self.end - self.start).total_seconds() // 60) + 1
        event = {
            "station": self.station,
            "city": self.city,
            "year": self.year,
            "start_time": ts_to_text(self.start),
            "end_time": ts_to_text(self.end),
            "elapsed_minutes_including_merged_gaps": elapsed_minutes,
            "icing_observation_minutes": self.obs_minutes,
            "peak_ice_thickness": round(self.max_thickness, 3),
            "peak_time": ts_to_text(self.peak_time),
            "source_file": self.source_file,
        }
        self.active = False
        return event


def clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def missing_mask(series: pd.Series) -> pd.Series:
    clean = clean_text_series(series)
    return clean.isin(MISSING_MARKERS)


def numeric_series(series: pd.Series) -> pd.Series:
    clean = clean_text_series(series)
    return pd.to_numeric(clean.where(~clean.isin(MISSING_MARKERS), None), errors="coerce")


def first_non_missing(series: pd.Series) -> str:
    clean = clean_text_series(series)
    valid = clean[~clean.isin(MISSING_MARKERS)]
    return valid.iloc[0] if not valid.empty else ""


def infer_city(station: str) -> str:
    for prefix in CITY_PREFIXES:
        if station.startswith(prefix):
            return prefix
    return "Unknown"


def infer_year(path: Path) -> str:
    match = re.search(r"(20\d{2})", str(path.parent))
    return match.group(1) if match else ""


def expected_minutes_for_year(year: str) -> int | None:
    if not year:
        return None
    y = int(year)
    return 527_040 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 525_600


def ts_to_text(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def month_key_from_ts(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y-%m")


def parse_times(series: pd.Series) -> pd.Series:
    return pd.to_datetime(clean_text_series(series), format="mixed", errors="coerce")


def pct_col_name(condition_name: str) -> str:
    return f"{condition_name}_percentage"


def update_time_quality(agg: Aggregate, times: pd.Series) -> None:
    valid = times.dropna()
    agg.time_parse_errors += int(times.isna().sum())
    if valid.empty:
        return

    current_first = valid.min()
    current_last = valid.max()
    agg.first_time = current_first if agg.first_time is None else min(agg.first_time, current_first)
    agg.last_time = current_last if agg.last_time is None else max(agg.last_time, current_last)

    ordered = valid.reset_index(drop=True)
    diffs = ordered.diff()
    if agg.prev_time is not None:
        first_diff = ordered.iloc[0] - agg.prev_time
        if first_diff == pd.Timedelta(0):
            agg.duplicate_adjacent_minutes += 1
        elif first_diff < pd.Timedelta(0):
            agg.out_of_order_rows += 1
        elif first_diff > pd.Timedelta(minutes=1):
            agg.missing_sequence_minutes += int(first_diff.total_seconds() // 60) - 1

    if ordered.shape[0] > 1:
        inner_diffs = diffs.iloc[1:]
        agg.duplicate_adjacent_minutes += int((inner_diffs == pd.Timedelta(0)).sum())
        agg.out_of_order_rows += int((inner_diffs < pd.Timedelta(0)).sum())
        gap_minutes = inner_diffs[inner_diffs > pd.Timedelta(minutes=1)].dt.total_seconds() // 60
        agg.missing_sequence_minutes += int((gap_minutes - 1).sum())

    agg.prev_time = ordered.iloc[-1]


def add_numeric_stats(agg: Aggregate, numeric: dict[str, pd.Series]) -> None:
    for col, values in numeric.items():
        agg.stats[col].add_series(values)


def add_anomalies(agg: Aggregate, numeric: dict[str, pd.Series]) -> None:
    temp = numeric[TEMP_COL]
    rh = numeric[RH_COL]
    pressure = numeric[PRESSURE_COL]
    wind10 = numeric[WIND10_COL]
    vis = numeric[VIS_COL]
    ice = numeric[ICE_THICKNESS_COL]
    voltage = numeric[VOLTAGE_COL]

    agg.anomaly_counts["air_temperature_outside_-50_60"] += int(((temp < -50) | (temp > 60)).sum())
    agg.anomaly_counts["relative_humidity_outside_0_100"] += int(((rh < 0) | (rh > 100)).sum())
    agg.anomaly_counts["station_pressure_outside_300_1100"] += int(((pressure < 300) | (pressure > 1100)).sum())
    agg.anomaly_counts["ten_minute_wind_speed_outside_0_75"] += int(((wind10 < 0) | (wind10 > 75)).sum())
    agg.anomaly_counts["visibility_outside_0_50000"] += int(((vis < 0) | (vis > 50000)).sum())
    agg.anomaly_counts["negative_ice_thickness"] += int((ice < 0).sum())
    agg.anomaly_counts["device_voltage_below_11v"] += int((voltage < 11).sum())


def add_conditions(agg: Aggregate, chunk: pd.DataFrame, numeric: dict[str, pd.Series]) -> dict[str, pd.Series]:
    temp = numeric[TEMP_COL]
    rh = numeric[RH_COL]
    wind10 = numeric[WIND10_COL]
    vis = numeric[VIS_COL]
    vis10 = numeric[VIS10_COL]
    rain = numeric[RAIN_COL]
    hourly_rain = numeric[HOURLY_RAIN_COL]
    ice = numeric[ICE_THICKNESS_COL]

    ice_type_text = clean_text_series(chunk[ICE_TYPE_COL])
    precip_text = clean_text_series(chunk[PRECIP_PHENOM_COL])
    obstacle_text = clean_text_series(chunk[VIS_OBSTACLE_COL])
    ice_type_present = ~ice_type_text.isin(MISSING_MARKERS)
    precip_present_text = ~precip_text.isin(MISSING_MARKERS)
    obstacle_present_text = ~obstacle_text.isin(MISSING_MARKERS)

    ice_positive = ice > 0
    fog_text = obstacle_text.str.contains(RAW_FOG_TOKEN, regex=False, na=False)
    lowvis = vis < 1000
    lowvis10 = vis10 < 1000
    fog_or_lowvis = fog_text | lowvis | lowvis10
    precip_record = precip_present_text | (rain > 0) | (hourly_rain > 0)

    conditions = {
        "positive_ice_thickness_minutes": ice_positive,
        "recorded_ice_type_minutes": ice_type_present,
        "recorded_ice_type_zero_thickness_minutes": ice_type_present & ~ice_positive,
        "air_temperature_le_0_minutes": temp <= 0,
        "air_temperature_le_2_rh_ge_95_minutes": (temp <= 2) & (rh >= 95),
        "air_temperature_le_0_rh_ge_90_minutes": (temp <= 0) & (rh >= 90),
        "relative_humidity_ge_95_minutes": rh >= 95,
        "fog_or_visibility_lt_1000m_minutes": fog_or_lowvis,
        "visibility_lt_1000m_minutes": lowvis,
        "visibility_lt_500m_minutes": vis < 500,
        "precipitation_record_minutes": precip_record,
        "ten_minute_wind_speed_ge_5mps_minutes": wind10 >= 5,
        "ten_minute_wind_speed_ge_10mps_minutes": wind10 >= 10,
    }
    for key, mask in conditions.items():
        agg.condition_counts[key] += int(mask.sum())

    for value, count in precip_text[precip_present_text].value_counts().items():
        agg.precip_counter[value] += int(count)
    for value, count in obstacle_text[obstacle_present_text].value_counts().items():
        agg.obstacle_counter[value] += int(count)

    return conditions


def update_group_aggregate(target: dict[tuple[str, ...], Aggregate], key: tuple[str, ...], base: Aggregate, chunk_rows: int, times: pd.Series, numeric: dict[str, pd.Series], conditions: dict[str, pd.Series], chunk: pd.DataFrame) -> None:
    agg = target[key]
    if not agg.station:
        agg.station = base.station
        agg.station_code = base.station_code
        agg.station_id = base.station_id
        agg.city = base.city
        agg.folder_year = base.folder_year
    agg.rows += chunk_rows
    valid_times = times.dropna()
    if not valid_times.empty:
        current_first = valid_times.min()
        current_last = valid_times.max()
        agg.first_time = current_first if agg.first_time is None else min(agg.first_time, current_first)
        agg.last_time = current_last if agg.last_time is None else max(agg.last_time, current_last)
    add_numeric_stats(agg, numeric)
    add_anomalies(agg, numeric)
    for condition_key, mask in conditions.items():
        agg.condition_counts[condition_key] += int(mask.sum())
    for col in SELECT_COLS:
        agg.missing_text[col] += int(missing_mask(chunk[col]).sum())
    precip_text = clean_text_series(chunk[PRECIP_PHENOM_COL])
    obstacle_text = clean_text_series(chunk[VIS_OBSTACLE_COL])
    precip_present = ~precip_text.isin(MISSING_MARKERS)
    obstacle_present = ~obstacle_text.isin(MISSING_MARKERS)
    for value, count in precip_text[precip_present].value_counts().items():
        agg.precip_counter[value] += int(count)
    for value, count in obstacle_text[obstacle_present].value_counts().items():
        agg.obstacle_counter[value] += int(count)


def process_ice_events(event_state: EventState, events: list[dict[str, Any]], station: str, city: str, year: str, source_file: str, times: pd.Series, thickness: pd.Series) -> None:
    mask = (thickness > 0) & times.notna()
    if not mask.any():
        return
    ice_times = times[mask].reset_index(drop=True)
    ice_values = thickness[mask].reset_index(drop=True)
    for ts, thick in zip(ice_times, ice_values):
        if not event_state.active:
            event_state.start_event(station, city, year, source_file, ts, float(thick))
            continue
        assert event_state.end is not None
        gap_minutes = (ts - event_state.end).total_seconds() / 60
        if gap_minutes <= MERGE_GAP_MINUTES + 1:
            event_state.add_observation(ts, float(thick))
        else:
            closed = event_state.close()
            if closed:
                events.append(closed)
            event_state.start_event(station, city, year, source_file, ts, float(thick))


def file_size_gb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths) / (1024**3)


def aggregate_to_row(agg: Aggregate, label: str = "") -> dict[str, Any]:
    full_expected = expected_minutes_for_year(agg.folder_year)
    range_expected = None
    if agg.first_time is not None and agg.last_time is not None:
        range_expected = int((agg.last_time - agg.first_time).total_seconds() // 60) + 1
    unique_estimate = max(0, agg.rows - agg.duplicate_adjacent_minutes)
    row = {
        "group": label,
        "city": agg.city,
        "station": agg.station,
        "station_code": agg.station_code,
        "station_id": agg.station_id,
        "year": agg.folder_year,
        "source_file": agg.source_file,
        "records": agg.rows,
        "start_time": ts_to_text(agg.first_time),
        "end_time": ts_to_text(agg.last_time),
        "expected_minutes_in_observed_range": range_expected if range_expected is not None else "",
        "observed_range_completeness_percentage": round(pct(unique_estimate, range_expected), 3) if range_expected else "",
        "expected_minutes_in_calendar_year": full_expected if full_expected else "",
        "calendar_year_completeness_percentage": round(pct(unique_estimate, full_expected), 3) if full_expected else "",
        "adjacent_duplicate_timestamps": agg.duplicate_adjacent_minutes,
        "out_of_order_records": agg.out_of_order_rows,
        "sequence_gap_minutes": agg.missing_sequence_minutes,
        "timestamp_parse_failures": agg.time_parse_errors,
    }
    metric_map = {
        TEMP_COL: "air_temperature",
        RH_COL: "relative_humidity",
        PRESSURE_COL: "station_pressure",
        WIND10_COL: "ten_minute_wind_speed",
        VIS_COL: "visibility",
        ICE_THICKNESS_COL: "ice_thickness",
        VOLTAGE_COL: "device_voltage",
    }
    for col, metric_name in metric_map.items():
        stats = agg.stats[col]
        row[f"{metric_name}_mean"] = round(stats.mean, 3) if stats.mean is not None else ""
        row[f"{metric_name}_minimum"] = round(stats.min_value, 3) if stats.min_value is not None else ""
        row[f"{metric_name}_maximum"] = round(stats.max_value, 3) if stats.max_value is not None else ""
    for key in [
        "positive_ice_thickness_minutes",
        "recorded_ice_type_minutes",
        "recorded_ice_type_zero_thickness_minutes",
        "air_temperature_le_0_minutes",
        "air_temperature_le_2_rh_ge_95_minutes",
        "air_temperature_le_0_rh_ge_90_minutes",
        "relative_humidity_ge_95_minutes",
        "fog_or_visibility_lt_1000m_minutes",
        "visibility_lt_1000m_minutes",
        "visibility_lt_500m_minutes",
        "precipitation_record_minutes",
        "ten_minute_wind_speed_ge_5mps_minutes",
        "ten_minute_wind_speed_ge_10mps_minutes",
    ]:
        value = agg.condition_counts[key]
        row[key] = value
        row[pct_col_name(key)] = round(pct(value, agg.rows), 3) if agg.rows else ""
    row["total_anomaly_records"] = sum(agg.anomaly_counts.values())
    return row


def month_row(month: str, agg: Aggregate) -> dict[str, Any]:
    row = aggregate_to_row(agg)
    row["month"] = month
    row.pop("group", None)
    row.pop("year", None)
    row.pop("source_file", None)
    row.pop("expected_minutes_in_calendar_year", None)
    row.pop("calendar_year_completeness_percentage", None)
    row.pop("expected_minutes_in_observed_range", None)
    row.pop("observed_range_completeness_percentage", None)
    row.pop("adjacent_duplicate_timestamps", None)
    row.pop("out_of_order_records", None)
    row.pop("sequence_gap_minutes", None)
    row.pop("timestamp_parse_failures", None)
    row.pop("start_time", None)
    row.pop("end_time", None)
    return row


def safe_div(value: float, denominator: float) -> float:
    return value / denominator if denominator else 0.0


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "\nNone.\n"
    rows = rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if abs(value) >= 1000:
        return f"{value:,.{digits}f}"
    return f"{value:.{digits}f}"


def top_counter(counter: Counter, n: int = 15) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {"category": key, "records": value, "percentage": round(pct(value, total), 3)}
        for key, value in counter.most_common(n)
    ]


def svg_bar_chart(items: list[tuple[str, float]], title: str, width: int = 920, height: int = 380, value_suffix: str = "") -> str:
    if not items:
        return "<p>No data.</p>"
    max_value = max(v for _, v in items) or 1
    pad_left, pad_right, pad_top, pad_bottom = 150, 30, 42, 42
    plot_w = width - pad_left - pad_right
    row_h = max(22, (height - pad_top - pad_bottom) // len(items))
    height = pad_top + pad_bottom + row_h * len(items)
    lines = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{pad_left}" y="24" font-size="18" font-weight="700">{html.escape(title)}</text>',
    ]
    for i, (label, value) in enumerate(items):
        y = pad_top + i * row_h
        bar_w = plot_w * value / max_value
        color = "#2f6f73" if i % 2 == 0 else "#7a5c2e"
        lines.append(f'<text x="{pad_left - 8}" y="{y + row_h * 0.65:.1f}" text-anchor="end" font-size="12">{html.escape(label)}</text>')
        lines.append(f'<rect x="{pad_left}" y="{y + 4}" width="{bar_w:.1f}" height="{max(8, row_h - 8)}" rx="3" fill="{color}" opacity="0.88"></rect>')
        lines.append(f'<text x="{pad_left + bar_w + 6:.1f}" y="{y + row_h * 0.65:.1f}" font-size="12">{value:.3f}{html.escape(value_suffix)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def svg_line_chart(items: list[tuple[str, float]], title: str, width: int = 920, height: int = 300, value_suffix: str = "%") -> str:
    if not items:
        return "<p>No data.</p>"
    pad_left, pad_right, pad_top, pad_bottom = 56, 24, 42, 56
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    max_value = max(v for _, v in items) or 1
    min_value = min(0, min(v for _, v in items))
    span = max_value - min_value or 1
    points = []
    for i, (_, value) in enumerate(items):
        x = pad_left + (plot_w * i / max(1, len(items) - 1))
        y = pad_top + plot_h - ((value - min_value) / span) * plot_h
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    lines = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{pad_left}" y="24" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}" stroke="#9aa0a6"/>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}" stroke="#9aa0a6"/>',
        f'<polyline points="{polyline}" fill="none" stroke="#2f6f73" stroke-width="3"/>',
    ]
    for i, ((label, value), (x, y)) in enumerate(zip(items, points)):
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#7a5c2e"/>')
        if i % max(1, len(items) // 12) == 0:
            lines.append(f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" font-size="11" transform="rotate(45 {x:.1f},{height - 22})">{html.escape(label)}</text>')
        if i == len(items) - 1 or value == max_value:
            lines.append(f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-size="11">{value:.2f}{html.escape(value_suffix)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def html_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 20) -> str:
    rows = rows[:max_rows]
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    all_fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_csv_chunks(path: Path):
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            reader = pd.read_csv(
                path,
                usecols=lambda col: col in READ_COLS,
                dtype=str,
                chunksize=CHUNKSIZE,
                encoding=encoding,
                keep_default_na=False,
                low_memory=False,
            )
            for chunk in reader:
                yield chunk
            return
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error


def build_reports(
    csv_paths: list[Path],
    station_year_rows: list[dict[str, Any]],
    month_rows: list[dict[str, Any]],
    station_month_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    global_agg: Aggregate,
    started_at: datetime,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    station_year_rows_sorted = sorted(
        station_year_rows,
        key=lambda r: (r.get("city", ""), r.get("station", ""), str(r.get("year", ""))),
    )
    low_coverage = sorted(
        station_year_rows,
        key=lambda r: float(r.get("calendar_year_completeness_percentage", 0) if r.get("calendar_year_completeness_percentage", "") != "" else 0),
    )[:12]
    top_ice_stations = sorted(
        station_year_rows,
        key=lambda r: float(r.get("positive_ice_thickness_minutes", 0)),
        reverse=True,
    )[:15]
    top_fog_stations = sorted(
        station_year_rows,
        key=lambda r: float(r.get("fog_or_visibility_lt_1000m_minutes_percentage", 0) if r.get("fog_or_visibility_lt_1000m_minutes_percentage", "") != "" else 0),
        reverse=True,
    )[:15]
    top_events = sorted(events, key=lambda e: (e["peak_ice_thickness"], e["elapsed_minutes_including_merged_gaps"]), reverse=True)[:20]

    write_csv(OUTPUT_ROOT / "station_year_summary.csv", station_year_rows_sorted)
    write_csv(OUTPUT_ROOT / "month_summary.csv", month_rows)
    write_csv(OUTPUT_ROOT / "station_month_summary.csv", station_month_rows)
    write_csv(OUTPUT_ROOT / "top_icing_events.csv", top_events)

    missing_rows = []
    for col in SELECT_COLS:
        missing = global_agg.missing_text[col]
        missing_rows.append({"field": col, "missing_or_marker_records": missing, "missing_percentage": round(pct(missing, global_agg.rows), 4)})
    write_csv(OUTPUT_ROOT / "missingness_summary.csv", missing_rows)

    anomaly_rows = [
        {"anomaly_type": key, "records": value, "percentage": round(pct(value, global_agg.rows), 5)}
        for key, value in global_agg.anomaly_counts.most_common()
    ]
    write_csv(OUTPUT_ROOT / "anomaly_summary.csv", anomaly_rows)

    phenomena_rows = []
    for row in top_counter(global_agg.precip_counter, 50):
        row["field"] = PRECIP_PHENOM_COL
        phenomena_rows.append(row)
    for row in top_counter(global_agg.obstacle_counter, 50):
        row["field"] = VIS_OBSTACLE_COL
        phenomena_rows.append(row)
    write_csv(OUTPUT_ROOT / "phenomena_summary.csv", phenomena_rows)

    station_count = len({r["station"] for r in station_year_rows if r.get("station")})
    year_count = sorted({str(r["year"]) for r in station_year_rows if r.get("year")})
    total_size = file_size_gb(csv_paths)
    avg_temp = global_agg.stats[TEMP_COL].mean
    min_temp = global_agg.stats[TEMP_COL].min_value
    max_temp = global_agg.stats[TEMP_COL].max_value
    avg_rh = global_agg.stats[RH_COL].mean
    avg_vis = global_agg.stats[VIS_COL].mean
    max_ice = global_agg.stats[ICE_THICKNESS_COL].max_value
    ice_minutes = global_agg.condition_counts["positive_ice_thickness_minutes"]
    ice_type_minutes = global_agg.condition_counts["recorded_ice_type_minutes"]
    fog_minutes = global_agg.condition_counts["fog_or_visibility_lt_1000m_minutes"]
    cold_moist_minutes = global_agg.condition_counts["air_temperature_le_2_rh_ge_95_minutes"]
    freezing_humid_minutes = global_agg.condition_counts["air_temperature_le_0_rh_ge_90_minutes"]

    month_ice_chart_items = [
        (r["month"], float(r.get("positive_ice_thickness_minutes_percentage", 0) or 0))
        for r in sorted(month_rows, key=lambda x: x["month"])
    ]
    month_fog_chart_items = [
        (r["month"], float(r.get("fog_or_visibility_lt_1000m_minutes_percentage", 0) or 0))
        for r in sorted(month_rows, key=lambda x: x["month"])
    ]
    top_ice_chart_items = [
        (f'{r.get("station","")} {r.get("year","")}', float(r.get("positive_ice_thickness_minutes_percentage", 0) or 0))
        for r in top_ice_stations
    ]
    top_fog_chart_items = [
        (f'{r.get("station","")} {r.get("year","")}', float(r.get("fog_or_visibility_lt_1000m_minutes_percentage", 0) or 0))
        for r in top_fog_stations
    ]
    low_cov_chart_items = [
        (f'{r.get("station","")} {r.get("year","")}', float(r.get("calendar_year_completeness_percentage", 0) or 0))
        for r in low_coverage
    ]

    md = f"""# Condensation-Weather Data Analysis Report

Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Data overview

- Data directory: `{DATA_ROOT}`
- CSV files: {len(csv_paths)}, totaling approximately {total_size:.2f} GB
- Years: {", ".join(year_count)}
- Stations: {station_count}; station-year files: {len(station_year_rows)}
- Temporal resolution: 1 minute
- Total records: {global_agg.rows:,}
- Overall date range: {ts_to_text(global_agg.first_time)} to {ts_to_text(global_agg.last_time)}

## 2. Key findings

1. The data primarily contain minute-level weather and icing observations from mountain and high-elevation stations in Fujian during 2022–2024. Field completeness is generally good, but some station-year files are substantially incomplete.
2. Positive ice-thickness observations total {ice_minutes:,} minutes ({pct(ice_minutes, global_agg.rows):.4f}% of all records). Recorded ice-type values total {ice_type_minutes:,} minutes ({pct(ice_type_minutes, global_agg.rows):.4f}%).
3. Fog or low visibility occurs for {fog_minutes:,} minutes ({pct(fog_minutes, global_agg.rows):.2f}% of all records), so condensation or fog environments are much more common than positive ice-thickness events.
4. Potential cold-humid conditions occur for {cold_moist_minutes:,} minutes at air temperature <=2 °C and relative humidity >=95%, and {freezing_humid_minutes:,} minutes at air temperature <=0 °C and relative humidity >=90%.
5. Mean air temperature is {format_number(avg_temp)} °C, ranging from {format_number(min_temp)} °C to {format_number(max_temp)} °C. Mean relative humidity is {format_number(avg_rh)}%, mean visibility is {format_number(avg_vis)} m, and maximum ice thickness is {format_number(max_ice, 3)}.

## 3. Data-quality audit

### Station-years with the lowest coverage

{markdown_table(low_coverage, ["city", "station", "year", "records", "calendar_year_completeness_percentage", "start_time", "end_time", "sequence_gap_minutes"], 12)}

### Fields with the highest missingness

{markdown_table(sorted(missing_rows, key=lambda r: r["missing_percentage"], reverse=True), ["field", "missing_or_marker_records", "missing_percentage"], 12)}

### Anomaly checks

{markdown_table(anomaly_rows, ["anomaly_type", "records", "percentage"], 12)}

## 4. Icing-event analysis

An icing event is defined by positive ice thickness. Consecutive positive observations separated by no more than {MERGE_GAP_MINUTES} minutes are merged into one event, allowing short transmission gaps or brief sensor resets to zero.

- Icing events: {len(events):,}
- Positive-icing observation minutes: {ice_minutes:,}
- Positive-icing share of all minutes: {pct(ice_minutes, global_agg.rows):.4f}%
- Maximum ice thickness: {format_number(max_ice, 3)}

### Events with the greatest peak ice thickness

{markdown_table(top_events, ["station", "city", "year", "start_time", "end_time", "elapsed_minutes_including_merged_gaps", "icing_observation_minutes", "peak_ice_thickness", "peak_time"], 20)}

### Station-years with the most positive-icing minutes

{markdown_table(top_ice_stations, ["city", "station", "year", "positive_ice_thickness_minutes", "positive_ice_thickness_minutes_percentage", "ice_thickness_maximum", "air_temperature_mean", "relative_humidity_mean"], 15)}

## 5. Fog, low visibility, and condensation environments

### Station-years with the largest fog or low-visibility share

{markdown_table(top_fog_stations, ["city", "station", "year", "fog_or_visibility_lt_1000m_minutes", "fog_or_visibility_lt_1000m_minutes_percentage", "visibility_mean", "air_temperature_mean", "relative_humidity_mean"], 15)}

### Top 15 precipitation phenomena

{markdown_table(top_counter(global_agg.precip_counter, 15), ["category", "records", "percentage"], 15)}

### Top 15 visibility obstructions

{markdown_table(top_counter(global_agg.obstacle_counter, 15), ["category", "records", "percentage"], 15)}

## 6. Seasonality

See `analysis_outputs/month_summary.csv` for monthly statistics. Winter and early spring deserve particular attention because cold-humid conditions and observed icing are more likely to coincide. Fog and low visibility occur in more months and should be screened jointly with temperature.

## 7. Modelling recommendations

1. Use positive ice thickness as a strong alert-modelling label. Candidate features include joint temperature <=2 °C and relative humidity >=95%, fog or visibility below 1,000 m, ten-minute mean wind speed, pressure change, and hourly rain.
2. Hourly rain is an hourly field repeated in minute data and must not be summed by minute. Estimate hourly or daily precipitation after taking an hourly maximum or final value.
3. Exclude or down-weight low-coverage files, especially severely incomplete station-years, to avoid distorting seasonal or station rankings.
4. Build a structured station table from the original register, including elevation, coordinates, and terrain exposure, to strengthen spatial interpretation.
5. Ice thickness may be stratified as minor (0–1), moderate (1–5), and strong (>5) and combined with duration in an event-intensity index.

## 8. Output files

- `analysis_outputs/station_year_summary.csv`: complete station-year statistics
- `analysis_outputs/month_summary.csv`: monthly statistics for all data
- `analysis_outputs/station_month_summary.csv`: station-month statistics
- `analysis_outputs/top_icing_events.csv`: peak icing events
- `analysis_outputs/missingness_summary.csv`: field missingness
- `analysis_outputs/anomaly_summary.csv`: anomaly scan
- `analysis_outputs/phenomena_summary.csv`: weather-phenomenon and visibility-obstruction frequencies
- `analysis_outputs/report.html`: HTML report with charts
"""
    (OUTPUT_ROOT / "comprehensive_report.md").write_text(md, encoding="utf-8")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Condensation-Weather Data Analysis Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #1f2933; background: #f6f7f4; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
h1, h2 {{ color: #143642; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 20px 0; }}
.metric {{ background: #fff; border: 1px solid #d8ded8; border-radius: 6px; padding: 14px 16px; }}
.metric b {{ display: block; font-size: 22px; color: #2f6f73; margin-bottom: 4px; }}
.panel {{ background: #fff; border: 1px solid #d8ded8; border-radius: 6px; padding: 18px; margin: 18px 0; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e4e7e2; padding: 7px 8px; text-align: left; white-space: nowrap; }}
th {{ background: #eef2ec; color: #253238; }}
.note {{ color: #56636b; line-height: 1.65; }}
code {{ background: #eef2ec; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
<h1>Condensation-Weather Data Analysis Report</h1>
<p class="note">Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. Event definition: positive ice thickness, merging gaps no longer than {MERGE_GAP_MINUTES} minutes.</p>
<section class="summary">
<div class="metric"><b>{len(csv_paths)}</b>CSV files</div>
<div class="metric"><b>{station_count}</b>stations</div>
<div class="metric"><b>{global_agg.rows:,}</b>minute records</div>
<div class="metric"><b>{ice_minutes:,}</b>positive-icing minutes</div>
<div class="metric"><b>{pct(ice_minutes, global_agg.rows):.4f}%</b>positive-icing share</div>
<div class="metric"><b>{fog_minutes:,}</b>fog/low-visibility minutes</div>
</section>
<div class="panel">{svg_line_chart(month_ice_chart_items, "Monthly positive-icing share", value_suffix="%")}</div>
<div class="panel">{svg_line_chart(month_fog_chart_items, "Monthly fog or low-visibility share", value_suffix="%")}</div>
<div class="panel">{svg_bar_chart(top_ice_chart_items, "Station-years with the greatest positive-icing share", value_suffix="%")}</div>
<div class="panel">{svg_bar_chart(top_fog_chart_items, "Station-years with the greatest fog or low-visibility share", value_suffix="%")}</div>
<div class="panel">{svg_bar_chart(low_cov_chart_items, "Station-years with the lowest completeness", value_suffix="%")}</div>
<h2>Top 20 Events by Peak Ice Thickness</h2>
<div class="panel">{html_table(top_events, ["station", "city", "year", "start_time", "end_time", "elapsed_minutes_including_merged_gaps", "icing_observation_minutes", "peak_ice_thickness", "peak_time"], 20)}</div>
<h2>Top 12 Fields by Missingness</h2>
<div class="panel">{html_table(sorted(missing_rows, key=lambda r: r["missing_percentage"], reverse=True), ["field", "missing_or_marker_records", "missing_percentage"], 12)}</div>
<h2>Notes</h2>
<p class="note">Precipitation is not summed directly across minute rows. Aggregate by hourly windows before estimating daily or annual precipitation. Complete CSV outputs are in <code>analysis_outputs</code>.</p>
</main>
</body>
</html>
"""
    (OUTPUT_ROOT / "report.html").write_text(html_doc, encoding="utf-8")


def analyze(max_files: int | None = None) -> None:
    started_at = datetime.now()
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")

    csv_paths = sorted(DATA_ROOT.glob("**/*.csv"))
    if max_files is not None:
        csv_paths = csv_paths[:max_files]
    if not csv_paths:
        raise FileNotFoundError("No CSV files found.")

    global_agg = Aggregate(source_file="ALL")
    station_year_aggs: list[Aggregate] = []
    month_aggs: dict[tuple[str], Aggregate] = defaultdict(Aggregate)
    station_month_aggs: dict[tuple[str, str], Aggregate] = defaultdict(Aggregate)
    events: list[dict[str, Any]] = []

    for idx, path in enumerate(csv_paths, start=1):
        rel_path = str(path.relative_to(PROJECT_ROOT))
        year = infer_year(path)
        file_agg = Aggregate(folder_year=year, source_file=rel_path)
        event_state = EventState()
        print(f"[{idx}/{len(csv_paths)}] {rel_path}", flush=True)

        try:
            for chunk in iter_csv_chunks(path):
                for alias, canonical in ALIAS_COLS.items():
                    if alias in chunk.columns and canonical not in chunk.columns:
                        chunk[canonical] = chunk[alias]
                for col in SELECT_COLS:
                    if col not in chunk.columns:
                        chunk[col] = ""
                chunk = chunk[SELECT_COLS]
                file_agg.rows += int(chunk.shape[0])
                global_agg.rows += int(chunk.shape[0])

                if not file_agg.station:
                    file_agg.station = first_non_missing(chunk[STATION_COL]) or path.stem.split("20")[0]
                    file_agg.station_code = first_non_missing(chunk[STATION_CODE_COL])
                    file_agg.station_id = first_non_missing(chunk[STATION_ID_COL])
                    file_agg.city = infer_city(file_agg.station)
                if not global_agg.first_time:
                    global_agg.station = "All data"

                times = parse_times(chunk[TIME_COL])
                numeric = {col: numeric_series(chunk[col]) for col in NUMERIC_COLS}

                update_time_quality(file_agg, times)
                update_time_quality(global_agg, times)
                add_numeric_stats(file_agg, numeric)
                add_numeric_stats(global_agg, numeric)
                add_anomalies(file_agg, numeric)
                add_anomalies(global_agg, numeric)

                for col in SELECT_COLS:
                    miss = int(missing_mask(chunk[col]).sum())
                    file_agg.missing_text[col] += miss
                    global_agg.missing_text[col] += miss

                conditions = add_conditions(file_agg, chunk, numeric)
                global_conditions = add_conditions(global_agg, chunk, numeric)
                assert conditions.keys() == global_conditions.keys()

                process_ice_events(
                    event_state,
                    events,
                    file_agg.station,
                    file_agg.city,
                    file_agg.folder_year,
                    rel_path,
                    times,
                    numeric[ICE_THICKNESS_COL],
                )

                valid_months = month_key_from_ts(times)
                month_values = sorted(valid_months.dropna().unique())
                for month in month_values:
                    month_mask = valid_months == month
                    sub_numeric = {col: values[month_mask] for col, values in numeric.items()}
                    sub_chunk = chunk.loc[month_mask]
                    month_base = Aggregate(station="All data", city="All data", folder_year=month[:4])
                    month_conditions = {name: mask[month_mask] for name, mask in conditions.items()}
                    update_group_aggregate(
                        month_aggs,
                        (month,),
                        month_base,
                        int(month_mask.sum()),
                        times[month_mask],
                        sub_numeric,
                        month_conditions,
                        sub_chunk,
                    )
                    station_month_base = Aggregate(
                        station=file_agg.station,
                        station_code=file_agg.station_code,
                        station_id=file_agg.station_id,
                        city=file_agg.city,
                        folder_year=month[:4],
                    )
                    update_group_aggregate(
                        station_month_aggs,
                        (file_agg.station, month),
                        station_month_base,
                        int(month_mask.sum()),
                        times[month_mask],
                        sub_numeric,
                        month_conditions,
                        sub_chunk,
                    )
        except Exception as exc:
            file_agg.anomaly_counts["file_read_failure"] += 1
            print(f"  !! failed: {exc}", flush=True)

        closed = event_state.close()
        if closed:
            events.append(closed)
        station_year_aggs.append(file_agg)

    station_year_rows = [aggregate_to_row(agg, "station-year") for agg in station_year_aggs]
    month_rows = [month_row(key[0], agg) for key, agg in sorted(month_aggs.items())]
    station_month_rows = [month_row(key[1], agg) for key, agg in sorted(station_month_aggs.items(), key=lambda item: (item[0][0], item[0][1]))]

    build_reports(
        csv_paths,
        station_year_rows,
        month_rows,
        station_month_rows,
        events,
        global_agg,
        started_at,
    )
    elapsed = datetime.now() - started_at
    print(f"Done in {elapsed}. Outputs: {OUTPUT_ROOT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one-minute icing and condensation weather CSV files.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit file count for a quick subset run.")
    args = parser.parse_args()
    analyze(max_files=args.max_files)


if __name__ == "__main__":
    main()

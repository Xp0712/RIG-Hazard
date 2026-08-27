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


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "(林立铮2025.6.21)凝结类天气数据"
OUTPUT_ROOT = PROJECT_ROOT / "analysis_outputs"

TIME_COL = "观测时间"
STATION_COL = "站点名称"
STATION_CODE_COL = "站点编号"
STATION_ID_COL = "站点ID"
VOLTAGE_COL = "设备电压"
ICE_THICKNESS_COL = "覆冰厚度"
ICE_FREQ_COL = "结冰传感器频率"
ICE_TYPE_COL = "覆冰类型"
TEMP_COL = "气温"
RH_COL = "相对湿度"
PRESSURE_COL = "气压"
RAIN_COL = "雨"
HOURLY_RAIN_COL = "小时降雨"
WIND10_COL = "十分钟平均风速"
VIS_COL = "能见度"
VIS10_COL = "十分钟能见度"
PRECIP_PHENOM_COL = "降水天气现象"
VIS_OBSTACLE_COL = "视程障碍"

ALIAS_COLS = {
    "一分钟能见度": VIS_COL,
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
CITY_PREFIXES = ["武夷山", "龙岩", "泉州", "宁德", "南平", "三明"]
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
            "站点": self.station,
            "城市": self.city,
            "年份": self.year,
            "开始时间": ts_to_text(self.start),
            "结束时间": ts_to_text(self.end),
            "持续分钟_含合并间断": elapsed_minutes,
            "覆冰观测分钟": self.obs_minutes,
            "峰值覆冰厚度": round(self.max_thickness, 3),
            "峰值时间": ts_to_text(self.peak_time),
            "来源文件": self.source_file,
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
    return "未识别"


def infer_year(path: Path) -> str:
    match = re.search(r"(20\d{2})年", str(path.parent))
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
    return f"{condition_name[:-2]}占比%" if condition_name.endswith("分钟") else f"{condition_name}占比%"


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

    agg.anomaly_counts["气温<-50或>60"] += int(((temp < -50) | (temp > 60)).sum())
    agg.anomaly_counts["相对湿度<0或>100"] += int(((rh < 0) | (rh > 100)).sum())
    agg.anomaly_counts["气压<300或>1100"] += int(((pressure < 300) | (pressure > 1100)).sum())
    agg.anomaly_counts["十分钟风速<0或>75"] += int(((wind10 < 0) | (wind10 > 75)).sum())
    agg.anomaly_counts["能见度<0或>50000"] += int(((vis < 0) | (vis > 50000)).sum())
    agg.anomaly_counts["覆冰厚度<0"] += int((ice < 0).sum())
    agg.anomaly_counts["设备电压<11V"] += int((voltage < 11).sum())


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
    fog_text = obstacle_text.str.contains("雾", regex=False, na=False)
    lowvis = vis < 1000
    lowvis10 = vis10 < 1000
    fog_or_lowvis = fog_text | lowvis | lowvis10
    precip_record = precip_present_text | (rain > 0) | (hourly_rain > 0)

    conditions = {
        "覆冰厚度>0分钟": ice_positive,
        "覆冰类型有记录分钟": ice_type_present,
        "覆冰类型有记录但厚度为0分钟": ice_type_present & ~ice_positive,
        "气温<=0分钟": temp <= 0,
        "气温<=2且湿度>=95分钟": (temp <= 2) & (rh >= 95),
        "气温<=0且湿度>=90分钟": (temp <= 0) & (rh >= 90),
        "湿度>=95分钟": rh >= 95,
        "雾或能见度<1000m分钟": fog_or_lowvis,
        "能见度<1000m分钟": lowvis,
        "能见度<500m分钟": vis < 500,
        "降水记录分钟": precip_record,
        "十分钟风速>=5m/s分钟": wind10 >= 5,
        "十分钟风速>=10m/s分钟": wind10 >= 10,
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
        "分组": label,
        "城市": agg.city,
        "站点": agg.station,
        "站点编号": agg.station_code,
        "站点ID": agg.station_id,
        "年份": agg.folder_year,
        "来源文件": agg.source_file,
        "记录数": agg.rows,
        "起始时间": ts_to_text(agg.first_time),
        "结束时间": ts_to_text(agg.last_time),
        "按首末时间应有分钟": range_expected if range_expected is not None else "",
        "按首末时间完整率%": round(pct(unique_estimate, range_expected), 3) if range_expected else "",
        "按自然年应有分钟": full_expected if full_expected else "",
        "按自然年完整率%": round(pct(unique_estimate, full_expected), 3) if full_expected else "",
        "相邻重复时间戳": agg.duplicate_adjacent_minutes,
        "时间倒序记录": agg.out_of_order_rows,
        "序列缺口分钟": agg.missing_sequence_minutes,
        "时间解析失败": agg.time_parse_errors,
    }
    metric_map = {
        TEMP_COL: "气温",
        RH_COL: "相对湿度",
        PRESSURE_COL: "气压",
        WIND10_COL: "十分钟风速",
        VIS_COL: "能见度",
        ICE_THICKNESS_COL: "覆冰厚度",
        VOLTAGE_COL: "设备电压",
    }
    for col, label_cn in metric_map.items():
        stats = agg.stats[col]
        row[f"{label_cn}均值"] = round(stats.mean, 3) if stats.mean is not None else ""
        row[f"{label_cn}最小值"] = round(stats.min_value, 3) if stats.min_value is not None else ""
        row[f"{label_cn}最大值"] = round(stats.max_value, 3) if stats.max_value is not None else ""
    for key in [
        "覆冰厚度>0分钟",
        "覆冰类型有记录分钟",
        "覆冰类型有记录但厚度为0分钟",
        "气温<=0分钟",
        "气温<=2且湿度>=95分钟",
        "气温<=0且湿度>=90分钟",
        "湿度>=95分钟",
        "雾或能见度<1000m分钟",
        "能见度<1000m分钟",
        "能见度<500m分钟",
        "降水记录分钟",
        "十分钟风速>=5m/s分钟",
        "十分钟风速>=10m/s分钟",
    ]:
        value = agg.condition_counts[key]
        row[key] = value
        row[pct_col_name(key)] = round(pct(value, agg.rows), 3) if agg.rows else ""
    row["异常记录合计"] = sum(agg.anomaly_counts.values())
    return row


def month_row(month: str, agg: Aggregate) -> dict[str, Any]:
    row = aggregate_to_row(agg)
    row["月份"] = month
    row.pop("分组", None)
    row.pop("年份", None)
    row.pop("来源文件", None)
    row.pop("按自然年应有分钟", None)
    row.pop("按自然年完整率%", None)
    row.pop("按首末时间应有分钟", None)
    row.pop("按首末时间完整率%", None)
    row.pop("相邻重复时间戳", None)
    row.pop("时间倒序记录", None)
    row.pop("序列缺口分钟", None)
    row.pop("时间解析失败", None)
    row.pop("起始时间", None)
    row.pop("结束时间", None)
    return row


def safe_div(value: float, denominator: float) -> float:
    return value / denominator if denominator else 0.0


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "\n无。\n"
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
        {"类别": key, "记录数": value, "占比%": round(pct(value, total), 3)}
        for key, value in counter.most_common(n)
    ]


def svg_bar_chart(items: list[tuple[str, float]], title: str, width: int = 920, height: int = 380, value_suffix: str = "") -> str:
    if not items:
        return "<p>暂无数据</p>"
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
        return "<p>暂无数据</p>"
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
        key=lambda r: (r.get("城市", ""), r.get("站点", ""), str(r.get("年份", ""))),
    )
    low_coverage = sorted(
        station_year_rows,
        key=lambda r: float(r.get("按自然年完整率%", 0) if r.get("按自然年完整率%", "") != "" else 0),
    )[:12]
    top_ice_stations = sorted(
        station_year_rows,
        key=lambda r: float(r.get("覆冰厚度>0分钟", 0)),
        reverse=True,
    )[:15]
    top_fog_stations = sorted(
        station_year_rows,
        key=lambda r: float(r.get("雾或能见度<1000m分钟占比%", 0) if r.get("雾或能见度<1000m分钟占比%", "") != "" else 0),
        reverse=True,
    )[:15]
    top_events = sorted(events, key=lambda e: (e["峰值覆冰厚度"], e["持续分钟_含合并间断"]), reverse=True)[:20]

    write_csv(OUTPUT_ROOT / "station_year_summary.csv", station_year_rows_sorted)
    write_csv(OUTPUT_ROOT / "month_summary.csv", month_rows)
    write_csv(OUTPUT_ROOT / "station_month_summary.csv", station_month_rows)
    write_csv(OUTPUT_ROOT / "top_icing_events.csv", top_events)

    missing_rows = []
    for col in SELECT_COLS:
        missing = global_agg.missing_text[col]
        missing_rows.append({"字段": col, "缺失或--记录数": missing, "缺失率%": round(pct(missing, global_agg.rows), 4)})
    write_csv(OUTPUT_ROOT / "missingness_summary.csv", missing_rows)

    anomaly_rows = [
        {"异常类型": key, "记录数": value, "占比%": round(pct(value, global_agg.rows), 5)}
        for key, value in global_agg.anomaly_counts.most_common()
    ]
    write_csv(OUTPUT_ROOT / "anomaly_summary.csv", anomaly_rows)

    phenomena_rows = []
    for row in top_counter(global_agg.precip_counter, 50):
        row["字段"] = PRECIP_PHENOM_COL
        phenomena_rows.append(row)
    for row in top_counter(global_agg.obstacle_counter, 50):
        row["字段"] = VIS_OBSTACLE_COL
        phenomena_rows.append(row)
    write_csv(OUTPUT_ROOT / "phenomena_summary.csv", phenomena_rows)

    station_count = len({r["站点"] for r in station_year_rows if r.get("站点")})
    year_count = sorted({str(r["年份"]) for r in station_year_rows if r.get("年份")})
    total_size = file_size_gb(csv_paths)
    avg_temp = global_agg.stats[TEMP_COL].mean
    min_temp = global_agg.stats[TEMP_COL].min_value
    max_temp = global_agg.stats[TEMP_COL].max_value
    avg_rh = global_agg.stats[RH_COL].mean
    avg_vis = global_agg.stats[VIS_COL].mean
    max_ice = global_agg.stats[ICE_THICKNESS_COL].max_value
    ice_minutes = global_agg.condition_counts["覆冰厚度>0分钟"]
    ice_type_minutes = global_agg.condition_counts["覆冰类型有记录分钟"]
    fog_minutes = global_agg.condition_counts["雾或能见度<1000m分钟"]
    cold_moist_minutes = global_agg.condition_counts["气温<=2且湿度>=95分钟"]
    freezing_humid_minutes = global_agg.condition_counts["气温<=0且湿度>=90分钟"]

    month_ice_chart_items = [
        (r["月份"], float(r.get("覆冰厚度>0分钟占比%", 0) or 0))
        for r in sorted(month_rows, key=lambda x: x["月份"])
    ]
    month_fog_chart_items = [
        (r["月份"], float(r.get("雾或能见度<1000m分钟占比%", 0) or 0))
        for r in sorted(month_rows, key=lambda x: x["月份"])
    ]
    top_ice_chart_items = [
        (f'{r.get("站点","")} {r.get("年份","")}', float(r.get("覆冰厚度>0分钟占比%", 0) or 0))
        for r in top_ice_stations
    ]
    top_fog_chart_items = [
        (f'{r.get("站点","")} {r.get("年份","")}', float(r.get("雾或能见度<1000m分钟占比%", 0) or 0))
        for r in top_fog_stations
    ]
    low_cov_chart_items = [
        (f'{r.get("站点","")} {r.get("年份","")}', float(r.get("按自然年完整率%", 0) or 0))
        for r in low_coverage
    ]

    md = f"""# 凝结类天气数据综合分析报告

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 数据概况

- 数据目录：`{DATA_ROOT}`
- CSV 文件：{len(csv_paths)} 个，合计约 {total_size:.2f} GB
- 覆盖年份：{", ".join(year_count)}
- 站点数量：{station_count} 个；站点-年份文件：{len(station_year_rows)} 个
- 时间粒度：1 分钟级观测
- 总记录数：{global_agg.rows:,} 条
- 全库时间范围：{ts_to_text(global_agg.first_time)} 至 {ts_to_text(global_agg.last_time)}

## 2. 关键结论

1. 数据主体是 2022-2024 年福建山地/高海拔站点的分钟级气象与覆冰观测，字段完整度整体较好，但个别站点-年份文件明显不完整。
2. `覆冰厚度 > 0` 的实际覆冰记录共有 {ice_minutes:,} 分钟，占全库 {pct(ice_minutes, global_agg.rows):.4f}%；`覆冰类型` 有记录共有 {ice_type_minutes:,} 分钟，占 {pct(ice_type_minutes, global_agg.rows):.4f}%。
3. 雾或低能见度记录共有 {fog_minutes:,} 分钟，占全库 {pct(fog_minutes, global_agg.rows):.2f}%，说明“凝结/雾”环境远比实际覆冰厚度事件更常见。
4. 低温高湿潜势条件较集中：`气温<=2℃且湿度>=95%` 共 {cold_moist_minutes:,} 分钟，`气温<=0℃且湿度>=90%` 共 {freezing_humid_minutes:,} 分钟。
5. 全库平均气温 {format_number(avg_temp)}℃，温度范围 {format_number(min_temp)}℃ 至 {format_number(max_temp)}℃；平均相对湿度 {format_number(avg_rh)}%；平均能见度 {format_number(avg_vis)} m；最大覆冰厚度 {format_number(max_ice, 3)}。

## 3. 数据质量审计

### 覆盖率最低的站点-年份

{markdown_table(low_coverage, ["城市", "站点", "年份", "记录数", "按自然年完整率%", "起始时间", "结束时间", "序列缺口分钟"], 12)}

### 缺失率最高字段

{markdown_table(sorted(missing_rows, key=lambda r: r["缺失率%"], reverse=True), ["字段", "缺失或--记录数", "缺失率%"], 12)}

### 异常值检查

{markdown_table(anomaly_rows, ["异常类型", "记录数", "占比%"], 12)}

## 4. 覆冰事件分析

事件定义：以 `覆冰厚度 > 0` 为实际覆冰判据；连续覆冰观测之间若间隔不超过 {MERGE_GAP_MINUTES} 分钟，则合并为同一事件。该定义能容忍短时传输缺测或传感器短暂回零。

- 覆冰事件数：{len(events):,}
- 覆冰观测分钟：{ice_minutes:,}
- 全库覆冰分钟占比：{pct(ice_minutes, global_agg.rows):.4f}%
- 最大覆冰厚度：{format_number(max_ice, 3)}

### 峰值覆冰厚度最高事件

{markdown_table(top_events, ["站点", "城市", "年份", "开始时间", "结束时间", "持续分钟_含合并间断", "覆冰观测分钟", "峰值覆冰厚度", "峰值时间"], 20)}

### 覆冰分钟最多的站点-年份

{markdown_table(top_ice_stations, ["城市", "站点", "年份", "覆冰厚度>0分钟", "覆冰厚度>0占比%", "覆冰厚度最大值", "气温均值", "相对湿度均值"], 15)}

## 5. 雾、低能见度与凝结环境

### 雾/低能见度占比最高的站点-年份

{markdown_table(top_fog_stations, ["城市", "站点", "年份", "雾或能见度<1000m分钟", "雾或能见度<1000m占比%", "能见度均值", "气温均值", "相对湿度均值"], 15)}

### 降水天气现象 Top 15

{markdown_table(top_counter(global_agg.precip_counter, 15), ["类别", "记录数", "占比%"], 15)}

### 视程障碍 Top 15

{markdown_table(top_counter(global_agg.obstacle_counter, 15), ["类别", "记录数", "占比%"], 15)}

## 6. 季节性

月度统计表见 `analysis_outputs/month_summary.csv`。整体建议重点关注冬季及早春月份，因为低温高湿和实际覆冰更容易同时出现；雾/低能见度则在多个月份均较常见，需要和温度阈值联合筛选。

## 7. 建模与业务建议

1. 后续若要做覆冰预警模型，建议将 `覆冰厚度>0` 作为强标签，将 `气温<=2℃且湿度>=95%`、`雾或能见度<1000m`、`十分钟平均风速`、`气压变化`、`小时降雨` 作为候选特征。
2. `小时降雨` 是分钟表中的小时尺度字段，不建议直接逐分钟累加；若要估计小时/日降水量，应先按小时取最大值或末值，再汇总。
3. 对低覆盖文件应单独剔除或降权，尤其是完整率极低的站点-年份，避免拉偏季节性或站点排序。
4. 建议从经纬度图片或原始站点台账中整理结构化站点表，增加海拔、经纬度、地形暴露度，空间解释会明显增强。
5. 对覆冰厚度可进一步设分级：轻微 `0-1`、中等 `1-5`、较强 `>5`，结合持续时间构建“事件强度指数”。

## 8. 输出文件

- `analysis_outputs/station_year_summary.csv`：站点-年份完整统计
- `analysis_outputs/month_summary.csv`：全库月度统计
- `analysis_outputs/station_month_summary.csv`：站点-月份统计
- `analysis_outputs/top_icing_events.csv`：峰值覆冰事件
- `analysis_outputs/missingness_summary.csv`：字段缺失率
- `analysis_outputs/anomaly_summary.csv`：异常值扫描
- `analysis_outputs/phenomena_summary.csv`：天气现象/视程障碍频次
- `analysis_outputs/report.html`：带图表的 HTML 报告
"""
    (OUTPUT_ROOT / "comprehensive_report.md").write_text(md, encoding="utf-8")

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>凝结类天气数据综合分析报告</title>
<style>
body {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; margin: 0; color: #1f2933; background: #f6f7f4; }}
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
<h1>凝结类天气数据综合分析报告</h1>
<p class="note">生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}；事件定义：<code>覆冰厚度 &gt; 0</code>，间断不超过 {MERGE_GAP_MINUTES} 分钟合并。</p>
<section class="summary">
<div class="metric"><b>{len(csv_paths)}</b>CSV 文件</div>
<div class="metric"><b>{station_count}</b>站点</div>
<div class="metric"><b>{global_agg.rows:,}</b>分钟记录</div>
<div class="metric"><b>{ice_minutes:,}</b>覆冰分钟</div>
<div class="metric"><b>{pct(ice_minutes, global_agg.rows):.4f}%</b>覆冰分钟占比</div>
<div class="metric"><b>{fog_minutes:,}</b>雾/低能见度分钟</div>
</section>
<div class="panel">{svg_line_chart(month_ice_chart_items, "月度覆冰分钟占比", value_suffix="%")}</div>
<div class="panel">{svg_line_chart(month_fog_chart_items, "月度雾或低能见度分钟占比", value_suffix="%")}</div>
<div class="panel">{svg_bar_chart(top_ice_chart_items, "覆冰占比最高的站点-年份", value_suffix="%")}</div>
<div class="panel">{svg_bar_chart(top_fog_chart_items, "雾/低能见度占比最高的站点-年份", value_suffix="%")}</div>
<div class="panel">{svg_bar_chart(low_cov_chart_items, "完整率最低的站点-年份", value_suffix="%")}</div>
<h2>峰值覆冰事件 Top 20</h2>
<div class="panel">{html_table(top_events, ["站点", "城市", "年份", "开始时间", "结束时间", "持续分钟_含合并间断", "覆冰观测分钟", "峰值覆冰厚度", "峰值时间"], 20)}</div>
<h2>字段缺失率 Top 12</h2>
<div class="panel">{html_table(sorted(missing_rows, key=lambda r: r["缺失率%"], reverse=True), ["字段", "缺失或--记录数", "缺失率%"], 12)}</div>
<h2>说明</h2>
<p class="note">降水量字段没有直接逐分钟累加；若需要日降水或年降水，应先按小时窗口聚合。完整 CSV 输出位于 <code>analysis_outputs</code>。</p>
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
                    global_agg.station = "全库"

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
                    month_base = Aggregate(station="全库", city="全库", folder_year=month[:4])
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
            file_agg.anomaly_counts["文件读取失败"] += 1
            print(f"  !! failed: {exc}", flush=True)

        closed = event_state.close()
        if closed:
            events.append(closed)
        station_year_aggs.append(file_agg)

    station_year_rows = [aggregate_to_row(agg, "站点-年份") for agg in station_year_aggs]
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

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "(林立铮2025.6.21)凝结类天气数据"
OUTPUT_ROOT = PROJECT_ROOT / "起冰事件统计"

TIME_COL = "观测时间"
STATION_COL = "站点名称"
STATION_CODE_COL = "站点编号"
STATION_ID_COL = "站点ID"
ICE_THICKNESS_COL = "覆冰厚度"
TEMP_COL = "气温"
RH_COL = "相对湿度"
VIS_COL = "能见度"
VIS1_COL = "一分钟能见度"
VIS_OBSTACLE_COL = "视程障碍"

READ_COLS = [
    TIME_COL,
    STATION_COL,
    STATION_CODE_COL,
    STATION_ID_COL,
    ICE_THICKNESS_COL,
    TEMP_COL,
    RH_COL,
    VIS_COL,
    VIS1_COL,
    VIS_OBSTACLE_COL,
]

MISSING_MARKERS = {"", "--", "nan", "NaN", "NAN", "null", "NULL", "None", "none"}
CITY_PREFIXES = ["武夷山", "龙岩", "泉州", "宁德", "南平", "三明"]


@dataclass
class EventState:
    active: bool = False
    station: str = ""
    city: str = ""
    year: str = ""
    station_code: str = ""
    station_id: str = ""
    source_file: str = ""
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    obs_minutes: int = 0
    elapsed_minutes: int = 0
    max_thickness: float = 0.0
    peak_time: pd.Timestamp | None = None
    min_temp: float | None = None
    max_temp: float | None = None
    max_rh: float | None = None
    min_vis: float | None = None
    exposure_obs_minutes: int = 0

    def start_event(
        self,
        station: str,
        city: str,
        year: str,
        station_code: str,
        station_id: str,
        source_file: str,
        ts: pd.Timestamp,
        thickness: float,
        temp: float | None,
        rh: float | None,
        vis: float | None,
        exposure: bool,
    ) -> None:
        self.active = True
        self.station = station
        self.city = city
        self.year = year
        self.station_code = station_code
        self.station_id = station_id
        self.source_file = source_file
        self.start = ts
        self.end = ts
        self.obs_minutes = 1
        self.elapsed_minutes = 1
        self.max_thickness = float(thickness)
        self.peak_time = ts
        self.min_temp = temp
        self.max_temp = temp
        self.max_rh = rh
        self.min_vis = vis
        self.exposure_obs_minutes = 1 if exposure else 0

    def add_observation(
        self,
        ts: pd.Timestamp,
        thickness: float,
        temp: float | None,
        rh: float | None,
        vis: float | None,
        exposure: bool,
    ) -> None:
        self.end = ts
        self.obs_minutes += 1
        if float(thickness) > self.max_thickness:
            self.max_thickness = float(thickness)
            self.peak_time = ts
        if temp is not None:
            self.min_temp = temp if self.min_temp is None else min(self.min_temp, temp)
            self.max_temp = temp if self.max_temp is None else max(self.max_temp, temp)
        if rh is not None:
            self.max_rh = rh if self.max_rh is None else max(self.max_rh, rh)
        if vis is not None:
            self.min_vis = vis if self.min_vis is None else min(self.min_vis, vis)
        if exposure:
            self.exposure_obs_minutes += 1

    def close(self, event_id: int, merge_gap_minutes: int) -> dict[str, Any] | None:
        if not self.active or self.start is None or self.end is None:
            return None
        elapsed = int((self.end - self.start).total_seconds() // 60) + 1
        self.elapsed_minutes = max(1, elapsed)
        cold_plausible = self.min_temp is not None and self.min_temp <= 2.0
        freezing_plausible = self.min_temp is not None and self.min_temp <= 0.0
        exposure_supported = self.exposure_obs_minutes > 0
        event = {
            "event_id": event_id,
            "station": self.station,
            "city": self.city,
            "year": self.year,
            "station_code": self.station_code,
            "station_id": self.station_id,
            "start_time": ts_text(self.start),
            "end_time": ts_text(self.end),
            "peak_time": ts_text(self.peak_time),
            "elapsed_minutes": self.elapsed_minutes,
            "ice_observation_minutes": self.obs_minutes,
            "max_thickness": round(self.max_thickness, 3),
            "min_temp": round_float(self.min_temp),
            "max_temp": round_float(self.max_temp),
            "max_rh": round_float(self.max_rh),
            "min_visibility": round_float(self.min_vis),
            "exposure_observation_minutes": self.exposure_obs_minutes,
            "cold_plausible_min_temp_le_2": int(cold_plausible),
            "freezing_plausible_min_temp_le_0": int(freezing_plausible),
            "exposure_supported": int(exposure_supported),
            "definition": f"ice_thickness>0, merge_positive_gaps<={merge_gap_minutes}min",
            "source_file": self.source_file,
        }
        self.active = False
        return event


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def numeric(series: pd.Series) -> pd.Series:
    text = clean_text(series)
    return pd.to_numeric(text.where(~text.isin(MISSING_MARKERS), None), errors="coerce")


def parse_times(series: pd.Series) -> pd.Series:
    return pd.to_datetime(clean_text(series), format="mixed", errors="coerce")


def ts_text(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def round_float(value: float | None, digits: int = 3) -> float | str:
    if value is None or pd.isna(value):
        return ""
    return round(float(value), digits)


def infer_city(station: str) -> str:
    for prefix in CITY_PREFIXES:
        if station.startswith(prefix):
            return prefix
    return "未识别"


def infer_year(path: Path) -> str:
    match = re.search(r"(20\d{2})年", str(path.parent))
    return match.group(1) if match else ""


def first_non_missing(series: pd.Series) -> str:
    text = clean_text(series)
    valid = text[~text.isin(MISSING_MARKERS)]
    return valid.iloc[0] if not valid.empty else ""


def iter_csv_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            reader = pd.read_csv(
                path,
                usecols=lambda col: col in READ_COLS,
                dtype=str,
                chunksize=chunksize,
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return round(s[lo] * (1 - frac) + s[hi] * frac, 3)


def summarize_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    station_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    station_year_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    year_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for event in events:
        station_groups[event["station"]].append(event)
        station_year_groups[(event["station"], event["year"])].append(event)
        year_groups[event["year"]].append(event)

    station_rows = []
    for station, rows in sorted(station_groups.items()):
        cold = [r for r in rows if int(r["cold_plausible_min_temp_le_2"]) == 1]
        freezing = [r for r in rows if int(r["freezing_plausible_min_temp_le_0"]) == 1]
        peak_values = [float(r["max_thickness"]) for r in rows]
        cold_peak_values = [float(r["max_thickness"]) for r in cold]
        durations = [float(r["elapsed_minutes"]) for r in rows]
        cold_durations = [float(r["elapsed_minutes"]) for r in cold]
        active_years = sorted({r["year"] for r in rows if r["year"]})
        station_rows.append(
            {
                "station": station,
                "city": rows[0]["city"],
                "all_events": len(rows),
                "cold_plausible_events_min_temp_le_2": len(cold),
                "freezing_plausible_events_min_temp_le_0": len(freezing),
                "active_years": ",".join(active_years),
                "n_active_years": len(active_years),
                "ice_observation_minutes": sum(int(r["ice_observation_minutes"]) for r in rows),
                "cold_plausible_ice_observation_minutes": sum(int(r["ice_observation_minutes"]) for r in cold),
                "median_event_elapsed_minutes": percentile(durations, 0.5),
                "p90_event_elapsed_minutes": percentile(durations, 0.9),
                "median_cold_event_elapsed_minutes": percentile(cold_durations, 0.5),
                "max_event_thickness": max(peak_values) if peak_values else "",
                "max_cold_plausible_thickness": max(cold_peak_values) if cold_peak_values else "",
                "enough_ge_5_cold_events": int(len(cold) >= 5),
                "enough_ge_10_cold_events": int(len(cold) >= 10),
                "enough_ge_20_cold_events": int(len(cold) >= 20),
            }
        )

    station_year_rows = []
    for (station, year), rows in sorted(station_year_groups.items()):
        cold = [r for r in rows if int(r["cold_plausible_min_temp_le_2"]) == 1]
        station_year_rows.append(
            {
                "station": station,
                "city": rows[0]["city"],
                "year": year,
                "all_events": len(rows),
                "cold_plausible_events_min_temp_le_2": len(cold),
                "ice_observation_minutes": sum(int(r["ice_observation_minutes"]) for r in rows),
                "cold_plausible_ice_observation_minutes": sum(int(r["ice_observation_minutes"]) for r in cold),
                "max_thickness": max(float(r["max_thickness"]) for r in rows),
                "max_cold_plausible_thickness": max([float(r["max_thickness"]) for r in cold], default=""),
            }
        )

    year_rows = []
    for year, rows in sorted(year_groups.items()):
        cold = [r for r in rows if int(r["cold_plausible_min_temp_le_2"]) == 1]
        year_rows.append(
            {
                "year": year,
                "all_events": len(rows),
                "cold_plausible_events_min_temp_le_2": len(cold),
                "stations_with_all_events": len({r["station"] for r in rows}),
                "stations_with_cold_plausible_events": len({r["station"] for r in cold}),
                "ice_observation_minutes": sum(int(r["ice_observation_minutes"]) for r in rows),
                "cold_plausible_ice_observation_minutes": sum(int(r["ice_observation_minutes"]) for r in cold),
            }
        )

    station_rows.sort(key=lambda r: (r["cold_plausible_events_min_temp_le_2"], r["all_events"]), reverse=True)
    station_year_rows.sort(key=lambda r: (r["year"], r["cold_plausible_events_min_temp_le_2"], r["all_events"]), reverse=True)
    return station_rows, station_year_rows, year_rows


def build_report(events: list[dict[str, Any]], station_rows: list[dict[str, Any]], station_year_rows: list[dict[str, Any]], year_rows: list[dict[str, Any]], merge_gap_minutes: int) -> str:
    cold_events = [e for e in events if int(e["cold_plausible_min_temp_le_2"]) == 1]
    freezing_events = [e for e in events if int(e["freezing_plausible_min_temp_le_0"]) == 1]
    station_count = len({e["station"] for e in events})
    cold_station_count = len({e["station"] for e in cold_events})
    ge5 = sum(1 for row in station_rows if int(row["enough_ge_5_cold_events"]) == 1)
    ge10 = sum(1 for row in station_rows if int(row["enough_ge_10_cold_events"]) == 1)
    ge20 = sum(1 for row in station_rows if int(row["enough_ge_20_cold_events"]) == 1)
    total_ice_minutes = sum(int(e["ice_observation_minutes"]) for e in events)
    cold_ice_minutes = sum(int(e["ice_observation_minutes"]) for e in cold_events)

    top_cols = [
        "station",
        "city",
        "all_events",
        "cold_plausible_events_min_temp_le_2",
        "freezing_plausible_events_min_temp_le_0",
        "active_years",
        "cold_plausible_ice_observation_minutes",
        "max_cold_plausible_thickness",
        "enough_ge_10_cold_events",
    ]

    return f"""# 同站点复发起冰事件数据量判断

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 事件定义

- 独立起冰事件：`覆冰厚度 > 0` 的连续记录；相邻正覆冰记录间隔不超过 {merge_gap_minutes} 分钟时合并为同一事件。
- 冷条件可信事件：独立事件中，事件期间最低气温 `<= 2℃`。该口径用于剔除部分暖季或高温下疑似传感器异常的覆冰厚度正值。
- 冻结可信事件：独立事件中，事件期间最低气温 `<= 0℃`，作为更严格口径。

## 总体结论

- 全部 `覆冰厚度 > 0` 独立事件：{len(events):,} 次。
- 冷条件可信独立起冰事件：{len(cold_events):,} 次。
- 冻结可信独立起冰事件：{len(freezing_events):,} 次。
- 有覆冰事件的站点：{station_count} 个；有冷条件可信事件的站点：{cold_station_count} 个。
- 全部覆冰正记录分钟：{total_ice_minutes:,} 分钟；冷条件可信覆冰正记录分钟：{cold_ice_minutes:,} 分钟。
- 至少 5 次冷条件可信起冰事件的站点：{ge5} 个。
- 至少 10 次冷条件可信起冰事件的站点：{ge10} 个。
- 至少 20 次冷条件可信起冰事件的站点：{ge20} 个。

## 年度分布

{markdown_table(year_rows, ["year", "all_events", "cold_plausible_events_min_temp_le_2", "stations_with_cold_plausible_events", "cold_plausible_ice_observation_minutes"], 10)}

## 冷条件可信事件最多的站点

{markdown_table(station_rows, top_cols, 32)}

## 判断

若论文主线要求“同站点存在复发起冰事件”，建议使用 `冷条件可信事件` 作为主口径。判断标准可设为：

- `>=5` 次：可用于站点级案例和复发性论证。
- `>=10` 次：可用于站点屏障或站点异质性建模。
- `>=20` 次：可用于较稳健的站点级统计评估。

完整事件表见 `icing_events.csv`；站点汇总见 `station_event_summary.csv`；站点-年份汇总见 `station_year_event_summary.csv`。
"""


def markdown_table(rows: list[dict[str, Any]], cols: list[str], max_rows: int) -> str:
    if not rows:
        return "无。"
    rows = rows[:max_rows]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def count_events(max_files: int | None, chunksize: int, merge_gap_minutes: int) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(DATA_ROOT.glob("**/*.csv"))
    if max_files is not None:
        csv_paths = csv_paths[:max_files]

    events: list[dict[str, Any]] = []
    event_counter = 0
    file_rows: list[dict[str, Any]] = []

    for idx, path in enumerate(csv_paths, start=1):
        rel = str(path.relative_to(PROJECT_ROOT))
        year = infer_year(path)
        state = EventState()
        file_row_count = 0
        file_ice_rows = 0
        station = ""
        station_code = ""
        station_id = ""
        city = ""
        print(f"[{idx}/{len(csv_paths)}] {rel}", flush=True)

        for chunk in iter_csv_chunks(path, chunksize):
            if VIS_COL not in chunk.columns and VIS1_COL in chunk.columns:
                chunk[VIS_COL] = chunk[VIS1_COL]
            for col in READ_COLS:
                if col not in chunk.columns:
                    chunk[col] = ""

            file_row_count += int(chunk.shape[0])
            if not station:
                station = first_non_missing(chunk[STATION_COL]) or path.stem.split("20")[0]
                station_code = first_non_missing(chunk[STATION_CODE_COL])
                station_id = first_non_missing(chunk[STATION_ID_COL])
                city = infer_city(station)

            times = parse_times(chunk[TIME_COL])
            ice = numeric(chunk[ICE_THICKNESS_COL])
            temp = numeric(chunk[TEMP_COL])
            rh = numeric(chunk[RH_COL])
            vis = numeric(chunk[VIS_COL])
            obstacle = clean_text(chunk[VIS_OBSTACLE_COL])
            exposure = ((temp <= 2) & (rh >= 95)) | (vis < 1000) | obstacle.str.contains("雾", regex=False, na=False)
            positive = (ice > 0) & times.notna()
            if not positive.any():
                continue

            positive_frame = pd.DataFrame(
                {
                    "time": times[positive],
                    "ice": ice[positive],
                    "temp": temp[positive],
                    "rh": rh[positive],
                    "vis": vis[positive],
                    "exposure": exposure[positive],
                }
            ).sort_values("time")
            file_ice_rows += int(positive_frame.shape[0])

            for row in positive_frame.itertuples(index=False):
                ts = row.time
                thick = float(row.ice)
                temp_val = None if pd.isna(row.temp) else float(row.temp)
                rh_val = None if pd.isna(row.rh) else float(row.rh)
                vis_val = None if pd.isna(row.vis) else float(row.vis)
                exposure_val = bool(row.exposure)
                if not state.active:
                    state.start_event(station, city, year, station_code, station_id, rel, ts, thick, temp_val, rh_val, vis_val, exposure_val)
                    continue
                assert state.end is not None
                gap_minutes = (ts - state.end).total_seconds() / 60
                if gap_minutes <= merge_gap_minutes + 1:
                    state.add_observation(ts, thick, temp_val, rh_val, vis_val, exposure_val)
                else:
                    event_counter += 1
                    closed = state.close(event_counter, merge_gap_minutes)
                    if closed:
                        events.append(closed)
                    state.start_event(station, city, year, station_code, station_id, rel, ts, thick, temp_val, rh_val, vis_val, exposure_val)

        if state.active:
            event_counter += 1
            closed = state.close(event_counter, merge_gap_minutes)
            if closed:
                events.append(closed)

        file_rows.append(
            {
                "source_file": rel,
                "station": station,
                "city": city,
                "year": year,
                "rows": file_row_count,
                "ice_positive_rows": file_ice_rows,
            }
        )

    station_rows, station_year_rows, year_rows = summarize_events(events)
    write_csv(OUTPUT_ROOT / "icing_events.csv", events)
    write_csv(OUTPUT_ROOT / "station_event_summary.csv", station_rows)
    write_csv(OUTPUT_ROOT / "station_year_event_summary.csv", station_year_rows)
    write_csv(OUTPUT_ROOT / "year_event_summary.csv", year_rows)
    write_csv(OUTPUT_ROOT / "file_scan_summary.csv", file_rows)
    report = build_report(events, station_rows, station_year_rows, year_rows, merge_gap_minutes)
    (OUTPUT_ROOT / "event_count_report.md").write_text(report, encoding="utf-8")
    print(f"Done. Outputs: {OUTPUT_ROOT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count recurrent independent icing-onset events per station.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files for a quick subset run.")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--merge-gap-minutes", type=int, default=10)
    args = parser.parse_args()
    count_events(args.max_files, args.chunksize, args.merge_gap_minutes)


if __name__ == "__main__":
    main()

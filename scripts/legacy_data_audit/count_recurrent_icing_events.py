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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "meteorology_raw"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "legacy_exploration" / "icing_event_statistics"

TIME_COL = "\u89c2\u6d4b\u65f6\u95f4"
STATION_COL = "\u7ad9\u70b9\u540d\u79f0"
STATION_CODE_COL = "\u7ad9\u70b9\u7f16\u53f7"
STATION_ID_COL = "\u7ad9\u70b9ID"
ICE_THICKNESS_COL = "\u8986\u51b0\u539a\u5ea6"
TEMP_COL = "\u6c14\u6e29"
RH_COL = "\u76f8\u5bf9\u6e7f\u5ea6"
VIS_COL = "\u80fd\u89c1\u5ea6"
VIS1_COL = "\u4e00\u5206\u949f\u80fd\u89c1\u5ea6"
VIS_OBSTACLE_COL = "\u89c6\u7a0b\u969c\u788d"

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
CITY_PREFIXES = [
    "\u6b66\u5937\u5c71",
    "\u9f99\u5ca9",
    "\u6cc9\u5dde",
    "\u5b81\u5fb7",
    "\u5357\u5e73",
    "\u4e09\u660e",
]
RAW_FOG_TOKEN = "\u96fe"


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
    return "Unknown"


def infer_year(path: Path) -> str:
    match = re.search(r"(20\d{2})", str(path.parent))
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

    return f"""# Recurrent Icing Event Volume by Station

Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Event definitions

- Independent icing event: consecutive records with ice thickness greater than zero; positive records separated by no more than {merge_gap_minutes} minutes are merged into one event.
- Cold-plausible event: an independent event with a minimum temperature no greater than 2 °C. This rule removes some warm-season or high-temperature positives likely caused by sensor anomalies.
- Freezing-plausible event: an independent event with a minimum temperature no greater than 0 °C, used as a stricter definition.

## Overall results

- Independent positive-thickness events: {len(events):,}.
- Cold-plausible independent icing events: {len(cold_events):,}.
- Freezing-plausible independent icing events: {len(freezing_events):,}.
- Stations with icing events: {station_count}; stations with cold-plausible events: {cold_station_count}.
- Positive-icing observation minutes: {total_ice_minutes:,}; cold-plausible positive-icing minutes: {cold_ice_minutes:,}.
- Stations with at least 5 cold-plausible events: {ge5}.
- Stations with at least 10 cold-plausible events: {ge10}.
- Stations with at least 20 cold-plausible events: {ge20}.

## Annual distribution

{markdown_table(year_rows, ["year", "all_events", "cold_plausible_events_min_temp_le_2", "stations_with_cold_plausible_events", "cold_plausible_ice_observation_minutes"], 10)}

## Stations with the most cold-plausible events

{markdown_table(station_rows, top_cols, 32)}

## Interpretation

Use cold-plausible events as the primary definition when the manuscript argument requires recurrent icing at the same station. Suggested evidence thresholds are:

- `>=5`: station-level case study and recurrence evidence.
- `>=10`: station-barrier or station-heterogeneity modelling.
- `>=20`: more robust station-level statistical evaluation.

See `icing_events.csv` for the full event table, `station_event_summary.csv` for station summaries, and `station_year_event_summary.csv` for station-year summaries.
"""


def markdown_table(rows: list[dict[str, Any]], cols: list[str], max_rows: int) -> str:
    if not rows:
        return "None."
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
            exposure = ((temp <= 2) & (rh >= 95)) | (vis < 1000) | obstacle.str.contains(RAW_FOG_TOKEN, regex=False, na=False)
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

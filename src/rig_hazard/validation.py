from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_project_path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_timeline(path: Path, usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    if path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, nrows=nrows, low_memory=False)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=usecols)
        return frame.head(nrows) if nrows is not None else frame
    raise ValueError(f"Unsupported timeline format: {path}")


def expected_year_rows(year: int, step_minutes: int) -> int:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    return int((end - start).total_seconds() // (step_minutes * 60))


def expected_split(issue_year: pd.Series, split: dict[str, list[int]]) -> pd.Series:
    mapping: dict[int, str] = {}
    for year in split.get("train_years", []):
        mapping[int(year)] = "train"
    for year in split.get("validation_years", []):
        mapping[int(year)] = "validation"
    for year in split.get("test_years", []):
        mapping[int(year)] = "test"
    return issue_year.map(mapping).fillna("excluded")


def add_failure(failures: list[str], condition: bool, message: str) -> None:
    if condition:
        failures.append(message)


def validate_preprocessed(output_root_value: str | Path, strict: bool = True) -> Path:
    output_root = resolve_project_path(output_root_value)
    failures: list[str] = []
    warnings: list[str] = []
    required_artifacts = [
        "resolved_config.json",
        "feature_contract.json",
        "events_recurrent.csv",
        "station_catalog.csv",
        "source_files.csv",
        "station_preprocessing_summary.csv",
        "split_summary.csv",
        "normalization_development.json",
        "normalization_final.json",
    ]
    for name in required_artifacts:
        add_failure(failures, not (output_root / name).is_file(), f"Missing artifact: {name}")
    if failures:
        return write_validation_outputs(output_root, failures, warnings, {}, strict)

    config = read_json(output_root / "resolved_config.json")
    contract = read_json(output_root / "feature_contract.json")
    step = int(config["time_step_minutes"])
    horizons = [int(value) for value in config["risk_set"]["forecast_horizons_hours"]]
    source_files = pd.read_csv(output_root / "source_files.csv", encoding="utf-8-sig", keep_default_na=False)
    station_catalog = pd.read_csv(output_root / "station_catalog.csv", encoding="utf-8-sig", keep_default_na=False)
    events = pd.read_csv(output_root / "events_recurrent.csv", encoding="utf-8-sig", keep_default_na=False)
    station_summary = pd.read_csv(output_root / "station_preprocessing_summary.csv", encoding="utf-8-sig", keep_default_na=False)

    timeline_paths = sorted(
        path
        for path in (output_root / "timelines").rglob("*")
        if path.is_file() and (path.name.endswith(".csv.gz") or path.suffix in {".csv", ".parquet"})
    )
    expected_file_count = int((source_files["empty"].astype(str).str.lower().isin({"false", "0"})).sum())
    add_failure(failures, len(timeline_paths) != expected_file_count, f"Timeline file count {len(timeline_paths)} != non-empty source count {expected_file_count}")

    allowed_features = set(contract["continuous_dynamic_features"]) | set(contract["binary_dynamic_features"])
    forbidden = set(contract["forbidden_as_model_inputs"])
    add_failure(failures, bool(allowed_features & forbidden), f"Allowed and forbidden feature lists overlap: {sorted(allowed_features & forbidden)}")

    audit_columns = [
        "station_code",
        "year",
        "issue_year",
        "bin_start",
        "issue_time",
        "ice_positive_minutes",
        "physical_at_risk",
        "history_valid",
        "future_step_label_valid",
        "risk_set",
        "hazard_label",
        "exposure_recent_1h",
        "split_development",
        "split_final",
    ]
    for horizon in horizons:
        audit_columns.extend([f"onset_within_{horizon}h", f"future_label_valid_{horizon}h", f"hard_negative_{horizon}h"])

    totals: dict[str, int] = defaultdict(int)
    files_by_year: dict[int, int] = defaultdict(int)
    for index, path in enumerate(timeline_paths, start=1):
        header = read_timeline(path, nrows=0)
        missing_columns = (allowed_features | set(audit_columns)) - set(header.columns)
        add_failure(failures, bool(missing_columns), f"{path.name}: missing columns {sorted(missing_columns)}")
        if missing_columns:
            continue
        frame = read_timeline(path, usecols=audit_columns)
        year = int(path.parent.name)
        files_by_year[year] += 1
        expected_rows = expected_year_rows(year, step)
        add_failure(failures, frame.shape[0] != expected_rows, f"{path.name}: {frame.shape[0]} rows != expected {expected_rows}")
        bin_start = pd.to_datetime(frame["bin_start"], errors="coerce")
        issue_time = pd.to_datetime(frame["issue_time"], errors="coerce")
        add_failure(failures, bool(bin_start.isna().any() or issue_time.isna().any()), f"{path.name}: invalid timestamps")
        add_failure(failures, bool(issue_time.duplicated().any()), f"{path.name}: duplicated issue_time")
        add_failure(failures, not issue_time.is_monotonic_increasing, f"{path.name}: issue_time is not monotonic")
        deltas = (issue_time - bin_start).dt.total_seconds().div(60)
        add_failure(failures, bool(deltas.ne(step).any()), f"{path.name}: issue_time/bin_start offset is not {step} minutes")
        add_failure(failures, bool(bin_start.dt.year.ne(year).any()), f"{path.name}: bin_start outside directory year")

        risk_mask = frame["risk_set"].eq(1)
        positive_mask = frame["hazard_label"].eq(1)
        add_failure(failures, bool((positive_mask & ~risk_mask).any()), f"{path.name}: hazard positive outside risk set")
        add_failure(failures, bool((risk_mask & frame["physical_at_risk"].ne(1)).any()), f"{path.name}: risk row is not physically at risk")
        add_failure(failures, bool((risk_mask & frame["history_valid"].ne(1)).any()), f"{path.name}: risk row has invalid history")
        add_failure(failures, bool((risk_mask & frame["future_step_label_valid"].ne(1)).any()), f"{path.name}: risk row has invalid next-step label coverage")
        add_failure(failures, bool((risk_mask & frame["ice_positive_minutes"].gt(0)).any()), f"{path.name}: risk row contains current-bin icing")

        issue_year = pd.to_numeric(frame["issue_year"], errors="coerce").astype("Int64")
        expected_development = expected_split(issue_year, config["splits"]["development"])
        expected_final = expected_split(issue_year, config["splits"]["final"])
        add_failure(failures, bool(frame["split_development"].astype(str).ne(expected_development).any()), f"{path.name}: development split mismatch")
        add_failure(failures, bool(frame["split_final"].astype(str).ne(expected_final).any()), f"{path.name}: final split mismatch")

        totals["timeline_rows"] += int(frame.shape[0])
        totals["risk_rows"] += int(risk_mask.sum())
        totals["hazard_positive_rows"] += int((risk_mask & positive_mask).sum())
        for horizon in horizons:
            hard = frame[f"hard_negative_{horizon}h"].eq(1)
            invalid_hard = hard & (
                ~risk_mask
                | frame["exposure_recent_1h"].ne(1)
                | frame[f"onset_within_{horizon}h"].ne(0)
                | frame[f"future_label_valid_{horizon}h"].ne(1)
            )
            add_failure(failures, bool(invalid_hard.any()), f"{path.name}: invalid hard_negative_{horizon}h rows")
            totals[f"hard_negative_{horizon}h"] += int(hard.sum())

        if index % 10 == 0 or index == len(timeline_paths):
            print(f"Validated {index}/{len(timeline_paths)} timeline files", flush=True)

    summary_metrics = ["timeline_rows", "risk_rows", "hazard_positive_rows", *[f"hard_negative_{horizon}h" for horizon in horizons]]
    for column in summary_metrics:
        expected_total = int(pd.to_numeric(station_summary[column], errors="coerce").fillna(0).sum())
        add_failure(failures, totals[column] != expected_total, f"Timeline total {column}={totals[column]} != station summary {expected_total}")

    add_failure(failures, bool(events["event_id"].duplicated().any()), "Duplicate event_id values")
    numeric_event_columns = ["cold_candidate_event", "cooldown_eligible_onset", "valid_target_event", "eligible_hazard_label"]
    for column in numeric_event_columns:
        events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0).astype(int)
    add_failure(failures, bool((events["valid_target_event"].eq(1) & events["cold_candidate_event"].ne(1)).any()), "Target event is not a cold candidate")
    add_failure(failures, bool((events["valid_target_event"].eq(1) & events["cooldown_eligible_onset"].ne(1)).any()), "Target event is inside cooldown")
    add_failure(failures, bool((events["eligible_hazard_label"].eq(1) & events["valid_target_event"].ne(1)).any()), "Eligible event is not a target event")
    eligible_events = int(events["eligible_hazard_label"].sum())
    add_failure(failures, eligible_events != totals["hazard_positive_rows"], f"Eligible event count {eligible_events} != timeline hazard positives {totals['hazard_positive_rows']}")

    add_failure(failures, bool(station_catalog["station_code"].duplicated().any()), "Duplicate station codes")
    add_failure(failures, bool(pd.to_numeric(station_catalog["station_index"], errors="coerce").duplicated().any()), "Duplicate station indexes")
    coordinate_missing = pd.to_numeric(station_catalog["longitude"], errors="coerce").isna() | pd.to_numeric(station_catalog["latitude"], errors="coerce").isna()
    if coordinate_missing.any():
        warnings.append(f"{int(coordinate_missing.sum())} stations are missing coordinates")
    elevation_missing = pd.to_numeric(station_catalog["elevation_m"], errors="coerce").isna()
    if elevation_missing.any():
        missing_codes = ", ".join(station_catalog.loc[elevation_missing, "station_code"].astype(str))
        warnings.append(f"{int(elevation_missing.sum())} stations are missing elevation: {missing_codes}")
    if "metadata_name_differs" in station_catalog:
        name_difference = pd.to_numeric(station_catalog["metadata_name_differs"], errors="coerce").fillna(0).eq(1)
        if name_difference.any():
            difference_text = ", ".join(
                f"{row.station_code}({row.station_name}/{row.metadata_station_name})"
                for row in station_catalog.loc[name_difference, ["station_code", "station_name", "metadata_station_name"]].itertuples(index=False)
            )
            warnings.append(
                "Station names differ from the coordinate metadata; records were joined "
                f"by station_code: {difference_text}"
            )

    candidate_path = output_root / "spatial_candidate_edges.csv"
    if candidate_path.is_file() and candidate_path.stat().st_size > 3:
        candidates = pd.read_csv(candidate_path, encoding="utf-8-sig")
        add_failure(failures, bool(candidates["source_station_code"].eq(candidates["target_station_code"]).any()), "Spatial candidates contain self edges")
        add_failure(failures, bool(pd.to_numeric(candidates["distance_km"], errors="coerce").le(0).any()), "Spatial candidates contain non-positive distances")

    for name in ("normalization_development.json", "normalization_final.json"):
        stats = read_json(output_root / name)
        missing_stats = set(contract["continuous_dynamic_features"]) - set(stats)
        add_failure(failures, bool(missing_stats), f"{name}: missing statistics for {sorted(missing_stats)}")
        for feature in contract["continuous_dynamic_features"]:
            item = stats.get(feature, {})
            count = int(item.get("count") or 0)
            mean = item.get("mean")
            std = item.get("std")
            add_failure(failures, count <= 0, f"{name}: no training values for {feature}")
            add_failure(failures, mean is None or not math.isfinite(float(mean)), f"{name}: invalid mean for {feature}")
            add_failure(failures, std is None or not math.isfinite(float(std)), f"{name}: invalid std for {feature}")

    metrics = {
        **dict(totals),
        "timeline_files": len(timeline_paths),
        "files_by_year": dict(sorted(files_by_year.items())),
        "stations": int(station_catalog.shape[0]),
        "all_events": int(events.shape[0]),
        "cold_candidate_events": int(events["cold_candidate_event"].sum()),
        "valid_target_events": int(events["valid_target_event"].sum()),
        "eligible_events": eligible_events,
    }
    return write_validation_outputs(output_root, failures, warnings, metrics, strict)


def write_validation_outputs(output_root: Path, failures: list[str], warnings: list[str], metrics: dict[str, Any], strict: bool) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    status = "PASS" if not failures else "FAIL"
    report_path = output_root / "validation_report.md"
    metric_lines = "\n".join(f"- `{key}`: {value}" for key, value in metrics.items()) or "- None."
    failure_lines = "\n".join(f"- {value}" for value in failures) or "- None."
    warning_lines = "\n".join(f"- {value}" for value in warnings) or "- None."
    report = f"""# RIG-Hazard Preprocessing Validation Report

Validated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: **{status}**

## Core counts

{metric_lines}

## Failures

{failure_lines}

## Warnings

{warning_lines}
"""
    report_path.write_text(report, encoding="utf-8")
    (output_root / "validation_summary.json").write_text(
        json.dumps({"status": status, "failures": failures, "warnings": warnings, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failures and strict:
        raise RuntimeError(f"Preprocessed data validation failed with {len(failures)} issue(s). See {report_path}")
    print(f"Validation {status}: {report_path}", flush=True)
    return report_path

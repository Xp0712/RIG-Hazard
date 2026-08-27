from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_project_path


INDEX_FIELDS = ("file_id", "row_index", "sample_weight", "stratum")


def load_index(cache_root: Path, name: str) -> dict[str, np.ndarray]:
    path = cache_root / f"index_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        return {field: values[field].copy() for field in INDEX_FIELDS}


def save_index(cache_root: Path, name: str, values: dict[str, np.ndarray]) -> None:
    missing = sorted(set(INDEX_FIELDS) - set(values))
    if missing:
        raise ValueError(f"Index is missing fields: {missing}")
    sizes = {np.asarray(values[field]).size for field in INDEX_FIELDS}
    if len(sizes) != 1:
        raise ValueError("Index fields must have the same length")
    np.savez_compressed(cache_root / f"index_{name}.npz", **{field: values[field] for field in INDEX_FIELDS})


def subset_index(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    selected = np.asarray(mask, dtype=bool)
    return {field: np.asarray(values[field])[selected] for field in INDEX_FIELDS}


def index_issue_metadata(cache_root: Path, index: dict[str, np.ndarray]) -> pd.DataFrame:
    manifest = json.loads((cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
    files = {int(row["file_id"]): row for row in manifest["files"]}
    count = int(index["row_index"].size)
    issue_time_ns = np.empty(count, dtype=np.int64)
    hazard_label = np.empty(count, dtype=np.int8)
    station_code = np.empty(count, dtype=object)
    year = np.empty(count, dtype=np.int16)
    for file_id in np.unique(index["file_id"]):
        positions = np.flatnonzero(index["file_id"] == file_id)
        metadata = files[int(file_id)]
        rows = index["row_index"][positions].astype(np.int64)
        with np.load(cache_root / metadata["target_path"], allow_pickle=False) as targets:
            issue_time_ns[positions] = targets["issue_time_ns"][rows]
            hazard_label[positions] = targets["hazard_label"][rows].astype(np.int8)
        station_code[positions] = str(metadata["station_code"])
        year[positions] = int(metadata["year"])
    return pd.DataFrame(
        {
            "issue_time_ns": issue_time_ns,
            "hazard_label": hazard_label,
            "station_code": station_code,
            "year": year,
        }
    )


def calendar_block_start(issue_time_ns: np.ndarray, block_days: int = 7) -> np.ndarray:
    if block_days < 1:
        raise ValueError("block_days must be positive")
    times = pd.to_datetime(np.asarray(issue_time_ns, dtype=np.int64), unit="ns")
    origin = pd.Timestamp(f"{int(times.year.min())}-01-01")
    elapsed_days = ((times.normalize() - origin) / pd.Timedelta(days=1)).astype(np.int64)
    block = np.floor_divide(elapsed_days, int(block_days))
    return (origin.value + block * int(block_days) * 86_400_000_000_000).astype(np.int64)


def assign_event_balanced_blocks(
    metadata: pd.DataFrame,
    number_folds: int = 5,
    block_days: int = 7,
) -> pd.DataFrame:
    if number_folds < 2:
        raise ValueError("number_folds must be at least two")
    frame = metadata.copy()
    frame["block_start_ns"] = calendar_block_start(frame["issue_time_ns"].to_numpy(), block_days)
    blocks = (
        frame.groupby("block_start_ns", as_index=False)
        .agg(risk_rows=("issue_time_ns", "size"), event_rows=("hazard_label", "sum"))
        .sort_values("block_start_ns")
    )
    if blocks.shape[0] < number_folds:
        raise ValueError("There are fewer calendar blocks than requested folds")
    fold_events = np.zeros(number_folds, dtype=np.int64)
    fold_rows = np.zeros(number_folds, dtype=np.int64)
    fold_blocks = np.zeros(number_folds, dtype=np.int64)
    assignments: dict[int, int] = {}
    event_blocks = blocks.loc[blocks["event_rows"].gt(0)].sort_values(
        ["event_rows", "risk_rows", "block_start_ns"], ascending=[False, False, True]
    )
    zero_event_blocks = blocks.loc[blocks["event_rows"].eq(0)].sort_values(
        ["risk_rows", "block_start_ns"], ascending=[False, True]
    )
    # First distribute event-bearing weeks by event burden. Then distribute quiet
    # weeks by exposure. A single lexicographic event objective sends every quiet
    # week to the fold with the fewest events and can create a validation fold
    # tens of times larger than the others.
    for _, block in event_blocks.iterrows():
        fold = min(
            range(number_folds),
            key=lambda value: (fold_events[value], fold_rows[value], fold_blocks[value], value),
        )
        block_start = int(block["block_start_ns"])
        assignments[block_start] = int(fold)
        fold_events[fold] += int(block["event_rows"])
        fold_rows[fold] += int(block["risk_rows"])
        fold_blocks[fold] += 1
    for _, block in zero_event_blocks.iterrows():
        fold = min(
            range(number_folds),
            key=lambda value: (fold_rows[value], fold_blocks[value], fold_events[value], value),
        )
        block_start = int(block["block_start_ns"])
        assignments[block_start] = int(fold)
        fold_rows[fold] += int(block["risk_rows"])
        fold_blocks[fold] += 1
    blocks["fold"] = blocks["block_start_ns"].map(assignments).astype(int)
    blocks["block_start"] = pd.to_datetime(blocks["block_start_ns"], unit="ns")
    blocks["block_end"] = blocks["block_start"] + pd.Timedelta(days=block_days)
    return blocks.sort_values("block_start").reset_index(drop=True)


def expanded_interval_mask(
    issue_time_ns: np.ndarray,
    block_starts_ns: np.ndarray,
    block_days: int,
    purge_hours: float,
) -> np.ndarray:
    times = np.asarray(issue_time_ns, dtype=np.int64)
    mask = np.zeros(times.size, dtype=bool)
    purge = int(float(purge_hours) * 3_600_000_000_000)
    duration = int(block_days) * 86_400_000_000_000
    for start in np.asarray(block_starts_ns, dtype=np.int64):
        mask |= (times >= int(start) - purge) & (times < int(start) + duration + purge)
    return mask


def map_block_folds(
    block_values_ns: np.ndarray,
    assigned_block_starts_ns: np.ndarray,
    assigned_folds: np.ndarray,
) -> np.ndarray:
    """Map int64 nanosecond block starts to folds without platform-sized integer casts."""

    values = np.asarray(block_values_ns, dtype=np.int64)
    keys = np.asarray(assigned_block_starts_ns, dtype=np.int64)
    folds = np.asarray(assigned_folds, dtype=np.int16)
    if keys.size != folds.size:
        raise ValueError("Assigned block starts and folds must have the same length")
    if values.size == 0:
        return np.empty(0, dtype=np.int16)
    if keys.size == 0:
        raise ValueError("Cannot map block folds without assigned calendar blocks")
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_folds = folds[order]
    if np.any(np.diff(sorted_keys) == 0):
        raise ValueError("Calendar block assignments contain duplicate block starts")
    positions = np.searchsorted(sorted_keys, values)
    safe_positions = np.minimum(positions, sorted_keys.size - 1)
    matched = (positions < sorted_keys.size) & (sorted_keys[safe_positions] == values)
    if not matched.all():
        missing = np.unique(values[~matched])[:5].tolist()
        raise ValueError(f"Calendar blocks are missing fold assignments: {missing}")
    return sorted_folds[positions]


def _copy_index_alias(cache_root: Path, source: str, destination: str) -> None:
    values = load_index(cache_root, source)
    save_index(cache_root, destination, values)


def build_temporal_protocol_indices(
    cache_root: str | Path,
    output_root: str | Path = "results/recurrence_analysis/temporal_protocol",
    number_folds: int = 5,
    block_days: int = 7,
    purge_hours: float = 30.0,
    overwrite: bool = False,
) -> Path:
    cache = resolve_project_path(cache_root)
    destination = resolve_project_path(output_root)
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    sampled_train = load_index(cache, "train")
    full_train = load_index(cache, "train_full")
    sampled_metadata = index_issue_metadata(cache, sampled_train)
    full_metadata = index_issue_metadata(cache, full_train)
    if set(full_metadata["year"].unique()) != {2022}:
        raise ValueError("The protocol training index must contain only 2022")
    blocks = assign_event_balanced_blocks(full_metadata, number_folds=number_folds, block_days=block_days)
    full_blocks = calendar_block_start(full_metadata["issue_time_ns"].to_numpy(), block_days)
    full_block_folds = map_block_folds(
        full_blocks,
        blocks["block_start_ns"].to_numpy(dtype=np.int64),
        blocks["fold"].to_numpy(dtype=np.int16),
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(number_folds):
        validation_starts = blocks.loc[blocks["fold"].eq(fold), "block_start_ns"].to_numpy(dtype=np.int64)
        validation_mask = full_block_folds == fold
        purge_mask = expanded_interval_mask(
            sampled_metadata["issue_time_ns"].to_numpy(),
            validation_starts,
            block_days=block_days,
            purge_hours=purge_hours,
        )
        training_mask = ~purge_mask
        training_index = subset_index(sampled_train, training_mask)
        validation_index = subset_index(full_train, validation_mask)
        save_index(cache, f"cv_fold{fold}_train", training_index)
        save_index(cache, f"cv_fold{fold}_validation", validation_index)
        fold_rows.append(
            {
                "fold": fold,
                "validation_blocks": int(validation_starts.size),
                "validation_rows": int(validation_index["row_index"].size),
                "validation_event_rows": int(full_metadata.loc[validation_mask, "hazard_label"].sum()),
                "training_rows_after_purge": int(training_index["row_index"].size),
                "purged_sampled_training_rows": int(purge_mask.sum()),
            }
        )

    _copy_index_alias(cache, "validation", "cross_year_2023")
    _copy_index_alias(cache, "test", "final_time_2024")
    _copy_index_alias(cache, "train", "selection_2022")
    _copy_index_alias(cache, "train_full", "selection_2022_full")
    blocks.to_csv(destination / "calendar_block_assignments.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(destination / "fold_summary.csv", index=False)
    protocol = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_root": str(cache),
        "number_folds": int(number_folds),
        "block_days": int(block_days),
        "purge_hours": float(purge_hours),
        "purge_rationale": "24-hour input history plus 6-hour forecast horizon",
        "selection_year": 2022,
        "cross_year_validation": 2023,
        "final_time_test": 2024,
        "folds": fold_rows,
        "index_names": {
            "cross_year": "cross_year_2023",
            "final_time": "final_time_2024",
            "selection": "selection_2022",
        },
    }
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    report = [
        "# Temporal evaluation protocol",
        "",
        "- 2022: event-balanced calendar-block cross-validation for selection, calibration, and budget choice.",
        "- 2023: locked cross-year validation.",
        "- 2024: frozen final time test.",
        "- Purge: 30 hours around each held-out block (24-hour history plus 6-hour horizon).",
        "- All stations in the same calendar block are assigned to the same fold.",
        "",
        folds.to_markdown(index=False),
        "",
        "The 2024 data have previously been inspected in this project. Report them as a retrospectively locked temporal confirmation set, not as a never-observed independent cohort.",
    ]
    (destination / "temporal_protocol_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Temporal protocol indices complete: {destination}", flush=True)
    return destination

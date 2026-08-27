from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np
import pandas as pd

from .baseline_experiment import FeatureTransformer, load_feature_transformer, read_timeline, split_years, timeline_files
from .config import resolve_project_path
from .preprocessing import causal_recurrence_features, prepare_output_root, write_json
from .torch_runtime import Dataset, torch


TARGET_COLUMNS = [
    "issue_time",
    "risk_set",
    "hazard_label",
    "hard_negative_1h",
    "hard_negative_3h",
    "hard_negative_6h",
    "exposure_e1_cold_humid",
    "exposure_e2_fog_low_visibility",
    "exposure_e3_any",
    "risk_spell_index",
    "risk_spell_step",
    "next_recurrent_event_index",
    "station_index",
    "city_index",
    "seen_in_development",
]

RECURRENCE_CONTINUOUS_FEATURES = [
    "time_since_last_recurrent_event_hours",
    "current_event_order",
    "events_past_7d",
    "events_past_30d",
    "previous_event_duration_hours",
    "previous_event_max_thickness",
    "previous_event_severity",
]
RECURRENCE_BINARY_FEATURES = ["previous_recurrent_event_missing"]


@dataclass(frozen=True)
class DeepSampleContract:
    feature_names: list[str]
    history_steps: int
    horizon_steps: int
    step_minutes: int
    split_column: str
    forbidden_input_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "feature_count": len(self.feature_names),
            "history_steps": self.history_steps,
            "history_hours": self.history_steps * self.step_minutes / 60.0,
            "horizon_steps": self.horizon_steps,
            "horizon_hours": self.horizon_steps * self.step_minutes / 60.0,
            "step_minutes": self.step_minutes,
            "split_column": self.split_column,
            "history_semantics": "Rows [issue_index-history_steps+1, issue_index], ending at issue_time; no future row is used.",
            "target_semantics": "Step k labels onset in [issue_time+k*step, issue_time+(k+1)*step).",
            "forbidden_input_columns": self.forbidden_input_columns,
            "future_only_metadata": ["next_recurrent_event_index", "target_event_id", "future_event_step"],
        }


def validate_feature_contract(preprocessed_root: Path, feature_names: list[str]) -> list[str]:
    contract = json.loads((preprocessed_root / "feature_contract.json").read_text(encoding="utf-8"))
    forbidden = set(contract["forbidden_as_model_inputs"])
    forbidden.update({"next_recurrent_event_index", "event_id", "reference_issue_time"})
    leaked = sorted(forbidden.intersection(feature_names))
    if leaked:
        raise ValueError(f"Deep feature contract contains future/label leakage columns: {leaked}")
    return sorted(forbidden)


def contiguous_history_mask(issue_time_ns: np.ndarray, history_steps: int, step_minutes: int) -> np.ndarray:
    times = np.asarray(issue_time_ns, dtype=np.int64)
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    valid = np.zeros(times.size, dtype=bool)
    if times.size < history_steps:
        return valid
    expected = int(step_minutes * 60 * 1_000_000_000)
    contiguous = np.diff(times) == expected
    run = 1
    for index in range(times.size):
        if index > 0:
            run = run + 1 if contiguous[index - 1] else 1
        valid[index] = run >= history_steps
    return valid


def multistep_targets(
    risk_set: np.ndarray,
    hazard_label: np.ndarray,
    issue_time_ns: np.ndarray,
    start_index: int,
    horizon_steps: int,
    step_minutes: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a right-censored discrete-time event target without reading beyond the requested horizon."""

    risk = np.asarray(risk_set, dtype=np.int8)
    hazard = np.asarray(hazard_label, dtype=np.int8)
    times = np.asarray(issue_time_ns, dtype=np.int64)
    labels = np.zeros(horizon_steps, dtype=np.float32)
    mask = np.zeros(horizon_steps, dtype=np.float32)
    event_step = -1
    expected_delta = int(step_minutes * 60 * 1_000_000_000)
    previous_time: int | None = None
    for step in range(horizon_steps):
        index = start_index + step
        if index >= risk.size:
            break
        current_time = int(times[index])
        if previous_time is not None and current_time - previous_time != expected_delta:
            break
        if risk[index] != 1:
            break
        mask[step] = 1.0
        if hazard[index] == 1:
            labels[step] = 1.0
            event_step = step
            break
        previous_time = current_time
    return labels, mask, event_step


def batched_multistep_targets(
    risk_set: np.ndarray,
    hazard_label: np.ndarray,
    issue_time_ns: np.ndarray,
    start_indices: np.ndarray,
    horizon_steps: int,
    step_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``multistep_targets`` for file-grouped training batches."""

    risk = np.asarray(risk_set, dtype=np.int8)
    hazard = np.asarray(hazard_label, dtype=np.int8)
    times = np.asarray(issue_time_ns, dtype=np.int64)
    starts = np.asarray(start_indices, dtype=np.int64).reshape(-1)
    steps = np.arange(horizon_steps, dtype=np.int64)
    indices = starts[:, None] + steps[None, :]
    in_bounds = indices < risk.size
    safe_indices = np.clip(indices, 0, max(risk.size - 1, 0))
    expected_delta = int(step_minutes * 60 * 1_000_000_000)
    expected_times = times[starts, None] + steps[None, :] * expected_delta
    raw_valid = in_bounds & (times[safe_indices] == expected_times) & (risk[safe_indices] == 1)
    valid_prefix = np.logical_and.accumulate(raw_valid, axis=1)
    raw_events = (hazard[safe_indices] == 1) & raw_valid
    no_prior_event = (np.cumsum(raw_events, axis=1) - raw_events.astype(np.int64)) == 0
    mask = valid_prefix & no_prior_event
    labels = raw_events & mask
    has_event = labels.any(axis=1)
    event_steps = np.where(has_event, labels.argmax(axis=1), -1).astype(np.int64)
    return labels.astype(np.float32), mask.astype(np.float32), event_steps


def _timeline_cache_paths(cache_root: Path, source_path: Path) -> tuple[Path, Path]:
    year = source_path.parent.name
    stem = source_path.name.removesuffix(".csv.gz").removesuffix(".csv").removesuffix(".parquet")
    return cache_root / "features" / year / f"{stem}.npy", cache_root / "targets" / year / f"{stem}.npz"


def _save_timeline_cache(
    frame: pd.DataFrame,
    feature_values: np.ndarray,
    feature_path: Path,
    target_path: Path,
) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, np.asarray(feature_values, dtype=np.float32), allow_pickle=False)
    times = pd.to_datetime(frame["issue_time"], errors="coerce").astype("int64").to_numpy(dtype=np.int64)
    arrays: dict[str, np.ndarray] = {"issue_time_ns": times}
    for column in TARGET_COLUMNS:
        if column == "issue_time":
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        dtype = np.float32 if column.endswith("_hours") else np.int32
        arrays[column] = values.to_numpy(dtype=dtype)
    np.savez_compressed(target_path, **arrays)


def _eligible_rows(
    frame: pd.DataFrame,
    split_column: str,
    split_name: str,
    history_steps: int,
    step_minutes: int,
) -> np.ndarray:
    issue_time = pd.to_datetime(frame["issue_time"], errors="coerce").astype("int64").to_numpy(dtype=np.int64)
    history_valid = contiguous_history_mask(issue_time, history_steps, step_minutes)
    return (
        frame[split_column].eq(split_name).to_numpy()
        & frame["risk_set"].eq(1).to_numpy()
        & history_valid
    )


def _valid_events_by_station(events_path: Path) -> dict[str, list[dict[str, Any]]]:
    events = pd.read_csv(events_path, low_memory=False)
    valid = events.loc[pd.to_numeric(events["valid_target_event"], errors="coerce").eq(1)].copy()
    valid["onset_time"] = pd.to_datetime(valid["onset_time"], errors="coerce")
    valid["end_time"] = pd.to_datetime(valid["end_time"], errors="coerce")
    valid = valid.dropna(subset=["station_code", "onset_time", "end_time"])
    return {
        str(station): group.sort_values("onset_time").to_dict(orient="records")
        for station, group in valid.groupby("station_code", sort=True)
    }


def _add_recurrence_columns(
    frame: pd.DataFrame,
    events_by_station: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    station_code = str(frame["station_code"].iloc[0])
    values = causal_recurrence_features(
        pd.to_datetime(frame["issue_time"], errors="coerce"),
        events_by_station.get(station_code, []),
    )
    for name, array in values.items():
        frame[name] = array
    return frame


def _extend_transformer_with_recurrence(
    base: FeatureTransformer,
    training_paths: list[Path],
    split_column: str,
    events_by_station: dict[str, list[dict[str, Any]]],
) -> FeatureTransformer:
    added_continuous = [name for name in RECURRENCE_CONTINUOUS_FEATURES if name not in base.feature_names]
    added_binary = [name for name in RECURRENCE_BINARY_FEATURES if name not in base.feature_names]
    if not added_continuous and not added_binary:
        return base
    counts = np.zeros(len(added_continuous), dtype=np.int64)
    sums = np.zeros(len(added_continuous), dtype=np.float64)
    sums_sq = np.zeros(len(added_continuous), dtype=np.float64)
    for index, path in enumerate(training_paths, start=1):
        frame = read_timeline(path, ["station_code", "issue_time", "risk_set", split_column])
        frame = _add_recurrence_columns(frame, events_by_station)
        mask = frame[split_column].eq("train").to_numpy() & frame["risk_set"].eq(1).to_numpy()
        values = frame.loc[mask, added_continuous].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        counts += finite.sum(axis=0)
        sums += np.where(finite, values, 0.0).sum(axis=0)
        sums_sq += np.where(finite, np.square(values), 0.0).sum(axis=0)
        if index % 10 == 0 or index == len(training_paths):
            print(f"Estimated recurrence normalization from {index}/{len(training_paths)} files", flush=True)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.divide(sums_sq, counts, out=np.ones_like(sums_sq), where=counts > 0) - np.square(means)
    stds = np.sqrt(np.maximum(variances, 1e-8))
    return FeatureTransformer(
        continuous_features=[*base.continuous_features, *added_continuous],
        binary_features=[*base.binary_features, *added_binary],
        means=np.concatenate([base.means, means]),
        stds=np.concatenate([base.stds, stds]),
        clip_z=base.clip_z,
    )


def build_deep_cache(config: dict[str, Any], config_path: Path, overwrite: bool = False) -> Path:
    preprocessed_root = resolve_project_path(config["preprocessed_root"])
    cache_root = resolve_project_path(config["cache_root"])
    prepare_output_root(cache_root, overwrite)
    step_minutes = int(config["step_minutes"])
    history_steps = int(round(float(config["history_hours"]) * 60 / step_minutes))
    horizon_steps = int(config["horizon_steps"])
    scheme = str(config["scheme"])
    split_column = f"split_{scheme}"
    transformer = load_feature_transformer(preprocessed_root, scheme, float(config["continuous_clip_z"]))
    stored_feature_names = list(transformer.feature_names)
    events_by_station: dict[str, list[dict[str, Any]]] = {}
    recurrence_enabled = bool(config.get("recurrence_covariates", {}).get("enabled", False))
    preprocessed_config = json.loads((preprocessed_root / "resolved_config.json").read_text(encoding="utf-8"))
    if recurrence_enabled:
        events_path = resolve_project_path(
            config.get("recurrence_covariates", {}).get(
                "events_path", preprocessed_root / "events_recurrent.csv"
            )
        )
        events_by_station = _valid_events_by_station(events_path)
        training_years = split_years(preprocessed_config, scheme, "train")
        transformer = _extend_transformer_with_recurrence(
            transformer,
            timeline_files(preprocessed_root, training_years),
            split_column,
            events_by_station,
        )
    forbidden = validate_feature_contract(preprocessed_root, transformer.feature_names)
    contract = DeepSampleContract(
        transformer.feature_names,
        history_steps,
        horizon_steps,
        step_minutes,
        split_column,
        forbidden,
    )
    write_json(cache_root / "sample_contract.json", contract.to_dict())
    write_json(cache_root / "feature_transformer.json", transformer.to_dict())
    write_json(cache_root / "resolved_config.json", config)

    split_names = list(config.get("cache_splits", ["train", "validation"]))
    years_by_split = {name: split_years(preprocessed_config, scheme, name) for name in split_names}
    all_years = sorted({year for years in years_by_split.values() for year in years})
    paths = timeline_files(preprocessed_root, all_years)
    count_columns = [split_column, "issue_time", "risk_set", "onset_within_6h", "hard_negative_6h"]
    counts = {"near_event": 0, "hard_negative": 0, "easy_negative": 0}
    per_file_counts: dict[str, dict[str, int]] = {}
    for index, path in enumerate(paths, start=1):
        frame = read_timeline(path, count_columns)
        eligible = _eligible_rows(frame, split_column, "train", history_steps, step_minutes)
        near = eligible & frame["onset_within_6h"].eq(1).to_numpy()
        hard = eligible & ~near & frame["hard_negative_6h"].eq(1).to_numpy()
        easy = eligible & ~near & ~hard
        file_count = {"near_event": int(near.sum()), "hard_negative": int(hard.sum()), "easy_negative": int(easy.sum())}
        per_file_counts[str(path.resolve())] = file_count
        for key, value in file_count.items():
            counts[key] += value
        if index % 10 == 0 or index == len(paths):
            print(f"Counted deep training strata in {index}/{len(paths)} files", flush=True)

    sampling = config["sampling"]
    hard_probability = min(1.0, float(sampling["hard_negative_target"]) / max(counts["hard_negative"], 1))
    easy_probability = min(1.0, float(sampling["easy_negative_target"]) / max(counts["easy_negative"], 1))
    base_seed = int(sampling["seed"])
    index_parts: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"file_id": [], "row_index": [], "sample_weight": [], "stratum": []}
        for name in [*split_names, "train_full"]
    }
    file_manifest: list[dict[str, Any]] = []
    read_columns = list(
        dict.fromkeys(
            [
                "station_code",
                *stored_feature_names,
                *TARGET_COLUMNS,
                split_column,
                "onset_within_6h",
                "hard_negative_6h",
            ]
        )
    )

    for file_id, path in enumerate(paths):
        frame = read_timeline(path, read_columns)
        if recurrence_enabled:
            frame = _add_recurrence_columns(frame, events_by_station)
        feature_path, target_path = _timeline_cache_paths(cache_root, path)
        feature_values = transformer.transform(frame)
        _save_timeline_cache(frame, feature_values, feature_path, target_path)
        station_code = str(frame["station_code"].iloc[0])
        file_manifest.append(
            {
                "file_id": file_id,
                "source_path": str(path.resolve()),
                "feature_path": str(feature_path.relative_to(cache_root)),
                "target_path": str(target_path.relative_to(cache_root)),
                "station_code": station_code,
                "year": int(path.parent.name),
                "rows": int(frame.shape[0]),
            }
        )
        for split_name in split_names:
            eligible = _eligible_rows(frame, split_column, split_name, history_steps, step_minutes)
            if split_name == "train":
                near = eligible & frame["onset_within_6h"].eq(1).to_numpy()
                hard = eligible & ~near & frame["hard_negative_6h"].eq(1).to_numpy()
                easy = eligible & ~near & ~hard
                rng = np.random.default_rng(base_seed + file_id * 104729)
                selected_hard = hard & (rng.random(frame.shape[0]) < hard_probability)
                selected_easy = easy & (rng.random(frame.shape[0]) < easy_probability)
                selected = near | selected_hard | selected_easy
                rows = np.flatnonzero(selected).astype(np.int32)
                weights = np.ones(rows.size, dtype=np.float32)
                selected_hard_at_rows = selected_hard[rows]
                selected_easy_at_rows = selected_easy[rows]
                weights[selected_hard_at_rows] = np.float32(1.0 / hard_probability)
                weights[selected_easy_at_rows] = np.float32(1.0 / easy_probability)
                strata = np.zeros(rows.size, dtype=np.int8)
                strata[selected_hard_at_rows] = 1
                strata[selected_easy_at_rows] = 2
            else:
                rows = np.flatnonzero(eligible).astype(np.int32)
                weights = np.ones(rows.size, dtype=np.float32)
                near_at_rows = frame["onset_within_6h"].eq(1).to_numpy()[rows]
                hard_at_rows = (~near_at_rows) & frame["hard_negative_6h"].eq(1).to_numpy()[rows]
                strata = np.full(rows.size, 2, dtype=np.int8)
                strata[hard_at_rows] = 1
                strata[near_at_rows] = 0
            index_parts[split_name]["file_id"].append(np.full(rows.size, file_id, dtype=np.int16))
            index_parts[split_name]["row_index"].append(rows)
            index_parts[split_name]["sample_weight"].append(weights)
            index_parts[split_name]["stratum"].append(strata)
        full_train_eligible = _eligible_rows(frame, split_column, "train", history_steps, step_minutes)
        full_train_rows = np.flatnonzero(full_train_eligible).astype(np.int32)
        full_near = frame["onset_within_6h"].eq(1).to_numpy()[full_train_rows]
        full_hard = (~full_near) & frame["hard_negative_6h"].eq(1).to_numpy()[full_train_rows]
        full_strata = np.full(full_train_rows.size, 2, dtype=np.int8)
        full_strata[full_hard] = 1
        full_strata[full_near] = 0
        index_parts["train_full"]["file_id"].append(
            np.full(full_train_rows.size, file_id, dtype=np.int16)
        )
        index_parts["train_full"]["row_index"].append(full_train_rows)
        index_parts["train_full"]["sample_weight"].append(
            np.ones(full_train_rows.size, dtype=np.float32)
        )
        index_parts["train_full"]["stratum"].append(full_strata)
        if (file_id + 1) % 10 == 0 or file_id + 1 == len(paths):
            print(f"Built deep timeline cache from {file_id + 1}/{len(paths)} files", flush=True)

    write_json(cache_root / "timeline_manifest.json", {"files": file_manifest})
    index_summary: dict[str, Any] = {}
    for split_name, values in index_parts.items():
        arrays = {key: np.concatenate(parts) for key, parts in values.items()}
        np.savez_compressed(cache_root / f"index_{split_name}.npz", **arrays)
        index_summary[split_name] = {
            "rows": int(arrays["row_index"].size),
            "near_event_rows": int((arrays["stratum"] == 0).sum()),
            "hard_negative_rows": int((arrays["stratum"] == 1).sum()),
            "easy_negative_rows": int((arrays["stratum"] == 2).sum()),
            "represented_weight_sum": float(arrays["sample_weight"].sum()),
        }
    write_json(
        cache_root / "cache_summary.json",
        {
            "config_path": str(config_path),
            "training_stratum_totals": counts,
            "hard_sampling_probability": hard_probability,
            "easy_sampling_probability": easy_probability,
            "file_count": len(file_manifest),
            "splits": index_summary,
        },
    )
    print(f"Deep timeline cache complete: {cache_root}", flush=True)
    return cache_root


class DeepHazardWindowDataset(Dataset):
    def __init__(self, cache_root: str | Path, split: str, maximum_open_files: int = 4):
        self.cache_root = Path(cache_root).resolve()
        self.contract = json.loads((self.cache_root / "sample_contract.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
        self.files = {int(value["file_id"]): value for value in manifest["files"]}
        with np.load(self.cache_root / f"index_{split}.npz", allow_pickle=False) as index:
            self.file_ids = index["file_id"].copy()
            self.row_indices = index["row_index"].copy()
            self.sample_weights = index["sample_weight"].copy()
            self.strata = index["stratum"].copy()
        self.maximum_open_files = max(int(maximum_open_files), 1)
        self._cache: OrderedDict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = OrderedDict()

    def __len__(self) -> int:
        return int(self.row_indices.size)

    def _open_file(self, file_id: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if file_id in self._cache:
            value = self._cache.pop(file_id)
            self._cache[file_id] = value
            return value
        metadata = self.files[file_id]
        features = np.load(self.cache_root / metadata["feature_path"], mmap_mode="r", allow_pickle=False)
        with np.load(self.cache_root / metadata["target_path"], allow_pickle=False) as values:
            targets = {key: values[key].copy() for key in values.files}
        self._cache[file_id] = (features, targets)
        while len(self._cache) > self.maximum_open_files:
            self._cache.popitem(last=False)
        return features, targets

    def __getitem__(self, item: int) -> dict[str, Any]:
        file_id = int(self.file_ids[item])
        row_index = int(self.row_indices[item])
        features, targets = self._open_file(file_id)
        history_steps = int(self.contract["history_steps"])
        start = row_index - history_steps + 1
        if start < 0:
            raise RuntimeError("Cached issue row does not contain the required history")
        history = np.array(features[start : row_index + 1], dtype=np.float32, copy=True)
        labels, risk_mask, event_step = multistep_targets(
            targets["risk_set"],
            targets["hazard_label"],
            targets["issue_time_ns"],
            row_index,
            int(self.contract["horizon_steps"]),
            int(self.contract["step_minutes"]),
        )
        next_event = int(targets["next_recurrent_event_index"][row_index])
        station_code = str(self.files[file_id]["station_code"])
        target_event_id = f"{station_code}-E{next_event:04d}" if event_step >= 0 and next_event > 0 else ""
        hard_flags = np.asarray(
            [
                targets["hard_negative_1h"][row_index],
                targets["hard_negative_3h"][row_index],
                targets["hard_negative_6h"][row_index],
            ],
            dtype=np.float32,
        )
        exposure_flags = np.asarray(
            [
                targets["exposure_e1_cold_humid"][row_index],
                targets["exposure_e2_fog_low_visibility"][row_index],
                targets["exposure_e3_any"][row_index],
            ],
            dtype=np.float32,
        )
        return {
            "history": torch.from_numpy(history),
            "hazard_target": torch.from_numpy(labels),
            "risk_mask": torch.from_numpy(risk_mask),
            "sample_weight": torch.tensor(float(self.sample_weights[item]), dtype=torch.float32),
            "station_index": torch.tensor(int(targets["station_index"][row_index]), dtype=torch.long),
            "city_index": torch.tensor(int(targets["city_index"][row_index]), dtype=torch.long),
            "risk_spell_index": torch.tensor(int(targets["risk_spell_index"][row_index]), dtype=torch.long),
            "hard_negative_flags": torch.from_numpy(hard_flags),
            "exposure_flags": torch.from_numpy(exposure_flags),
            "issue_time_ns": torch.tensor(int(targets["issue_time_ns"][row_index]), dtype=torch.long),
            "future_event_step": torch.tensor(event_step, dtype=torch.long),
            "target_event_id": target_event_id,
            "file_id": torch.tensor(file_id, dtype=torch.long),
            "row_index": torch.tensor(row_index, dtype=torch.long),
            "stratum": torch.tensor(int(self.strata[item]), dtype=torch.long),
        }


def stratified_subsample_positions(
    strata: np.ndarray,
    maximum_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(strata, dtype=np.int8)
    total = int(values.size)
    if maximum_samples is None or maximum_samples >= total:
        positions = np.arange(total, dtype=np.int64)
        return positions, np.ones(total, dtype=np.float32), {
            "requested_maximum": maximum_samples,
            "selected": total,
            "selection_probability_by_stratum": {str(key): 1.0 for key in [0, 1, 2]},
        }
    maximum = max(int(maximum_samples), 3)
    rng = np.random.default_rng(seed)
    available = {key: np.flatnonzero(values == key) for key in [0, 1, 2]}
    near_goal = len(available[0]) if len(available[0]) <= maximum // 2 else maximum // 4
    remaining = maximum - near_goal
    hard_goal = min(len(available[1]), remaining // 2)
    easy_goal = min(len(available[2]), remaining - hard_goal)
    remaining -= hard_goal + easy_goal
    goals = {0: near_goal, 1: hard_goal, 2: easy_goal}
    while remaining > 0:
        changed = False
        for key in [1, 2, 0]:
            if goals[key] < len(available[key]):
                goals[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    selected_parts: list[np.ndarray] = []
    factors: list[np.ndarray] = []
    probabilities: dict[str, float] = {}
    selected_counts: dict[str, int] = {}
    for key in [0, 1, 2]:
        count = min(goals[key], len(available[key]))
        selected = rng.choice(available[key], size=count, replace=False) if count else np.empty(0, dtype=np.int64)
        probability = count / len(available[key]) if len(available[key]) else 1.0
        selected_parts.append(np.asarray(selected, dtype=np.int64))
        factors.append(np.full(count, 1.0 / max(probability, 1e-12), dtype=np.float32))
        probabilities[str(key)] = float(probability)
        selected_counts[str(key)] = int(count)
    positions = np.concatenate(selected_parts)
    selection_weights = np.concatenate(factors)
    order = rng.permutation(positions.size)
    return positions[order], selection_weights[order], {
        "requested_maximum": maximum_samples,
        "selected": int(positions.size),
        "selected_by_stratum": selected_counts,
        "selection_probability_by_stratum": probabilities,
    }


def uniform_negative_subsample_positions(
    strata: np.ndarray,
    maximum_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep onset-near rows and sample all negative strata without hard-negative enrichment."""

    values = np.asarray(strata, dtype=np.int8)
    total = int(values.size)
    if maximum_samples is None or maximum_samples >= total:
        positions = np.arange(total, dtype=np.int64)
        return positions, np.ones(total, dtype=np.float32), {
            "strategy": "uniform_negative",
            "requested_maximum": maximum_samples,
            "selected": total,
            "selection_probability_by_group": {"near_event": 1.0, "all_negative": 1.0},
        }
    maximum = max(int(maximum_samples), 2)
    rng = np.random.default_rng(seed)
    near = np.flatnonzero(values == 0)
    negative = np.flatnonzero(values != 0)
    near_count = min(near.size, maximum)
    negative_count = min(negative.size, maximum - near_count)
    selected_near = near if near_count == near.size else rng.choice(near, size=near_count, replace=False)
    selected_negative = (
        rng.choice(negative, size=negative_count, replace=False)
        if negative_count < negative.size
        else negative
    )
    near_probability = near_count / max(near.size, 1)
    negative_probability = negative_count / max(negative.size, 1)
    positions = np.concatenate([selected_near, selected_negative]).astype(np.int64, copy=False)
    weights = np.concatenate(
        [
            np.full(near_count, 1.0 / max(near_probability, 1e-12), dtype=np.float32),
            np.full(negative_count, 1.0 / max(negative_probability, 1e-12), dtype=np.float32),
        ]
    )
    order = rng.permutation(positions.size)
    selected_values = values[positions]
    return positions[order], weights[order], {
        "strategy": "uniform_negative",
        "requested_maximum": maximum_samples,
        "selected": int(positions.size),
        "selected_by_stratum": {
            str(key): int((selected_values == key).sum()) for key in [0, 1, 2]
        },
        "selection_probability_by_group": {
            "near_event": float(near_probability),
            "all_negative": float(negative_probability),
        },
    }


class DeepCacheBatchSource:
    """File-grouped, vectorized batch reader for practical multi-step training and evaluation."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        maximum_samples: int | None = None,
        selection_seed: int = 0,
        sampling_strategy: str = "stratified",
        permuted_feature_indices: list[int] | None = None,
        permutation_block_days: int = 7,
        include_file_ids: set[int] | None = None,
        feature_center: np.ndarray | None = None,
        feature_scale: np.ndarray | None = None,
        prefetch_batches: int = 0,
    ):
        self.cache_root = Path(cache_root).resolve()
        self.contract = json.loads((self.cache_root / "sample_contract.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
        self.files = {int(value["file_id"]): value for value in manifest["files"]}
        with np.load(self.cache_root / f"index_{split}.npz", allow_pickle=False) as index:
            all_file_ids = index["file_id"].copy()
            all_row_indices = index["row_index"].copy()
            all_sample_weights = index["sample_weight"].copy()
            all_strata = index["stratum"].copy()
        if include_file_ids is not None:
            included = np.isin(all_file_ids, np.asarray(sorted(include_file_ids), dtype=np.int64))
            all_file_ids = all_file_ids[included]
            all_row_indices = all_row_indices[included]
            all_sample_weights = all_sample_weights[included]
            all_strata = all_strata[included]
            if all_file_ids.size == 0:
                raise ValueError(f"No rows remain in split {split} after file filtering")
        if sampling_strategy == "stratified":
            positions, selection_weights, summary = stratified_subsample_positions(
                all_strata, maximum_samples, selection_seed
            )
            summary["strategy"] = "hard_negative_enriched"
        elif sampling_strategy == "uniform_negative":
            positions, selection_weights, summary = uniform_negative_subsample_positions(
                all_strata, maximum_samples, selection_seed
            )
        else:
            raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")
        self.file_ids = all_file_ids[positions]
        self.row_indices = all_row_indices[positions]
        self.sample_weights = all_sample_weights[positions] * selection_weights
        self.strata = all_strata[positions]
        self.selection_summary = summary
        self.selection_summary["split"] = split
        self.selection_summary["represented_weight_sum"] = float(self.sample_weights.sum())
        self.selection_summary["included_file_ids"] = (
            None if include_file_ids is None else sorted(int(value) for value in include_file_ids)
        )
        self.permuted_feature_indices = tuple(int(value) for value in (permuted_feature_indices or []))
        self.permutation_block_days = max(int(permutation_block_days), 1)
        self.selection_seed = int(selection_seed)
        self.prefetch_batches = max(int(prefetch_batches), 0)
        feature_count = int(self.contract["feature_count"])
        self.feature_center = np.zeros(feature_count, dtype=np.float32)
        self.feature_scale = np.ones(feature_count, dtype=np.float32)
        if feature_center is not None:
            center = np.asarray(feature_center, dtype=np.float32)
            if center.shape != (feature_count,):
                raise ValueError("feature_center must match the cache feature count")
            self.feature_center = center
        if feature_scale is not None:
            scale = np.asarray(feature_scale, dtype=np.float32)
            if scale.shape != (feature_count,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError("feature_scale must be finite, positive, and match the feature count")
            self.feature_scale = scale
        self._positions_by_file = {
            int(file_id): np.flatnonzero(self.file_ids == file_id) for file_id in np.unique(self.file_ids)
        }

    def __len__(self) -> int:
        return int(self.row_indices.size)

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        include_file_ids: set[int] | None = None,
    ):
        iterator = self._iter_batches_sync(batch_size, shuffle, seed, include_file_ids)
        if self.prefetch_batches <= 0:
            yield from iterator
            return
        queue: Queue[Any] = Queue(maxsize=self.prefetch_batches)
        sentinel = object()

        def produce() -> None:
            try:
                for batch in iterator:
                    queue.put(batch)
            except BaseException as error:
                queue.put(error)
            finally:
                queue.put(sentinel)

        worker = Thread(target=produce, name="deep-cache-prefetch", daemon=True)
        worker.start()
        while True:
            value = queue.get()
            if value is sentinel:
                break
            if isinstance(value, BaseException):
                raise value
            yield value
        worker.join()

    def _iter_batches_sync(
        self,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        include_file_ids: set[int] | None = None,
    ):
        batch_size = max(int(batch_size), 1)
        rng = np.random.default_rng(seed)
        file_ids = np.asarray(list(self._positions_by_file), dtype=np.int64)
        if include_file_ids is not None:
            file_ids = np.asarray(
                [file_id for file_id in file_ids if int(file_id) in include_file_ids], dtype=np.int64
            )
        if shuffle:
            rng.shuffle(file_ids)
        history_steps = int(self.contract["history_steps"])
        horizon_steps = int(self.contract["horizon_steps"])
        step_minutes = int(self.contract["step_minutes"])
        history_offsets = np.arange(history_steps - 1, -1, -1, dtype=np.int64)
        target_names = [
            "issue_time_ns",
            "risk_set",
            "hazard_label",
            "station_index",
            "city_index",
            "risk_spell_index",
            "hard_negative_1h",
            "hard_negative_3h",
            "hard_negative_6h",
            "exposure_e1_cold_humid",
            "exposure_e2_fog_low_visibility",
            "exposure_e3_any",
        ]
        for file_id in file_ids:
            positions = self._positions_by_file[int(file_id)].copy()
            if shuffle:
                rng.shuffle(positions)
            metadata = self.files[int(file_id)]
            features = np.load(self.cache_root / metadata["feature_path"], mmap_mode="r", allow_pickle=False)
            with np.load(self.cache_root / metadata["target_path"], allow_pickle=False) as values:
                targets = {name: values[name].copy() for name in target_names}
            permutation = None
            if self.permuted_feature_indices:
                timestamps = pd.to_datetime(targets["issue_time_ns"])
                season_year = np.where(timestamps.month >= 11, timestamps.year, timestamps.year - 1)
                permutation = np.arange(timestamps.size, dtype=np.int64)
                shift = max(int(round(self.permutation_block_days * 24 * 60 / step_minutes)), 1)
                for season in np.unique(season_year):
                    season_rows = np.flatnonzero(season_year == season)
                    if season_rows.size > 1:
                        offset = shift % season_rows.size
                        permutation[season_rows] = np.roll(season_rows, offset)
            for start in range(0, positions.size, batch_size):
                selected = positions[start : start + batch_size]
                rows = self.row_indices[selected].astype(np.int64, copy=False)
                history_indices = rows[:, None] - history_offsets[None, :]
                if np.any(history_indices < 0):
                    raise RuntimeError("Deep cache contains a row without its contracted history")
                history = np.asarray(features[history_indices], dtype=np.float32)
                if permutation is not None:
                    permuted_history = np.asarray(features[permutation[history_indices]], dtype=np.float32)
                    history[:, :, self.permuted_feature_indices] = permuted_history[
                        :, :, self.permuted_feature_indices
                    ]
                history = (history - self.feature_center[None, None, :]) / self.feature_scale[
                    None, None, :
                ]
                labels, risk_mask, event_steps = batched_multistep_targets(
                    targets["risk_set"],
                    targets["hazard_label"],
                    targets["issue_time_ns"],
                    rows,
                    horizon_steps,
                    step_minutes,
                )
                hard_flags = np.column_stack(
                    [targets[f"hard_negative_{horizon}h"][rows] for horizon in [1, 3, 6]]
                ).astype(np.float32)
                exposure_flags = np.column_stack(
                    [
                        targets["exposure_e1_cold_humid"][rows],
                        targets["exposure_e2_fog_low_visibility"][rows],
                        targets["exposure_e3_any"][rows],
                    ]
                ).astype(np.float32)
                yield {
                    "history": torch.from_numpy(history),
                    "hazard_target": torch.from_numpy(labels),
                    "risk_mask": torch.from_numpy(risk_mask),
                    "sample_weight": torch.from_numpy(self.sample_weights[selected].astype(np.float32, copy=False)),
                    "station_index": torch.from_numpy(targets["station_index"][rows].astype(np.int64)),
                    "city_index": torch.from_numpy(targets["city_index"][rows].astype(np.int64)),
                    "risk_spell_index": torch.from_numpy(targets["risk_spell_index"][rows].astype(np.int64)),
                    "hard_negative_flags": torch.from_numpy(hard_flags),
                    "exposure_flags": torch.from_numpy(exposure_flags),
                    "issue_time_ns": torch.from_numpy(targets["issue_time_ns"][rows].astype(np.int64)),
                    "future_event_step": torch.from_numpy(event_steps),
                    "file_id": torch.full((rows.size,), int(file_id), dtype=torch.long),
                    "row_index": torch.from_numpy(rows),
                    "stratum": torch.from_numpy(self.strata[selected].astype(np.int64, copy=False)),
                }

from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import resolve_project_path
from .deep_data import DeepCacheBatchSource
from .deep_training import (
    WeightedTrajectoryCalibrator,
    _device_from_name,
    build_model,
    collect_trajectory_arrays,
    cumulative_probability_at_step,
    evaluate_model_nll,
    set_reproducible_seed,
    train_epoch,
    trajectory_probability_rows,
    weighted_probability_metric_row,
)
from .torch_runtime import torch


ECE_EDGES = np.asarray(
    [
        0.0,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0 + 1e-9,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SpatialHoldout:
    protocol: str
    fold_id: str
    heldout_stations: tuple[str, ...]
    heldout_region: str | None = None


def _prepare_output(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot be used together")
    if path.exists() and any(path.iterdir()) and not resume:
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_or_validate_request(path: Path, request: dict[str, Any], resume: bool) -> None:
    if resume:
        if not path.exists():
            raise FileNotFoundError(f"Resume requested but protocol request is missing: {path}")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != request:
            raise ValueError(
                "Resume parameters differ from the original spatial protocol request; "
                "use the original parameters or start with --overwrite."
            )
        return
    path.write_text(json.dumps(request, indent=2), encoding="utf-8")


def _manifest(cache_root: Path) -> tuple[dict[int, dict[str, Any]], dict[str, set[int]]]:
    value = json.loads((cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
    files = {int(row["file_id"]): row for row in value["files"]}
    by_station: dict[str, set[int]] = {}
    for file_id, row in files.items():
        by_station.setdefault(str(row["station_code"]), set()).add(file_id)
    return files, by_station


def build_spatial_holdouts(
    catalog: pd.DataFrame,
    events: pd.DataFrame,
    station_folds: int = 5,
) -> tuple[list[SpatialHoldout], pd.DataFrame]:
    catalog = catalog.copy()
    catalog["station_code"] = catalog["station_code"].astype(str)
    catalog["city"] = catalog["city"].astype(str)
    valid = events.loc[
        pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).eq(1)
        & pd.to_datetime(events["onset_time"], errors="coerce").dt.year.eq(2022)
    ]
    event_count = valid.groupby(valid["station_code"].astype(str)).size().to_dict()
    fold_events = np.zeros(station_folds, dtype=np.int64)
    fold_stations = np.zeros(station_folds, dtype=np.int64)
    city_counts: dict[str, np.ndarray] = {
        city: np.zeros(station_folds, dtype=np.int64) for city in sorted(catalog["city"].unique())
    }
    assignments: list[dict[str, Any]] = []
    ordered = catalog.assign(
        event_count_2022=catalog["station_code"].map(event_count).fillna(0).astype(int)
    ).sort_values(["event_count_2022", "city", "station_code"], ascending=[False, True, True])
    for row in ordered.itertuples(index=False):
        city = str(row.city)
        fold = min(
            range(station_folds),
            key=lambda value: (
                city_counts[city][value],
                fold_events[value],
                fold_stations[value],
                value,
            ),
        )
        count = int(row.event_count_2022)
        fold_events[fold] += count
        fold_stations[fold] += 1
        city_counts[city][fold] += 1
        assignments.append(
            {
                "protocol": "station_group_cv",
                "fold_id": f"station_fold_{fold}",
                "station_code": str(row.station_code),
                "city": city,
                "event_count_2022": count,
            }
        )
    assignment_frame = pd.DataFrame(assignments)
    holdouts: list[SpatialHoldout] = []
    for fold_id, group in assignment_frame.groupby("fold_id", sort=True):
        holdouts.append(
            SpatialHoldout(
                protocol="station_group_cv",
                fold_id=str(fold_id),
                heldout_stations=tuple(sorted(group["station_code"].astype(str))),
            )
        )
    for city, group in catalog.groupby("city", sort=True):
        holdouts.append(
            SpatialHoldout(
                protocol="region_loco",
                fold_id=f"region_{int(group['city_index'].iloc[0])}",
                heldout_stations=tuple(sorted(group["station_code"].astype(str))),
                heldout_region=str(city),
            )
        )
    region_rows = [
        {
            "protocol": holdout.protocol,
            "fold_id": holdout.fold_id,
            "station_code": station,
            "city": holdout.heldout_region,
            "event_count_2022": int(event_count.get(station, 0)),
        }
        for holdout in holdouts
        if holdout.protocol == "region_loco"
        for station in holdout.heldout_stations
    ]
    return holdouts, pd.concat([assignment_frame, pd.DataFrame(region_rows)], ignore_index=True)


def training_only_feature_affine(
    cache_root: Path,
    files: dict[int, dict[str, Any]],
    included_stations: set[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    contract = json.loads((cache_root / "sample_contract.json").read_text(encoding="utf-8"))
    transformer = json.loads(
        (cache_root / "feature_transformer.json").read_text(encoding="utf-8")
    )
    names = list(contract["feature_names"])
    continuous_indices = np.asarray(
        [names.index(name) for name in transformer["continuous_features"]], dtype=np.int64
    )
    count = 0
    total = np.zeros(continuous_indices.size, dtype=np.float64)
    total_square = np.zeros(continuous_indices.size, dtype=np.float64)
    source_files = []
    for file_id, metadata in files.items():
        if int(metadata["year"]) != 2022 or str(metadata["station_code"]) not in included_stations:
            continue
        values = np.load(
            cache_root / metadata["feature_path"], mmap_mode="r", allow_pickle=False
        )
        current = np.asarray(values[:, continuous_indices], dtype=np.float64)
        count += current.shape[0]
        total += current.sum(axis=0)
        total_square += np.square(current).sum(axis=0)
        source_files.append(int(file_id))
    if count == 0:
        raise ValueError("No seen-station 2022 rows are available for normalization")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-8)
    center = np.zeros(len(names), dtype=np.float32)
    scale = np.ones(len(names), dtype=np.float32)
    center[continuous_indices] = mean.astype(np.float32)
    scale[continuous_indices] = np.sqrt(variance).astype(np.float32)
    return center, scale, {
        "rows": int(count),
        "source_file_ids": source_files,
        "included_stations": sorted(included_stations),
        "excluded_station_count": int(len(set(str(row["station_code"]) for row in files.values()) - included_stations)),
        "semantics": "Affine normalization estimated from 2022 covariates of seen stations only.",
    }


def _source(
    cache_root: Path,
    split: str,
    file_ids: set[int],
    maximum_samples: int | None,
    seed: int,
    center: np.ndarray,
    scale: np.ndarray,
    prefetch_batches: int,
) -> DeepCacheBatchSource:
    return DeepCacheBatchSource(
        cache_root,
        split,
        maximum_samples,
        seed,
        include_file_ids=file_ids,
        feature_center=center,
        feature_scale=scale,
        prefetch_batches=prefetch_batches,
    )


def _train_inner_fold(
    model_name: str,
    seed: int,
    inner_fold: int,
    config: dict[str, Any],
    cache_root: Path,
    seen_file_ids: set[int],
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    maximum_train_samples: int | None,
    maximum_early_samples: int | None,
    maximum_calibration_samples: int | None,
    training_batch_size: int,
    inference_batch_size: int,
    prefetch_batches: int,
) -> tuple[int, dict[str, np.ndarray], list[dict[str, Any]]]:
    training = config["training"]
    train_source = _source(
        cache_root,
        f"cv_fold{inner_fold}_train",
        seen_file_ids,
        maximum_train_samples,
        seed + inner_fold * 1009,
        center,
        scale,
        prefetch_batches,
    )
    early_source = _source(
        cache_root,
        f"cv_fold{inner_fold}_validation",
        seen_file_ids,
        maximum_early_samples,
        seed + inner_fold * 1009 + 1,
        center,
        scale,
        prefetch_batches,
    )
    calibration_source = _source(
        cache_root,
        f"cv_fold{inner_fold}_validation",
        seen_file_ids,
        maximum_calibration_samples,
        seed + inner_fold * 1009 + 2,
        center,
        scale,
        prefetch_batches,
    )
    set_reproducible_seed(seed + inner_fold * 7919)
    model = build_model(
        model_name,
        int(train_source.contract["feature_count"]),
        int(train_source.contract["horizon_steps"]),
        config,
        feature_names=list(train_source.contract["feature_names"]),
    ).to(device)
    if model.has_hierarchical_barrier:
        raise ValueError("Spatial generalization requires no learned identity embedding/barrier")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    patience = int(training["early_stopping_patience"])
    maximum_epochs = int(training["maximum_epochs"])
    best_epoch = 0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        train_summary = train_epoch(
            model,
            train_source,
            optimizer,
            device,
            training_batch_size,
            seed + inner_fold * 7919 + epoch * 101,
            float(training["gradient_clip_norm"]),
            float(training["hard_negative_loss_multiplier"]),
            return_details=True,
        )
        development = evaluate_model_nll(
            model, early_source, device, inference_batch_size, return_details=True
        )
        loss = float(development["nll"])
        scheduler.step(loss)
        improved = loss < best_loss - 1e-8
        if improved:
            best_loss = loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            without_improvement = 0
        else:
            without_improvement += 1
        history.append(
            {
                "inner_fold": inner_fold,
                "epoch": epoch,
                "training_nll": float(train_summary["nll"]),
                "development_nll": loss,
                "best_so_far": int(improved),
                "training_positive_bin_rate": float(train_summary["positive_bin_rate"]),
                "development_positive_bin_rate": float(development["positive_bin_rate"]),
            }
        )
        if without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Spatial inner fold did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    arrays = collect_trajectory_arrays(
        model,
        calibration_source,
        device,
        inference_batch_size,
        include_current_features=False,
        include_metadata=False,
    )
    return best_epoch, arrays, history


def _fixed_refit(
    model_name: str,
    seed: int,
    fixed_epochs: int,
    config: dict[str, Any],
    cache_root: Path,
    seen_file_ids: set[int],
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    maximum_train_samples: int | None,
    training_batch_size: int,
    prefetch_batches: int,
) -> tuple[Any, list[dict[str, Any]]]:
    source = _source(
        cache_root,
        "selection_2022",
        seen_file_ids,
        maximum_train_samples,
        seed + 7001,
        center,
        scale,
        prefetch_batches,
    )
    set_reproducible_seed(seed + 7001)
    model = build_model(
        model_name,
        int(source.contract["feature_count"]),
        int(source.contract["horizon_steps"]),
        config,
        feature_names=list(source.contract["feature_names"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history = []
    for epoch in range(1, fixed_epochs + 1):
        summary = train_epoch(
            model,
            source,
            optimizer,
            device,
            training_batch_size,
            seed + epoch * 1009,
            float(config["training"]["gradient_clip_norm"]),
            float(config["training"]["hard_negative_loss_multiplier"]),
            return_details=True,
        )
        history.append(
            {"epoch": epoch, "training_nll": float(summary["nll"]), "stage": "fixed_refit"}
        )
    return model, history


def _save_prediction(
    path: Path,
    model: Any,
    calibrator: WeightedTrajectoryCalibrator,
    source: DeepCacheBatchSource,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    arrays = collect_trajectory_arrays(
        model,
        source,
        device,
        batch_size,
        include_current_features=False,
        include_metadata=True,
    )
    risk = cumulative_probability_at_step(
        arrays["eta"],
        arrays["eta"].shape[1],
        eta_shift=calibrator.log_rate_shift,
        eta_slope=calibrator.slope,
    )
    event = arrays["target"].max(axis=1) > 0.5
    observed = event | (arrays["risk_mask"][:, -1] > 0.5)
    order = np.lexsort((arrays["row_index"], arrays["file_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        file_id=arrays["file_id"][order],
        row_index=arrays["row_index"][order],
        issue_time_ns=arrays["issue_time_ns"][order],
        risk_6h=risk[order].astype(np.float32),
        onset_within_6h=event[order].astype(np.int8),
        observed_6h=observed[order].astype(np.int8),
        sample_weight=arrays["sample_weight"][order].astype(np.float64),
    )
    return trajectory_probability_rows(
        "rec_none_hazard",
        arrays["eta"],
        arrays["target"],
        arrays["risk_mask"],
        arrays["sample_weight"],
        int(source.contract["step_minutes"]),
        "calibrated_seen_station_2022",
        eta_shift=calibrator.log_rate_shift,
        eta_slope=calibrator.slope,
    )


def _run_holdout_seed(
    holdout: SpatialHoldout,
    model_name: str,
    seed: int,
    inner_folds: list[int],
    config: dict[str, Any],
    cache_root: Path,
    files: dict[int, dict[str, Any]],
    by_station: dict[str, set[int]],
    all_stations: set[str],
    output_root: Path,
    device: torch.device,
    maximum_train_samples: int | None,
    maximum_early_samples: int | None,
    maximum_calibration_samples: int | None,
    maximum_evaluation_samples: int | None,
    training_batch_size: int,
    inference_batch_size: int,
    prefetch_batches: int,
    resume: bool,
) -> list[dict[str, Any]]:
    run_root = output_root / "runs" / holdout.protocol / holdout.fold_id / f"seed_{seed}"
    complete_path = run_root / "complete.json"
    metric_path = run_root / "metrics.csv"
    if resume and complete_path.exists() and metric_path.exists():
        print(f"Reused spatial holdout: {holdout.protocol} {holdout.fold_id} seed={seed}", flush=True)
        return pd.read_csv(metric_path).to_dict(orient="records")
    heldout = set(holdout.heldout_stations)
    seen = all_stations - heldout
    if not seen or seen & heldout:
        raise ValueError("Spatial seen and heldout station sets must be nonempty and disjoint")
    seen_file_ids = set().union(*(by_station[station] for station in seen))
    heldout_file_ids = set().union(*(by_station[station] for station in heldout))
    if seen_file_ids & heldout_file_ids:
        raise RuntimeError("Spatial file leakage detected")
    normalization_root = output_root / "normalization" / holdout.protocol / holdout.fold_id
    normalization_array_path = normalization_root / "feature_affine.npz"
    normalization_manifest_path = normalization_root / "normalization_manifest.json"
    if normalization_array_path.exists() and normalization_manifest_path.exists():
        with np.load(normalization_array_path, allow_pickle=False) as values:
            center = values["center"].copy()
            scale = values["scale"].copy()
        normalization = json.loads(
            normalization_manifest_path.read_text(encoding="utf-8")
        )
        if normalization["included_stations"] != sorted(seen):
            raise ValueError("Cached spatial normalization uses a different seen-station set")
    else:
        center, scale, normalization = training_only_feature_affine(
            cache_root, files, seen
        )
        normalization_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(normalization_array_path, center=center, scale=scale)
        normalization_manifest_path.write_text(
            json.dumps(normalization, indent=2), encoding="utf-8"
        )
    best_epochs = []
    calibration_parts: list[dict[str, np.ndarray]] = []
    history_rows: list[dict[str, Any]] = []
    for inner_fold in inner_folds:
        best_epoch, arrays, history = _train_inner_fold(
            model_name,
            seed,
            inner_fold,
            config,
            cache_root,
            seen_file_ids,
            center,
            scale,
            device,
            maximum_train_samples,
            maximum_early_samples,
            maximum_calibration_samples,
            training_batch_size,
            inference_batch_size,
            prefetch_batches,
        )
        best_epochs.append(best_epoch)
        calibration_parts.append(arrays)
        history_rows.extend(
            {
                **row,
                "protocol": holdout.protocol,
                "fold_id": holdout.fold_id,
                "seed": seed,
            }
            for row in history
        )
    pooled = {
        key: np.concatenate([part[key] for part in calibration_parts], axis=0)
        for key in calibration_parts[0]
    }
    calibrator = WeightedTrajectoryCalibrator().fit(
        pooled["eta"], pooled["target"], pooled["risk_mask"], pooled["sample_weight"]
    )
    fixed_epochs = max(1, int(np.median(best_epochs)))
    trained_model, training_history = _fixed_refit(
        model_name,
        seed,
        fixed_epochs,
        config,
        cache_root,
        seen_file_ids,
        center,
        scale,
        device,
        maximum_train_samples,
        training_batch_size,
        prefetch_batches,
    )
    history_rows.extend(
        {
            **row,
            "protocol": holdout.protocol,
            "fold_id": holdout.fold_id,
            "seed": seed,
        }
        for row in training_history
    )
    metric_rows: list[dict[str, Any]] = []
    for split, display in (
        ("selection_2022_full", "2022_spatial_holdout"),
        ("cross_year_2023", "2023_spatial_temporal"),
        ("final_time_2024", "2024_spatial_temporal"),
    ):
        source = _source(
            cache_root,
            split,
            heldout_file_ids,
            maximum_evaluation_samples,
            seed + int(display[:4]),
            center,
            scale,
            prefetch_batches,
        )
        rows = _save_prediction(
            run_root / "predictions" / f"{display}.npz",
            trained_model,
            calibrator,
            source,
            device,
            inference_batch_size,
        )
        for row in rows:
            row.update(
                {
                    "protocol": holdout.protocol,
                    "fold_id": holdout.fold_id,
                    "heldout_region": holdout.heldout_region,
                    "heldout_station_count": len(heldout),
                    "seen_station_count": len(seen),
                    "seed": seed,
                    "split": display,
                    "fixed_epochs": fixed_epochs,
                }
            )
        metric_rows.extend(rows)
    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(run_root / "training_history.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
    torch.save(
        {
            "model_name": model_name,
            "seed": seed,
            "model_config": trained_model.to_config(),
            "state_dict": {
                name: value.detach().cpu() for name, value in trained_model.state_dict().items()
            },
            "calibrator": calibrator.to_dict(),
            "fixed_epochs": fixed_epochs,
            "inner_best_epochs": best_epochs,
            "seen_stations": sorted(seen),
            "heldout_stations": sorted(heldout),
            "normalization": normalization,
            "feature_center": center,
            "feature_scale": scale,
        },
        run_root / "checkpoint.pt",
    )
    complete_path.write_text(
        json.dumps(
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "protocol": holdout.protocol,
                "fold_id": holdout.fold_id,
                "seed": seed,
                "heldout_stations": sorted(heldout),
                "seen_stations": sorted(seen),
                "fixed_epochs": fixed_epochs,
                "inner_best_epochs": best_epochs,
                "leakage_audit_passed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Spatial holdout complete: {holdout.protocol} {holdout.fold_id} seed={seed}",
        flush=True,
    )
    return metric_rows


def _load_seed_prediction(
    output_root: Path,
    protocol: str,
    holdouts: list[SpatialHoldout],
    seed: int,
    split: str,
) -> dict[str, np.ndarray]:
    parts = []
    for holdout in holdouts:
        if holdout.protocol != protocol:
            continue
        path = (
            output_root
            / "runs"
            / protocol
            / holdout.fold_id
            / f"seed_{seed}"
            / "predictions"
            / f"{split}.npz"
        )
        with np.load(path, allow_pickle=False) as values:
            parts.append({key: values[key].copy() for key in values.files})
    combined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.lexsort((combined["row_index"], combined["file_id"]))
    return {key: value[order] for key, value in combined.items()}


def _ensemble_protocol_predictions(
    output_root: Path,
    protocol: str,
    holdouts: list[SpatialHoldout],
    seeds: list[int],
    split: str,
) -> dict[str, np.ndarray]:
    parts = [
        _load_seed_prediction(output_root, protocol, holdouts, seed, split) for seed in seeds
    ]
    identity = (
        "file_id",
        "row_index",
        "issue_time_ns",
        "onset_within_6h",
        "observed_6h",
    )
    for part in parts[1:]:
        for key in identity:
            if not np.array_equal(parts[0][key], part[key]):
                raise ValueError(f"Spatial prediction alignment failed: {protocol} {split} {key}")
    result = {key: parts[0][key] for key in parts[0] if key != "risk_6h"}
    result["risk_6h"] = np.mean(
        np.stack([part["risk_6h"].astype(np.float64) for part in parts]), axis=0
    )
    return result


def _metric_table(
    predictions: dict[str, np.ndarray],
    file_station: dict[int, str],
    station_city: dict[str, str],
    protocol: str,
    split: str,
) -> pd.DataFrame:
    observed = predictions["observed_6h"].astype(bool)
    label = predictions["onset_within_6h"][observed]
    probability = predictions["risk_6h"][observed]
    weight = predictions["sample_weight"][observed]
    stations = np.asarray([file_station[int(value)] for value in predictions["file_id"][observed]])
    rows = []
    groups = [("all", np.ones(label.size, dtype=bool))]
    groups.extend((f"station::{station}", stations == station) for station in np.unique(stations))
    for city in sorted(set(station_city.values())):
        groups.append(
            (f"region::{city}", np.asarray([station_city[value] == city for value in stations]))
        )
    for group, selected in groups:
        if not selected.any() or label[selected].sum() == 0:
            continue
        row = weighted_probability_metric_row(
            "rec_none_unseen",
            "five_seed_ensemble",
            "6h",
            label[selected],
            probability[selected],
            weight[selected],
        )
        row.update(
            {
                "protocol": protocol,
                "split": split,
                "evaluation_group": group,
                "stations": int(np.unique(stations[selected]).size),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _station_metric_arrays(
    label: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    station_index: np.ndarray,
    station_count: int,
) -> dict[str, np.ndarray]:
    clipped = np.clip(score.astype(np.float64), 1e-12, 1.0 - 1e-12)
    result = {
        "weight": np.bincount(station_index, weights=weight, minlength=station_count),
        "log_loss": np.bincount(
            station_index,
            weights=weight
            * (-(label * np.log(clipped) + (1.0 - label) * np.log1p(-clipped))),
            minlength=station_count,
        ),
        "brier": np.bincount(
            station_index,
            weights=weight * np.square(clipped - label),
            minlength=station_count,
        ),
    }
    bins = np.clip(np.digitize(clipped, ECE_EDGES) - 1, 0, ECE_EDGES.size - 2)
    flat = station_index * (ECE_EDGES.size - 1) + bins
    shape = (station_count, ECE_EDGES.size - 1)
    result["bin_weight"] = np.bincount(
        flat, weights=weight, minlength=int(np.prod(shape))
    ).reshape(shape)
    result["bin_positive"] = np.bincount(
        flat, weights=weight * label, minlength=int(np.prod(shape))
    ).reshape(shape)
    result["bin_probability"] = np.bincount(
        flat, weights=weight * clipped, minlength=int(np.prod(shape))
    ).reshape(shape)
    return result


def _vector_metrics(
    counts: np.ndarray, arrays: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    total = counts @ arrays["weight"]
    log_loss = np.divide(counts @ arrays["log_loss"], total)
    brier = np.divide(counts @ arrays["brier"], total)
    bin_weight = counts @ arrays["bin_weight"]
    positive = counts @ arrays["bin_positive"]
    probability = counts @ arrays["bin_probability"]
    observed = np.divide(
        positive, bin_weight, out=np.zeros_like(positive), where=bin_weight > 0
    )
    predicted = np.divide(
        probability, bin_weight, out=np.zeros_like(probability), where=bin_weight > 0
    )
    ece = np.divide((bin_weight * np.abs(observed - predicted)).sum(axis=1), total)
    return {"log_loss": log_loss, "brier": brier, "ece": ece}


def _prepare_average_precision(
    label: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    station_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    group_end = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    return label[order], weight[order], station_index[order], np.flatnonzero(group_end)


def _average_precision_batch(
    counts: np.ndarray,
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    label, weight, station_index, group_ends = prepared
    weighted = counts[:, station_index] * weight[None, :]
    total_positive = (weighted * label[None, :]).sum(axis=1)
    cumulative_weight = np.cumsum(weighted, axis=1)
    cumulative_positive = np.cumsum(weighted * label[None, :], axis=1)
    true_positive = cumulative_positive[:, group_ends]
    precision = np.divide(
        true_positive,
        cumulative_weight[:, group_ends],
        out=np.zeros_like(true_positive),
        where=cumulative_weight[:, group_ends] > 0,
    )
    delta_positive = np.diff(
        true_positive, axis=1, prepend=np.zeros((true_positive.shape[0], 1))
    )
    return np.divide(
        (delta_positive * precision).sum(axis=1),
        total_positive,
        out=np.full(total_positive.shape, np.nan),
        where=total_positive > 0,
    )


def _paired_station_gap_bootstrap(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    file_station: dict[int, str],
    first_name: str,
    second_name: str,
    split: str,
    samples: int,
    seed: int,
    bootstrap_root: Path,
    batch_size: int,
) -> pd.DataFrame:
    for key in (
        "file_id",
        "row_index",
        "issue_time_ns",
        "onset_within_6h",
        "observed_6h",
    ):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Generalization comparison is not aligned: {split} {key}")
    observed = first["observed_6h"].astype(bool)
    label = first["onset_within_6h"][observed].astype(np.float64)
    weight = first["sample_weight"][observed].astype(np.float64)
    station = np.asarray([file_station[int(value)] for value in first["file_id"][observed]])
    stations = np.sort(np.unique(station))
    lookup = {value: index for index, value in enumerate(stations)}
    station_index = np.asarray([lookup[value] for value in station], dtype=np.int16)
    p_first = np.clip(first["risk_6h"][observed], 1e-12, 1.0 - 1e-12)
    p_second = np.clip(second["risk_6h"][observed], 1e-12, 1.0 - 1e-12)
    if "sample_weight" in second and not np.allclose(
        first["sample_weight"], second["sample_weight"], rtol=1e-6, atol=1e-8
    ):
        raise ValueError(f"Generalization comparison has unequal weights: {split}")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        stations.size, np.full(stations.size, 1.0 / stations.size), size=samples
    ).astype(np.int16)
    scores = {first_name: p_first, second_name: p_second}
    values: dict[str, dict[str, np.ndarray]] = {}
    prepared = {}
    for name, score in scores.items():
        metric_arrays = _station_metric_arrays(
            label, score, weight, station_index, stations.size
        )
        values[name] = _vector_metrics(counts, metric_arrays)
        prepared[name] = _prepare_average_precision(
            label, score, weight, station_index
        )
        values[name]["pr_auc"] = np.full(samples, np.nan)

    bootstrap_root.mkdir(parents=True, exist_ok=True)
    slug = f"{split}__{first_name}_minus_{second_name}"
    final_path = bootstrap_root / f"{slug}.npz"
    partial_path = bootstrap_root / f"{slug}.partial.npz"
    completed = 0
    if final_path.exists():
        with np.load(final_path, allow_pickle=False) as saved:
            draws_by_metric = {key: saved[key].copy() for key in saved.files}
    else:
        if partial_path.exists():
            with np.load(partial_path, allow_pickle=False) as saved:
                completed = int(saved["completed"])
                for name in scores:
                    values[name]["pr_auc"][:completed] = saved[f"pr_auc_{name}"][:completed]
            print(f"Spatial bootstrap resumed: {slug} {completed}/{samples}", flush=True)
        for start in range(completed, samples, max(int(batch_size), 1)):
            stop = min(start + max(int(batch_size), 1), samples)
            for name in scores:
                values[name]["pr_auc"][start:stop] = _average_precision_batch(
                    counts[start:stop], prepared[name]
                )
            if stop % 100 < max(int(batch_size), 1) or stop == samples:
                np.savez(
                    partial_path,
                    completed=np.asarray(stop),
                    **{f"pr_auc_{name}": values[name]["pr_auc"] for name in scores},
                )
                print(f"Spatial bootstrap: {slug} {stop}/{samples}", flush=True)
        draws_by_metric = {
            "delta_pr_auc": values[first_name]["pr_auc"] - values[second_name]["pr_auc"],
            "delta_log_loss": values[first_name]["log_loss"]
            - values[second_name]["log_loss"],
            "delta_brier": values[first_name]["brier"] - values[second_name]["brier"],
            "delta_ece": values[first_name]["ece"] - values[second_name]["ece"],
        }
        np.savez_compressed(final_path, **draws_by_metric)
        partial_path.unlink(missing_ok=True)

    rows = []
    for metric, draws in draws_by_metric.items():
        lower_is_better = metric != "delta_pr_auc"
        rows.append(
            {
                "model_a": first_name,
                "model_b": second_name,
                "split": split,
                "metric": metric,
                "estimate": float(np.nanmean(draws)),
                "ci95_low": float(np.nanquantile(draws, 0.025)),
                "ci95_high": float(np.nanquantile(draws, 0.975)),
                "probability_model_a_better": float(
                    np.nanmean(draws < 0) if lower_is_better else np.nanmean(draws > 0)
                ),
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "stations": int(stations.size),
                "bootstrap_samples": samples,
            }
        )
    return pd.DataFrame(rows)


def _load_reference_ensemble(
    protocol_root: Path,
    seeds: list[int],
    split: str,
) -> dict[str, np.ndarray]:
    prediction_root = (
        protocol_root / "oof_predictions"
        if split == "2022_spatial_holdout"
        else protocol_root / "locked_predictions" / split[:4]
    )
    parts = []
    for seed in seeds:
        with np.load(
            prediction_root / f"rec_none_seed_{seed}.npz", allow_pickle=False
        ) as values:
            parts.append({key: values[key].copy() for key in values.files})
    identity = ("file_id", "row_index", "issue_time_ns", "onset_within_6h", "observed_6h")
    for part in parts[1:]:
        for key in identity:
            if not np.array_equal(parts[0][key], part[key]):
                raise ValueError(f"Reference ensemble is not aligned: {split} {key}")
    result = {key: parts[0][key] for key in parts[0] if key != "risk_6h"}
    result["risk_6h"] = np.mean(
        np.stack([part["risk_6h"].astype(np.float64) for part in parts]), axis=0
    )
    return result


def run_spatial_generalization(
    config_path: str | Path,
    output_root: str | Path,
    model_name: str = "rec_none",
    seeds: Iterable[int] | None = None,
    station_folds: int = 5,
    inner_folds: Iterable[int] = (0, 1, 2),
    protocols: Iterable[str] = ("station_group_cv", "region_loco"),
    maximum_train_samples: int | None = 120000,
    maximum_early_samples: int | None = 20000,
    maximum_calibration_samples: int | None = 120000,
    maximum_evaluation_samples: int | None = None,
    training_batch_size: int = 512,
    inference_batch_size: int = 4096,
    prefetch_batches: int = 2,
    cpu_threads: int = 8,
    bootstrap_samples: int = 5000,
    bootstrap_batch_size: int = 16,
    compare_reference: bool = True,
    device_name: str = "auto",
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    config_path = resolve_project_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_root = resolve_project_path(config["cache_root"])
    output_root = resolve_project_path(output_root)
    _prepare_output(output_root, overwrite, resume)
    selected_seeds = (
        [int(value) for value in seeds]
        if seeds is not None
        else [int(value) for value in config["training"]["seeds"]]
    )
    selected_protocols = list(protocols)
    selected_inner_folds = [int(value) for value in inner_folds]
    request = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": model_name,
        "seeds": selected_seeds,
        "station_folds": int(station_folds),
        "inner_folds": selected_inner_folds,
        "protocols": selected_protocols,
        "maximum_train_samples": maximum_train_samples,
        "maximum_early_samples": maximum_early_samples,
        "maximum_calibration_samples": maximum_calibration_samples,
        "maximum_evaluation_samples": maximum_evaluation_samples,
        "training_batch_size": int(training_batch_size),
        "inference_batch_size": int(inference_batch_size),
        "prefetch_batches": int(prefetch_batches),
        "cpu_threads": int(cpu_threads),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_batch_size": int(bootstrap_batch_size),
        "compare_reference": bool(compare_reference),
    }
    _write_or_validate_request(output_root / "protocol_request.json", request, resume)
    files, by_station = _manifest(cache_root)
    catalog = pd.read_csv(
        resolve_project_path(config["preprocessed_root"]) / "station_catalog.csv",
        encoding="utf-8-sig",
    )
    catalog = catalog.loc[catalog["station_code"].astype(str).isin(by_station)].copy()
    events = pd.read_csv(
        resolve_project_path(config["preprocessed_root"]) / "events_recurrent.csv",
        low_memory=False,
    )
    holdouts, assignments = build_spatial_holdouts(catalog, events, station_folds)
    holdouts = [value for value in holdouts if value.protocol in selected_protocols]
    assignments = assignments.loc[assignments["protocol"].isin(selected_protocols)].copy()
    assignments.to_csv(output_root / "spatial_fold_assignments.csv", index=False)
    all_stations = set(catalog["station_code"].astype(str))
    device = _device_from_name(device_name)
    torch.set_num_threads(max(int(cpu_threads), 1))
    metric_rows: list[dict[str, Any]] = []
    for holdout in holdouts:
        for seed in selected_seeds:
            metric_rows.extend(
                _run_holdout_seed(
                    holdout,
                    model_name,
                    seed,
                    selected_inner_folds,
                    config,
                    cache_root,
                    files,
                    by_station,
                    all_stations,
                    output_root,
                    device,
                    maximum_train_samples,
                    maximum_early_samples,
                    maximum_calibration_samples,
                    maximum_evaluation_samples,
                    training_batch_size,
                    inference_batch_size,
                    prefetch_batches,
                    resume,
                )
            )
    pd.DataFrame(metric_rows).to_csv(output_root / "seed_metrics.csv", index=False)

    file_station = {file_id: str(row["station_code"]) for file_id, row in files.items()}
    station_city = dict(zip(catalog["station_code"].astype(str), catalog["city"].astype(str)))
    split_names = (
        "2022_spatial_holdout",
        "2023_spatial_temporal",
        "2024_spatial_temporal",
    )
    ensembles: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    metric_parts = []
    for protocol in selected_protocols:
        for split in split_names:
            ensemble = _ensemble_protocol_predictions(
                output_root, protocol, holdouts, selected_seeds, split
            )
            ensembles[(protocol, split)] = ensemble
            metric_parts.append(
                _metric_table(
                    ensemble, file_station, station_city, protocol, split
                )
            )
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(output_root / "ensemble_generalization_metrics.csv", index=False)

    gap_parts = []
    if compare_reference:
        reference_root = resolve_project_path(config["deep_protocol_output_root"])
        for split_index, split in enumerate(split_names):
            reference = _load_reference_ensemble(reference_root, selected_seeds, split)
            for protocol_index, protocol in enumerate(selected_protocols):
                gap_parts.append(
                    _paired_station_gap_bootstrap(
                        ensembles[(protocol, split)],
                        reference,
                        file_station,
                        protocol,
                        "all_station_reference",
                        split,
                        bootstrap_samples,
                        20260807 + split_index * 1009 + protocol_index * 101,
                        output_root / "bootstrap_replicates",
                        bootstrap_batch_size,
                    )
                )
            if {"station_group_cv", "region_loco"}.issubset(selected_protocols):
                gap_parts.append(
                    _paired_station_gap_bootstrap(
                        ensembles[("region_loco", split)],
                        ensembles[("station_group_cv", split)],
                        file_station,
                        "region_loco",
                        "station_group_cv",
                        split,
                        bootstrap_samples,
                        20260807 + split_index * 1009 + 777,
                        output_root / "bootstrap_replicates",
                        bootstrap_batch_size,
                    )
                )
    gap_frame = pd.concat(gap_parts, ignore_index=True) if gap_parts else pd.DataFrame()
    gap_frame.to_csv(output_root / "paired_generalization_gap_bootstrap.csv", index=False)
    audit = []
    for holdout in holdouts:
        heldout = set(holdout.heldout_stations)
        seen = all_stations - heldout
        audit.append(
            {
                **asdict(holdout),
                "seen_station_count": len(seen),
                "heldout_station_count": len(heldout),
                "station_overlap": len(seen & heldout),
                "training_years": "2022",
                "evaluation_years": "2022,2023,2024",
                "early_stopping_excludes_holdout": 1,
                "calibration_excludes_holdout": 1,
                "normalization_excludes_holdout": 1,
                "identity_embedding_disabled": 1,
            }
        )
    pd.DataFrame(audit).to_csv(output_root / "spatial_leakage_audit.csv", index=False)
    (output_root / "spatial_generalization_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "config_path": str(config_path),
                "model": model_name,
                "protocols": selected_protocols,
                "seeds": selected_seeds,
                "station_folds": station_folds,
                "region_folds": int(catalog["city"].nunique()),
                "inner_temporal_folds": selected_inner_folds,
                "training_year": 2022,
                "evaluation_splits": list(split_names),
                "normalization": "2022 seen-station covariates only",
                "early_stopping": "2022 seen-station inner temporal validation only",
                "calibration": "pooled 2022 seen-station inner temporal predictions only",
                "bootstrap_unit": "station",
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_batch_size": bootstrap_batch_size,
                "reference_comparison": bool(compare_reference),
                "maximum_train_samples": maximum_train_samples,
                "maximum_early_samples": maximum_early_samples,
                "maximum_calibration_samples": maximum_calibration_samples,
                "maximum_evaluation_samples": maximum_evaluation_samples,
                "training_batch_size": training_batch_size,
                "inference_batch_size": inference_batch_size,
                "prefetch_batches": prefetch_batches,
                "cpu_threads": cpu_threads,
                "2024_status": "retrospectively locked because previously inspected",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = [
        "# Unseen-station and unseen-region generalization",
        "",
        "Station-group CV holds out complete stations; region-LOCO holds out every station in one "
        "city. Heldout stations are excluded from training, early stopping, calibration, and "
        "normalization. The final encoder has no station or region identity embedding.",
        "",
        "2022 measures pure spatial transfer. 2023 and 2024 jointly measure spatial and temporal "
        "transfer. Region-LOCO is the strongest internal external-validity test.",
        "",
        "Paired station bootstrap reports the degradation relative to the all-station temporal "
        "reference and the additional region-holdout penalty. Positive delta log loss is worse.",
    ]
    (output_root / "spatial_generalization_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Spatial generalization complete: {output_root}", flush=True)
    return output_root

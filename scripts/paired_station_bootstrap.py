from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_PATH = Path("configs/rig_hazard_recurrence_modeling.json")
OUTPUT = Path("results/recurrence_modeling/paired_station_bootstrap")
CONTROL = "rec_none"
EXPERIMENT = "rec_full"
ECE_EDGES = np.asarray(
    [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
     3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0 + 1e-9],
    dtype=np.float64,
)


def _weights_from_cache(
    cache_root: Path, year: int, file_id: np.ndarray, row_index: np.ndarray
) -> np.ndarray:
    split = {
        2022: "selection_2022_full",
        2023: "cross_year_2023",
        2024: "final_time_2024",
    }[int(year)]
    with np.load(cache_root / f"index_{split}.npz", allow_pickle=False) as values:
        source_key = (
            values["file_id"].astype(np.int64) * np.int64(1 << 32)
            + values["row_index"].astype(np.int64)
        )
        source_weight = values["sample_weight"].astype(np.float64)
    order = np.argsort(source_key, kind="mergesort")
    source_key = source_key[order]
    source_weight = source_weight[order]
    requested = (
        file_id.astype(np.int64) * np.int64(1 << 32) + row_index.astype(np.int64)
    )
    positions = np.searchsorted(source_key, requested)
    if np.any(positions >= source_key.size) or not np.array_equal(
        source_key[np.minimum(positions, source_key.size - 1)], requested
    ):
        raise ValueError(f"Prediction identities do not align to cache split {split}")
    return source_weight[positions]


def load_ensemble(
    root: Path,
    year: int,
    model: str,
    seeds: list[int],
    cache_root: Path | None = None,
) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    required = (
        "file_id", "row_index", "issue_time_ns", "onset_within_6h",
        "observed_6h", "risk_6h",
    )
    for seed in seeds:
        path = (
            root / "oof_predictions" / f"{model}_seed_{seed}.npz"
            if int(year) == 2022
            else root / "locked_predictions" / str(year) / f"{model}_seed_{seed}.npz"
        )
        with np.load(path, allow_pickle=False) as values:
            missing = [key for key in required if key not in values.files]
            if missing:
                raise ValueError(f"Prediction archive is missing {missing}: {path}")
            # Specification runs also contain the full 36-step trajectory.  The
            # bootstrap only needs the fixed 6-hour endpoint, so do not load the
            # much larger matrices into memory.
            keys = [*required]
            if "sample_weight" in values.files:
                keys.append("sample_weight")
            parts.append({key: values[key].copy() for key in keys})
    identity = ("file_id", "row_index", "issue_time_ns", "onset_within_6h", "observed_6h")
    for part in parts[1:]:
        for key in identity:
            if not np.array_equal(parts[0][key], part[key]):
                raise ValueError(f"Unaligned ensemble field: year={year} model={model} field={key}")
    result = {key: parts[0][key] for key in parts[0] if key != "risk_6h"}
    result["risk_6h"] = np.mean(
        np.stack([part["risk_6h"].astype(np.float64) for part in parts]), axis=0
    )
    if "sample_weight" not in result:
        if cache_root is None:
            raise ValueError(
                f"Predictions omit sample_weight and no cache root was supplied: {root} {model} {year}"
            )
        result["sample_weight"] = _weights_from_cache(
            cache_root, year, result["file_id"], result["row_index"]
        )
    return result


def station_metric_arrays(
    label: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    station_index: np.ndarray,
    station_count: int,
) -> dict[str, np.ndarray]:
    clipped = np.clip(score.astype(np.float64), 1e-12, 1.0 - 1e-12)
    arrays = {
        "weight": np.bincount(station_index, weights=weight, minlength=station_count),
        "log_loss": np.bincount(
            station_index,
            weights=weight * (-(label * np.log(clipped) + (1 - label) * np.log1p(-clipped))),
            minlength=station_count,
        ),
        "brier": np.bincount(
            station_index, weights=weight * np.square(clipped - label), minlength=station_count
        ),
    }
    bins = np.clip(np.digitize(clipped, ECE_EDGES) - 1, 0, ECE_EDGES.size - 2)
    flat = station_index * (ECE_EDGES.size - 1) + bins
    shape = (station_count, ECE_EDGES.size - 1)
    arrays["bin_weight"] = np.bincount(flat, weights=weight, minlength=np.prod(shape)).reshape(shape)
    arrays["bin_positive"] = np.bincount(
        flat, weights=weight * label, minlength=np.prod(shape)
    ).reshape(shape)
    arrays["bin_probability"] = np.bincount(
        flat, weights=weight * clipped, minlength=np.prod(shape)
    ).reshape(shape)
    return arrays


def vector_metrics(counts: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    total = counts @ arrays["weight"]
    log_loss = (counts @ arrays["log_loss"]) / total
    brier = (counts @ arrays["brier"]) / total
    bin_weight = counts @ arrays["bin_weight"]
    positive = counts @ arrays["bin_positive"]
    probability = counts @ arrays["bin_probability"]
    observed = np.divide(positive, bin_weight, out=np.zeros_like(positive), where=bin_weight > 0)
    predicted = np.divide(
        probability, bin_weight, out=np.zeros_like(probability), where=bin_weight > 0
    )
    ece = (bin_weight * np.abs(observed - predicted)).sum(axis=1) / total
    return {"log_loss": log_loss, "brier": brier, "ece": ece}


def prepare_average_precision(
    label: np.ndarray, score: np.ndarray, weight: np.ndarray, station_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    group_end = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    return label[order], weight[order], station_index[order], np.flatnonzero(group_end)


def average_precision_batch(
    counts: np.ndarray,
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    label, weight, station_index, group_ends = prepared
    weighted = counts[:, station_index] * weight[None, :]
    total_positive = (weighted * label[None, :]).sum(axis=1)
    cumulative_weight = np.cumsum(weighted, axis=1)
    cumulative_positive = np.cumsum(weighted * label[None, :], axis=1)
    tp = cumulative_positive[:, group_ends]
    precision = np.divide(
        tp, cumulative_weight[:, group_ends], out=np.zeros_like(tp),
        where=cumulative_weight[:, group_ends] > 0,
    )
    delta_tp = np.diff(tp, axis=1, prepend=np.zeros((tp.shape[0], 1)))
    return np.divide(
        (delta_tp * precision).sum(axis=1), total_positive,
        out=np.full(total_positive.shape, np.nan), where=total_positive > 0,
    )


def run_year(
    year: int,
    protocol_root: Path,
    seeds: list[int],
    file_station: dict[int, str],
    replicates: int,
) -> pd.DataFrame:
    experiment = load_ensemble(protocol_root, year, EXPERIMENT, seeds)
    control = load_ensemble(protocol_root, year, CONTROL, seeds)
    for key in ("file_id", "row_index", "onset_within_6h", "observed_6h"):
        if not np.array_equal(experiment[key], control[key]):
            raise ValueError(f"Unaligned paired models: year={year} field={key}")
    observed = experiment["observed_6h"].astype(bool)
    label = experiment["onset_within_6h"][observed].astype(np.int8)
    weight = experiment["sample_weight"][observed].astype(np.float64)
    station_names = np.asarray([file_station[int(value)] for value in experiment["file_id"][observed]])
    unique_stations = np.unique(station_names)
    station_lookup = {value: index for index, value in enumerate(unique_stations)}
    station_index = np.asarray([station_lookup[value] for value in station_names], dtype=np.int16)
    station_count = unique_stations.size
    rng = np.random.default_rng(20260807 + year)
    sampled = rng.integers(0, station_count, size=(replicates, station_count))
    counts = np.zeros((replicates, station_count), dtype=np.int16)
    for replicate in range(replicates):
        counts[replicate] = np.bincount(sampled[replicate], minlength=station_count)
    scores = {
        EXPERIMENT: experiment["risk_6h"][observed].astype(np.float64),
        CONTROL: control["risk_6h"][observed].astype(np.float64),
    }
    metric_values: dict[str, dict[str, np.ndarray]] = {}
    prepared = {}
    for model in (EXPERIMENT, CONTROL):
        arrays = station_metric_arrays(label, scores[model], weight, station_index, station_count)
        metric_values[model] = vector_metrics(counts, arrays)
        prepared[model] = prepare_average_precision(label, scores[model], weight, station_index)
        metric_values[model]["pr_auc"] = np.full(replicates, np.nan)

    partial = OUTPUT / f"bootstrap_{year}_partial.npz"
    completed = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as saved:
            completed = int(saved["completed"])
            for model in (EXPERIMENT, CONTROL):
                metric_values[model]["pr_auc"][:completed] = saved[f"pr_auc_{model}"][:completed]
        print(f"Bootstrap {year}: resumed at {completed}/{replicates}", flush=True)
    batch_size = 16
    for start in range(completed, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        for model in (EXPERIMENT, CONTROL):
            metric_values[model]["pr_auc"][start:stop] = average_precision_batch(
                counts[start:stop], prepared[model]
            )
        if stop % 100 < batch_size or stop == replicates:
            np.savez(
                partial, completed=np.asarray(stop),
                **{f"pr_auc_{model}": metric_values[model]["pr_auc"] for model in (EXPERIMENT, CONTROL)},
            )
            print(f"Bootstrap {year}: {stop}/{replicates}", flush=True)
    partial.unlink(missing_ok=True)
    return pd.DataFrame(
        {
            "year": year,
            "replicate": np.arange(replicates),
            "delta_pr_auc": metric_values[EXPERIMENT]["pr_auc"] - metric_values[CONTROL]["pr_auc"],
            "delta_log_loss": metric_values[EXPERIMENT]["log_loss"] - metric_values[CONTROL]["log_loss"],
            "delta_brier": metric_values[EXPERIMENT]["brier"] - metric_values[CONTROL]["brier"],
            "delta_ece": metric_values[EXPERIMENT]["ece"] - metric_values[CONTROL]["ece"],
        }
    )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol_root = Path(config["deep_protocol_output_root"])
    manifest = json.loads((Path(config["cache_root"]) / "timeline_manifest.json").read_text(encoding="utf-8"))
    file_station = {int(row["file_id"]): str(row["station_code"]) for row in manifest["files"]}
    replicates = int(
        config["recurrence_evaluation"]["paired_station_bootstrap_replicates"]
    )
    seeds = [int(value) for value in config["training"]["seeds"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = [run_year(year, protocol_root, seeds, file_station, replicates) for year in (2023, 2024)]
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(OUTPUT / "probability_bootstrap_replicates.csv.gz", index=False, compression="gzip")
    summary_rows = []
    for year, frame in detail.groupby("year"):
        for metric in ("delta_pr_auc", "delta_log_loss", "delta_brier", "delta_ece"):
            summary_rows.append(
                {
                    "year": year, "comparison": f"{EXPERIMENT}_minus_{CONTROL}", "metric": metric,
                    "mean": frame[metric].mean(), "ci95_low": frame[metric].quantile(0.025),
                    "ci95_high": frame[metric].quantile(0.975), "replicates": replicates,
                }
            )
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "probability_bootstrap_summary.csv", index=False)
    (OUTPUT / "bootstrap_manifest.json").write_text(
        json.dumps(
            {
                "comparison": f"{EXPERIMENT} minus {CONTROL}", "years": [2023, 2024],
                "replicates_per_year": replicates, "cluster_unit": "station",
                "station_draws_per_replicate": 30, "seed_handling": "five-seed ensemble before bootstrap",
                "paired": True, "metrics": ["PR-AUC", "log-loss", "Brier", "ECE"],
            }, indent=2,
        ), encoding="utf-8",
    )
    print(f"Paired station bootstrap complete: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

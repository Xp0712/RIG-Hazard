from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.paired_station_bootstrap import (
    average_precision_batch,
    load_ensemble,
    prepare_average_precision,
    station_metric_arrays,
    vector_metrics,
)


def _comparisons(value: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in value.split(","):
        names = [name.strip() for name in item.split(":")]
        if len(names) != 2 or not all(names):
            raise argparse.ArgumentTypeError(
                "comparisons must use model_a:model_b comma-separated syntax"
            )
        pairs.append((names[0], names[1]))
    return pairs


def _pair_slug(model_a: str, model_b: str) -> str:
    return f"{model_a}_minus_{model_b}"


def _model_roots(value: str) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    if not value.strip():
        return roots
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "model roots must use model=path comma-separated syntax"
            )
        model, path = (part.strip() for part in item.split("=", 1))
        if not model or not path:
            raise argparse.ArgumentTypeError("model roots cannot contain empty values")
        roots[model] = Path(path)
    return roots


def _years(value: str) -> list[int]:
    years = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not years or any(year not in (2022, 2023, 2024) for year in years):
        raise argparse.ArgumentTypeError("years must be a subset of 2022,2023,2024")
    return years


def run_pair_year(
    year: int,
    model_a: str,
    model_b: str,
    protocol_root: Path,
    model_roots: dict[str, Path],
    cache_root: Path,
    seeds: list[int],
    file_station: dict[int, str],
    replicates: int,
    batch_size: int,
    output: Path,
    bootstrap_seed: int,
) -> pd.DataFrame:
    first = load_ensemble(
        model_roots.get(model_a, protocol_root), year, model_a, seeds, cache_root
    )
    second = load_ensemble(
        model_roots.get(model_b, protocol_root), year, model_b, seeds, cache_root
    )
    for key in ("file_id", "row_index", "onset_within_6h", "observed_6h"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(
                f"Unaligned paired models: year={year} comparison={model_a}:{model_b} field={key}"
            )
    observed = first["observed_6h"].astype(bool)
    label = first["onset_within_6h"][observed].astype(np.int8)
    weight = first["sample_weight"][observed].astype(np.float64)
    station_names = np.asarray(
        [file_station[int(value)] for value in first["file_id"][observed]]
    )
    stations = np.unique(station_names)
    lookup = {value: index for index, value in enumerate(stations)}
    station_index = np.asarray([lookup[value] for value in station_names], dtype=np.int16)
    station_count = stations.size
    rng = np.random.default_rng(bootstrap_seed)
    counts = rng.multinomial(
        station_count,
        np.full(station_count, 1.0 / station_count),
        size=replicates,
    ).astype(np.int16)
    scores = {
        model_a: first["risk_6h"][observed].astype(np.float64),
        model_b: second["risk_6h"][observed].astype(np.float64),
    }
    metric_values: dict[str, dict[str, np.ndarray]] = {}
    prepared = {}
    for model in (model_a, model_b):
        arrays = station_metric_arrays(
            label, scores[model], weight, station_index, station_count
        )
        metric_values[model] = vector_metrics(counts, arrays)
        prepared[model] = prepare_average_precision(
            label, scores[model], weight, station_index
        )
        metric_values[model]["pr_auc"] = np.full(replicates, np.nan)

    slug = _pair_slug(model_a, model_b)
    partial = output / f"{slug}_{year}_partial.npz"
    completed = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as saved:
            completed = int(saved["completed"])
            for model in (model_a, model_b):
                metric_values[model]["pr_auc"][:completed] = saved[
                    f"pr_auc_{model}"
                ][:completed]
        print(f"{slug} {year}: resumed at {completed}/{replicates}", flush=True)
    for start in range(completed, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        for model in (model_a, model_b):
            metric_values[model]["pr_auc"][start:stop] = average_precision_batch(
                counts[start:stop], prepared[model]
            )
        if stop % 100 < batch_size or stop == replicates:
            np.savez(
                partial,
                completed=np.asarray(stop),
                **{
                    f"pr_auc_{model}": metric_values[model]["pr_auc"]
                    for model in (model_a, model_b)
                },
            )
            print(f"{slug} {year}: {stop}/{replicates}", flush=True)
    partial.unlink(missing_ok=True)
    return pd.DataFrame(
        {
            "year": year,
            "comparison": slug,
            "replicate": np.arange(replicates),
            "delta_pr_auc": metric_values[model_a]["pr_auc"]
            - metric_values[model_b]["pr_auc"],
            "delta_log_loss": metric_values[model_a]["log_loss"]
            - metric_values[model_b]["log_loss"],
            "delta_brier": metric_values[model_a]["brier"]
            - metric_values[model_b]["brier"],
            "delta_ece": metric_values[model_a]["ece"]
            - metric_values[model_b]["ece"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact paired station bootstrap for targeted probability-model comparisons."
    )
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument(
        "--comparisons",
        type=_comparisons,
        default=_comparisons("rec_load:rec_none,rec_previous:rec_none"),
    )
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--years", type=_years, default=[2023, 2024])
    parser.add_argument(
        "--model-roots",
        type=_model_roots,
        default={},
        help="Optional model=protocol_root mappings for paired cross-root comparisons.",
    )
    parser.add_argument(
        "--output-root",
        default="results/recurrence_modeling/targeted_probability_bootstrap",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    protocol_root = Path(config["deep_protocol_output_root"])
    manifest = json.loads(
        (Path(config["cache_root"]) / "timeline_manifest.json").read_text(encoding="utf-8")
    )
    cache_root = Path(config["cache_root"])
    file_station = {
        int(row["file_id"]): str(row["station_code"]) for row in manifest["files"]
    }
    seeds = [int(value) for value in config["training"]["seeds"]]
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    detail_parts: list[pd.DataFrame] = []
    for comparison_index, (model_a, model_b) in enumerate(args.comparisons):
        slug = _pair_slug(model_a, model_b)
        detail_path = output / f"{slug}_replicates.csv.gz"
        if detail_path.exists():
            print(f"Reused completed comparison: {detail_path}", flush=True)
            detail_parts.append(pd.read_csv(detail_path))
            continue
        frames = [
            run_pair_year(
                year,
                model_a,
                model_b,
                protocol_root,
                args.model_roots,
                cache_root,
                seeds,
                file_station,
                args.replicates,
                args.batch_size,
                output,
                20260807 + year * 1009 + comparison_index * 7919,
            )
            for year in args.years
        ]
        detail = pd.concat(frames, ignore_index=True)
        detail.to_csv(detail_path, index=False, compression="gzip")
        detail_parts.append(detail)

    all_detail = pd.concat(detail_parts, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for (comparison, year), frame in all_detail.groupby(["comparison", "year"]):
        for metric in ("delta_pr_auc", "delta_log_loss", "delta_brier", "delta_ece"):
            values = frame[metric].dropna()
            summary_rows.append(
                {
                    "comparison": comparison,
                    "year": int(year),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci95_low": float(values.quantile(0.025)),
                    "ci95_high": float(values.quantile(0.975)),
                    "replicates": int(values.shape[0]),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        output / "targeted_probability_bootstrap_summary.csv", index=False
    )
    (output / "bootstrap_manifest.json").write_text(
        json.dumps(
            {
                "comparisons": [
                    _pair_slug(model_a, model_b) for model_a, model_b in args.comparisons
                ],
                "years": args.years,
                "replicates_per_year": args.replicates,
                "cluster_unit": "station",
                "paired": True,
                "seed_handling": "five-seed ensemble before station bootstrap",
                "metrics": ["PR-AUC", "log-loss", "Brier", "ECE"],
                "model_roots": {
                    model: str(path) for model, path in args.model_roots.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Targeted probability bootstrap complete: {output}", flush=True)


if __name__ == "__main__":
    main()

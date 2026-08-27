from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.spatial_generalization import run_spatial_generalization


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in csv_strings(value)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unseen-station and unseen-region frozen generalization protocols."
    )
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument(
        "--output-root",
        default="results/recurrence_modeling/spatial_generalization",
    )
    parser.add_argument("--model", default="rec_none")
    parser.add_argument("--seeds", type=csv_ints)
    parser.add_argument("--station-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=csv_ints, default=csv_ints("0,1,2"))
    parser.add_argument(
        "--protocols",
        type=csv_strings,
        default=csv_strings("station_group_cv,region_loco"),
    )
    parser.add_argument("--maximum-train-samples", type=int, default=120000)
    parser.add_argument("--maximum-early-samples", type=int, default=20000)
    parser.add_argument("--maximum-calibration-samples", type=int, default=120000)
    parser.add_argument("--maximum-evaluation-samples", type=int)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-reference-comparison",
        action="store_true",
        help="Validation-only switch; formal experiments must not use this option.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_spatial_generalization(
        args.config,
        args.output_root,
        model_name=args.model,
        seeds=args.seeds,
        station_folds=args.station_folds,
        inner_folds=args.inner_folds,
        protocols=args.protocols,
        maximum_train_samples=args.maximum_train_samples,
        maximum_early_samples=args.maximum_early_samples,
        maximum_calibration_samples=args.maximum_calibration_samples,
        maximum_evaluation_samples=args.maximum_evaluation_samples,
        training_batch_size=args.training_batch_size,
        inference_batch_size=args.inference_batch_size,
        prefetch_batches=args.prefetch_batches,
        cpu_threads=args.cpu_threads,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_batch_size=args.bootstrap_batch_size,
        compare_reference=not args.skip_reference_comparison,
        device_name=args.device,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.utility_warning import DEFAULT_MODELS, run_utility_warning_experiments


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv_strings(value)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run utility-first frozen warning and seasonal recurrence experiments."
    )
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument(
        "--output-root",
        default="results/recurrence_modeling/utility_warning",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--budgets", default="2,5,10,20")
    parser.add_argument(
        "--quantiles", default="0.80,0.85,0.90,0.92,0.94,0.96,0.98,0.99,0.995"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_utility_warning_experiments(
        args.config,
        args.output_root,
        models=_csv_strings(args.models),
        budgets=_csv_floats(args.budgets),
        quantiles=_csv_floats(args.quantiles),
        bootstrap_samples=args.bootstrap_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

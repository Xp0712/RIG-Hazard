from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.seasonal_risk_structure import run_seasonal_risk_structure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nested seasonal first-versus-recurrent discrete hazard tests."
    )
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument(
        "--output-root",
        default="results/recurrence_modeling/seasonal_risk_structure",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--maximum-train-rows", type=int)
    parser.add_argument("--maximum-evaluation-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_seasonal_risk_structure(
        args.config,
        args.output_root,
        number_folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        maximum_train_rows=args.maximum_train_rows,
        maximum_evaluation_rows=args.maximum_evaluation_rows,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

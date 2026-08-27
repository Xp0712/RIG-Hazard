from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.icing_experiment_protocol import audit_icing_experiment_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experiment protocol audit from the experiment specification.")
    parser.add_argument("--config", default="configs/rig_hazard_icing_model_experiments.json")
    parser.add_argument(
        "--output-root",
        default="results/icing_model_experiments/protocol_audit",
    )
    parser.add_argument("--skip-split-manifest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit_icing_experiment_protocol(
        args.config,
        args.output_root,
        write_split_manifest=not args.skip_split_manifest,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

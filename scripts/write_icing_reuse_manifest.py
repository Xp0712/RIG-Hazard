from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _state(path: Path) -> dict[str, object]:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.is_file() else None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record completed experiments reused by the icing-model protocol."
    )
    parser.add_argument(
        "--output",
        default="results/icing_model_experiments/reused_experiments.json",
    )
    args = parser.parse_args()
    paths = {
        "legacy_probability_metrics": Path(
            "results/recurrence_analysis/model_comparison/deep_protocol/locked_year_metrics.csv"
        ),
        "legacy_probability_manifest": Path(
            "results/recurrence_analysis/model_comparison/deep_protocol/run_manifest.json"
        ),
        "recurrence_probability_metrics": Path(
            "results/recurrence_modeling/probability_models/locked_year_metrics.csv"
        ),
        "fair_baseline_metrics": Path(
            "results/recurrence_modeling/fair_baselines/locked_year_metrics.csv"
        ),
        "fair_baseline_complete": Path(
            "results/recurrence_modeling/supplementary_experiments/.complete_fair_baselines"
        ),
        "fair_baseline_summary_complete": Path(
            "results/recurrence_modeling/supplementary_experiments/.complete_fair_baseline_summary"
        ),
        "spatial_generalization_complete": Path(
            "results/recurrence_modeling/spatial_generalization/.complete_spatial_generalization"
        ),
        "utility_warning_complete": Path(
            "results/recurrence_modeling/supplementary_experiments/.complete_utility_warning"
        ),
        "candidate_attribution_complete": Path(
            "results/recurrence_modeling/supplementary_experiments/.complete_candidate_attribution"
        ),
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": (
            "Artifacts with compatible frozen preprocessing, folds, target, seeds, and completion "
            "evidence are reused. This pipeline never launches fair baselines or spatial generalization."
        ),
        "artifacts": {name: _state(path) for name, path in paths.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()

"""Consolidate project data and completed experiment artifacts safely.

The script only moves explicitly listed legacy paths. It never deletes project
documents or experiment results.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
REQUIRED_DOCUMENTS = {
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "results_analysis.md",
}
LEGACY_MOVES = (
    (
        "recurrence_experiments_complete_20260813",
        "results/archive/source_snapshot/directory",
    ),
    (
        "recurrence_experiments_complete_20260813.tar",
        "results/archive/source_snapshot/source_snapshot.tar",
    ),
    ("analysis_outputs", "results/legacy_exploration/analysis_outputs"),
    ("\u8d77\u51b0\u4e8b\u4ef6\u7edf\u8ba1", "results/legacy_exploration/icing_event_statistics"),
    ("\u65f6\u5e8f\u9884\u6d4b\u5206\u6790", "results/legacy_exploration/timeseries_data_audit"),
    ("(\u6797\u7acb\u94ee2025.6.21)\u51dd\u7ed3\u7c7b\u5929\u6c14\u6570\u636e", "data/meteorology_raw"),
    ("\u590d\u53d1\u6570\u636e\u96c6", "data/public_benchmarks"),
    ("analyze_icing_dataset.py", "scripts/legacy_data_audit/analyze_icing_dataset.py"),
    (
        "count_recurrent_icing_events.py",
        "scripts/legacy_data_audit/count_recurrent_icing_events.py",
    ),
)


def move_if_present(
    source: Path,
    destination: Path,
    operations: list[dict[str, str]],
    dry_run: bool,
) -> None:
    if not source.exists():
        return
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    operations.append(
        {
            "action": "move",
            "source": str(source.relative_to(PROJECT_ROOT)),
            "destination": str(destination.relative_to(PROJECT_ROOT)),
        }
    )
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize completed project artifacts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without modifying files.",
    )
    args = parser.parse_args()

    missing_docs = [path for path in REQUIRED_DOCUMENTS if not path.is_file()]
    if missing_docs:
        raise FileNotFoundError(f"Required merged Markdown document missing: {missing_docs}")

    partial = list(PROJECT_ROOT.rglob("*.downloading"))
    if partial:
        raise RuntimeError(
            "Result synchronization is still active; complete it before organizing: "
            + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in partial[:5])
        )

    operations: list[dict[str, str]] = []
    for source, destination in LEGACY_MOVES:
        move_if_present(
            PROJECT_ROOT / source,
            PROJECT_ROOT / destination,
            operations,
            args.dry_run,
        )

    preserved_markdown = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in PROJECT_ROOT.rglob("*.md")
        if ".python_deps" not in path.parts
    )

    if not args.dry_run:
        manifest = {
            "organized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "operations": operations,
            "preserved_markdown_count": len(preserved_markdown),
            "preserved_markdown": preserved_markdown,
        }
        RESULTS_ROOT.mkdir(exist_ok=True)
        (RESULTS_ROOT / "workspace_layout_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Console sessions on Windows may use GBK and cannot render every legacy
    # filename.  Escape only the preview output; the on-disk manifest retains
    # readable Unicode names.
    print(
        json.dumps(
            {"operations": operations, "preserved_markdown": preserved_markdown},
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

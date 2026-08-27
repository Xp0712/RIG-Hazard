"""Regression checks for active project file and directory names."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATHS = (
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "requirements",
    PROJECT_ROOT / "src" / "rig_hazard",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "visualization",
)
NUMBERED_STAGE_PATTERN = re.compile(
    r"(?:^|[_-])(?:v\d+|m\d+|p\d+|e\d+)(?:[_\.-]|$)",
    flags=re.IGNORECASE,
)
TEMPORAL_NAME_PATTERN = re.compile(
    r"(?:^|[_-])(?:next_stage|remaining_experiments|latest|new|final\d*)(?:[_\.-]|$)",
    flags=re.IGNORECASE,
)
NUMBERED_MODEL_IDENTIFIER = re.compile(r"\b(?:m\d+|e\d+)_", flags=re.IGNORECASE)


class ProjectNamingTests(unittest.TestCase):
    def test_active_paths_do_not_use_numbered_stages(self) -> None:
        invalid: list[str] = []
        for root in ACTIVE_PATHS:
            for path in root.rglob("*"):
                if "__pycache__" in path.parts:
                    continue
                if NUMBERED_STAGE_PATTERN.search(path.name) or TEMPORAL_NAME_PATTERN.search(
                    path.name
                ):
                    invalid.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(invalid, [], f"Numbered stage names found: {invalid}")

    def test_canonical_project_files_exist(self) -> None:
        expected = (
            "src/rig_hazard/__init__.py",
            "src/rig_hazard/__main__.py",
            "configs/rig_hazard_preprocessing.json",
            "requirements/runtime.txt",
            "requirements/deep_learning.txt",
            "scripts/resume_main_experiments.sh",
            "scripts/run_spatial_generalization_queue.sh",
            "scripts/upload_spatial_generalization_queue.py",
            "visualization/README.md",
        )
        missing = [relative for relative in expected if not (PROJECT_ROOT / relative).is_file()]
        self.assertEqual(missing, [], f"Canonical project files missing: {missing}")

    def test_legacy_top_level_directories_are_absent(self) -> None:
        legacy = ("rig_hazard", "rig_hazard_outputs", "tools", "logs")
        present = [name for name in legacy if (PROJECT_ROOT / name).exists()]
        self.assertEqual(present, [], f"Legacy top-level directories found: {present}")

    def test_legacy_result_directories_are_absent(self) -> None:
        legacy = (
            "deep_rig_hazard",
            "local_hazard_baselines",
            "paper_figures",
            "preprocessed_10min",
            "recurrence_experiments",
            "recurrence_next_stage",
            "recurrent_icing_spec",
        )
        present = [name for name in legacy if (PROJECT_ROOT / "results" / name).exists()]
        self.assertEqual(present, [], f"Legacy result directories found: {present}")

    def test_active_code_uses_semantic_model_identifiers(self) -> None:
        invalid: list[str] = []
        for root in ACTIVE_PATHS:
            for path in root.rglob("*"):
                if path.suffix not in {".py", ".json", ".sh"}:
                    continue
                if path.name == "naming.py":
                    continue
                text = path.read_text(encoding="utf-8")
                if NUMBERED_MODEL_IDENTIFIER.search(text):
                    invalid.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(
            invalid,
            [],
            f"Numbered model identifiers found outside compatibility boundary: {invalid}",
        )


if __name__ == "__main__":
    unittest.main()

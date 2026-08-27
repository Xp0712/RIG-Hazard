from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_project_path
from .deep_data import validate_feature_contract


ANCHORS = (
    ("gru", "2023_cross_year", 0.1185, 0.011598),
    ("recurrent_dual", "2023_cross_year", 0.1522, 0.010423),
    ("gru", "2024_final_time", 0.1133, 0.015761),
    ("recurrent_dual", "2024_final_time", 0.1556, 0.013852),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(cache_root: Path, name: str) -> dict[str, np.ndarray]:
    path = cache_root / f"index_{name}.npz"
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _keys(values: dict[str, np.ndarray]) -> np.ndarray:
    return (
        values["file_id"].astype(np.int64) * np.int64(1 << 32)
        + values["row_index"].astype(np.int64)
    )


def _split_audit(cache_root: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    selection = _index(cache_root, "selection_2022_full")
    cross_year = _index(cache_root, "cross_year_2023")
    final_time = _index(cache_root, "final_time_2024")
    named = {
        "selection_2022": selection,
        "cross_year_2023": cross_year,
        "final_time_2024": final_time,
    }
    key_sets = {name: set(_keys(values).tolist()) for name, values in named.items()}
    rows: list[dict[str, Any]] = []
    for left, right in (
        ("selection_2022", "cross_year_2023"),
        ("selection_2022", "final_time_2024"),
        ("cross_year_2023", "final_time_2024"),
    ):
        overlap = len(key_sets[left].intersection(key_sets[right]))
        rows.append(
            {
                "check": f"{left}_disjoint_{right}",
                "status": "PASS" if overlap == 0 else "FAIL",
                "observed": overlap,
                "expected": 0,
            }
        )
    validation_keys: set[int] = set()
    for fold in range(5):
        train = _index(cache_root, f"cv_fold{fold}_train")
        validation = _index(cache_root, f"cv_fold{fold}_validation")
        train_keys, fold_keys = set(_keys(train).tolist()), set(_keys(validation).tolist())
        overlap = len(train_keys.intersection(fold_keys))
        rows.append(
            {
                "check": f"fold{fold}_train_validation_disjoint",
                "status": "PASS" if overlap == 0 else "FAIL",
                "observed": overlap,
                "expected": 0,
            }
        )
        duplicate_validation = len(validation_keys.intersection(fold_keys))
        rows.append(
            {
                "check": f"fold{fold}_validation_unique_across_folds",
                "status": "PASS" if duplicate_validation == 0 else "FAIL",
                "observed": duplicate_validation,
                "expected": 0,
            }
        )
        validation_keys.update(fold_keys)
        named[f"cv_fold{fold}_validation"] = validation
    selection_keys = key_sets["selection_2022"]
    symmetric_difference = len(validation_keys.symmetric_difference(selection_keys))
    rows.append(
        {
            "check": "five_validation_folds_cover_selection_2022",
            "status": "PASS" if symmetric_difference == 0 else "FAIL",
            "observed": symmetric_difference,
            "expected": 0,
        }
    )
    return rows, named


def _write_split_manifest(
    cache_root: Path,
    output_path: Path,
    named_indices: dict[str, np.ndarray],
) -> None:
    timeline = json.loads((cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
    files = {int(value["file_id"]): value for value in timeline["files"]}
    fold_by_origin: dict[int, int] = {}
    for fold in range(5):
        values = named_indices[f"cv_fold{fold}_validation"]
        fold_by_origin.update({int(key): fold for key in _keys(values)})
    if output_path.exists():
        output_path.unlink()
    header = True
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        for split_name in ("selection_2022", "cross_year_2023", "final_time_2024"):
            values = named_indices[split_name]
            for file_id in np.unique(values["file_id"]):
                mask = values["file_id"] == file_id
                rows = values["row_index"][mask].astype(np.int64)
                metadata = files[int(file_id)]
                target_path = cache_root / metadata["target_path"]
                with np.load(target_path, allow_pickle=False) as targets:
                    issue_time_ns = targets["issue_time_ns"][rows]
                origin_id = int(file_id) * np.int64(1 << 32) + rows
                frame = pd.DataFrame(
                    {
                        "origin_id": origin_id,
                        "station_id": str(metadata["station_code"]),
                        "timestamp": pd.to_datetime(issue_time_ns),
                        "split": split_name,
                        "fold": [fold_by_origin.get(int(value), -1) for value in origin_id],
                        "purge_flag": 0,
                    }
                )
                frame.to_csv(handle, index=False, header=header)
                header = False


def _baseline_reproduction(project_root: Path) -> pd.DataFrame:
    path = (
        project_root
        / "results/recurrence_analysis/model_comparison/deep_protocol/locked_year_metrics.csv"
    )
    columns = [
        "model", "split", "metric", "expected", "observed", "absolute_difference",
        "seeds", "status", "source",
    ]
    if not path.exists():
        return pd.DataFrame(
            [["", "", "", np.nan, np.nan, np.nan, 0, "MISSING", str(path)]],
            columns=columns,
        )
    metrics = pd.read_csv(path, low_memory=False)
    rows: list[dict[str, Any]] = []
    for model, split, expected_pr, expected_loss in ANCHORS:
        selected = metrics.loc[
            metrics["encoder"].eq(model)
            & metrics["split"].eq(split)
            & metrics["calibration"].eq("calibrated_2022_oof")
            & metrics["horizon"].eq("6h")
        ]
        for metric, expected in (("pr_auc", expected_pr), ("log_loss", expected_loss)):
            observed = float(pd.to_numeric(selected[metric], errors="coerce").mean())
            difference = abs(observed - expected)
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "expected": expected,
                    "observed": observed,
                    "absolute_difference": difference,
                    "seeds": int(selected.shape[0]),
                    "status": "PASS" if difference <= 5e-4 else "REVIEW",
                    "source": str(path),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def audit_icing_experiment_protocol(
    config_path: str | Path,
    output_root: str | Path,
    write_split_manifest: bool = True,
    overwrite: bool = False,
) -> Path:
    config_path = resolve_project_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    output_root = resolve_project_path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = resolve_project_path(config["cache_root"])
    preprocessed_root = resolve_project_path(config["preprocessed_root"])
    contract = json.loads((cache_root / "sample_contract.json").read_text(encoding="utf-8"))
    leaked = validate_feature_contract(preprocessed_root, list(contract["feature_names"]))
    split_rows, named_indices = _split_audit(cache_root)
    split_audit = pd.DataFrame(split_rows)
    split_audit.to_csv(output_root / "split_audit.csv", index=False)
    if write_split_manifest:
        _write_split_manifest(
            cache_root, output_root / "split_manifest.csv.gz", named_indices
        )

    events = pd.read_csv(preprocessed_root / "events_recurrent.csv", low_memory=False)
    validation = json.loads(
        (preprocessed_root / "validation_summary.json").read_text(encoding="utf-8")
    )
    preprocessing_manifest = preprocessed_root / "manifest.json"
    cache_contract = cache_root / "sample_contract.json"
    source_files = pd.read_csv(preprocessed_root / "source_files.csv", low_memory=False)
    data_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_signature": _sha256(preprocessing_manifest),
        "feature_contract_sha256": _sha256(cache_contract),
        "config_sha256": _sha256(config_path),
        "raw_file_inventory_rows": int(source_files.shape[0]),
        "raw_bytes": int(pd.to_numeric(source_files["size_bytes"], errors="coerce").fillna(0).sum()),
        "stations": int(source_files["station_code"].replace("", np.nan).nunique()),
        "valid_reference_events": int(
            pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).eq(1).sum()
        ),
        "eligible_hazard_events": int(validation["metrics"]["eligible_events"]),
        "feature_count": int(contract["feature_count"]),
        "history_steps": int(contract["history_steps"]),
        "horizon_steps": int(contract["horizon_steps"]),
        "split_rows": {
            name: int(values["row_index"].size)
            for name, values in named_indices.items()
            if name in {"selection_2022", "cross_year_2023", "final_time_2024"}
        },
        "validation_status": validation["status"],
        "validation_warnings": validation["warnings"],
        "leakage_columns_forbidden": leaked,
    }
    (output_root / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metric_contract = {
        "probability_target": "36 conditional 10-minute onset hazards",
        "cumulative_risk": "F[k] = 1 - product(1-h[j]), j=1..k",
        "primary_horizon": "6h (k=36)",
        "selection_data": "2022 five-fold purged OOF only",
        "frozen_years": {"2023": "cross-year", "2024": "retrospectively locked"},
        "event_hit": "At least one active alarm in [onset-6h, onset)",
        "lead": "Hours from first active alarm in the valid warning window to onset",
        "false_alarm_time": "Active 10-minute bins outside every valid event warning window",
        "station_month": "Calendar station-month; report raw hours and normalized exposure",
        "bootstrap": "5000 paired station-cluster samples retaining complete within-station rows/events",
        "budget_nesting": "A_2 subset A_5 subset A_10 subset A_20",
    }
    (output_root / "metric_contract.json").write_text(
        json.dumps(metric_contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    reproduction = _baseline_reproduction(project_root)
    reproduction.to_csv(output_root / "baseline_reproduction.csv", index=False)

    failures = split_audit.loc[split_audit["status"].eq("FAIL")]
    known_issues = [
        "2024 has already been inspected and is retrospectively locked, not a never-seen test set.",
        *[str(value) for value in validation.get("warnings", [])],
        "Frozen runs store only selected horizons; current protocol runs save all 36 hazard and cumulative-risk steps.",
        "Existing completed models use the frozen project seed/training contract and are reused rather than retrained with the document's illustrative seed list.",
        "The existing fair-baseline suite uses the earlier common feature contract. "
        "It is reused as requested; the weather ablation trains a weather-only GRU "
        "because the protocol's exact field set is a materially different input experiment.",
    ]
    (output_root / "known_issues.md").write_text(
        "# Known issues\n\n" + "\n".join(f"- {value}" for value in known_issues) + "\n",
        encoding="utf-8",
    )
    status = "PASS" if failures.empty and validation["status"] == "PASS" else "FAIL"
    report = [
        "# experiment protocol audit",
        "",
        f"Status: **{status}**",
        "",
        "The existing preprocessed timelines and deep cache remain the single data source; they are not rebuilt for completed experiments.",
        "",
        "## Frozen counts",
        "",
        f"- Reference events: {data_manifest['valid_reference_events']}",
        f"- Eligible hazard events: {data_manifest['eligible_hazard_events']}",
        f"- Features: {data_manifest['feature_count']}",
        f"- History/horizon: {data_manifest['history_steps']}/{data_manifest['horizon_steps']} steps",
        "",
        "## Split checks",
        "",
        split_audit.to_markdown(index=False),
        "",
        "## Baseline anchors",
        "",
        reproduction.to_markdown(index=False),
        "",
        "See `known_issues.md` for non-blocking interpretation limits.",
    ]
    (output_root / "protocol_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(
            "Experiment protocol audit failed; dependent experiments are blocked"
        )
    return output_root

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.config import resolve_project_path
from rig_hazard.deep_data import DeepCacheBatchSource
from rig_hazard.deep_protocol import _concatenate_arrays, _variant_source_options
from rig_hazard.deep_training import (
    _device_from_name,
    collect_trajectory_arrays,
    load_deep_checkpoint,
    trajectory_probability_matrices,
)
from rig_hazard.evaluation_provenance import sha256_file
from rig_hazard.naming import FROZEN_MAIN_MODEL_ARTIFACT_ID
from rig_hazard.risk_trajectory import (
    audit_trajectory_arrays,
    multihorizon_probability_metrics,
)
from rig_hazard.torch_runtime import torch


SPLITS = {
    "selection_2022_oof": "selection_2022_oof",
    "2023_cross_year": "cross_year_2023",
    "2024_final_time": "final_time_2024",
}


def _checkpoint(root: Path, model: str, seed: int, fold: int | None) -> Path:
    if fold is None:
        return root / "trained_models" / "checkpoints" / f"{model}_seed_{seed}.pt"
    return root / "folds" / f"fold{fold}" / f"{model}_seed_{seed}" / "checkpoints" / f"{model}_seed_{seed}.pt"


def _seed_arrays(
    root: Path,
    model_name: str,
    seed: int,
    display_split: str,
    cache_root: Path,
    config: dict,
    device: torch.device,
    maximum_samples: int | None,
    folds: int,
) -> tuple[dict[str, np.ndarray], dict]:
    options = _variant_source_options(model_name, config)
    if display_split == "selection_2022_oof":
        parts = []
        for fold in range(int(folds)):
            model, _ = load_deep_checkpoint(_checkpoint(root, model_name, seed, fold), str(device))
            source = DeepCacheBatchSource(
                cache_root,
                f"cv_fold{fold}_validation",
                maximum_samples,
                int(config["sampling"]["seed"]) + fold * 1009,
                **options,
            )
            parts.append(
                collect_trajectory_arrays(
                    model,
                    source,
                    device,
                    int(config["training"]["batch_size"]),
                    include_current_features=False,
                    include_metadata=True,
                )
            )
            del model
        arrays = _concatenate_arrays(parts)
        _, checkpoint_payload = load_deep_checkpoint(
            _checkpoint(root, model_name, seed, None), str(device)
        )
        return arrays, checkpoint_payload

    model, checkpoint_payload = load_deep_checkpoint(
        _checkpoint(root, model_name, seed, None), str(device)
    )
    source = DeepCacheBatchSource(
        cache_root,
        SPLITS[display_split],
        maximum_samples,
        int(config["sampling"]["seed"]),
        **options,
    )
    arrays = collect_trajectory_arrays(
        model,
        source,
        device,
        int(config["training"]["batch_size"]),
        include_current_features=False,
        include_metadata=True,
    )
    del model
    return arrays, checkpoint_payload


def _export_split(
    root: Path,
    output: Path,
    model_name: str,
    seeds: list[int],
    display_split: str,
    cache_root: Path,
    config: dict,
    device: torch.device,
    resume: bool,
    maximum_samples: int | None,
    folds: int,
) -> tuple[dict, object, str]:
    destination = output / f"{display_split}.npz"
    if resume and destination.exists():
        print(f"Reused selected trajectory export: {destination}", flush=True)
        with np.load(destination, allow_pickle=False) as values:
            payload = {key: values[key].copy() for key in values.files}
        audit = audit_trajectory_arrays(payload, step_minutes=int(config["step_minutes"]))
        metrics = multihorizon_probability_metrics(
            payload, display_split, step_minutes=int(config["step_minutes"])
        )
        return audit.to_dict(), metrics, sha256_file(destination)
    hazard_sum: np.ndarray | None = None
    cumulative_sum: np.ndarray | None = None
    gate_sum: np.ndarray | None = None
    identity: dict[str, np.ndarray] | None = None
    for index, seed in enumerate(seeds, start=1):
        arrays, checkpoint = _seed_arrays(
            root, model_name, seed, display_split, cache_root, config, device,
            maximum_samples, folds,
        )
        order = np.lexsort((arrays["row_index"], arrays["file_id"]))
        calibrator = checkpoint["trajectory_calibrator"]
        hazard, cumulative = trajectory_probability_matrices(
            arrays["eta"],
            float(calibrator["log_rate_shift"]),
            float(calibrator["slope"]),
        )
        hazard, cumulative = hazard[order], cumulative[order]
        current_identity = {
            "file_id": arrays["file_id"][order].astype(np.int16, copy=False),
            "row_index": arrays["row_index"][order].astype(np.int32, copy=False),
            "issue_time_ns": arrays["issue_time_ns"][order].astype(np.int64, copy=False),
            "y_hazard": arrays["target"][order].astype(np.int8, copy=False),
            "censor_mask": arrays["risk_mask"][order].astype(np.int8, copy=False),
            "sample_weight": arrays["sample_weight"][order].astype(np.float64, copy=False),
        }
        if identity is None:
            identity = {key: value.copy() for key, value in current_identity.items()}
            hazard_sum = hazard.astype(np.float32, copy=True)
            cumulative_sum = cumulative.astype(np.float32, copy=True)
        else:
            for key in ("file_id", "row_index", "issue_time_ns", "y_hazard", "censor_mask"):
                if not np.array_equal(identity[key], current_identity[key]):
                    raise ValueError(
                        f"Unaligned selected trajectory ensemble: split={display_split} seed={seed} field={key}"
                    )
            hazard_sum += hazard.astype(np.float32, copy=False)
            cumulative_sum += cumulative.astype(np.float32, copy=False)
        if "rec_gate" in arrays:
            current_gate = arrays["rec_gate"][order].astype(np.float32, copy=False)
            if gate_sum is None:
                gate_sum = current_gate.copy()
            else:
                gate_sum += current_gate
        print(
            f"Selected trajectory {display_split}: seed {index}/{len(seeds)} complete",
            flush=True,
        )
        del arrays, hazard, cumulative, current_identity

    if identity is None or hazard_sum is None or cumulative_sum is None:
        raise RuntimeError(f"No predictions exported for {display_split}")
    cumulative_mean = cumulative_sum / np.float32(len(seeds))
    if not np.all(np.diff(cumulative_mean, axis=1) >= -1e-6):
        raise ValueError(f"Non-monotone ensemble cumulative risk in {display_split}")
    # A probability ensemble is defined by the mean CDF.  Recover its unique
    # conditional hazard so h and F remain exactly coherent after ensembling.
    survival = np.clip(1.0 - cumulative_mean, 1e-12, 1.0)
    survival_before = np.concatenate(
        [np.ones((survival.shape[0], 1), dtype=np.float32), survival[:, :-1]], axis=1
    )
    hazard_mean = np.clip(1.0 - survival / survival_before, 0.0, 1.0)
    origin_id = (
        identity["file_id"].astype(np.int64) * np.int64(1 << 32)
        + identity["row_index"].astype(np.int64)
    )
    event = identity["y_hazard"].max(axis=1) > 0
    observed = event | (identity["censor_mask"][:, -1] > 0)
    payload = {
        "origin_id": origin_id,
        **identity,
        "h_trajectory": hazard_mean.astype(np.float32, copy=False),
        "F_trajectory": cumulative_mean.astype(np.float32, copy=False),
        "risk_6h": cumulative_mean[:, -1].astype(np.float32, copy=False),
        "onset_within_6h": event.astype(np.int8),
        "observed_6h": observed.astype(np.int8),
    }
    if gate_sum is not None:
        payload["rec_gate"] = (gate_sum / np.float32(len(seeds))).astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    print(f"Selected trajectory export complete: {destination}", flush=True)
    audit = audit_trajectory_arrays(payload, step_minutes=int(config["step_minutes"]))
    if not audit.passed:
        raise RuntimeError(f"Exported trajectory failed audit: {audit.to_dict()}")
    metrics = multihorizon_probability_metrics(
        payload, display_split, step_minutes=int(config["step_minutes"])
    )
    return audit.to_dict(), metrics, sha256_file(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one frozen five-seed model as coherent 36-step ensemble hazards and risks."
    )
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument(
        "--protocol-root",
        default="results/recurrence_modeling/probability_models",
    )
    parser.add_argument(
        "--model",
        default=FROZEN_MAIN_MODEL_ARTIFACT_ID,
        help="Frozen checkpoint artifact identifier.",
    )
    parser.add_argument(
        "--output-root",
        default="results/dynamic_hard_budget/local_weather_hazard_trajectory",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maximum-samples", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--splits", nargs="+", choices=tuple(SPLITS), default=list(SPLITS)
    )
    args = parser.parse_args()

    config = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    cache_root = resolve_project_path(config["cache_root"])
    protocol_root = resolve_project_path(args.protocol_root)
    output = resolve_project_path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in config["training"]["seeds"]]
    device = _device_from_name(args.device)
    torch.set_num_threads(max(int(config["training"].get("torch_num_threads", 1)), 1))
    audit_rows = []
    metric_parts = []
    output_hashes = {}
    for display_split in args.splits:
        audit, metrics, output_hash = _export_split(
            protocol_root,
            output,
            args.model,
            seeds,
            display_split,
            cache_root,
            config,
            device,
            args.resume,
            args.maximum_samples,
            args.folds,
        )
        audit_rows.append({"split": display_split, **audit})
        metric_parts.append(metrics)
        output_hashes[display_split] = output_hash
    import pandas as pd

    pd.DataFrame(audit_rows).to_csv(output / "trajectory_audit.csv", index=False)
    pd.concat(metric_parts, ignore_index=True).to_csv(
        output / "multihorizon_probability_metrics.csv", index=False
    )
    checkpoint_hashes = {}
    for seed in seeds:
        for fold in range(int(args.folds)):
            path = _checkpoint(protocol_root, args.model, seed, fold)
            checkpoint_hashes[str(path.relative_to(protocol_root))] = sha256_file(path)
        path = _checkpoint(protocol_root, args.model, seed, None)
        checkpoint_hashes[str(path.relative_to(protocol_root))] = sha256_file(path)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output / "export_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "protocol_root": str(protocol_root),
                "seeds": seeds,
                "selection_data": "2022 pooled OOF",
                "locked_evaluation": [2023, 2024],
                "config_sha256": config_hash,
                "checkpoint_sha256": checkpoint_hashes,
                "trajectory_sha256": output_hashes,
                "maximum_samples": args.maximum_samples,
                "shape": ["rows", int(config["horizon_steps"])],
                "dtype": "float32",
                "coherence": "Five-seed mean CDF; conditional h is recovered from that CDF so F[k] = 1-product(1-h[j])",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output / ".complete_selected_full_trajectory").touch()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.config import resolve_project_path
from rig_hazard.deep_data import DeepCacheBatchSource
from rig_hazard.deep_training import (
    collect_trajectory_arrays,
    load_deep_checkpoint,
    trajectory_probability_rows,
)


FEATURE_GROUPS = {
    "gap_time": ["time_since_last_recurrent_event_hours"],
    "event_load_7d_30d": ["events_past_7d", "events_past_30d"],
    "event_order": ["current_event_order"],
    "previous_event_attributes": [
        "previous_event_duration_hours",
        "previous_event_max_thickness",
        "previous_event_severity",
    ],
    "previous_event_missing": ["previous_recurrent_event_missing"],
}


def _models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _six_hour_metrics(
    model: torch.nn.Module,
    source: DeepCacheBatchSource,
    device: torch.device,
    batch_size: int,
    calibrator: dict,
    step_minutes: int,
    label: str,
) -> list[dict]:
    arrays = collect_trajectory_arrays(model, source, device, batch_size)
    rows = trajectory_probability_rows(
        label,
        arrays["eta"],
        arrays["target"],
        arrays["risk_mask"],
        arrays["sample_weight"],
        step_minutes,
        "calibrated_2022_oof",
        eta_shift=float(calibrator["log_rate_shift"]),
        eta_slope=float(calibrator["slope"]),
    )
    return [row for row in rows if row["horizon"] == "6h"]


def run_model(
    model_name: str,
    config: dict,
    output_root: Path,
    device: torch.device,
    permutation_rows: int,
    ig_rows: int,
    ig_steps: int,
    batch_size: int,
    overwrite: bool,
) -> None:
    protocol_root = Path(config["deep_protocol_output_root"])
    seed = int(config["training"]["seeds"][0])
    checkpoint_path = (
        protocol_root / "trained_models" / "checkpoints" / f"{model_name}_seed_{seed}.pt"
    )
    model, checkpoint = load_deep_checkpoint(checkpoint_path, str(device))
    calibrator = checkpoint["trajectory_calibrator"]
    cache_root = resolve_project_path(config["cache_root"])
    names = list(checkpoint["sample_contract"]["feature_names"])
    model_output = output_root / model_name
    model_output.mkdir(parents=True, exist_ok=True)

    permutation_path = model_output / "conditional_permutation_2023.csv"
    if overwrite or not permutation_path.exists():
        rows = _six_hour_metrics(
            model,
            DeepCacheBatchSource(cache_root, "cross_year_2023", permutation_rows, seed),
            device,
            batch_size,
            calibrator,
            int(config["step_minutes"]),
            "original",
        )
        for group, features in FEATURE_GROUPS.items():
            source = DeepCacheBatchSource(
                cache_root,
                "cross_year_2023",
                permutation_rows,
                seed,
                permuted_feature_indices=[names.index(name) for name in features],
                permutation_block_days=7,
            )
            rows.extend(
                _six_hour_metrics(
                    model,
                    source,
                    device,
                    batch_size,
                    calibrator,
                    int(config["step_minutes"]),
                    f"conditional_block_permutation_{group}",
                )
            )
        pd.DataFrame(rows).to_csv(permutation_path, index=False)
    else:
        print(f"Reused conditional permutation: {permutation_path}", flush=True)

    ig_path = model_output / "integrated_gradients_2023.csv"
    if overwrite or not ig_path.exists():
        source = DeepCacheBatchSource(cache_root, "cross_year_2023", ig_rows, seed)
        absolute_sum = np.zeros(len(names), dtype=np.float64)
        denominator = 0
        model.eval()
        for batch in source.iter_batches(batch_size, shuffle=False):
            actual = batch["history"].to(device)
            baseline = torch.zeros_like(actual)
            total_gradient = torch.zeros_like(actual)
            station = batch["station_index"].to(device)
            city = batch["city_index"].to(device)
            for alpha in torch.linspace(0.0, 1.0, ig_steps, device=device):
                interpolated = (
                    baseline + alpha * (actual - baseline)
                ).detach().requires_grad_(True)
                with torch.backends.cudnn.flags(enabled=False):
                    eta = model(interpolated, station, city)
                    calibrated_eta = (
                        float(calibrator["log_rate_shift"])
                        + float(calibrator["slope"]) * eta
                    )
                    probability = 1.0 - torch.exp(
                        -torch.exp(torch.clamp(calibrated_eta, max=15.0)).sum(dim=1)
                    )
                    gradient = torch.autograd.grad(probability.sum(), interpolated)[0]
                total_gradient += gradient.detach()
            integrated = (actual - baseline) * total_gradient / ig_steps
            absolute_sum += integrated.abs().sum(dim=(0, 1)).detach().cpu().numpy()
            denominator += int(integrated.shape[0] * integrated.shape[1])
        importance = absolute_sum / max(denominator, 1)
        ig = pd.DataFrame(
            {"feature": names, "mean_absolute_integrated_gradient": importance}
        )
        ig["rank"] = ig["mean_absolute_integrated_gradient"].rank(
            method="min", ascending=False
        ).astype(int)
        ig.sort_values("rank").to_csv(ig_path, index=False)
    else:
        print(f"Reused integrated gradients: {ig_path}", flush=True)

    (model_output / "attribution_manifest.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "seed": seed,
                "split": "2023 after model and controller freeze",
                "conditional_permutation": (
                    "within station-season continuous 7-day block circular shift"
                ),
                "permutation_rows": permutation_rows,
                "integrated_gradients_steps": ig_steps,
                "integrated_gradients_rows": ig_rows,
                "integrated_gradients_probability": "2022-OOF calibrated 6h risk",
                "selection_prohibition": "Attribution outputs must not select a model.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Candidate attribution complete: {model_output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downstream attribution for locked recurrence candidate models."
    )
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument("--models", type=_models, default=_models("rec_full,rec_load,rec_previous"))
    parser.add_argument("--permutation-rows", type=int, default=120000)
    parser.add_argument("--ig-rows", type=int, default=2048)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-root",
        default="results/recurrence_modeling/candidate_attribution",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        run_model(
            model_name,
            config,
            output_root,
            device,
            args.permutation_rows,
            args.ig_rows,
            args.ig_steps,
            args.batch_size,
            args.overwrite,
        )


if __name__ == "__main__":
    main()

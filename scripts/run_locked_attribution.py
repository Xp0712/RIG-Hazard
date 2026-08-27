from __future__ import annotations

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


config = json.loads(Path("configs/rig_hazard_recurrence_modeling.json").read_text(encoding="utf-8"))
root = Path(config["deep_protocol_output_root"])
selection = json.loads((root / "frozen_model_selection.json").read_text(encoding="utf-8"))
model_name = selection["selected_model"]
seed = int(config["training"]["seeds"][0])
checkpoint_path = root / "trained_models" / "checkpoints" / f"{model_name}_seed_{seed}.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, checkpoint = load_deep_checkpoint(checkpoint_path, device)
calibrator = checkpoint["trajectory_calibrator"]
cache_root = resolve_project_path(config["cache_root"])
names = list(checkpoint["sample_contract"]["feature_names"])
groups = {
    "gap_time": ["time_since_last_recurrent_event_hours", "previous_recurrent_event_missing"],
    "event_load_7d_30d": ["events_past_7d", "events_past_30d"],
    "event_order": ["current_event_order"],
    "previous_event_attributes": [
        "previous_event_duration_hours", "previous_event_max_thickness", "previous_event_severity"
    ],
}
output = Path("results/recurrence_modeling/locked_attribution")
output.mkdir(parents=True, exist_ok=True)


def metrics(source: DeepCacheBatchSource, label: str) -> list[dict]:
    arrays = collect_trajectory_arrays(model, source, torch.device(device), 256)
    rows = trajectory_probability_rows(
        label, arrays["eta"], arrays["target"], arrays["risk_mask"], arrays["sample_weight"],
        int(config["step_minutes"]), "calibrated_2022_oof",
        eta_shift=float(calibrator["log_rate_shift"]), eta_slope=float(calibrator["slope"]),
    )
    return [row for row in rows if row["horizon"] == "6h"]


permutation_path = output / "conditional_permutation_2023.csv"
if permutation_path.exists():
    print(f"Reused conditional permutation results: {permutation_path}", flush=True)
else:
    metric_rows = metrics(DeepCacheBatchSource(cache_root, "cross_year_2023", 120000, seed), "original")
    for group, features in groups.items():
        source = DeepCacheBatchSource(
            cache_root, "cross_year_2023", 120000, seed,
            permuted_feature_indices=[names.index(name) for name in features],
            permutation_block_days=7,
        )
        metric_rows.extend(metrics(source, f"conditional_block_permutation_{group}"))
    pd.DataFrame(metric_rows).to_csv(permutation_path, index=False)

source = DeepCacheBatchSource(cache_root, "cross_year_2023", 2048, seed)
ig_sum = np.zeros(len(names), dtype=np.float64)
ig_count = 0
steps = 32
model.eval()
for batch in source.iter_batches(128, shuffle=False):
    actual = batch["history"].to(device)
    baseline = torch.zeros_like(actual)
    total_gradient = torch.zeros_like(actual)
    station = batch["station_index"].to(device)
    city = batch["city_index"].to(device)
    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        interpolated = (baseline + alpha * (actual - baseline)).detach().requires_grad_(True)
        # cuDNN RNNs do not expose backward graphs from eval-mode forwards.
        # The native backend supports deterministic attribution while retaining eval-mode dropout.
        with torch.backends.cudnn.flags(enabled=False):
            eta = model(interpolated, station, city)
            probability = 1.0 - torch.exp(-torch.exp(torch.clamp(eta, max=15.0)).sum(dim=1))
            gradient = torch.autograd.grad(probability.sum(), interpolated)[0]
        total_gradient += gradient.detach()
    integrated = (actual - baseline) * total_gradient / steps
    ig_sum += integrated.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    ig_count += 1
importance = ig_sum / max(ig_count, 1)
ig = pd.DataFrame({"feature": names, "mean_absolute_integrated_gradient": importance})
ig["rank"] = ig["mean_absolute_integrated_gradient"].rank(method="min", ascending=False).astype(int)
ig.sort_values("rank").to_csv(output / "integrated_gradients_2023.csv", index=False)
(output / "attribution_manifest.json").write_text(json.dumps({
    "model": model_name, "seed": seed, "split": "2023 only after model freeze",
    "conditional_permutation": "within station-year icing-season, circular 7-day block shift",
    "integrated_gradients_steps": steps, "integrated_gradients_rows": 2048,
    "selection_prohibition": "attribution outputs must not be used to select the model",
}, indent=2), encoding="utf-8")
print(output)

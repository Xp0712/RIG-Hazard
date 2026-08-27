from __future__ import annotations

import json
from pathlib import Path


BASE = Path("configs/rig_hazard_recurrence_models.json")
OUTPUT = Path("configs/rig_hazard_recurrence_modeling.json")

RECURRENCE = [
    "time_since_last_recurrent_event_hours",
    "current_event_order",
    "events_past_7d",
    "events_past_30d",
    "previous_event_duration_hours",
    "previous_event_max_thickness",
    "previous_event_severity",
    "previous_recurrent_event_missing",
]


def masked(kept: set[str]) -> list[str]:
    return [name for name in RECURRENCE if name not in kept]


config = json.loads(BASE.read_text(encoding="utf-8"))
config["deep_protocol_output_root"] = "results/recurrence_modeling/probability_models"
config["stability_root"] = "results/recurrence_modeling/budget_2022_oof"
config["warning_evaluation"]["bootstrap_samples"] = 5000
config["training"]["batch_size"] = 1024
config["training"]["torch_num_threads"] = 8
config["training"]["early_stopping_validation_samples"] = 20000
config["training"]["development_metric_samples"] = 120000
config["warning_evaluation"]["inference_batch_size"] = 4096
config["performance"] = {
    "allow_tf32": True,
    "cudnn_benchmark": True,
    "training_batch_size": 1024,
    "inference_batch_size": 4096,
}
config["protocol_model_variants"] = {
    "rec_none": {"encoder": "recurrent_dual", "masked_features": RECURRENCE},
    "rec_gap": {
        "encoder": "recurrent_dual",
        "masked_features": masked({"time_since_last_recurrent_event_hours", "previous_recurrent_event_missing"}),
    },
    "rec_load": {
        "encoder": "recurrent_dual",
        "masked_features": masked({"events_past_7d", "events_past_30d"}),
    },
    "rec_order": {"encoder": "recurrent_dual", "masked_features": masked({"current_event_order"})},
    "rec_previous": {
        "encoder": "recurrent_dual",
        "masked_features": masked({
            "previous_event_duration_hours", "previous_event_max_thickness",
            "previous_event_severity", "previous_recurrent_event_missing",
        }),
    },
    "rec_full": {"encoder": "recurrent_dual", "masked_features": []},
    "rec_shuffled": {
        "encoder": "recurrent_dual", "masked_features": [],
        "permuted_features": RECURRENCE, "permutation_block_days": 7,
    },
    "rec_full_uniform": {
        "encoder": "recurrent_dual", "masked_features": [],
        "negative_sampling": "uniform_negative",
    },
}
config["recurrence_evaluation"] = {
    "selection_data": "2022 pooled purged block OOF only",
    "report_only_data": [2023, 2024],
    "primary_metric": "6h calibrated PR-AUC",
    "guardrails": ["6h calibrated log_loss", "6h calibrated ece"],
    "paired_station_bootstrap_replicates": 5000,
    "bootstrap_unit": "station with complete within-station time series",
    "seed_policy": "report mean and standard deviation; seeds are not inferential replicates",
}
OUTPUT.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(OUTPUT)

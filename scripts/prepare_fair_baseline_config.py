from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("configs/rig_hazard_recurrence_modeling.json")
OUTPUT = Path("configs/rig_hazard_fair_baselines.json")
RECURRENCE_FEATURES = [
    "time_since_last_recurrent_event_hours",
    "current_event_order",
    "events_past_7d",
    "events_past_30d",
    "previous_event_duration_hours",
    "previous_event_max_thickness",
    "previous_event_severity",
    "previous_recurrent_event_missing",
]


config = json.loads(SOURCE.read_text(encoding="utf-8"))
models = {
    "fair_gru": "gru",
    "fair_tcn": "tcn",
    "fair_patchtst": "patchtst",
    "fair_timesnet": "timesnet",
    "fair_itransformer": "itransformer",
}
config["deep_protocol_output_root"] = (
    "results/recurrence_modeling/fair_baselines"
)
config["default_local_models"] = list(models)
config["protocol_model_variants"] = {
    name: {"encoder": encoder, "masked_features": RECURRENCE_FEATURES}
    for name, encoder in models.items()
}
config["performance"] = {"allow_tf32": True}
config["fair_baseline_contract"] = {
    "models": models,
    "recurrence_features_masked": RECURRENCE_FEATURES,
    "reference_model": "rec_none from configs/rig_hazard_recurrence_modeling.json",
    "same_cache": True,
    "same_temporal_folds": True,
    "same_training_hyperparameters": True,
    "selection_year": 2022,
    "frozen_evaluation_years": [2023, 2024],
}
OUTPUT.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(OUTPUT)

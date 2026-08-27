from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RECURRENCE_MODELING_ROOT = Path("results/recurrence_modeling")
FAIR_BASELINES_ROOT = RECURRENCE_MODELING_ROOT / "fair_baselines"
REFERENCE_MODELS_ROOT = RECURRENCE_MODELING_ROOT / "probability_models"
OUTPUT_PATH = FAIR_BASELINES_ROOT / "fair_baseline_summary.csv"


def six_hour_calibrated(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["horizon"].eq("6h") & frame["calibration"].str.contains("calibrated")
    ].copy()
    selected["source"] = source
    return selected


oof = pd.concat(
    [
        six_hour_calibrated(
            FAIR_BASELINES_ROOT / "oof_metrics.csv", "fair_baselines"
        ),
        six_hour_calibrated(
            REFERENCE_MODELS_ROOT / "oof_metrics.csv", "rec_none_reference"
        ),
    ],
    ignore_index=True,
)
oof = oof.loc[
    oof["encoder"].isin(
        ["fair_gru", "fair_tcn", "fair_patchtst", "fair_timesnet", "fair_itransformer", "rec_none"]
    )
    & oof["fold"].astype(str).eq("pooled")
].copy()
oof["evaluation"] = "2022_pooled_oof"

locked = pd.concat(
    [
        six_hour_calibrated(
            FAIR_BASELINES_ROOT / "locked_year_metrics.csv", "fair_baselines"
        ),
        six_hour_calibrated(
            REFERENCE_MODELS_ROOT / "locked_year_metrics.csv", "rec_none_reference"
        ),
    ],
    ignore_index=True,
)
locked = locked.loc[
    locked["encoder"].isin(
        ["fair_gru", "fair_tcn", "fair_patchtst", "fair_timesnet", "fair_itransformer", "rec_none"]
    )
].copy()
locked["evaluation"] = locked["split"]
combined = pd.concat([oof, locked], ignore_index=True)

metrics = ["pr_auc", "log_loss", "brier_score", "ece", "roc_auc"]
summary = (
    combined.groupby(["evaluation", "encoder"], as_index=False)[metrics]
    .agg(["mean", "std"])
)
summary.columns = [
    "_".join(str(value) for value in column if str(value))
    for column in summary.columns.to_flat_index()
]
summary.to_csv(OUTPUT_PATH, index=False)
(FAIR_BASELINES_ROOT / "fair_baseline_summary_manifest.json").write_text(
    json.dumps(
        {
            "reference": "rec_none",
            "input_contract": "All models mask the eight recurrence-history variables.",
            "selection": "2022 pooled block OOF only",
            "locked_evaluation": [2023, 2024],
            "seed_reporting": "mean and standard deviation across five seeds",
            "inference_warning": (
                "Seeds describe optimization stability and are not independent inferential units."
            ),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(OUTPUT_PATH)

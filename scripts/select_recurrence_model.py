from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


root = Path("results/recurrence_modeling/probability_models")
metrics = pd.read_csv(root / "oof_metrics.csv")
selected = metrics.loc[
    metrics["stage"].eq("2022_oof_calibration")
    & metrics["horizon"].eq("6h")
    & metrics["calibration"].eq("calibrated_2022_oof_pooled")
].copy()
summary = selected.groupby("encoder", as_index=False).agg(
    pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
    log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"),
    ece_mean=("ece", "mean"), ece_sd=("ece", "std"),
)
control = summary.loc[summary["encoder"].eq("rec_none")].iloc[0]
summary["log_loss_guardrail"] = summary["log_loss_mean"].le(control["log_loss_mean"])
summary["ece_guardrail"] = summary["ece_mean"].le(control["ece_mean"])
eligible = summary.loc[summary["log_loss_guardrail"] & summary["ece_guardrail"]].copy()
if eligible.empty:
    eligible = summary.copy()
winner = eligible.sort_values(
    ["pr_auc_mean", "log_loss_mean", "ece_mean"], ascending=[False, True, True]
).iloc[0]
summary["selected_2022_only"] = summary["encoder"].eq(winner["encoder"]).astype(int)
summary.to_csv(root / "2022_only_model_selection.csv", index=False, encoding="utf-8-sig")
bundle = {
    "selected_model": str(winner["encoder"]),
    "selection_source": "2022 pooled purged block OOF only",
    "primary_rule": "maximum mean calibrated 6h PR-AUC among models not worse than rec_none on mean log-loss and ECE",
    "tie_breakers": ["lower mean log-loss", "lower mean ECE"],
    "forbidden_for_selection": ["2023_cross_year", "2024_final_time"],
}
(root / "frozen_model_selection.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
config_path = Path("configs/rig_hazard_recurrence_modeling.json")
config = json.loads(config_path.read_text(encoding="utf-8"))
config["budget_selection"]["selection_models"] = [bundle["selected_model"]]
config["budget_selection"]["reference_model"] = bundle["selected_model"]
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(bundle["selected_model"])

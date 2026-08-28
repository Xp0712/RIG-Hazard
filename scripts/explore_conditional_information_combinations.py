from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.explore_conditional_information_value import (
    BASE_MODEL,
    _bootstrap_delta,
    _conditions,
    _raw_current_features,
    _seasonal_prior_count,
    _weighted_metrics,
)
from scripts.paired_station_bootstrap import load_ensemble


DIRECT_MODELS = ("rec_gap", "rec_order", "rec_load", "rec_previous", "rec_full")


def _score_combinations(scores: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    combinations: dict[str, np.ndarray] = {}
    definitions: dict[str, Any] = {}
    for model in DIRECT_MODELS:
        combinations[model] = scores[model]
        definitions[model] = {"type": "direct_feature_group_model", "models": [model]}
        for alpha in (0.25, 0.5):
            name = f"blend_none{int((1-alpha)*100)}_{model}{int(alpha*100)}"
            combinations[name] = (1.0 - alpha) * scores[BASE_MODEL] + alpha * scores[model]
            definitions[name] = {
                "type": "fixed_probability_blend",
                "weights": {BASE_MODEL: 1.0 - alpha, model: alpha},
            }
    for partner in ("rec_gap", "rec_order", "rec_load"):
        name = f"blend_previous50_{partner}50"
        combinations[name] = 0.5 * scores["rec_previous"] + 0.5 * scores[partner]
        definitions[name] = {
            "type": "fixed_probability_blend",
            "weights": {"rec_previous": 0.5, partner: 0.5},
        }
    name = "blend_gap_order_load_previous_equal"
    combinations[name] = np.mean(
        np.stack([scores[name] for name in ("rec_gap", "rec_order", "rec_load", "rec_previous")]),
        axis=0,
    )
    definitions[name] = {
        "type": "fixed_probability_blend",
        "weights": {name: 0.25 for name in ("rec_gap", "rec_order", "rec_load", "rec_previous")},
    }
    return combinations, definitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore recurrence-information combinations locally")
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument(
        "--output-root",
        default="results/local_conditional_information_combinations",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--bootstrap-top", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    cache_root = PROJECT_ROOT / config["cache_root"]
    protocol_root = PROJECT_ROOT / config["deep_protocol_output_root"]
    output_root = PROJECT_ROOT / args.output_root
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in config["training"]["seeds"]]

    predictions = {
        model: load_ensemble(protocol_root, 2022, model, seeds, cache_root)
        for model in (BASE_MODEL, *DIRECT_MODELS)
    }
    base = predictions[BASE_MODEL]
    for model in DIRECT_MODELS:
        for key in ("file_id", "row_index", "issue_time_ns", "onset_within_6h", "observed_6h"):
            if not np.array_equal(base[key], predictions[model][key]):
                raise ValueError(f"Unaligned prediction field: {model} {key}")

    observed = base["observed_6h"].astype(bool)
    feature, station = _raw_current_features(cache_root, base["file_id"], base["row_index"])
    issue_time = pd.DatetimeIndex(pd.to_datetime(base["issue_time_ns"], unit="ns"))
    seasonal_prior = _seasonal_prior_count(
        station,
        issue_time,
        PROJECT_ROOT
        / "results/recurrence_modeling/seasonal_recurrence/event_global_seasonal_mapping.csv",
    )
    feature = feature.loc[observed].reset_index(drop=True)
    station = station[observed]
    seasonal_prior = seasonal_prior[observed]
    label = base["onset_within_6h"][observed].astype(np.int8)
    weight = base["sample_weight"][observed].astype(np.float64)
    direct_scores = {
        model: predictions[model]["risk_6h"][observed].astype(np.float64)
        for model in (BASE_MODEL, *DIRECT_MODELS)
    }
    candidate_scores, definitions = _score_combinations(direct_scores)
    conditions, thresholds = _conditions(feature, direct_scores[BASE_MODEL], seasonal_prior)

    metric_rows: list[dict[str, Any]] = []
    for condition, raw_mask in conditions.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.sum() < 100:
            continue
        base_metric = _weighted_metrics(
            label[mask], direct_scores[BASE_MODEL][mask], weight[mask]
        )
        for candidate, score in candidate_scores.items():
            current = _weighted_metrics(label[mask], score[mask], weight[mask])
            metric_rows.append(
                {
                    "condition": condition,
                    "candidate": candidate,
                    "candidate_type": definitions[candidate]["type"],
                    "rows": int(mask.sum()),
                    "coverage": float(mask.mean()),
                    "positive_rows": int(label[mask].sum()),
                    "pr_auc": current.pr_auc,
                    "log_loss": current.log_loss,
                    "brier": current.brier,
                    "ece": current.ece,
                    "delta_pr_auc_vs_rec_none": current.pr_auc - base_metric.pr_auc,
                    "delta_log_loss_vs_rec_none": current.log_loss - base_metric.log_loss,
                    "delta_brier_vs_rec_none": current.brier - base_metric.brier,
                    "delta_ece_vs_rec_none": current.ece - base_metric.ece,
                }
            )
        print(f"Combination metrics: {condition}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics["coherent_probability_gain"] = (
        metrics["delta_log_loss_vs_rec_none"].lt(0)
        & metrics["delta_brier_vs_rec_none"].lt(0)
    ).astype(int)
    metrics["coherent_full_gain"] = (
        metrics["coherent_probability_gain"].eq(1)
        & metrics["delta_pr_auc_vs_rec_none"].gt(0)
    ).astype(int)
    metrics.to_csv(output_root / "combination_condition_metrics_2022_oof.csv", index=False)

    eligible = metrics.loc[
        metrics["coherent_full_gain"].eq(1)
        & metrics["positive_rows"].ge(20)
        & metrics["coverage"].between(0.005, 0.95)
    ].copy()
    eligible = eligible.sort_values(
        ["delta_log_loss_vs_rec_none", "delta_pr_auc_vs_rec_none"],
        ascending=[True, False],
    )
    top = eligible.head(int(args.bootstrap_top)).copy()
    stations = np.unique(station)
    station_lookup = {value: index for index, value in enumerate(stations)}
    station_index = np.asarray([station_lookup[value] for value in station], dtype=np.int16)
    bootstrap_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        mask = np.asarray(conditions[str(row.condition)], dtype=bool)
        intervals = _bootstrap_delta(
            label[mask],
            direct_scores[BASE_MODEL][mask],
            candidate_scores[str(row.candidate)][mask],
            weight[mask],
            station_index[mask],
            stations.size,
            int(args.bootstrap_replicates),
            20260807 + rank * 7919,
        )
        for metric, (low, high) in intervals.items():
            bootstrap_rows.append(
                {
                    "rank": rank,
                    "condition": str(row.condition),
                    "candidate": str(row.candidate),
                    "metric": metric,
                    "ci95_low": low,
                    "ci95_high": high,
                    "replicates": int(args.bootstrap_replicates),
                }
            )
        print(f"Bootstrap candidate {rank}/{top.shape[0]}", flush=True)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output_root / "top_combination_station_bootstrap.csv", index=False)
    top.to_csv(output_root / "top_coherent_combinations.csv", index=False)

    hard = metrics.loc[metrics["condition"].eq("hard_negative")].sort_values(
        "delta_brier_vs_rec_none"
    ).head(12)
    overall = metrics.loc[metrics["condition"].eq("all_rows")].sort_values(
        ["delta_log_loss_vs_rec_none", "delta_pr_auc_vs_rec_none"]
    )
    report = [
        "# Additional-Information Combination Exploration (2022 Five-Fold OOF)",
        "",
        "These results only screen hypotheses for later freezing. Probability blending does not mean that the corresponding joint-feature model has been trained.",
        "",
        "## Leading combinations that improve PR-AUC, Log Loss, and Brier score",
        "",
        top[
            [
                "condition",
                "candidate",
                "rows",
                "positive_rows",
                "delta_pr_auc_vs_rec_none",
                "delta_log_loss_vs_rec_none",
                "delta_brier_vs_rec_none",
                "delta_ece_vs_rec_none",
            ]
        ].to_markdown(index=False)
        if not top.empty
        else "No combination satisfies all criteria.",
        "",
        "## Full sample",
        "",
        overall[
            [
                "candidate",
                "delta_pr_auc_vs_rec_none",
                "delta_log_loss_vs_rec_none",
                "delta_brier_vs_rec_none",
                "delta_ece_vs_rec_none",
            ]
        ].head(15).to_markdown(index=False),
        "",
        "## Hard-negative diagnostic",
        "",
        hard[
            [
                "candidate",
                "delta_log_loss_vs_rec_none",
                "delta_brier_vs_rec_none",
                "delta_ece_vs_rec_none",
            ]
        ].to_markdown(index=False),
    ]
    (output_root / "combination_exploration_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "split": "2022 purged five-fold OOF",
                "base_model": BASE_MODEL,
                "direct_models": list(DIRECT_MODELS),
                "candidate_definitions": definitions,
                "condition_thresholds": thresholds,
                "bootstrap_replicates": int(args.bootstrap_replicates),
                "bootstrap_scope": "top point-estimate coherent gains only",
                "multiple_testing_warning": (
                    "This is hypothesis generation. Freeze at most one or two combinations "
                    "before any 2023/2024 evaluation."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Combination exploration complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()

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
    _conditions,
    _raw_current_features,
    _seasonal_prior_count,
    _weighted_metrics,
)
from scripts.paired_station_bootstrap import load_ensemble


MODELS = ("rec_gap", "rec_order", "rec_load", "rec_previous", "rec_full")


def _weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(values * weight) / max(float(weight.sum()), 1e-12))


def _brier_decomposition(
    label: np.ndarray,
    base: np.ndarray,
    extra: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float | str]:
    difference = extra - base
    movement_cost = _weighted_mean(np.square(difference), weight)
    residual_correction = 2.0 * _weighted_mean(difference * (base - label), weight)
    delta = movement_cost + residual_correction
    if delta < 0:
        mechanism = "net_residual_correction"
    elif residual_correction < 0:
        mechanism = "directionally_useful_but_movement_cost_dominates"
    else:
        mechanism = "wrong_direction_or_redundant_shift"
    positive = label == 1
    negative = ~positive
    return {
        "brier_delta_exact": delta,
        "movement_cost": movement_cost,
        "residual_correction": residual_correction,
        "mean_score_shift_positive": (
            _weighted_mean(difference[positive], weight[positive]) if positive.any() else float("nan")
        ),
        "mean_score_shift_negative": (
            _weighted_mean(difference[negative], weight[negative]) if negative.any() else float("nan")
        ),
        "mean_absolute_score_shift": _weighted_mean(np.abs(difference), weight),
        "failure_mechanism": mechanism,
    }


def _weighted_correlation(first: np.ndarray, second: np.ndarray, weight: np.ndarray) -> float:
    total = float(weight.sum())
    if total <= 0:
        return float("nan")
    first_mean = float(np.sum(weight * first) / total)
    second_mean = float(np.sum(weight * second) / total)
    covariance = float(np.sum(weight * (first - first_mean) * (second - second_mean)) / total)
    first_var = float(np.sum(weight * np.square(first - first_mean)) / total)
    second_var = float(np.sum(weight * np.square(second - second_mean)) / total)
    denominator = np.sqrt(first_var * second_var)
    return covariance / denominator if denominator > 0 else float("nan")


def _fine_gap_conditions(feature: pd.DataFrame) -> dict[str, np.ndarray]:
    gap = feature["time_since_last_recurrent_event_hours"].to_numpy(dtype=float)
    available = feature["previous_recurrent_event_missing"].to_numpy(dtype=float) < 0.5
    return {
        "gap_missing": ~available,
        "gap_0_6h": available & (gap <= 6),
        "gap_6_24h": available & (gap > 6) & (gap <= 24),
        "gap_1_3d": available & (gap > 24) & (gap <= 72),
        "gap_3_7d": available & (gap > 72) & (gap <= 168),
        "gap_7_14d": available & (gap > 168) & (gap <= 336),
        "gap_14_30d": available & (gap > 336) & (gap <= 720),
        "gap_30_90d": available & (gap > 720) & (gap <= 2160),
        "gap_gt_90d": available & (gap > 2160),
    }


def _station_heterogeneity(
    condition: str,
    model: str,
    mask: np.ndarray,
    label: np.ndarray,
    base: np.ndarray,
    extra: np.ndarray,
    weight: np.ndarray,
    station: np.ndarray,
) -> dict[str, Any]:
    base = np.clip(base, 1e-12, 1 - 1e-12)
    extra = np.clip(extra, 1e-12, 1 - 1e-12)
    base_loss = -(label * np.log(base) + (1 - label) * np.log1p(-base))
    extra_loss = -(label * np.log(extra) + (1 - label) * np.log1p(-extra))
    rows = []
    positive_by_station = []
    for code in np.unique(station[mask]):
        selected = mask & (station == code)
        current_weight = weight[selected]
        if current_weight.sum() <= 0:
            continue
        rows.append(
            (
                _weighted_mean(extra_loss[selected] - base_loss[selected], current_weight),
                _weighted_mean(
                    np.square(extra[selected] - label[selected])
                    - np.square(base[selected] - label[selected]),
                    current_weight,
                ),
            )
        )
        positive_by_station.append(int(label[selected].sum()))
    values = np.asarray(rows, dtype=float)
    positive = np.asarray(positive_by_station, dtype=float)
    positive_total = max(float(positive.sum()), 1.0)
    concentration = float(np.square(positive / positive_total).sum())
    return {
        "condition": condition,
        "model": model,
        "stations": int(values.shape[0]),
        "positive_stations": int((positive > 0).sum()),
        "effective_positive_station_count": float(1.0 / concentration) if concentration > 0 else 0.0,
        "stations_improved_log_loss_fraction": float((values[:, 0] < 0).mean()),
        "stations_improved_brier_fraction": float((values[:, 1] < 0).mean()),
        "station_median_delta_log_loss": float(np.median(values[:, 0])),
        "station_q25_delta_log_loss": float(np.quantile(values[:, 0], 0.25)),
        "station_q75_delta_log_loss": float(np.quantile(values[:, 0], 0.75)),
        "station_median_delta_brier": float(np.median(values[:, 1])),
    }


def _attribution_diagnostics(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_by_model = {
        "rec_load": "conditional_block_permutation_event_load_7d_30d",
        "rec_previous": "conditional_block_permutation_previous_event_attributes",
        "rec_full": None,
    }
    for model, selected_group in group_by_model.items():
        path = root / model / "conditional_permutation_2023.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        original = frame.loc[frame["model_name"].eq("original")].iloc[0]
        groups = (
            frame.loc[~frame["model_name"].eq("original")]
            if selected_group is None
            else frame.loc[frame["model_name"].eq(selected_group)]
        )
        for row in groups.itertuples(index=False):
            rows.append(
                {
                    "model": model,
                    "permutation": row.model_name,
                    "delta_pr_auc_permuted_minus_original": float(row.pr_auc - original.pr_auc),
                    "delta_log_loss_permuted_minus_original": float(row.log_loss - original.log_loss),
                    "delta_brier_permuted_minus_original": float(row.brier_score - original.brier_score),
                    "interpretation": (
                        "model_uses_feature_group"
                        if row.pr_auc < original.pr_auc and row.log_loss > original.log_loss
                        else "feature_group_mixed_or_harmful"
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why recurrence information fails")
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument(
        "--output-root",
        default="results/local_recurrence_failure_diagnostics",
    )
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
    prediction = {
        model: load_ensemble(protocol_root, 2022, model, seeds, cache_root)
        for model in (BASE_MODEL, *MODELS)
    }
    base_prediction = prediction[BASE_MODEL]
    observed = base_prediction["observed_6h"].astype(bool)
    feature, station = _raw_current_features(
        cache_root, base_prediction["file_id"], base_prediction["row_index"]
    )
    issue_time = pd.DatetimeIndex(pd.to_datetime(base_prediction["issue_time_ns"], unit="ns"))
    seasonal_prior = _seasonal_prior_count(
        station,
        issue_time,
        PROJECT_ROOT
        / "results/recurrence_modeling/seasonal_recurrence/event_global_seasonal_mapping.csv",
    )
    feature = feature.loc[observed].reset_index(drop=True)
    station = station[observed]
    seasonal_prior = seasonal_prior[observed]
    label = base_prediction["onset_within_6h"][observed].astype(np.int8)
    weight = base_prediction["sample_weight"][observed].astype(np.float64)
    score = {
        model: prediction[model]["risk_6h"][observed].astype(np.float64)
        for model in (BASE_MODEL, *MODELS)
    }
    broad_conditions, thresholds = _conditions(feature, score[BASE_MODEL], seasonal_prior)
    fine_conditions = _fine_gap_conditions(feature)
    selected_names = (
        "all_rows",
        "season_first_risk",
        "season_recurrent_risk",
        "event_load_7d_positive",
        "previous_severity_high",
        "previous_duration_long",
        "hard_negative",
        "gap_1_7d_and_fog_low_visibility",
        "gap_1_7d_and_previous_long",
        "gap_1_7d_and_previous_severe",
    )
    conditions = {name: broad_conditions[name] for name in selected_names}
    conditions.update(fine_conditions)

    decomposition_rows: list[dict[str, Any]] = []
    heterogeneity_rows: list[dict[str, Any]] = []
    for condition, raw_mask in conditions.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.sum() < 100:
            continue
        base_metric = _weighted_metrics(label[mask], score[BASE_MODEL][mask], weight[mask])
        for model in MODELS:
            current = _weighted_metrics(label[mask], score[model][mask], weight[mask])
            decomposition_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "rows": int(mask.sum()),
                    "positive_rows": int(label[mask].sum()),
                    "positive_rate": float(label[mask].mean()),
                    "score_correlation_with_rec_none": _weighted_correlation(
                        score[BASE_MODEL][mask], score[model][mask], weight[mask]
                    ),
                    "delta_pr_auc": current.pr_auc - base_metric.pr_auc,
                    "delta_log_loss": current.log_loss - base_metric.log_loss,
                    "delta_brier": current.brier - base_metric.brier,
                    "delta_ece": current.ece - base_metric.ece,
                    **_brier_decomposition(
                        label[mask],
                        score[BASE_MODEL][mask],
                        score[model][mask],
                        weight[mask],
                    ),
                }
            )
            heterogeneity_rows.append(
                _station_heterogeneity(
                    condition,
                    model,
                    mask,
                    label,
                    score[BASE_MODEL],
                    score[model],
                    weight,
                    station,
                )
            )
        print(f"Failure diagnostics: {condition}", flush=True)
    decomposition = pd.DataFrame(decomposition_rows)
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    attribution = _attribution_diagnostics(
        PROJECT_ROOT / "results/recurrence_modeling/candidate_attribution"
    )
    decomposition.to_csv(output_root / "brier_failure_decomposition_2022_oof.csv", index=False)
    heterogeneity.to_csv(output_root / "station_heterogeneity_2022_oof.csv", index=False)
    attribution.to_csv(output_root / "existing_2023_attribution_diagnostics.csv", index=False)

    overall = decomposition.loc[decomposition["condition"].eq("all_rows")].sort_values(
        "delta_brier"
    )
    fine_gap = decomposition.loc[
        decomposition["condition"].str.startswith("gap_")
        & ~decomposition["condition"].str.contains("and_")
    ].sort_values(["model", "condition"])
    station_overall = heterogeneity.loc[
        heterogeneity["condition"].eq("all_rows")
    ].sort_values("stations_improved_brier_fraction", ascending=False)
    report = [
        "# Why Most Recurrence Information Does Not Improve Model Performance",
        "",
        "The Brier-score difference is decomposed exactly into probability-shift cost plus correction of the `rec_none` residual.",
        "Shift cost is always nonnegative; additional information has a net benefit only when the residual correction is sufficiently negative.",
        "",
        "## Full-sample decomposition",
        "",
        overall[
            [
                "model",
                "delta_pr_auc",
                "delta_log_loss",
                "delta_brier",
                "movement_cost",
                "residual_correction",
                "failure_mechanism",
            ]
        ].to_markdown(index=False),
        "",
        "## Fine-grained gap windows",
        "",
        fine_gap[
            [
                "condition",
                "model",
                "rows",
                "positive_rows",
                "delta_pr_auc",
                "delta_log_loss",
                "delta_brier",
                "failure_mechanism",
            ]
        ].to_markdown(index=False),
        "",
        "## Station heterogeneity",
        "",
        station_overall[
            [
                "model",
                "positive_stations",
                "effective_positive_station_count",
                "stations_improved_log_loss_fraction",
                "stations_improved_brier_fraction",
                "station_median_delta_log_loss",
            ]
        ].to_markdown(index=False),
        "",
        "## Conditional-permutation cross-evidence for 2023",
        "",
        attribution.to_markdown(index=False) if not attribution.empty else "No attribution results are available.",
    ]
    (output_root / "failure_mechanism_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "split": "2022 purged five-fold OOF plus existing 2023 attribution",
                "models": list(MODELS),
                "brier_identity": "delta = E[(p_extra-p_base)^2] + 2E[(p_extra-p_base)(p_base-y)]",
                "condition_thresholds": thresholds,
                "interpretation_scope": "mechanism diagnosis, not causal proof",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Recurrence failure diagnostics complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()

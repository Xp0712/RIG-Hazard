from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.paired_station_bootstrap import (
    ECE_EDGES,
    average_precision_batch,
    load_ensemble,
    prepare_average_precision,
)


BASE_MODEL = "rec_none"
EXTRA_MODELS = ("rec_load", "rec_previous")
FEATURES = (
    "time_since_last_recurrent_event_hours",
    "current_event_order",
    "events_past_7d",
    "events_past_30d",
    "previous_event_duration_hours",
    "previous_event_max_thickness",
    "previous_event_severity",
    "previous_recurrent_event_missing",
)
TARGET_FLAGS = (
    "hard_negative_6h",
    "exposure_e1_cold_humid",
    "exposure_e2_fog_low_visibility",
    "exposure_e3_any",
)


@dataclass(frozen=True)
class MetricRow:
    pr_auc: float
    log_loss: float
    brier: float
    ece: float


def _weighted_metrics(label: np.ndarray, score: np.ndarray, weight: np.ndarray) -> MetricRow:
    probability = np.clip(np.asarray(score, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    label = np.asarray(label, dtype=np.int8)
    weight = np.asarray(weight, dtype=np.float64)
    total = float(weight.sum())
    if total <= 0:
        return MetricRow(float("nan"), float("nan"), float("nan"), float("nan"))
    log_loss = float(
        np.sum(
            weight
            * (-(label * np.log(probability) + (1 - label) * np.log1p(-probability)))
        )
        / total
    )
    brier = float(np.sum(weight * np.square(probability - label)) / total)
    bins = np.clip(np.digitize(probability, ECE_EDGES) - 1, 0, ECE_EDGES.size - 2)
    ece = 0.0
    for bin_index in range(ECE_EDGES.size - 1):
        selected = bins == bin_index
        bin_weight = float(weight[selected].sum())
        if bin_weight <= 0:
            continue
        observed = float(np.sum(weight[selected] * label[selected]) / bin_weight)
        predicted = float(np.sum(weight[selected] * probability[selected]) / bin_weight)
        ece += bin_weight * abs(observed - predicted) / total
    pr_auc = (
        float(average_precision_score(label, probability, sample_weight=weight))
        if np.unique(label).size >= 2
        else float("nan")
    )
    return MetricRow(
        pr_auc,
        log_loss,
        brier,
        float(ece),
    )


def _raw_current_features(
    cache_root: Path,
    file_id: np.ndarray,
    row_index: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    contract = json.loads((cache_root / "sample_contract.json").read_text(encoding="utf-8"))
    transformer = json.loads(
        (cache_root / "feature_transformer.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
    file_rows = {int(row["file_id"]): row for row in manifest["files"]}
    feature_names = list(contract["feature_names"])
    indices = [feature_names.index(name) for name in FEATURES]
    continuous = list(transformer["continuous_features"])
    center = dict(zip(continuous, transformer["means"]))
    scale = dict(zip(continuous, transformer["stds"]))
    output = np.full((file_id.size, len(FEATURES)), np.nan, dtype=np.float32)
    flags = np.zeros((file_id.size, len(TARGET_FLAGS)), dtype=np.int8)
    stations = np.empty(file_id.size, dtype=object)

    def cache_file(metadata: dict[str, Any], kind: str) -> Path:
        declared = cache_root / metadata[f"{kind}_path"]
        if declared.exists():
            return declared
        folder = cache_root / kind.replace("feature", "features").replace("target", "targets") / str(metadata["year"])
        matches = list(folder.glob(f"{metadata['station_code']}_*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Cannot uniquely resolve local {kind} cache for file_id={metadata['file_id']}: {matches}"
            )
        return matches[0]

    for current_file in np.unique(file_id):
        positions = np.flatnonzero(file_id == current_file)
        metadata = file_rows[int(current_file)]
        rows = row_index[positions].astype(np.int64)
        features = np.load(cache_file(metadata, "feature"), mmap_mode="r")
        values = np.asarray(features[np.ix_(rows, indices)], dtype=np.float32)
        for column_index, name in enumerate(FEATURES):
            if name in center:
                values[:, column_index] = (
                    values[:, column_index] * float(scale[name]) + float(center[name])
                )
        output[positions] = values
        with np.load(cache_file(metadata, "target"), allow_pickle=False) as target:
            for flag_index, name in enumerate(TARGET_FLAGS):
                flags[positions, flag_index] = target[name][rows].astype(np.int8)
        stations[positions] = str(metadata["station_code"])
    frame = pd.DataFrame(output, columns=FEATURES)
    for index, name in enumerate(TARGET_FLAGS):
        frame[name] = flags[:, index]
    return frame, stations.astype(str)


def _seasonal_prior_count(
    station: np.ndarray,
    issue_time: pd.DatetimeIndex,
    events_path: Path,
) -> np.ndarray:
    events = pd.read_csv(events_path, low_memory=False)
    events = events.loc[
        pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).eq(1)
    ].copy()
    events["onset_time"] = pd.to_datetime(events["onset_time"], errors="coerce")
    events = events.dropna(subset=["onset_time"])
    event_lookup = {
        (str(code), int(season)): group["onset_time"].sort_values().astype("int64").to_numpy()
        for (code, season), group in events.groupby(
            ["station_code", "icing_season_start_year"], sort=False
        )
    }
    season = np.where(issue_time.month >= 11, issue_time.year, issue_time.year - 1)
    result = np.zeros(issue_time.size, dtype=np.int16)
    for code in np.unique(station):
        station_positions = np.flatnonzero(station == code)
        for current_season in np.unique(season[station_positions]):
            positions = station_positions[season[station_positions] == current_season]
            onsets = event_lookup.get((str(code), int(current_season)))
            if onsets is not None and onsets.size:
                result[positions] = np.searchsorted(
                    onsets, issue_time.asi8[positions], side="right"
                ).astype(np.int16)
    return result


def _conditions(
    feature: pd.DataFrame,
    base_score: np.ndarray,
    seasonal_prior: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    finite_base = base_score[np.isfinite(base_score)]
    base_quantiles = {
        f"base_q{int(q * 100)}": float(np.quantile(finite_base, q))
        for q in (0.5, 0.9, 0.99)
    }
    previous_available = feature["previous_recurrent_event_missing"].to_numpy() < 0.5
    gap = feature["time_since_last_recurrent_event_hours"].to_numpy(dtype=float)
    severity = feature["previous_event_severity"].to_numpy(dtype=float)
    duration = feature["previous_event_duration_hours"].to_numpy(dtype=float)
    thickness = feature["previous_event_max_thickness"].to_numpy(dtype=float)
    valid_severity = severity[previous_available & np.isfinite(severity)]
    valid_duration = duration[previous_available & np.isfinite(duration)]
    valid_thickness = thickness[previous_available & np.isfinite(thickness)]
    thresholds = {
        **base_quantiles,
        "previous_severity_q75": float(np.quantile(valid_severity, 0.75)),
        "previous_duration_q75": float(np.quantile(valid_duration, 0.75)),
        "previous_thickness_q75": float(np.quantile(valid_thickness, 0.75)),
    }
    q50, q90, q99 = (base_quantiles[f"base_q{value}"] for value in (50, 90, 99))
    recurrent = seasonal_prior >= 1
    conditions = {
        "all_rows": np.ones(base_score.size, dtype=bool),
        "season_first_risk": seasonal_prior == 0,
        "season_recurrent_risk": recurrent,
        "season_after_exactly_one": seasonal_prior == 1,
        "season_after_two_plus": seasonal_prior >= 2,
        "previous_event_available": previous_available,
        "gap_le_24h": previous_available & (gap <= 24),
        "gap_1_to_7d": previous_available & (gap > 24) & (gap <= 168),
        "gap_7_to_30d": previous_available & (gap > 168) & (gap <= 720),
        "gap_gt_30d": previous_available & (gap > 720),
        "event_load_7d_positive": feature["events_past_7d"].to_numpy(dtype=float) > 0,
        "event_load_30d_positive": feature["events_past_30d"].to_numpy(dtype=float) > 0,
        "previous_severity_high": previous_available
        & (severity >= thresholds["previous_severity_q75"]),
        "previous_duration_long": previous_available
        & (duration >= thresholds["previous_duration_q75"]),
        "previous_thickness_high": previous_available
        & (thickness >= thresholds["previous_thickness_q75"]),
        "hard_negative": feature["hard_negative_6h"].to_numpy(dtype=int) == 1,
        "cold_humid_exposure": feature["exposure_e1_cold_humid"].to_numpy(dtype=int) == 1,
        "fog_low_visibility_exposure": feature[
            "exposure_e2_fog_low_visibility"
        ].to_numpy(dtype=int)
        == 1,
        "any_condensation_exposure": feature["exposure_e3_any"].to_numpy(dtype=int) == 1,
        "base_risk_low_q00_q50": base_score <= q50,
        "base_risk_middle_q50_q90": (base_score > q50) & (base_score <= q90),
        "base_risk_high_q90_q99": (base_score > q90) & (base_score <= q99),
        "base_risk_extreme_q99_q100": base_score > q99,
        "recurrent_and_base_middle": recurrent & (base_score > q50) & (base_score <= q90),
        "recurrent_and_base_high": recurrent & (base_score > q90) & (base_score <= q99),
        "recent_7d_and_hard_negative": (gap <= 168)
        & previous_available
        & (feature["hard_negative_6h"].to_numpy(dtype=int) == 1),
        "gap_1_7d_and_previous_severe": previous_available
        & (gap > 24)
        & (gap <= 168)
        & (severity >= thresholds["previous_severity_q75"]),
        "gap_1_7d_and_previous_long": previous_available
        & (gap > 24)
        & (gap <= 168)
        & (duration >= thresholds["previous_duration_q75"]),
        "gap_1_7d_and_previous_thick": previous_available
        & (gap > 24)
        & (gap <= 168)
        & (thickness >= thresholds["previous_thickness_q75"]),
        "gap_1_7d_and_season_second": previous_available
        & (gap > 24)
        & (gap <= 168)
        & (seasonal_prior == 1),
        "gap_1_7d_and_season_third_plus": previous_available
        & (gap > 24)
        & (gap <= 168)
        & (seasonal_prior >= 2),
        "gap_1_7d_and_cold_humid": previous_available
        & (gap > 24)
        & (gap <= 168)
        & (feature["exposure_e1_cold_humid"].to_numpy(dtype=int) == 1),
        "gap_1_7d_and_fog_low_visibility": previous_available
        & (gap > 24)
        & (gap <= 168)
        & (feature["exposure_e2_fog_low_visibility"].to_numpy(dtype=int) == 1),
        "gap_1_7d_and_base_high": previous_available
        & (gap > 24)
        & (gap <= 168)
        & (base_score > q90),
    }
    return conditions, thresholds


def _bootstrap_delta(
    label: np.ndarray,
    base_score: np.ndarray,
    extra_score: np.ndarray,
    weight: np.ndarray,
    station_index: np.ndarray,
    station_count: int,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        station_count,
        np.full(station_count, 1.0 / station_count),
        size=replicates,
    ).astype(np.int16)
    base = np.clip(base_score, 1e-12, 1.0 - 1e-12)
    extra = np.clip(extra_score, 1e-12, 1.0 - 1e-12)
    base_log = -(label * np.log(base) + (1 - label) * np.log1p(-base))
    extra_log = -(label * np.log(extra) + (1 - label) * np.log1p(-extra))
    denominator = np.bincount(station_index, weights=weight, minlength=station_count)
    differences = {
        "delta_log_loss": np.bincount(
            station_index, weights=weight * (extra_log - base_log), minlength=station_count
        ),
        "delta_brier": np.bincount(
            station_index,
            weights=weight * (np.square(extra - label) - np.square(base - label)),
            minlength=station_count,
        ),
    }
    total = counts @ denominator
    result: dict[str, tuple[float, float]] = {}
    for metric, station_sum in differences.items():
        values = np.divide(
            counts @ station_sum,
            total,
            out=np.full(replicates, np.nan),
            where=total > 0,
        )
        result[metric] = (
            float(np.nanquantile(values, 0.025)),
            float(np.nanquantile(values, 0.975)),
        )
    if np.unique(label).size >= 2 and int(label.sum()) >= 10:
        prepared_base = prepare_average_precision(label, base, weight, station_index)
        prepared_extra = prepare_average_precision(label, extra, weight, station_index)
        delta_parts = []
        for start in range(0, replicates, 16):
            selected = counts[start : start + 16]
            delta_parts.append(
                average_precision_batch(selected, prepared_extra)
                - average_precision_batch(selected, prepared_base)
            )
        values = np.concatenate(delta_parts)
        result["delta_pr_auc"] = (
            float(np.nanquantile(values, 0.025)),
            float(np.nanquantile(values, 0.975)),
        )
    else:
        result["delta_pr_auc"] = (float("nan"), float("nan"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore when recurrence information adds value")
    parser.add_argument("--config", default="configs/rig_hazard_recurrence_modeling.json")
    parser.add_argument(
        "--output-root",
        default="results/local_conditional_information_exploration",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
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
        for model in (BASE_MODEL, *EXTRA_MODELS)
    }
    base = predictions[BASE_MODEL]
    for model in EXTRA_MODELS:
        for key in ("file_id", "row_index", "issue_time_ns", "onset_within_6h", "observed_6h"):
            if not np.array_equal(base[key], predictions[model][key]):
                raise ValueError(f"Unaligned prediction field: {model} {key}")
    observed = base["observed_6h"].astype(bool)
    feature, station = _raw_current_features(
        cache_root, base["file_id"], base["row_index"]
    )
    issue_time = pd.DatetimeIndex(pd.to_datetime(base["issue_time_ns"], unit="ns"))
    seasonal_prior = _seasonal_prior_count(
        station,
        issue_time,
        PROJECT_ROOT
        / "results/recurrence_modeling/seasonal_recurrence/event_global_seasonal_mapping.csv",
    )
    feature = feature.loc[observed].reset_index(drop=True)
    station = station[observed]
    issue_time = issue_time[observed]
    seasonal_prior = seasonal_prior[observed]
    label = base["onset_within_6h"][observed].astype(np.int8)
    weight = base["sample_weight"][observed].astype(np.float64)
    scores = {
        model: predictions[model]["risk_6h"][observed].astype(np.float64)
        for model in (BASE_MODEL, *EXTRA_MODELS)
    }
    conditions, thresholds = _conditions(feature, scores[BASE_MODEL], seasonal_prior)
    stations = np.unique(station)
    station_lookup = {value: index for index, value in enumerate(stations)}
    station_index = np.asarray([station_lookup[value] for value in station], dtype=np.int16)

    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for condition_index, (condition, raw_mask) in enumerate(conditions.items()):
        mask = np.asarray(raw_mask, dtype=bool)
        positive_rows = int(label[mask].sum())
        negative_rows = int(mask.sum() - positive_rows)
        if mask.sum() < 100:
            continue
        base_metric = _weighted_metrics(label[mask], scores[BASE_MODEL][mask], weight[mask])
        metric_rows.append(
            {
                "condition": condition,
                "model": BASE_MODEL,
                "rows": int(mask.sum()),
                "coverage": float(mask.mean()),
                "positive_rows": positive_rows,
                "weighted_positive_rate": float(
                    np.sum(weight[mask] * label[mask]) / np.sum(weight[mask])
                ),
                **base_metric.__dict__,
                "delta_pr_auc_vs_rec_none": 0.0,
                "delta_log_loss_vs_rec_none": 0.0,
                "delta_brier_vs_rec_none": 0.0,
                "delta_ece_vs_rec_none": 0.0,
            }
        )
        local_station = station_index[mask]
        for model_index, model in enumerate(EXTRA_MODELS):
            current = _weighted_metrics(label[mask], scores[model][mask], weight[mask])
            metric_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "rows": int(mask.sum()),
                    "coverage": float(mask.mean()),
                    "positive_rows": positive_rows,
                    "weighted_positive_rate": float(
                        np.sum(weight[mask] * label[mask]) / np.sum(weight[mask])
                    ),
                    **current.__dict__,
                    "delta_pr_auc_vs_rec_none": current.pr_auc - base_metric.pr_auc,
                    "delta_log_loss_vs_rec_none": current.log_loss - base_metric.log_loss,
                    "delta_brier_vs_rec_none": current.brier - base_metric.brier,
                    "delta_ece_vs_rec_none": current.ece - base_metric.ece,
                }
            )
            intervals = _bootstrap_delta(
                label[mask],
                scores[BASE_MODEL][mask],
                scores[model][mask],
                weight[mask],
                local_station,
                stations.size,
                int(args.bootstrap_replicates),
                20260807 + condition_index * 101 + model_index * 10007,
            )
            for metric, (low, high) in intervals.items():
                bootstrap_rows.append(
                    {
                        "condition": condition,
                        "comparison": f"{model}_minus_{BASE_MODEL}",
                        "metric": metric,
                        "ci95_low": low,
                        "ci95_high": high,
                        "replicates": int(args.bootstrap_replicates),
                    }
                )
        print(f"Conditional metrics: {condition}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    metrics.to_csv(output_root / "conditional_probability_metrics_2022_oof.csv", index=False)
    bootstrap.to_csv(output_root / "paired_station_bootstrap_2022_oof.csv", index=False)

    baseline_all = _weighted_metrics(label, scores[BASE_MODEL], weight)
    gate_rows: list[dict[str, Any]] = []
    for condition, raw_mask in conditions.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.mean() < 0.005 or mask.mean() > 0.95 or int(label[mask].sum()) < 10:
            continue
        for model in EXTRA_MODELS:
            for alpha in (0.25, 0.5, 0.75, 1.0):
                gated = scores[BASE_MODEL].copy()
                gated[mask] = (
                    (1.0 - alpha) * scores[BASE_MODEL][mask]
                    + alpha * scores[model][mask]
                )
                current = _weighted_metrics(label, gated, weight)
                gate_rows.append(
                    {
                        "condition": condition,
                        "extra_model": model,
                        "blend_alpha": alpha,
                        "condition_coverage": float(mask.mean()),
                        "condition_positive_rows": int(label[mask].sum()),
                        **current.__dict__,
                        "delta_pr_auc_vs_rec_none": current.pr_auc - baseline_all.pr_auc,
                        "delta_log_loss_vs_rec_none": current.log_loss - baseline_all.log_loss,
                        "delta_brier_vs_rec_none": current.brier - baseline_all.brier,
                        "delta_ece_vs_rec_none": current.ece - baseline_all.ece,
                    }
                )
    gates = pd.DataFrame(gate_rows)
    gates["eligible_exploratory_gate"] = (
        gates["delta_log_loss_vs_rec_none"].lt(0)
        & gates["delta_brier_vs_rec_none"].lt(0)
        & gates["delta_pr_auc_vs_rec_none"].ge(-1e-4)
    ).astype(int)
    gates.to_csv(output_root / "conditional_gate_candidates_2022_oof.csv", index=False)
    selected = (
        gates.loc[gates["eligible_exploratory_gate"].eq(1)]
        .sort_values(
            ["extra_model", "delta_log_loss_vs_rec_none", "delta_pr_auc_vs_rec_none"],
            ascending=[True, True, False],
        )
        .groupby("extra_model", as_index=False)
        .head(1)
    )
    selected.to_csv(output_root / "selected_exploratory_gates.csv", index=False)

    extra = metrics.loc[metrics["model"].isin(EXTRA_MODELS)].copy()
    top = extra.sort_values(
        ["delta_log_loss_vs_rec_none", "delta_pr_auc_vs_rec_none"],
        ascending=[True, False],
    ).head(12)
    report = [
        "# 条件增量信息探索（2022年五折OOF）",
        "",
        "本结果用于提出假设，不是2023/2024冻结确认结论。普通accuracy不适合极稀有事件，",
        "因此以PR-AUC、LogLoss、Brier和ECE为主。",
        "",
        "## 条件子群中LogLoss改善最大的组合",
        "",
        top[
            [
                "condition",
                "model",
                "rows",
                "positive_rows",
                "delta_pr_auc_vs_rec_none",
                "delta_log_loss_vs_rec_none",
                "delta_brier_vs_rec_none",
                "delta_ece_vs_rec_none",
            ]
        ].to_markdown(index=False),
        "",
        "## 探索性条件门控",
        "",
        selected[
            [
                "condition",
                "extra_model",
                "blend_alpha",
                "condition_coverage",
                "delta_pr_auc_vs_rec_none",
                "delta_log_loss_vs_rec_none",
                "delta_brier_vs_rec_none",
                "delta_ece_vs_rec_none",
            ]
        ].to_markdown(index=False)
        if not selected.empty
        else "没有候选同时改善LogLoss与Brier且不降低PR-AUC。",
    ]
    (output_root / "conditional_information_exploration_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split": "2022 purged five-fold OOF",
        "base_model": BASE_MODEL,
        "extra_models": list(EXTRA_MODELS),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "cluster_unit": "station",
        "thresholds_fitted_on": "2022 OOF exploration only",
        "thresholds": thresholds,
        "diagnostic_only_conditions": [
            "hard_negative",
            "recent_7d_and_hard_negative"
        ],
        "diagnostic_only_note": (
            "Hard-negative membership uses the absence of a future event and cannot be a "
            "prospective gate; it is retained only to diagnose false-alarm behavior."
        ),
        "interpretation": (
            "Hypothesis generation only. Any condition or gate must be frozen and tested "
            "on 2023/2024 before being claimed."
        ),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Conditional information exploration complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .baseline_models import HazardRateCalibrator, PlattCalibrator, WeightedBinaryGLM, cloglog_probability, logit_probability
from .config import PROJECT_ROOT, resolve_project_path
from .preprocessing import prepare_output_root, write_json


@dataclass
class FeatureTransformer:
    continuous_features: list[str]
    binary_features: list[str]
    means: np.ndarray
    stds: np.ndarray
    clip_z: float

    @property
    def feature_names(self) -> list[str]:
        return [*self.continuous_features, *self.binary_features]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        continuous = frame[self.continuous_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        missing = ~np.isfinite(continuous)
        if missing.any():
            continuous[missing] = np.broadcast_to(self.means, continuous.shape)[missing]
        continuous = (continuous - self.means) / self.stds
        continuous = np.clip(continuous, -self.clip_z, self.clip_z)
        binary = frame[self.binary_features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        return np.concatenate([continuous, binary], axis=1).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "continuous_features": self.continuous_features,
            "binary_features": self.binary_features,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "clip_z": self.clip_z,
            "feature_names": self.feature_names,
        }


def load_feature_transformer(preprocessed_root: Path, scheme: str, clip_z: float) -> FeatureTransformer:
    contract = json.loads((preprocessed_root / "feature_contract.json").read_text(encoding="utf-8"))
    stats_name = "normalization_development.json" if scheme == "development" else "normalization_final.json"
    stats = json.loads((preprocessed_root / stats_name).read_text(encoding="utf-8"))
    continuous = list(contract["continuous_dynamic_features"])
    binary = list(contract["binary_dynamic_features"])
    means = np.asarray([float(stats[name]["mean"]) for name in continuous], dtype=np.float64)
    stds = np.asarray([max(float(stats[name]["std"]), 1e-8) for name in continuous], dtype=np.float64)
    return FeatureTransformer(continuous, binary, means, stds, clip_z)


def read_timeline(path: Path, usecols: list[str]) -> pd.DataFrame:
    if path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    raise ValueError(f"Unsupported timeline file: {path}")


def timeline_files(preprocessed_root: Path, years: Iterable[int]) -> list[Path]:
    paths: list[Path] = []
    for year in sorted(set(int(value) for value in years)):
        year_root = preprocessed_root / "timelines" / str(year)
        if year_root.exists():
            paths.extend(sorted(path for path in year_root.iterdir() if path.is_file() and (path.name.endswith(".csv.gz") or path.suffix in {".csv", ".parquet"})))
    return paths


def split_years(preprocessed_config: dict[str, Any], scheme: str, split_name: str) -> list[int]:
    split = preprocessed_config["splits"][scheme]
    key = {"train": "train_years", "validation": "validation_years", "test": "test_years"}[split_name]
    return [int(year) for year in split.get(key, [])]


def build_training_sample(
    paths: list[Path],
    transformer: FeatureTransformer,
    split_column: str,
    split_name: str,
    horizons: list[int],
    sampling: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    max_horizon = max(horizons)
    count_columns = [split_column, "risk_set", f"onset_within_{max_horizon}h", f"hard_negative_{max_horizon}h"]
    stratum_totals = {"near_event": 0, "hard_negative": 0, "easy_negative": 0}
    file_counts: list[dict[str, int]] = []
    for index, path in enumerate(paths, start=1):
        frame = read_timeline(path, count_columns)
        frame = frame.loc[frame[split_column].eq(split_name) & frame["risk_set"].eq(1)]
        near = frame[f"onset_within_{max_horizon}h"].eq(1)
        hard = ~near & frame[f"hard_negative_{max_horizon}h"].eq(1)
        counts = {
            "near_event": int(near.sum()),
            "hard_negative": int(hard.sum()),
            "easy_negative": int((~near & ~hard).sum()),
        }
        file_counts.append(counts)
        for key, value in counts.items():
            stratum_totals[key] += value
        if index % 10 == 0 or index == len(paths):
            print(f"Counted training strata in {index}/{len(paths)} files", flush=True)

    hard_probability = min(1.0, float(sampling["hard_negative_target"]) / max(stratum_totals["hard_negative"], 1))
    easy_probability = min(1.0, float(sampling["easy_negative_target"]) / max(stratum_totals["easy_negative"], 1))
    base_seed = int(sampling["seed"])
    feature_columns = transformer.feature_names
    label_columns = ["hazard_label", *[f"onset_within_{horizon}h" for horizon in horizons]]
    read_columns = list(dict.fromkeys([*feature_columns, *count_columns, *label_columns]))
    x_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    label_parts: dict[str, list[np.ndarray]] = {column: [] for column in label_columns}
    selected_counts = {"near_event": 0, "hard_negative": 0, "easy_negative": 0}

    for index, path in enumerate(paths):
        frame = read_timeline(path, read_columns)
        frame = frame.loc[frame[split_column].eq(split_name) & frame["risk_set"].eq(1)].reset_index(drop=True)
        if frame.empty:
            continue
        near = frame[f"onset_within_{max_horizon}h"].eq(1).to_numpy()
        hard = (~near) & frame[f"hard_negative_{max_horizon}h"].eq(1).to_numpy()
        easy = (~near) & (~hard)
        rng = np.random.default_rng(base_seed + index * 104729)
        selected_hard = hard & (rng.random(frame.shape[0]) < hard_probability)
        selected_easy = easy & (rng.random(frame.shape[0]) < easy_probability)
        selected = near | selected_hard | selected_easy
        if not selected.any():
            continue
        selected_frame = frame.loc[selected]
        x_parts.append(transformer.transform(selected_frame))
        weights = np.ones(int(selected.sum()), dtype=np.float64)
        selected_strata_hard = selected_hard[selected]
        selected_strata_easy = selected_easy[selected]
        if hard_probability > 0:
            weights[selected_strata_hard] = 1.0 / hard_probability
        if easy_probability > 0:
            weights[selected_strata_easy] = 1.0 / easy_probability
        weight_parts.append(weights)
        for column in label_columns:
            label_parts[column].append(pd.to_numeric(selected_frame[column], errors="coerce").fillna(0).to_numpy(dtype=np.int8))
        selected_counts["near_event"] += int(near.sum())
        selected_counts["hard_negative"] += int(selected_hard.sum())
        selected_counts["easy_negative"] += int(selected_easy.sum())
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f"Sampled training rows from {index + 1}/{len(paths)} files", flush=True)

    x = np.concatenate(x_parts, axis=0)
    sample_weight = np.concatenate(weight_parts)
    labels = {column: np.concatenate(parts) for column, parts in label_parts.items()}
    order_rng = np.random.default_rng(base_seed + 991)
    order = order_rng.permutation(x.shape[0])
    x = x[order]
    sample_weight = sample_weight[order]
    labels = {column: values[order] for column, values in labels.items()}
    summary = {
        "stratum_totals": stratum_totals,
        "selected_counts": selected_counts,
        "hard_sampling_probability": hard_probability,
        "easy_sampling_probability": easy_probability,
        "sample_rows": int(x.shape[0]),
        "represented_weight_sum": float(sample_weight.sum()),
        "label_positive_counts_sample": {column: int(values.sum()) for column, values in labels.items()},
        "label_positive_weighted_estimates": {column: float(np.dot(sample_weight, values)) for column, values in labels.items()},
        "feature_count": int(x.shape[1]),
    }
    return x, labels, sample_weight, summary


def fit_models(
    x: np.ndarray,
    labels: dict[str, np.ndarray],
    sample_weight: np.ndarray,
    horizons: list[int],
    optimization: dict[str, Any],
) -> dict[str, WeightedBinaryGLM]:
    common = {
        "l2": float(optimization["l2"]),
        "max_iter": int(optimization["max_iter"]),
        "tolerance": float(optimization["tolerance"]),
    }
    specifications = {
        "local_weather_hazard": ("cloglog", "hazard_label"),
        "ordinary_logit_step": ("logit", "hazard_label"),
        **{f"direct_logit_{horizon}h": ("logit", f"onset_within_{horizon}h") for horizon in horizons},
    }
    models: dict[str, WeightedBinaryGLM] = {}
    for name, (link, label) in specifications.items():
        print(f"Fitting {name} on {x.shape[0]:,} sampled rows", flush=True)
        model = WeightedBinaryGLM(link=link, **common).fit(x, labels[label], sample_weight)
        models[name] = model
        print(f"  converged={model.converged} iterations={model.iterations} objective={model.objective:.8f}", flush=True)
    return models


def collect_calibration_scores(
    paths: list[Path],
    transformer: FeatureTransformer,
    models: dict[str, WeightedBinaryGLM],
    split_column: str,
    split_name: str,
    horizons: list[int],
) -> dict[str, np.ndarray]:
    feature_columns = transformer.feature_names
    label_columns = ["hazard_label", *[f"onset_within_{horizon}h" for horizon in horizons]]
    read_columns = list(dict.fromkeys([*feature_columns, split_column, "risk_set", *label_columns]))
    parts: dict[str, list[np.ndarray]] = {column: [] for column in label_columns}
    for model_name in models:
        parts[f"score::{model_name}"] = []
    for index, path in enumerate(paths, start=1):
        frame = read_timeline(path, read_columns)
        frame = frame.loc[frame[split_column].eq(split_name) & frame["risk_set"].eq(1)]
        if frame.empty:
            continue
        x = transformer.transform(frame)
        for column in label_columns:
            parts[column].append(pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype=np.int8))
        for model_name, model in models.items():
            parts[f"score::{model_name}"].append(model.decision_function(x).astype(np.float64))
        if index % 10 == 0 or index == len(paths):
            print(f"Collected calibration scores from {index}/{len(paths)} files", flush=True)
    return {key: np.concatenate(values) for key, values in parts.items()}


def fit_calibrators(scores: dict[str, np.ndarray], horizons: list[int]) -> dict[str, Any]:
    calibrators: dict[str, Any] = {}
    hazard = HazardRateCalibrator().fit(scores["score::local_weather_hazard"], scores["hazard_label"])
    calibrators["local_weather_hazard"] = hazard
    step = PlattCalibrator().fit(scores["score::ordinary_logit_step"], scores["hazard_label"])
    calibrators["ordinary_logit_step"] = step
    for horizon in horizons:
        name = f"direct_logit_{horizon}h"
        calibrators[name] = PlattCalibrator().fit(scores[f"score::{name}"], scores[f"onset_within_{horizon}h"])
    return calibrators


def write_model_bundle(
    path: Path,
    transformer: FeatureTransformer,
    models: dict[str, WeightedBinaryGLM],
    calibrators: dict[str, Any],
    horizons: list[int],
    experiment_config: dict[str, Any],
) -> None:
    bundle = {
        "model_family": "RIG-Hazard rule/local-hazard baseline",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "time_step_minutes": 10,
        "horizons_hours": horizons,
        "feature_transformer": transformer.to_dict(),
        "models": {name: model.to_dict() for name, model in models.items()},
        "calibrators": {name: calibrator.to_dict() for name, calibrator in calibrators.items()},
        "experiment_config": experiment_config,
        "hazard_accumulation": "P(T<=H|T>t)=1-exp(-n_steps*exp(eta+calibration_shift)); covariates are frozen at issue_time.",
    }
    write_json(path, bundle)


def generate_predictions(
    paths: list[Path],
    output_root: Path,
    transformer: FeatureTransformer,
    models: dict[str, WeightedBinaryGLM],
    calibrators: dict[str, Any],
    split_column: str,
    split_name: str,
    horizons: list[int],
    time_step_minutes: int,
    compression_level: int,
) -> list[Path]:
    feature_columns = transformer.feature_names
    audit_columns = [
        "station_code",
        "station_name",
        "city",
        "issue_time",
        "issue_year",
        "risk_spell_index",
        "risk_spell_step",
        "hazard_label",
        "exposure_e1_cold_humid",
        "exposure_e2_fog_low_visibility",
        "exposure_e3_any",
        *[f"onset_within_{horizon}h" for horizon in horizons],
        *[f"hard_negative_{horizon}h" for horizon in horizons],
    ]
    read_columns = list(dict.fromkeys([*feature_columns, *audit_columns, split_column, "risk_set"]))
    prediction_paths: list[Path] = []
    for index, path in enumerate(paths, start=1):
        frame = read_timeline(path, read_columns)
        frame = frame.loc[frame[split_column].eq(split_name) & frame["risk_set"].eq(1)].reset_index(drop=True)
        if frame.empty:
            continue
        x = transformer.transform(frame)
        output = frame[audit_columns].copy()
        output["cold_humid_rule"] = frame["exposure_e1_cold_humid"].astype(np.int8)
        output["strict_condensation_rule"] = (
            frame["exposure_e1_cold_humid"].eq(1) & frame["exposure_e2_fog_low_visibility"].eq(1)
        ).astype(np.int8)

        hazard_eta = models["local_weather_hazard"].decision_function(x)
        output["local_weather_hazard_raw_step"] = cloglog_probability(hazard_eta).astype(np.float32)
        output["local_weather_hazard_step"] = calibrators["local_weather_hazard"].predict(hazard_eta).astype(np.float32)
        step_eta = models["ordinary_logit_step"].decision_function(x)
        output["ordinary_logit_raw_step"] = logit_probability(step_eta).astype(np.float32)
        output["ordinary_logit_step"] = calibrators["ordinary_logit_step"].predict(step_eta).astype(np.float32)
        for horizon in horizons:
            steps = horizon * 60 / time_step_minutes
            output[f"local_weather_hazard_raw_{horizon}h"] = cloglog_probability(hazard_eta, steps=steps).astype(np.float32)
            output[f"local_weather_hazard_{horizon}h"] = calibrators["local_weather_hazard"].predict(hazard_eta, steps=steps).astype(np.float32)
            name = f"direct_logit_{horizon}h"
            direct_eta = models[name].decision_function(x)
            output[f"direct_logit_raw_{horizon}h"] = logit_probability(direct_eta).astype(np.float32)
            output[f"direct_logit_{horizon}h"] = calibrators[name].predict(direct_eta).astype(np.float32)

        destination = output_root / split_name / path.parent.name / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name.endswith(".csv.gz"):
            output.to_csv(destination, index=False, encoding="utf-8-sig", compression={"method": "gzip", "compresslevel": compression_level})
        elif destination.suffix == ".csv":
            output.to_csv(destination, index=False, encoding="utf-8-sig")
        else:
            destination = destination.with_suffix(".parquet")
            output.to_parquet(destination, index=False)
        prediction_paths.append(destination)
        if index % 10 == 0 or index == len(paths):
            print(f"Generated {split_name} predictions for {index}/{len(paths)} files", flush=True)
    return prediction_paths


def probability_metric_row(model: str, label: str, y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, probability))
    reference_brier = prevalence * (1.0 - prevalence)
    brier_skill = 1.0 - brier / reference_brier if reference_brier > 0 else float("nan")
    edges = np.asarray([0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0 + 1e-9])
    assignments = np.clip(np.digitize(probability, edges, right=False) - 1, 0, edges.size - 2)
    ece = 0.0
    maximum_gap = 0.0
    populated_bins = 0
    for bin_index in range(edges.size - 1):
        mask = assignments == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        populated_bins += 1
        gap = abs(float(probability[mask].mean()) - float(y[mask].mean()))
        ece += count / y.size * gap
        maximum_gap = max(maximum_gap, gap)
    return {
        "model": model,
        "label": label,
        "n": int(y.size),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "mean_probability": float(probability.mean()),
        "pr_auc": float(average_precision_score(y, probability)) if 0 < y.sum() < y.size else float("nan"),
        "roc_auc": float(roc_auc_score(y, probability)) if 0 < y.sum() < y.size else float("nan"),
        "brier_score": brier,
        "brier_skill": brier_skill,
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "ece": ece,
        "maximum_calibration_gap": maximum_gap,
        "populated_calibration_bins": populated_bins,
    }


def evaluate_probability_predictions(paths: list[Path], horizons: list[int]) -> list[dict[str, Any]]:
    label_columns = ["hazard_label", *[f"onset_within_{horizon}h" for horizon in horizons]]
    prediction_columns = [
        "cold_humid_rule",
        "strict_condensation_rule",
        "local_weather_hazard_raw_step",
        "local_weather_hazard_step",
        "ordinary_logit_raw_step",
        "ordinary_logit_step",
    ]
    for horizon in horizons:
        prediction_columns.extend(
            [
                f"local_weather_hazard_raw_{horizon}h",
                f"local_weather_hazard_{horizon}h",
                f"direct_logit_raw_{horizon}h",
                f"direct_logit_{horizon}h",
            ]
        )
    columns = [*label_columns, *prediction_columns]
    parts: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    for index, path in enumerate(paths, start=1):
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=columns, low_memory=False)
        for column in columns:
            parts[column].append(pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy())
        if index % 10 == 0 or index == len(paths):
            print(f"Loaded probability metrics from {index}/{len(paths)} prediction files", flush=True)
    arrays = {column: np.concatenate(values) for column, values in parts.items()}
    pairs: list[tuple[str, str]] = [
        ("local_weather_hazard_raw_step", "hazard_label"),
        ("local_weather_hazard_step", "hazard_label"),
        ("ordinary_logit_raw_step", "hazard_label"),
        ("ordinary_logit_step", "hazard_label"),
    ]
    for horizon in horizons:
        label = f"onset_within_{horizon}h"
        pairs.extend(
            [
                (f"local_weather_hazard_raw_{horizon}h", label),
                (f"local_weather_hazard_{horizon}h", label),
                (f"direct_logit_raw_{horizon}h", label),
                (f"direct_logit_{horizon}h", label),
            ]
        )
    max_horizon = max(horizons)
    pairs.extend(
        [
            ("cold_humid_rule", f"onset_within_{max_horizon}h"),
            ("strict_condensation_rule", f"onset_within_{max_horizon}h"),
        ]
    )
    return [probability_metric_row(model, label, arrays[label], arrays[model]) for model, label in pairs]


def load_warning_frame(paths: list[Path], horizon: int, model_columns: list[str]) -> pd.DataFrame:
    columns = [
        "station_code",
        "station_name",
        "issue_time",
        f"onset_within_{horizon}h",
        f"hard_negative_{horizon}h",
        *model_columns,
    ]
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(paths, start=1):
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=columns, low_memory=False)
        frames.append(frame)
        if index % 10 == 0 or index == len(paths):
            print(f"Loaded warning data from {index}/{len(paths)} prediction files", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result["issue_time"] = pd.to_datetime(result["issue_time"], errors="coerce")
    result["station_month"] = result["station_code"].astype(str) + "|" + result["issue_time"].dt.to_period("M").astype(str)
    return result


def threshold_for_false_alarm_budget(
    frame: pd.DataFrame,
    model_column: str,
    horizon: int,
    false_alarm_hours_per_station_month: float,
    step_minutes: int,
) -> float:
    negative_scores = pd.to_numeric(
        frame.loc[frame[f"onset_within_{horizon}h"].eq(0), model_column], errors="coerce"
    ).dropna().to_numpy(dtype=np.float64)
    station_months = max(int(frame["station_month"].nunique()), 1)
    budget_bins = int(math.floor(false_alarm_hours_per_station_month * 60 / step_minutes * station_months))
    if negative_scores.size <= budget_bins:
        return 0.0
    low = 0.0
    high = float(np.nextafter(1.0, 2.0))
    for _ in range(64):
        midpoint = (low + high) / 2.0
        if int((negative_scores >= midpoint).sum()) > budget_bins:
            low = midpoint
        else:
            high = midpoint
    return high


def observed_warning_rows(frame: pd.DataFrame, observed_column: str) -> pd.DataFrame:
    """Return rows whose complete warning horizon is observable."""

    if observed_column not in frame.columns:
        raise KeyError(f"Missing warning-horizon observability column: {observed_column}")
    observed = pd.to_numeric(frame[observed_column], errors="coerce").fillna(0).eq(1)
    result = frame.loc[observed].copy()
    if result.empty:
        raise ValueError(f"No observable warning horizons in column {observed_column}")
    return result


def evaluate_warning_model(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    model_column: str,
    threshold: float,
    horizon: int,
    step_minutes: int,
    false_alarm_budget: float,
    operating_point: str,
    minimum_consecutive_alarm_bins: int,
    alarm_merge_gap_minutes: int,
    maximum_silence_before_event_minutes: int,
) -> dict[str, Any]:
    score = pd.to_numeric(frame[model_column], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    alarm = score >= threshold
    future_event = frame[f"onset_within_{horizon}h"].eq(1).to_numpy()
    hard_negative = frame[f"hard_negative_{horizon}h"].eq(1).to_numpy()
    station_months = max(int(frame["station_month"].nunique()), 1)
    false_bins = int((alarm & ~future_event).sum())
    alarm_bins = int(alarm.sum())
    false_hours_per_station_month = false_bins * step_minutes / 60.0 / station_months
    alarm_hours_per_station_month = alarm_bins * step_minutes / 60.0 / station_months
    hard_total = int(hard_negative.sum())
    hard_false = int((alarm & hard_negative).sum())

    station_frames: dict[str, pd.DataFrame] = {}
    for station_code, station_frame in frame.assign(_alarm=alarm).groupby("station_code", sort=False):
        station_frames[str(station_code)] = station_frame.sort_values("issue_time")

    horizon_delta = pd.Timedelta(hours=horizon)
    target_events = events.loc[events["valid_target_event"].eq(1)].copy()
    evaluable = 0
    hits = 0
    effective_leads: list[float] = []
    last_alarm_leads: list[float] = []
    for event in target_events.itertuples(index=False):
        station_frame = station_frames.get(str(event.station_code))
        if station_frame is None:
            continue
        onset = pd.Timestamp(event.onset_time)
        window = station_frame.loc[
            station_frame["issue_time"].ge(onset - horizon_delta) & station_frame["issue_time"].lt(onset)
        ]
        if window.empty:
            continue
        evaluable += 1
        alarm_times = window.loc[window["_alarm"], "issue_time"]
        if alarm_times.empty:
            continue
        alarm_times = alarm_times.sort_values()
        if (onset - alarm_times.max()).total_seconds() / 60.0 > maximum_silence_before_event_minutes:
            continue
        gaps = alarm_times.diff().dt.total_seconds().div(60).fillna(0).to_numpy()
        break_positions = np.flatnonzero(gaps > alarm_merge_gap_minutes)
        final_episode_start = int(break_positions[-1]) if break_positions.size else 0
        final_episode = alarm_times.iloc[final_episode_start:]
        if final_episode.shape[0] < minimum_consecutive_alarm_bins:
            continue
        hits += 1
        effective_leads.append(float((onset - final_episode.min()).total_seconds() / 3600.0))
        last_alarm_leads.append(float((onset - final_episode.max()).total_seconds() / 3600.0))

    return {
        "model": model_column,
        "operating_point": operating_point,
        "threshold": float(threshold),
        "false_alarm_budget_hours_per_station_month": false_alarm_budget,
        "false_alarm_hours_per_station_month": false_hours_per_station_month,
        "alarm_hours_per_station_month": alarm_hours_per_station_month,
        "budget_met": int(false_hours_per_station_month <= false_alarm_budget + 1e-9),
        "hard_negative_far": hard_false / hard_total if hard_total else float("nan"),
        "hard_negative_alarm_bins": hard_false,
        "hard_negative_bins": hard_total,
        "target_events": int(target_events.shape[0]),
        "evaluable_events": evaluable,
        "hit_events": hits,
        "event_hit_rate": hits / evaluable if evaluable else float("nan"),
        "mean_effective_lead_hours": float(np.mean(effective_leads)) if effective_leads else float("nan"),
        "median_effective_lead_hours": float(np.median(effective_leads)) if effective_leads else float("nan"),
        "mean_last_alarm_lead_hours": float(np.mean(last_alarm_leads)) if last_alarm_leads else float("nan"),
        "median_last_alarm_lead_hours": float(np.median(last_alarm_leads)) if last_alarm_leads else float("nan"),
        "minimum_consecutive_alarm_bins": minimum_consecutive_alarm_bins,
        "alarm_merge_gap_minutes": alarm_merge_gap_minutes,
        "maximum_silence_before_event_minutes": maximum_silence_before_event_minutes,
        "station_months": station_months,
    }


def evaluate_warning_operating_points(
    validation_paths: list[Path],
    test_paths: list[Path],
    events: pd.DataFrame,
    horizon: int,
    step_minutes: int,
    false_alarm_budget: float,
    minimum_consecutive_alarm_bins: int,
    alarm_merge_gap_minutes: int,
    maximum_silence_before_event_minutes: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    learned_models = [f"local_weather_hazard_{horizon}h", f"direct_logit_{horizon}h"]
    rule_models = ["cold_humid_rule", "strict_condensation_rule"]
    model_columns = [*learned_models, *rule_models]
    validation = load_warning_frame(validation_paths, horizon, model_columns)
    test = load_warning_frame(test_paths, horizon, model_columns)
    events = events.copy()
    events["onset_time"] = pd.to_datetime(events["onset_time"], errors="coerce")
    events["valid_target_event"] = pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).astype(int)
    validation_years = set(validation["issue_time"].dt.year.unique())
    test_years = set(test["issue_time"].dt.year.unique())
    validation_events = events.loc[events["onset_time"].dt.year.isin(validation_years)]
    test_events = events.loc[events["onset_time"].dt.year.isin(test_years)]

    thresholds: dict[str, float] = {
        model: threshold_for_false_alarm_budget(validation, model, horizon, false_alarm_budget, step_minutes)
        for model in learned_models
    }
    thresholds.update({model: 0.5 for model in rule_models})
    rows: list[dict[str, Any]] = []
    for model in model_columns:
        source = "validation_budget" if model in learned_models else "fixed_rule"
        rows.append(
            evaluate_warning_model(
                validation, validation_events, model, thresholds[model], horizon, step_minutes, false_alarm_budget,
                f"validation::{source}", minimum_consecutive_alarm_bins, alarm_merge_gap_minutes, maximum_silence_before_event_minutes
            )
        )
        rows.append(
            evaluate_warning_model(
                test, test_events, model, thresholds[model], horizon, step_minutes, false_alarm_budget,
                f"test::locked_{source}", minimum_consecutive_alarm_bins, alarm_merge_gap_minutes, maximum_silence_before_event_minutes
            )
        )
    for model in learned_models:
        matched_threshold = threshold_for_false_alarm_budget(test, model, horizon, false_alarm_budget, step_minutes)
        rows.append(
            evaluate_warning_model(
                test, test_events, model, matched_threshold, horizon, step_minutes, false_alarm_budget,
                "test::matched_budget_diagnostic", minimum_consecutive_alarm_bins, alarm_merge_gap_minutes, maximum_silence_before_event_minutes
            )
        )
    return rows, thresholds


def markdown_table(rows: list[dict[str, Any]], columns: list[str], digits: int = 5) -> str:
    if not rows:
        return "无。"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_baseline_report(
    sample_summary: dict[str, Any],
    models: dict[str, WeightedBinaryGLM],
    calibrators: dict[str, Any],
    test_metrics: list[dict[str, Any]],
    warning_rows: list[dict[str, Any]],
    output_root: Path,
) -> str:
    step_models = [row for row in test_metrics if row["model"] in {"local_weather_hazard_step", "ordinary_logit_step"}]
    hazard_step = next(row for row in step_models if row["model"] == "local_weather_hazard_step")
    logit_step = next(row for row in step_models if row["model"] == "ordinary_logit_step")
    relative_improvements = {
        metric: (logit_step[metric] - hazard_step[metric]) / max(abs(logit_step[metric]), 1e-12)
        for metric in ("brier_score", "log_loss", "ece")
    }
    meaningful_wins = sum(value > 0.01 for value in relative_improvements.values())
    meaningful_losses = sum(value < -0.01 for value in relative_improvements.values())
    if meaningful_wins >= 2:
        calibration_judgment = "当前线性local weather hazard在至少两个校准指标上取得超过1%的相对改善，初步支持hazard校准假设。"
    elif meaningful_losses >= 2:
        calibration_judgment = "当前线性local weather hazard在至少两个校准指标上明显差于普通logit单步分类器，hazard校准假设暂不成立。"
    else:
        calibration_judgment = "local_weather_hazard与普通logit单步分类器的校准差异不足1%，当前应判定为实质相当，不能宣称hazard更好。"
    matched = [row for row in warning_rows if row["operating_point"] == "test::matched_budget_diagnostic"]
    hazard_warning = next(row for row in matched if row["model"].startswith("local_weather_hazard_"))
    direct_warning = next(row for row in matched if row["model"].startswith("direct_logit_"))
    lead_judgment = (
        "在测试集匹配误报预算下，local_weather_hazard的平均有效提前量至少增加0.1小时。"
        if hazard_warning["mean_effective_lead_hours"] > direct_warning["mean_effective_lead_hours"] + 0.1
        else "在测试集匹配误报预算下，local_weather_hazard尚未获得至少0.1小时的有效提前量改善。"
    )
    optimizer_rows = [
        {
            "model": name,
            "link": model.link,
            "converged": model.converged,
            "iterations": model.iterations,
            "objective": model.objective,
        }
        for name, model in models.items()
    ]
    calibration_rows = [
        {"model": name, **calibrator.to_dict()} for name, calibrator in calibrators.items()
    ]
    probability_columns = ["model", "label", "positives", "pr_auc", "brier_score", "brier_skill", "log_loss", "ece", "mean_probability"]
    warning_columns = [
        "model",
        "operating_point",
        "threshold",
        "false_alarm_hours_per_station_month",
        "event_hit_rate",
        "mean_effective_lead_hours",
        "median_effective_lead_hours",
        "hard_negative_far",
    ]
    return f"""# RIG-Hazard rule/local-hazard基线实验报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验定位

- `cold_humid_rule`、`strict_condensation_rule`：非学习型规则基线。
- `local_weather_hazard`：同站点局地特征的cloglog离散时间hazard。
- 普通分类器：同风险集、同特征的logit单步分类器，以及1/3/6小时直接logit分类器。
- `local_weather_hazard`多时效风险使用当前发报时刻的事件率进行冻结协变量累积，不读取未来气象观测。
- 训练保留全部事件前6小时窗口，对抽样的hard/easy negatives使用逆概率权重。

## 训练样本

```json
{json.dumps(sample_summary, ensure_ascii=False, indent=2)}
```

## 优化状态

{markdown_table(optimizer_rows, ['model', 'link', 'converged', 'iterations', 'objective'])}

## 校准参数

{markdown_table(calibration_rows, sorted({key for row in calibration_rows for key in row}))}

## 测试集概率指标

{markdown_table(test_metrics, probability_columns)}

## 事件预警指标

{markdown_table(warning_rows, warning_columns)}

## 当前判断

1. {calibration_judgment}
2. {lead_judgment}
3. 这些结果只代表线性、非图local weather hazard基线；hierarchical barrier层次站点屏障和unfiltered graph稀疏滞后图尚未加入。

完整模型、预测和CSV指标位于：`{output_root}`
"""


def run_baseline_experiment(config: dict[str, Any], config_path: Path, overwrite: bool = False) -> Path:
    preprocessed_root = resolve_project_path(config["preprocessed_root"])
    output_root = resolve_project_path(config["output_root"])
    prepare_output_root(output_root, overwrite)
    validation_summary_path = preprocessed_root / "validation_summary.json"
    if not validation_summary_path.is_file():
        raise FileNotFoundError("Preprocessed validation_summary.json is missing. Run `python run_rig_hazard.py validate` first.")
    validation_status = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    if validation_status.get("status") != "PASS":
        raise RuntimeError("Preprocessed data validation did not pass")

    preprocessed_config = json.loads((preprocessed_root / "resolved_config.json").read_text(encoding="utf-8"))
    contract = json.loads((preprocessed_root / "feature_contract.json").read_text(encoding="utf-8"))
    horizons = [int(value.replace("onset_within_", "").replace("h", "")) for value in contract["multi_horizon_labels"]]
    scheme = str(config["scheme"])
    split_column = f"split_{scheme}"
    train_name = str(config["train_split"])
    validation_name = str(config["validation_split"])
    test_name = str(config["test_split"])
    train_paths = timeline_files(preprocessed_root, split_years(preprocessed_config, scheme, train_name))
    validation_paths = timeline_files(preprocessed_root, split_years(preprocessed_config, scheme, validation_name))
    test_paths = timeline_files(preprocessed_root, split_years(preprocessed_config, scheme, test_name))
    if not train_paths or not validation_paths or not test_paths:
        raise RuntimeError("Train, validation, and test timeline files are required")

    transformer = load_feature_transformer(preprocessed_root, scheme, float(config["optimization"]["continuous_clip_z"]))
    x, labels, sample_weight, sample_summary = build_training_sample(
        train_paths, transformer, split_column, train_name, horizons, config["sampling"]
    )
    write_json(output_root / "training_sample_summary.json", sample_summary)
    models = fit_models(x, labels, sample_weight, horizons, config["optimization"])
    del x, labels, sample_weight

    calibration_scores = collect_calibration_scores(
        validation_paths, transformer, models, split_column, validation_name, horizons
    )
    calibrators = fit_calibrators(calibration_scores, horizons)
    del calibration_scores
    write_model_bundle(output_root / "model_bundle.json", transformer, models, calibrators, horizons, config)
    write_json(output_root / "resolved_experiment_config.json", config)

    step_minutes = int(preprocessed_config["time_step_minutes"])
    validation_prediction_paths = generate_predictions(
        validation_paths,
        output_root / "predictions",
        transformer,
        models,
        calibrators,
        split_column,
        validation_name,
        horizons,
        step_minutes,
        int(config.get("prediction_compression_level", 1)),
    )
    test_prediction_paths = generate_predictions(
        test_paths,
        output_root / "predictions",
        transformer,
        models,
        calibrators,
        split_column,
        test_name,
        horizons,
        step_minutes,
        int(config.get("prediction_compression_level", 1)),
    )

    validation_metrics = evaluate_probability_predictions(validation_prediction_paths, horizons)
    test_metrics = evaluate_probability_predictions(test_prediction_paths, horizons)
    pd.DataFrame(validation_metrics).to_csv(output_root / "probability_metrics_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_metrics).to_csv(output_root / "probability_metrics_test.csv", index=False, encoding="utf-8-sig")

    events = pd.read_csv(preprocessed_root / "events_recurrent.csv", encoding="utf-8-sig")
    warning_horizon = int(config["warning"]["horizon_hours"])
    false_alarm_budget = float(config["warning"]["false_alarm_hours_per_station_month"])
    minimum_consecutive_alarm_bins = int(config["warning"].get("minimum_consecutive_alarm_bins", 2))
    alarm_merge_gap_minutes = int(config["warning"].get("alarm_merge_gap_minutes", step_minutes * 2))
    maximum_silence_before_event_minutes = int(config["warning"].get("maximum_silence_before_event_minutes", step_minutes * 3))
    warning_rows, thresholds = evaluate_warning_operating_points(
        validation_prediction_paths,
        test_prediction_paths,
        events,
        warning_horizon,
        step_minutes,
        false_alarm_budget,
        minimum_consecutive_alarm_bins,
        alarm_merge_gap_minutes,
        maximum_silence_before_event_minutes,
    )
    pd.DataFrame(warning_rows).to_csv(output_root / "warning_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output_root / "warning_thresholds.json", thresholds)
    report = build_baseline_report(sample_summary, models, calibrators, test_metrics, warning_rows, output_root)
    (output_root / "baseline_report.md").write_text(report, encoding="utf-8")
    print(f"Baseline experiment complete: {output_root}", flush=True)
    return output_root

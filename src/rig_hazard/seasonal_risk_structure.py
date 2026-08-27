from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .baseline_models import HazardRateCalibrator, WeightedBinaryGLM, cloglog_probability
from .config import resolve_project_path
from .deep_training import weighted_probability_metric_row
from .recurrence_models import HORIZON_STEPS, CurrentRiskData, load_current_risk_data


MODEL_NAMES = ("weather_only", "recurrence_state", "weather_recurrence_interaction", "event_history")
NESTED_COMPARISONS = (
    ("recurrence_state", "weather_only"),
    ("weather_recurrence_interaction", "recurrence_state"),
    ("event_history", "weather_recurrence_interaction"),
)
WEATHER_FEATURES = (
    "temperature_mean",
    "temperature_min",
    "temperature_change_30m",
    "rh_mean",
    "rh_max",
    "rh_change_30m",
    "pressure_mean",
    "rain_mean",
    "hourly_rain_mean",
    "wind_speed_mean",
    "visibility_mean",
    "visibility_min",
    "visibility_change_30m",
    "fog_fraction",
    "precipitation_fraction",
    "snow_fraction",
    "exposure_fraction_1h",
    "exposure_e1_cold_humid",
    "exposure_e2_fog_low_visibility",
    "exposure_recent_1h",
    "time_since_risk_entry_hours",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)
INTERACTION_FEATURES = (
    "temperature_mean",
    "rh_mean",
    "exposure_e1_cold_humid",
    "snow_fraction",
    "visibility_mean",
)
HISTORY_FEATURES = (
    "time_since_last_recurrent_event_hours",
    "events_past_7d",
    "events_past_30d",
    "previous_event_duration_hours",
    "previous_event_max_thickness",
    "previous_event_severity",
)
EVENT_GROUPS = ("all", "e1", "e2", "e3plus", "recurrent")
SEASON_COHORTS = ("all_seasons", "complete_seasons", "censored_seasons")


@dataclass
class SeasonalRiskData:
    features: np.ndarray
    labels: dict[str, np.ndarray]
    observed: dict[str, np.ndarray]
    sample_weight: np.ndarray
    issue_time_ns: np.ndarray
    station_code: np.ndarray
    season_start_year: np.ndarray
    recurrence_state: np.ndarray
    next_seasonal_order: np.ndarray
    complete_season: np.ndarray
    feature_names: list[str]


@dataclass
class SeasonalHazardDesign:
    model_name: str
    source_feature_names: list[str]
    station_levels: list[str]
    feature_names: list[str]
    penalty_weights: np.ndarray

    @classmethod
    def fit(cls, model_name: str, data: SeasonalRiskData) -> "SeasonalHazardDesign":
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unsupported seasonal risk model: {model_name}")
        missing = sorted(
            set((*WEATHER_FEATURES, *INTERACTION_FEATURES, *HISTORY_FEATURES))
            - set(data.feature_names)
        )
        if missing:
            raise ValueError(f"Risk cache is missing required features: {missing}")
        station_levels = sorted(set(data.station_code.tolist()))
        names = list(WEATHER_FEATURES)
        penalties = [1.0] * len(names)
        if model_name != "weather_only":
            names.append("seasonal_recurrence_state")
            penalties.append(1.0)
        if model_name in {"weather_recurrence_interaction", "event_history"}:
            names.extend([f"state_x_{name}" for name in INTERACTION_FEATURES])
            penalties.extend([1.0] * len(INTERACTION_FEATURES))
        if model_name == "event_history":
            names.extend([f"state_history_{name}" for name in HISTORY_FEATURES])
            penalties.extend([1.0] * len(HISTORY_FEATURES))
        # Penalized station indicators provide a low-capacity frailty approximation.
        names.extend([f"station_frailty[{station}]" for station in station_levels])
        penalties.extend([10.0] * len(station_levels))
        return cls(
            model_name=model_name,
            source_feature_names=list(data.feature_names),
            station_levels=station_levels,
            feature_names=names,
            penalty_weights=np.asarray(penalties, dtype=np.float64),
        )

    def transform(self, data: SeasonalRiskData) -> np.ndarray:
        if list(data.feature_names) != self.source_feature_names:
            raise ValueError("Seasonal risk feature contract changed between fit and transform")
        index = {name: value for value, name in enumerate(data.feature_names)}
        parts = [data.features[:, [index[name] for name in WEATHER_FEATURES]]]
        state = data.recurrence_state.astype(np.float64)[:, None]
        if self.model_name != "weather_only":
            parts.append(state)
        if self.model_name in {"weather_recurrence_interaction", "event_history"}:
            weather = data.features[:, [index[name] for name in INTERACTION_FEATURES]]
            parts.append(weather * state)
        if self.model_name == "event_history":
            history = data.features[:, [index[name] for name in HISTORY_FEATURES]]
            # History is explicitly inactive during the seasonal first-event risk period.
            parts.append(history * state)
        station = np.column_stack(
            [data.station_code == level for level in self.station_levels]
        ).astype(np.float64)
        parts.append(station)
        return np.concatenate(parts, axis=1).astype(np.float64, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "source_feature_names": self.source_feature_names,
            "station_levels": self.station_levels,
            "feature_names": self.feature_names,
            "penalty_weights": self.penalty_weights.tolist(),
        }


def icing_season_start_year(issue_time_ns: np.ndarray) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(issue_time_ns, unit="ns"))
    in_season = (times.month >= 11) | (times.month <= 4)
    year = np.where(times.month >= 11, times.year, times.year - 1).astype(np.int16)
    return np.where(in_season, year, -1).astype(np.int16)


def _event_onsets_by_station_season(events: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    valid = events.loc[events["valid_target_event"].eq(1)].copy()
    valid["onset_time"] = pd.to_datetime(valid["onset_time"], errors="coerce")
    valid = valid.dropna(subset=["station_code", "onset_time"])
    valid["season_start_year"] = icing_season_start_year(
        valid["onset_time"].astype("int64").to_numpy(dtype=np.int64)
    )
    return {
        (str(station), int(season)): np.sort(
            group["onset_time"].astype("int64").to_numpy(dtype=np.int64)
        )
        for (station, season), group in valid.groupby(
            ["station_code", "season_start_year"], sort=True
        )
    }


def add_seasonal_state(
    data: CurrentRiskData,
    events: pd.DataFrame,
    observation_start: pd.Timestamp,
    observation_end: pd.Timestamp,
) -> SeasonalRiskData:
    season_year = icing_season_start_year(data.issue_time_ns)
    keep = season_year >= 0
    onsets = _event_onsets_by_station_season(events)
    previous_count = np.zeros(data.issue_time_ns.size, dtype=np.int16)
    for station in np.unique(data.station_code):
        station_positions = np.flatnonzero(data.station_code == station)
        for season in np.unique(season_year[station_positions]):
            if season < 0:
                continue
            positions = station_positions[season_year[station_positions] == season]
            event_times = onsets.get((str(station), int(season)), np.empty(0, dtype=np.int64))
            previous_count[positions] = np.searchsorted(
                event_times, data.issue_time_ns[positions], side="right"
            ).astype(np.int16)
    season_start = pd.to_datetime(
        [f"{int(year)}-11-01" if year >= 0 else "1900-01-01" for year in season_year]
    )
    season_end = pd.to_datetime(
        [f"{int(year) + 1}-05-01" if year >= 0 else "1900-01-02" for year in season_year]
    )
    complete = np.asarray(
        (season_start >= pd.Timestamp(observation_start))
        & (season_end <= pd.Timestamp(observation_end) + pd.Timedelta(minutes=10)),
        dtype=bool,
    )
    return SeasonalRiskData(
        features=data.features[keep],
        labels={name: values[keep] for name, values in data.labels.items()},
        observed={name: values[keep] for name, values in data.observed.items()},
        sample_weight=data.sample_weight[keep],
        issue_time_ns=data.issue_time_ns[keep],
        station_code=data.station_code[keep],
        season_start_year=season_year[keep],
        recurrence_state=(previous_count[keep] >= 1).astype(np.int8),
        next_seasonal_order=(previous_count[keep] + 1).astype(np.int16),
        complete_season=complete[keep],
        feature_names=list(data.feature_names),
    )


def compact_data(data: SeasonalRiskData) -> SeasonalRiskData:
    return SeasonalRiskData(
        features=np.empty((data.sample_weight.size, 0), dtype=np.float32),
        labels=data.labels,
        observed=data.observed,
        sample_weight=data.sample_weight,
        issue_time_ns=data.issue_time_ns,
        station_code=data.station_code,
        season_start_year=data.season_start_year,
        recurrence_state=data.recurrence_state,
        next_seasonal_order=data.next_seasonal_order,
        complete_season=data.complete_season,
        feature_names=[],
    )


def concatenate_data(parts: list[SeasonalRiskData]) -> SeasonalRiskData:
    if not parts:
        raise ValueError("At least one seasonal risk dataset is required")
    return SeasonalRiskData(
        features=np.concatenate([part.features for part in parts]),
        labels={name: np.concatenate([part.labels[name] for part in parts]) for name in HORIZON_STEPS},
        observed={name: np.concatenate([part.observed[name] for part in parts]) for name in HORIZON_STEPS},
        sample_weight=np.concatenate([part.sample_weight for part in parts]),
        issue_time_ns=np.concatenate([part.issue_time_ns for part in parts]),
        station_code=np.concatenate([part.station_code for part in parts]),
        season_start_year=np.concatenate([part.season_start_year for part in parts]),
        recurrence_state=np.concatenate([part.recurrence_state for part in parts]),
        next_seasonal_order=np.concatenate([part.next_seasonal_order for part in parts]),
        complete_season=np.concatenate([part.complete_season for part in parts]),
        feature_names=list(parts[0].feature_names),
    )


def _group_mask(data: SeasonalRiskData, cohort: str, event_group: str) -> np.ndarray:
    if cohort == "complete_seasons":
        mask = data.complete_season.copy()
    elif cohort == "censored_seasons":
        mask = ~data.complete_season
    elif cohort == "all_seasons":
        mask = np.ones(data.sample_weight.size, dtype=bool)
    else:
        raise ValueError(f"Unknown season cohort: {cohort}")
    order = data.next_seasonal_order
    if event_group == "e1":
        mask &= order == 1
    elif event_group == "e2":
        mask &= order == 2
    elif event_group == "e3plus":
        mask &= order >= 3
    elif event_group == "recurrent":
        mask &= order >= 2
    elif event_group != "all":
        raise ValueError(f"Unknown event group: {event_group}")
    return mask


def probability_metric_rows(
    model_name: str,
    split: str,
    eta: np.ndarray,
    data: SeasonalRiskData,
    calibrator: HazardRateCalibrator | None,
    fold: int | str,
) -> list[dict[str, Any]]:
    modes = [("raw", None)]
    if calibrator is not None:
        modes.append(("calibrated_2022_oof", calibrator))
    rows: list[dict[str, Any]] = []
    for calibration, calibration_model in modes:
        for horizon, steps in HORIZON_STEPS.items():
            probability = (
                cloglog_probability(eta, steps=steps)
                if calibration_model is None
                else calibration_model.predict(eta, steps=steps)
            )
            for cohort in SEASON_COHORTS:
                for event_group in EVENT_GROUPS:
                    selected = data.observed[horizon] & _group_mask(data, cohort, event_group)
                    if not selected.any():
                        continue
                    row = weighted_probability_metric_row(
                        model_name,
                        calibration,
                        horizon,
                        data.labels[horizon][selected],
                        probability[selected],
                        data.sample_weight[selected],
                    )
                    row.update(
                        {
                            "split": split,
                            "fold": fold,
                            "season_cohort": cohort,
                            "event_stage": event_group,
                            "event_rows": int(data.labels[horizon][selected].sum()),
                            "stations": int(np.unique(data.station_code[selected]).size),
                        }
                    )
                    rows.append(row)
    return rows


def _fit_model(
    model_name: str,
    data: SeasonalRiskData,
    settings: dict[str, Any],
) -> tuple[SeasonalHazardDesign, WeightedBinaryGLM]:
    design = SeasonalHazardDesign.fit(model_name, data)
    x = design.transform(data)
    model = WeightedBinaryGLM(
        "cloglog",
        l2=float(settings.get("l2", 1e-4)),
        max_iter=int(settings.get("maximum_iterations", 250)),
        tolerance=float(settings.get("tolerance", 1e-8)),
        penalty_weights=design.penalty_weights,
    ).fit(x, data.labels["step"], data.sample_weight)
    return design, model


def _weighted_log_loss(label: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability.astype(np.float64), 1e-12, 1.0 - 1e-12)
    return -(label * np.log(clipped) + (1.0 - label) * np.log1p(-clipped))


def paired_station_loss_bootstrap(
    data: SeasonalRiskData,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    model_a: str,
    model_b: str,
    horizon: str,
    calibration: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort_index, cohort in enumerate(SEASON_COHORTS):
        for group_index, event_group in enumerate(EVENT_GROUPS):
            selected = data.observed[horizon] & _group_mask(data, cohort, event_group)
            if not selected.any():
                continue
            stations = np.sort(np.unique(data.station_code[selected]))
            lookup = {station: index for index, station in enumerate(stations)}
            station_index = np.asarray(
                [lookup[value] for value in data.station_code[selected]], dtype=np.int16
            )
            weight = data.sample_weight[selected].astype(np.float64)
            label = data.labels[horizon][selected].astype(np.float64)
            difference = _weighted_log_loss(label, probability_a[selected]) - _weighted_log_loss(
                label, probability_b[selected]
            )
            station_weight = np.bincount(
                station_index, weights=weight, minlength=stations.size
            )
            station_loss = np.bincount(
                station_index, weights=weight * difference, minlength=stations.size
            )
            rng = np.random.default_rng(seed + cohort_index * 101 + group_index * 17)
            counts = rng.multinomial(
                stations.size,
                np.full(stations.size, 1.0 / stations.size),
                size=bootstrap_samples,
            )
            denominator = counts @ station_weight
            draws = np.divide(
                counts @ station_loss,
                denominator,
                out=np.full(bootstrap_samples, np.nan),
                where=denominator > 0,
            )
            finite = draws[np.isfinite(draws)]
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": "delta_log_loss",
                    "calibration": calibration,
                    "horizon": horizon,
                    "season_cohort": cohort,
                    "event_stage": event_group,
                    "estimate": float(station_loss.sum() / station_weight.sum()),
                    "ci95_low": float(np.quantile(finite, 0.025)),
                    "ci95_high": float(np.quantile(finite, 0.975)),
                    "probability_model_a_better": float((finite < 0).mean()),
                    "risk_rows": int(selected.sum()),
                    "event_rows": int(label.sum()),
                    "stations": int(stations.size),
                    "bootstrap_samples": int(finite.size),
                }
            )
    return pd.DataFrame(rows)


def _cluster_sandwich(
    design: SeasonalHazardDesign,
    model: WeightedBinaryGLM,
    data: SeasonalRiskData,
) -> tuple[pd.DataFrame, np.ndarray]:
    x = design.transform(data)
    z = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    parameters = np.concatenate([[model.intercept], model.coefficients])
    eta = np.clip(z @ parameters, -30.0, 15.0)
    rate = np.exp(eta)
    label = data.labels["step"].astype(bool)
    weight = data.sample_weight.astype(np.float64)
    weight /= max(float(weight.mean()), 1e-12)
    positive_gradient = np.zeros_like(rate)
    stable_gradient = rate < 50.0
    positive_gradient[stable_gradient] = -rate[stable_gradient] / np.expm1(
        rate[stable_gradient]
    )
    gradient_eta = np.where(label, positive_gradient, rate)
    small = rate < 1e-4
    positive_information = np.empty_like(rate)
    positive_information[small] = rate[small] / 2.0
    middle = (~small) & (rate < 50.0)
    exp_rate = np.exp(rate[middle])
    positive_information[middle] = (
        rate[middle]
        * (exp_rate * (rate[middle] - 1.0) + 1.0)
        / np.square(np.expm1(rate[middle]))
    )
    positive_information[rate >= 50.0] = 0.0
    information = np.where(label, positive_information, rate)
    bread = z.T @ (z * (weight * information)[:, None])
    penalty = np.concatenate([[0.0], design.penalty_weights])
    bread += float(model.l2) * float(weight.sum()) * np.diag(penalty)
    inverse_bread = np.linalg.pinv(bread, rcond=1e-10)
    cluster_scores = []
    for station in np.unique(data.station_code):
        selected = data.station_code == station
        cluster_scores.append(
            z[selected].T @ (weight[selected] * gradient_eta[selected])
        )
    score_matrix = np.vstack(cluster_scores)
    meat = score_matrix.T @ score_matrix
    covariance = inverse_bread @ meat @ inverse_bread
    clusters = score_matrix.shape[0]
    if clusters > 1 and z.shape[0] > z.shape[1]:
        covariance *= (clusters / (clusters - 1)) * (
            (z.shape[0] - 1) / (z.shape[0] - z.shape[1])
        )
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistic = np.divide(
        parameters,
        standard_error,
        out=np.full_like(parameters, np.nan),
        where=standard_error > 0,
    )
    names = ["intercept", *design.feature_names]
    table = pd.DataFrame(
        {
            "term": names,
            "coefficient": parameters,
            "hazard_rate_ratio": np.exp(np.clip(parameters, -20.0, 20.0)),
            "cluster_robust_se": standard_error,
            "z_statistic": statistic,
            "p_value": 2.0 * norm.sf(np.abs(statistic)),
            "ci95_low": parameters - 1.96 * standard_error,
            "ci95_high": parameters + 1.96 * standard_error,
            "clusters": clusters,
            "covariance_note": "ridge-adjusted station-cluster sandwich approximation",
        }
    )
    return table, covariance


def _joint_interaction_test(
    coefficient_table: pd.DataFrame,
    covariance: np.ndarray,
) -> dict[str, Any]:
    names = coefficient_table["term"].tolist()
    indices = [names.index(f"state_x_{name}") for name in INTERACTION_FEATURES]
    coefficients = coefficient_table.loc[indices, "coefficient"].to_numpy(dtype=np.float64)
    sub_covariance = covariance[np.ix_(indices, indices)]
    statistic = float(coefficients @ np.linalg.pinv(sub_covariance) @ coefficients)
    return {
        "test": "joint_weather_by_recurrence_state_interactions",
        "terms": ",".join(f"state_x_{name}" for name in INTERACTION_FEATURES),
        "wald_chi_square": statistic,
        "degrees_of_freedom": len(indices),
        "p_value": float(chi2.sf(statistic, len(indices))),
        "covariance": "ridge-adjusted station-cluster sandwich approximation",
    }


def _observation_bounds(cache_root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    manifest = json.loads((cache_root / "timeline_manifest.json").read_text(encoding="utf-8"))
    years = [int(metadata["year"]) for metadata in manifest["files"]]
    if not years:
        raise ValueError("Timeline manifest has no files")
    step_minutes = int(
        json.loads((cache_root / "sample_contract.json").read_text(encoding="utf-8"))[
            "step_minutes"
        ]
    )
    start = pd.Timestamp(f"{min(years)}-01-01")
    end = pd.Timestamp(f"{max(years) + 1}-01-01") - pd.Timedelta(minutes=step_minutes)
    return start, end


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_seasonal_risk_structure(
    config_path: str | Path,
    output_root: str | Path,
    number_folds: int = 5,
    bootstrap_samples: int = 5000,
    maximum_train_rows: int | None = None,
    maximum_evaluation_rows: int | None = None,
    overwrite: bool = False,
) -> Path:
    config_path = resolve_project_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_root = resolve_project_path(config["cache_root"])
    output_root = resolve_project_path(output_root)
    _prepare_output(output_root, overwrite)
    settings = dict(config.get("seasonal_risk_structure", {}))
    settings.setdefault("l2", 1e-4)
    settings.setdefault("maximum_iterations", 250)
    settings.setdefault("tolerance", 1e-8)
    seed = int(config["sampling"]["seed"])
    events = pd.read_csv(
        resolve_project_path(
            "results/recurrence_modeling/seasonal_recurrence/"
            "event_global_seasonal_mapping.csv"
        ),
        low_memory=False,
    )
    events["valid_target_event"] = pd.to_numeric(
        events["valid_target_event"], errors="coerce"
    ).fillna(0).astype(int)
    observation_start, observation_end = _observation_bounds(cache_root)

    fold_rows: list[dict[str, Any]] = []
    fold_coefficients: list[dict[str, Any]] = []
    oof_eta: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_NAMES}
    oof_data_parts: list[SeasonalRiskData] = []
    for fold in range(number_folds):
        train = add_seasonal_state(
            load_current_risk_data(
                cache_root, f"cv_fold{fold}_train", maximum_train_rows, seed + fold * 101
            ),
            events,
            observation_start,
            observation_end,
        )
        validation = add_seasonal_state(
            load_current_risk_data(
                cache_root,
                f"cv_fold{fold}_validation",
                maximum_evaluation_rows,
                seed + fold * 101 + 1,
            ),
            events,
            observation_start,
            observation_end,
        )
        oof_data_parts.append(compact_data(validation))
        for model_name in MODEL_NAMES:
            design, model = _fit_model(model_name, train, settings)
            eta = model.decision_function(design.transform(validation))
            oof_eta[model_name].append(eta)
            fold_rows.extend(
                probability_metric_rows(
                    model_name, "2022_block_validation", eta, validation, None, fold
                )
            )
            for term, coefficient in zip(design.feature_names, model.coefficients):
                if term.startswith("state_x_") or term == "seasonal_recurrence_state":
                    fold_coefficients.append(
                        {
                            "fold": fold,
                            "model": model_name,
                            "term": term,
                            "coefficient": float(coefficient),
                        }
                    )
        print(f"Seasonal risk structure fold: {fold + 1}/{number_folds}", flush=True)

    pooled_data = concatenate_data(oof_data_parts)
    calibrators: dict[str, HazardRateCalibrator] = {}
    pooled_rows: list[dict[str, Any]] = []
    pooled_eta = {name: np.concatenate(parts) for name, parts in oof_eta.items()}
    for model_name in MODEL_NAMES:
        calibrator = HazardRateCalibrator().fit(
            pooled_eta[model_name], pooled_data.labels["step"], pooled_data.sample_weight
        )
        calibrators[model_name] = calibrator
        pooled_rows.extend(
            probability_metric_rows(
                model_name,
                "2022_pooled_oof",
                pooled_eta[model_name],
                pooled_data,
                calibrator,
                "pooled",
            )
        )

    training = add_seasonal_state(
        load_current_risk_data(
            cache_root, "selection_2022", maximum_train_rows, seed + 7001
        ),
        events,
        observation_start,
        observation_end,
    )
    trained_models: dict[str, tuple[SeasonalHazardDesign, WeightedBinaryGLM]] = {}
    bundles: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        design, model = _fit_model(model_name, training, settings)
        trained_models[model_name] = (design, model)
        bundles[model_name] = {
            "design": design.to_dict(),
            "glm": model.to_dict(),
            "calibrator": calibrators[model_name].to_dict(),
        }

    coefficient_rows: list[pd.DataFrame] = []
    joint_rows: list[dict[str, Any]] = []
    for model_name in ("weather_recurrence_interaction", "event_history"):
        design, model = trained_models[model_name]
        table, covariance = _cluster_sandwich(design, model, training)
        table["model"] = model_name
        coefficient_rows.append(table)
        joint = _joint_interaction_test(table, covariance)
        joint["model"] = model_name
        joint_rows.append(joint)

    evaluation_rows: list[dict[str, Any]] = []
    bootstrap_parts: list[pd.DataFrame] = []

    def evaluate_split(
        data: SeasonalRiskData,
        split_name: str,
        eta_by_model: dict[str, np.ndarray],
        split_seed: int,
    ) -> None:
        probability: dict[tuple[str, str, str], np.ndarray] = {}
        for model_name, eta in eta_by_model.items():
            evaluation_rows.extend(
                probability_metric_rows(
                    model_name, split_name, eta, data, calibrators[model_name], "frozen"
                )
            )
            probability[(model_name, "raw", "step")] = cloglog_probability(eta, steps=1)
            probability[(model_name, "calibrated_2022_oof", "6h")] = calibrators[
                model_name
            ].predict(eta, steps=36)
        for comparison_index, (model_a, model_b) in enumerate(NESTED_COMPARISONS):
            for mode_index, (calibration, horizon) in enumerate(
                (("raw", "step"), ("calibrated_2022_oof", "6h"))
            ):
                result = paired_station_loss_bootstrap(
                    data,
                    probability[(model_a, calibration, horizon)],
                    probability[(model_b, calibration, horizon)],
                    model_a,
                    model_b,
                    horizon,
                    calibration,
                    bootstrap_samples,
                    split_seed + comparison_index * 1009 + mode_index * 101,
                )
                result["split"] = split_name
                bootstrap_parts.append(result)

    evaluate_split(pooled_data, "2022_pooled_oof", pooled_eta, seed + 2200)
    for cache_split, display_split, year_seed in (
        ("cross_year_2023", "2023_cross_year", 2023),
        ("final_time_2024", "2024_final_time", 2024),
    ):
        data = add_seasonal_state(
            load_current_risk_data(
                cache_root, cache_split, maximum_evaluation_rows, seed + year_seed
            ),
            events,
            observation_start,
            observation_end,
        )
        eta_by_model = {
            model_name: model.decision_function(design.transform(data))
            for model_name, (design, model) in trained_models.items()
        }
        evaluate_split(data, display_split, eta_by_model, seed + year_seed * 17)
        print(f"Seasonal risk structure frozen evaluation: {display_split}", flush=True)

    pd.DataFrame(fold_rows).to_csv(output_root / "fold_metrics.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(output_root / "pooled_oof_metrics.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(output_root / "locked_year_metrics.csv", index=False)
    pd.concat(bootstrap_parts, ignore_index=True).to_csv(
        output_root / "paired_station_nested_bootstrap.csv", index=False
    )
    pd.concat(coefficient_rows, ignore_index=True).to_csv(
        output_root / "interaction_coefficients.csv", index=False
    )
    pd.DataFrame(joint_rows).to_csv(output_root / "joint_interaction_tests.csv", index=False)
    pd.DataFrame(fold_coefficients).to_csv(
        output_root / "fold_interaction_stability.csv", index=False
    )
    (output_root / "seasonal_risk_structure_bundle.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "config_path": str(config_path),
                "models": bundles,
                "nesting": {
                    "weather_only": "weather plus penalized station frailty",
                    "recurrence_state": (
                        "weather_only plus causal within-season recurrence state"
                    ),
                    "weather_recurrence_interaction": (
                        "recurrence_state plus five prespecified weather-by-state "
                        "interactions"
                    ),
                    "event_history": (
                        "weather_recurrence_interaction plus gap, previous-event "
                        "attributes, and 7/30-day load"
                    ),
                },
                "interaction_features": list(INTERACTION_FEATURES),
                "selection_protocol": "2022 purged block OOF only",
                "frozen_evaluation": [2023, 2024],
                "primary_test": (
                    "weather_recurrence_interaction minus recurrence_state paired "
                    "station bootstrap delta log loss"
                ),
                "primary_cohort": "complete seasons",
                "event_stages": list(EVENT_GROUPS),
                "bootstrap_samples": bootstrap_samples,
                "observation_start": str(observation_start),
                "observation_end": str(observation_end),
                "causal_state_rule": (
                    "At issue time, state=1 only after at least one valid event onset in the "
                    "same November-April icing season; active-event rows are outside the risk set."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = [
        "# Seasonal first-versus-recurrent risk structure test",
        "",
        "The weather-only, recurrence-state, weather-recurrence-interaction, and "
        "event-history models are nested complementary-log-log discrete hazard "
        "models. All state and history variables are causal at issue time. The "
        "interaction model versus the recurrence-state model is the prespecified "
        "test of different weather risk structure after seasonal recurrence.",
        "",
        "Primary evidence is the paired station-cluster bootstrap change in 2022 OOF raw one-step "
        "and calibrated 6-hour log loss, followed by frozen 2023 and retrospectively locked 2024 "
        "replication. Negative delta favors the expanded model.",
        "",
        "Interaction Wald tests use a ridge-adjusted station-cluster sandwich approximation and "
        "are supporting inference; they must be interpreted with the OOF and cross-year results.",
        "",
        "E1, E2, E3+ and all recurrent risk periods are reported separately. Complete icing "
        "seasons are primary; boundary-censored seasons are sensitivity analyses.",
    ]
    (output_root / "seasonal_risk_structure_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Seasonal risk structure experiment complete: {output_root}", flush=True)
    return output_root

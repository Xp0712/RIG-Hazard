from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .baseline_experiment import FeatureTransformer, read_timeline, timeline_files
from .baseline_models import WeightedBinaryGLM
from .naming import MAIN_MODEL_NAME, artifact_value
from .preprocessing import write_json


@dataclass
class LocalHazardBundle:
    transformer: FeatureTransformer
    model: WeightedBinaryGLM


@dataclass
class GraphSchema:
    feature_names: list[str]
    base_feature_count: int
    city_columns: dict[str, int]
    station_columns: dict[str, int]
    edge_features: pd.DataFrame
    seen_stations: set[str]

    @property
    def graph_feature_count(self) -> int:
        return len(self.feature_names) - self.base_feature_count


def load_local_hazard_bundle(baseline_root: Path) -> LocalHazardBundle:
    bundle = json.loads((baseline_root / "model_bundle.json").read_text(encoding="utf-8"))
    transformer_value = bundle["feature_transformer"]
    transformer = FeatureTransformer(
        continuous_features=list(transformer_value["continuous_features"]),
        binary_features=list(transformer_value["binary_features"]),
        means=np.asarray(transformer_value["means"], dtype=np.float64),
        stds=np.asarray(transformer_value["stds"], dtype=np.float64),
        clip_z=float(transformer_value["clip_z"]),
    )
    model = WeightedBinaryGLM.from_dict(
        artifact_value(bundle["models"], MAIN_MODEL_NAME)
    )
    return LocalHazardBundle(transformer=transformer, model=model)


def fit_source_signal_normalization(
    paths: list[Path],
    bundle: LocalHazardBundle,
    baseline_quantile: float,
    scale_quantile: float,
) -> dict[str, dict[str, float | int]]:
    if not 0 < baseline_quantile < scale_quantile < 1:
        raise ValueError("source signal quantiles must satisfy 0 < baseline < scale < 1")
    columns = [*bundle.transformer.feature_names, "station_code", "risk_set"]
    normalization: dict[str, dict[str, float | int]] = {}
    for index, path in enumerate(paths, start=1):
        frame = read_timeline(path, columns)
        risk = frame["risk_set"].eq(1)
        station_code = str(frame["station_code"].iloc[0])
        if risk.any():
            eta = bundle.model.decision_function(bundle.transformer.transform(frame.loc[risk]))
            low = float(np.quantile(eta, baseline_quantile))
            high = float(np.quantile(eta, scale_quantile))
            scale = max(high - low, 0.1)
            normalization[station_code] = {
                "baseline_eta": low,
                "scale_eta": scale,
                "scale_quantile_eta": high,
                "risk_rows": int(risk.sum()),
            }
        if index % 10 == 0 or index == len(paths):
            print(f"Fitted source normalization for {index}/{len(paths)} stations", flush=True)
    return normalization


def generate_source_signal_cache(
    paths: list[Path],
    cache_root: Path,
    bundle: LocalHazardBundle,
    normalization: dict[str, dict[str, float | int]],
    signal_clip: float,
    compression_level: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    columns = [*bundle.transformer.feature_names, "station_code", "issue_time", "risk_set"]
    for index, path in enumerate(paths, start=1):
        frame = read_timeline(path, columns)
        station_code = str(frame["station_code"].iloc[0])
        issue_time = pd.to_datetime(frame["issue_time"], errors="coerce")
        year = int(issue_time.dt.year.mode().iloc[0])
        signal = np.zeros(frame.shape[0], dtype=np.float32)
        risk = frame["risk_set"].eq(1).to_numpy()
        normalizer = normalization.get(station_code)
        if risk.any() and normalizer is not None:
            eta = bundle.model.decision_function(bundle.transformer.transform(frame.loc[risk]))
            values = (eta - float(normalizer["baseline_eta"])) / float(normalizer["scale_eta"])
            signal[risk] = np.clip(values, 0.0, signal_clip).astype(np.float32)
        destination = cache_root / str(year) / f"{station_code}.csv.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"issue_time": issue_time, "source_signal": signal}).to_csv(
            destination,
            index=False,
            encoding="utf-8-sig",
            compression={"method": "gzip", "compresslevel": compression_level},
        )
        summaries.append(
            {
                "station_code": station_code,
                "year": year,
                "rows": int(signal.size),
                "positive_signal_rows": int((signal > 0).sum()),
                "positive_signal_fraction": float((signal > 0).mean()),
                "mean_positive_signal": float(signal[signal > 0].mean()) if (signal > 0).any() else 0.0,
            }
        )
        if index % 10 == 0 or index == len(paths):
            print(f"Generated source signals for {index}/{len(paths)} station-years", flush=True)
    return summaries


class SignalBank:
    def __init__(self, cache_root: Path, year: int, step_minutes: int, rolling_window_minutes: int):
        if rolling_window_minutes % step_minutes:
            raise ValueError("rolling_window_minutes must be a multiple of step_minutes")
        self.cache_root = cache_root
        self.year = int(year)
        self.step_minutes = int(step_minutes)
        self.window_bins = max(1, int(rolling_window_minutes // step_minutes))
        self._series: dict[str, pd.Series] = {}
        self._lagged: dict[tuple[str, int], pd.Series] = {}

    def source_series(self, station_code: str) -> pd.Series:
        if station_code not in self._series:
            path = self.cache_root / str(self.year) / f"{station_code}.csv.gz"
            if not path.exists():
                self._series[station_code] = pd.Series(dtype=np.float32)
            else:
                frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["issue_time"])
                frame = frame.dropna(subset=["issue_time"]).sort_values("issue_time")
                self._series[station_code] = pd.Series(
                    pd.to_numeric(frame["source_signal"], errors="coerce").fillna(0).to_numpy(dtype=np.float32),
                    index=pd.DatetimeIndex(frame["issue_time"]),
                )
        return self._series[station_code]

    def lagged_values(self, station_code: str, lag_minutes: int, issue_times: pd.DatetimeIndex) -> np.ndarray:
        if lag_minutes <= 0 or lag_minutes % self.step_minutes:
            raise ValueError("graph lags must be positive multiples of the time step")
        key = (station_code, int(lag_minutes))
        if key not in self._lagged:
            source = self.source_series(station_code)
            lag_bins = int(lag_minutes // self.step_minutes)
            self._lagged[key] = source.rolling(self.window_bins, min_periods=1).max().shift(lag_bins)
        return self._lagged[key].reindex(issue_times).fillna(0).to_numpy(dtype=np.float32)


def build_graph_schema(
    station_catalog: pd.DataFrame,
    candidate_edges: pd.DataFrame,
    lags_minutes: list[int],
    candidate_neighbors: int,
    max_distance_km: float,
) -> GraphSchema:
    catalog = station_catalog.copy()
    catalog["station_code"] = catalog["station_code"].astype(str)
    catalog["city"] = catalog["city"].astype(str)
    seen_mask = pd.to_numeric(catalog["seen_in_development"], errors="coerce").fillna(0).eq(1)
    seen = set(catalog.loc[seen_mask, "station_code"])
    cities = sorted(catalog.loc[seen_mask, "city"].dropna().unique().tolist())
    stations = sorted(seen)

    feature_names = ["local::eta"]
    city_columns: dict[str, int] = {}
    for city in cities:
        city_columns[city] = len(feature_names)
        feature_names.append(f"barrier::city::{city}")
    station_columns: dict[str, int] = {}
    for station in stations:
        station_columns[station] = len(feature_names)
        feature_names.append(f"barrier::station::{station}")
    base_feature_count = len(feature_names)

    edges = candidate_edges.copy()
    for column in ["source_station_code", "target_station_code"]:
        edges[column] = edges[column].astype(str)
    edges["distance_rank"] = pd.to_numeric(edges["distance_rank"], errors="coerce")
    edges["distance_km"] = pd.to_numeric(edges["distance_km"], errors="coerce")
    edges = edges.loc[
        edges["source_station_code"].isin(seen)
        & edges["target_station_code"].isin(seen)
        & edges["distance_rank"].le(candidate_neighbors)
        & edges["distance_km"].le(max_distance_km)
    ].copy()
    edges = edges.sort_values(["target_station_code", "distance_rank", "source_station_code"])

    rows: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        for lag in sorted(set(int(value) for value in lags_minutes)):
            name = f"graph::{edge.source_station_code}->{edge.target_station_code}::lag{lag}m"
            column_index = len(feature_names)
            feature_names.append(name)
            rows.append(
                {
                    "feature_name": name,
                    "column_index": column_index,
                    "source_station_code": str(edge.source_station_code),
                    "source_station_name": str(edge.source_station_name),
                    "target_station_code": str(edge.target_station_code),
                    "target_station_name": str(edge.target_station_name),
                    "lag_minutes": lag,
                    "distance_km": float(edge.distance_km),
                    "distance_rank": int(edge.distance_rank),
                    "source_minus_target_elevation_m": edge.source_minus_target_elevation_m,
                    "same_city": int(edge.same_city),
                }
            )
    return GraphSchema(
        feature_names=feature_names,
        base_feature_count=base_feature_count,
        city_columns=city_columns,
        station_columns=station_columns,
        edge_features=pd.DataFrame(rows),
        seen_stations=seen,
    )


def build_structured_design(
    frame: pd.DataFrame,
    schema: GraphSchema,
    signal_bank: SignalBank,
    bundle: LocalHazardBundle,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    if frame.empty:
        return sparse.csr_matrix((0, len(schema.feature_names)), dtype=np.float64), np.empty(0, dtype=np.float64)
    station_code = str(frame["station_code"].iloc[0])
    city = str(frame["city"].iloc[0])
    issue_times = pd.DatetimeIndex(pd.to_datetime(frame["issue_time"], errors="coerce"))
    local_eta = bundle.model.decision_function(bundle.transformer.transform(frame))
    row_index = np.arange(frame.shape[0], dtype=np.int32)

    rows: list[np.ndarray] = [row_index]
    columns: list[np.ndarray] = [np.zeros(frame.shape[0], dtype=np.int32)]
    values: list[np.ndarray] = [local_eta.astype(np.float64)]
    city_column = schema.city_columns.get(city)
    if city_column is not None:
        rows.append(row_index)
        columns.append(np.full(frame.shape[0], city_column, dtype=np.int32))
        values.append(np.ones(frame.shape[0], dtype=np.float64))
    station_column = schema.station_columns.get(station_code)
    if station_column is not None:
        rows.append(row_index)
        columns.append(np.full(frame.shape[0], station_column, dtype=np.int32))
        values.append(np.ones(frame.shape[0], dtype=np.float64))

    target_edges = schema.edge_features.loc[schema.edge_features["target_station_code"].eq(station_code)]
    for edge in target_edges.itertuples(index=False):
        signal = signal_bank.lagged_values(str(edge.source_station_code), int(edge.lag_minutes), issue_times)
        active = np.flatnonzero(signal > 0)
        if active.size:
            rows.append(active.astype(np.int32))
            columns.append(np.full(active.size, int(edge.column_index), dtype=np.int32))
            values.append(signal[active].astype(np.float64))

    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(frame.shape[0], len(schema.feature_names)),
        dtype=np.float64,
    ).tocsr()
    return matrix, local_eta


def build_structured_training_sample(
    paths: list[Path],
    schema: GraphSchema,
    signal_cache_root: Path,
    bundle: LocalHazardBundle,
    split_column: str,
    split_name: str,
    max_horizon: int,
    sampling: dict[str, Any],
    step_minutes: int,
    rolling_window_minutes: int,
    return_metadata: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, Any]] | tuple[
    sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame
]:
    count_columns = [split_column, "risk_set", f"onset_within_{max_horizon}h", f"hard_negative_{max_horizon}h"]
    totals = {"near_event": 0, "hard_negative": 0, "easy_negative": 0}
    for index, path in enumerate(paths, start=1):
        frame = read_timeline(path, count_columns)
        frame = frame.loc[frame[split_column].eq(split_name) & frame["risk_set"].eq(1)]
        near = frame[f"onset_within_{max_horizon}h"].eq(1)
        hard = ~near & frame[f"hard_negative_{max_horizon}h"].eq(1)
        totals["near_event"] += int(near.sum())
        totals["hard_negative"] += int(hard.sum())
        totals["easy_negative"] += int((~near & ~hard).sum())
        if index % 10 == 0 or index == len(paths):
            print(f"Counted graph training strata in {index}/{len(paths)} files", flush=True)

    hard_probability = min(1.0, float(sampling["hard_negative_target"]) / max(totals["hard_negative"], 1))
    easy_probability = min(1.0, float(sampling["easy_negative_target"]) / max(totals["easy_negative"], 1))
    base_seed = int(sampling["seed"])
    read_columns = list(
        dict.fromkeys(
            [
                *bundle.transformer.feature_names,
                *count_columns,
                "hazard_label",
                "station_code",
                "city",
                "issue_time",
            ]
        )
    )
    matrix_parts: list[sparse.csr_matrix] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    selected_counts = {"near_event": 0, "hard_negative": 0, "easy_negative": 0}
    metadata_parts: list[pd.DataFrame] = []
    banks: dict[int, SignalBank] = {}

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
        selected_frame = frame.loc[selected].reset_index(drop=True)
        year = int(pd.to_datetime(selected_frame["issue_time"], errors="coerce").dt.year.mode().iloc[0])
        bank = banks.setdefault(year, SignalBank(signal_cache_root, year, step_minutes, rolling_window_minutes))
        matrix, _ = build_structured_design(selected_frame, schema, bank, bundle)
        matrix_parts.append(matrix)
        labels.append(pd.to_numeric(selected_frame["hazard_label"], errors="coerce").fillna(0).to_numpy(dtype=np.int8))
        row_weights = np.ones(int(selected.sum()), dtype=np.float64)
        row_weights[selected_hard[selected]] = 1.0 / hard_probability
        row_weights[selected_easy[selected]] = 1.0 / easy_probability
        weights.append(row_weights)
        if return_metadata:
            metadata = selected_frame[["station_code", "issue_time"]].copy()
            metadata["issue_time"] = pd.to_datetime(metadata["issue_time"], errors="coerce")
            metadata["station_month_block"] = (
                metadata["station_code"].astype(str)
                + "|"
                + metadata["issue_time"].dt.to_period("M").astype(str)
            )
            metadata_parts.append(metadata)
        selected_counts["near_event"] += int(near.sum())
        selected_counts["hard_negative"] += int(selected_hard.sum())
        selected_counts["easy_negative"] += int(selected_easy.sum())
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f"Built graph training design from {index + 1}/{len(paths)} files", flush=True)

    matrix = sparse.vstack(matrix_parts, format="csr")
    y = np.concatenate(labels)
    sample_weight = np.concatenate(weights)
    summary = {
        "stratum_totals": totals,
        "selected_counts": selected_counts,
        "hard_sampling_probability": hard_probability,
        "easy_sampling_probability": easy_probability,
        "sample_rows": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "base_feature_count": schema.base_feature_count,
        "graph_feature_count": schema.graph_feature_count,
        "matrix_nonzero": int(matrix.nnz),
        "matrix_density": float(matrix.nnz / max(matrix.shape[0] * matrix.shape[1], 1)),
        "hazard_positive_rows": int(y.sum()),
        "represented_weight_sum": float(sample_weight.sum()),
    }
    if return_metadata:
        return matrix, y, sample_weight, summary, pd.concat(metadata_parts, ignore_index=True)
    return matrix, y, sample_weight, summary


def prepare_source_signals(
    preprocessed_root: Path,
    output_root: Path,
    bundle: LocalHazardBundle,
    train_years: Iterable[int],
    all_years: Iterable[int],
    source_config: dict[str, Any],
    compression_level: int,
    allowed_station_codes: set[str] | None = None,
) -> tuple[Path, dict[str, dict[str, float | int]], pd.DataFrame]:
    def allowed(path: Path) -> bool:
        return allowed_station_codes is None or path.name.split("_", 1)[0] in allowed_station_codes

    train_paths = [path for path in timeline_files(preprocessed_root, train_years) if allowed(path)]
    normalization = fit_source_signal_normalization(
        train_paths,
        bundle,
        float(source_config["baseline_quantile"]),
        float(source_config["scale_quantile"]),
    )
    write_json(output_root / "source_signal_normalization.json", normalization)
    cache_root = output_root / "source_signals"
    summaries = generate_source_signal_cache(
        [path for path in timeline_files(preprocessed_root, all_years) if allowed(path)],
        cache_root,
        bundle,
        normalization,
        float(source_config["clip"]),
        compression_level,
    )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_root / "source_signal_summary.csv", index=False, encoding="utf-8-sig")
    return cache_root, normalization, summary_frame

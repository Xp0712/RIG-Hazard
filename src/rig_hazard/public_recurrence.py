from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PublicDatasetContract:
    name: str
    step_minutes: int
    horizon_bins: int
    train_end: str
    selection_end: str
    test_end: str
    budgets_hours: tuple[float, ...]

    @property
    def horizon_hours(self) -> float:
        return self.horizon_bins * self.step_minutes / 60.0


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def aggregate_ecommerce(
    source: str | Path,
    output: str | Path,
    maximum_entities: int,
) -> Path:
    import duckdb

    source_path = Path(source)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    source_sql = _sql_path(source_path)
    output_sql = _sql_path(output_path)
    columns = (
        "{'event_time':'VARCHAR','event_type':'VARCHAR','product_id':'VARCHAR',"
        "'category_id':'VARCHAR','category_code':'VARCHAR','brand':'VARCHAR',"
        "'price':'VARCHAR','user_id':'BIGINT','user_session':'VARCHAR'}"
    )
    relation = f"read_csv('{source_sql}', header=true, columns={columns}, null_padding=true)"
    connection = duckdb.connect()
    connection.execute(
        f"""
        CREATE TEMP TABLE selected_entities AS
        SELECT user_id
        FROM {relation}
        WHERE event_type = 'purchase' AND user_id IS NOT NULL
        GROUP BY user_id
        HAVING count(*) >= 2
        ORDER BY count(*) DESC, user_id
        LIMIT {int(maximum_entities)}
        """
    )
    connection.execute(
        f"""
        COPY (
          SELECT
            CAST(events.user_id AS VARCHAR) AS entity_id,
            date_trunc('hour', try_strptime(events.event_time, '%Y-%m-%d %H:%M:%S UTC')) AS issue_time,
            sum(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END)::INTEGER AS event_count,
            sum(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END)::INTEGER AS view_count,
            sum(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END)::INTEGER AS cart_count,
            sum(CASE WHEN event_type = 'remove_from_cart' THEN 1 ELSE 0 END)::INTEGER AS remove_count
          FROM {relation} AS events
          INNER JOIN selected_entities USING (user_id)
          GROUP BY events.user_id, issue_time
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.close()
    return output_path


def aggregate_us_accidents(
    source: str | Path,
    output: str | Path,
    maximum_entities: int,
    grid_degrees: float,
) -> Path:
    import duckdb

    source_path = Path(source)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    source_sql = _sql_path(source_path)
    output_sql = _sql_path(output_path)
    relation = f"read_csv_auto('{source_sql}', header=true, sample_size=200000, null_padding=true)"
    grid = float(grid_degrees)
    entity = (
        "concat(coalesce(State, 'NA'), '|', "
        f"printf('%.4f', floor(Start_Lat / {grid}) * {grid}), '|', "
        f"printf('%.4f', floor(Start_Lng / {grid}) * {grid}))"
    )
    connection = duckdb.connect()
    connection.execute(
        f"""
        CREATE TEMP TABLE selected_entities AS
        SELECT {entity} AS entity_id
        FROM {relation}
        WHERE Start_Lat IS NOT NULL AND Start_Lng IS NOT NULL
          AND try_cast(Start_Time AS TIMESTAMP) IS NOT NULL
        GROUP BY entity_id
        HAVING count(*) >= 5
        ORDER BY count(*) DESC, entity_id
        LIMIT {int(maximum_entities)}
        """
    )
    connection.execute(
        f"""
        COPY (
          SELECT
            {entity} AS entity_id,
            date_trunc('day', try_cast(Start_Time AS TIMESTAMP)) AS issue_time,
            count(*)::INTEGER AS event_count,
            avg(try_cast(Severity AS DOUBLE))::FLOAT AS severity_mean
          FROM {relation} AS accidents
          INNER JOIN selected_entities ON selected_entities.entity_id = {entity}
          GROUP BY 1, 2
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.close()
    return output_path


def read_parquet_with_duckdb(path: str | Path) -> pd.DataFrame:
    import duckdb

    return duckdb.sql(f"SELECT * FROM read_parquet('{_sql_path(Path(path))}')").df()


def build_complete_panel(
    aggregated: pd.DataFrame,
    step_minutes: int,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    frame = aggregated.copy()
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], errors="coerce")
    frame = frame.dropna(subset=["entity_id", "issue_time"])
    frequency = f"{int(step_minutes)}min"
    timeline = pd.date_range(start, end, freq=frequency, inclusive="left")
    count_columns = [
        column
        for column in ("event_count", "view_count", "cart_count", "remove_count")
        if column in frame
    ]
    extra_columns = [column for column in ("severity_mean",) if column in frame]
    parts: list[pd.DataFrame] = []
    for entity, group in frame.groupby("entity_id", sort=True):
        group = group.set_index("issue_time").sort_index()
        current = group.reindex(timeline)
        current["entity_id"] = str(entity)
        for column in count_columns:
            current[column] = pd.to_numeric(current[column], errors="coerce").fillna(0)
        for column in extra_columns:
            current[column] = pd.to_numeric(current[column], errors="coerce")
        current["issue_time"] = timeline
        parts.append(current.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def _time_since_previous_event(event: np.ndarray) -> np.ndarray:
    result = np.full(event.size, np.nan, dtype=np.float32)
    previous: int | None = None
    for index, value in enumerate(event):
        if previous is not None:
            result[index] = float(index - previous)
        if value > 0:
            previous = index
    return result


def add_causal_recurrence_features(
    panel: pd.DataFrame,
    horizon_bins: int,
) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.DataFrame] = []
    base_counts = [
        column
        for column in ("event_count", "view_count", "cart_count", "remove_count")
        if column in panel
    ]
    feature_names: list[str] = []
    for entity, group in panel.groupby("entity_id", sort=False):
        group = group.sort_values("issue_time").copy()
        event = pd.to_numeric(group["event_count"], errors="coerce").fillna(0)
        for column in base_counts:
            lag = pd.to_numeric(group[column], errors="coerce").fillna(0).shift(1).fillna(0)
            lag_name = f"{column}_lag1"
            group[lag_name] = lag
            if lag_name not in feature_names:
                feature_names.append(lag_name)
            for window in (6, 24, 168):
                name = f"{column}_past_{window}bins"
                group[name] = lag.rolling(window, min_periods=1).sum()
                if name not in feature_names:
                    feature_names.append(name)
        group["time_since_previous_event_bins"] = _time_since_previous_event(
            event.to_numpy(dtype=np.float64)
        )
        group["time_since_previous_event_missing"] = group[
            "time_since_previous_event_bins"
        ].isna().astype(np.int8)
        group["time_since_previous_event_bins"] = group[
            "time_since_previous_event_bins"
        ].fillna(1e4).clip(0, 1e4)
        prior_count = np.arange(group.shape[0], dtype=np.float64)
        prior_events = event.cumsum().shift(1).fillna(0).to_numpy(dtype=np.float64)
        group["prior_event_rate"] = np.divide(
            prior_events,
            np.maximum(prior_count, 1),
            out=np.zeros_like(prior_events),
            where=prior_count > 0,
        )
        times = pd.to_datetime(group["issue_time"])
        group["hour_sin"] = np.sin(2 * np.pi * times.dt.hour / 24.0)
        group["hour_cos"] = np.cos(2 * np.pi * times.dt.hour / 24.0)
        group["day_of_year_sin"] = np.sin(2 * np.pi * times.dt.dayofyear / 365.25)
        group["day_of_year_cos"] = np.cos(2 * np.pi * times.dt.dayofyear / 365.25)
        future = np.zeros(group.shape[0], dtype=np.int8)
        observed = np.zeros(group.shape[0], dtype=np.int8)
        event_array = event.to_numpy(dtype=np.float64) > 0
        for step in range(1, int(horizon_bins) + 1):
            shifted = np.r_[event_array[step:], np.zeros(step, dtype=bool)]
            future |= shifted.astype(np.int8)
        if group.shape[0] > horizon_bins:
            observed[: -int(horizon_bins)] = 1
        group["onset_within_horizon"] = future
        group["observed_horizon"] = observed
        parts.append(group)
    feature_names.extend(
        [
            "time_since_previous_event_bins",
            "time_since_previous_event_missing",
            "prior_event_rate",
            "hour_sin",
            "hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ]
    )
    return pd.concat(parts, ignore_index=True), list(dict.fromkeys(feature_names))


def event_table(panel: pd.DataFrame) -> pd.DataFrame:
    events = panel.loc[pd.to_numeric(panel["event_count"], errors="coerce").fillna(0).gt(0)].copy()
    events = events.sort_values(["entity_id", "issue_time"])
    events["event_order"] = events.groupby("entity_id").cumcount() + 1
    events["event_id"] = (
        events["entity_id"].astype(str)
        + "-E"
        + events["event_order"].astype(str).str.zfill(6)
    )
    events["valid_target_event"] = 1
    return events.rename(
        columns={"entity_id": "station_code", "issue_time": "onset_time"}
    )[["event_id", "station_code", "onset_time", "valid_target_event", "event_order"]]


def chronological_masks(
    frame: pd.DataFrame,
    contract: PublicDatasetContract,
) -> dict[str, np.ndarray]:
    time = pd.to_datetime(frame["issue_time"])
    horizon = pd.Timedelta(minutes=contract.horizon_bins * contract.step_minutes)
    train_end = pd.Timestamp(contract.train_end)
    selection_end = pd.Timestamp(contract.selection_end)
    test_end = pd.Timestamp(contract.test_end)
    return {
        "train": (time + horizon <= train_end).to_numpy(),
        "selection": ((time >= train_end) & (time + horizon <= selection_end)).to_numpy(),
        "test": ((time >= selection_end) & (time + horizon <= test_end)).to_numpy(),
    }


def fit_public_xgboost(
    frame: pd.DataFrame,
    feature_names: list[str],
    train_mask: np.ndarray,
    seed: int,
    maximum_training_rows: int,
    settings: dict[str, Any],
):
    from xgboost import XGBClassifier

    eligible = train_mask & frame["observed_horizon"].eq(1).to_numpy()
    positions = np.flatnonzero(eligible)
    if positions.size > int(maximum_training_rows):
        label = frame["onset_within_horizon"].to_numpy(dtype=np.int8)[positions]
        positive = positions[label == 1]
        negative = positions[label == 0]
        keep_negative = max(int(maximum_training_rows) - positive.size, 1)
        rng = np.random.default_rng(seed)
        negative = rng.choice(
            negative, size=min(negative.size, keep_negative), replace=False
        )
        positions = np.sort(np.r_[positive, negative])
    features = frame.loc[positions, feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    label = frame.loc[positions, "onset_within_horizon"].to_numpy(dtype=np.int8)
    negatives = max(int((label == 0).sum()), 1)
    positives = max(int((label == 1).sum()), 1)
    model = XGBClassifier(
        n_estimators=int(settings.get("n_estimators", 300)),
        max_depth=int(settings.get("max_depth", 6)),
        learning_rate=float(settings.get("learning_rate", 0.05)),
        subsample=float(settings.get("subsample", 0.8)),
        colsample_bytree=float(settings.get("colsample_bytree", 0.8)),
        min_child_weight=float(settings.get("min_child_weight", 5)),
        reg_lambda=float(settings.get("reg_lambda", 2.0)),
        scale_pos_weight=min(negatives / positives, 100.0),
        tree_method="hist",
        device=str(settings.get("device", "cuda")),
        n_jobs=int(settings.get("n_jobs", 8)),
        random_state=int(seed),
        eval_metric="logloss",
    )
    fit_features: Any = features
    fit_label: Any = label
    if str(settings.get("device", "cuda")).startswith("cuda"):
        try:
            import cupy as cp

            fit_features = cp.asarray(features.to_numpy(dtype=np.float32, copy=False))
            fit_label = cp.asarray(label)
        except ImportError:
            warnings.warn(
                "CuPy is unavailable; XGBoost will transfer CPU training data to CUDA.",
                RuntimeWarning,
                stacklevel=2,
            )
    model.fit(fit_features, fit_label)
    return model, {"training_rows": int(positions.size), "positive_rows": int(label.sum())}


def score_public_panel(
    model: Any,
    frame: pd.DataFrame,
    feature_names: list[str],
    batch_size: int = 250000,
) -> np.ndarray:
    result = np.zeros(frame.shape[0], dtype=np.float32)
    device = str(model.get_params().get("device", "cpu"))
    cupy: Any | None = None
    booster: Any | None = None
    if device.startswith("cuda"):
        try:
            import cupy as cp

            cupy = cp
            booster = model.get_booster()
        except ImportError:
            warnings.warn(
                "CuPy is unavailable; XGBoost CUDA inference will fall back to CPU input.",
                RuntimeWarning,
                stacklevel=2,
            )
    for start in range(0, frame.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), frame.shape[0])
        features = (
            frame.iloc[start:stop][feature_names]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        if cupy is not None and booster is not None:
            gpu_features = cupy.asarray(
                features.to_numpy(dtype=np.float32, copy=False)
            )
            gpu_prediction = booster.inplace_predict(
                gpu_features,
                validate_features=False,
            )
            result[start:stop] = cupy.asnumpy(gpu_prediction)
        else:
            result[start:stop] = model.predict_proba(features)[:, 1]
    return result

from __future__ import annotations

"""Paired station-cluster inference for frozen alert-controller comparisons."""

from typing import Iterable

import numpy as np
import pandas as pd


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return result
    order = finite[np.argsort(values[finite])]
    adjusted = 0.0
    total = order.size
    for rank, index in enumerate(order):
        adjusted = max(adjusted, (total - rank) * values[index])
        result[index] = min(adjusted, 1.0)
    return result


def _metric(group: pd.DataFrame, name: str) -> float:
    hit = pd.to_numeric(group["hit"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    lead = pd.to_numeric(group["effective_lead_hours"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    if name == "hit_rate":
        return float(hit.mean()) if hit.size else float("nan")
    if name == "mean_lead":
        selected = lead[hit > 0.5]
        return float(selected.mean()) if selected.size else 0.0
    if name == "lead_utility":
        return float(np.mean(hit * lead)) if hit.size else float("nan")
    raise ValueError(f"Unsupported event metric: {name}")


def paired_station_cluster_bootstrap(
    event_records: pd.DataFrame,
    method_a: str,
    method_b: str,
    samples: int = 5000,
    seed: int = 20260825,
    queue: str = "operational",
) -> pd.DataFrame:
    """Bootstrap paired differences while retaining each station as a cluster."""

    queue_column = "operational_evaluable" if queue == "operational" else "evaluable"
    required = {"method", "event_id", "station_code", queue_column, "hit", "effective_lead_hours"}
    missing = required.difference(event_records.columns)
    if missing:
        raise ValueError(f"Missing paired-bootstrap columns: {sorted(missing)}")
    left = event_records.loc[event_records["method"].eq(method_a)].copy()
    right = event_records.loc[event_records["method"].eq(method_b)].copy()
    keys = ["event_id", "station_code"]
    merged = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    merged = merged.loc[
        merged[f"{queue_column}_a"].eq(1) & merged[f"{queue_column}_b"].eq(1)
    ].copy()
    if merged.empty:
        raise ValueError("No paired evaluable events are available")
    stations = np.sort(merged["station_code"].astype(str).unique())
    station_parts = {
        station: merged.loc[merged["station_code"].astype(str).eq(station)]
        for station in stations
    }
    rng = np.random.default_rng(seed)
    metrics = ("hit_rate", "mean_lead", "lead_utility")
    draws = {metric: np.empty(samples, dtype=np.float64) for metric in metrics}
    for sample in range(samples):
        chosen = rng.choice(stations, size=stations.size, replace=True)
        sampled = pd.concat([station_parts[value] for value in chosen], ignore_index=True)
        for metric in metrics:
            a = sampled.rename(
                columns={"hit_a": "hit", "effective_lead_hours_a": "effective_lead_hours"}
            )
            b = sampled.rename(
                columns={"hit_b": "hit", "effective_lead_hours_b": "effective_lead_hours"}
            )
            draws[metric][sample] = _metric(a, metric) - _metric(b, metric)
    rows = []
    for metric in metrics:
        values = draws[metric]
        point_a = left.loc[left[queue_column].eq(1)]
        point_b = right.loc[right[queue_column].eq(1)]
        point = _metric(point_a, metric) - _metric(point_b, metric)
        p_value = min(1.0, 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean())))
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "queue": queue,
                "metric": metric,
                "difference": point,
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "p_value": p_value,
                "bootstrap_samples": int(samples),
                "stations": int(stations.size),
                "paired_events": int(merged.shape[0]),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_p_value"] = holm_adjust(result["p_value"])
    return result

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def station_cluster_bootstrap(
    event_records: pd.DataFrame,
    station_months: pd.DataFrame,
    *,
    samples: int = 5000,
    seed: int = 20260824,
    entity_column: str = "station_code",
) -> pd.DataFrame:
    """Bootstrap frozen alert metrics by resampling complete station clusters."""

    if int(samples) < 1:
        raise ValueError("samples must be positive")
    stations = sorted(
        set(event_records[entity_column].astype(str))
        | set(station_months[entity_column].astype(str))
    )
    if not stations:
        raise ValueError("At least one station cluster is required")
    rows: list[dict[str, Any]] = []
    for station in stations:
        events = event_records.loc[event_records[entity_column].astype(str).eq(station)]
        monthly = station_months.loc[station_months[entity_column].astype(str).eq(station)]
        standard = events.loc[pd.to_numeric(events["evaluable"], errors="coerce").fillna(0).eq(1)]
        operational = events.loc[
            pd.to_numeric(events["operational_evaluable"], errors="coerce").fillna(0).eq(1)
        ]
        standard_hits = standard.loc[pd.to_numeric(standard["hit"], errors="coerce").fillna(0).eq(1)]
        operational_hits = operational.loc[
            pd.to_numeric(operational["hit"], errors="coerce").fillna(0).eq(1)
        ]
        rows.append(
            {
                entity_column: station,
                "standard_events": int(standard.shape[0]),
                "standard_hits": int(standard_hits.shape[0]),
                "standard_lead_sum": float(
                    pd.to_numeric(
                        standard_hits["effective_lead_hours"], errors="coerce"
                    ).sum()
                ),
                "operational_events": int(operational.shape[0]),
                "operational_hits": int(operational_hits.shape[0]),
                "operational_lead_sum": float(
                    pd.to_numeric(
                        operational_hits["effective_lead_hours"], errors="coerce"
                    ).sum()
                ),
                "station_months": int(monthly.shape[0]),
                "false_alarm_hours_sum": float(
                    pd.to_numeric(monthly["false_alarm_hours"], errors="coerce").sum()
                ),
                "unsettled_alarm_hours_sum": float(
                    pd.to_numeric(monthly["unsettled_alarm_hours"], errors="coerce").sum()
                ),
                "reserved_budget_utilization_sum": float(
                    pd.to_numeric(
                        monthly["reserved_budget_utilization"], errors="coerce"
                    ).sum()
                ),
                "budget_exceeded_months": int(
                    pd.to_numeric(monthly["budget_exceeded"], errors="coerce").fillna(0).sum()
                ),
                "reserved_budget_exceeded_months": int(
                    pd.to_numeric(
                        monthly["reserved_budget_exceeded"], errors="coerce"
                    ).fillna(0).sum()
                ),
                "station_maximum_false_alarm_hours": float(
                    pd.to_numeric(monthly["false_alarm_hours"], errors="coerce").max()
                ),
            }
        )
    cluster = pd.DataFrame(rows).fillna(0)
    numeric = cluster.drop(columns=[entity_column]).to_numpy(dtype=np.float64)
    columns = {name: index for index, name in enumerate(cluster.columns[1:])}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(stations), size=(int(samples), len(stations)))
    sampled = numeric[draws]
    totals = sampled.sum(axis=1)
    standard_hit_rate = _safe_ratio(
        totals[:, columns["standard_hits"]], totals[:, columns["standard_events"]]
    )
    standard_mean_lead = _safe_ratio(
        totals[:, columns["standard_lead_sum"]], totals[:, columns["standard_hits"]]
    )
    operational_hit_rate = _safe_ratio(
        totals[:, columns["operational_hits"]], totals[:, columns["operational_events"]]
    )
    operational_mean_lead = _safe_ratio(
        totals[:, columns["operational_lead_sum"]], totals[:, columns["operational_hits"]]
    )
    station_month_count = totals[:, columns["station_months"]]
    distributions = {
        "strict_event_hit_rate": standard_hit_rate,
        "strict_mean_effective_lead_hours": standard_mean_lead,
        "strict_lead_utility_hours": standard_hit_rate * standard_mean_lead,
        "strict_operational_event_hit_rate": operational_hit_rate,
        "strict_operational_mean_effective_lead_hours": operational_mean_lead,
        "strict_operational_lead_utility_hours": operational_hit_rate
        * operational_mean_lead,
        "mean_false_alarm_hours": _safe_ratio(
            totals[:, columns["false_alarm_hours_sum"]], station_month_count
        ),
        "mean_unsettled_alarm_hours": _safe_ratio(
            totals[:, columns["unsettled_alarm_hours_sum"]], station_month_count
        ),
        "mean_reserved_budget_utilization": _safe_ratio(
            totals[:, columns["reserved_budget_utilization_sum"]], station_month_count
        ),
        "station_month_exceedance_rate": _safe_ratio(
            totals[:, columns["budget_exceeded_months"]], station_month_count
        ),
        "reserved_station_month_exceedance_rate": _safe_ratio(
            totals[:, columns["reserved_budget_exceeded_months"]], station_month_count
        ),
        "maximum_false_alarm_hours": sampled[
            :, :, columns["station_maximum_false_alarm_hours"]
        ].max(axis=1),
    }
    for metric, values in distributions.items():
        finite = values[np.isfinite(values)]
        rows = finite if finite.size else np.array([np.nan])
        yield_row = {
            "metric": metric,
            "bootstrap_samples": int(samples),
            "cluster_unit": "station",
            "clusters": len(stations),
            "ci_lower_2_5": float(np.nanquantile(rows, 0.025)),
            "ci_median": float(np.nanquantile(rows, 0.5)),
            "ci_upper_97_5": float(np.nanquantile(rows, 0.975)),
        }
        yield yield_row


def station_cluster_bootstrap_frame(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return pd.DataFrame(list(station_cluster_bootstrap(*args, **kwargs)))

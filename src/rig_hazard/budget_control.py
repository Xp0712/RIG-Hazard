from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalBudgetConfig:
    step_minutes: int = 10
    monthly_budget_hours: float = 10.0
    trailing_history_days: int = 30
    candidate_quantile: float = 0.98
    minimum_history_rows: int = 144
    burst_allowance_hours: float = 1.0
    minimum_candidate_run_bins: int = 2
    minimum_alarm_run_bins: int = 2

    @property
    def monthly_budget_bins(self) -> int:
        return int(np.floor(self.monthly_budget_hours * 60 / self.step_minutes))

    @property
    def burst_allowance_bins(self) -> int:
        return int(np.floor(self.burst_allowance_hours * 60 / self.step_minutes))


def causal_budget_alarms(
    issue_time: pd.Series | pd.DatetimeIndex,
    score: np.ndarray | pd.Series,
    config: CausalBudgetConfig,
) -> pd.DataFrame:
    """Create alarms using past scores only and a hard calendar-month cap."""

    times = pd.DatetimeIndex(pd.to_datetime(issue_time, errors="coerce"))
    values = np.asarray(score, dtype=np.float64)
    if times.size != values.size:
        raise ValueError("issue_time and score must have the same length")
    if times.hasnans or not times.is_monotonic_increasing:
        raise ValueError("issue_time must be valid and sorted")
    if not 0 < config.candidate_quantile < 1:
        raise ValueError("candidate_quantile must be between zero and one")
    if config.monthly_budget_bins < 1:
        raise ValueError("monthly budget must contain at least one time bin")

    score_series = pd.Series(values, index=times)
    threshold = (
        score_series.rolling(
            f"{int(config.trailing_history_days)}D",
            min_periods=int(config.minimum_history_rows),
            closed="both",
        )
        .quantile(float(config.candidate_quantile))
        .shift(1)
    )
    candidate = values > threshold.to_numpy(dtype=np.float64)
    candidate &= np.isfinite(values) & np.isfinite(threshold.to_numpy(dtype=np.float64))

    if config.minimum_candidate_run_bins < 1 or config.minimum_alarm_run_bins < 1:
        raise ValueError("minimum run lengths must be positive")
    alarms = np.zeros(values.size, dtype=np.int8)
    periods = times.to_period("M")
    month_codes = periods.asi8
    time_ns = times.asi8
    month_start_ns = periods.start_time.asi8
    next_month_start_ns = (periods + 1).start_time.asi8
    elapsed_ns = np.maximum(time_ns - month_start_ns, 0)
    duration_ns = np.maximum(next_month_start_ns - month_start_ns, 1)
    paced_allowance = (
        np.floor(config.monthly_budget_bins * elapsed_ns / duration_ns).astype(np.int64)
        + config.burst_allowance_bins
    )
    allowance = np.minimum(
        config.monthly_budget_bins,
        np.maximum(config.burst_allowance_bins, paced_allowance),
    )

    current_month_code: int | None = None
    used = 0
    candidate_run = 0
    latched_remaining = 0
    for index in range(values.size):
        month_code = int(month_codes[index])
        eligible = bool(candidate[index])
        if month_code != current_month_code:
            current_month_code = month_code
            used = 0
            candidate_run = 0
            latched_remaining = 0
        allowed = int(allowance[index])
        if latched_remaining > 0:
            if used < allowed:
                alarms[index] = 1
                used += 1
                latched_remaining -= 1
            else:
                latched_remaining = 0
            candidate_run = candidate_run + 1 if eligible else 0
            continue
        candidate_run = candidate_run + 1 if eligible else 0
        if candidate_run >= config.minimum_candidate_run_bins and used < allowed:
            alarms[index] = 1
            used += 1
            latched_remaining = config.minimum_alarm_run_bins - 1

    return pd.DataFrame(
        {
            "issue_time": times,
            "causal_threshold": threshold.to_numpy(dtype=np.float64),
            "candidate_alarm": candidate.astype(np.int8),
            "budget_alarm": alarms,
        }
    )


def apply_causal_budget_by_station(
    frame: pd.DataFrame,
    score_columns: list[str],
    config: CausalBudgetConfig,
) -> pd.DataFrame:
    required = {"station_code", "issue_time", *score_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing budget-control columns: {sorted(missing)}")
    output_parts: list[pd.DataFrame] = []
    for _, station_frame in frame.groupby("station_code", sort=False):
        station_frame = station_frame.copy()
        station_frame["issue_time"] = pd.to_datetime(station_frame["issue_time"], errors="coerce")
        station_frame = station_frame.sort_values("issue_time")
        for column in score_columns:
            controlled = causal_budget_alarms(
                station_frame["issue_time"],
                pd.to_numeric(station_frame[column], errors="coerce").to_numpy(dtype=np.float64),
                config,
            )
            station_frame[f"{column}__causal_threshold"] = controlled["causal_threshold"].to_numpy()
            station_frame[f"{column}__candidate_alarm"] = controlled["candidate_alarm"].to_numpy(dtype=np.int8)
            station_frame[f"{column}__budget_alarm"] = controlled["budget_alarm"].to_numpy(dtype=np.int8)
        output_parts.append(station_frame)
    return pd.concat(output_parts, ignore_index=True)

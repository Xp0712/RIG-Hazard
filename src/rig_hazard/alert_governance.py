from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


STRICT_EVENT_CONTRACT_VERSION = "strict_event_grid_reset_2026-08-25"


@dataclass(frozen=True)
class CandidatePolicy:
    strategy: str
    threshold: float = 0.5
    rolling_quantile: float = 0.98
    trailing_history_days: int = 30
    minimum_history_rows: int = 144
    ema_alpha: float = 1.0
    off_threshold_ratio: float = 0.8
    minimum_consecutive_bins: int = 1
    hold_bins: int = 1

    def validate(self) -> None:
        if self.strategy not in {"fixed_threshold", "rolling_quantile", "hysteresis"}:
            raise ValueError(f"Unsupported candidate strategy: {self.strategy}")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0 < self.rolling_quantile < 1:
            raise ValueError("rolling_quantile must be in (0, 1)")
        if not 0 < self.off_threshold_ratio <= 1:
            raise ValueError("off_threshold_ratio must be in (0, 1]")
        if self.minimum_consecutive_bins < 1 or self.hold_bins < 1:
            raise ValueError("run and hold lengths must be positive")


@dataclass(frozen=True)
class SafeBudgetPolicy:
    step_minutes: int
    budget_hours: float
    burst_allowance_hours: float = 0.5
    pace_multiplier: float = 1.0

    @property
    def budget_bins(self) -> int:
        return int(np.floor(self.budget_hours * 60.0 / self.step_minutes))

    @property
    def burst_bins(self) -> int:
        return int(np.floor(self.burst_allowance_hours * 60.0 / self.step_minutes))

    def validate(self) -> None:
        if self.step_minutes < 1 or self.budget_bins < 1:
            raise ValueError("The budget must contain at least one time bin")
        if self.burst_allowance_hours < 0 or self.pace_multiplier <= 0:
            raise ValueError("Burst allowance must be non-negative and pacing positive")


def causal_ema(values: np.ndarray | pd.Series, alpha: float) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not 0 < float(alpha) <= 1:
        raise ValueError("alpha must be in (0, 1]")
    result = np.full(raw.size, np.nan, dtype=np.float64)
    previous: float | None = None
    for index, value in enumerate(raw):
        if not np.isfinite(value):
            previous = None
            continue
        previous = (
            float(value)
            if previous is None
            else float(alpha) * float(value) + (1.0 - float(alpha)) * previous
        )
        result[index] = previous
    return result


def past_only_rolling_threshold(
    issue_time: pd.Series | pd.DatetimeIndex,
    values: np.ndarray | pd.Series,
    quantile: float,
    history_days: int,
    minimum_rows: int,
) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(issue_time, errors="coerce"))
    scores = np.asarray(values, dtype=np.float64)
    if times.hasnans or not times.is_monotonic_increasing or times.size != scores.size:
        raise ValueError("issue_time must be valid, sorted, and aligned with values")
    threshold = (
        pd.Series(scores, index=times)
        .rolling(
            f"{int(history_days)}D",
            min_periods=max(int(minimum_rows), 1),
            closed="both",
        )
        .quantile(float(quantile))
        .shift(1)
    )
    return threshold.to_numpy(dtype=np.float64)


def causal_hysteresis_mask(
    values: np.ndarray | pd.Series,
    on_threshold: float | np.ndarray,
    policy: CandidatePolicy,
) -> tuple[np.ndarray, np.ndarray]:
    policy.validate()
    smoothed = causal_ema(values, policy.ema_alpha)
    on = np.asarray(on_threshold, dtype=np.float64)
    if on.ndim == 0:
        on = np.full(smoothed.size, float(on), dtype=np.float64)
    if on.shape != smoothed.shape:
        raise ValueError("on_threshold must be scalar or aligned with values")
    off = on * float(policy.off_threshold_ratio)
    alarm = np.zeros(smoothed.size, dtype=np.int8)
    run = 0
    active = False
    hold_remaining = 0
    for index, score in enumerate(smoothed):
        if not np.isfinite(score) or not np.isfinite(on[index]):
            run = 0
            active = False
            hold_remaining = 0
            continue
        run = run + 1 if score >= on[index] else 0
        if not active and run >= int(policy.minimum_consecutive_bins):
            active = True
            hold_remaining = int(policy.hold_bins) - 1
        if active:
            alarm[index] = 1
            if score >= off[index]:
                hold_remaining = max(hold_remaining, int(policy.hold_bins) - 1)
            elif hold_remaining > 0:
                hold_remaining -= 1
            else:
                active = False
                run = 0
    return alarm, smoothed


def candidate_alarm_mask(
    issue_time: pd.Series | pd.DatetimeIndex,
    values: np.ndarray | pd.Series,
    policy: CandidatePolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy.validate()
    scores = np.asarray(values, dtype=np.float64)
    if policy.strategy == "rolling_quantile":
        threshold = past_only_rolling_threshold(
            issue_time,
            scores,
            policy.rolling_quantile,
            policy.trailing_history_days,
            policy.minimum_history_rows,
        )
    else:
        threshold = np.full(scores.size, float(policy.threshold), dtype=np.float64)
    if policy.strategy in {"fixed_threshold", "rolling_quantile"}:
        smoothed = causal_ema(scores, policy.ema_alpha)
        alarm = (
            np.isfinite(smoothed)
            & np.isfinite(threshold)
            & (smoothed >= threshold)
        ).astype(np.int8)
    else:
        alarm, smoothed = causal_hysteresis_mask(scores, threshold, policy)
    return alarm, smoothed, threshold


def causal_monthly_budget_cap(
    issue_time: pd.Series | pd.DatetimeIndex,
    candidate: np.ndarray | pd.Series,
    policy: SafeBudgetPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a past-only, per-entity calendar-month hard cap to candidate alarms."""

    policy.validate()
    times = pd.DatetimeIndex(pd.to_datetime(issue_time, errors="coerce"))
    candidates = np.asarray(candidate, dtype=bool)
    if times.hasnans or not times.is_monotonic_increasing or times.size != candidates.size:
        raise ValueError("issue_time must be valid, sorted, and aligned with candidate")
    periods = times.to_period("M")
    elapsed = np.maximum(times.asi8 - periods.start_time.asi8, 0)
    duration = np.maximum((periods + 1).start_time.asi8 - periods.start_time.asi8, 1)
    paced = np.floor(
        policy.budget_bins * policy.pace_multiplier * elapsed / duration
    ).astype(np.int64)
    allowance = np.minimum(
        policy.budget_bins,
        np.maximum(policy.burst_bins, paced + policy.burst_bins),
    )
    alarm = np.zeros(times.size, dtype=np.int8)
    month_code: int | None = None
    used = 0
    for index, eligible in enumerate(candidates):
        code = int(periods.asi8[index])
        if code != month_code:
            month_code = code
            used = 0
        if eligible and used < int(allowance[index]) and used < policy.budget_bins:
            alarm[index] = 1
            used += 1
    return alarm, allowance


def apply_candidate_policy_by_entity(
    frame: pd.DataFrame,
    score_column: str,
    candidate_policy: CandidatePolicy,
    entity_column: str = "station_code",
    time_column: str = "issue_time",
    safe_budget: SafeBudgetPolicy | None = None,
) -> pd.DataFrame:
    required = {entity_column, time_column, score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing controller columns: {sorted(missing)}")
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(entity_column, sort=False):
        group = group.copy()
        group[time_column] = pd.to_datetime(group[time_column], errors="coerce")
        group = group.sort_values(time_column)
        candidate, smoothed, threshold = candidate_alarm_mask(
            group[time_column],
            pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=np.float64),
            candidate_policy,
        )
        group["candidate_alarm"] = candidate
        group["causal_smoothed_score"] = smoothed
        group["causal_threshold"] = threshold
        if safe_budget is None:
            group["budget_alarm"] = candidate
            group["causal_budget_allowance_bins"] = np.nan
        else:
            alarm, allowance = causal_monthly_budget_cap(
                group[time_column], candidate, safe_budget
            )
            group["budget_alarm"] = alarm
            group["causal_budget_allowance_bins"] = allowance
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _segments(
    times: pd.Series,
    alarm: np.ndarray,
    step_minutes: int,
    reset_times: pd.Series | pd.DatetimeIndex | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    index = np.flatnonzero(np.asarray(alarm, dtype=bool))
    if index.size == 0:
        return []
    timestamps = pd.DatetimeIndex(pd.to_datetime(times, errors="coerce"))
    gap_ns = int(step_minutes * 60 * 1_000_000_000)
    breaks = np.r_[True, np.diff(timestamps.asi8[index]) > gap_ns]
    if reset_times is not None and index.size > 1:
        resets = pd.DatetimeIndex(pd.to_datetime(reset_times, errors="coerce")).dropna()
        if resets.size:
            reset_ns = np.sort(resets.asi8)
            previous_alarm_ns = timestamps.asi8[index[:-1]]
            current_alarm_ns = timestamps.asi8[index[1:]]
            # The online controller observes an onset before making the action
            # at that issue time.  Therefore an onset in (previous, current]
            # terminates the previous segment and the current alarm starts a
            # new one, even when the two alarm bins are grid-consecutive.
            reset_after_previous = np.searchsorted(
                reset_ns, previous_alarm_ns, side="right"
            )
            reset_through_current = np.searchsorted(
                reset_ns, current_alarm_ns, side="right"
            )
            breaks[1:] |= reset_through_current > reset_after_previous
    starts = np.flatnonzero(breaks)
    stops = np.r_[starts[1:] - 1, index.size - 1]
    return [
        {
            "start": pd.Timestamp(timestamps[index[start]]),
            "end": pd.Timestamp(timestamps[index[stop]]),
            "positions": index[start : stop + 1],
        }
        for start, stop in zip(starts, stops)
    ]


def strict_pre_event_prediction_grid(
    onset_time: Any,
    step_minutes: int = 10,
    horizon_hours: float = 6.0,
) -> pd.DatetimeIndex:
    """Return the exact prediction grid strictly preceding an event onset.

    Event onsets are allowed to occur between prediction-grid timestamps.  The
    final prediction is the last grid point strictly before the onset, rather
    than ``onset - step``.  This preserves a complete, fixed-length horizon
    without silently requiring event timestamps to be grid-aligned.
    """

    onset = pd.Timestamp(onset_time)
    if pd.isna(onset):
        raise ValueError("onset_time must be a valid timestamp")
    if int(step_minutes) < 1 or float(horizon_hours) <= 0:
        raise ValueError("step_minutes and horizon_hours must be positive")
    expected_bins_float = float(horizon_hours) * 60.0 / int(step_minutes)
    expected_bins = int(round(expected_bins_float))
    if not np.isclose(expected_bins_float, expected_bins):
        raise ValueError("horizon_hours must contain an integer number of prediction bins")

    step_ns = int(step_minutes) * 60 * 1_000_000_000
    onset_ns = int(onset.value)
    window_end_ns = ((onset_ns - 1) // step_ns) * step_ns
    offsets = np.arange(expected_bins - 1, -1, -1, dtype=np.int64) * step_ns
    grid_ns = window_end_ns - offsets
    if onset.tzinfo is None:
        grid = pd.DatetimeIndex(pd.to_datetime(grid_ns, unit="ns"))
    else:
        grid = pd.DatetimeIndex(pd.to_datetime(grid_ns, unit="ns", utc=True)).tz_convert(
            onset.tz
        )

    expected_delta = pd.Timedelta(minutes=int(step_minutes))
    horizon = pd.Timedelta(hours=float(horizon_hours))
    leads = onset - grid
    invariants_hold = bool(
        grid.size == expected_bins
        and grid[-1] < onset
        and onset - grid[-1] <= expected_delta
        and grid[0] == grid[-1] - (expected_bins - 1) * expected_delta
        and np.all(np.diff(grid.asi8) == expected_delta.value)
        and np.all(leads > pd.Timedelta(0))
        and np.all(leads <= horizon)
    )
    if not invariants_hold:
        raise RuntimeError("Strict pre-event prediction-grid invariants were violated")
    return grid


def strict_event_alert_evaluation(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    alarm_column: str = "budget_alarm",
    entity_column: str = "station_code",
    time_column: str = "issue_time",
    event_time_column: str = "onset_time",
    event_id_column: str = "event_id",
    step_minutes: int = 10,
    horizon_hours: float = 6.0,
    observability_column: str | None = "observed_6h",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Grid-anchored one-segment/one-event matching under a single event contract.

    The standardized queue contains events with all expected prediction bins.
    The operational queue contains events with at least one legal pre-event
    prediction.  Known events can be matched even when their cached future
    label is right-censored.  Unmatched alarm bins are confirmed false only
    after a complete observable horizon; otherwise they remain unsettled and
    reserve budget capacity.
    """

    required = {entity_column, time_column, alarm_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing strict evaluation columns: {sorted(missing)}")
    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working.dropna(subset=[time_column]).sort_values([entity_column, time_column])
    working["_alarm"] = pd.to_numeric(working[alarm_column], errors="coerce").fillna(0).ge(0.5)
    working["_strict_true_alarm"] = False
    if observability_column is not None and observability_column in working:
        working["_horizon_observed"] = pd.to_numeric(
            working[observability_column], errors="coerce"
        ).fillna(0).ge(0.5)
    else:
        working["_horizon_observed"] = True
    event_frame = events.copy()
    event_frame[event_time_column] = pd.to_datetime(
        event_frame[event_time_column], errors="coerce"
    )
    if "valid_target_event" in event_frame:
        event_frame = event_frame.loc[
            pd.to_numeric(event_frame["valid_target_event"], errors="coerce").fillna(0).eq(1)
        ]
    event_frame = event_frame.dropna(subset=[event_time_column])
    horizon = pd.Timedelta(hours=float(horizon_hours))
    expected_bins = int(round(float(horizon_hours) * 60.0 / step_minutes))
    records: list[dict[str, Any]] = []
    segment_total = 0
    matched_segment_keys: set[str] = set()
    working_groups = {
        str(entity): group.sort_values(time_column)
        for entity, group in working.groupby(entity_column, sort=False)
    }
    event_groups = {
        str(entity): group.sort_values(event_time_column)
        for entity, group in event_frame.groupby(entity_column, sort=False)
    }
    for entity, station_events in event_groups.items():
        group = working_groups.get(entity)
        if group is None:
            group_indices = np.array([], dtype=np.int64)
            times = pd.DatetimeIndex([])
            segments: list[dict[str, Any]] = []
            observed = np.array([], dtype=bool)
        else:
            group_indices = group.index.to_numpy()
            times = pd.DatetimeIndex(group[time_column])
            segments = _segments(
                group[time_column],
                group["_alarm"].to_numpy(),
                step_minutes,
                reset_times=station_events[event_time_column],
            )
            observed = group["_horizon_observed"].to_numpy(dtype=bool)
        local_records: list[dict[str, Any]] = []
        for event in station_events.itertuples(index=False):
            onset = pd.Timestamp(getattr(event, event_time_column))
            expected_grid = strict_pre_event_prediction_grid(
                onset, step_minutes=step_minutes, horizon_hours=horizon_hours
            )
            window_positions = np.flatnonzero(
                (times >= expected_grid[0]) & (times <= expected_grid[-1])
            )
            available_times = times[window_positions]
            complete = bool(
                available_times.size == expected_bins
                and np.array_equal(available_times.asi8, expected_grid.asi8)
            )
            available_bins = int(available_times.size)
            operational = available_bins > 0
            observed_bins = int(observed[window_positions].sum()) if available_bins else 0
            reasons: list[str] = []
            if group is None:
                reasons.append("station_id_not_matched")
            elif not operational:
                reasons.append("no_legal_risk_prediction_in_window")
            elif not complete:
                reasons.append("incomplete_legal_prediction_window")
            record = {
                "event_id": str(getattr(event, event_id_column)),
                entity_column: str(entity),
                "onset_time": onset,
                "window_start": expected_grid[0],
                "window_end": expected_grid[-1],
                "expected_prediction_bins": expected_bins,
                "available_prediction_bins": available_bins,
                "observed_prediction_bins": observed_bins,
                "right_censored_prediction_bins": available_bins - observed_bins,
                "available_warning_window_hours": (
                    float((onset - available_times[0]).total_seconds() / 3600.0)
                    if operational
                    else 0.0
                ),
                "evaluable": int(complete),
                "standardized_queue": int(complete),
                "operational_evaluable": int(operational),
                "operational_queue": int(operational),
                "primary_exclusion_reason": reasons[0] if reasons else "",
                "all_exclusion_reasons": ";".join(reasons),
                "known_event_overrides_horizon_censoring": int(
                    operational and observed_bins < available_bins
                ),
                "hit": 0,
                "segment_id": "",
                "effective_lead_hours": float("nan"),
                "carry_in_from_before_window": 0,
            }
            for metadata_column in (
                "reference_issue_time",
                "eligible_hazard_label",
                "model_eligibility_exclusion_reason",
                "within_season_event_order",
                "global_event_order",
                "seasonal_event_class",
                "icing_season_start_year",
                "end_time",
            ):
                if hasattr(event, metadata_column):
                    record[metadata_column] = getattr(event, metadata_column)
            local_records.append(record)
        segment_total += len(segments)
        for segment_index, segment in enumerate(segments):
            candidates = [
                row
                for row in local_records
                if row["operational_evaluable"] == 1
                and row["hit"] == 0
                and segment["start"] < row["onset_time"]
                and segment["end"] >= row["onset_time"] - horizon
            ]
            if not candidates:
                continue
            selected = min(candidates, key=lambda row: row["onset_time"])
            onset = selected["onset_time"]
            segment_id = f"{entity}:{segment_index}"
            if segment_id in matched_segment_keys:
                raise RuntimeError("A strict alert segment was matched more than once")
            matched_segment_keys.add(segment_id)
            first_eligible = max(segment["start"], onset - horizon)
            selected.update(
                {
                    "hit": 1,
                    "segment_id": segment_id,
                    "effective_lead_hours": float(
                        (onset - first_eligible).total_seconds() / 3600.0
                    ),
                    "carry_in_from_before_window": int(segment["start"] < onset - horizon),
                }
            )
            local_positions = np.asarray(segment["positions"], dtype=np.int64)
            local_times = times[local_positions]
            true_positions = local_positions[(local_times >= onset - horizon) & (local_times < onset)]
            working.loc[group_indices[true_positions], "_strict_true_alarm"] = True
        records.extend(local_records)
    for entity, group in working_groups.items():
        if entity not in event_groups:
            segment_total += len(
                _segments(group[time_column], group["_alarm"].to_numpy(), step_minutes)
            )
    event_records = pd.DataFrame(records)
    evaluable = event_records.loc[event_records["evaluable"].eq(1)] if not event_records.empty else event_records
    hits = evaluable.loc[evaluable["hit"].eq(1)] if not evaluable.empty else evaluable
    operational = (
        event_records.loc[event_records["operational_evaluable"].eq(1)]
        if not event_records.empty
        else event_records
    )
    operational_hits = (
        operational.loc[operational["hit"].eq(1)] if not operational.empty else operational
    )
    alarm = working["_alarm"].to_numpy(dtype=bool)
    strict_true = working["_strict_true_alarm"].to_numpy(dtype=bool)
    horizon_observed = working["_horizon_observed"].to_numpy(dtype=bool)
    working["_strict_false_alarm"] = alarm & ~strict_true & horizon_observed
    working["_strict_unsettled_alarm"] = alarm & ~strict_true & ~horizon_observed
    working["_strict_reserved_alarm"] = (
        working["_strict_false_alarm"] | working["_strict_unsettled_alarm"]
    )
    hit_rate = float(evaluable["hit"].mean()) if not evaluable.empty else float("nan")
    mean_lead = float(hits["effective_lead_hours"].mean()) if not hits.empty else float("nan")
    operational_hit_rate = (
        float(operational["hit"].mean()) if not operational.empty else float("nan")
    )
    operational_mean_lead = (
        float(operational_hits["effective_lead_hours"].mean())
        if not operational_hits.empty
        else float("nan")
    )
    if not operational_hits.empty:
        leads = operational_hits["effective_lead_hours"].to_numpy(dtype=np.float64)
        if np.any(leads <= 0) or np.any(leads > float(horizon_hours) + 1e-12):
            raise RuntimeError("Matched-event lead times violate the strict event contract")
    metrics = {
        "evaluation_contract_version": STRICT_EVENT_CONTRACT_VERSION,
        "target_events": int(event_frame.shape[0]),
        "evaluable_events": int(evaluable.shape[0]),
        "hit_events": int(hits.shape[0]),
        "event_hit_rate": hit_rate,
        "mean_effective_lead_hours": mean_lead,
        "median_effective_lead_hours": (
            float(hits["effective_lead_hours"].median()) if not hits.empty else float("nan")
        ),
        "lead_utility_hours": (
            hit_rate * mean_lead if np.isfinite(hit_rate) and np.isfinite(mean_lead) else 0.0
        ),
        "operational_evaluable_events": int(operational.shape[0]),
        "operational_hit_events": int(operational_hits.shape[0]),
        "operational_event_hit_rate": operational_hit_rate,
        "operational_mean_effective_lead_hours": operational_mean_lead,
        "operational_median_effective_lead_hours": (
            float(operational_hits["effective_lead_hours"].median())
            if not operational_hits.empty
            else float("nan")
        ),
        "operational_lead_utility_hours": (
            operational_hit_rate * operational_mean_lead
            if np.isfinite(operational_hit_rate) and np.isfinite(operational_mean_lead)
            else 0.0
        ),
        "alert_segments": int(segment_total),
        "matched_segments": int(len(matched_segment_keys)),
        "duplicate_event_matches": int(
            hits.duplicated([event_id_column, entity_column]).sum()
        ) if not hits.empty else 0,
        "duplicate_segment_matches": int(hits["segment_id"].duplicated().sum()) if not hits.empty else 0,
        "carry_in_hits": int(hits["carry_in_from_before_window"].sum()) if not hits.empty else 0,
        "strict_false_alarm_bins": int(working["_strict_false_alarm"].sum()),
        "strict_unsettled_alarm_bins": int(working["_strict_unsettled_alarm"].sum()),
        "strict_reserved_alarm_bins": int(working["_strict_reserved_alarm"].sum()),
    }
    return metrics, event_records, working


def station_month_budget_distribution(
    frame: pd.DataFrame,
    alarm_column: str,
    budget_hours: float,
    step_minutes: int,
    entity_column: str = "station_code",
    time_column: str = "issue_time",
    future_column: str = "onset_within_6h",
    strict_false_column: str | None = None,
    unsettled_column: str | None = None,
    reserved_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working["station_month"] = (
        working[entity_column].astype(str)
        + "|"
        + working[time_column].dt.to_period("M").astype(str)
    )
    alarm = pd.to_numeric(working[alarm_column], errors="coerce").fillna(0).ge(0.5)
    if strict_false_column is not None and strict_false_column in working:
        false_alarm = pd.to_numeric(
            working[strict_false_column], errors="coerce"
        ).fillna(0).ge(0.5)
    else:
        future = pd.to_numeric(working[future_column], errors="coerce").fillna(0).ge(0.5)
        false_alarm = alarm & ~future
    if unsettled_column is not None and unsettled_column in working:
        unsettled = pd.to_numeric(
            working[unsettled_column], errors="coerce"
        ).fillna(0).ge(0.5)
    else:
        unsettled = pd.Series(False, index=working.index)
    if reserved_column is not None and reserved_column in working:
        reserved = pd.to_numeric(
            working[reserved_column], errors="coerce"
        ).fillna(0).ge(0.5)
    else:
        reserved = false_alarm | unsettled
    working["_alarm_bins"] = alarm.astype(np.int64)
    working["_false_alarm_bins"] = false_alarm.astype(np.int64)
    working["_unsettled_alarm_bins"] = unsettled.astype(np.int64)
    working["_reserved_alarm_bins"] = reserved.astype(np.int64)
    monthly = working.groupby("station_month", as_index=False).agg(
        alarm_bins=("_alarm_bins", "sum"),
        false_alarm_bins=("_false_alarm_bins", "sum"),
        unsettled_alarm_bins=("_unsettled_alarm_bins", "sum"),
        reserved_alarm_bins=("_reserved_alarm_bins", "sum"),
        observed_bins=("_alarm_bins", "size"),
    )
    count_columns = [
        "alarm_bins",
        "false_alarm_bins",
        "unsettled_alarm_bins",
        "reserved_alarm_bins",
        "observed_bins",
    ]
    monthly[count_columns] = monthly[count_columns].astype(np.int64)
    monthly[entity_column] = monthly["station_month"].str.rsplit("|", n=1).str[0]
    step_hours = float(step_minutes) / 60.0
    monthly["alarm_hours"] = monthly["alarm_bins"] * step_hours
    monthly["false_alarm_hours"] = monthly["false_alarm_bins"] * step_hours
    monthly["unsettled_alarm_hours"] = monthly["unsettled_alarm_bins"] * step_hours
    monthly["reserved_alarm_hours"] = monthly["reserved_alarm_bins"] * step_hours
    monthly["budget_hours"] = float(budget_hours)
    monthly["budget_utilization"] = monthly["alarm_hours"] / float(budget_hours)
    monthly["false_alarm_budget_utilization"] = monthly["false_alarm_hours"] / float(budget_hours)
    monthly["reserved_budget_utilization"] = monthly["reserved_alarm_hours"] / float(budget_hours)
    monthly["budget_exceeded"] = monthly["false_alarm_hours"] > float(budget_hours) + 1e-9
    monthly["reserved_budget_exceeded"] = (
        monthly["reserved_alarm_hours"] > float(budget_hours) + 1e-9
    )
    summary = {
        "station_months": int(monthly.shape[0]),
        "mean_false_alarm_hours": float(monthly["false_alarm_hours"].mean()),
        "p95_false_alarm_hours": float(monthly["false_alarm_hours"].quantile(0.95)),
        "maximum_false_alarm_hours": float(monthly["false_alarm_hours"].max()),
        "station_month_exceedance_rate": float(monthly["budget_exceeded"].mean()),
        "mean_unsettled_alarm_hours": float(monthly["unsettled_alarm_hours"].mean()),
        "p95_unsettled_alarm_hours": float(monthly["unsettled_alarm_hours"].quantile(0.95)),
        "maximum_unsettled_alarm_hours": float(monthly["unsettled_alarm_hours"].max()),
        "mean_reserved_alarm_hours": float(monthly["reserved_alarm_hours"].mean()),
        "p95_reserved_alarm_hours": float(monthly["reserved_alarm_hours"].quantile(0.95)),
        "maximum_reserved_alarm_hours": float(monthly["reserved_alarm_hours"].max()),
        "reserved_station_month_exceedance_rate": float(
            monthly["reserved_budget_exceeded"].mean()
        ),
        "mean_budget_utilization": float(monthly["budget_utilization"].mean()),
        "p95_budget_utilization": float(monthly["budget_utilization"].quantile(0.95)),
        "maximum_budget_utilization": float(monthly["budget_utilization"].max()),
        "mean_reserved_budget_utilization": float(
            monthly["reserved_budget_utilization"].mean()
        ),
        "p95_reserved_budget_utilization": float(
            monthly["reserved_budget_utilization"].quantile(0.95)
        ),
        "maximum_reserved_budget_utilization": float(
            monthly["reserved_budget_utilization"].max()
        ),
    }
    return monthly, summary


def assert_prefix_invariance(
    issue_time: pd.Series | pd.DatetimeIndex,
    values: np.ndarray | pd.Series,
    policy: CandidatePolicy,
    safe_budget: SafeBudgetPolicy | None = None,
    cut_points: Iterable[int] = (),
) -> None:
    times = pd.Series(pd.to_datetime(issue_time, errors="coerce"))
    scores = np.asarray(values, dtype=np.float64)

    def run(end: int) -> np.ndarray:
        candidate, _, _ = candidate_alarm_mask(times.iloc[:end], scores[:end], policy)
        if safe_budget is None:
            return candidate
        return causal_monthly_budget_cap(times.iloc[:end], candidate, safe_budget)[0]

    full = run(scores.size)
    points = list(cut_points) or [max(scores.size // 3, 1), max(2 * scores.size // 3, 1)]
    for point in points:
        point = min(max(int(point), 1), scores.size)
        np.testing.assert_array_equal(run(point), full[:point])

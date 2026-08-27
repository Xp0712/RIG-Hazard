from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlertShape:
    ema_alpha: float = 1.0
    minimum_consecutive_bins: int = 1
    hold_bins: int = 3
    merge_gap_bins: int = 2

    def validate(self) -> None:
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.minimum_consecutive_bins < 1:
            raise ValueError("minimum_consecutive_bins must be positive")
        if self.hold_bins < 1 or self.merge_gap_bins < 0:
            raise ValueError("hold_bins must be positive and merge_gap_bins non-negative")


def causal_alert_mask(
    score: np.ndarray | pd.Series,
    threshold: float,
    shape: AlertShape,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an online EMA/run/hold/grace state machine to one station timeline."""

    masks, smoothed = causal_alert_masks(score, [threshold], shape)
    return masks[:, 0], smoothed


def _alarm_from_smoothed(smoothed: np.ndarray, threshold: float, shape: AlertShape) -> np.ndarray:
    """Vectorized equivalent of the causal run/hold/grace state transition."""

    alarm = np.zeros(smoothed.size, dtype=np.int8)
    finite = np.isfinite(smoothed)
    transitions = np.diff(np.r_[False, finite, False].astype(np.int8))
    starts, stops = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    for start, stop in zip(starts, stops):
        above = smoothed[start:stop] >= float(threshold)
        consecutive = int(shape.minimum_consecutive_bins)
        trigger = np.zeros(above.size, dtype=bool)
        if above.size >= consecutive:
            cumulative = np.r_[0, np.cumsum(above, dtype=np.int64)]
            trigger[consecutive - 1 :] = (
                cumulative[consecutive:] - cumulative[:-consecutive]
            ) == consecutive
        active_bins = int(shape.hold_bins + shape.merge_gap_bins)
        cumulative_trigger = np.r_[0, np.cumsum(trigger, dtype=np.int64)]
        indices = np.arange(trigger.size)
        left = np.maximum(indices - active_bins + 1, 0)
        alarm[start:stop] = (
            cumulative_trigger[indices + 1] - cumulative_trigger[left] > 0
        ).astype(np.int8)
    return alarm


def causal_alert_masks(
    score: np.ndarray | pd.Series,
    thresholds: Iterable[float],
    shape: AlertShape,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one causal smoother and many frozen absolute thresholds."""

    shape.validate()
    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("score must be one-dimensional")
    smoothed = np.full(values.size, np.nan, dtype=np.float64)
    previous: float | None = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            previous = None
            continue
        previous = (
            float(value)
            if previous is None
            else shape.ema_alpha * float(value) + (1.0 - shape.ema_alpha) * previous
        )
        smoothed[index] = previous
    threshold_values = [float(value) for value in thresholds]
    alarms = np.column_stack(
        [_alarm_from_smoothed(smoothed, value, shape) for value in threshold_values]
    ) if threshold_values else np.empty((values.size, 0), dtype=np.int8)
    return alarms, smoothed


def apply_alert_policy_grid(
    frame: pd.DataFrame,
    score_column: str,
    thresholds: Iterable[float],
    shape: AlertShape,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return station-time-sorted rows and aligned alarm columns for thresholds."""

    required = {"station_code", "issue_time", score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing alert-policy columns: {sorted(missing)}")
    threshold_values = [float(value) for value in thresholds]
    ordered = frame.copy()
    ordered["issue_time"] = pd.to_datetime(ordered["issue_time"], errors="coerce")
    ordered = ordered.sort_values(["station_code", "issue_time"]).reset_index(drop=True)
    if ordered["issue_time"].isna().any():
        raise ValueError("Alert-policy issue_time values must be valid")
    matrix = np.zeros((ordered.shape[0], len(threshold_values)), dtype=np.int8)
    for positions in ordered.groupby("station_code", sort=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        station_masks, _ = causal_alert_masks(
            pd.to_numeric(ordered.loc[positions, score_column], errors="coerce").to_numpy(),
            threshold_values,
            shape,
        )
        matrix[positions] = station_masks
    return ordered, matrix


def apply_alert_policy(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    shape: AlertShape,
    alarm_column: str = "budget_alarm",
) -> pd.DataFrame:
    required = {"station_code", "issue_time", score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing alert-policy columns: {sorted(missing)}")
    parts: list[pd.DataFrame] = []
    for _, station in frame.groupby("station_code", sort=False):
        station = station.copy()
        station["issue_time"] = pd.to_datetime(station["issue_time"], errors="coerce")
        station = station.sort_values("issue_time")
        if station["issue_time"].isna().any():
            raise ValueError("Alert-policy issue_time values must be valid")
        alarms, smoothed = causal_alert_mask(
            pd.to_numeric(station[score_column], errors="coerce").to_numpy(),
            threshold,
            shape,
        )
        station[alarm_column] = alarms
        station[f"{score_column}__ema"] = smoothed
        parts.append(station)
    return pd.concat(parts, ignore_index=True)


def assert_nested_alert_masks(
    masks: dict[float, np.ndarray | pd.Series],
    budgets: Iterable[float] = (2.0, 5.0, 10.0, 20.0),
) -> None:
    ordered = [float(value) for value in budgets]
    for lower, upper in zip(ordered, ordered[1:]):
        left = np.asarray(masks[lower], dtype=bool)
        right = np.asarray(masks[upper], dtype=bool)
        if left.shape != right.shape or np.any(left & ~right):
            raise ValueError(f"Alert masks are not nested: A_{lower:g} is not a subset of A_{upper:g}")


def _alert_segments(times: pd.Series, alarm: np.ndarray, step_minutes: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    selected = pd.DatetimeIndex(pd.to_datetime(times, errors="coerce"))[np.asarray(alarm, dtype=bool)]
    if selected.empty:
        return []
    breaks = np.r_[True, np.diff(selected.asi8) > int(step_minutes * 60 * 1_000_000_000)]
    starts = np.flatnonzero(breaks)
    stops = np.r_[starts[1:] - 1, selected.size - 1]
    return [(pd.Timestamp(selected[start]), pd.Timestamp(selected[stop])) for start, stop in zip(starts, stops)]


def evaluate_event_alerts(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    alarm_column: str,
    budget_hours: float,
    step_minutes: int = 10,
    horizon_hours: int = 6,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate final binary alarms using one-segment/one-event greedy matching."""

    working = frame.copy()
    working["issue_time"] = pd.to_datetime(working["issue_time"], errors="coerce")
    observed = pd.to_numeric(working.get("observed_6h", 1), errors="coerce").fillna(0).eq(1)
    working = working.loc[observed].copy()
    working["_alarm"] = pd.to_numeric(working[alarm_column], errors="coerce").fillna(0).ge(0.5)
    working["_future"] = pd.to_numeric(working["onset_within_6h"], errors="coerce").fillna(0).eq(1)
    working["_hard"] = pd.to_numeric(working.get("hard_negative_6h", 0), errors="coerce").fillna(0).eq(1)
    station_months = max(int(working["station_month"].nunique()), 1)
    false_bins = int((working["_alarm"] & ~working["_future"]).sum())
    alarm_bins = int(working["_alarm"].sum())
    hard_total = int(working["_hard"].sum())
    hard_false = int((working["_alarm"] & working["_hard"]).sum())
    event_rows: list[dict[str, Any]] = []
    valid_events = events.loc[pd.to_numeric(events["valid_target_event"], errors="coerce").fillna(0).eq(1)].copy()
    valid_events["onset_time"] = pd.to_datetime(valid_events["onset_time"], errors="coerce")
    horizon = pd.Timedelta(hours=horizon_hours)
    segment_count = 0
    for station_code, station_frame in working.groupby("station_code", sort=False):
        station_frame = station_frame.sort_values("issue_time")
        segments = _alert_segments(
            station_frame["issue_time"], station_frame["_alarm"].to_numpy(), step_minutes
        )
        segment_count += len(segments)
        station_events = valid_events.loc[
            valid_events["station_code"].astype(str).eq(str(station_code))
        ].sort_values("onset_time")
        evaluable: dict[str, dict[str, Any]] = {}
        for event in station_events.itertuples(index=False):
            onset = pd.Timestamp(event.onset_time)
            window = station_frame.loc[
                station_frame["issue_time"].ge(onset - horizon)
                & station_frame["issue_time"].lt(onset)
            ]
            evaluable[str(event.event_id)] = {
                "model": alarm_column,
                "event_id": str(event.event_id),
                "station_code": str(station_code),
                "onset_time": onset,
                "evaluable": int(not window.empty),
                "hit": 0,
                "effective_lead_hours": float("nan"),
                "lead_utility_hours": 0.0,
            }
        unmatched = [
            event for event in station_events.itertuples(index=False)
            if evaluable[str(event.event_id)]["evaluable"] == 1
        ]
        matched_ids: set[str] = set()
        for segment_start, segment_end in segments:
            candidates = [
                event for event in unmatched
                if str(event.event_id) not in matched_ids
                and pd.Timestamp(event.onset_time) > segment_start
                and pd.Timestamp(event.onset_time) <= segment_end + horizon
            ]
            if not candidates:
                continue
            event = min(candidates, key=lambda value: pd.Timestamp(value.onset_time))
            onset = pd.Timestamp(event.onset_time)
            valid_start = max(segment_start, onset - horizon)
            lead = float((onset - valid_start).total_seconds() / 3600.0)
            row = evaluable[str(event.event_id)]
            row.update({"hit": 1, "effective_lead_hours": lead, "lead_utility_hours": lead})
            matched_ids.add(str(event.event_id))
        event_rows.extend(evaluable.values())
    records = pd.DataFrame(event_rows)
    evaluable_records = records.loc[records["evaluable"].eq(1)]
    hits = evaluable_records.loc[evaluable_records["hit"].eq(1)]
    hit_rate = float(evaluable_records["hit"].mean()) if not evaluable_records.empty else float("nan")
    mean_lead = float(hits["effective_lead_hours"].mean()) if not hits.empty else float("nan")
    false_hours = false_bins * step_minutes / 60.0 / station_months
    result = {
        "model": alarm_column,
        "false_alarm_budget_hours_per_station_month": float(budget_hours),
        "false_alarm_hours_per_station_month": float(false_hours),
        "alarm_hours_per_station_month": alarm_bins * step_minutes / 60.0 / station_months,
        "budget_met": int(false_hours <= float(budget_hours) + 1e-9),
        "hard_negative_far": hard_false / hard_total if hard_total else float("nan"),
        "hard_negative_alarm_bins": hard_false,
        "hard_negative_bins": hard_total,
        "target_events": int(valid_events.shape[0]),
        "evaluable_events": int(evaluable_records.shape[0]),
        "hit_events": int(hits.shape[0]),
        "event_hit_rate": hit_rate,
        "mean_effective_lead_hours": mean_lead,
        "median_effective_lead_hours": float(hits["effective_lead_hours"].median()) if not hits.empty else float("nan"),
        "lead_utility_hours": hit_rate * mean_lead if np.isfinite(hit_rate) and np.isfinite(mean_lead) else 0.0,
        "alert_segments": int(segment_count),
        "alert_segments_per_station_month": segment_count / station_months,
        "station_months": station_months,
    }
    return result, records


def select_nested_thresholds(
    candidates: pd.DataFrame,
    budgets: Iterable[float] = (2.0, 5.0, 10.0, 20.0),
) -> pd.DataFrame:
    """Dynamic-programming selection under threshold monotonicity."""

    ordered_budgets = [float(value) for value in budgets]
    states: dict[float, tuple[tuple[float, float, float, float], list[pd.Series]]] = {}
    for budget_index, budget in enumerate(ordered_budgets):
        rows = candidates.loc[
            np.isclose(candidates["budget_hours"], budget)
            & candidates["budget_met"].eq(1)
        ]
        if rows.empty:
            raise ValueError(f"No alert threshold satisfies budget {budget:g}h")
        next_states: dict[float, tuple[tuple[float, float, float, float], list[pd.Series]]] = {}
        for _, row in rows.iterrows():
            threshold = float(row["threshold"])
            score = (
                float(row["lead_utility_hours"]),
                float(row["event_hit_rate"]),
                float(row["median_effective_lead_hours"] if np.isfinite(row["median_effective_lead_hours"]) else -1),
                -float(row["alert_segments"]),
            )
            if budget_index == 0:
                next_states[threshold] = (score, [row])
                continue
            feasible = [value for previous_threshold, value in states.items() if previous_threshold >= threshold]
            if not feasible:
                continue
            previous_score, previous_path = max(feasible, key=lambda value: value[0])
            combined = tuple(previous_score[index] + score[index] for index in range(4))
            current = next_states.get(threshold)
            if current is None or combined > current[0]:
                next_states[threshold] = (combined, [*previous_path, row])
        if not next_states:
            raise ValueError("No monotone nested threshold path satisfies all budgets")
        states = next_states
    _, selected = max(states.values(), key=lambda value: value[0])
    result = pd.DataFrame([dict(row) for row in selected]).reset_index(drop=True)
    result["selected"] = 1
    return result


def policy_payload(shape: AlertShape, selected: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": asdict(shape),
        "thresholds": {
            f"{float(row.budget_hours):g}": float(row.threshold)
            for row in selected.itertuples(index=False)
        },
        "selection_year": 2022,
        "score": "calibrated six-hour cumulative risk F[36]",
        "causal": True,
        "budget_nesting": "A_2 subset A_5 subset A_10 subset A_20",
    }

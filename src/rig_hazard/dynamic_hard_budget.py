from __future__ import annotations

"""Causal utility-aware alert control with a station-month hard budget.

The controller never inspects a future event at decision time.  Event labels
are consumed only when their onset becomes observable, and a non-event alert
is settled only after its complete follow-up horizon is observable.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .risk_trajectory import trajectory_value


DYNAMIC_BUDGET_CONTRACT_VERSION = "utility-adaptive-hard-budget-causal-reserve"


@dataclass(frozen=True)
class DynamicBudgetConfig:
    step_minutes: int = 10
    horizon_steps: int = 36
    method: str = "uadhbac"
    utility: str = "linear"
    score_threshold: float = 0.02
    price_initial: float = 0.0
    price_learning_rate: float = 2.0
    pending_pressure: float = 1.0
    pacing_slack_bins: int = 1
    deduplication_bins: int = 1
    use_lead_utility: bool = True
    use_dynamic_price: bool = True
    use_remaining_time: bool = True
    reserve_pending: bool = True
    release_true_reserve: bool = True
    deduplicate_events: bool = True
    couple_budgets: bool = True
    dual_step_scale: float = 1.0
    dual_initial_price: float = 0.0
    switch_high_threshold: float = 0.02
    switch_low_threshold: float = 0.01
    switch_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.step_minutes <= 0 or self.horizon_steps <= 0:
            raise ValueError("step_minutes and horizon_steps must be positive")
        if not np.isfinite(self.score_threshold) or self.score_threshold < 0:
            raise ValueError("score_threshold must be finite and non-negative")
        if self.pacing_slack_bins < 0 or self.deduplication_bins < 0:
            raise ValueError("integer controller settings must be non-negative")
        if not np.isfinite(self.dual_step_scale) or self.dual_step_scale <= 0:
            raise ValueError("dual_step_scale must be finite and positive")
        if not np.isfinite(self.dual_initial_price) or self.dual_initial_price < 0:
            raise ValueError("dual_initial_price must be finite and non-negative")
        if (
            not np.isfinite(self.switch_low_threshold)
            or not np.isfinite(self.switch_high_threshold)
            or self.switch_low_threshold < 0
            or self.switch_high_threshold < self.switch_low_threshold
        ):
            raise ValueError("switch thresholds must satisfy 0 <= low <= high")
        if not np.isfinite(self.switch_fraction) or not 0 <= self.switch_fraction <= 1:
            raise ValueError("switch_fraction must lie in [0, 1]")

    @property
    def horizon(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.step_minutes * self.horizon_steps)

    @property
    def horizon_ns(self) -> int:
        return int(self.step_minutes * self.horizon_steps * 60 * 1_000_000_000)


@dataclass
class _Reservation:
    issue_time_ns: int
    maturity_time_ns: int
    month_key: str
    followup_observed: bool
    segment_id: int = 0
    capacity_reserved: bool = True
    active: bool = True


@dataclass
class _MonthState:
    budget_hours: float
    capacity_bins: int
    confirmed_false_bins: int = 0
    pending_bins: int = 0
    locked_true_bins: int = 0
    released_true_bins: int = 0
    action_bins: int = 0
    peak_pending_bins: int = 0
    peak_occupied_bins: int = 0
    peak_confirmed_plus_pending_bins: int = 0
    rejected_capacity: int = 0
    rejected_pacing: int = 0
    calendar_bins: int = 0
    last_calendar_index: int = -1
    dual_price: float = 0.0

    @property
    def occupied_bins(self) -> int:
        return self.confirmed_false_bins + self.pending_bins + self.locked_true_bins

    def update_peaks(self) -> None:
        self.peak_pending_bins = max(self.peak_pending_bins, self.pending_bins)
        self.peak_occupied_bins = max(self.peak_occupied_bins, self.occupied_bins)
        self.peak_confirmed_plus_pending_bins = max(
            self.peak_confirmed_plus_pending_bins,
            self.confirmed_false_bins + self.pending_bins,
        )


def budget_capacity_bins(budget_hours: float, step_minutes: int) -> int:
    """Return the number of complete decision bins representable by a budget."""

    if not np.isfinite(budget_hours) or budget_hours < 0:
        raise ValueError("budget_hours must be finite and non-negative")
    # Floor in integer minutes: a sub-bin remainder can never fund an action.
    return int(np.floor((float(budget_hours) * 60.0 + 1e-10) / int(step_minutes)))


def _month_key(station: str, time: pd.Timestamp) -> str:
    return f"{station}|{time.to_period('M')}"


def _month_elapsed(time: pd.Timestamp) -> float:
    start = time.to_period("M").start_time
    end = (time.to_period("M") + 1).start_time
    return float(np.clip((time - start) / (end - start), 0.0, 1.0))


def _settle_mature_reservations(
    states: dict[str, _MonthState],
    active: list[_Reservation],
    now_ns: int,
) -> None:
    remaining: list[_Reservation] = []
    touched: set[str] = set()
    for reservation in active:
        if not reservation.active:
            continue
        if reservation.maturity_time_ns > now_ns:
            remaining.append(reservation)
            continue
        state = states[reservation.month_key]
        touched.add(reservation.month_key)
        if reservation.followup_observed:
            reservation.active = False
            if reservation.capacity_reserved:
                state.pending_bins -= 1
            state.confirmed_false_bins += 1
        # A right-censored reservation remains charged to its issue month but
        # cannot match an event after its horizon, so it leaves the hot queue.
    active[:] = remaining
    for key in touched:
        states[key].update_peaks()


def _release_for_event(
    states: dict[str, _MonthState],
    active: list[_Reservation],
    onset_ns: int,
    horizon_ns: int,
    release_true_reserve: bool,
    matched_segments: set[int],
) -> int:
    candidates = sorted(
        {
            reservation.segment_id
            for reservation in active
            if reservation.active
            and reservation.segment_id not in matched_segments
            and onset_ns - horizon_ns <= reservation.issue_time_ns < onset_ns
        }
    )
    if not candidates:
        return 0
    selected_segment = candidates[0]
    matched_segments.add(selected_segment)
    released = 0
    remaining: list[_Reservation] = []
    touched: set[str] = set()
    for reservation in active:
        if (
            reservation.active
            and reservation.segment_id == selected_segment
            and onset_ns - horizon_ns <= reservation.issue_time_ns < onset_ns
        ):
            state = states[reservation.month_key]
            touched.add(reservation.month_key)
            reservation.active = False
            if reservation.capacity_reserved:
                state.pending_bins -= 1
            if release_true_reserve:
                state.released_true_bins += 1
            else:
                state.locked_true_bins += 1
            released += 1
        elif reservation.active:
            remaining.append(reservation)
    active[:] = remaining
    for key in touched:
        states[key].update_peaks()
    return released


def _score_and_threshold(
    risk_6h: float,
    utility_value: float,
    state: _MonthState,
    config: DynamicBudgetConfig,
    elapsed: float,
) -> tuple[float, float]:
    score = utility_value if config.use_lead_utility else risk_6h
    threshold = float(config.score_threshold)
    if not config.use_dynamic_price or state.capacity_bins <= 0:
        return score, threshold
    occupied_fraction = state.occupied_bins / max(state.capacity_bins, 1)
    pending_fraction = state.pending_bins / max(state.capacity_bins, 1)
    elapsed = float(elapsed) if config.use_remaining_time else 0.5
    price = (
        float(config.price_initial)
        + float(config.price_learning_rate) * (occupied_fraction - elapsed)
        + float(config.pending_pressure) * pending_fraction
    )
    # Clipping makes the online price numerically stable in stress tests.
    return score, threshold * float(np.exp(np.clip(price, -12.0, 12.0)))


def _pacing_allows(state: _MonthState, elapsed: float, slack: int) -> bool:
    allowance = int(np.floor(float(elapsed) * state.capacity_bins)) + int(slack)
    return state.occupied_bins + 1 <= min(state.capacity_bins, allowance)


def _monthly_audit(
    station_states: dict[float, dict[str, _MonthState]],
    station: str,
    step_minutes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hours = float(step_minutes) / 60.0
    for budget, states in station_states.items():
        for month_key, state in states.items():
            rows.append(
                {
                    "station_code": station,
                    "station_month": month_key,
                    "budget_hours": float(budget),
                    "capacity_bins": int(state.capacity_bins),
                    "action_bins": int(state.action_bins),
                    "confirmed_false_bins": int(state.confirmed_false_bins),
                    "unsettled_bins": int(state.pending_bins),
                    "locked_true_bins": int(state.locked_true_bins),
                    "released_true_bins": int(state.released_true_bins),
                    "peak_pending_bins": int(state.peak_pending_bins),
                    "peak_occupied_bins": int(state.peak_occupied_bins),
                    "peak_confirmed_plus_pending_bins": int(
                        state.peak_confirmed_plus_pending_bins
                    ),
                    "false_alarm_hours": state.confirmed_false_bins * hours,
                    "unsettled_hours": state.pending_bins * hours,
                    "maximum_reserved_hours": state.peak_occupied_bins * hours,
                    "budget_utilization": state.action_bins / max(state.capacity_bins, 1),
                    "rejected_capacity": int(state.rejected_capacity),
                    "rejected_pacing": int(state.rejected_pacing),
                    "calendar_bins": int(state.calendar_bins),
                    "final_dual_price": float(state.dual_price),
                    "hard_budget_met": int(state.peak_occupied_bins <= state.capacity_bins),
                    "contract_version": DYNAMIC_BUDGET_CONTRACT_VERSION,
                }
            )
    return rows


def apply_dynamic_hard_budget(
    frame: pd.DataFrame,
    hazard_trajectory: np.ndarray,
    budgets_hours: Iterable[float],
    config: DynamicBudgetConfig,
    events: pd.DataFrame | None = None,
    entity_column: str = "station_code",
    time_column: str = "issue_time",
    observability_column: str = "observed_6h",
    event_time_column: str = "onset_time",
    candidate_masks: dict[float, np.ndarray] | None = None,
    trace_all_steps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply coupled controllers and return decisions, month audit, and trace.

    ``events`` may be supplied for offline replay, but an onset is processed
    only when ``onset_time <= current issue_time``.  Candidate masks must be
    produced causally (for example by a frozen threshold policy).
    """

    required = {entity_column, time_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing controller columns: {sorted(missing)}")
    budgets = sorted(set(float(value) for value in budgets_hours))
    if not budgets:
        raise ValueError("At least one budget is required")
    capacities = {value: budget_capacity_bins(value, config.step_minutes) for value in budgets}
    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    if working[time_column].isna().any():
        raise ValueError("Controller issue times must be valid")
    hazard = np.asarray(hazard_trajectory, dtype=np.float64)
    if hazard.shape != (working.shape[0], config.horizon_steps):
        raise ValueError("hazard_trajectory must align with frame rows and configured horizon")
    if not np.isfinite(hazard).all() or np.any((hazard < 0) | (hazard > 1)):
        raise ValueError("hazard_trajectory must be finite and lie in [0, 1]")
    working["_source_position"] = np.arange(working.shape[0], dtype=np.int64)
    ordered = working.sort_values([entity_column, time_column, "_source_position"]).copy()
    ordered_positions = ordered["_source_position"].to_numpy(dtype=np.int64)
    ordered_hazard = hazard[ordered_positions]
    ordered_time_ns = ordered[time_column].astype("int64").to_numpy(dtype=np.int64)
    periods = ordered[time_column].dt.to_period("M")
    month_start_ns = periods.dt.start_time.astype("int64").to_numpy(dtype=np.int64)
    month_end_ns = (periods + 1).dt.start_time.astype("int64").to_numpy(dtype=np.int64)
    step_ns = int(config.step_minutes * 60 * 1_000_000_000)
    calendar_index_by_row = np.floor_divide(
        ordered_time_ns - month_start_ns, step_ns
    ).astype(np.int64)
    calendar_bins_by_row = np.floor_divide(
        month_end_ns - month_start_ns, step_ns
    ).astype(np.int64)
    elapsed_by_row = np.clip(
        (ordered_time_ns - month_start_ns) / (month_end_ns - month_start_ns), 0.0, 1.0
    )
    station_month_by_row = (
        ordered[entity_column].astype(str) + "|" + periods.astype(str)
    ).to_numpy(dtype=object)
    observed_source = (
        ordered[observability_column]
        if observability_column in ordered
        else pd.Series(1, index=ordered.index)
    )
    ordered_observed = (
        pd.to_numeric(observed_source, errors="coerce")
        .fillna(0)
        .ge(0.5)
        .to_numpy(dtype=bool)
    )
    ordered_candidates = (
        {
            budget: np.asarray(mask, dtype=bool)[ordered_positions]
            for budget, mask in candidate_masks.items()
        }
        if candidate_masks is not None
        else {}
    )
    risk = 1.0 - np.prod(1.0 - ordered_hazard, axis=1)
    utility = trajectory_value(
        ordered_hazard,
        step_minutes=config.step_minutes,
        utility=config.utility,
    )
    if config.method == "dynamic_with_F6_only":
        utility = risk.copy()

    event_groups: dict[str, list[int]] = {}
    if events is not None and not events.empty:
        event_frame = events.copy()
        event_frame[event_time_column] = pd.to_datetime(
            event_frame[event_time_column], errors="coerce"
        )
        event_frame = event_frame.dropna(subset=[event_time_column])
        if "valid_target_event" in event_frame:
            event_frame = event_frame.loc[
                pd.to_numeric(event_frame["valid_target_event"], errors="coerce")
                .fillna(0)
                .eq(1)
            ]
        event_groups = {
            str(entity): sorted(int(pd.Timestamp(value).value) for value in group[event_time_column])
            for entity, group in event_frame.groupby(entity_column, sort=False)
        }

    decisions = {budget: np.zeros(ordered.shape[0], dtype=np.int8) for budget in budgets}
    trace_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    offset = 0
    for station, group in ordered.groupby(entity_column, sort=False):
        count = group.shape[0]
        positions = np.arange(offset, offset + count, dtype=np.int64)
        offset += count
        station_events = event_groups.get(str(station), [])
        event_pointer = 0
        states_by_budget: dict[float, dict[str, _MonthState]] = {
            budget: {} for budget in budgets
        }
        last_actions: dict[float, int | None] = {budget: None for budget in budgets}
        segment_ids: dict[float, int] = {budget: 0 for budget in budgets}
        matched_segments: dict[float, set[int]] = {budget: set() for budget in budgets}
        active_reservations: dict[float, list[_Reservation]] = {
            budget: [] for budget in budgets
        }
        for local_index, ordered_index in enumerate(positions):
            now_ns = int(ordered_time_ns[ordered_index])
            # Labels enter controller state only at their observed onset.
            while event_pointer < len(station_events) and station_events[event_pointer] <= now_ns:
                onset_ns = station_events[event_pointer]
                for budget in budgets:
                    _release_for_event(
                        states_by_budget[budget], active_reservations[budget], onset_ns,
                        config.horizon_ns,
                        config.release_true_reserve,
                        matched_segments[budget],
                    )
                    # An observed event terminates the current alarm episode;
                    # later actions belong to a new segment and can cover a
                    # later recurrence under one-segment/one-event matching.
                    last_actions[budget] = None
                event_pointer += 1
            for budget in budgets:
                _settle_mature_reservations(
                    states_by_budget[budget], active_reservations[budget], now_ns
                )

            observed = bool(ordered_observed[ordered_index])
            current_month_key = str(station_month_by_row[ordered_index])
            elapsed_fraction = float(elapsed_by_row[ordered_index])
            desired: list[bool] = []
            capacity_ok: list[bool] = []
            scores: list[float] = []
            thresholds: list[float] = []
            current_states: list[_MonthState] = []
            for budget in budgets:
                state = states_by_budget[budget].setdefault(
                    current_month_key,
                    _MonthState(
                        budget,
                        capacities[budget],
                        calendar_bins=int(calendar_bins_by_row[ordered_index]),
                        dual_price=float(config.dual_initial_price),
                    ),
                )
                current_states.append(state)
                if config.method == "dual_mirror_descent":
                    calendar_index = int(calendar_index_by_row[ordered_index])
                    skipped = max(0, calendar_index - state.last_calendar_index - 1)
                    target_rate = state.capacity_bins / max(state.calendar_bins, 1)
                    step_size = config.dual_step_scale / np.sqrt(max(state.calendar_bins, 1))
                    if skipped:
                        state.dual_price = max(
                            0.0,
                            state.dual_price - step_size * target_rate * skipped,
                        )
                    horizon_hours = config.step_minutes * config.horizon_steps / 60.0
                    score = float(np.clip(utility[ordered_index] / horizon_hours, 0.0, 1.0))
                    threshold = float(state.dual_price)
                elif config.method == "switch_over_knapsack":
                    score = float(utility[ordered_index])
                    threshold = float(
                        config.switch_high_threshold
                        if elapsed_fraction < config.switch_fraction
                        else config.switch_low_threshold
                    )
                else:
                    score, threshold = _score_and_threshold(
                        risk[ordered_index], utility[ordered_index], state, config,
                        elapsed_fraction,
                    )
                scores.append(score)
                thresholds.append(threshold)
                if budget in ordered_candidates:
                    candidate = bool(ordered_candidates[budget][ordered_index])
                elif config.method == "dual_mirror_descent":
                    # The void action wins ties in the canonical primal response.
                    candidate = bool(score > threshold)
                else:
                    candidate = bool(score >= threshold)
                if config.deduplicate_events and config.deduplication_bins > 0:
                    previous = last_actions[budget]
                    if previous is not None:
                        candidate &= now_ns - previous >= int(
                            config.step_minutes * config.deduplication_bins * 60 * 1_000_000_000
                        )
                if config.method == "uniform_pacing":
                    paced = _pacing_allows(state, elapsed_fraction, config.pacing_slack_bins)
                    if candidate and not paced:
                        state.rejected_pacing += 1
                    candidate &= paced
                desired.append(candidate)
                capacity_ok.append(
                    state.occupied_bins + (1 if config.reserve_pending else 0)
                    <= state.capacity_bins
                )

            if config.couple_budgets:
                # A larger budget inherits every lower-budget candidate.
                for index in range(1, len(desired)):
                    desired[index] = desired[index] or desired[index - 1]
                accepted = [False] * len(budgets)
                for index in range(len(budgets)):
                    accepted[index] = bool(
                        desired[index] and all(capacity_ok[index:])
                    )
            else:
                accepted = [bool(want and room) for want, room in zip(desired, capacity_ok)]

            for index, budget in enumerate(budgets):
                state = current_states[index]
                if desired[index] and not accepted[index]:
                    state.rejected_capacity += 1
                if accepted[index]:
                    decisions[budget][ordered_index] = 1
                    state.action_bins += 1
                    previous_action = last_actions[budget]
                    if previous_action is None or now_ns - previous_action > int(
                        config.step_minutes * 60 * 1_000_000_000
                    ):
                        segment_ids[budget] += 1
                    last_actions[budget] = now_ns
                    reservation = _Reservation(
                        issue_time_ns=now_ns,
                        maturity_time_ns=now_ns + config.horizon_ns,
                        month_key=current_month_key,
                        followup_observed=observed,
                        segment_id=segment_ids[budget],
                        capacity_reserved=config.reserve_pending,
                    )
                    active_reservations[budget].append(reservation)
                    if config.reserve_pending:
                        state.pending_bins += 1
                    else:
                        # Keep the reservation for delayed settlement without
                        # charging capacity: this is the explicit unsafe ablation.
                        reservation.active = True
                    state.update_peaks()
                if config.reserve_pending and state.occupied_bins > state.capacity_bins:
                    raise RuntimeError("Hard budget invariant was violated")
                if config.method == "dual_mirror_descent":
                    target_rate = state.capacity_bins / max(state.calendar_bins, 1)
                    step_size = config.dual_step_scale / np.sqrt(max(state.calendar_bins, 1))
                    state.dual_price = max(
                        0.0,
                        state.dual_price + step_size * (float(desired[index]) - target_rate),
                    )
                    state.last_calendar_index = int(calendar_index_by_row[ordered_index])
                if trace_all_steps or accepted[index]:
                    trace_rows.append(
                        {
                            "station_code": str(station),
                            "issue_time": pd.Timestamp(now_ns),
                            "budget_hours": budget,
                            "risk_6h": risk[ordered_index],
                            "utility_value": utility[ordered_index],
                            "decision_score": scores[index],
                            "decision_threshold": thresholds[index],
                            "desired": int(desired[index]),
                            "alarm": int(accepted[index]),
                            "confirmed_false_bins": state.confirmed_false_bins,
                            "pending_bins": state.pending_bins,
                            "occupied_bins": state.occupied_bins,
                            "capacity_bins": state.capacity_bins,
                            "dual_price": state.dual_price,
                        }
                    )
        monthly_rows.extend(_monthly_audit(states_by_budget, str(station), config.step_minutes))

    # Restore caller row order.
    output = frame.copy()
    inverse = np.empty(ordered.shape[0], dtype=np.int64)
    inverse[ordered_positions] = np.arange(ordered.shape[0], dtype=np.int64)
    for budget in budgets:
        label = f"budget_alarm_{budget:g}h"
        output[label] = decisions[budget][inverse].astype(np.int8)
    output["trajectory_risk_6h"] = risk[inverse]
    output["trajectory_lead_utility"] = utility[inverse]
    output["controller_contract_version"] = DYNAMIC_BUDGET_CONTRACT_VERSION
    return output, pd.DataFrame(monthly_rows), pd.DataFrame(trace_rows)


def nesting_audit(
    controlled: pd.DataFrame,
    budgets_hours: Iterable[float],
) -> pd.DataFrame:
    budgets = sorted(set(float(value) for value in budgets_hours))
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(budgets[:-1], budgets[1:]):
        low = pd.to_numeric(controlled[f"budget_alarm_{lower:g}h"], errors="coerce").fillna(0).ge(0.5)
        high = pd.to_numeric(controlled[f"budget_alarm_{upper:g}h"], errors="coerce").fillna(0).ge(0.5)
        conflict = low & ~high
        rows.append(
            {
                "lower_budget_hours": lower,
                "upper_budget_hours": upper,
                "lower_alarm_bins": int(low.sum()),
                "upper_alarm_bins": int(high.sum()),
                "nesting_violations": int(conflict.sum()),
                "nested": int(not conflict.any()),
            }
        )
    return pd.DataFrame(rows)


def offline_score_oracle(
    frame: pd.DataFrame,
    score: np.ndarray,
    budgets_hours: Iterable[float],
    step_minutes: int = 10,
    entity_column: str = "station_code",
    time_column: str = "issue_time",
) -> pd.DataFrame:
    """Diagnostic non-deployable upper reference using a full month score rank.

    No event labels are used.  Because it sees all *predicted* scores in a
    month before allocating capacity, this output must be labelled offline and
    must never be presented as a causal baseline.
    """

    values = np.asarray(score, dtype=np.float64)
    if values.shape != (frame.shape[0],) or not np.isfinite(values).all():
        raise ValueError("score must be one finite value per frame row")
    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    if working[time_column].isna().any():
        raise ValueError("offline oracle requires valid issue times")
    working["_score"] = values
    working["_position"] = np.arange(working.shape[0], dtype=np.int64)
    working["_station_month"] = (
        working[entity_column].astype(str)
        + "|"
        + working[time_column].dt.to_period("M").astype(str)
    )
    output = frame.copy()
    budgets = sorted(set(float(value) for value in budgets_hours))
    for budget in budgets:
        mask = np.zeros(frame.shape[0], dtype=np.int8)
        capacity = budget_capacity_bins(budget, step_minutes)
        if capacity > 0:
            for _, group in working.groupby("_station_month", sort=False):
                selected = group.sort_values(
                    ["_score", time_column], ascending=[False, True]
                ).head(capacity)
                mask[selected["_position"].to_numpy(dtype=np.int64)] = 1
        output[f"offline_score_oracle_{budget:g}h"] = mask
    output["deployable"] = 0
    output["oracle_uses_event_labels"] = 0
    return output


def controller_ablation(name: str, base: DynamicBudgetConfig) -> DynamicBudgetConfig:
    mapping: dict[str, dict[str, Any]] = {
        "dynamic_without_lead_utility": {"use_lead_utility": False},
        "dynamic_without_budget_pricing": {"use_dynamic_price": False},
        "dynamic_without_remaining_time": {"use_remaining_time": False},
        "dynamic_without_pending_reserve": {"reserve_pending": False},
        "dynamic_without_reserve_release": {"release_true_reserve": False},
        "dynamic_without_event_dedup": {"deduplicate_events": False},
        "dynamic_without_budget_coupling": {"couple_budgets": False},
        "dynamic_with_F6_only": {"method": "dynamic_with_F6_only", "use_lead_utility": False},
    }
    if name not in mapping:
        raise ValueError(f"Unknown ablation: {name}")
    return replace(base, **mapping[name])

# Recurrent Icing Risk Modelling: Frozen Results Analysis

**Data scope:** Mountain automatic-weather-station records from 2022 through
2024, aggregated to a 10-minute grid, with a 24-hour input history and a 6-hour
new-icing forecast horizon.

**Study population:** Station timestamps that are ice-free, outside the
post-event cooldown period, and supported by complete historical inputs.

**Evidence rule:** This document states only conclusions directly supported by
completed experiments. Associations, single-year advantages, and exploratory
results are not described as stable predictive gains or causal mechanisms.

**Frozen contract:** The primary probability model is
`local_weather_hazard`. The controller structure and all parameters are selected
from 2022 pooled out-of-fold (OOF) predictions only; 2023 and 2024 are frozen
evaluation years. The strict end-to-end recomputation uses event contract
`strict_event_grid_reset_2026-08-25`.

## 1. Main conclusion

> Mountain icing is usefully represented as a recurrent discrete-time risk
> process. Recent local temperature, humidity, cold-humid exposure, visibility,
> and their temporal evolution are the primary signals for the next icing onset.
> Explicit recurrence history, lagged neighboring-station signals, and station
> heterogeneity are interpretable, but they have not shown a stable incremental
> predictive benefit across years.

Deployment value depends on the false-alarm budget. After strict grid alignment,
the complete-window queue covers 84 of 116 events in 2023 and 116 of 181 events
in 2024; the operational queue covers 113 of 116 and 170 of 181 events,
respectively. Fixed-threshold, hysteresis, and rolling-quantile policies can have
high utility but substantially overspend in individual station-months.

The `uadhbac` controller simultaneously provides zero station-month overspend,
capacity reservation for unsettled alerts, and cross-budget nesting. At the
prespecified 5-hour and 10-hour budgets, all four year-budget comparisons show a
significant operational lead-time utility improvement over `budget_safe`.
Generalization to unseen stations remains the main deployment limitation.

## 2. Data, events, and risk set

| Level | Count |
| --- | ---: |
| Raw minute records | 43,077,910 |
| 10-minute station timestamps | 4,472,064 |
| Initial positive-icing episodes | 709 |
| Cold-plausible candidate events | 635 |
| Valid recurrent events | 476 |
| Events mapped to a leakage-free issue time | 447 |
| Risk-set timestamps | 4,161,734 |

Under station-wise global ordering, 27 of 30 event stations have recurrent
icing, and 446 of 476 events are recurrences. After resetting each season from
November through April, the data contain 80 within-season first events and 396
within-season recurrences. Across 36 event-definition sensitivity settings, the
valid event count ranges from 326 to 476, while the number of recurrent stations
remains 27. The recurrence pattern is robust, but the exact event count depends
on merge and cooldown rules.

## 3. Predictive information

Model attribution and ablation results identify local meteorological history as
the main predictive source: mean, minimum, and maximum temperature; joint
cold-humid exposure; condensation exposure during the previous hour;
visibility; snow; and changes in humidity and temperature. These variables
describe the environment for a new icing onset, not the current icing state.

### 3.1 Initial rules, classifiers, and hazard models

The table reports calibrated 6-hour results. Calibration does not change PR-AUC.

| Model | 2022 PR-AUC | 2022 Log Loss | 2024 PR-AUC | 2024 Log Loss |
| --- | ---: | ---: | ---: | ---: |
| `cold_humid_rule` | 0.0784 | 0.56530 | 0.0809 | 0.61129 |
| `strict_condensation_rule` | 0.0695 | 0.26107 | 0.0435 | 0.31741 |
| `direct_logit` | 0.1071 | **0.00998** | **0.1330** | **0.01377** |
| `local_cloglog_hazard` | **0.1105** | 0.01060 | 0.1154 | 0.01450 |

Cold-humid rules identify physically plausible conditions but do not replace a
probability model. The direct classifier ranks and calibrates better in this
2024 comparison; this does not establish universal superiority under the later
unified risk-set or spatial-generalization contracts.

### 3.2 Recurrence-history features

| Model | 2022 OOF PR-AUC | 2023 PR-AUC | 2024 PR-AUC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `local_weather_hazard` | **0.2609** | 0.1866 | **0.1692** | Most stable primary probability model |
| `rec_gap` | 0.2586 | 0.1825 | 0.1679 | No stable increment |
| `rec_previous` | 0.2577 | 0.1838 | **0.1871** | Ranking signal in 2024 only |
| `rec_load` | 0.2116 | **0.1873** | 0.1757 | Unstable across years |
| `rec_full` | 0.1988 | 0.1636 | 0.1614 | Overall degradation |

The complete recurrence module also has substantially worse 2024 Log Loss.
Recurrence information is therefore most defensible for risk-set construction
and stage interpretation, not as a general feature that reliably improves
cross-year prediction.

### 3.3 Fair comparison with temporal encoders

Every fair baseline uses the same cached data, temporal folds, random seeds,
training settings, and recurrence-feature masks. No model dominates across all
years and metrics.

| Model | 2022 PR-AUC | 2023 PR-AUC | 2024 PR-AUC | 2023 Log Loss | 2024 Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `local_weather_hazard` | **0.2609** | 0.1866 | **0.1692** | **0.00967** | 0.01340 |
| GRU | 0.2600 | **0.2029** | 0.1679 | 0.00979 | 0.01379 |
| TimesNet | 0.2414 | 0.1963 | 0.1612 | 0.00984 | **0.01303** |
| PatchTST | 0.1919 | 0.1572 | 0.1292 | 0.01244 | 0.01640 |
| TCN | 0.1625 | 0.1723 | 0.1457 | 0.01008 | 0.01413 |
| iTransformer | 0.1819 | 0.1802 | 0.1582 | 0.01111 | 0.01512 |

These results do not support treating encoder complexity itself as a stable
advantage. `local_weather_hazard` remains the primary model; GRU and TimesNet
are informative secondary comparisons.

### 3.4 Multiscale weather and recurrence-gate candidates

A separate supporting experiment screened fast-history, slow-history,
dual-scale weather, full-GRU, and recurrence-gated encoders. It used the same
2022-only selection principle but belongs to the experimental icing-model
contract rather than the frozen primary-model contract.

| Candidate | 2022 OOF PR-AUC | 2022 OOF Log Loss | Log-Loss guard |
| --- | ---: | ---: | --- |
| Fast-history encoder | **0.2196** | **0.01072** | Pass |
| Slow-history encoder | 0.2170 | 0.01128 | Pass |
| Full GRU encoder | 0.1988 | 0.01138 | Pass |
| Dual-scale weather encoder | 0.1912 | 0.01104 | Pass |
| Dual-scale recurrence gate | 0.2015 | 0.01123 | Pass |
| Dual-scale full recurrence | 0.1656 | 0.01172 | Fail |

The 5,000-sample locked-year station bootstrap shows that the dual-scale weather
encoder is worse than the fast-history encoder in 2024 PR-AUC by 0.0273 (95% CI
[-0.0557, -0.0070]); the 2023 interval includes zero. Adding full recurrence
raises Log Loss relative to dual-scale weather in both frozen years. The gated
version has no significant PR-AUC improvement and raises 2024 Log Loss by
0.000670 (95% CI [0.000082, 0.001559]). These experiments reinforce model
simplification but do not replace the fair comparison in Section 3.3.

### 3.5 Conditional information and failure diagnostics

Conditional recurrence screens use 2022 purged OOF predictions for hypothesis
generation. In the all-row screen, `rec_load` is significantly worse than
`local_weather_hazard` in station-bootstrap PR-AUC, with a 95% interval for the
difference of [-0.0843, -0.0297]. The interval for `rec_previous` crosses zero.
Among the 20 highest-ranked condition-model or probability-blend combinations,
none has simultaneous 95% station-bootstrap support for higher PR-AUC, lower
Log Loss, and lower Brier score.

Conditional permutation diagnostics confirm that the recurrence models use
their assigned feature groups. Model use is not the same as incremental value:
the additional probability shifts often add squared-error cost without enough
residual correction. Conditions defined using future outcomes, including hard
negatives, are diagnostic only and cannot be prospective gates. No conditional
screen was promoted into the frozen 2023/2024 model contract.

## 4. First-event and recurrent-event risk structure

A statistical interaction model shows different risk structures for first and
recurrent events. The joint weather-by-recurrence-state test has Wald
$\chi^2=175.31$ with 5 degrees of freedom ($P=5.37\times10^{-36}$). Temperature
and cold-humid exposure slopes change during recurrence, and the recent 7-day
event count and previous maximum ice thickness are positively associated with
subsequent risk.

This supports heterogeneity in conditional associations, but predictive gains
appear in only some years. The evidence supports “first and recurrent events
have different risk structures,” not “stage-specific modelling consistently
improves cross-year prediction.”

## 5. Alert governance

Three resource quantities must remain distinct:

- Mean false-alarm hours (FAH) describe average resource use.
- Station-month hard feasibility requires the FAH of every station-month to
  remain within budget.
- Reserved-capacity feasibility additionally charges alerts whose follow-up is
  incomplete and therefore unsettled.

This document uses “hard-feasible” only when the latter two conditions hold.

### 5.1 Strict grid alignment and two evaluation queues

The previous evaluator floored the onset timestamp and then shifted one extra
10-minute step backward, which dropped a valid prediction for events exactly on
the grid. The corrected final prediction is the nearest 10-minute grid point
strictly before onset. An event at 22:30 and one at 22:28 therefore both use
22:20 as the final prediction.

Runtime and tests enforce `window_end < onset_time`, a gap of no more than 10
minutes, 36 strictly increasing predictions spaced by 10 minutes, a 350-minute
window span, and `0 < lead <= 6 h` for every valid lead time.

| Year | Target events | Complete-window queue | Operational queue | Complete-window exclusions | Operational exclusions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 OOF | 179 | 86 | 167 | 93 | 12 |
| 2023 frozen | 116 | 84 | 113 | 32 | 3 |
| 2024 frozen | 181 | 116 | 170 | 65 | 11 |

The complete-window queue requires all 36 predictions before an event and is
appropriate for standardized comparison. The operational queue requires at
least one valid prediction in the 6-hour horizon and measures deployment
coverage. Complete-window inclusion is selective: excluded events have much
shorter intervals since the previous event. Primary reporting must therefore
include operational-queue sensitivity rather than relying on complete windows
alone.

### 5.2 Frozen hard-feasible reference baseline

The five historical strategy families and all parameters were selected on 2022
OOF predictions. Only `budget_safe` and `original_simple` remain hard-feasible
for every budget in both frozen years. The stronger of these references,
`budget_safe`, has the following complete-window performance:

| Year | Budget (h) | Hit rate (95% station-cluster CI) | Mean lead (h) | Lead utility (95% CI) | Mean FAH | Maximum FAH | Mean reserved utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 2 | 19.05% [9.78%, 30.88%] | 3.939 | 0.750 [0.351, 1.275] | 0.227 | 1.833 | 12.04% |
| 2023 | 5 | 41.67% [32.35%, 50.63%] | 3.890 | 1.621 [1.194, 2.136] | 0.899 | 4.833 | 18.54% |
| 2023 | 10 | 67.86% [58.02%, 77.32%] | 3.963 | 2.689 [2.244, 3.182] | 1.728 | 9.667 | 17.85% |
| 2023 | 20 | 97.62% [94.38%, 100.00%] | 4.202 | 4.102 [3.763, 4.528] | 4.580 | 19.500 | 23.71% |
| 2024 | 2 | 27.59% [20.72%, 34.44%] | 3.301 | 0.911 [0.681, 1.250] | 0.175 | 1.833 | 10.00% |
| 2024 | 5 | 58.62% [46.43%, 68.89%] | 3.128 | 1.834 [1.534, 2.197] | 0.712 | 4.833 | 16.23% |
| 2024 | 10 | 75.86% [64.08%, 84.62%] | 3.175 | 2.408 [2.084, 2.800] | 1.386 | 9.833 | 15.73% |
| 2024 | 20 | 94.83% [90.53%, 98.08%] | 3.411 | 3.234 [2.873, 3.707] | 3.570 | 19.833 | 20.95% |

The cost is underuse: `budget_safe` occupies about 10.0% to 23.7% of nominal
capacity. Non-hard strategies can achieve greater utility by overspending in
local months and are not valid winners under the same resource constraint.

### 5.3 Dynamic hard-budget controller

`uadhbac` converts the 36-step hazard trajectory into lead-time utility and
combines confirmed false alarms, unsettled reservations, time remaining in the
month, and a dynamic budget price. Every accepted action is charged to capacity
before its outcome is known. The 2022 OOF selection fixes the utility quantile
at 0.98, the dynamic price learning rate at 2.0, and unsettled pressure at 1.0.

| Year | Budget (h) | Complete hit | Operational hit | Complete mean lead (h) | Complete utility | Operational utility | Mean FAH | Maximum FAH | Mean reserved utilization | Hard-feasible / nested |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | 2 | 40.48% | 42.48% | 4.047 | 1.638 | 1.503 | 0.558 | 2.000 | 28.68% | Yes / Yes |
| 2023 | 5 | 52.38% | 55.75% | 4.809 | 2.519 | 2.341 | 1.306 | 5.000 | 26.95% | Yes / Yes |
| 2023 | 10 | 61.90% | 63.72% | 4.969 | 3.076 | 2.896 | 2.452 | 10.000 | 25.38% | Yes / Yes |
| 2023 | 20 | 76.19% | 80.53% | 5.008 | 3.816 | 3.604 | 4.431 | 20.000 | 22.99% | Yes / Yes |
| 2024 | 2 | 56.90% | 49.41% | 3.520 | 2.003 | 1.698 | 0.495 | 2.000 | 28.55% | Yes / Yes |
| 2024 | 5 | 68.97% | 62.35% | 3.734 | 2.575 | 2.336 | 1.174 | 5.000 | 27.12% | Yes / Yes |
| 2024 | 10 | 76.72% | 70.00% | 3.969 | 3.046 | 2.869 | 2.215 | 10.000 | 25.39% | Yes / Yes |
| 2024 | 20 | 83.62% | 82.94% | 4.310 | 3.604 | 3.552 | 3.846 | 20.000 | 22.00% | Yes / Yes |

All eight points have zero online hard-budget violations and zero cross-budget
nesting violations. At the four prespecified 5-hour and 10-hour operational
points, the lead-time utility gains over `budget_safe` are:

| Year | Budget (h) | Utility difference (h) | 95% CI | Holm-adjusted P |
| --- | ---: | ---: | ---: | ---: |
| 2023 | 5 | +0.973 | [0.545, 1.454] | <0.001 |
| 2023 | 10 | +0.601 | [0.207, 0.946] | 0.0096 |
| 2024 | 5 | +0.758 | [0.367, 1.222] | 0.0008 |
| 2024 | 10 | +0.835 | [0.476, 1.270] | <0.001 |

The gain is primarily longer effective lead time; hit-rate differences are not
significant in all four comparisons. Eight ablations further delimit the
mechanism: removing budget coupling creates 2,952 nesting conflicts; removing
unsettled-capacity reservation breaks hard feasibility; removing the dynamic
price reduces mean operational utility from 2.600 to 1.812. Across dense budgets
from 1 to 30 hours, four utility definitions, and 1-hour, 3-hour, and 6-hour
horizons, the complete controller retains zero hard-budget and nesting
violations.

### 5.4 Deterministic guarantees

For entity-month $q$, let budgets be $B_1<\cdots<B_K$ and define the integer
10-minute capacity as $C_k=\lfloor60B_k/10\rfloor$. At time $t$, let
$F_{qk}(t)$, $P_{qk}(t)$, and $L_{qk}(t)$ denote confirmed-false, unsettled, and
post-hit locked alert bins. Occupied capacity is
$O_{qk}(t)=F_{qk}(t)+P_{qk}(t)+L_{qk}(t)$.

The implementation relies on six conditions:

1. Each accepted action occupies exactly one decision bin.
2. Every candidate is reserved before issue (`reserve_pending=true`), and
   right-censored actions remain charged to their issue month.
3. Settlement never increases occupancy: false settlement moves one bin from
   $P$ to $F$; an event releases $P$ or transfers it equally to $L$.
4. Timestamps are processed atomically in nondecreasing order within an entity,
   and an event becomes observable only when `onset_time <= current_time`.
5. Cross-budget coupling first inherits lower-budget candidates, then accepts an
   action only if every affected higher budget has capacity.
6. Hazard calibration and independent, identically distributed observations are
   not required for feasibility; distributional assumptions affect utility only.

**Theorem 1 (entity-month hard budget).** Under conditions 1–4, if
$O_{qk}(0)=0$, then $O_{qk}(t)\le C_k$ for every prediction trajectory, event
sequence, censoring pattern, and time $t$. The result follows by induction:
settlement does not increase occupancy, and acceptance is permitted only after
verifying $O+1\le C$. Alerts that cross a month boundary remain charged to their
issue month.

**Theorem 2 (cross-budget nesting).** Under conditions 1–5, acceptance at budget
$B_k$ implies acceptance at $B_{k+1}$. Candidate inheritance supplies the
higher-budget candidate, while its required capacity checks are a subset of
those already passed by the lower budget. Applying this argument to every
adjacent pair yields $A(B_1)\subseteq\cdots\subseteq A(B_K)$.

These are deterministic safety results, not optimality or regret bounds. The DMD
reference follows
[Dual Mirror Descent for online resource allocation](https://proceedings.mlr.press/v119/balseiro20a.html),
but the present problem releases capacity after true events and handles right
censoring. It therefore does not inherit the original $O(\sqrt{T})$ regret bound
for non-replenishable resources and stochastic inputs.

### 5.5 Runtime and memory complexity

Let $N$ be prediction rows, $E$ events, $K$ budgets, $H$ follow-up bins
($H=36$), and $M$ entity-months. The batch implementation sorts data, computes
the trajectory, and scans at most $H$ active reservations per budget, giving
$O(N\log N+NH+K(N+E)H)$ time. With fixed $H$ and $E\le N$, this is
$O(N\log N+NK)$; the sorted online decision kernel is $O(NK)$. Materialized
evaluation uses $O(N(H+K)+MK)$ space, while streaming deployment requires
$O(KH)$ additional state for one active entity.

At four budgets, the 2023 DMD replay processes 1,321,566 rows in 65.14 seconds
(20,289 rows/s); switch-over requires 22.76 seconds (58,077 rows/s). In 2024,
the corresponding times for 1,453,371 rows are 72.74 and 25.43 seconds. The full
theory-supplement stage includes selection, frozen evaluation, and two
5,000-sample bootstraps; its end-to-end runtime is not an online latency measure.

### 5.6 Standard online baselines

The original six hard-guard families are `budget_safe`, `original_simple`,
`uniform_pacing_hard_guard`, `risk_greedy_hard_guard`,
`utility_static_hard_guard`, and `primal_dual_hard_guard`. Generic capacity
wrappers around fixed threshold, hysteresis, and rolling quantiles are diagnostic
variants, not additional algorithm families.

Standard DMD and switch-over online-knapsack baselines were selected on 2022 OOF
only. DMD freezes candidate `dmd_03` (step-size scale 2.0); switch-over freezes
`switch_08` (75% switch point and high/low risk quantiles 0.98/0.95).

| Year | Budget (h) | Baseline | Operational hit | Operational utility | Mean reserved utilization | Maximum reserved capacity | Hard-feasible | Nesting conflicts |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2023 | 5 | DMD | 48.67% | 1.712 | 98.47% | 5.000 | Yes | 20,830 |
| 2023 | 10 | DMD | 59.29% | 2.245 | 98.35% | 10.000 | Yes | 20,830 |
| 2023 | 5 | switch-over | 31.86% | 1.168 | 31.41% | 5.000 | Yes | 0 |
| 2023 | 10 | switch-over | 46.90% | 1.821 | 29.62% | 10.000 | Yes | 0 |
| 2024 | 5 | DMD | 57.06% | 1.927 | 91.96% | 5.000 | Yes | 22,661 |
| 2024 | 10 | DMD | 65.88% | 2.479 | 91.32% | 10.000 | Yes | 22,661 |
| 2024 | 5 | switch-over | 30.59% | 1.224 | 32.95% | 5.000 | Yes | 0 |
| 2024 | 10 | switch-over | 48.82% | 1.948 | 30.87% | 10.000 | Yes | 0 |

DMD nearly fills capacity but is not nested. Switch-over is hard-feasible and
nested but has low utilization. `uadhbac` has greater point-estimate utility at
all eight comparisons. Seven remain significant after Holm correction; the
2024 10-hour difference from DMD is +0.390 hours with a 95% CI of
[0.009, 0.804], but its Holm-adjusted $P=0.1428$.

### 5.7 Cost of nesting

Nesting ensures that increasing available resources cannot retract an alert
issued at a lower service level. This matters for service-tier consistency,
mid-month capacity changes, procurement curves, and auditability.

Against `dynamic_without_budget_coupling`, the aggregate frozen point-estimate
utility cost is 0.623 hours, or 2.91% of total uncoupled utility. This reduces
2,574 adjacent-budget conflicts in 2023 and 2,952 in 2024 to zero. A
5,000-sample station-cluster bootstrap finds that only the 2023 10-hour
operational cost remains significant after Holm correction; no nonzero
complete-window cost remains significant. The 2.91% value is therefore a frozen
sample estimate, not a universal population cost.

### 5.8 Precursor alert experiments and result precedence

Earlier warning experiments used average false-alarm hours, non-strict event
matching, or independently selected budget thresholds. They remain useful for
development history but do not satisfy the final combination of station-month
hard feasibility, unsettled-capacity reservation, strict event alignment, and
cross-budget nesting.

The earlier nested-alert experiment illustrates the distinction. Its nominal
2-hour, 5-hour, and 10-hour policies exceed their mean false-alarm budgets in
both 2023 and 2024; only the 20-hour policies meet the mean budget. Its high hit
rates therefore cannot be compared as feasible results against the final hard
budget controller. The `recurrence_modeling/utility_warning` results meet their
average budgets but do not prove per-station-month reserved-capacity
feasibility. The later `alert_governance` strict-grid runs correct matching and
policy comparisons, while `dynamic_hard_budget` is the authoritative frozen
contract that supersedes all of these precursor controller results.

## 6. Neighbor signals and spatial generalization

Stability selection retains 168 of 560 candidate edge-lag terms, representing
62 unique directed edges, but these do not produce stable gains in probability,
hit rate, or lead time. Edges represent predictive lagged association after
conditioning on local history, not physical propagation or causality.

Unseen-station evaluations show a clear loss. The PR-AUC difference is -0.0626
(95% CI [-0.1161, -0.0078]) under `2022_spatial_holdout` and -0.0521
([-0.0964, -0.0175]) under `2023_spatial_temporal`. These values belong to
different temporal and spatial contracts and are not conflicting versions of
one experiment. Region holdout is not consistently worse than station-group
holdout; the primary issue is the unseen station itself.

At a fixed 10-hour budget in the matched 2023 comparison, the stable graph and
its graph-removed ablation have the same 80.53% hit rate. Mean effective lead
time is 3.712 versus 3.853 hours, with a paired difference of -0.114 hours
(95% CI [-0.263, 0.028]). This controlled comparison does not support an
independent alerting benefit from the graph.

## 7. Public recurrence diagnostics

The ecommerce repeat-purchase and US accident recurrence datasets test whether
the event-risk, temporal-ordering, budget-control, and strict-matching workflow
can operate in different recurrent-event domains. They are not external icing
validation.

| Dataset | Budget (h/entity-month) | Complete hit | Operational hit | Mean lead (h) | Lead utility (h) | Mean FAH | Maximum FAH | Mean reserved utilization | Online violations | Hard-feasible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ecommerce | 4 | 4.41% | 4.52% | 2.900 | 0.128 | 0.438 | 3 | 11.01% | 0 | Yes |
| Ecommerce | 8 | 11.74% | 11.86% | 3.038 | 0.356 | 1.054 | 5 | 13.24% | 0 | Yes |
| Ecommerce | 16 | 21.02% | 21.16% | 3.244 | 0.682 | 2.373 | 11 | 14.92% | 0 | Yes |
| Ecommerce | 32 | 32.79% | 33.20% | 3.526 | 1.156 | 5.113 | 23 | 16.07% | 0 | Yes |
| US accidents | 24 | 87.11% | 87.15% | 41.591 | 36.232 | 12.513 | 24 | 52.14% | 0 | Yes |
| US accidents | 48 | 95.37% | 95.37% | 46.740 | 44.576 | 20.959 | 48 | 45.33% | 0 | Yes |
| US accidents | 96 | 98.94% | 98.92% | 56.415 | 55.816 | 36.197 | 96 | 39.85% | 0 | Yes |
| US accidents | 192 | 99.93% | 99.92% | 76.747 | 76.697 | 67.951 | 192 | 37.46% | 0 | Yes |

The US accident benchmark uses daily steps and nearly all-positive training
labels. Its lead times cannot be compared with the 6-hour icing horizon, and its
performance must be interpreted only as a diagnostic stress test.

## 8. Supported and unsupported claims

### Supported by the completed experiments

1. A seasonally reset recurrent-event risk set supports discrete-time icing-onset
   modelling.
2. Local cold-humid conditions and meteorological evolution dominate prediction
   of the next onset.
3. First and recurrent events have statistically identifiable risk-structure
   heterogeneity.
4. Explicit recurrence history, neighbor lags, and station heterogeneity do not
   provide a stable cross-year gain, supporting model simplification.
5. Mean budget compliance is not station-month hard feasibility; complete and
   operational queues provide different evidence about standardized performance
   and deployment coverage.
6. `uadhbac` has zero station-month overspend and zero nesting violations at all
   frozen budgets in 2023 and 2024.
7. Relative to `budget_safe`, `uadhbac` significantly improves prespecified
   operational lead-time utility at 5 and 10 hours in both frozen years.

### Not supported by the current evidence

1. Complete recurrence features consistently improve predictive accuracy.
2. Neighboring stations consistently provide earlier or more accurate warnings.
3. The discrete-time hazard model universally calibrates better than a direct
   classifier.
4. The current model is deployment-ready at unseen stations or regions.
5. Public recurrence benchmarks externally validate icing or meteorological
   mechanisms.
6. Average-FAH compliance or alert nesting alone establishes hard feasibility.
7. The complete-window queue alone represents short-interval recurrences.
8. The dynamic controller significantly improves hit rate at every budget; the
   confirmed result is lead-time utility at the four prespecified comparisons.

## 9. Recommended next work

1. Keep `local_weather_hazard` and the 2022-selected `uadhbac` parameters frozen;
   do not retune them in response to 2023 or 2024.
2. Continue reporting the operational queue and stratify by recurrence order,
   inter-event interval, station, and season.
3. Use held-out-station cluster bootstrap as the standard while developing
   domain generalization, hierarchical calibration, or unlabeled test-time
   calibration.
4. Keep public datasets in the workflow-diagnostic role, especially until the
   US accident label imbalance and timescale mismatch are resolved.
5. Obtain 2025 or external utility-grid data for a genuinely unseen confirmation
   of probability trajectories, budget feasibility, and lead-time utility.

## 10. Canonical result artifacts

All paths below are relative to `results/dynamic_hard_budget/`. The pipeline
does not create alias copies of canonical tables.

| Artifact | Content and role |
| --- | --- |
| `experiments/frozen_main_results.csv` | 176 frozen method-budget runs; the 64 rows whose `method` starts with `dynamic_` are the canonical ablation results |
| `experiments/paired_station_cluster_bootstrap.csv` | 48 prespecified paired station-cluster bootstrap comparisons |
| `experiments/frozen_nesting_audit.csv` | 78-row cross-budget nesting audit |
| `experiments/frozen_station_month_audit.csv.gz` | 59,752 station-month FAH, unsettled-capacity, reserved-capacity, and overspend audit rows |
| `experiments/frozen_event_records.csv.gz` | 26,136 event records derived from the same event table, trajectories, alerts, and matcher |
| `experiments/selected_controller_2022.json` | Dynamic-controller parameters selected from 2022 OOF only |
| `experiments/standard_baseline_selection_2022.csv` | Selection results for 11 standard-algorithm candidates |
| `experiments/selected_standard_baselines_2022.json` | Frozen DMD and switch-over parameters |
| `experiments/standard_algorithm_baselines.csv` | 16 frozen standard online-baseline runs |
| `experiments/standard_baseline_station_cluster_bootstrap.csv` | 48 paired comparisons against the standard baselines |
| `experiments/nesting_utility_cost.csv` | Eight point estimates for the utility cost of nesting |
| `experiments/nesting_cost_station_cluster_bootstrap.csv` | 48 station-cluster bootstrap rows for nesting cost |
| `experiments/controller_complexity_audit.csv` | Asymptotic and measured runtime and resource results |
| `experiments/theory_supplement_manifest.json` | Theory-supplement runtime, contract version, and key hashes |
| `experiments/frozen_run_manifest.json` | Frozen event, trajectory, controller, and evaluation contract |
| `experiments/resolved_experiment_contract.json` | Fully resolved run configuration |
| `experiments/dense_budget_sensitivity.csv` | Dense budget sensitivity |
| `experiments/utility_function_sensitivity.csv` | Lead-time utility sensitivity |
| `experiments/prediction_window_sensitivity.csv` | Forecast-horizon sensitivity |
| `public_recurrence_benchmarks/*/frozen_test_metrics.csv` | All frozen public-dataset diagnostic runs |

Supporting, exploratory, superseded, and legacy result families are routed in
[results_catalog.md](results_catalog.md). That catalog covers every top-level
directory under `results/` and prevents older contracts from being interpreted
as duplicate versions of the frozen primary result.

# RIG-Hazard: Recurrent Icing Risk and Hard-Budget Alerts

RIG-Hazard is an end-to-end framework for rare recurrent-event forecasting and
resource-constrained alerting. It uses the previous 24 hours of observations to
predict new icing onset over the next 6 hours on a 10-minute grid, then decides
whether to alert under a false-alarm budget defined for each station-month.

The primary probability model is `local_weather_hazard`. Model and controller
selection use only pooled out-of-fold (OOF) predictions from 2022. Data from
2023 and 2024 are reserved for frozen evaluation.

See [docs/results_analysis.md](docs/results_analysis.md) for the complete result
interpretation and [docs/project_conventions.md](docs/project_conventions.md)
for repository and documentation standards.

## Key findings

- Mountain icing can be represented as a recurrent discrete-time hazard process.
  Recent local cold-humid conditions and their temporal evolution provide the
  most stable predictive information.
- Explicit recurrence history, lagged neighboring-station signals, and complex
  temporal encoders do not provide a stable cross-year gain over
  `local_weather_hazard`.
- After strict grid alignment, the complete-window queue covers 84 of 116 events
  in 2023 and 116 of 181 events in 2024. The operational queue covers 113 of 116
  and 170 of 181 events, respectively.
- `uadhbac` has zero station-month hard-budget violations, feasible unsettled
  capacity, and strict cross-budget nesting at 2, 5, 10, and 20 hours in both
  frozen years.
- At the prespecified 5-hour and 10-hour budgets, `uadhbac` improves operational
  lead-time utility over `budget_safe` in all four year-budget comparisons under
  5,000-sample station-cluster bootstrap tests.
- The standard dual mirror descent (DMD) baseline uses 90.4% to 98.8% of reserved
  capacity but violates cross-budget nesting. The switch-over online-knapsack
  baseline is hard-feasible and nested but uses only 27.0% to 34.3% of capacity.
- Public recurrence datasets are workflow stress tests, not external validation
  of icing prediction or meteorological mechanisms.

## Frozen evaluation contract

| Component | Fixed setting |
| --- | --- |
| Primary probability model | `local_weather_hazard` |
| Input history | 24 hours |
| Decision interval | 10 minutes |
| Forecast horizon | 6 hours (36 steps) |
| Selection data | 2022 pooled OOF predictions |
| Frozen evaluation years | 2023 and 2024 |
| Primary budgets | 2, 5, 10, and 20 hours per station-month |
| Primary evaluation queue | Operational queue, with complete-window results also reported |
| Hard-budget definition | Confirmed false alarms plus reserved unsettled alerts must remain within budget |
| Cross-budget requirement | Every lower-budget alert set must be a subset of each higher-budget alert set |
| Bootstrap | 5,000 station-cluster resamples |
| Strict event contract | `strict_event_grid_reset_2026-08-25` |

The strict grid contract requires the final prediction timestamp to be earlier
than the event onset and no more than one 10-minute step away. Both the online
controller and offline evaluator split alert segments at observed event
boundaries so that capacity settlement and false-alarm auditing use the same
segment definition.

## Probability-model baseline comparison

The primary probability model was compared with five temporal encoders under a
fair protocol: all models use the same cached samples, temporal folds, random
seeds, training settings, recurrence-feature masks, 6-hour horizon, and
calibration procedure.

| Model | 2022 OOF PR-AUC | 2023 PR-AUC | 2024 PR-AUC | 2023 Log Loss | 2024 Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `local_weather_hazard` | **0.2609** | 0.1866 | **0.1692** | **0.00967** | 0.01340 |
| GRU | 0.2600 | **0.2029** | 0.1679 | 0.00979 | 0.01379 |
| TimesNet | 0.2414 | 0.1963 | 0.1612 | 0.00984 | **0.01303** |
| PatchTST | 0.1919 | 0.1572 | 0.1292 | 0.01244 | 0.01640 |
| TCN | 0.1625 | 0.1723 | 0.1457 | 0.01008 | 0.01413 |
| iTransformer | 0.1819 | 0.1802 | 0.1582 | 0.01111 | 0.01512 |

GRU has the highest 2023 PR-AUC, and TimesNet has the lowest 2024 Log Loss, but
neither gain is stable across years and metrics. `local_weather_hazard` has the
highest selection-year OOF PR-AUC, the highest 2024 PR-AUC, and the lowest 2023
Log Loss. It therefore remains the frozen primary model rather than being
replaced after inspecting later years.

The canonical comparison table is
`results/recurrence_modeling/fair_baselines/fair_baseline_summary.csv`. The
underlying OOF and frozen-year metrics are in `oof_metrics.csv` and
`locked_year_metrics.csv` in the same directory. Recurrence-feature comparisons
and exact paired station bootstrap results are stored under
`results/recurrence_modeling/probability_models/` and
`results/recurrence_modeling/paired_station_bootstrap/`, respectively. Full
interpretation, including rules, classifiers, hazard models, and
recurrence-history ablations, is provided in
[docs/results_analysis.md](docs/results_analysis.md#3-predictive-information).

## Repository layout

```text
ice_project/
├── README.md                       # Project overview and reproduction guide
├── configs/                        # Experiment contracts and model configuration
├── data/                           # Raw and public benchmark data (release asset)
├── docs/                           # Results and project conventions
├── requirements/                   # Layered dependency files
├── scripts/                        # Experiment and maintenance entry points
├── src/rig_hazard/                 # Reusable modelling, control, and evaluation code
├── tests/                          # Unit and regression tests
├── visualization/                  # Central manuscript figures and source data
│   └── manuscript_figures/         # Main and supplementary figure assets
└── results/                        # Metrics, predictions, checkpoints, and logs
    ├── dynamic_hard_budget/        # Frozen primary results
    │   ├── local_weather_hazard_trajectory/ # Frozen primary-model trajectories
    │   ├── experiments/            # Controllers, baselines, ablations, and sensitivity analyses
    │   └── public_recurrence_benchmarks/ # Cross-domain workflow diagnostics
    ├── alert_governance/           # Historical alert policies and strict-grid baselines
    ├── recurrence_analysis/        # Event definitions and statistical models
    ├── recurrence_modeling/        # Fair baselines and spatial generalization
    ├── icing_model_experiments/    # Weather, recurrence-gate, and earlier nested-alert studies
    └── legacy_exploration/         # Historical exploratory artifacts
```

`results/dynamic_hard_budget/` is the authoritative frozen-results root, and
`visualization/manuscript_figures/` is the authoritative manuscript-figure root.
The `src/` source root and `src/rig_hazard/` import package are intentionally
separate; merging them would break editable installation and the
`import rig_hazard` package contract. Historical result directories remain for
provenance and must not be mixed with the frozen primary tables.

## Data and result assets

Approximately 31 GiB of data and generated results are excluded from Git history.
They are distributed as split 7-Zip archives in the
[project-assets-initial release](https://github.com/Xp0712/RIG-Hazard/releases/tag/project-assets-initial).

After downloading every `data.7z.*` and `results.7z.*` part, run the following
commands from the repository root:

```powershell
7z x data.7z.001 -o.
7z x results.7z.001 -o.
```

7-Zip reads the remaining parts automatically. Verify the reconstructed assets
against the SHA-256 manifest included with the release.

## Result artifact map

The frozen pipeline uses the following canonical artifacts. No alias copies are
created because duplicate result files can diverge.

| Common reference | Canonical artifact | Content |
| --- | --- | --- |
| `frozen_main_comparison.csv` | `results/dynamic_hard_budget/experiments/frozen_main_results.csv` | 176-row frozen table containing baselines, online controllers, hard guards, eight ablations, and oracle diagnostics |
| `paired_station_bootstrap.csv` | `results/dynamic_hard_budget/experiments/paired_station_cluster_bootstrap.csv` | 48 prespecified comparisons with 5,000 station-cluster bootstrap samples |
| `ablation_results.csv` | 64 rows in `frozen_main_results.csv` whose `method` starts with `dynamic_` | Ablations are part of the canonical main table and are not duplicated |
| `nestedness_audit.csv` | `results/dynamic_hard_budget/experiments/frozen_nesting_audit.csv` | 78-row cross-budget nesting audit |
| `station_month_budget_audit.csv` | `results/dynamic_hard_budget/experiments/frozen_station_month_audit.csv.gz` | 59,752 method-station-month budget audit rows, gzip-compressed |

Other primary artifacts are listed below. Paths are relative to
`results/dynamic_hard_budget/`.

| Artifact | Purpose |
| --- | --- |
| `experiments/selected_controller_2022.json` | Controller parameters selected from 2022 OOF data and trajectory hashes |
| `experiments/frozen_event_records.csv.gz` | 26,136 strict event-matching records |
| `experiments/dense_budget_sensitivity.csv` | Sensitivity at 1, 2, 3, 5, 7, 10, 15, 20, and 30 hours |
| `experiments/utility_function_sensitivity.csv` | Sensitivity across four lead-time utility definitions |
| `experiments/prediction_window_sensitivity.csv` | Sensitivity at 1-hour, 3-hour, and 6-hour horizons |
| `experiments/hard_guard_algorithm_taxonomy.csv` | Algorithm-family audit for hard guards, DMD, and switch-over baselines |
| `experiments/standard_baseline_selection_2022.csv` | Selection results for 11 standard-baseline candidates |
| `experiments/selected_standard_baselines_2022.json` | Frozen DMD and switch-over parameters |
| `experiments/standard_algorithm_baselines.csv` | Standard online-algorithm results for 2023 and 2024 |
| `experiments/standard_baseline_station_cluster_bootstrap.csv` | Paired comparisons against the standard baselines |
| `experiments/nesting_utility_cost.csv` | Year-budget nesting cost for coupled and uncoupled controllers |
| `experiments/nesting_cost_station_cluster_bootstrap.csv` | Station-cluster bootstrap for nesting cost |
| `experiments/controller_complexity_audit.csv` | Asymptotic and measured runtime and memory audit |
| `experiments/theory_supplement_manifest.json` | Theory-supplement contract, runtime, and key SHA-256 values |
| `experiments/frozen_run_manifest.json` | Frozen data, model, evaluation, and artifact contract |
| `public_recurrence_benchmarks/*/frozen_test_metrics.csv` | Frozen public-dataset diagnostic metrics |
| `public_recurrence_benchmarks/*/run_manifest.json` | Public-dataset run contracts and artifact hashes |

## Installation and verification

Python 3.10 or newer is required.

```powershell
python -m pip install -r requirements/runtime.txt
python -m pip install -r requirements/deep_learning.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

After editable installation, use `rig-hazard --help` or
`python -m rig_hazard --help` for the unified command-line interface. Standalone
experiments remain available through `scripts/`. GPU acceleration is used for
risk-trajectory export and CUDA-enabled XGBoost training; controller replay,
strict matching, monthly auditing, and bootstrap analysis are primarily CPU tasks.

## Reproduction entry points

```powershell
# Export the frozen 36-step local_weather_hazard risk trajectory.
python scripts/export_selected_full_trajectory.py --device cuda

# Select on 2022 OOF, evaluate 2023/2024, and run ablations and sensitivities.
python scripts/run_dynamic_hard_budget_experiments.py --phase all

# Run only the standard online baselines, nesting-cost bootstrap, and complexity audit.
python scripts/run_dynamic_hard_budget_experiments.py --phase theory

# Run public recurrence diagnostics; --resume reuses model and policy checkpoints.
python scripts/run_public_recurrence_benchmarks.py `
  --datasets ecommerce us_accidents `
  --output-root results/dynamic_hard_budget/public_recurrence_benchmarks `
  --resume
```

Do not overwrite completed frozen results in place. Use a new output directory
for exploratory runs, and derive formal conclusions only from the frozen primary
table and its run manifests.

## Terminology

| Term | Definition |
| --- | --- |
| Risk set | Timestamps at which a station is ice-free, outside cooldown, and has valid historical inputs |
| Complete-window queue | Events with all 36 valid prediction timestamps before onset; used for standardized comparison |
| Operational queue | Events with at least one valid prediction timestamp in the 6-hour horizon; used for deployment coverage |
| Unsettled alert | An alert whose follow-up is incomplete and cannot yet be classified as false |
| Reserved capacity | Budget occupied by confirmed false alarms plus unsettled or locked alerts |
| Hard-feasible | Reserved capacity never exceeds budget for any station-month |
| Nested | Every lower-budget alert is also present at each higher budget |
| Lead-time utility | Utility combining event hit rate with effective warning lead time |

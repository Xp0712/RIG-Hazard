# Complete Result Catalog

This catalog makes every result family under `results/` discoverable without
mixing experiments that use different event, model-selection, or budget
contracts. Numerical interpretation of the current scientific conclusions is
in [results_analysis.md](results_analysis.md). Raw predictions, checkpoints,
bootstrap replicates, and logs are distributed in the project release and are
not duplicated in Git-tracked documentation.

## Evidence precedence

When two artifacts report different values, use the following order:

1. **Frozen primary:** `results/dynamic_hard_budget/` under event contract
   `strict_event_grid_reset_2026-08-25`.
2. **Confirmatory supporting:** leakage-controlled probability, recurrence,
   bootstrap, and spatial experiments under `results/recurrence_analysis/` and
   `results/recurrence_modeling/`.
3. **Supporting or diagnostic:** candidate screens, conditional analyses,
   public-dataset stress tests, and mechanism diagnostics.
4. **Superseded or historical:** earlier event matching, average-budget warning
   policies, engineering validations, and exploratory graph controllers.

A later evidence tier may explain model behavior or preserve provenance, but it
must not override a higher-tier frozen result.

## Coverage of every top-level result directory

| Result root | Tier | Recorded role and conclusion |
| --- | --- | --- |
| `results/dynamic_hard_budget/` | Frozen primary | Authoritative probability trajectories, strict queues, controller selection, frozen 2023/2024 results, hard-budget and nesting audits, standard online baselines, sensitivities, complexity, and public diagnostics |
| `results/recurrence_modeling/` | Confirmatory supporting | Primary probability-model selection, fair temporal encoders, paired probability inference, attribution, recurrence sensitivity, seasonal structure, spatial generalization, and precursor utility-warning experiments |
| `results/recurrence_analysis/` | Confirmatory supporting | Event structure, definition sensitivity, temporal folds, statistical models, earlier deep-model comparisons, and warning-policy evaluations |
| `results/preprocessed_data_10min/` | Supporting data contract | Ten-minute station data, split summaries, station/year preprocessing summaries, and validation manifests |
| `results/icing_model_experiments/` | Supporting model screen | Multiscale weather encoders, recurrence gates, targeted bootstrap tests, selected trajectories, protocol audits, and an earlier nested-alert experiment |
| `results/recurrence_information_story/` | Exploratory | Conditional recurrence-value screens, probability blends, station bootstraps, and failure-mechanism diagnostics; no candidate was promoted to the frozen contract |
| `results/hierarchical_sparse_lag_graph/` | Supporting negative result | Sparse lagged neighbor graph, probability and warning comparisons; selected edges do not yield a stable incremental benefit |
| `results/alert_governance/` | Superseded strict predecessor | XGBoost diagnostic, strict-grid strategy comparison, eligibility audit, nested-alert audit, and the predecessor public recurrence runs |
| `results/local_weather_hazard_baselines/` | Historical baseline | Early physical rules, classifier, and complementary-log-log hazard comparisons retained for context |
| `results/deep_model_experiments/` | Engineering and historical | GRU, TCN, modern encoder, recurrent-barrier, graph-gate, convergence, throughput, resume, and implementation validation runs |
| `results/stable_graph_causal_budget/` | Historical experiment | Earlier stable-graph probability and causal-budget warning experiment; superseded by the strict frozen controller |
| `results/local_conditional_information_exploration/` | Earlier exploratory copy | Earlier location for conditional recurrence screens; the maintained narrative uses `results/recurrence_information_story/conditional_value/` |
| `results/local_conditional_information_combinations/` | Earlier exploratory copy | Earlier location for conditional combination screens; the maintained narrative uses `results/recurrence_information_story/conditional_combinations/` |
| `results/local_recurrence_failure_diagnostics/` | Earlier diagnostic copy | Earlier location for recurrence failure diagnostics; the maintained narrative uses `results/recurrence_information_story/failure_diagnostics/` |
| `results/legacy_exploration/` | Legacy provenance | Dataset audits, early event statistics, archived key tables, and the workspace-organization inventory; not part of the frozen evidence |
| `results/logs/` | Operational provenance | Execution logs only; no scientific result is selected from this directory |
| `results/seasonal_risk_structure_validation/` | Empty placeholder | Contains no files; the populated validation is `results/recurrence_modeling/seasonal_risk_structure_validation/` |

## Frozen primary artifacts

All paths in this section are relative to `results/dynamic_hard_budget/`.

| Scope | Canonical artifacts | Result represented |
| --- | --- | --- |
| Primary trajectory | `local_weather_hazard_trajectory/multihorizon_probability_metrics.csv`, `trajectory_audit.csv`, `export_manifest.json` | Frozen 36-step probability trajectory and integrity checks |
| Controller selection | `experiments/selection_search.csv`, `selected_controller_2022.json` | Parameters selected from 2022 pooled OOF only |
| Frozen controller table | `experiments/frozen_main_results.csv` | Baselines, `uadhbac`, hard guards, eight ablations, and oracle diagnostics across two years and four budgets |
| Paired inference | `experiments/paired_station_cluster_bootstrap.csv` | Forty-eight prespecified 5,000-sample station-cluster comparisons |
| Event queues | `experiments/queue_contract_audit.csv`, `frozen_event_subgroups.csv`, `frozen_event_records.csv.gz` | Complete-window and operational eligibility, subgroups, matching, hits, and lead times |
| Feasibility | `experiments/frozen_station_month_audit.csv.gz`, `frozen_nesting_audit.csv` | Station-month reserved-capacity and cross-budget nesting audits |
| Probability integrity | `experiments/probability_metrics_2023.csv`, `probability_metrics_2024.csv`, `multihorizon_probability_metrics.csv`, `trajectory_integrity_audit.csv` | Frozen-year calibration, ranking, multihorizon, and trajectory checks |
| Mechanism ablations | The 64 `dynamic_*` rows in `experiments/frozen_main_results.csv` | Budget coupling, pending reservation, dynamic price, utility, horizon, and state ablations |
| Robustness | `experiments/dense_budget_sensitivity.csv`, `utility_function_sensitivity.csv`, `prediction_window_sensitivity.csv` | Budget-grid, utility-definition, and horizon sensitivity |
| Standard algorithms | `experiments/standard_baseline_selection_2022.csv`, `selected_standard_baselines_2022.json`, `standard_algorithm_baselines.csv` | Frozen DMD and switch-over selection and evaluation |
| Standard-algorithm inference | `experiments/standard_baseline_station_cluster_bootstrap.csv`, `standard_algorithm_station_month_audit.csv.gz`, `standard_algorithm_nesting_audit.csv`, `standard_algorithm_event_records.csv.gz` | Utility comparisons, feasibility, nesting, and event records |
| Nesting cost | `experiments/nesting_utility_cost.csv`, `nesting_cost_station_cluster_bootstrap.csv` | Point and station-bootstrap estimates of the cost of coupled nesting |
| Complexity and taxonomy | `experiments/controller_complexity_audit.csv`, `hard_guard_algorithm_taxonomy.csv` | Runtime, memory, asymptotic complexity, and algorithm-family definitions |
| Frozen provenance | `experiments/resolved_experiment_contract.json`, `frozen_run_manifest.json`, `theory_supplement_manifest.json` | Resolved configuration, hashes, elapsed time, and claim scope |
| Public diagnostics | `public_recurrence_benchmarks/*/frozen_test_metrics.csv`, `frozen_station_month_metrics.csv.gz`, `run_manifest.json` | Ecommerce and US-accident workflow stress tests |

## Probability and recurrence artifacts

| Experiment family | Summary artifacts | Recorded conclusion |
| --- | --- | --- |
| Early local baselines | `results/local_weather_hazard_baselines/probability_metrics_validation.csv`, `probability_metrics_test.csv`, `warning_metrics.csv` | Physical rules are useful screens but not replacements for calibrated probability models |
| Event structure | `results/recurrence_analysis/structure/event_order_summary.csv`, `recurrence_gap_summary.csv`, `first_recurrent_cluster_bootstrap.csv` | Recurrence is common and inter-event intervals are heterogeneous |
| Definition sensitivity | `results/recurrence_modeling/definition_sensitivity/sensitivity_summary.csv` and supporting year/station tables | Valid event counts vary from 326 to 476 across 36 definitions while recurrent-station count remains 27 |
| Seasonal reset | `results/recurrence_modeling/seasonal_recurrence/global_vs_seasonal_summary.csv` | Global ordering gives 30 first and 446 recurrent events; seasonal reset gives 80 first and 396 recurrent events |
| Probability selection | `results/recurrence_modeling/probability_models/2022_only_model_selection.csv`, `frozen_model_selection.json`, `oof_metrics.csv`, `locked_year_metrics.csv` | `rec_none`, exposed publicly as `local_weather_hazard`, is selected on 2022 OOF |
| Broad paired probability test | `results/recurrence_modeling/paired_station_bootstrap/probability_bootstrap_summary.csv` | Full recurrence does not significantly improve frozen-year PR-AUC and worsens 2024 Log Loss |
| Targeted paired probability test | `results/recurrence_modeling/targeted_probability_bootstrap/targeted_probability_bootstrap_summary.csv` | `rec_previous` improves 2024 PR-AUC only; recurrence gains are not stable across years and calibration metrics |
| Fair temporal encoders | `results/recurrence_modeling/fair_baselines/fair_baseline_summary.csv`, `oof_metrics.csv`, `locked_year_metrics.csv` | GRU and TimesNet have isolated wins, but no encoder has a stable cross-year advantage over the selected local model |
| Locked attribution | `results/recurrence_modeling/locked_attribution/`, `candidate_attribution/` | Local temperature, humidity, cold-humid exposure, condensation, visibility, snow, and temporal changes dominate attribution |
| Seasonal risk structure | `results/recurrence_modeling/seasonal_risk_structure/` and `seasonal_risk_structure_validation/` | First and recurrent events have different conditional associations without a stable predictive increment |
| Spatial generalization | `results/recurrence_modeling/spatial_generalization/ensemble_generalization_metrics.csv`, `paired_generalization_gap_bootstrap.csv`, `spatial_leakage_audit.csv` | Unseen stations have significant PR-AUC loss; region holdout is not consistently worse than station-group holdout |
| Precursor utility warning | `results/recurrence_modeling/utility_warning/frozen_2023_2024_warning_metrics.csv`, `paired_station_bootstrap_summary.csv` | Average-budget warning results are retained but do not establish final hard feasibility |

## Supporting model and mechanism screens

| Experiment family | Summary artifacts | Recorded conclusion |
| --- | --- | --- |
| Multiscale weather candidates | `results/icing_model_experiments/selection_year_model_selection/candidate_2022_oof_metrics.csv`, `weather_ablation/oof_metrics.csv`, `weather_ablation/locked_year_metrics.csv` | The fast-history encoder leads the candidate screen; dual-scale weather is not a stable improvement |
| Recurrence-gate candidates | `results/icing_model_experiments/recurrence_gate_ablation/oof_metrics.csv`, `locked_year_metrics.csv` | Full recurrence worsens calibration; gating has no stable PR-AUC gain |
| Candidate paired inference | `results/icing_model_experiments/selection_year_bootstrap/targeted_probability_bootstrap_summary.csv`, `locked_year_bootstrap/targeted_probability_bootstrap_summary.csv` | Five-thousand-sample intervals support simplification rather than promotion of a recurrence gate |
| Candidate trajectory | `results/icing_model_experiments/selected_model_trajectory/` | Full-curve output for the experimental selected model; not the primary frozen trajectory |
| Earlier nested alert | `results/icing_model_experiments/nested_alert/frozen_warning_metrics.csv` | High hit rates accompany mean-budget excess at 2, 5, and 10 hours in both frozen years |
| Conditional value | `results/recurrence_information_story/conditional_value/conditional_probability_metrics_2022_oof.csv`, `paired_station_bootstrap_2022_oof.csv` | Recurrence value is condition-dependent but not stable enough for promotion |
| Conditional combinations | `results/recurrence_information_story/conditional_combinations/combination_condition_metrics_2022_oof.csv`, `top_combination_station_bootstrap.csv` | None of the top 20 candidates has coherent station-bootstrap improvement across all three primary probability metrics |
| Failure diagnostics | `results/recurrence_information_story/failure_diagnostics/existing_2023_attribution_diagnostics.csv` and decomposition tables | Extra models use recurrence features, but their probability shifts often add error without sufficient residual correction |
| Neighbor graph | `results/hierarchical_sparse_lag_graph/graph_regularization_selection.csv`, `probability_metrics_test.csv`, `paired_event_comparisons.csv`, `warning_metrics.csv` | Selected lagged edges are predictive associations, not causal propagation, and do not add stable warning value |

## Superseded and engineering artifact routing

| Result family | Retained summaries | Why it is not primary evidence |
| --- | --- | --- |
| Strict-grid predecessor | `results/alert_governance/strategy_comparison/` and `nested_alert_audit/` | Precedes the final dynamic reserved-capacity and cross-budget contract |
| XGBoost diagnostic | `results/alert_governance/xgboost_baseline/run_manifest.json` and its metrics in the strategy comparison | Controller-independence diagnostic, not the selected probability model |
| Predecessor public diagnostics | `results/alert_governance/public_recurrence_benchmarks/` | Recomputed under `results/dynamic_hard_budget/public_recurrence_benchmarks/` for the final contract |
| Earlier temporal-model protocol | `results/recurrence_analysis/model_comparison/` | Contains statistical, GRU, modern-encoder, implementation, and warning runs that precede the maintained fair-baseline summaries |
| Recurrence selection support | `results/recurrence_modeling/budget_2022_oof/`, `remote_final_summaries/`, and `supplementary_experiments/` | Selection support, downloaded summary snapshots, and completion markers rather than separate scientific conclusions |
| Deep local baselines | `results/deep_model_experiments/local_baselines/deep_local_summary.csv` | Earlier model-development contract |
| Modern encoder validation | `results/deep_model_experiments/modern_baseline_validation/deep_local_summary.csv` | Engineering validation later replaced by the fair-baseline protocol |
| Recurrent barriers and graph gate | `results/deep_model_experiments/recurrent_barrier_extensions/`, `recurrent_barrier_warning/`, `stable_graph_gate/` | Experimental extensions that were not promoted |
| Deep contract and warning development | `results/deep_model_experiments/model_contract_audit/`, `event_warning/`, and `extension_validation/` | Audits model inputs and evaluates precursor warning and extension behavior |
| Training engineering | `results/deep_model_experiments/implementation_validation/`, `full_epoch_benchmark/`, `gru_convergence/`, `tcn_convergence/`, `tcn_stability_validation/`, `throughput_validation/`, `resume_validation/`, `partial_resume_validation/`, and `cache/` | Validates implementation, convergence, throughput, restart behavior, and reusable inputs rather than a scientific claim |
| Icing-model protocol support | `results/icing_model_experiments/protocol_audit/`, `protocol_audit_local_validation/`, `deep_protocol_engineering_validation/`, `model_selection_local_validation/`, `bootstrap_local_validation/`, and `pipeline/` | Protocol checks, reduced local checks, engineering validation, and pipeline completion state |
| Earlier stable graph and budget | `results/stable_graph_causal_budget/` | Earlier graph and budget contract superseded by strict matching and reserved capacity |
| Legacy exploration | `results/legacy_exploration/` | Archived audits and historical tables retained only for provenance |
| Operational logs | `results/alert_governance/logs/`, `results/recurrence_modeling/logs/`, and `results/logs/` | Execution provenance only; not result-selection inputs |

## Row-level and binary artifacts

The catalog intentionally does not reproduce millions of prediction rows or
binary checkpoints in Markdown. They remain part of the result archive:

- `folds/`, `runs/`, and `completed_runs/` contain seed- and fold-level metrics.
- `oof_predictions/`, `locked_predictions/`, `predictions*.csv.gz`, and
  `event_records*.csv.gz` contain row-level evaluation evidence.
- `checkpoints/`, `trained_models/`, `.pt`, `.npz`, and model JSON files contain
  reusable fitted state.
- `bootstrap_replicates*.csv.gz` contains resampled distributions behind the
  reported confidence intervals.
- `logs/` and completion markers record execution state but are not scientific
  result tables.

These files are available through the split release assets described in the
root README. Their manifests and SHA-256 values provide provenance without
copying the same result values into multiple tracked files.

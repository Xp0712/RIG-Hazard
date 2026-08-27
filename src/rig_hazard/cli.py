from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline_experiment import run_baseline_experiment
from .config import PROJECT_ROOT, load_config
from .graph_experiment import run_graph_experiment
from .preprocessing import preprocess
from .stability_experiment import run_stability_experiment
from .validation import validate_preprocessed


def comma_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def year_set(value: str | None) -> set[int] | None:
    items = comma_set(value)
    return {int(item) for item in items} if items else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rig-hazard",
        description="Unified command-line entry point for RIG-Hazard experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preprocess_parser = subparsers.add_parser("preprocess", help="Build recurrent-event timelines and hazard risk sets.")
    preprocess_parser.add_argument(
        "--config",
        default="configs/rig_hazard_preprocessing.json",
        help="Preprocessing configuration path.",
    )
    preprocess_parser.add_argument("--output-root", default=None, help="Override the configured output directory.")
    preprocess_parser.add_argument("--max-stations", type=int, default=None, help="Limit station count for a quick subset run.")
    preprocess_parser.add_argument("--stations", default=None, help="Comma-separated station codes or names.")
    preprocess_parser.add_argument("--years", default=None, help="Comma-separated years, for example 2022,2023.")
    preprocess_parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output directory.")
    validate_parser = subparsers.add_parser("validate", help="Audit a preprocessed dataset before model training.")
    validate_parser.add_argument("--output-root", default="results/preprocessed_data_10min", help="Preprocessed output directory.")
    validate_parser.add_argument("--no-strict", action="store_true", help="Write failures without returning a non-zero exit.")
    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Train and evaluate rule, local-hazard, and classifier baselines.",
    )
    baseline_parser.add_argument("--config", default="configs/rig_hazard_baseline.json", help="Baseline experiment JSON configuration.")
    baseline_parser.add_argument("--overwrite", action="store_true", help="Replace an existing baseline output directory.")
    graph_parser = subparsers.add_parser(
        "graph",
        help="Train and evaluate hierarchical-barrier and sparse-lag-graph models.",
    )
    graph_parser.add_argument(
        "--config",
        default="configs/rig_hazard_graph.json",
        help="Hierarchical sparse-lag graph experiment configuration.",
    )
    graph_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing graph experiment output directory.",
    )
    stability_parser = subparsers.add_parser(
        "stability", help="Run block graph-stability selection and causal monthly alarm-budget evaluation."
    )
    stability_parser.add_argument(
        "--config", default="configs/rig_hazard_stability.json", help="Stability experiment JSON configuration."
    )
    stability_parser.add_argument("--overwrite", action="store_true", help="Replace an existing stability output directory.")
    deep_audit_parser = subparsers.add_parser(
        "deep-audit", help="Freeze and audit the statistical baseline before deep hazard experiments."
    )
    deep_audit_parser.add_argument(
        "--config", default="configs/rig_hazard_deep.json", help="Deep RIG-Hazard experiment JSON configuration."
    )
    deep_audit_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing model-contract audit directory.",
    )
    deep_cache_parser = subparsers.add_parser(
        "deep-cache", help="Build leakage-safe temporal windows for local deep hazard models."
    )
    deep_cache_parser.add_argument(
        "--config", default="configs/rig_hazard_deep.json", help="Deep RIG-Hazard experiment JSON configuration."
    )
    deep_cache_parser.add_argument("--overwrite", action="store_true", help="Replace an existing deep cache directory.")
    deep_local_parser = subparsers.add_parser(
        "deep-local", help="Train non-graph GRU/TCN multi-step hazard baselines on the frozen deep cache."
    )
    deep_local_parser.add_argument(
        "--config", default="configs/rig_hazard_deep.json", help="Deep RIG-Hazard experiment JSON configuration."
    )
    deep_local_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated encoders: gru,tcn,patchtst,timesnet,itransformer,recurrent_dual.",
    )
    deep_local_parser.add_argument("--seeds", default=None, help="Comma-separated random seeds.")
    deep_local_parser.add_argument("--max-epochs", type=int, default=None, help="Override maximum training epochs.")
    deep_local_parser.add_argument("--max-train-samples", type=int, default=None, help="Limit stratified training rows.")
    deep_local_parser.add_argument(
        "--max-validation-samples", type=int, default=None, help="Limit stratified development rows."
    )
    deep_local_parser.add_argument("--output-root", default=None, help="Override the configured local baseline output.")
    deep_local_parser.add_argument("--train-split", default="train", help="Cache index name used for fitting.")
    deep_local_parser.add_argument(
        "--validation-split", default="validation", help="Cache index name used for early stopping and metrics."
    )
    deep_local_parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu, or cuda.")
    deep_local_parser.add_argument("--resume", action="store_true", help="Reuse completed model/seed runs in the output directory.")
    deep_local_parser.add_argument("--overwrite", action="store_true", help="Replace an existing local baseline directory.")
    deep_warning_parser = subparsers.add_parser(
        "deep-warning", help="Evaluate five-seed deep ensembles under the frozen causal monthly alarm budget."
    )
    deep_warning_parser.add_argument(
        "--config", default="configs/rig_hazard_deep.json", help="Deep RIG-Hazard experiment JSON configuration."
    )
    deep_warning_parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu, or cuda.")
    deep_warning_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated encoders to ensemble; defaults to the configured local model list.",
    )
    deep_warning_parser.add_argument(
        "--local-root", default=None, help="Override the directory containing trained checkpoints."
    )
    deep_warning_parser.add_argument(
        "--output-root", default=None, help="Override the configured warning output directory."
    )
    deep_warning_parser.add_argument("--evaluation-year", type=int, default=None, help="Override warning evaluation year.")
    deep_warning_parser.add_argument("--history-split", default=None, help="Cache index supplying the preceding year.")
    deep_warning_parser.add_argument("--evaluation-split", default=None, help="Cache index supplying the evaluation year.")
    deep_warning_parser.add_argument(
        "--disable-matched-diagnostic",
        action="store_true",
        help="Disable label-matched thresholds, required for the locked 2024 run.",
    )
    deep_warning_parser.add_argument("--resume", action="store_true", help="Reuse completed station-year ensemble predictions.")
    deep_warning_parser.add_argument("--overwrite", action="store_true", help="Replace an existing warning output directory.")
    deep_extension_parser = subparsers.add_parser(
        "deep-extensions", help="Run recurrent-history, negative-sampling, and hierarchical-barrier experiments."
    )
    deep_extension_parser.add_argument(
        "--config", default="configs/rig_hazard_deep.json", help="Deep RIG-Hazard experiment JSON configuration."
    )
    deep_extension_parser.add_argument("--variants", default=None, help="Comma-separated configured extension variants.")
    deep_extension_parser.add_argument("--seeds", default=None, help="Comma-separated random seeds.")
    deep_extension_parser.add_argument("--max-epochs", type=int, default=None, help="Override maximum training epochs.")
    deep_extension_parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training rows.")
    deep_extension_parser.add_argument(
        "--max-validation-samples", type=int, default=None, help="Limit development rows."
    )
    deep_extension_parser.add_argument("--output-root", default=None, help="Override the extension output directory.")
    deep_extension_parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu, or cuda.")
    deep_extension_parser.add_argument("--resume", action="store_true", help="Reuse completed variant/seed runs.")
    deep_extension_parser.add_argument("--overwrite", action="store_true", help="Replace an existing extension directory.")
    deep_extension_warning_parser = subparsers.add_parser(
        "deep-extension-warning", help="Evaluate extension ensembles under frozen and matched false-alarm controls."
    )
    deep_extension_warning_parser.add_argument(
        "--config", default="configs/rig_hazard_deep.json", help="Deep RIG-Hazard experiment JSON configuration."
    )
    deep_extension_warning_parser.add_argument(
        "--device", default="auto", help="Torch device, for example auto, cpu, or cuda."
    )
    deep_extension_warning_parser.add_argument(
        "--resume", action="store_true", help="Reuse completed station-year extension predictions."
    )
    deep_extension_warning_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing extension warning directory."
    )
    deep_graph_gate_parser = subparsers.add_parser(
        "deep-graph-gate", help="Run 2023 stable-graph lead and time-misalignment placebo gates."
    )
    deep_graph_gate_parser.add_argument(
        "--config", default="configs/rig_hazard_deep.json", help="Deep RIG-Hazard experiment JSON configuration."
    )
    deep_graph_gate_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing stable-graph gate directory."
    )
    recurrence_structure_parser = subparsers.add_parser(
        "recurrence-structure",
        help="Describe recurrent-event order, gap time, seasonality, and first-versus-recurrent performance.",
    )
    recurrence_structure_parser.add_argument(
        "--events",
        default="results/preprocessed_data_10min/events_recurrent.csv",
        help="Recurrent event table.",
    )
    recurrence_structure_parser.add_argument(
        "--event-records",
        default="",
        help="Comma-separated event-level warning record files; use an empty value to skip performance analysis.",
    )
    recurrence_structure_parser.add_argument(
        "--output-root",
        default="results/recurrence_analysis/structure",
        help="Output directory.",
    )
    seasonal_parser = subparsers.add_parser(
        "recurrence-seasonal", help="Compare global and November-April seasonal recurrence orders."
    )
    seasonal_parser.add_argument(
        "--events", default="results/preprocessed_data_10min/events_recurrent.csv"
    )
    seasonal_parser.add_argument(
        "--output-root",
        default="results/recurrence_modeling/seasonal_recurrence",
    )
    seasonal_parser.add_argument("--overwrite", action="store_true")
    recurrence_structure_parser.add_argument(
        "--bootstrap-replicates", type=int, default=1000, help="Station-cluster bootstrap replicates."
    )
    recurrence_structure_parser.add_argument("--overwrite", action="store_true", help="Replace the output directory.")
    recurrence_sensitivity_parser = subparsers.add_parser(
        "recurrence-sensitivity",
        help="Run the full factorial recurrence-definition sensitivity analysis.",
    )
    recurrence_sensitivity_parser.add_argument(
        "--config",
        default="configs/rig_hazard_preprocessing.json",
        help="Preprocessing configuration path.",
    )
    recurrence_sensitivity_parser.add_argument(
        "--output-root",
        default="results/recurrence_analysis/definition_sensitivity",
        help="Output directory.",
    )
    recurrence_sensitivity_parser.add_argument(
        "--persistence-records",
        type=int,
        default=3,
        help="Consecutive records required by the resolution-aware thickness rule.",
    )
    recurrence_sensitivity_parser.add_argument("--overwrite", action="store_true", help="Replace the output directory.")
    protocol_parser = subparsers.add_parser(
        "protocol-folds",
        help="Build purged 2022 calendar-block folds and locked 2023/2024 cache aliases.",
    )
    protocol_parser.add_argument(
        "--cache-root",
        default="results/recurrence_analysis/model_comparison/cache",
        help="Deep cache containing train, validation, and test indices.",
    )
    protocol_parser.add_argument(
        "--output-root",
        default="results/recurrence_analysis/temporal_protocol",
        help="Protocol metadata output directory.",
    )
    protocol_parser.add_argument("--folds", type=int, default=5, help="Number of 2022 folds.")
    protocol_parser.add_argument("--block-days", type=int, default=7, help="Calendar block length in days.")
    protocol_parser.add_argument("--purge-hours", type=float, default=30.0, help="Purge around held-out blocks.")
    protocol_parser.add_argument("--overwrite", action="store_true", help="Replace protocol metadata.")
    recurrence_model_parser = subparsers.add_parser(
        "recurrence-statistical",
        help="Run 2022 block-CV PWP gap-time and station-frailty cloglog comparisons.",
    )
    recurrence_model_parser.add_argument(
        "--config",
        default="configs/rig_hazard_recurrence_models.json",
        help="Recurrence model JSON configuration.",
    )
    recurrence_model_parser.add_argument("--output-root", default=None, help="Override statistical output directory.")
    recurrence_model_parser.add_argument("--folds", type=int, default=5, help="Number of prepared 2022 folds.")
    recurrence_model_parser.add_argument(
        "--max-train-rows", type=int, default=None, help="Optional engineering-check training limit."
    )
    recurrence_model_parser.add_argument(
        "--max-evaluation-rows", type=int, default=None, help="Optional engineering-check evaluation limit."
    )
    recurrence_model_parser.add_argument(
        "--no-save-predictions", action="store_true", help="Skip compressed row-level locked-year predictions."
    )
    recurrence_model_parser.add_argument("--overwrite", action="store_true", help="Replace statistical outputs.")
    deep_protocol_parser = subparsers.add_parser(
        "deep-protocol",
        help="Run 2022 block-CV calibration, fixed 2022 refit, and locked 2023/2024 deep evaluation.",
    )
    deep_protocol_parser.add_argument(
        "--config",
        default="configs/rig_hazard_recurrence_models.json",
        help="Recurrence model JSON configuration.",
    )
    deep_protocol_parser.add_argument(
        "--models", default=None, help="Comma-separated encoders, for example gru,recurrent_dual."
    )
    deep_protocol_parser.add_argument("--seeds", default=None, help="Comma-separated random seeds.")
    deep_protocol_parser.add_argument("--folds", type=int, default=5, help="Number of prepared 2022 folds.")
    deep_protocol_parser.add_argument("--max-epochs", type=int, default=None, help="Maximum fold-training epochs.")
    deep_protocol_parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Optional engineering-check training limit."
    )
    deep_protocol_parser.add_argument(
        "--max-evaluation-samples", type=int, default=None, help="Optional engineering-check evaluation limit."
    )
    deep_protocol_parser.add_argument("--output-root", default=None, help="Override deep protocol output directory.")
    deep_protocol_parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu, or cuda.")
    deep_protocol_parser.add_argument("--resume", action="store_true", help="Reuse compatible completed folds and final fits.")
    deep_protocol_parser.add_argument("--overwrite", action="store_true", help="Replace deep protocol outputs.")
    budget_parser = subparsers.add_parser(
        "select-budget-2022",
        help="Freeze model-specific causal alarm controls under one 2022 OOF monthly budget.",
    )
    budget_parser.add_argument(
        "--config",
        default="configs/rig_hazard_recurrence_models.json",
        help="Recurrence model JSON configuration.",
    )
    budget_parser.add_argument("--deep-protocol-root", default=None, help="Override the deep protocol output root.")
    budget_parser.add_argument("--output-root", default=None, help="Override the frozen budget output root.")
    budget_parser.add_argument("--overwrite", action="store_true", help="Replace budget-selection outputs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "preprocess":
        config, config_path, config_hash = load_config(Path(args.config))
        preprocess(
            config=config,
            config_path=config_path,
            config_hash=config_hash,
            output_root_override=args.output_root,
            max_stations=args.max_stations,
            selected_stations=comma_set(args.stations),
            selected_years=year_set(args.years),
            overwrite=args.overwrite,
        )
        return
    if args.command == "validate":
        validate_preprocessed(args.output_root, strict=not args.no_strict)
        return
    if args.command == "baseline":
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_baseline_experiment(config, config_path, overwrite=args.overwrite)
        return
    if args.command == "graph":
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_graph_experiment(config, config_path, overwrite=args.overwrite)
        return
    if args.command == "stability":
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_stability_experiment(config, config_path, overwrite=args.overwrite)
        return
    if args.command == "deep-audit":
        from .deep_audit import run_deep_audit

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_deep_audit(config, config_path, overwrite=args.overwrite)
        return
    if args.command == "deep-cache":
        from .deep_data import build_deep_cache

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        build_deep_cache(config, config_path, overwrite=args.overwrite)
        return
    if args.command == "deep-local":
        from .deep_training import run_deep_local_experiment

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_deep_local_experiment(
            config,
            config_path,
            overwrite=args.overwrite,
            models=sorted(comma_set(args.models)) if args.models else None,
            seeds=sorted(year_set(args.seeds)) if args.seeds else None,
            maximum_epochs=args.max_epochs,
            maximum_train_samples=args.max_train_samples,
            maximum_validation_samples=args.max_validation_samples,
            output_root_override=args.output_root,
            device_name=args.device,
            resume=args.resume,
            train_split=args.train_split,
            validation_split=args.validation_split,
        )
        return
    if args.command == "deep-warning":
        from .deep_warning import run_deep_warning

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_deep_warning(
            config,
            config_path,
            overwrite=args.overwrite,
            resume=args.resume,
            device_name=args.device,
            models=sorted(comma_set(args.models)) if args.models else None,
            local_root_override=args.local_root,
            output_root_override=args.output_root,
            evaluation_year_override=args.evaluation_year,
            history_split_override=args.history_split,
            evaluation_split_override=args.evaluation_split,
            disable_matched_diagnostic=args.disable_matched_diagnostic,
        )
        return
    if args.command == "deep-extensions":
        from .deep_extensions import run_deep_extension_experiment

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_deep_extension_experiment(
            config,
            config_path,
            overwrite=args.overwrite,
            resume=args.resume,
            variants=sorted(comma_set(args.variants)) if args.variants else None,
            seeds=sorted(year_set(args.seeds)) if args.seeds else None,
            maximum_epochs=args.max_epochs,
            maximum_train_samples=args.max_train_samples,
            maximum_validation_samples=args.max_validation_samples,
            output_root_override=args.output_root,
            device_name=args.device,
        )
        return
    if args.command == "deep-extension-warning":
        from .deep_extension_warning import run_deep_extension_warning

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_deep_extension_warning(
            config,
            config_path,
            overwrite=args.overwrite,
            resume=args.resume,
            device_name=args.device,
        )
        return
    if args.command == "deep-graph-gate":
        from .deep_graph_gate import run_deep_graph_gate

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_deep_graph_gate(config, config_path, overwrite=args.overwrite)
        return
    if args.command == "recurrence-structure":
        from .recurrence_analysis import run_recurrence_structure

        record_paths = [item.strip() for item in args.event_records.split(",") if item.strip()]
        run_recurrence_structure(
            events_path=args.events,
            output_root=args.output_root,
            event_record_paths=record_paths,
            bootstrap_replicates=args.bootstrap_replicates,
            overwrite=args.overwrite,
        )
        return
    if args.command == "recurrence-seasonal":
        from .seasonal_recurrence import run_seasonal_recurrence

        run_seasonal_recurrence(args.events, args.output_root, overwrite=args.overwrite)
        return
    if args.command == "recurrence-sensitivity":
        from .recurrence_sensitivity import run_recurrence_sensitivity

        run_recurrence_sensitivity(
            config_path=args.config,
            output_root=args.output_root,
            persistence_records=args.persistence_records,
            overwrite=args.overwrite,
        )
        return
    if args.command == "protocol-folds":
        from .temporal_protocol import build_temporal_protocol_indices

        build_temporal_protocol_indices(
            cache_root=args.cache_root,
            output_root=args.output_root,
            number_folds=args.folds,
            block_days=args.block_days,
            purge_hours=args.purge_hours,
            overwrite=args.overwrite,
        )
        return
    if args.command == "recurrence-statistical":
        from .recurrence_models import run_recurrence_statistical_models

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_recurrence_statistical_models(
            config,
            config_path,
            output_root_override=args.output_root,
            number_folds=args.folds,
            maximum_train_rows=args.max_train_rows,
            maximum_evaluation_rows=args.max_evaluation_rows,
            save_predictions=not args.no_save_predictions,
            overwrite=args.overwrite,
        )
        return
    if args.command == "deep-protocol":
        from .deep_protocol import run_deep_temporal_protocol

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_deep_temporal_protocol(
            config,
            config_path,
            models=sorted(comma_set(args.models)) if args.models else None,
            seeds=sorted(year_set(args.seeds)) if args.seeds else None,
            number_folds=args.folds,
            maximum_epochs=args.max_epochs,
            maximum_train_samples=args.max_train_samples,
            maximum_evaluation_samples=args.max_evaluation_samples,
            output_root_override=args.output_root,
            device_name=args.device,
            overwrite=args.overwrite,
            resume=args.resume,
        )
        return
    if args.command == "select-budget-2022":
        from .protocol_budget import select_2022_oof_budget

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        select_2022_oof_budget(
            config,
            config_path,
            deep_protocol_root=args.deep_protocol_root,
            output_root_override=args.output_root,
            overwrite=args.overwrite,
        )
        return
    parser.error(f"Unknown command: {args.command}")

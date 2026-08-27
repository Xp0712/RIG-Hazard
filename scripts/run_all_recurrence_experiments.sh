#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/ice_project"
PYTHON="/root/miniconda3/bin/python"
LOG_DIR="$PROJECT_ROOT/results/logs"
CURRENT_STAGE="initialization"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONPATH="$PROJECT_ROOT/src"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

on_error() {
  local exit_code=$?
  printf '\n[%s] FAILED stage=%s exit_code=%s line=%s\n' \
    "$(date --iso-8601=seconds)" "$CURRENT_STAGE" "$exit_code" "$1"
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

run_stage() {
  CURRENT_STAGE="$1"
  shift
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds)" "$CURRENT_STAGE"
  "$@"
  printf '[%s] COMPLETE %s\n' "$(date --iso-8601=seconds)" "$CURRENT_STAGE"
}

wait_for_gpu() {
  CURRENT_STAGE="wait_for_gpu"
  while ! nvidia-smi -L >/dev/null 2>&1; do
    printf '[%s] GPU unavailable; waiting 60 seconds before deep training\n' \
      "$(date --iso-8601=seconds)"
    sleep 60
  done
  printf '[%s] GPU available; starting deep training\n' "$(date --iso-8601=seconds)"
}

run_stage "install_dependencies" \
  "$PYTHON" -m pip install -r requirements/deep_learning.txt

run_stage "unit_tests" \
  "$PYTHON" -m unittest discover -s tests

run_stage "compile_check" \
  "$PYTHON" -m compileall -q src scripts

run_stage "build_recurrence_cache" \
  "$PYTHON" -m rig_hazard deep-cache \
  --config configs/rig_hazard_recurrence_models.json --overwrite

run_stage "build_temporal_folds" \
  "$PYTHON" -m rig_hazard protocol-folds \
  --cache-root results/recurrence_analysis/model_comparison/cache \
  --folds 5 --block-days 7 --purge-hours 30 --overwrite

run_stage "recurrence_structure_initial" \
  "$PYTHON" -m rig_hazard recurrence-structure --overwrite

run_stage "recurrence_definition_sensitivity" \
  "$PYTHON" -m rig_hazard recurrence-sensitivity \
  --config configs/rig_hazard_preprocessing.json --persistence-records 3 --overwrite

run_stage "recurrence_statistical_models" \
  "$PYTHON" -m rig_hazard recurrence-statistical \
  --config configs/rig_hazard_recurrence_models.json --folds 5 --overwrite

wait_for_gpu

run_stage "deep_core_models" \
  "$PYTHON" -u -m rig_hazard deep-protocol \
  --config configs/rig_hazard_recurrence_models.json \
  --models gru,recurrent_dual \
  --seeds 20260807,20260817,20260827,20260837,20260847 \
  --folds 5 --device auto --overwrite

run_stage "deep_modern_baselines" \
  "$PYTHON" -u -m rig_hazard deep-protocol \
  --config configs/rig_hazard_recurrence_models.json \
  --models tcn,patchtst,timesnet,itransformer \
  --seeds 20260807,20260817,20260827,20260837,20260847 \
  --folds 5 --device auto \
  --output-root results/recurrence_analysis/model_comparison/deep_protocol_modern \
  --overwrite

run_stage "select_2022_warning_budget" \
  "$PYTHON" -m rig_hazard select-budget-2022 \
  --config configs/rig_hazard_recurrence_models.json --overwrite

run_stage "warning_evaluation_2023" \
  "$PYTHON" -m rig_hazard deep-warning \
  --config configs/rig_hazard_recurrence_models.json \
  --models gru,recurrent_dual \
  --local-root results/recurrence_analysis/model_comparison/deep_protocol/trained_models \
  --evaluation-year 2023 --history-split selection_2022_full \
  --evaluation-split cross_year_2023 \
  --output-root results/recurrence_analysis/model_comparison/warning_2023 \
  --device auto --overwrite

run_stage "warning_evaluation_2024" \
  "$PYTHON" -m rig_hazard deep-warning \
  --config configs/rig_hazard_recurrence_models.json \
  --models gru,recurrent_dual \
  --local-root results/recurrence_analysis/model_comparison/deep_protocol/trained_models \
  --evaluation-year 2024 --history-split cross_year_2023 \
  --evaluation-split final_time_2024 --disable-matched-diagnostic \
  --output-root results/recurrence_analysis/model_comparison/warning_2024 \
  --device auto --overwrite

run_stage "recurrence_structure_with_warnings" \
  "$PYTHON" -m rig_hazard recurrence-structure \
  --event-records "results/recurrence_analysis/model_comparison/warning_2023/event_records.csv,results/recurrence_analysis/model_comparison/warning_2024/event_records.csv" \
  --overwrite

CURRENT_STAGE="all_experiments"
printf '\n[%s] COMPLETE all experiments\n' "$(date --iso-8601=seconds)"

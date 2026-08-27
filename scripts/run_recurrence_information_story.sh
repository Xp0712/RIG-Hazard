#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/ice_project}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
BOOTSTRAP_TOP="${BOOTSTRAP_TOP:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/recurrence_information_story}"
LOG_ROOT="${LOG_ROOT:-results/logs/recurrence_information_story}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

MASTER_LOG="$LOG_ROOT/master.log"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MASTER_LOG"
}

run_stage() {
  local stage="$1"
  shift
  local marker="$OUTPUT_ROOT/.complete_${stage}"
  local stage_log="$LOG_ROOT/${stage}.log"

  if [[ -f "$marker" ]]; then
    log "SKIP $stage (completion marker exists)"
    return 0
  fi

  log "START $stage"
  "$@" 2>&1 | tee "$stage_log"
  touch "$marker"
  log "COMPLETE $stage"
}

log "RUN recurrence information story; bootstrap_replicates=$BOOTSTRAP_REPLICATES bootstrap_top=$BOOTSTRAP_TOP"

run_stage conditional_value \
  "$PYTHON_BIN" -u scripts/explore_conditional_information_value.py \
  --config configs/rig_hazard_recurrence_modeling.json \
  --output-root "$OUTPUT_ROOT/conditional_value" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --overwrite

run_stage conditional_combinations \
  "$PYTHON_BIN" -u scripts/explore_conditional_information_combinations.py \
  --config configs/rig_hazard_recurrence_modeling.json \
  --output-root "$OUTPUT_ROOT/conditional_combinations" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --bootstrap-top "$BOOTSTRAP_TOP" \
  --overwrite

run_stage failure_diagnostics \
  "$PYTHON_BIN" -u scripts/diagnose_recurrence_information_failures.py \
  --config configs/rig_hazard_recurrence_modeling.json \
  --output-root "$OUTPUT_ROOT/failure_diagnostics" \
  --overwrite

touch "$OUTPUT_ROOT/.complete_all"
log "COMPLETE all recurrence information story analyses"

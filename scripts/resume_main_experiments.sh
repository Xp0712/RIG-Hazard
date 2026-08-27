#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/ice_project"
PYTHON="/root/miniconda3/bin/python"
CURRENT_STAGE="initialization"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONPATH="$PROJECT_ROOT/src"

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

run_stage "strict_event_contract_unit_tests" \
  "$PYTHON" -m unittest discover -s tests

run_stage "frozen_dynamic_hard_budget" \
  "$PYTHON" -u scripts/run_dynamic_hard_budget_experiments.py --phase all

run_stage "public_recurrence_benchmarks" \
  "$PYTHON" -u scripts/run_public_recurrence_benchmarks.py \
  --datasets ecommerce us_accidents \
  --output-root results/dynamic_hard_budget/public_recurrence_benchmarks \
  --resume

CURRENT_STAGE="all_strict_event_contract_experiments"
printf '\n[%s] COMPLETE all strict-event-contract experiments\n' "$(date --iso-8601=seconds)"

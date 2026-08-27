#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ice_project
PYTHON=/root/miniconda3/bin/python
CONFIG=configs/rig_hazard_alert_governance.json
RUN_ROOT=results/alert_governance
mkdir -p "$RUN_ROOT/logs"
export PYTHONPATH=/root/autodl-tmp/ice_project/src
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
  local name="$1"
  shift
  if [[ -f "$RUN_ROOT/.complete_${name}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${name}"
    return
  fi
  echo "[$(date -Iseconds)] START ${name}"
  "$@" 2>&1 | tee "$RUN_ROOT/logs/${name}.log"
  touch "$RUN_ROOT/.complete_${name}"
  echo "[$(date -Iseconds)] COMPLETE ${name}"
}

echo "[$(date -Iseconds)] LOCK main=rec_none baselines=GRU,TimesNet,XGBoost"
echo "[$(date -Iseconds)] STOPPED extensions=rec_full,dual_time_scale,station_graph"

stage nested_alert_audit "$PYTHON" -u scripts/run_nested_alert_audit.py \
  --config "$CONFIG" --overwrite

stage xgboost_baseline "$PYTHON" -u scripts/run_xgboost_hazard_baseline.py \
  --config "$CONFIG" --resume

stage alert_strategy_comparison "$PYTHON" -u scripts/run_alert_strategy_comparison.py \
  --config "$CONFIG" --resume

stage public_recurrence_benchmarks "$PYTHON" -u scripts/run_public_recurrence_benchmarks.py \
  --config "$CONFIG" --resume

echo "[$(date -Iseconds)] ALL ALERT GOVERNANCE EXPERIMENTS COMPLETE"

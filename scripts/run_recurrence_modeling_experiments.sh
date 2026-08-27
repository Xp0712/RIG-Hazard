#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ice_project
PYTHON=/root/miniconda3/bin/python
ROOT=results/recurrence_modeling
mkdir -p "$ROOT/logs"
export PYTHONPATH=/root/autodl-tmp/ice_project/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export RIG_HAZARD_FAST_CUDA=1

stage() {
  local name="$1"
  shift
  if [[ -f "$ROOT/.complete_${name}" ]]; then
    echo "SKIP ${name}"
    return
  fi
  echo "[$(date -Iseconds)] START ${name}"
  "$@" 2>&1 | tee "$ROOT/logs/${name}.log"
  touch "$ROOT/.complete_${name}"
  echo "[$(date -Iseconds)] COMPLETE ${name}"
}

stage lock_protocol "$PYTHON" scripts/prepare_recurrence_modeling_config.py
stage seasonal_recurrence "$PYTHON" -m rig_hazard recurrence-seasonal --overwrite
stage recurrence_sensitivity "$PYTHON" -m rig_hazard recurrence-sensitivity \
  --config configs/rig_hazard_preprocessing.json \
  --output-root "$ROOT/definition_sensitivity" --overwrite

MODELS=rec_none,rec_gap,rec_load,rec_order,rec_previous,rec_full,rec_shuffled,rec_full_uniform
SEEDS=20260807,20260817,20260827,20260837,20260847
run_probability_models() {
  local mode=--overwrite
  if [[ -f "$ROOT/probability_models/protocol_request.json" ]]; then
    mode=--resume
  fi
  "$PYTHON" -u -m rig_hazard deep-protocol \
    --config configs/rig_hazard_recurrence_modeling.json --models "$MODELS" --seeds "$SEEDS" \
    --folds 5 --device auto "$mode"
}
stage probability_models run_probability_models
stage freeze_model "$PYTHON" scripts/select_recurrence_model.py
stage probability_bootstrap "$PYTHON" scripts/paired_station_bootstrap.py

SELECTED=$($PYTHON -c 'import json; print(json.load(open("results/recurrence_modeling/probability_models/frozen_model_selection.json"))["selected_model"])')
stage budget_utility "$PYTHON" -m rig_hazard select-budget-2022 \
  --config configs/rig_hazard_recurrence_modeling.json \
  --deep-protocol-root "$ROOT/probability_models" \
  --output-root "$ROOT/budget_2022_oof" --overwrite
stage locked_attribution "$PYTHON" scripts/run_locked_attribution.py

echo "Selected probability model: $SELECTED"
echo "Next-stage pipeline complete. Attribution was run strictly downstream of model freeze."

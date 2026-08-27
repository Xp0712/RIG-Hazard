#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ice_project
PYTHON=/root/miniconda3/bin/python
ROOT=results/recurrence_modeling
SUPPLEMENTARY_ROOT="$ROOT/supplementary_experiments"
RISK_ROOT="$ROOT/seasonal_risk_structure"
SPATIAL_ROOT="$ROOT/spatial_generalization"
mkdir -p "$SPATIAL_ROOT"

export PYTHONPATH=/root/autodl-tmp/ice_project/src
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export RIG_HAZARD_FAST_CUDA=1

wait_for_file_or_process() {
  local required_file="$1"
  local process_pattern="$2"
  local stage_name="$3"
  while [[ ! -f "$required_file" ]]; do
    if ! pgrep -af "$process_pattern" >/dev/null; then
      echo "[$(date -Iseconds)] ERROR ${stage_name}: process ended without ${required_file}"
      exit 1
    fi
    echo "[$(date -Iseconds)] WAIT ${stage_name}"
    sleep 300
  done
  echo "[$(date -Iseconds)] READY ${stage_name}"
}

wait_for_file_or_process \
  "$SUPPLEMENTARY_ROOT/.complete_fair_baseline_summary" \
  "run_supplementary_experiments.sh" \
  "supplementary_experiments"

wait_for_file_or_process \
  "$RISK_ROOT/seasonal_risk_structure_bundle.json" \
  "run_seasonal_risk_structure|seasonal-risk-structure" \
  "seasonal_risk_structure"

if [[ -f "$SPATIAL_ROOT/spatial_generalization_manifest.json" ]]; then
  echo "[$(date -Iseconds)] SKIP spatial_generalization already complete"
  exit 0
fi

mode=--overwrite
if [[ -f "$SPATIAL_ROOT/protocol_request.json" ]]; then
  mode=--resume
fi

echo "[$(date -Iseconds)] START spatial_generalization mode=${mode}"
"$PYTHON" -u scripts/run_spatial_generalization.py \
  --config configs/rig_hazard_recurrence_modeling.json \
  --output-root "$SPATIAL_ROOT" \
  --model rec_none \
  --seeds 20260807,20260817,20260827,20260837,20260847 \
  --station-folds 5 \
  --inner-folds 0,1,2 \
  --protocols station_group_cv,region_loco \
  --maximum-train-samples 120000 \
  --maximum-early-samples 20000 \
  --maximum-calibration-samples 120000 \
  --training-batch-size 512 \
  --inference-batch-size 4096 \
  --prefetch-batches 2 \
  --cpu-threads 8 \
  --bootstrap-samples 5000 \
  --bootstrap-batch-size 16 \
  --device auto \
  "$mode"
touch "$SPATIAL_ROOT/.complete_spatial_generalization"
echo "[$(date -Iseconds)] COMPLETE spatial_generalization"

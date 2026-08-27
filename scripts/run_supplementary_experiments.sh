#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ice_project
PYTHON=/root/miniconda3/bin/python
ROOT=results/recurrence_modeling
RUN_ROOT="$ROOT/supplementary_experiments"
mkdir -p "$RUN_ROOT/logs"
export PYTHONPATH=/root/autodl-tmp/ice_project/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export RIG_HAZARD_FAST_CUDA=1
export RIG_HAZARD_PATCHTST_MICROBATCH_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
  local name="$1"
  shift
  if [[ -f "$RUN_ROOT/.complete_${name}" ]]; then
    echo "SKIP ${name}"
    return
  fi
  echo "[$(date -Iseconds)] START ${name}"
  "$@" 2>&1 | tee "$RUN_ROOT/logs/${name}.log"
  touch "$RUN_ROOT/.complete_${name}"
  echo "[$(date -Iseconds)] COMPLETE ${name}"
}

wait_for_gpu() {
  local gpu_state
  gpu_state=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || echo "GPU state unavailable")
  echo "[$(date -Iseconds)] SHARED_GPU no wait; current state: ${gpu_state}"
}

stage utility_warning "$PYTHON" -u scripts/run_utility_warning_experiments.py \
  --config configs/rig_hazard_recurrence_modeling.json \
  --models rec_none,rec_load,rec_previous,rec_full,rec_full_uniform \
  --budgets 2,5,10,20 \
  --quantiles 0.80,0.85,0.90,0.92,0.94,0.96,0.98,0.99,0.995 \
  --bootstrap-samples 5000 --overwrite

stage targeted_probability_bootstrap "$PYTHON" -u scripts/paired_probability_bootstrap.py \
  --config configs/rig_hazard_recurrence_modeling.json \
  --comparisons rec_load:rec_none,rec_previous:rec_none \
  --replicates 5000 --batch-size 16

wait_for_gpu
stage candidate_attribution "$PYTHON" -u scripts/run_candidate_attribution.py \
  --config configs/rig_hazard_recurrence_modeling.json \
  --models rec_full,rec_load,rec_previous \
  --permutation-rows 120000 --ig-rows 2048 --ig-steps 32 \
  --batch-size 256 --device auto

stage prepare_fair_baselines "$PYTHON" scripts/prepare_fair_baseline_config.py

run_fair_baselines() {
  local mode=--overwrite
  if [[ -f "$ROOT/fair_baselines/protocol_request.json" ]]; then
    mode=--resume
  fi
  "$PYTHON" -u -m rig_hazard deep-protocol \
    --config configs/rig_hazard_fair_baselines.json \
    --models fair_gru,fair_tcn,fair_patchtst,fair_timesnet,fair_itransformer \
    --seeds 20260807,20260817,20260827,20260837,20260847 \
    --folds 5 --device auto "$mode"
}

stage fair_baselines run_fair_baselines
stage fair_baseline_summary "$PYTHON" scripts/summarize_fair_baselines.py

echo "[$(date -Iseconds)] ALL REQUESTED REMAINING EXPERIMENTS COMPLETE"

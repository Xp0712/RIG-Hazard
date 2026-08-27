#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ice_project
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
CONFIG=configs/rig_hazard_icing_model_experiments.json
ROOT=results/icing_model_experiments
WEATHER_ABLATION_ROOT="$ROOT/weather_ablation"
RECURRENCE_GATE_ROOT="$ROOT/recurrence_gate_ablation"
RUN_ROOT="$ROOT/pipeline"
mkdir -p "$RUN_ROOT/logs"

export PYTHONPATH=/root/autodl-tmp/ice_project/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export RIG_HAZARD_FAST_CUDA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
  local name="$1"
  shift
  if [[ -f "$RUN_ROOT/.complete_${name}" ]]; then
    echo "[$(date -Iseconds)] SKIP completed ${name}"
    return
  fi
  echo "[$(date -Iseconds)] START ${name}"
  "$@" 2>&1 | tee "$RUN_ROOT/logs/${name}.log"
  touch "$RUN_ROOT/.complete_${name}"
  echo "[$(date -Iseconds)] COMPLETE ${name}"
}

run_deep_stage() {
  local output_root="$1"
  local models="$2"
  local mode=--overwrite
  if [[ -f "$output_root/protocol_request.json" ]]; then
    mode=--resume
  fi
  "$PYTHON" -u -m rig_hazard deep-protocol \
    --config "$CONFIG" \
    --models "$models" \
    --seeds 20260807,20260817,20260827,20260837,20260847 \
    --folds 5 --device auto --output-root "$output_root" "$mode"
}

stage reuse_inventory "$PYTHON" scripts/write_icing_reuse_manifest.py

stage protocol_audit "$PYTHON" -u scripts/audit_icing_experiment_protocol.py \
  --config "$CONFIG" --output-root "$ROOT/protocol_audit" --overwrite

# Only the orthogonal weather models are trained here. Existing rule and
# local hazard baselines are reused, as are completed fair baselines.
stage weather_ablation run_deep_stage "$WEATHER_ABLATION_ROOT" \
  gru_weather_reference,fast_weather_encoder,slow_weather_encoder,dual_weather_encoder

# These two recurrence variants are new. Their scientific retention is judged
# later on locked years, but downstream selection remains 2022-OOF-only.
stage recurrence_gate_ablation run_deep_stage "$RECURRENCE_GATE_ROOT" \
  dual_weather_recurrence,gated_dual_weather_recurrence

stage selection_year_model_selection "$PYTHON" -u scripts/select_icing_probability_model.py \
  --candidate "gru_weather_reference=$WEATHER_ABLATION_ROOT" \
  --candidate "fast_weather_encoder=$WEATHER_ABLATION_ROOT" \
  --candidate "slow_weather_encoder=$WEATHER_ABLATION_ROOT" \
  --candidate "dual_weather_encoder=$WEATHER_ABLATION_ROOT" \
  --candidate "dual_weather_recurrence=$RECURRENCE_GATE_ROOT" \
  --candidate "gated_dual_weather_recurrence=$RECURRENCE_GATE_ROOT" \
  --reference gru_weather_reference --output-root "$ROOT/selection_year_model_selection"

SELECTED_MODEL=$(
  "$PYTHON" -c "import json; print(json.load(open('$ROOT/selection_year_model_selection/selected_model.json'))['selected_model'])"
)
SELECTED_ROOT=$(
  "$PYTHON" -c "import json; print(json.load(open('$ROOT/selection_year_model_selection/selected_model.json'))['selected_protocol_root'])"
)
SELECTED_CONFIG="$CONFIG"
echo "[$(date -Iseconds)] FROZEN 2022 selection model=${SELECTED_MODEL} root=${SELECTED_ROOT}"

stage selected_model_trajectory "$PYTHON" -u scripts/export_selected_full_trajectory.py \
  --config "$SELECTED_CONFIG" --protocol-root "$SELECTED_ROOT" \
  --model "$SELECTED_MODEL" --output-root "$ROOT/selected_model_trajectory" \
  --device auto --resume

stage nested_alert "$PYTHON" -u scripts/run_nested_alert_policy.py \
  --config "$SELECTED_CONFIG" --protocol-root "$SELECTED_ROOT" \
  --model "$SELECTED_MODEL" --output-root "$ROOT/nested_alert" --resume

MODEL_ROOTS="gru_weather_reference=$WEATHER_ABLATION_ROOT,fast_weather_encoder=$WEATHER_ABLATION_ROOT,slow_weather_encoder=$WEATHER_ABLATION_ROOT,dual_weather_encoder=$WEATHER_ABLATION_ROOT,dual_weather_recurrence=$RECURRENCE_GATE_ROOT,gated_dual_weather_recurrence=$RECURRENCE_GATE_ROOT"
COMPARISONS="dual_weather_encoder:gru_weather_reference,dual_weather_encoder:fast_weather_encoder,dual_weather_encoder:slow_weather_encoder,dual_weather_recurrence:dual_weather_encoder,gated_dual_weather_recurrence:dual_weather_encoder"

stage selection_year_bootstrap "$PYTHON" -u scripts/paired_probability_bootstrap.py \
  --config "$CONFIG" --comparisons "$COMPARISONS" --model-roots "$MODEL_ROOTS" \
  --years 2022 --replicates 5000 --batch-size 16 \
  --output-root "$ROOT/selection_year_bootstrap"

stage locked_year_bootstrap "$PYTHON" -u scripts/paired_probability_bootstrap.py \
  --config "$CONFIG" --comparisons "$COMPARISONS" --model-roots "$MODEL_ROOTS" \
  --years 2023,2024 --replicates 5000 --batch-size 16 \
  --output-root "$ROOT/locked_year_bootstrap"

stage reuse_inventory_refresh "$PYTHON" scripts/write_icing_reuse_manifest.py
touch "$ROOT/.complete_icing_model_experiments"
echo "[$(date -Iseconds)] ICING MODEL EXPERIMENT PIPELINE COMPLETE"

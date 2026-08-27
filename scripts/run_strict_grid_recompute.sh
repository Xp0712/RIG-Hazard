#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/ice_project
PYTHON=/root/miniconda3/bin/python
CONFIG=configs/rig_hazard_alert_governance.json
OUTPUT_ROOT=results/alert_governance
STRATEGY_FINAL="$OUTPUT_ROOT/strategy_comparison"
NESTED_FINAL="$OUTPUT_ROOT/nested_alert_audit"
STRATEGY_STAGING="$OUTPUT_ROOT/strategy_comparison_staging"
NESTED_STAGING="$OUTPUT_ROOT/nested_alert_audit_staging"
RUN_ID="strict_event_grid_$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="/root/autodl-tmp/ice_project_result_backups/$RUN_ID"

cd "$PROJECT_ROOT"
mkdir -p results/logs "$BACKUP_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

echo "[$(date -Iseconds)] START strict-event-grid strategy comparison"
"$PYTHON" -u scripts/run_alert_strategy_comparison.py \
  --config "$CONFIG" --output-root "$STRATEGY_STAGING" --resume

echo "[$(date -Iseconds)] START strict-event-grid nested audit"
"$PYTHON" -u scripts/run_nested_alert_audit.py \
  --config "$CONFIG" --output-root "$NESTED_STAGING" --overwrite

"$PYTHON" - "$STRATEGY_STAGING" "$NESTED_STAGING" <<'PY'
from pathlib import Path
import sys
import pandas as pd

strategy = Path(sys.argv[1])
nested = Path(sys.argv[2])
main = pd.read_csv(strategy / "frozen_main_baseline_table.csv")
audit = pd.read_csv(strategy / "event_eligibility_audit.csv")
funnel = pd.read_csv(strategy / "event_eligibility_funnel.csv")
assert main.shape[0] == 40, main.shape
assert set(main["year"]) == {2023, 2024}
assert main.groupby(["year", "budget_hours"])["strategy"].nunique().eq(5).all()
assert not audit["event_id"].duplicated().any()
frozen = funnel.loc[funnel["year"].isin([2023, 2024])].set_index("year")
assert int(frozen.loc[2023, "standardized_queue_events"]) == 84
assert int(frozen.loc[2024, "standardized_queue_events"]) == 116
assert int(frozen.loc[2023, "operational_queue_events"]) == 113
assert int(frozen.loc[2024, "operational_queue_events"]) == 170
required = {
    "evaluation_contract_version",
    "event_table_sha256",
    "prediction_bundle_sha256",
    "selection_policy_file_sha256",
}
assert required.issubset(main.columns), required.difference(main.columns)
assert (nested / "strict_unified_metrics.csv").exists()
assert (nested / "legacy_contract_metrics.csv").exists()
print("STRICT_EVENT_GRID_VALIDATION_OK")
PY

if [[ -d "$STRATEGY_FINAL" ]]; then
  mv "$STRATEGY_FINAL" "$BACKUP_ROOT/strategy_comparison"
fi
if [[ -d "$NESTED_FINAL" ]]; then
  mv "$NESTED_FINAL" "$BACKUP_ROOT/nested_alert_audit"
fi
mv "$STRATEGY_STAGING" "$STRATEGY_FINAL"
mv "$NESTED_STAGING" "$NESTED_FINAL"

echo "[$(date -Iseconds)] COMPLETE strict-event-grid; backup=$BACKUP_ROOT"

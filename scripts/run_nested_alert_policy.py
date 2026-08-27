from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.config import resolve_project_path
from rig_hazard.nested_alert import (
    AlertShape,
    apply_alert_policy,
    apply_alert_policy_grid,
    assert_nested_alert_masks,
    evaluate_event_alerts,
    policy_payload,
    select_nested_thresholds,
)
from rig_hazard.utility_warning import _ensemble_frame, _events


BUDGETS = (2.0, 5.0, 10.0, 20.0)


def _shape_id(shape: AlertShape) -> str:
    return (
        f"a{shape.ema_alpha:g}_c{shape.minimum_consecutive_bins}_"
        f"h{shape.hold_bins}_m{shape.merge_gap_bins}"
    ).replace(".", "p")


def _candidate_table(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    shape: AlertShape,
    quantiles: list[float],
    step_minutes: int,
) -> pd.DataFrame:
    values = pd.to_numeric(frame["risk_6h"], errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    thresholds = sorted({float(np.quantile(finite, value)) for value in quantiles}, reverse=True)
    prepared, alarm_matrix = apply_alert_policy_grid(
        frame, "risk_6h", thresholds, shape
    )
    rows: list[dict[str, object]] = []
    for threshold_index, threshold in enumerate(thresholds):
        controlled = prepared.assign(
            budget_alarm=alarm_matrix[:, threshold_index]
        )
        metrics, _ = evaluate_event_alerts(
            controlled,
            events,
            "budget_alarm",
            max(BUDGETS),
            step_minutes=step_minutes,
        )
        for budget in BUDGETS:
            rows.append(
                {
                    **asdict(shape),
                    "shape_id": _shape_id(shape),
                    "threshold": threshold,
                    "budget_hours": budget,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"model", "false_alarm_budget_hours_per_station_month", "budget_met"}
                    },
                    "budget_met": int(
                        float(metrics["false_alarm_hours_per_station_month"]) <= budget + 1e-9
                    ),
                }
            )
    return pd.DataFrame(rows)


def _load_year(
    cache_root: Path,
    protocol_root: Path,
    model: str,
    seeds: list[int],
    year: int,
) -> pd.DataFrame:
    return _ensemble_frame(
        cache_root,
        protocol_root / "locked_predictions" / str(year),
        model,
        seeds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and freeze a causal nested alert policy.")
    parser.add_argument("--config", default="configs/rig_hazard_icing_model_experiments.json")
    parser.add_argument("--protocol-root", default=None)
    parser.add_argument("--model", default="dual_weather_encoder")
    parser.add_argument(
        "--output-root",
        default="results/icing_model_experiments/nested_alert",
    )
    parser.add_argument(
        "--quantiles",
        default=(
            "0.80,0.85,0.90,0.92,0.94,0.96,0.97,0.98,0.985,0.99,"
            "0.992,0.995,0.997,0.998,0.999,0.9995,0.9998"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve_project_path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_root = resolve_project_path(args.protocol_root or config["deep_protocol_output_root"])
    cache_root = resolve_project_path(config["cache_root"])
    seeds = [int(value) for value in config["training"]["seeds"]]
    quantiles = [float(value) for value in args.quantiles.split(",")]
    seasonal_path = resolve_project_path(
        "results/recurrence_modeling/seasonal_recurrence/event_global_seasonal_mapping.csv"
    )
    events = _events(config, seasonal_path)
    selection_events = events.loc[events["onset_time"].dt.year.eq(2022)].copy()
    selection = _ensemble_frame(
        cache_root, protocol_root / "oof_predictions", args.model, seeds
    )
    step_minutes = int(config["step_minutes"])
    grid = config.get("nested_alert", {})
    shapes = [
        AlertShape(float(alpha), int(consecutive), int(hold // step_minutes), int(merge // step_minutes))
        for alpha, consecutive, hold, merge in product(
            grid.get("ema_alpha", [1.0, 0.5, 0.25]),
            grid.get("minimum_consecutive_bins", [1, 2, 3]),
            grid.get("hold_minutes", [30, 60, 120]),
            grid.get("merge_gap_minutes", [20, 40, 60]),
        )
    ]
    selected_shapes: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    shape_root = output_root / "shape_candidates"
    shape_root.mkdir(parents=True, exist_ok=True)
    for index, shape in enumerate(shapes, start=1):
        path = shape_root / f"{_shape_id(shape)}.csv"
        if args.resume and path.exists():
            candidates = pd.read_csv(path)
        else:
            candidates = _candidate_table(
                selection, selection_events, shape, quantiles, step_minutes
            )
            candidates.to_csv(path, index=False)
        candidate_parts.append(candidates)
        try:
            chosen = select_nested_thresholds(candidates, BUDGETS)
        except ValueError:
            continue
        chosen["mean_utility_across_budgets"] = float(chosen["lead_utility_hours"].mean())
        chosen["mean_hit_rate_across_budgets"] = float(chosen["event_hit_rate"].mean())
        selected_shapes.append(chosen)
        print(f"Nested alert shape {index}/{len(shapes)} complete: {_shape_id(shape)}", flush=True)
    if not selected_shapes:
        raise RuntimeError("No common causal alert shape satisfies all four budgets")
    all_selected = pd.concat(selected_shapes, ignore_index=True)
    shape_summary = all_selected.groupby("shape_id", as_index=False).first()
    best_shape_id = str(
        shape_summary.sort_values(
            ["mean_utility_across_budgets", "mean_hit_rate_across_budgets", "alert_segments"],
            ascending=[False, False, True],
        ).iloc[0]["shape_id"]
    )
    selected = all_selected.loc[all_selected["shape_id"].eq(best_shape_id)].copy()
    shape = AlertShape(
        float(selected["ema_alpha"].iloc[0]),
        int(selected["minimum_consecutive_bins"].iloc[0]),
        int(selected["hold_bins"].iloc[0]),
        int(selected["merge_gap_bins"].iloc[0]),
    )
    pd.concat(candidate_parts, ignore_index=True).to_csv(
        output_root / "candidate_metrics.csv.gz", index=False, compression="gzip"
    )
    all_selected.to_csv(output_root / "selected_thresholds_by_shape.csv", index=False)
    selected.to_csv(output_root / "selected_nested_thresholds.csv", index=False)
    policy = policy_payload(shape, selected)
    policy.update(
        {
            "model": args.model,
            "seeds": seeds,
            "protocol_root": str(protocol_root),
            "selected_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    (output_root / "policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )

    year_frames = {2023: _load_year(cache_root, protocol_root, args.model, seeds, 2023)}
    year_frames[2024] = _load_year(cache_root, protocol_root, args.model, seeds, 2024)
    metric_rows: list[dict[str, object]] = []
    event_parts: list[pd.DataFrame] = []
    for year in (2023, 2024):
        history = selection if year == 2023 else year_frames[2023]
        combined = pd.concat([history, year_frames[year]], ignore_index=True).sort_values(
            ["station_code", "issue_time"]
        )
        masks: dict[float, np.ndarray] = {}
        year_events = events.loc[events["onset_time"].dt.year.eq(year)]
        for row in selected.sort_values("budget_hours").itertuples(index=False):
            budget = float(row.budget_hours)
            controlled = apply_alert_policy(combined, "risk_6h", float(row.threshold), shape)
            current = controlled.loc[pd.to_datetime(controlled["issue_time"]).dt.year.eq(year)].copy()
            masks[budget] = current["budget_alarm"].to_numpy(dtype=np.int8)
            metrics, records = evaluate_event_alerts(
                current,
                year_events,
                "budget_alarm",
                budget,
                step_minutes=step_minutes,
            )
            metric_rows.append(
                {
                    **metrics,
                    "year": year,
                    "budget_hours": budget,
                    "threshold": float(row.threshold),
                    "shape_id": best_shape_id,
                }
            )
            records["year"] = year
            records["budget_hours"] = budget
            event_parts.append(records)
        assert_nested_alert_masks(masks, BUDGETS)
    pd.DataFrame(metric_rows).to_csv(output_root / "frozen_warning_metrics.csv", index=False)
    pd.concat(event_parts, ignore_index=True).to_csv(
        output_root / "frozen_event_records.csv.gz", index=False, compression="gzip"
    )
    (output_root / ".complete_nested_alert").touch()
    print(f"Nested alert policy complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_compat import read_artifact_csv
from .baseline_experiment import evaluate_warning_model
from .config import resolve_project_path
from .deep_warning import (
    monthly_false_alarm_distribution_audit,
    select_matched_false_alarm_thresholds,
)
from .graph_experiment import cluster_bootstrap_event_comparison, event_alarm_records
from .naming import artifact_value
from .preprocessing import prepare_output_root, write_json


GRAPH_GATE_SCORE_COLUMNS = [
    "stable_graph_original_6h",
    "stable_graph_ablation_6h",
    "graph_extra_lag_6h",
    "graph_shift_7d_6h",
    "graph_station_permutation_6h",
    "graph_future_positive_control_6h",
]
GRAPH_GATE_PAIRS = [
    ("stable_graph_original", "stable_graph_ablation"),
    ("stable_graph_original", "graph_extra_lag"),
    ("stable_graph_original", "graph_shift_7d"),
    ("stable_graph_original", "graph_station_permutation"),
    ("graph_future_positive_control", "stable_graph_original"),
]


def aligned_graph_contribution(
    frame: pd.DataFrame,
    time_delta: pd.Timedelta,
    station_mapping: dict[str, str] | None = None,
) -> np.ndarray:
    """Look up a graph contribution at an exact shifted time and optional placebo station."""

    lookup = frame.set_index(["station_code", "issue_time"])["stable_graph_contribution"]
    stations = frame["station_code"].astype(str)
    if station_mapping is not None:
        stations = stations.map(station_mapping)
    keys = pd.MultiIndex.from_arrays(
        [stations.to_numpy(), (frame["issue_time"] + time_delta).to_numpy()],
        names=["station_code", "issue_time"],
    )
    return lookup.reindex(keys).fillna(0.0).to_numpy(dtype=np.float64)


def six_hour_probability_from_eta(eta: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    calibrated = float(calibrator["log_rate_shift"]) + float(calibrator["slope"]) * np.asarray(
        eta, dtype=np.float64
    )
    rate = np.exp(np.minimum(calibrated, 15.0))
    return -np.expm1(-36.0 * rate)


def load_graph_gate_frame(stability_root: Path, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = sorted((stability_root / "predictions" / str(year)).glob("*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No stable-graph predictions found for {year}")
    columns = [
        "station_code",
        "issue_time",
        "onset_within_6h",
        "hard_negative_6h",
        "stable_graph_eta",
        "stable_graph_ablation_eta",
        "stable_graph_contribution",
    ]
    parts: list[pd.DataFrame] = []
    for path in paths:
        part = read_artifact_csv(
            path,
            columns=columns,
            encoding="utf-8-sig",
            low_memory=False,
        )
        part["issue_time"] = pd.to_datetime(part["issue_time"], errors="coerce", format="mixed")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True).sort_values(["station_code", "issue_time"])
    if frame["issue_time"].isna().any() or frame.duplicated(["station_code", "issue_time"]).any():
        raise ValueError("Stable-graph gate requires unique valid station-time rows")
    bundle = json.loads((stability_root / "model_bundle.json").read_text(encoding="utf-8"))
    calibrator = artifact_value(bundle["calibrators"], "stable_graph")
    stations = sorted(frame["station_code"].astype(str).unique())
    permutation = {station: stations[(index + 1) % len(stations)] for index, station in enumerate(stations)}
    local_eta = pd.to_numeric(frame["stable_graph_ablation_eta"], errors="coerce").to_numpy(np.float64)
    original_eta = pd.to_numeric(frame["stable_graph_eta"], errors="coerce").to_numpy(np.float64)
    contribution = pd.to_numeric(
        frame["stable_graph_contribution"], errors="coerce"
    ).fillna(0).to_numpy(np.float64)
    eta_values = {
        "stable_graph_original_6h": original_eta,
        "stable_graph_ablation_6h": local_eta,
        "graph_extra_lag_6h": local_eta
        + aligned_graph_contribution(frame, -pd.Timedelta(hours=6)),
        "graph_shift_7d_6h": local_eta
        + aligned_graph_contribution(frame, -pd.Timedelta(days=7)),
        "graph_station_permutation_6h": local_eta
        + aligned_graph_contribution(frame, pd.Timedelta(0), permutation),
        "graph_future_positive_control_6h": local_eta
        + aligned_graph_contribution(frame, pd.Timedelta(hours=6)),
    }
    for column, eta in eta_values.items():
        frame[column] = six_hour_probability_from_eta(eta, calibrator).astype(np.float32)
    diagnostics = {
        "rows": int(frame.shape[0]),
        "stations": len(stations),
        "graph_nonzero_fraction": float((contribution != 0).mean()),
        "graph_contribution_mean": float(contribution.mean()),
        "graph_contribution_standard_deviation": float(contribution.std()),
        "placebo_semantics": {
            "graph_extra_lag": "At time t, use the same station's frozen graph contribution at t-6h.",
            "graph_shift_7d": "At time t, use the same station's frozen graph contribution at t-7d.",
            "graph_station_permutation": "At time t, use another station's contribution under a fixed cyclic permutation.",
            "graph_future_positive_control": "At time t, use t+6h contribution; intentionally anti-causal and never deployable.",
        },
    }
    return frame.reset_index(drop=True), diagnostics


def evaluate_graph_gate(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
    warning: dict[str, Any],
    year: int,
) -> dict[str, pd.DataFrame]:
    frame = frame.copy()
    frame["station_month"] = (
        frame["station_code"].astype(str) + "|" + frame["issue_time"].dt.to_period("M").astype(str)
    )
    event_frame = events.copy()
    event_frame["onset_time"] = pd.to_datetime(event_frame["onset_time"], errors="coerce")
    event_frame["valid_target_event"] = pd.to_numeric(
        event_frame["valid_target_event"], errors="coerce"
    ).fillna(0).astype(int)
    event_frame = event_frame.loc[event_frame["onset_time"].dt.year.eq(year)]
    target_hours = float(config["warning_evaluation"]["matched_false_alarm_diagnostic"]["target_hours_per_station_month"])
    selection = select_matched_false_alarm_thresholds(
        frame,
        GRAPH_GATE_SCORE_COLUMNS,
        int(warning["horizon_hours"]),
        target_hours,
        int(config["step_minutes"]),
    )
    alarm_columns: dict[str, str] = {}
    for row in selection.itertuples(index=False):
        alarm_column = f"{row.score_column}__matched_alarm"
        score = pd.to_numeric(frame[row.score_column], errors="coerce").fillna(0).to_numpy(np.float64)
        frame[alarm_column] = (score >= float(row.threshold)).astype(np.int8)
        alarm_columns[str(row.model_name)] = alarm_column
    rows: list[dict[str, Any]] = []
    records: list[pd.DataFrame] = []
    for model_name, alarm_column in alarm_columns.items():
        row = evaluate_warning_model(
            frame,
            event_frame,
            alarm_column,
            0.5,
            int(warning["horizon_hours"]),
            int(config["step_minutes"]),
            target_hours,
            f"{year}::graph_gate_matched_false_alarm",
            int(warning["minimum_consecutive_alarm_bins"]),
            int(warning["alarm_merge_gap_minutes"]),
            int(warning["maximum_silence_before_event_minutes"]),
        )
        row.update(
            {
                "model_name": model_name,
                "lead_utility_hours": (
                    float(row["event_hit_rate"] * row["mean_effective_lead_hours"])
                    if np.isfinite(row["mean_effective_lead_hours"])
                    else float("nan")
                ),
            }
        )
        rows.append(row)
        model_records = event_alarm_records(
            frame,
            event_frame,
            alarm_column,
            0.5,
            int(warning["horizon_hours"]),
            warning,
        )
        model_records["model_name"] = model_name
        records.append(model_records)
    record_frame = pd.concat(records, ignore_index=True)
    comparisons: list[pd.DataFrame] = []
    for index, (model_a, model_b) in enumerate(GRAPH_GATE_PAIRS):
        comparison = cluster_bootstrap_event_comparison(
            record_frame,
            alarm_columns[model_a],
            alarm_columns[model_b],
            int(config["warning_evaluation"]["bootstrap_samples"]),
            int(config["warning_evaluation"]["bootstrap_seed"]) + 100 + index,
        )
        comparison["model_a_name"] = model_a
        comparison["model_b_name"] = model_b
        comparisons.append(comparison)
    audit = monthly_false_alarm_distribution_audit(
        frame,
        alarm_columns,
        int(warning["horizon_hours"]),
        int(config["step_minutes"]),
        target_hours,
    )
    return {
        "graph_gate_metrics": pd.DataFrame(rows),
        "graph_gate_event_records": record_frame,
        "graph_gate_comparisons": pd.concat(comparisons, ignore_index=True),
        "graph_gate_thresholds": selection,
        "graph_gate_monthly_distribution": audit,
    }


def graph_gate_decision(comparisons: pd.DataFrame) -> dict[str, Any]:
    primary = comparisons.loc[
        comparisons["model_a_name"].eq("stable_graph_original")
        & comparisons["model_b_name"].eq("stable_graph_ablation")
    ].set_index("metric")
    utility = primary.loc["lead_utility_difference_hours"]
    common_lead = primary.loc["common_hit_lead_difference_hours"]
    primary_pass = bool(
        float(utility["ci95_low"]) > 0
        and float(common_lead["estimate"]) >= 0.1
        and float(common_lead["ci95_low"]) > 0
    )
    placebo_rows = comparisons.loc[
        comparisons["model_a_name"].eq("stable_graph_original")
        & comparisons["model_b_name"].isin(
            ["graph_extra_lag", "graph_shift_7d", "graph_station_permutation"]
        )
        & comparisons["metric"].eq("lead_utility_difference_hours")
    ]
    placebo_pass = bool(
        placebo_rows.shape[0] == 3
        and placebo_rows["estimate"].gt(0).all()
        and placebo_rows["ci95_low"].gt(0).all()
    )
    passed = primary_pass and placebo_pass
    return {
        "gate_passed": passed,
        "primary_increment_passed": primary_pass,
        "placebo_separation_passed": placebo_pass,
        "decision": "proceed_to_deep_graph" if passed else "stop_deep_graph_extension",
        "criteria": {
            "primary": "Original graph lead-utility CI lower bound > 0 and common-hit lead >= 0.1h with CI lower bound > 0.",
            "placebos": "Original graph lead utility must exceed all three causal placebos with CI lower bounds > 0.",
        },
    }


def build_graph_gate_report(
    results: dict[str, pd.DataFrame],
    diagnostics: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    metrics = results["graph_gate_metrics"]
    comparisons = results["graph_gate_comparisons"]
    return "\n".join(
        [
            "# Deep RIG-Hazard稳定图进入门控",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 设计",
            "",
            "- 只使用2023开发年；所有候选在总体10小时/站点月误报下比较。",
            "- 原稳定图与同一模型的图贡献置零版本比较。",
            "- 额外滞后6小时、错位7天和站点置换均保持局地与层次项不变，只破坏图贡献的正确时空对齐。",
            "- 未来6小时正对照故意读取未来信息，仅检查方向性，不属于可用模型。",
            f"- 图贡献非零比例：{diagnostics['graph_nonzero_fraction']:.4f}。",
            "",
            "## 同误报结果",
            "",
            metrics[
                [
                    "model_name",
                    "false_alarm_hours_per_station_month",
                    "event_hit_rate",
                    "mean_effective_lead_hours",
                    "lead_utility_hours",
                    "hard_negative_far",
                ]
            ].to_markdown(index=False),
            "",
            "## 配对站点聚类Bootstrap",
            "",
            comparisons[
                ["model_a_name", "model_b_name", "metric", "estimate", "ci95_low", "ci95_high"]
            ].to_markdown(index=False),
            "",
            "## 门控决定",
            "",
            f"- 主要增量门槛：{'通过' if decision['primary_increment_passed'] else '未通过'}。",
            f"- 安慰剂分离门槛：{'通过' if decision['placebo_separation_passed'] else '未通过'}。",
            f"- 总门槛：{'通过' if decision['gate_passed'] else '未通过'}。",
            f"- 决策：`{decision['decision']}`。",
            "",
            "门控未通过时，不训练深度图分支；这不是工程未完成，而是预先规定的防过拟合停止规则。",
        ]
    ) + "\n"


def run_deep_graph_gate(
    config: dict[str, Any],
    config_path: Path,
    overwrite: bool = False,
) -> Path:
    output_root = resolve_project_path(config["graph_gate_output_root"])
    prepare_output_root(output_root, overwrite)
    stability_root = resolve_project_path(config["stability_root"])
    year = int(config["warning_evaluation"]["year"])
    stability_config = json.loads(
        (stability_root / "resolved_experiment_config.json").read_text(encoding="utf-8")
    )
    frame, diagnostics = load_graph_gate_frame(stability_root, year)
    events = pd.read_csv(
        resolve_project_path(config["preprocessed_root"]) / "events_recurrent.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    results = evaluate_graph_gate(frame, events, config, stability_config["warning"], year)
    for name, result in results.items():
        result.to_csv(output_root / f"{name}.csv", index=False, encoding="utf-8-sig")
    decision = graph_gate_decision(results["graph_gate_comparisons"])
    write_json(output_root / "graph_gate_diagnostics.json", diagnostics)
    write_json(output_root / "graph_gate_decision.json", decision)
    write_json(
        output_root / "resolved_config.json",
        {**config, "config_path": str(config_path), "evaluation_year": year},
    )
    (output_root / "graph_gate_report.md").write_text(
        build_graph_gate_report(results, diagnostics, decision), encoding="utf-8"
    )
    print(f"Deep stable-graph gate complete: {output_root}", flush=True)
    return output_root

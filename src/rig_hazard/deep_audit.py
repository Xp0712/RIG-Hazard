from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .artifact_compat import read_artifact_csv
import scipy
import sklearn

from .baseline_experiment import read_timeline, timeline_files
from .config import PROJECT_ROOT, resolve_project_path
from .preprocessing import prepare_output_root, write_json


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _existing_files(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.exists() and path.is_file()})


def frozen_artifact_paths(config: dict[str, Any]) -> list[Path]:
    preprocessed_root = resolve_project_path(config["preprocessed_root"])
    baseline_root = resolve_project_path(config["baseline_root"])
    graph_root = resolve_project_path(config["graph_root"])
    stability_root = resolve_project_path(config["stability_root"])

    paths: list[Path] = []
    paths.extend((PROJECT_ROOT / "rig_hazard").glob("*.py"))
    paths.extend((PROJECT_ROOT / "configs").glob("rig_hazard*.json"))
    paths.extend(PROJECT_ROOT.glob("requirements*.txt"))
    for name in [
        "resolved_config.json",
        "feature_contract.json",
        "normalization_development.json",
        "events_recurrent.csv",
        "station_catalog.csv",
        "split_summary.csv",
        "validation_summary.json",
    ]:
        paths.append(preprocessed_root / name)
    for root in [baseline_root, graph_root, stability_root]:
        for pattern in [
            "*.json",
            "*.csv",
            "*.md",
            "predictions/**/*.csv.gz",
            "budget_predictions/**/*.csv.gz",
        ]:
            paths.extend(root.glob(pattern))
    return _existing_files(paths)


def build_artifact_manifest(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        stat = path.stat()
        rows.append(
            {
                "path": _project_relative(path),
                "bytes": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256_file(path),
            }
        )
        if index % 25 == 0 or index == len(paths):
            print(f"Hashed frozen artifacts {index}/{len(paths)}", flush=True)
    return pd.DataFrame(rows)


def build_sample_flow(preprocessed_root: Path, stability_root: Path) -> pd.DataFrame:
    validation = json.loads((preprocessed_root / "validation_summary.json").read_text(encoding="utf-8"))
    metrics = validation["metrics"]
    station_summary = pd.read_csv(
        preprocessed_root / "station_preprocessing_summary.csv", encoding="utf-8-sig", low_memory=False
    )
    events = pd.read_csv(preprocessed_root / "events_recurrent.csv", encoding="utf-8-sig", low_memory=False)
    rows: list[dict[str, Any]] = [
        {
            "stage": "raw_records",
            "statistical_unit": "minute_record",
            "scope": "all",
            "count": int(pd.to_numeric(station_summary["raw_rows"], errors="coerce").sum()),
            "filter": "source CSV rows after file discovery",
        },
        {
            "stage": "timeline",
            "statistical_unit": "station_10min",
            "scope": "all",
            "count": int(metrics["timeline_rows"]),
            "filter": "complete 10-minute station timelines",
        },
        {
            "stage": "all_icing_processes",
            "statistical_unit": "event",
            "scope": "all",
            "count": int(metrics["all_events"]),
            "filter": "positive-ice processes after gap merging",
        },
        {
            "stage": "cold_candidate_events",
            "statistical_unit": "event",
            "scope": "all",
            "count": int(metrics["cold_candidate_events"]),
            "filter": "cold-plausible target candidates",
        },
        {
            "stage": "valid_recurrent_events",
            "statistical_unit": "event",
            "scope": "all",
            "count": int(metrics["valid_target_events"]),
            "filter": "cold candidate and cooldown eligible",
        },
        {
            "stage": "eligible_hazard_events",
            "statistical_unit": "event",
            "scope": "all",
            "count": int(metrics["eligible_events"]),
            "filter": "onset maps to a valid no-leakage issue time",
        },
        {
            "stage": "risk_set_rows",
            "statistical_unit": "station_10min",
            "scope": "all",
            "count": int(metrics["risk_rows"]),
            "filter": "risk_set == 1",
        },
    ]

    columns = [
        "split_development",
        "risk_set",
        "hazard_label",
        "onset_within_1h",
        "onset_within_3h",
        "onset_within_6h",
    ]
    split_counts: dict[str, dict[str, int]] = {}
    all_years = sorted(int(value) for value in metrics["files_by_year"])
    paths = timeline_files(preprocessed_root, all_years)
    for index, path in enumerate(paths, start=1):
        frame = read_timeline(path, columns)
        frame = frame.loc[frame["risk_set"].eq(1)]
        for split_name, split_frame in frame.groupby("split_development", sort=False):
            target = split_counts.setdefault(
                str(split_name), {"risk_rows": 0, "hazard_label": 0, "onset_1h": 0, "onset_3h": 0, "onset_6h": 0}
            )
            target["risk_rows"] += int(split_frame.shape[0])
            target["hazard_label"] += int(pd.to_numeric(split_frame["hazard_label"], errors="coerce").fillna(0).sum())
            for horizon in [1, 3, 6]:
                target[f"onset_{horizon}h"] += int(
                    pd.to_numeric(split_frame[f"onset_within_{horizon}h"], errors="coerce").fillna(0).sum()
                )
        if index % 20 == 0 or index == len(paths):
            print(f"Audited sample flow from {index}/{len(paths)} timelines", flush=True)

    for split_name, values in sorted(split_counts.items()):
        for label in ["risk_rows", "hazard_label", "onset_1h", "onset_3h", "onset_6h"]:
            rows.append(
                {
                    "stage": label,
                    "statistical_unit": "station_10min",
                    "scope": f"development::{split_name}",
                    "count": values[label],
                    "filter": "risk_set == 1 and split_development matches",
                }
            )

    records_path = stability_root / "causal_budget_event_records.csv"
    if records_path.exists():
        records = read_artifact_csv(
            records_path,
            identifier_columns=("model_name",),
            encoding="utf-8-sig",
            low_memory=False,
        )
        records = records.loc[records["model_name"].eq("local_weather_hazard")]
        for year, yearly in records.groupby("year"):
            rows.append(
                {
                    "stage": "evaluable_warning_events",
                    "statistical_unit": "event",
                    "scope": str(int(year)),
                    "count": int(pd.to_numeric(yearly["evaluable"], errors="coerce").fillna(0).sum()),
                    "filter": "target event has a complete warning evaluation window",
                }
            )
    return pd.DataFrame(rows)


def diagnose_stable_graph(config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    graph_root = resolve_project_path(config["graph_root"])
    stability_root = resolve_project_path(config["stability_root"])
    graph_bundle = json.loads((graph_root / "model_bundle.json").read_text(encoding="utf-8"))
    stability_bundle = json.loads((stability_root / "model_bundle.json").read_text(encoding="utf-8"))
    original = graph_bundle["selected_graph_model"]
    stable = stability_bundle["stable_model"]
    if original["feature_names"] != stable["feature_names"]:
        raise RuntimeError(
            "Unfiltered and stable graph feature schemas differ; coefficient "
            "diagnostics are not aligned"
        )

    training_summary = json.loads((stability_root / "training_sample_summary.json").read_text(encoding="utf-8"))
    base_count = int(training_summary["base_feature_count"])
    coefficient_threshold = float(config["stable_graph_diagnostics"].get("coefficient_threshold", 1e-4))
    original_coef = np.asarray(original["coefficients"], dtype=np.float64)
    stable_coef = np.asarray(stable["coefficients"], dtype=np.float64)
    graph_original = original_coef[base_count:]
    graph_stable = stable_coef[base_count:]
    original_active = graph_original > coefficient_threshold
    stable_active = graph_stable > coefficient_threshold

    edges = pd.read_csv(stability_root / "graph_edge_stability.csv", encoding="utf-8-sig", low_memory=False)
    removed = edges.loc[
        edges["original_coefficient"].gt(coefficient_threshold) & edges["stable"].eq(0)
    ].copy()

    stride = max(int(config["stable_graph_diagnostics"].get("prediction_stride", 10)), 1)
    eta_parts: list[np.ndarray] = []
    stable_parts: list[np.ndarray] = []
    contribution_parts: list[np.ndarray] = []
    years = {int(value) for value in config["stable_graph_diagnostics"].get("years", [2023, 2024])}
    prediction_paths = sorted((stability_root / "predictions").glob("*/*.csv.gz"))
    selected_paths = [path for path in prediction_paths if int(path.parent.name) in years]
    for index, path in enumerate(selected_paths, start=1):
        frame = read_artifact_csv(
            path,
            columns=[
                "unfiltered_graph_eta",
                "stable_graph_eta",
                "stable_graph_contribution",
            ],
            encoding="utf-8-sig",
            low_memory=False,
        ).iloc[::stride]
        eta_parts.append(frame["unfiltered_graph_eta"].to_numpy(dtype=np.float64))
        stable_parts.append(frame["stable_graph_eta"].to_numpy(dtype=np.float64))
        contribution_parts.append(frame["stable_graph_contribution"].to_numpy(dtype=np.float64))
        if index % 20 == 0 or index == len(selected_paths):
            print(
                "Compared unfiltered/stable graph predictions from "
                f"{index}/{len(selected_paths)} files",
                flush=True,
            )
    unfiltered_graph_eta = np.concatenate(eta_parts)
    stable_graph_eta = np.concatenate(stable_parts)
    contribution = np.concatenate(contribution_parts)
    difference = stable_graph_eta - unfiltered_graph_eta
    pearson = float(np.corrcoef(unfiltered_graph_eta, stable_graph_eta)[0, 1])
    spearman = float(
        pd.Series(unfiltered_graph_eta).corr(
            pd.Series(stable_graph_eta), method="spearman"
        )
    )

    warning_metrics = read_artifact_csv(
        stability_root / "causal_budget_warning_metrics.csv",
        identifier_columns=("model_name",),
        encoding="utf-8-sig",
        low_memory=False,
    )
    exact_warning_matches: dict[str, bool] = {}
    event_metric_matches: dict[str, bool] = {}
    warning_metric_differences: dict[str, dict[str, float]] = {}
    for year in sorted(years):
        yearly = warning_metrics.loc[warning_metrics["year"].eq(year)].set_index("model_name")
        if {"unfiltered_graph", "stable_graph"}.issubset(yearly.index):
            columns = [
                "hit_events",
                "event_hit_rate",
                "mean_effective_lead_hours",
                "median_effective_lead_hours",
                "false_alarm_hours_per_station_month",
                "alarm_hours_per_station_month",
            ]
            differences = (
                yearly.loc["stable_graph", columns].astype(float)
                - yearly.loc["unfiltered_graph", columns].astype(float)
            )
            warning_metric_differences[str(year)] = {
                column: float(differences[column]) for column in columns
            }
            exact_warning_matches[str(year)] = bool(
                np.allclose(
                    yearly.loc["unfiltered_graph", columns].to_numpy(dtype=np.float64),
                    yearly.loc["stable_graph", columns].to_numpy(dtype=np.float64),
                    equal_nan=True,
                )
            )
            event_columns = [
                "hit_events",
                "event_hit_rate",
                "mean_effective_lead_hours",
                "median_effective_lead_hours",
            ]
            event_metric_matches[str(year)] = bool(
                np.allclose(
                    yearly.loc["unfiltered_graph", event_columns].to_numpy(dtype=np.float64),
                    yearly.loc["stable_graph", event_columns].to_numpy(dtype=np.float64),
                    equal_nan=True,
                )
            )

    diagnostics = {
        "unfiltered_graph_iterations": int(original.get("iterations", 0)),
        "stable_graph_refit_iterations": int(stable.get("iterations", 0)),
        "unfiltered_graph_converged": bool(original.get("converged", False)),
        "stable_graph_converged": bool(stable.get("converged", False)),
        "unfiltered_graph_objective": original.get("objective"),
        "stable_graph_objective": stable.get("objective"),
        "unfiltered_graph_message": original.get("message", ""),
        "stable_graph_message": stable.get("message", ""),
        "base_feature_count": base_count,
        "graph_feature_count": int(graph_original.size),
        "unfiltered_active_graph_terms": int(original_active.sum()),
        "stable_active_graph_terms": int(stable_active.sum()),
        "removed_original_graph_terms": int((original_active & ~stable_active).sum()),
        "base_coefficient_max_abs_difference": float(np.max(np.abs(stable_coef[:base_count] - original_coef[:base_count]))),
        "retained_graph_coefficient_rmse": float(
            np.sqrt(np.mean(np.square(graph_stable[stable_active] - graph_original[stable_active])))
        ),
        "prediction_rows_sampled": int(unfiltered_graph_eta.size),
        "prediction_stride": stride,
        "eta_pearson": pearson,
        "eta_spearman": spearman,
        "eta_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "eta_max_abs_difference": float(np.max(np.abs(difference))),
        "stable_graph_contribution_nonzero_fraction": float(np.mean(np.abs(contribution) > 1e-12)),
        "warning_metrics_exactly_match_unfiltered_graph": exact_warning_matches,
        "event_metrics_exactly_match_unfiltered_graph": event_metric_matches,
        "stable_minus_unfiltered_warning_metrics": warning_metric_differences,
        "interpretation": (
            "The stable graph starts from the converged unfiltered graph, fixes "
            "non-stable graph terms to zero, and refits on the same 2022 design. "
            "One optimizer iteration is plausible when the projected solution is "
            "already near the constrained optimum. Event hits and lead-time "
            "summaries match, while alarm-duration metrics differ slightly; the "
            "two graph models must not be described as fully identical."
        ),
    }
    return diagnostics, removed


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 100) -> str:
    if frame.empty:
        return "无。"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    lines = ["| " + " | ".join(subset.columns) + " |", "| " + " | ".join(["---"] * subset.shape[1]) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_audit_report(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    sample_flow: pd.DataFrame,
    diagnostics: dict[str, Any],
    removed_edges: pd.DataFrame,
) -> str:
    lines = [
        "# Deep RIG-Hazard 模型合同与样本口径审计",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 冻结范围",
        "",
        f"- 已冻结并计算SHA-256的文件：{manifest.shape[0]}个，总计{int(manifest['bytes'].sum())}字节。",
        "- 范围包括事件与特征契约、归一化参数、local weather hazard/unfiltered/stable-graph模型包、正式预测、预算预测、指标、配置和当前源代码。",
        "- 2024结果已经被查看，只能作为已开发的描述性跨年评价；后续深度模型不得据此选择结构或超参数。",
        "",
        "## 运行环境",
        "",
        "```json",
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## 数量流转",
        "",
        markdown_table(sample_flow, ["stage", "statistical_unit", "scope", "count", "filter"], limit=200),
        "",
        "## stable graph受限重拟合诊断",
        "",
        "```json",
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        "```",
        "",
        "### 被稳定性筛选移除的原unfiltered graph图项",
        "",
        markdown_table(
            removed_edges,
            [
                "source_station_code",
                "target_station_code",
                "lag_minutes",
                "selection_frequency",
                "original_coefficient",
                "coefficient_median",
            ],
            limit=100,
        ),
        "",
        "## 模型合同审计结论",
        "",
        "1. 当前统计基线及其正式预测已经用文件哈希冻结，后续深度模型必须使用相同事件、风险集与切分口径。",
        "2. stable graph一次迭代应结合投影热启动、系数差异和预测排序共同解释，不能仅凭迭代次数认定训练失败。",
        "3. 深度阶段首先比较非图GRU/TCN hazard与local weather hazard；图分支只有在非图编码器通过后才进入实验。",
        "4. 新模型选择只能使用2022训练和2023开发数据；真正最终结论需要新年份或外部区域。",
    ]
    return "\n".join(lines) + "\n"


def run_deep_audit(config: dict[str, Any], config_path: Path, overwrite: bool = False) -> Path:
    output_root = resolve_project_path(config["audit_output_root"])
    prepare_output_root(output_root, overwrite)
    write_json(output_root / "resolved_config.json", config)
    paths = frozen_artifact_paths(config)
    manifest = build_artifact_manifest(paths)
    manifest.to_csv(output_root / "artifact_hashes.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_root / "freeze_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "artifact_count": int(manifest.shape[0]),
            "total_bytes": int(manifest["bytes"].sum()),
            "manifest_sha256": hashlib.sha256(manifest.to_csv(index=False).encode("utf-8")).hexdigest(),
        },
    )
    preprocessed_root = resolve_project_path(config["preprocessed_root"])
    stability_root = resolve_project_path(config["stability_root"])
    sample_flow = build_sample_flow(preprocessed_root, stability_root)
    sample_flow.to_csv(output_root / "sample_flow.csv", index=False, encoding="utf-8-sig")
    diagnostics, removed_edges = diagnose_stable_graph(config)
    write_json(output_root / "stable_graph_refit_diagnostics.json", diagnostics)
    removed_edges.to_csv(
        output_root / "stable_graph_removed_terms.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = build_audit_report(config, manifest, sample_flow, diagnostics, removed_edges)
    (output_root / "experiment_audit_report.md").write_text(report, encoding="utf-8")
    print(f"Deep RIG-Hazard model-contract audit complete: {output_root}", flush=True)
    return output_root

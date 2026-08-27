from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.alert_governance import (
    STRICT_EVENT_CONTRACT_VERSION,
    station_month_budget_distribution,
    strict_event_alert_evaluation,
)
from rig_hazard.baseline_experiment import observed_warning_rows
from rig_hazard.config import resolve_project_path
from rig_hazard.evaluation_provenance import file_bundle_manifest, sha256_file
from rig_hazard.event_eligibility import (
    add_event_interval_metadata,
    annotate_timeline_exclusion_reasons,
    eligibility_bias_summary,
    eligibility_funnel,
    eligibility_station_season_distribution,
)
from rig_hazard.nested_alert import AlertShape, apply_alert_policy, evaluate_event_alerts
from rig_hazard.utility_warning import _ensemble_frame, _events


def _check(
    rows: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    detail: str,
    severity: str = "critical",
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "status": "PASS" if passed else ("FAIL" if severity == "critical" else "WARN"),
            "severity": severity,
            "detail": detail,
        }
    )


def _expected_future_label(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    horizon_hours: int = 6,
) -> np.ndarray:
    result = np.zeros(frame.shape[0], dtype=np.int8)
    horizon_ns = int(horizon_hours * 3600 * 1_000_000_000)
    for station, positions in frame.groupby("station_code", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        issue = (
            pd.to_datetime(frame.loc[positions, "issue_time"], errors="coerce")
            .astype("datetime64[ns]")
            .astype("int64")
            .to_numpy()
        )
        onsets = (
            pd.to_datetime(
                events.loc[events["station_code"].astype(str).eq(str(station)), "onset_time"],
                errors="coerce",
            )
            .dropna()
            .sort_values()
            .astype("datetime64[ns]")
            .astype("int64")
            .to_numpy()
        )
        if onsets.size == 0:
            continue
        next_index = np.searchsorted(onsets, issue, side="right")
        valid = next_index < onsets.size
        next_onset = np.zeros(issue.size, dtype=np.int64)
        next_onset[valid] = onsets[next_index[valid]]
        result[positions] = (
            valid & (next_onset > issue) & (next_onset <= issue + horizon_ns)
        ).astype(np.int8)
    return result


def _prefix_invariant(
    frame: pd.DataFrame,
    threshold: float,
    shape: AlertShape,
) -> tuple[bool, int]:
    checked = 0
    for _, station in frame.groupby("station_code", sort=False):
        station = station.sort_values("issue_time").reset_index(drop=True)
        if station.shape[0] < 500:
            continue
        cut = station.shape[0] // 2
        full = apply_alert_policy(station, "risk_6h", threshold, shape)
        prefix = apply_alert_policy(station.iloc[:cut], "risk_6h", threshold, shape)
        if not np.array_equal(
            full["budget_alarm"].to_numpy()[:cut],
            prefix["budget_alarm"].to_numpy(),
        ):
            return False, checked + 1
        perturbed = station.copy()
        perturbed["risk_6h"] = pd.to_numeric(
            perturbed["risk_6h"], errors="coerce"
        ).astype(np.float64)
        perturbed.loc[cut:, "risk_6h"] = np.nanmax(
            pd.to_numeric(perturbed["risk_6h"], errors="coerce")
        ) * 100.0 + 1.0
        changed = apply_alert_policy(perturbed, "risk_6h", threshold, shape)
        if not np.array_equal(
            full["budget_alarm"].to_numpy()[:cut],
            changed["budget_alarm"].to_numpy()[:cut],
        ):
            return False, checked + 1
        checked += 1
        if checked >= 5:
            break
    return checked > 0, checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the high-hit nested alert policy.")
    parser.add_argument("--config", default="configs/rig_hazard_alert_governance.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    section = config["nested_alert_audit"]
    output_root = resolve_project_path(
        args.output_root or section["output_root"]
    )
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_root = resolve_project_path(section["source_root"])
    policy = json.loads((source_root / "policy.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(source_root / "selected_nested_thresholds.csv")
    model = str(policy["model"])
    seeds = [int(value) for value in policy["seeds"]]
    protocol_root = resolve_project_path(section["protocol_root"])
    cache_root = resolve_project_path(config["cache_root"])
    seasonal_path = resolve_project_path(config["seasonal_event_mapping"])
    events = _events(config, seasonal_path)
    event_table_sha256 = sha256_file(seasonal_path)
    selection = _ensemble_frame(cache_root, protocol_root / "oof_predictions", model, seeds)
    locked = {
        year: _ensemble_frame(
            cache_root, protocol_root / "locked_predictions" / str(year), model, seeds
        )
        for year in (2023, 2024)
    }
    shape = AlertShape(**{key: policy["shape"][key] for key in AlertShape.__dataclass_fields__})
    step_minutes = int(config["step_minutes"])
    cache_config = json.loads((cache_root / "resolved_config.json").read_text(encoding="utf-8"))
    timeline_root = resolve_project_path(cache_config["preprocessed_root"]) / "timelines"
    prediction_manifests: dict[str, dict[str, Any]] = {}
    prediction_manifests["2022_oof"] = file_bundle_manifest(
        [protocol_root / "oof_predictions" / f"{model}_seed_{seed}.npz" for seed in seeds],
        PROJECT_ROOT,
    )
    for year in (2023, 2024):
        prediction_manifests[f"{year}_frozen"] = file_bundle_manifest(
            [
                protocol_root
                / "locked_predictions"
                / str(year)
                / f"{model}_seed_{seed}.npz"
                for seed in seeds
            ],
            PROJECT_ROOT,
        )
    selected_policy_file_sha256 = sha256_file(source_root / "selected_nested_thresholds.csv")
    checks: list[dict[str, Any]] = []
    selection_years = sorted(pd.to_datetime(selection["issue_time"]).dt.year.unique().tolist())
    _check(
        checks,
        "selection_rows_are_2022_only",
        selection_years == [2022] and int(policy["selection_year"]) == 2022,
        f"OOF years={selection_years}; policy selection_year={policy['selection_year']}",
    )
    locked_years = {
        year: sorted(pd.to_datetime(frame["issue_time"]).dt.year.unique().tolist())
        for year, frame in locked.items()
    }
    _check(
        checks,
        "locked_predictions_are_year_separated",
        all(values == [year] for year, values in locked_years.items()),
        json.dumps(locked_years, ensure_ascii=False),
    )
    _check(
        checks,
        "no_monthwise_top_k_contract",
        bool(policy.get("causal")) and set(policy.get("thresholds", {})) == {"2", "5", "10", "20"},
        "Policy contains four frozen absolute thresholds; no monthwise rank or Top-K parameter.",
    )
    _check(
        checks,
        "unique_selection_keys",
        not selection.duplicated(["station_code", "issue_time"]).any(),
        "rows="
        f"{selection.shape[0]}, duplicate_station_times="
        f"{int(selection.duplicated(['station_code', 'issue_time']).sum())}",
    )
    selection_keys = pd.MultiIndex.from_frame(selection[["station_code", "issue_time"]])
    for year, frame in locked.items():
        locked_keys = pd.MultiIndex.from_frame(frame[["station_code", "issue_time"]])
        overlap = selection_keys.intersection(locked_keys).size
        duplicate = int(frame.duplicated(["station_code", "issue_time"]).sum())
        _check(
            checks,
            f"unique_and_disjoint_locked_keys_{year}",
            duplicate == 0 and overlap == 0,
            f"duplicate_station_times={duplicate}; overlap_with_2022={overlap}",
        )
    label_expected = _expected_future_label(selection, events.loc[events["onset_time"].dt.year.eq(2022)])
    observed = pd.to_numeric(selection["observed_6h"], errors="coerce").fillna(0).eq(1).to_numpy()
    label_actual = pd.to_numeric(selection["onset_within_6h"], errors="coerce").fillna(0).astype(int).to_numpy()
    mismatch = int(((label_expected != label_actual) & observed).sum())
    _check(
        checks,
        "event_table_matches_horizon_labels",
        mismatch == 0,
        f"observed_rows={int(observed.sum())}, mismatched_rows={mismatch}",
        severity="diagnostic",
    )

    metric_rows: list[dict[str, Any]] = []
    monthly_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    eligibility_parts: list[pd.DataFrame] = []
    for year in (2023, 2024):
        history = selection if year == 2023 else locked[2023]
        combined = pd.concat([history, locked[year]], ignore_index=True).sort_values(
            ["station_code", "issue_time"]
        )
        year_events = events.loc[events["onset_time"].dt.year.eq(year)].copy()
        eligibility_frame = locked[year].copy()
        eligibility_frame["_eligibility_alarm"] = 0
        _, eligibility_records, _ = strict_event_alert_evaluation(
            eligibility_frame,
            year_events,
            alarm_column="_eligibility_alarm",
            step_minutes=step_minutes,
            horizon_hours=6,
        )
        eligibility_records["year"] = year
        eligibility_records["prediction_bundle_sha256"] = prediction_manifests[
            f"{year}_frozen"
        ]["bundle_sha256"]
        eligibility_parts.append(eligibility_records)
        for row in selected.sort_values("budget_hours").itertuples(index=False):
            budget = float(row.budget_hours)
            threshold = float(row.threshold)
            all_controlled = apply_alert_policy(combined, "risk_6h", threshold, shape)
            controlled = all_controlled.loc[
                pd.to_datetime(all_controlled["issue_time"]).dt.year.eq(year)
            ].copy()
            legacy_controlled = observed_warning_rows(controlled, "observed_6h")
            legacy, legacy_records = evaluate_event_alerts(
                legacy_controlled,
                year_events,
                "budget_alarm",
                budget,
                step_minutes=step_minutes,
            )
            strict, strict_records, strict_frame = strict_event_alert_evaluation(
                controlled,
                year_events,
                alarm_column="budget_alarm",
                step_minutes=step_minutes,
                horizon_hours=6,
            )
            legacy_monthly, legacy_distribution = station_month_budget_distribution(
                legacy_controlled,
                "budget_alarm",
                budget,
                step_minutes,
            )
            strict_monthly, strict_distribution = station_month_budget_distribution(
                strict_frame,
                "budget_alarm",
                budget,
                step_minutes,
                strict_false_column="_strict_false_alarm",
                unsettled_column="_strict_unsettled_alarm",
                reserved_column="_strict_reserved_alarm",
            )
            metric_rows.append(
                {
                    "year": year,
                    "budget_hours": budget,
                    "threshold": threshold,
                    "evaluation_contract_version": STRICT_EVENT_CONTRACT_VERSION,
                    "event_table_sha256": event_table_sha256,
                    "prediction_bundle_sha256": prediction_manifests[f"{year}_frozen"][
                        "bundle_sha256"
                    ],
                    "selection_policy_file_sha256": selected_policy_file_sha256,
                    **{f"legacy_{key}": value for key, value in legacy.items()},
                    **{f"strict_{key}": value for key, value in strict.items()},
                    **{f"legacy_monthly_{key}": value for key, value in legacy_distribution.items()},
                    **{f"strict_monthly_{key}": value for key, value in strict_distribution.items()},
                }
            )
            for name, monthly in (("legacy", legacy_monthly), ("strict", strict_monthly)):
                monthly["year"] = year
                monthly["budget_hours"] = budget
                monthly["false_alarm_definition"] = name
                monthly["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
                monthly["event_table_sha256"] = event_table_sha256
                monthly["prediction_bundle_sha256"] = prediction_manifests[
                    f"{year}_frozen"
                ]["bundle_sha256"]
                monthly["selection_policy_file_sha256"] = selected_policy_file_sha256
                monthly_parts.append(monthly)
            strict_records["year"] = year
            strict_records["budget_hours"] = budget
            strict_records["evaluation"] = "strict"
            legacy_records["year"] = year
            legacy_records["budget_hours"] = budget
            legacy_records["evaluation"] = "legacy"
            for records in (strict_records, legacy_records):
                records["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
                records["event_table_sha256"] = event_table_sha256
                records["prediction_bundle_sha256"] = prediction_manifests[
                    f"{year}_frozen"
                ]["bundle_sha256"]
                records["selection_policy_file_sha256"] = selected_policy_file_sha256
            event_parts.extend([strict_records, legacy_records])
            _check(
                checks,
                f"unique_event_and_segment_matches_{year}_{budget:g}h",
                int(strict["duplicate_event_matches"]) == 0
                and int(strict["duplicate_segment_matches"]) == 0,
                f"duplicate_events={strict['duplicate_event_matches']}; duplicate_segments={strict['duplicate_segment_matches']}",
            )
        first_threshold = float(selected.sort_values("budget_hours").iloc[0]["threshold"])
        invariant, stations_checked = _prefix_invariant(combined, first_threshold, shape)
        _check(
            checks,
            f"prefix_invariance_{year}",
            invariant,
            f"stations_checked={stations_checked}; future scores were perturbed after each cutoff",
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_root / "strict_vs_legacy_metrics.csv", index=False)
    provenance_columns = [
        "evaluation_contract_version",
        "event_table_sha256",
        "prediction_bundle_sha256",
        "selection_policy_file_sha256",
    ]
    legacy_columns = ["year", "budget_hours", "threshold", *provenance_columns] + [
        column for column in metrics if column.startswith("legacy_")
    ]
    strict_columns = [
        "year",
        "budget_hours",
        "threshold",
        *provenance_columns,
    ] + [column for column in metrics if column.startswith("strict_")]
    metrics[legacy_columns].to_csv(output_root / "legacy_contract_metrics.csv", index=False)
    metrics[strict_columns].to_csv(output_root / "strict_unified_metrics.csv", index=False)
    monthly_all = pd.concat(monthly_parts, ignore_index=True)
    monthly_all.to_csv(
        output_root / "station_month_budget_distribution.csv.gz",
        index=False,
        compression="gzip",
    )
    monthly_all.loc[monthly_all["false_alarm_definition"].eq("legacy")].to_csv(
        output_root / "legacy_station_month_budget_distribution.csv.gz",
        index=False,
        compression="gzip",
    )
    monthly_all.loc[monthly_all["false_alarm_definition"].eq("strict")].to_csv(
        output_root / "strict_station_month_budget_distribution.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.concat(event_parts, ignore_index=True).to_csv(
        output_root / "event_match_audit.csv.gz", index=False, compression="gzip"
    )
    eligibility = pd.concat(eligibility_parts, ignore_index=True)
    eligibility = add_event_interval_metadata(eligibility, events)
    eligibility = annotate_timeline_exclusion_reasons(
        eligibility, timeline_root, step_minutes
    )
    eligibility["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    eligibility["event_table_sha256"] = event_table_sha256
    eligibility["selection_policy_file_sha256"] = selected_policy_file_sha256
    eligibility.to_csv(output_root / "event_eligibility_audit.csv", index=False)
    funnel = eligibility_funnel(eligibility)
    funnel["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    funnel["event_table_sha256"] = event_table_sha256
    funnel["selection_policy_file_sha256"] = selected_policy_file_sha256
    funnel.to_csv(output_root / "event_eligibility_funnel.csv", index=False)
    bias = eligibility_bias_summary(eligibility)
    bias["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    bias["event_table_sha256"] = event_table_sha256
    bias["selection_policy_file_sha256"] = selected_policy_file_sha256
    bias.to_csv(output_root / "event_eligibility_bias_summary.csv", index=False)
    station_season = eligibility_station_season_distribution(eligibility)
    station_season["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    station_season["event_table_sha256"] = event_table_sha256
    station_season["selection_policy_file_sha256"] = selected_policy_file_sha256
    station_season.to_csv(
        output_root / "event_eligibility_station_season.csv", index=False
    )
    (output_root / "prediction_bundle_manifest.json").write_text(
        json.dumps(prediction_manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checks_frame = pd.DataFrame(checks)
    checks_frame["evaluation_contract_version"] = STRICT_EVENT_CONTRACT_VERSION
    checks_frame["event_table_sha256"] = event_table_sha256
    checks_frame["selection_policy_file_sha256"] = selected_policy_file_sha256
    checks_frame.to_csv(output_root / "audit_checks.csv", index=False)
    critical_failures = checks_frame.loc[
        checks_frame["severity"].eq("critical") & checks_frame["status"].eq("FAIL")
    ]
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_policy": str(source_root),
        "model": model,
        "selection_year": 2022,
        "evaluation_years": [2023, 2024],
        "evaluation_contract_version": STRICT_EVENT_CONTRACT_VERSION,
        "event_table_sha256": event_table_sha256,
        "selection_policy_file_sha256": selected_policy_file_sha256,
        "critical_failures": critical_failures["check_id"].tolist(),
        "audit_passed": bool(critical_failures.empty),
        "interpretation_rule": (
            "High hit rates are accepted only if strict matching remains high and all critical "
            "split, causality, duplicate-match, and frozen-threshold checks pass."
        ),
    }
    (output_root / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not critical_failures.empty:
        raise RuntimeError(
            "Nested alert audit failed critical checks: "
            + ", ".join(critical_failures["check_id"].tolist())
        )
    (output_root / ".complete_nested_alert_audit").touch()
    print(f"Nested alert audit complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()

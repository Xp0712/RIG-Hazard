from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use model=protocol_root syntax")
    model, path = (part.strip() for part in value.split("=", 1))
    if not model or not path:
        raise argparse.ArgumentTypeError("candidate cannot contain an empty model or path")
    return model, Path(path)


def _oof_row(model: str, root: Path) -> dict[str, object]:
    path = root / "oof_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing completed 2022 OOF metrics for {model}: {path}")
    frame = pd.read_csv(path, low_memory=False)
    selected = frame.loc[
        frame["encoder"].eq(model)
        & frame["fold"].astype(str).eq("pooled")
        & frame["horizon"].eq("6h")
        & frame["calibration"].eq("calibrated_2022_oof_pooled")
    ].copy()
    if selected.empty:
        raise ValueError(f"No pooled calibrated 2022 OOF rows for {model}: {path}")
    return {
        "model": model,
        "protocol_root": str(root),
        "seeds": int(selected["seed"].nunique()),
        "pr_auc_mean": float(pd.to_numeric(selected["pr_auc"]).mean()),
        "pr_auc_std": float(pd.to_numeric(selected["pr_auc"]).std()),
        "log_loss_mean": float(pd.to_numeric(selected["log_loss"]).mean()),
        "brier_mean": float(pd.to_numeric(selected["brier_score"]).mean()),
        "ece_mean": float(pd.to_numeric(selected["ece"]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the downstream probability model using 2022 pooled OOF metrics only. "
            "Locked 2023/2024 metrics are deliberately never read."
        )
    )
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--reference", default="fair_gru")
    parser.add_argument(
        "--maximum-log-loss-degradation",
        type=float,
        default=0.0,
        help="Allowed increase relative to the reference 2022 OOF log-loss.",
    )
    parser.add_argument(
        "--output-root",
        default="results/icing_model_experiments/selection_year_model_selection",
    )
    args = parser.parse_args()

    rows = [_oof_row(model, root) for model, root in args.candidate]
    table = pd.DataFrame(rows)
    reference_rows = table.loc[table["model"].eq(args.reference)]
    if reference_rows.shape[0] != 1:
        raise ValueError(f"Reference must appear exactly once: {args.reference}")
    reference_loss = float(reference_rows.iloc[0]["log_loss_mean"])
    table["passes_log_loss_guard"] = table["log_loss_mean"].le(
        reference_loss + float(args.maximum_log_loss_degradation)
    )
    eligible = table.loc[table["passes_log_loss_guard"]].copy()
    if eligible.empty:
        eligible = reference_rows.copy()
    # The hierarchy is frozen before inspecting either locked year: maximize
    # 2022 OOF PR-AUC, then prefer lower log-loss, Brier score, and complexity
    # order supplied by the caller for deterministic ties.
    eligible["candidate_order"] = [
        table.index[table["model"].eq(model)][0] for model in eligible["model"]
    ]
    selected = eligible.sort_values(
        ["pr_auc_mean", "log_loss_mean", "brier_mean", "candidate_order"],
        ascending=[False, True, True, True],
    ).iloc[0]

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "candidate_2022_oof_metrics.csv", index=False)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_model": str(selected["model"]),
        "selected_protocol_root": str(selected["protocol_root"]),
        "reference_model": args.reference,
        "selection_data": "2022 pooled purged OOF only",
        "locked_year_metrics_read": False,
        "selection_rule": (
            "Among candidates with mean 2022 OOF log-loss no worse than the reference "
            f"by more than {args.maximum_log_loss_degradation:g}, maximize mean PR-AUC; "
            "break ties by log-loss, Brier score, then declared order."
        ),
        "selected_metrics": {
            key: float(selected[key])
            for key in ("pr_auc_mean", "log_loss_mean", "brier_mean", "ece_mean")
        },
    }
    (output / "selected_model.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / ".complete_model_selection").touch()
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

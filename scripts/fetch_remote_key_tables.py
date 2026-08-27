from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath

import paramiko


KEY_TABLES = [
    "temporal_protocol/fold_summary.csv",
    "structure/station_event_counts.csv",
    "structure/event_order_summary.csv",
    "structure/recurrence_gap_summary.csv",
    "structure/monthly_event_counts.csv",
    "structure/seasonal_event_counts.csv",
    "definition_sensitivity/sensitivity_summary.csv",
    "definition_sensitivity/reference_onset_matching.csv",
    "model_comparison/statistical_models/pooled_oof_metrics.csv",
    "model_comparison/statistical_models/locked_year_metrics.csv",
    "model_comparison/deep_protocol/oof_metrics.csv",
    "model_comparison/deep_protocol/locked_year_metrics.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=os.environ["ICE_UPLOAD_PASSWORD"],
        timeout=30,
    )
    sftp = client.open_sftp()
    args.output.mkdir(parents=True, exist_ok=True)
    root = PurePosixPath(args.root)
    for relative in KEY_TABLES:
        destination = args.output.joinpath(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(str(root / relative), str(destination))
        print(relative)
    sftp.close()
    client.close()


if __name__ == "__main__":
    main()

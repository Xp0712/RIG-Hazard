from __future__ import annotations

import argparse
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import PurePosixPath
import stat

import pandas as pd
import paramiko


def walk(sftp: paramiko.SFTPClient, root: PurePosixPath):
    for entry in sftp.listdir_attr(str(root)):
        path = root / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            yield from walk(sftp, path)
        else:
            yield path, entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    password = os.environ["ICE_UPLOAD_PASSWORD"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, port=args.port, username=args.user, password=password, timeout=30)
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(20)
    _, stdout, _ = client.exec_command(
        "tail -n 80 /root/autodl-tmp/ice_project/results/logs/all_recurrence_experiments.log"
    )
    log_tail = stdout.read().decode("utf-8", errors="replace")
    sftp = client.open_sftp()
    root = PurePosixPath(args.root)
    rows: list[dict[str, object]] = []
    for path, attributes in walk(sftp, root):
        name = path.name.lower()
        if not (name.endswith(".csv") or name.endswith(".csv.gz")):
            continue
        relative = str(path.relative_to(root))
        row: dict[str, object] = {
            "relative_path": relative,
            "size_mb": round(attributes.st_size / 1024 / 1024, 3),
            "modified_at": datetime.fromtimestamp(attributes.st_mtime).isoformat(timespec="seconds"),
            "rows": None,
            "columns": None,
            "column_names": None,
            "read_status": "not_read_large" if attributes.st_size > 25 * 1024 * 1024 else "pending",
        }
        if attributes.st_size <= 25 * 1024 * 1024:
            try:
                with sftp.open(str(path), "rb") as handle:
                    payload = handle.read()
                frame = pd.read_csv(
                    BytesIO(payload),
                    compression="gzip" if name.endswith(".gz") else None,
                    low_memory=False,
                )
                row["rows"] = int(frame.shape[0])
                row["columns"] = int(frame.shape[1])
                row["column_names"] = ",".join(map(str, frame.columns))
                row["read_status"] = "ok"
            except Exception as error:
                row["read_status"] = f"error:{type(error).__name__}"
        rows.append(row)
    sftp.close()
    client.close()

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(rows).sort_values(["modified_at", "relative_path"]).to_csv(
        output, index=False, encoding="utf-8-sig"
    )
    print(json.dumps({"table_count": len(rows), "output": output, "log_tail": log_tail}, ensure_ascii=False))


if __name__ == "__main__":
    main()

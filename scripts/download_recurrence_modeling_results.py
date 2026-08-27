from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import stat

import paramiko


HOST = "connect.nmb1.seetacloud.com"
PORT = 23526
REMOTE_ROOT = PurePosixPath(
    "/root/autodl-tmp/ice_project/results/recurrence_modeling"
)
LOCAL_ROOT = Path("results/recurrence_modeling").resolve()


def walk(sftp: paramiko.SFTPClient, root: PurePosixPath):
    for entry in sftp.listdir_attr(str(root)):
        path = root / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            yield from walk(sftp, path)
        elif stat.S_ISREG(entry.st_mode):
            yield path, int(entry.st_size)


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    HOST, port=PORT, username="root", password=os.environ["ICE_UPLOAD_PASSWORD"],
    timeout=30, banner_timeout=30, auth_timeout=30,
)
transport = client.get_transport()
if transport is not None:
    transport.set_keepalive(20)
sftp = client.open_sftp()
files = list(walk(sftp, REMOTE_ROOT))
if os.environ.get("ICE_RESULTS_SUMMARY_ONLY") == "1":
    excluded = (
        PurePosixPath("probability_models/folds"),
        PurePosixPath("probability_models/oof_predictions"),
        PurePosixPath("probability_models/locked_predictions"),
    )
    files = [
        (path, size)
        for path, size in files
        if not any(
            path.relative_to(REMOTE_ROOT) == prefix
            or prefix in path.relative_to(REMOTE_ROOT).parents
            for prefix in excluded
        )
    ]
downloaded = 0
skipped = 0
for index, (remote, size) in enumerate(files, start=1):
    relative = remote.relative_to(REMOTE_ROOT)
    local = LOCAL_ROOT.joinpath(*relative.parts)
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size == size:
        skipped += 1
        continue
    temporary = local.with_suffix(local.suffix + ".downloading")
    print(f"[{index}/{len(files)}] {relative} ({size / 1024 / 1024:.1f} MiB)", flush=True)
    sftp.get(str(remote), str(temporary))
    if temporary.stat().st_size != size:
        raise IOError(f"Incomplete download: {relative}")
    temporary.replace(local)
    downloaded += 1
sftp.close()
client.close()
print(f"Download complete: downloaded={downloaded}, unchanged={skipped}, total={len(files)}")

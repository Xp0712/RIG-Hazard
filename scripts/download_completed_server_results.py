from __future__ import annotations

"""Synchronize completed server-side experiment outputs to this workspace.

Only result artifacts and execution logs are copied.  The transfer is
resumable: an existing local file with the same byte size is left untouched,
and partial transfers use a ``.downloading`` suffix until verified.
"""

import os
from pathlib import Path, PurePosixPath
import stat

import paramiko


HOST = "connect.nmb1.seetacloud.com"
PORT = 23526
USER = "root"
REMOTE_PROJECT = PurePosixPath("/root/autodl-tmp/ice_project")
INCLUDE = (PurePosixPath("results"), PurePosixPath("logs"))
LOCAL_PROJECT = Path(__file__).resolve().parents[1]


def walk(sftp: paramiko.SFTPClient, root: PurePosixPath):
    for entry in sftp.listdir_attr(str(root)):
        remote = root / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            yield from walk(sftp, remote)
        elif stat.S_ISREG(entry.st_mode):
            yield remote, int(entry.st_size)


def main() -> None:
    password = os.environ["ICE_UPLOAD_PASSWORD"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(20)

    sftp = client.open_sftp()
    files: list[tuple[PurePosixPath, int]] = []
    for relative_root in INCLUDE:
        files.extend(walk(sftp, REMOTE_PROJECT / relative_root))

    downloaded = 0
    unchanged = 0
    total_bytes = sum(size for _, size in files)
    print(
        f"Remote result files: {len(files)} "
        f"({total_bytes / 1024 / 1024 / 1024:.2f} GiB)",
        flush=True,
    )
    try:
        for index, (remote, size) in enumerate(files, start=1):
            relative = remote.relative_to(REMOTE_PROJECT)
            local = LOCAL_PROJECT.joinpath(*relative.parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and local.stat().st_size == size:
                unchanged += 1
                continue

            temporary = local.with_suffix(local.suffix + ".downloading")
            print(
                f"[{index}/{len(files)}] {relative} "
                f"({size / 1024 / 1024:.1f} MiB)",
                flush=True,
            )
            sftp.get(str(remote), str(temporary))
            if temporary.stat().st_size != size:
                raise IOError(f"Incomplete download: {relative}")
            temporary.replace(local)
            downloaded += 1
    finally:
        sftp.close()
        client.close()

    print(
        f"Download complete: downloaded={downloaded}, unchanged={unchanged}, "
        f"total={len(files)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

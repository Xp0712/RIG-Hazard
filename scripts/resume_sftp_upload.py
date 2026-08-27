from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import stat

import paramiko


EXCLUDED_NAMES = {".idea", ".python_deps", "__pycache__"}
EXCLUDED_PREFIXES = {Path("results/stable_graph_causal_budget")}


def included(path: Path, source: Path) -> bool:
    relative = path.relative_to(source)
    if any(part in EXCLUDED_NAMES for part in relative.parts):
        return False
    return not any(relative == prefix or prefix in relative.parents for prefix in EXCLUDED_PREFIXES)


def ensure_remote_directory(sftp: paramiko.SFTPClient, path: PurePosixPath) -> None:
    current = PurePosixPath("/")
    for part in path.parts[1:]:
        current /= part
        try:
            attributes = sftp.stat(str(current))
            if not stat.S_ISDIR(attributes.st_mode):
                raise NotADirectoryError(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume a project upload using SFTP file sizes.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    password = os.environ.get("ICE_UPLOAD_PASSWORD")
    if not password:
        raise RuntimeError("ICE_UPLOAD_PASSWORD is required")
    source = args.source.resolve()
    files = sorted(path for path in source.rglob("*") if path.is_file() and included(path, source))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport is unavailable")
    transport.set_keepalive(20)
    sftp = client.open_sftp()
    destination = PurePosixPath(args.destination)
    ensure_remote_directory(sftp, destination)
    directory_cache: set[PurePosixPath] = {destination}
    uploaded = 0
    skipped = 0
    for index, local_path in enumerate(files, start=1):
        relative = local_path.relative_to(source)
        remote_path = destination.joinpath(*relative.parts)
        if remote_path.parent not in directory_cache:
            ensure_remote_directory(sftp, remote_path.parent)
            directory_cache.add(remote_path.parent)
        local_size = local_path.stat().st_size
        try:
            remote_size = sftp.stat(str(remote_path)).st_size
        except FileNotFoundError:
            remote_size = -1
        if remote_size == local_size:
            skipped += 1
            continue
        print(
            f"[{index}/{len(files)}] uploading {relative} "
            f"({local_size / 1024 / 1024:.1f} MiB)",
            flush=True,
        )
        temporary_path = str(remote_path) + ".uploading"
        sftp.put(str(local_path), temporary_path, confirm=True)
        if remote_size >= 0:
            sftp.remove(str(remote_path))
        sftp.rename(temporary_path, str(remote_path))
        uploaded += 1
    sftp.close()
    client.close()
    print(f"Resume complete: uploaded={uploaded}, unchanged={skipped}, total={len(files)}")


if __name__ == "__main__":
    main()

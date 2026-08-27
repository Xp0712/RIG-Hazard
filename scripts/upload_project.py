from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import tarfile

import paramiko


EXCLUDED_NAMES = {".idea", ".python_deps", "__pycache__"}


def include_path(path: Path) -> bool:
    return not any(part in EXCLUDED_NAMES for part in path.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream the project to a remote SSH server.")
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
    if not source.is_dir():
        raise FileNotFoundError(source)

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
    destination = shlex.quote(args.destination)
    command = f"mkdir -p {destination} && tar -xzf - -C {destination}"
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport is unavailable")
    channel = transport.open_session()
    channel.set_combine_stderr(True)
    channel.exec_command(command)
    writer = channel.makefile("wb")

    roots = [path for path in sorted(source.iterdir()) if include_path(path.relative_to(source))]
    with tarfile.open(fileobj=writer, mode="w|gz", compresslevel=1) as archive:
        for index, path in enumerate(roots, start=1):
            print(f"[{index}/{len(roots)}] uploading {path.name}", flush=True)
            archive.add(
                path,
                arcname=path.name,
                recursive=True,
                filter=lambda info: info
                if include_path(Path(info.name))
                else None,
            )
    writer.close()
    output = channel.makefile("rb").read().decode("utf-8", errors="replace")
    exit_status = channel.recv_exit_status()
    client.close()
    if output.strip():
        print(output.strip())
    if exit_status != 0:
        raise RuntimeError(f"Remote extraction failed with exit status {exit_status}")
    print(f"Upload complete: {args.destination}")


if __name__ == "__main__":
    main()

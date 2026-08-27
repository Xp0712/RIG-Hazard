from __future__ import annotations

import getpass
import os

import paramiko


REMOTE_SCRIPT = r"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import os


ROOT = Path('/root/autodl-tmp/ice_project')
OUTPUTS = ROOT / 'results'
THRESHOLD = 500 * 1024 * 1024


def scan_tree(root: Path) -> tuple[int, int]:
    count = 0
    size = 0
    if not root.exists():
        return count, size
    for directory, _, files in os.walk(root):
        base = Path(directory)
        for name in files:
            path = base / name
            try:
                size += path.stat().st_size
                count += 1
            except OSError:
                pass
    return count, size


def artifact_rows(root: Path):
    if not root.exists():
        return
    for directory, _, files in os.walk(root):
        base = Path(directory)
        for name in files:
            path = base / name
            try:
                stat = path.stat()
            except OSError:
                continue
            yield stat.st_mtime, stat.st_size, path.relative_to(ROOT).as_posix()


print('SERVER_TIME=' + datetime.now().astimezone().isoformat(timespec='seconds'))
print('SECTION=TOP_LEVEL_OUTPUTS')
print('directory\tfile_count\ttotal_bytes')
for path in sorted((item for item in OUTPUTS.iterdir() if item.is_dir()), key=lambda item: item.name):
    count, size = scan_tree(path)
    print(f'{path.name}\t{count}\t{size}')

print('SECTION=LOGS')
count, size = scan_tree(ROOT / 'logs')
print('directory\tfile_count\ttotal_bytes')
print(f'logs\t{count}\t{size}')

alert_root = OUTPUTS / 'alert_governance'
public_root = alert_root / 'public_recurrence_benchmarks'
print('SECTION=ALERT_GOVERNANCE_ARTIFACTS_EXCLUDING_PUBLIC')
print('modified_time\tbytes\tpath')
for modified, size, relative in sorted(artifact_rows(alert_root) or []):
    if relative.startswith('results/alert_governance/public_recurrence_benchmarks/'):
        continue
    print(f'{datetime.fromtimestamp(modified).astimezone().isoformat(timespec="seconds")}\t{size}\t{relative}')

print('SECTION=PUBLIC_RECURRENCE_BENCHMARK_ARTIFACTS')
print('modified_time\tbytes\tpath')
for modified, size, relative in sorted(artifact_rows(public_root) or []):
    print(f'{datetime.fromtimestamp(modified).astimezone().isoformat(timespec="seconds")}\t{size}\t{relative}')

print('SECTION=LARGE_FILES_OVER_500MB')
print('bytes\tpath')
large_files = []
for directory, _, files in os.walk(OUTPUTS):
    base = Path(directory)
    for name in files:
        path = base / name
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > THRESHOLD:
            large_files.append((size, path.relative_to(ROOT).as_posix()))
for size, relative in sorted(large_files, reverse=True):
    print(f'{size}\t{relative}')
if not large_files:
    print('NONE')

print('SECTION=LARGE_CACHE_OR_CHECKPOINT_DIRS_OVER_500MB')
print('bytes\tpath')
candidates = []
keywords = ('cache', 'checkpoint', 'model')
for directory, dirs, _ in os.walk(OUTPUTS):
    for name in dirs:
        path = Path(directory) / name
        if not any(keyword in name.lower() for keyword in keywords):
            continue
        _, size = scan_tree(path)
        if size > THRESHOLD:
            candidates.append((size, path.relative_to(ROOT).as_posix()))
for size, relative in sorted(candidates, reverse=True):
    print(f'{size}\t{relative}')
if not candidates:
    print('NONE')
"""


def main() -> None:
    password = os.environ.get("ICE_REMOTE_PASSWORD") or getpass.getpass("SSH password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "connect.nmb1.seetacloud.com",
        port=23526,
        username="root",
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    _, stdout, stderr = client.exec_command(
        "/root/miniconda3/bin/python -",
        get_pty=False,
    )
    stdin = stdout.channel.makefile_stdin("wb")
    stdin.write(REMOTE_SCRIPT.encode("utf-8"))
    stdin.close()
    print(stdout.read().decode("utf-8", errors="replace"), end="")
    print(stderr.read().decode("utf-8", errors="replace"), end="")
    status = stdout.channel.recv_exit_status()
    client.close()
    if status != 0:
        raise SystemExit(status)


if __name__ == "__main__":
    main()

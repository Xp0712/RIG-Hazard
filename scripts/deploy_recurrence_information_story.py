from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shlex
import time
import getpass

import paramiko


HOST = "connect.nmb1.seetacloud.com"
PORT = 23526
USER = "root"
REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/ice_project")
FILES = (
    "scripts/explore_conditional_information_value.py",
    "scripts/explore_conditional_information_combinations.py",
    "scripts/diagnose_recurrence_information_failures.py",
    "scripts/run_recurrence_information_story.sh",
)


def run_remote(client: paramiko.SSHClient, command: str) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(command, get_pty=False)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    return status, output + error


def ensure_remote_directory(sftp: paramiko.SFTPClient, path: PurePosixPath) -> None:
    parts = path.parts
    current = PurePosixPath(parts[0])
    for part in parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def main() -> None:
    password = os.environ.get("ICE_UPLOAD_PASSWORD") or getpass.getpass("SSH password: ")

    local_root = Path(__file__).resolve().parents[1]
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

    status, output = run_remote(
        client,
        "cd /root/autodl-tmp/ice_project && "
        "date '+SERVER_TIME=%F %T %z' && "
        "test -f configs/rig_hazard_recurrence_modeling.json && "
        "test -f results/recurrence_modeling/seasonal_recurrence/"
        "event_global_seasonal_mapping.csv && "
        "printf 'PRECHECK=PASS\\n' && df -h . | tail -n 1",
    )
    print(output, end="")
    if status != 0:
        client.close()
        raise RuntimeError("Remote precheck failed")

    sftp = client.open_sftp()
    for relative in FILES:
        source = local_root / relative
        target = REMOTE_ROOT / PurePosixPath(relative)
        ensure_remote_directory(sftp, target.parent)
        temporary = PurePosixPath(f"{target}.codex-upload.tmp")
        sftp.put(str(source), str(temporary), confirm=True)
        sftp.posix_rename(str(temporary), str(target))
        print(f"UPLOADED {relative}", flush=True)
    sftp.chmod(str(REMOTE_ROOT / "scripts/run_recurrence_information_story.sh"), 0o755)
    sftp.close()

    quoted_files = " ".join(shlex.quote(path) for path in FILES[:3])
    status, output = run_remote(
        client,
        "cd /root/autodl-tmp/ice_project && "
        "PYTHONPATH=/root/autodl-tmp/ice_project/src "
        f"/root/miniconda3/bin/python -m py_compile {quoted_files} && "
        "bash -n scripts/run_recurrence_information_story.sh && "
        "printf 'REMOTE_VALIDATION=PASS\\n'",
    )
    print(output, end="")
    if status != 0:
        client.close()
        raise RuntimeError("Remote validation failed")

    status, output = run_remote(
        client,
        "cd /root/autodl-tmp/ice_project && mkdir -p results/logs && "
        "pid_file=results/logs/recurrence_information_story.pid; "
        "if test -s \"$pid_file\" && kill -0 \"$(cat \"$pid_file\")\" 2>/dev/null; then "
        "printf 'ALREADY_RUNNING PID=%s\\n' \"$(cat \"$pid_file\")\"; "
        "else "
        "nohup setsid bash scripts/run_recurrence_information_story.sh "
        "> results/logs/recurrence_information_story.nohup.log 2>&1 < /dev/null & "
        "new_pid=$!; printf '%s\\n' \"$new_pid\" > \"$pid_file\"; "
        "printf 'STARTED_PID=%s\\n' \"$new_pid\"; "
        "fi",
    )
    print(output, end="")
    if status != 0:
        client.close()
        raise RuntimeError("Remote start failed")

    time.sleep(8)
    status, output = run_remote(
        client,
        "cd /root/autodl-tmp/ice_project && "
        "printf '%s\\n' '--- PROCESS ---' && "
        "pid_file=results/logs/recurrence_information_story.pid; "
        "test -s \"$pid_file\" && ps -o pid,ppid,etime,stat,args -p \"$(cat \"$pid_file\")\"; "
        "pgrep -af '[e]xplore_conditional_information|[d]iagnose_recurrence_information' || true; "
        "printf '%s\\n' '--- LOG ---' && "
        "tail -n 40 results/logs/recurrence_information_story.nohup.log 2>/dev/null || true; "
        "printf '%s\\n' '--- MARKERS ---' && "
        "find results/recurrence_information_story -maxdepth 1 "
        "-name '.complete_*' -printf '%f\\n' 2>/dev/null | sort",
    )
    print(output, end="")
    client.close()
    if status != 0:
        raise RuntimeError("Remote post-start verification failed")


if __name__ == "__main__":
    main()

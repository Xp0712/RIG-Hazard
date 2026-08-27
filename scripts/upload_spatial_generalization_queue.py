from __future__ import annotations

import os
import time
from pathlib import Path, PurePosixPath

import paramiko


FILES = [
    "src/rig_hazard/deep_data.py",
    "src/rig_hazard/spatial_generalization.py",
    "scripts/run_spatial_generalization.py",
    "scripts/run_spatial_generalization_queue.sh",
    "tests/test_deep_data.py",
    "tests/test_spatial_generalization.py",
]
LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/ice_project")


def ensure_directory(sftp: paramiko.SFTPClient, path: PurePosixPath) -> None:
    current = PurePosixPath("/")
    for part in path.parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def run_checked(client: paramiko.SSHClient, command: str) -> str:
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(
            f"Remote command failed with exit code {status}:\n{output}\n{error}"
        )
    return output + error


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "connect.nmb1.seetacloud.com",
        port=23526,
        username="root",
        password=os.environ["ICE_UPLOAD_PASSWORD"],
    )
    sftp = client.open_sftp()
    for relative in FILES:
        remote = REMOTE_ROOT / PurePosixPath(relative)
        ensure_directory(sftp, remote.parent)
        temporary = f"{remote}.uploading"
        sftp.put(str(LOCAL_ROOT / relative), temporary, confirm=True)
        try:
            sftp.remove(str(remote))
        except FileNotFoundError:
            pass
        sftp.rename(temporary, str(remote))
        print(f"Uploaded {relative}", flush=True)
    queue_script = REMOTE_ROOT / "scripts/run_spatial_generalization_queue.sh"
    sftp.chmod(str(queue_script), 0o755)
    sftp.close()

    preflight = run_checked(
        client,
        "cd /root/autodl-tmp/ice_project; "
        "export PYTHONPATH=/root/autodl-tmp/ice_project/src; "
        "/root/miniconda3/bin/python -m py_compile "
        "src/rig_hazard/deep_data.py src/rig_hazard/spatial_generalization.py "
        "scripts/run_spatial_generalization.py; "
        "bash -n scripts/run_spatial_generalization_queue.sh; "
        "/root/miniconda3/bin/python -m unittest "
        "tests.test_deep_data tests.test_spatial_generalization"
    )
    print(preflight, flush=True)

    active_experiment = run_checked(
        client,
        "pgrep -af 'scripts/[r]un_spatial_generalization.py' || true",
    ).strip()
    if active_experiment:
        raise RuntimeError(
            "Spatial-generalization training is already active; refusing to replace its queue script:\n"
            + active_experiment
        )
    run_checked(
        client,
        "pkill -TERM -f '^bash scripts/run_spatial_generalization_queue.sh$' || true; "
        "sleep 2",
    )
    command = (
        "cd /root/autodl-tmp/ice_project; mkdir -p results/logs; "
        "nohup setsid bash scripts/run_spatial_generalization_queue.sh "
        ">> results/logs/spatial_generalization_queue.log 2>&1 < /dev/null & "
        "echo SPATIAL_QUEUE_PID=$!"
    )
    print(run_checked(client, command), flush=True)
    time.sleep(3)
    print(
        run_checked(
            client,
            "cd /root/autodl-tmp/ice_project; "
            "pgrep -af '^bash scripts/run_spatial_generalization_queue.sh$' || true; "
            "tail -n 20 results/logs/spatial_generalization_queue.log || true",
        ),
        flush=True,
    )
    client.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import getpass
import os
from pathlib import Path, PurePosixPath
import time

import paramiko


HOST = "connect.nmb1.seetacloud.com"
PORT = 23526
USER = "root"
REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/ice_project")
FILES = (
    "src/rig_hazard/alert_governance.py",
    "src/rig_hazard/public_recurrence.py",
    "scripts/run_public_recurrence_benchmarks.py",
    "tests/test_alert_governance.py",
    "tests/test_public_recurrence.py",
    "tests/test_public_recurrence_benchmarks.py",
)


def run(client: paramiko.SSHClient, command: str) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(command, get_pty=False)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    return status, output + error


def ensure_directory(sftp: paramiko.SFTPClient, path: PurePosixPath) -> None:
    current = PurePosixPath(path.parts[0])
    for part in path.parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def main() -> None:
    password = os.environ.get("ICE_REMOTE_PASSWORD") or getpass.getpass("SSH password: ")
    root = Path(__file__).resolve().parents[1]
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

    status, output = run(
        client,
        "cd /root/autodl-tmp/ice_project && date '+SERVER_TIME=%F %T %z' && "
        "printf '%s\\n' '--- PIPELINE BEFORE FIX ---' && "
        "pgrep -af '[r]un_alert_governance_experiments.sh|"
        "[r]un_public_recurrence_benchmarks.py' || true; "
        "printf '%s\\n' '--- COMPLETION MARKERS ---' && "
        "find results/alert_governance -maxdepth 1 "
        "-name '.complete_*' -printf '%f\\n' | sort",
    )
    print(output, end="")
    if status != 0:
        client.close()
        raise RuntimeError("Remote precheck failed")

    sftp = client.open_sftp()
    for relative in FILES:
        source = root / relative
        target = REMOTE_ROOT / PurePosixPath(relative)
        ensure_directory(sftp, target.parent)
        temporary = PurePosixPath(f"{target}.codex-upload.tmp")
        sftp.put(str(source), str(temporary), confirm=True)
        sftp.posix_rename(str(temporary), str(target))
        print(f"UPLOADED {relative}", flush=True)
    sftp.close()

    status, output = run(
        client,
        "set -e; cd /root/autodl-tmp/ice_project && "
        "export PYTHONPATH=/root/autodl-tmp/ice_project/src && "
        "/root/miniconda3/bin/python -m py_compile "
        "src/rig_hazard/alert_governance.py src/rig_hazard/public_recurrence.py "
        "scripts/run_public_recurrence_benchmarks.py && "
        "if ! /root/miniconda3/bin/python -c 'import cupy' 2>/dev/null; then "
        "/root/miniconda3/bin/python -m pip install --no-cache-dir cupy-cuda12x; "
        "fi && "
        "/root/miniconda3/bin/python -m unittest "
        "tests.test_alert_governance tests.test_public_recurrence "
        "tests.test_public_recurrence_benchmarks && "
        "/root/miniconda3/bin/python - <<'PY'\n"
        "import warnings\n"
        "import cupy as cp\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from xgboost import XGBClassifier\n"
        "from rig_hazard.public_recurrence import score_public_panel\n"
        "rng = np.random.default_rng(17)\n"
        "values = rng.normal(size=(512, 8)).astype(np.float32)\n"
        "labels = np.r_[np.zeros(448, dtype=np.int8), np.ones(64, dtype=np.int8)]\n"
        "model = XGBClassifier(n_estimators=3, max_depth=2, tree_method='hist', device='cuda')\n"
        "model.fit(cp.asarray(values), cp.asarray(labels))\n"
        "frame = pd.DataFrame(values, columns=[f'f{i}' for i in range(values.shape[1])])\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    prediction = score_public_panel(model, frame, list(frame.columns), batch_size=128)\n"
        "messages = [str(item.message) for item in caught]\n"
        "assert prediction.shape == (512,), prediction.shape\n"
        "assert np.isfinite(prediction).all()\n"
        "assert not any('mismatched devices' in message for message in messages), messages\n"
        "print('XGBOOST_GPU_INFERENCE=PASS')\n"
        "print('CUPY_VERSION=' + cp.__version__)\n"
        "PY\n"
        "printf 'REMOTE_VALIDATION=PASS\\n'",
    )
    print(output, end="")
    if status != 0:
        client.close()
        raise RuntimeError("Remote validation failed")

    status, output = run(
        client,
        "cd /root/autodl-tmp/ice_project && mkdir -p results/logs && "
        "pid_file=results/logs/alert_governance.pid; "
        "if test -f results/alert_governance/.complete_public_recurrence_benchmarks; then "
        "printf 'PUBLIC_BENCHMARK_ALREADY_COMPLETE\\n'; "
        "else "
        "if test -s \"$pid_file\" && kill -0 \"$(cat \"$pid_file\")\" 2>/dev/null; then "
        "old_pid=$(cat \"$pid_file\"); "
        "old_pgid=$(ps -o pgid= -p \"$old_pid\" | tr -d ' '); "
        "printf 'STOPPING_CPU_FALLBACK PID=%s PGID=%s\\n' \"$old_pid\" \"$old_pgid\"; "
        "if test \"$old_pgid\" = \"$old_pid\"; then kill -TERM -- \"-$old_pgid\"; "
        "else kill -TERM \"$old_pid\"; fi; "
        "for attempt in $(seq 1 20); do kill -0 \"$old_pid\" 2>/dev/null || break; sleep 1; done; "
        "if kill -0 \"$old_pid\" 2>/dev/null; then "
        "if test \"$old_pgid\" = \"$old_pid\"; then kill -KILL -- \"-$old_pgid\"; "
        "else kill -KILL \"$old_pid\"; fi; fi; "
        "fi; "
        "resume_log=results/logs/alert_governance_resume_$(date +%Y%m%d_%H%M%S).log; "
        "nohup setsid bash scripts/run_alert_governance_experiments.sh "
        "> \"$resume_log\" 2>&1 < /dev/null & "
        "new_pid=$!; printf '%s\\n' \"$new_pid\" > \"$pid_file\"; "
        "printf 'STARTED_PID=%s LOG=%s\\n' \"$new_pid\" \"$resume_log\"; "
        "fi",
    )
    print(output, end="")
    if status != 0:
        client.close()
        raise RuntimeError("Remote restart failed")

    time.sleep(10)
    status, output = run(
        client,
        "cd /root/autodl-tmp/ice_project && "
        "printf '%s\\n' '--- ACTIVE PROCESS ---' && "
        "pgrep -af '[r]un_alert_governance_experiments.sh|"
        "[r]un_public_recurrence_benchmarks.py' || true; "
        "printf '%s\\n' '--- LATEST RESUME LOG ---' && "
        "latest=$(ls -1t results/logs/alert_governance_resume_*.log 2>/dev/null | head -n 1); "
        "test -n \"$latest\" && tail -n 40 \"$latest\" || true; "
        "printf '%s\\n' '--- PUBLIC STAGE LOG ---' && "
        "tail -n 20 results/alert_governance/logs/"
        "public_recurrence_benchmarks.log 2>/dev/null || true",
    )
    print(output, end="")
    client.close()
    if status != 0:
        raise RuntimeError("Remote post-start verification failed")


if __name__ == "__main__":
    main()

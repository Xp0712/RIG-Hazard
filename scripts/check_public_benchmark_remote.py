from __future__ import annotations

import getpass
import os

import paramiko


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
    command = r"""
cd /root/autodl-tmp/ice_project
date '+SERVER_TIME=%F %T %z'
printf '%s\n' '--- PROCESSES ---'
pid_file=results/logs/alert_governance.pid
if test -s "$pid_file"; then
    pid=$(cat "$pid_file")
    ps -o pid,ppid,lstart,etime,stat,%cpu,%mem,args -p "$pid" || true
fi
pgrep -af '[r]un_alert_governance_experiments.sh|[r]un_public_recurrence_benchmarks.py' || true
printf '%s\n' '--- PUBLIC MARKERS ---'
find results/alert_governance/public_recurrence_benchmarks -name '.complete*' -printf '%p\n' 2>/dev/null | sort
find results/alert_governance -maxdepth 1 -name '.complete_public*' -printf '%p\n' 2>/dev/null | sort
printf '%s\n' '--- OUTPUT FILES ---'
find results/alert_governance/public_recurrence_benchmarks -maxdepth 2 -type f \( -name 'frozen_test_metrics.csv' -o -name 'run_manifest.json' \) -printf '%p %s bytes\n' 2>/dev/null | sort
printf '%s\n' '--- LOG TAIL ---'
tail -n 50 results/alert_governance/logs/public_recurrence_benchmarks.log 2>/dev/null || true
printf '%s\n' '--- RESUME LOG TAIL ---'
latest=$(ls -1t results/logs/alert_governance_resume_*.log 2>/dev/null | head -n 1)
test -n "$latest" && printf 'LATEST_LOG=%s\n' "$latest" && tail -n 60 "$latest" || true
printf '%s\n' '--- ERRORS SINCE RECOVERY ---'
if test -n "$latest"; then
    grep -nEi 'Traceback|CUDA error|out of memory|OOM|OverflowError|Killed|segmentation fault' "$latest" || true
fi
grep -nEi 'Traceback|CUDA error|out of memory|OOM|OverflowError|Killed|segmentation fault' results/alert_governance/logs/public_recurrence_benchmarks.log 2>/dev/null || true
printf '%s\n' '--- CUDA PYTHON ---'
/root/miniconda3/bin/python - <<'PY'
try:
    import cupy
    print('CUPY_VERSION=' + cupy.__version__)
    print('CUPY_DEVICE_COUNT=' + str(cupy.cuda.runtime.getDeviceCount()))
except Exception as exc:
    print('CUPY_UNAVAILABLE=' + repr(exc))
PY
printf '%s\n' '--- GPU ---'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
"""
    _, stdout, stderr = client.exec_command(command, get_pty=False)
    print(stdout.read().decode("utf-8", errors="replace"), end="")
    print(stderr.read().decode("utf-8", errors="replace"), end="")
    status = stdout.channel.recv_exit_status()
    client.close()
    if status != 0:
        raise SystemExit(status)


if __name__ == "__main__":
    main()

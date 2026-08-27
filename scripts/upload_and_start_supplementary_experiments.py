from __future__ import annotations

import os
import time
from pathlib import Path, PurePosixPath

import paramiko


FILES = [
    "src/rig_hazard/utility_warning.py",
    "scripts/run_utility_warning_experiments.py",
    "scripts/paired_probability_bootstrap.py",
    "scripts/run_candidate_attribution.py",
    "scripts/prepare_fair_baseline_config.py",
    "scripts/summarize_fair_baselines.py",
    "scripts/run_supplementary_experiments.sh",
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
sftp.chmod(str(REMOTE_ROOT / "scripts/run_supplementary_experiments.sh"), 0o755)
sftp.close()

preflight = (
    "cd /root/autodl-tmp/ice_project; "
    "/root/miniconda3/bin/python -m py_compile "
    "src/rig_hazard/utility_warning.py "
    "scripts/run_utility_warning_experiments.py "
    "scripts/paired_probability_bootstrap.py "
    "scripts/run_candidate_attribution.py "
    "scripts/prepare_fair_baseline_config.py "
    "scripts/summarize_fair_baselines.py; "
    "bash -n scripts/run_supplementary_experiments.sh; "
    "/root/miniconda3/bin/python -c \""
    "from pathlib import Path; "
    "r=Path('results/recurrence_modeling/probability_models'); "
    "m=['rec_none','rec_load','rec_previous','rec_full','rec_full_uniform']; "
    "s=[20260807,20260817,20260827,20260837,20260847]; "
    "mss=[str(p) for x in m for y in ['oof_predictions','locked_predictions/2023','locked_predictions/2024'] "
    "for z in s for p in [r/y/f'{x}_seed_{z}.npz'] if not p.exists()]; "
    "assert not mss, 'Missing predictions: '+str(mss); "
    "print('REMOTE_PREFLIGHT_OK prediction_files=75')\""
)
_, stdout, stderr = client.exec_command(preflight)
preflight_output = stdout.read().decode().strip()
preflight_error = stderr.read().decode().strip()
exit_code = stdout.channel.recv_exit_status()
if preflight_output:
    print(preflight_output, flush=True)
if exit_code != 0:
    if preflight_error:
        print(preflight_error, flush=True)
    client.close()
    raise SystemExit(f"Remote preflight failed with exit code {exit_code}")

command = (
    "cd /root/autodl-tmp/ice_project; mkdir -p results/logs; "
    "nohup setsid bash scripts/run_supplementary_experiments.sh "
    ">> results/logs/supplementary_experiments_master.log 2>&1 < /dev/null & echo $!"
)
_, stdout, stderr = client.exec_command(command)
pid = stdout.read().decode().strip()
error = stderr.read().decode().strip()
if error:
    print(error, flush=True)
print(f"REMOTE_PID={pid}", flush=True)
time.sleep(5)
_, stdout, stderr = client.exec_command(
    f"ps -p {pid} -o pid=,stat=,etime=,cmd=; "
    "tail -n 30 /root/autodl-tmp/ice_project/results/logs/supplementary_experiments_master.log"
)
print(stdout.read().decode(), flush=True)
error = stderr.read().decode().strip()
if error:
    print(error, flush=True)
client.close()

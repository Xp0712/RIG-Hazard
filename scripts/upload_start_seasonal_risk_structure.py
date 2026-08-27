from __future__ import annotations

import os
import time
from pathlib import Path, PurePosixPath

import paramiko


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/ice_project")
FILES = [
    "src/rig_hazard/seasonal_risk_structure.py",
    "scripts/run_seasonal_risk_structure.py",
]


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
    destination = REMOTE_ROOT / PurePosixPath(relative)
    temporary = f"{destination}.uploading"
    sftp.put(str(LOCAL_ROOT / relative), temporary, confirm=True)
    try:
        sftp.remove(str(destination))
    except FileNotFoundError:
        pass
    sftp.rename(temporary, str(destination))
    print(f"Uploaded {relative}", flush=True)
sftp.close()

smoke = " && ".join(
    [
        "cd /root/autodl-tmp/ice_project",
        "export PYTHONPATH=/root/autodl-tmp/ice_project/src",
        "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2",
        (
            "/root/miniconda3/bin/python -m py_compile "
            "src/rig_hazard/seasonal_risk_structure.py scripts/run_seasonal_risk_structure.py"
        ),
        (
            "/root/miniconda3/bin/python -u scripts/run_seasonal_risk_structure.py "
            "--folds 1 --bootstrap-samples 20 --maximum-train-rows 20000 "
            "--maximum-evaluation-rows 30000 "
            "--output-root results/recurrence_modeling/"
            "seasonal_risk_structure_check --overwrite"
        ),
    ]
)
_, stdout, stderr = client.exec_command(smoke, timeout=600)
for line in iter(stdout.readline, ""):
    print(line, end="")
error = stderr.read().decode("utf-8", "replace")
exit_code = stdout.channel.recv_exit_status()
if error:
    print(error, end="")
print(f"SMOKE_EXIT={exit_code}", flush=True)
if exit_code != 0:
    client.close()
    raise SystemExit(exit_code)

launch = (
    "cd /root/autodl-tmp/ice_project; mkdir -p results/logs; "
    "nohup setsid env PYTHONPATH=/root/autodl-tmp/ice_project/src "
    "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 "
    "/root/miniconda3/bin/python -u scripts/run_seasonal_risk_structure.py "
    "--config configs/rig_hazard_recurrence_modeling.json --folds 5 --bootstrap-samples 5000 "
    "--output-root results/recurrence_modeling/seasonal_risk_structure "
    "--overwrite >> results/logs/seasonal_risk_structure.log 2>&1 < /dev/null & echo $!"
)
_, stdout, stderr = client.exec_command(launch)
pid = stdout.read().decode().strip()
error = stderr.read().decode().strip()
if error:
    print(error, flush=True)
print(f"FULL_PID={pid}", flush=True)
time.sleep(5)
_, stdout, stderr = client.exec_command(
    f"ps -p {pid} -o pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,cmd=; "
    "tail -n 20 /root/autodl-tmp/ice_project/results/logs/seasonal_risk_structure.log"
)
print(stdout.read().decode("utf-8", "replace"), end="")
error = stderr.read().decode("utf-8", "replace")
if error:
    print(error, end="")
client.close()

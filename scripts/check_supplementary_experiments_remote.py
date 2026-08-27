from __future__ import annotations

import os

import paramiko


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "connect.nmb1.seetacloud.com",
    port=23526,
    username="root",
    password=os.environ["ICE_UPLOAD_PASSWORD"],
)
command = (
    "echo PROCESSES; "
    "pgrep -af 'run_supplementary_experiments|run_utility_warning|paired_probability|"
    "run_candidate_attribution|deep-protocol' || true; "
    "echo COMPLETED_STAGES; "
    "find /root/autodl-tmp/ice_project/results/recurrence_modeling/"
    "supplementary_experiments -maxdepth 1 -name '.complete_*' -printf '%f\\n' 2>/dev/null | sort; "
    "echo MASTER_LOG; "
    "tail -n 50 /root/autodl-tmp/ice_project/results/logs/supplementary_experiments_master.log; "
    "echo GPU; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
    "--format=csv,noheader"
)
_, stdout, stderr = client.exec_command(command)
print(stdout.read().decode(), end="")
error = stderr.read().decode().strip()
if error:
    print(error)
client.close()

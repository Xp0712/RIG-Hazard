from __future__ import annotations

import os
import time

import paramiko


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.nmb1.seetacloud.com", port=23526, username="root", password=os.environ["ICE_UPLOAD_PASSWORD"])
command = (
    "cd /root/autodl-tmp/ice_project; mkdir -p results/logs; "
    "nohup setsid bash scripts/run_recurrence_modeling_experiments.sh "
    ">> results/logs/recurrence_modeling_master.log 2>&1 < /dev/null & echo $!"
)
_, stdout, stderr = client.exec_command(command)
pid = stdout.read().decode().strip()
error = stderr.read().decode().strip()
if error:
    print(error)
print(f"PID={pid}")
time.sleep(5)
_, stdout, _ = client.exec_command(
    f"ps -p {pid} -o pid=,stat=,etime=,cmd=; tail -n 30 /root/autodl-tmp/ice_project/results/logs/recurrence_modeling_master.log"
)
print(stdout.read().decode())
client.close()

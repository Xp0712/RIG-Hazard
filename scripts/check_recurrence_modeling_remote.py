from __future__ import annotations

import os
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.nmb1.seetacloud.com", port=23526, username="root", password=os.environ["ICE_UPLOAD_PASSWORD"])
_, stdout, stderr = client.exec_command(
    "pgrep -af run_recurrence_modeling_experiments.sh; echo LOG; tail -n 40 /root/autodl-tmp/ice_project/results/logs/recurrence_modeling_master.log"
)
print(stdout.read().decode())
print(stderr.read().decode())
client.close()

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
    "ps -p 51448 -o pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,cmd=; "
    "echo RISK_LOG; tail -n 30 /root/autodl-tmp/ice_project/results/logs/"
    "seasonal_risk_structure.log; "
    "echo EXISTING_PIPELINE; pgrep -af 'paired_probability_bootstrap.py|"
    "run_supplementary_experiments.sh' | head -n 5"
)
_, stdout, stderr = client.exec_command(command)
print(stdout.read().decode("utf-8", "replace"), end="")
error = stderr.read().decode("utf-8", "replace")
if error:
    print(error, end="")
client.close()

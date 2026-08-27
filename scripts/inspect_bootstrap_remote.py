from __future__ import annotations

import os
import paramiko


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "connect.nmb1.seetacloud.com", port=23526, username="root",
    password=os.environ["ICE_UPLOAD_PASSWORD"], timeout=30,
)
command = r"""
echo ===PROCESS_TREE===
ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd --sort=-%cpu | head -n 20
echo ===MATCHING===
pgrep -af 'paired_station_bootstrap|run_recurrence_modeling_experiments'
echo ===OUTPUT===
find /root/autodl-tmp/ice_project/results/recurrence_modeling -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -n 25
echo ===DISK===
df -h /root/autodl-tmp
echo ===GPU===
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
"""
_, stdout, stderr = client.exec_command(command)
print(stdout.read().decode())
print(stderr.read().decode())
client.close()

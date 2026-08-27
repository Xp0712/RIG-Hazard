from __future__ import annotations

import os
import paramiko


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "connect.nmb1.seetacloud.com", port=23526, username="root",
    password=os.environ["ICE_UPLOAD_PASSWORD"], timeout=30,
)
_, stdout, stderr = client.exec_command(
    "kill 12450 12447 2>/dev/null || true; sleep 2; "
    "pgrep -af 'paired_station_bootstrap|run_recurrence_modeling_experiments' || true"
)
print(stdout.read().decode())
print(stderr.read().decode())
client.close()

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import paramiko


FILES = [
    "src/rig_hazard/cli.py", "src/rig_hazard/deep_data.py", "src/rig_hazard/deep_protocol.py",
    "src/rig_hazard/deep_training.py", "src/rig_hazard/seasonal_recurrence.py",
    "scripts/prepare_recurrence_modeling_config.py", "scripts/select_recurrence_model.py",
    "scripts/run_recurrence_modeling_experiments.sh", "scripts/run_locked_attribution.py",
    "scripts/paired_station_bootstrap.py",
]
root = Path(__file__).resolve().parents[1]
remote_root = PurePosixPath("/root/autodl-tmp/ice_project")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.nmb1.seetacloud.com", port=23526, username="root", password=os.environ["ICE_UPLOAD_PASSWORD"])
sftp = client.open_sftp()
for relative in FILES:
    target = remote_root / PurePosixPath(relative)
    try:
        sftp.stat(str(target.parent))
    except FileNotFoundError:
        sftp.mkdir(str(target.parent))
    sftp.put(str(root / relative), str(target), confirm=True)
    print(relative, flush=True)
sftp.chmod(str(remote_root / "scripts/run_recurrence_modeling_experiments.sh"), 0o755)
sftp.close()
client.close()

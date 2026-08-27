from __future__ import annotations

import json
import os
import time

import paramiko


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "connect.nmb1.seetacloud.com",
    port=23526,
    username="root",
    password=os.environ["ICE_UPLOAD_PASSWORD"],
)


def run(command: str) -> str:
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode()
    error = stderr.read().decode()
    if error:
        output += "\n" + error
    return output


print(run(
    "pgrep -af run_recurrence_modeling_experiments.sh; "
    "echo ===GPU===; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw "
    "--format=csv,noheader; echo ===LOG===; "
    "tail -n 25 /root/autodl-tmp/ice_project/results/logs/recurrence_modeling_master.log"
))

remote_config = "/root/autodl-tmp/ice_project/configs/rig_hazard_recurrence_modeling.json"
sftp = client.open_sftp()
with sftp.open(remote_config, "r") as handle:
    config = json.loads(handle.read().decode())
config["training"]["batch_size"] = 1024
config["training"]["torch_num_threads"] = 8
config["training"]["early_stopping_validation_samples"] = 20000
config["training"]["development_metric_samples"] = 120000
config["warning_evaluation"]["inference_batch_size"] = 4096
config["performance"] = {
    "allow_tf32": True,
    "cudnn_benchmark": True,
    "target_device": "RTX 4090",
    "training_batch_size": 1024,
    "inference_batch_size": 4096,
}
with sftp.open(remote_config, "w") as handle:
    handle.write(json.dumps(config, indent=2).encode())
sftp.close()
print("UPDATED batch_size=1024 torch_num_threads=8 inference_batch_size=4096")
client.close()

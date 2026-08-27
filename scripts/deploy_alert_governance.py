from __future__ import annotations

import argparse
import gzip
import json
import os
import posixpath
import shlex
import sys
import time
from pathlib import Path

import paramiko


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/autodl-tmp/ice_project"
CODE_FILES = (
    "src/rig_hazard/alert_governance.py",
    "src/rig_hazard/cluster_bootstrap.py",
    "src/rig_hazard/evaluation_provenance.py",
    "src/rig_hazard/event_eligibility.py",
    "src/rig_hazard/public_recurrence.py",
    "src/rig_hazard/xgb_baseline.py",
    "configs/rig_hazard_alert_governance.json",
    "scripts/run_nested_alert_audit.py",
    "scripts/run_alert_strategy_comparison.py",
    "scripts/run_public_recurrence_benchmarks.py",
    "scripts/prepare_public_aggregates.py",
    "scripts/run_xgboost_hazard_baseline.py",
    "scripts/run_alert_governance_experiments.sh",
    "scripts/run_strict_grid_recompute.sh",
    "tests/test_alert_governance.py",
    "tests/test_public_recurrence.py",
)
PUBLIC_AGGREGATES = (
    "results/alert_governance/public_recurrence_benchmarks/prepared/ecommerce_hourly.parquet",
    "results/alert_governance/public_recurrence_benchmarks/prepared/us_accidents_daily.parquet",
    "results/alert_governance/public_recurrence_benchmarks/prepared/public_aggregate_manifest.json",
)


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get("ICE_REMOTE_PASSWORD")
    key_file = os.environ.get("ICE_REMOTE_KEY_FILE")
    if not password and not key_file:
        raise RuntimeError(
            "Set ICE_REMOTE_PASSWORD or ICE_REMOTE_KEY_FILE before deployment"
        )
    if key_file and not Path(key_file).is_file():
        raise FileNotFoundError(f"SSH private key not found: {key_file}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password or None,
        key_filename=key_file or None,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    client.get_transport().set_keepalive(30)
    return client


def run(client: paramiko.SSHClient, command: str, check: bool = True) -> str:
    stdin, stdout, stderr = client.exec_command(command, get_pty=False)
    del stdin
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    combined = output + error
    if combined.strip():
        print(combined.rstrip(), flush=True)
    if check and status != 0:
        raise RuntimeError(f"Remote command failed ({status}): {command}")
    return combined


def ensure_remote_parent(client: paramiko.SSHClient, remote_path: str) -> None:
    parent = posixpath.dirname(remote_path)
    run(client, f"mkdir -p {shlex.quote(parent)}")


def atomic_upload(
    client: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    local_path: Path,
    remote_path: str,
) -> None:
    ensure_remote_parent(client, remote_path)
    temporary = remote_path + ".uploading"
    sftp.put(str(local_path), temporary)
    sftp.posix_rename(temporary, remote_path)
    print(f"UPLOADED {local_path.relative_to(PROJECT_ROOT)}", flush=True)


def remote_metadata(sftp: paramiko.SFTPClient, path: str) -> dict[str, int] | None:
    try:
        with sftp.open(path, "r") as source:
            value = source.read().decode("utf-8")
        parsed = json.loads(value)
        return {"source_size": int(parsed["source_size"])}
    except (FileNotFoundError, IOError, KeyError, ValueError, json.JSONDecodeError):
        return None


def upload_gzip(
    client: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    local_relative: str,
    remote_relative: str,
) -> None:
    local_path = PROJECT_ROOT / local_relative
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    source_size = local_path.stat().st_size
    compressed_path = PROJECT_ROOT / remote_relative
    local_metadata_path = compressed_path.with_suffix(compressed_path.suffix + ".source.json")
    local_metadata: dict[str, int] | None = None
    if compressed_path.exists() and local_metadata_path.exists():
        try:
            local_metadata = json.loads(local_metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            local_metadata = None
    if local_metadata != {"source_size": source_size}:
        compressed_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_local = compressed_path.with_suffix(compressed_path.suffix + ".building")
        processed = 0
        last_local_report = time.monotonic()
        with local_path.open("rb") as source, gzip.open(
            temporary_local, "wb", compresslevel=1
        ) as compressed:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                compressed.write(chunk)
                processed += len(chunk)
                if time.monotonic() - last_local_report >= 20:
                    print(
                        f"COMPRESS {local_path.name}: {processed / source_size:.1%}",
                        flush=True,
                    )
                    last_local_report = time.monotonic()
        os.replace(temporary_local, compressed_path)
        local_metadata_path.write_text(
            json.dumps({"source_size": source_size}), encoding="utf-8"
        )
        print(
            f"COMPRESSED {local_path.name}: {compressed_path.stat().st_size / 1024**3:.2f} GiB",
            flush=True,
        )
    remote_path = posixpath.join(REMOTE_ROOT, remote_relative.replace("\\", "/"))
    metadata_path = remote_path + ".source.json"
    metadata = remote_metadata(sftp, metadata_path)
    try:
        existing_size = int(sftp.stat(remote_path).st_size)
    except (FileNotFoundError, IOError):
        existing_size = 0
    if metadata == {"source_size": source_size} and existing_size > 0:
        print(f"REUSED {remote_relative} ({existing_size:,} compressed bytes)", flush=True)
        return
    ensure_remote_parent(client, remote_path)
    temporary = remote_path + ".uploading"
    last_report = time.monotonic()

    def progress(uploaded: int, total: int) -> None:
        nonlocal last_report
        if time.monotonic() - last_report >= 20 or uploaded == total:
            print(
                f"UPLOAD {compressed_path.name}: {uploaded / max(total, 1):.1%} "
                f"({uploaded / 1024**3:.2f}/{total / 1024**3:.2f} GiB)",
                flush=True,
            )
            last_report = time.monotonic()

    sftp.put(str(compressed_path), temporary, callback=progress)
    sftp.posix_rename(temporary, remote_path)
    payload = json.dumps({"source_size": source_size}).encode("utf-8")
    metadata_temporary = metadata_path + ".uploading"
    with sftp.open(metadata_temporary, "wb") as target:
        target.write(payload)
    sftp.posix_rename(metadata_temporary, metadata_path)
    compressed_size = int(sftp.stat(remote_path).st_size)
    print(
        f"UPLOADED {remote_relative}: {compressed_size / 1024**3:.2f} GiB compressed",
        flush=True,
    )


def check_server(client: paramiko.SSHClient) -> None:
    command = f"""
set -e
cd {shlex.quote(REMOTE_ROOT)}
echo CHECK_TIME=$(date -Iseconds)
df -h .
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo FAIR_PREDICTIONS
for model in fair_gru fair_timesnet; do
  printf '%s ' "$model"
  find results/recurrence_modeling/fair_baselines -type f \
    -name "${{model}}_seed_*.npz" | wc -l
done
echo REC_NONE_PREDICTIONS
find results/recurrence_modeling/probability_models -type f \
  -name 'rec_none_seed_*.npz' | wc -l
echo EXISTING_PIPELINE
pgrep -af '[r]un_alert_governance_experiments.sh' || true
echo MASTER_LOG_TAIL
tail -n 40 results/logs/alert_governance_master.log 2>/dev/null || true
"""
    run(client, command)


def validate_server(client: paramiko.SSHClient) -> None:
    python_files = " ".join(
        shlex.quote(path)
        for path in CODE_FILES
        if path.endswith(".py") and not path.startswith("tests/")
    )
    run(
        client,
        f"""
set -e
cd {shlex.quote(REMOTE_ROOT)}
export PYTHONPATH={shlex.quote(posixpath.join(REMOTE_ROOT, "src"))}
/root/miniconda3/bin/python -m py_compile {python_files}
bash -n scripts/run_alert_governance_experiments.sh
bash -n scripts/run_strict_grid_recompute.sh
/root/miniconda3/bin/python -m unittest tests.test_alert_governance tests.test_public_recurrence
/root/miniconda3/bin/python - <<'PY'
import importlib
import subprocess
import sys
missing = []
for package in ('duckdb', 'pyarrow', 'xgboost'):
    try:
        importlib.import_module(package)
    except ImportError:
        missing.append(package)
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
print('DEPENDENCIES_OK')
PY
/root/miniconda3/bin/python - <<'PY'
import numpy as np
import warnings
from xgboost import XGBClassifier
rng = np.random.default_rng(7)
features = rng.normal(size=(256, 8))
labels = np.r_[np.zeros(224, dtype=np.int8), np.ones(32, dtype=np.int8)]
model = XGBClassifier(n_estimators=2, max_depth=2, tree_method='hist', device='cuda')
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    model.fit(features, labels)
fallback = [str(item.message) for item in caught if 'GPU' in str(item.message) or 'device' in str(item.message)]
assert not fallback, fallback
assert model.predict_proba(features[:4]).shape == (4, 2)
print('XGBOOST_CUDA_OK')
PY
/root/miniconda3/bin/python - <<'PY'
import duckdb
for path in (
    'results/alert_governance/public_recurrence_benchmarks/prepared/ecommerce_hourly.parquet',
    'results/alert_governance/public_recurrence_benchmarks/prepared/us_accidents_daily.parquet',
):
    rows = duckdb.sql(f"SELECT count(*) FROM read_parquet('{{path}}')").fetchone()[0]
    assert rows > 0, path
    print(path, rows)
PY
echo REMOTE_VALIDATION_OK
""",
    )


def start_server(client: paramiko.SSHClient) -> None:
    existing = run(
        client,
        "pgrep -af '[r]un_alert_governance_experiments.sh' || true",
        check=False,
    ).strip()
    if existing:
        print("Pipeline already running; no duplicate was started.", flush=True)
        return
    output = run(
        client,
        f"""
cd {shlex.quote(REMOTE_ROOT)}
mkdir -p results/logs
nohup setsid bash scripts/run_alert_governance_experiments.sh \
  > results/logs/alert_governance_master.log 2>&1 < /dev/null &
echo NEW_PID=$!
sleep 8
ps -o pid,ppid,etime,stat,cmd -p $! || true
tail -n 30 results/logs/alert_governance_master.log || true
""",
    )
    if "NEW_PID=" not in output:
        raise RuntimeError("Remote pipeline did not return a PID")


def start_strict_grid_recompute(client: paramiko.SSHClient) -> None:
    existing = run(
        client,
        "pgrep -af '[r]un_strict_grid_recompute.sh' || true",
        check=False,
    ).strip()
    if existing:
        print("Strict-grid recompute is already running; no duplicate was started.", flush=True)
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"results/logs/strict_event_grid_recompute_{stamp}.log"
    output = run(
        client,
        f"""
cd {shlex.quote(REMOTE_ROOT)}
mkdir -p results/logs
nohup setsid bash scripts/run_strict_grid_recompute.sh \
  > {shlex.quote(log_path)} 2>&1 < /dev/null &
echo NEW_PID=$!
echo LOG_PATH={shlex.quote(log_path)}
sleep 5
ps -o pid,ppid,etime,stat,cmd -p $! || true
tail -n 20 {shlex.quote(log_path)} || true
""",
    )
    if "NEW_PID=" not in output:
        raise RuntimeError("Strict-grid recompute did not return a PID")


def install_cuda12_xgboost(client: paramiko.SSHClient) -> None:
    run(
        client,
        "/root/miniconda3/bin/python -m pip install "
        "'xgboost==3.2.0' 'nvidia-nccl-cu12'",
    )
    print("Installed CUDA-12-compatible XGBoost 3.2.0", flush=True)


def revoke_session_key(client: paramiko.SSHClient) -> None:
    key_file = os.environ.get("ICE_REMOTE_KEY_FILE")
    if not key_file:
        raise RuntimeError("ICE_REMOTE_KEY_FILE is required to revoke the session key")
    public_key = paramiko.Ed25519Key.from_private_key_file(key_file).get_base64()
    sed_script = f"\\|{public_key}|d"
    run(
        client,
        "set -e; "
        f"sed -i {shlex.quote(sed_script)} /root/.ssh/authorized_keys; "
        f"! grep -qF {shlex.quote(public_key)} /root/.ssh/authorized_keys; "
        "echo TEMP_KEY_REVOKED",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy and start alert-governance experiments")
    parser.add_argument("--host", default="connect.nmb1.seetacloud.com")
    parser.add_argument("--port", type=int, default=23526)
    parser.add_argument("--user", default="root")
    parser.add_argument(
        "--action",
        choices=(
            "check",
            "code",
            "data",
            "xgb-cuda12",
            "validate",
            "start",
            "strict-grid-start",
            "revoke-session-key",
            "all",
        ),
        default="all",
    )
    args = parser.parse_args()
    client = connect(args)
    try:
        sftp = client.open_sftp()
        if args.action in {"check", "all"}:
            check_server(client)
        if args.action in {"code", "all"}:
            for relative in CODE_FILES:
                local = PROJECT_ROOT / relative
                remote = posixpath.join(REMOTE_ROOT, relative.replace("\\", "/"))
                atomic_upload(client, sftp, local, remote)
            run(
                client,
                f"chmod 755 {REMOTE_ROOT}/scripts/run_alert_governance_experiments.sh "
                f"{REMOTE_ROOT}/scripts/run_strict_grid_recompute.sh",
            )
        if args.action in {"data", "all"}:
            for relative in PUBLIC_AGGREGATES:
                local = PROJECT_ROOT / relative
                if not local.exists():
                    raise FileNotFoundError(
                        f"Prepare public aggregates first: {local}"
                    )
                remote = posixpath.join(REMOTE_ROOT, relative.replace("\\", "/"))
                atomic_upload(client, sftp, local, remote)
        if args.action == "xgb-cuda12":
            install_cuda12_xgboost(client)
        if args.action in {"validate", "all"}:
            validate_server(client)
        if args.action in {"start", "all"}:
            start_server(client)
        if args.action == "strict-grid-start":
            start_strict_grid_recompute(client)
        if args.action == "revoke-session-key":
            revoke_session_key(client)
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DEPLOYMENT_FAILED: {exc}", file=sys.stderr, flush=True)
        raise

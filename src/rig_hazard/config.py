from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .naming import canonicalize_project_relative_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return config, config_path, config_hash


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / canonicalize_project_relative_path(str(path))
    return path.resolve()


def validate_config(config: dict[str, Any]) -> None:
    required = ["data_root", "output_root", "station_metadata", "time_step_minutes", "event", "risk_set", "exposure", "splits"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")

    step = int(config["time_step_minutes"])
    if step <= 0 or 60 % step != 0:
        raise ValueError("time_step_minutes must be a positive divisor of 60")

    horizons = [int(value) for value in config["risk_set"]["forecast_horizons_hours"]]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("forecast_horizons_hours must contain positive integers")

    event = config["event"]
    if int(event["merge_gap_minutes"]) < 0 or int(event["cooldown_minutes"]) < 0:
        raise ValueError("event gap and cooldown values must be non-negative")

    split_years: set[int] = set()
    for split_name in ("development", "final"):
        split = config["splits"][split_name]
        for key in ("train_years", "validation_years", "test_years"):
            split_years.update(int(year) for year in split.get(key, []))
    if not split_years:
        raise ValueError("At least one split year is required")

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .deep_data import DeepCacheBatchSource


@dataclass(frozen=True)
class XGBSummaryContract:
    included_feature_indices: tuple[int, ...]
    included_feature_names: tuple[str, ...]
    statistics: tuple[str, ...] = ("last", "mean", "minimum", "maximum")

    @property
    def output_feature_count(self) -> int:
        return len(self.included_feature_indices) * len(self.statistics)


def summary_contract(
    cache_root: str | Path,
    masked_feature_names: list[str] | tuple[str, ...],
) -> XGBSummaryContract:
    contract = json.loads(
        (Path(cache_root) / "sample_contract.json").read_text(encoding="utf-8")
    )
    names = list(contract["feature_names"])
    masked = set(masked_feature_names)
    indices = tuple(index for index, name in enumerate(names) if name not in masked)
    return XGBSummaryContract(indices, tuple(names[index] for index in indices))


def summarize_history(history: np.ndarray, contract: XGBSummaryContract) -> np.ndarray:
    values = np.asarray(history, dtype=np.float32)[:, :, contract.included_feature_indices]
    return np.concatenate(
        [
            values[:, -1, :],
            np.nanmean(values, axis=1),
            np.nanmin(values, axis=1),
            np.nanmax(values, axis=1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def iter_summary_batches(
    source: DeepCacheBatchSource,
    contract: XGBSummaryContract,
    batch_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    for batch in source.iter_batches(batch_size, shuffle=False):
        history = batch["history"].numpy()
        labels = batch["hazard_target"].numpy()
        mask = batch["risk_mask"].numpy()
        event = labels.max(axis=1) > 0.5
        observed = event | (mask.sum(axis=1) >= mask.shape[1])
        yield {
            "features": summarize_history(history, contract),
            "label": event.astype(np.int8),
            "observed": observed.astype(np.int8),
            "sample_weight": batch["sample_weight"].numpy().astype(np.float32),
            "file_id": batch["file_id"].numpy().astype(np.int16),
            "row_index": batch["row_index"].numpy().astype(np.int32),
            "issue_time_ns": batch["issue_time_ns"].numpy().astype(np.int64),
        }


def training_matrix(
    source: DeepCacheBatchSource,
    contract: XGBSummaryContract,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for batch in iter_summary_batches(source, contract, batch_size):
        keep = batch["observed"].astype(bool)
        feature_parts.append(batch["features"][keep])
        label_parts.append(batch["label"][keep])
        weight_parts.append(batch["sample_weight"][keep])
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    weights = np.concatenate(weight_parts)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return features, labels, weights


def predict_source(
    model: Any,
    source: DeepCacheBatchSource,
    contract: XGBSummaryContract,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {
        "file_id": [],
        "row_index": [],
        "issue_time_ns": [],
        "risk_6h": [],
        "onset_within_6h": [],
        "observed_6h": [],
    }
    for batch in iter_summary_batches(source, contract, batch_size):
        probability = model.predict_proba(batch["features"])[:, 1]
        output["file_id"].append(batch["file_id"])
        output["row_index"].append(batch["row_index"])
        output["issue_time_ns"].append(batch["issue_time_ns"])
        output["risk_6h"].append(np.asarray(probability, dtype=np.float32))
        output["onset_within_6h"].append(batch["label"])
        output["observed_6h"].append(batch["observed"])
    return {key: np.concatenate(parts) for key, parts in output.items()}


def fit_logit_calibrator(probability: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    raw = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    target = np.asarray(label, dtype=np.int8)
    observed = np.isfinite(logit[:, 0])
    model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    model.fit(logit[observed], target[observed])
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def apply_logit_calibrator(
    probability: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    raw = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    value = float(intercept) + float(slope) * np.log(raw / (1.0 - raw))
    return (1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))).astype(np.float32)


def save_prediction(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((arrays["row_index"], arrays["file_id"]))
    np.savez_compressed(path, **{key: value[order] for key, value in arrays.items()})

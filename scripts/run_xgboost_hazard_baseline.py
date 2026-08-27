from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.config import resolve_project_path
from rig_hazard.deep_data import DeepCacheBatchSource
from rig_hazard.xgb_baseline import (
    apply_logit_calibrator,
    fit_logit_calibrator,
    predict_source,
    save_prediction,
    summary_contract,
    training_matrix,
)


def _new_model(settings: dict, seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=int(settings.get("n_estimators", 350)),
        max_depth=int(settings.get("max_depth", 6)),
        learning_rate=float(settings.get("learning_rate", 0.05)),
        subsample=float(settings.get("subsample", 0.8)),
        colsample_bytree=float(settings.get("colsample_bytree", 0.8)),
        min_child_weight=float(settings.get("min_child_weight", 5.0)),
        reg_lambda=float(settings.get("reg_lambda", 2.0)),
        reg_alpha=float(settings.get("reg_alpha", 0.0)),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=str(settings.get("device", "cuda")),
        n_jobs=int(settings.get("n_jobs", 8)),
        random_state=int(seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2022-only XGBoost six-hour baseline.")
    parser.add_argument("--config", default="configs/rig_hazard_alert_governance.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    settings = config["xgboost_baseline"]
    output_root = resolve_project_path(args.output_root or settings["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / ".complete_xgboost_baseline"
    if args.resume and marker.exists():
        print(f"XGBoost baseline already complete: {output_root}", flush=True)
        return

    cache_root = resolve_project_path(config["cache_root"])
    seed = int(settings.get("seed", 20260807))
    batch_size = int(settings.get("batch_size", 2048))
    masked = list(settings["masked_features"])
    contract = summary_contract(cache_root, masked)
    oof_parts: list[dict[str, np.ndarray]] = []
    checkpoints = output_root / "fold_models"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for fold in range(int(settings.get("folds", 5))):
        prediction_path = output_root / "fold_predictions" / f"fold{fold}.npz"
        model_path = checkpoints / f"fold{fold}.json"
        if args.resume and prediction_path.exists() and model_path.exists():
            with np.load(prediction_path, allow_pickle=False) as values:
                oof_parts.append({key: values[key].copy() for key in values.files})
            print(f"XGBoost fold {fold} reused", flush=True)
            continue
        train = DeepCacheBatchSource(cache_root, f"cv_fold{fold}_train")
        validation = DeepCacheBatchSource(cache_root, f"cv_fold{fold}_validation")
        features, labels, weights = training_matrix(train, contract, batch_size)
        positive_multiplier = float(settings.get("positive_weight_multiplier", 1.0))
        weights = weights * np.where(labels == 1, positive_multiplier, 1.0)
        model = _new_model(settings, seed + fold)
        model.fit(features, labels, sample_weight=weights)
        prediction = predict_source(model, validation, contract, batch_size)
        model.save_model(model_path)
        save_prediction(prediction_path, prediction)
        oof_parts.append(prediction)
        print(
            f"XGBoost fold {fold + 1}/{settings.get('folds', 5)} complete: "
            f"train_rows={labels.size}, positives={int(labels.sum())}",
            flush=True,
        )

    keys = oof_parts[0].keys()
    oof = {key: np.concatenate([part[key] for part in oof_parts]) for key in keys}
    observed = oof["observed_6h"].astype(bool)
    intercept, slope = fit_logit_calibrator(
        oof["risk_6h"][observed], oof["onset_within_6h"][observed]
    )
    oof["risk_6h"] = apply_logit_calibrator(oof["risk_6h"], intercept, slope)
    model_name = str(settings.get("model_name", "xgboost"))
    save_prediction(
        output_root / "oof_predictions" / f"{model_name}_seed_{seed}.npz", oof
    )

    train = DeepCacheBatchSource(cache_root, "selection_2022")
    features, labels, weights = training_matrix(train, contract, batch_size)
    weights = weights * np.where(
        labels == 1, float(settings.get("positive_weight_multiplier", 1.0)), 1.0
    )
    trained_model = _new_model(settings, seed)
    trained_model.fit(features, labels, sample_weight=weights)
    trained_model_root = output_root / "trained_models"
    trained_model_root.mkdir(parents=True, exist_ok=True)
    trained_model.save_model(trained_model_root / "xgboost.json")
    for year, split in ((2023, "cross_year_2023"), (2024, "final_time_2024")):
        source = DeepCacheBatchSource(cache_root, split)
        prediction = predict_source(trained_model, source, contract, batch_size)
        prediction["risk_6h"] = apply_logit_calibrator(
            prediction["risk_6h"], intercept, slope
        )
        save_prediction(
            output_root
            / "locked_predictions"
            / str(year)
            / f"{model_name}_seed_{seed}.npz",
            prediction,
        )
        print(f"XGBoost locked prediction complete: {year}", flush=True)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection_data": "2022 purged pooled OOF only",
        "locked_years": [2023, 2024],
        "model_name": model_name,
        "seed": seed,
        "masked_features": masked,
        "summary_statistics": list(contract.statistics),
        "input_feature_count": contract.output_feature_count,
        "calibrator": {"intercept": intercept, "slope": slope},
        "role": "controller-independence tabular baseline, not a replacement main model",
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker.touch()
    print(f"XGBoost hazard baseline complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()

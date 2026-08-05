"""Ablate broad per-user C/D/V means in the advanced CatBoost."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_honest_advanced_data import ADVANCED_CAT_PARAMS as CAT_PARAMS
from fraud_vblock_features import add_vblock_user_features
from train_honest_advanced_catboost import FOLDS, prepare


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "honest_user_means_catboost"
REPORT_PATH = CACHE_DIR / "report.json"
SOURCE = "user_means_catboost"
TARGET = "isFraud"
MAX_ITERATIONS = 1_900
SEED = 1729


def parse_folds(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(fold not in {0, 1, 2} for fold in result):
        raise argparse.ArgumentTypeError("Use a comma-separated subset of 0,1,2")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=parse_folds, default=(2,))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    CACHE_DIR.mkdir(exist_ok=True)
    started = time.time()

    prepared, features, categorical = prepare()
    (
        prepared["train"],
        prepared["inference"],
        vblock_features,
        vblock_stats,
    ) = add_vblock_user_features(
        prepared["train"],
        prepared["inference"],
        prepared["train_components"],
        prepared["inference_components"],
    )
    mean_features = [
        column
        for column in vblock_features
        if re.fullmatch(r"wide_user_[CDV]\d+_mean", column)
    ]
    features = list(dict.fromkeys([*features, *mean_features]))
    day = prepared["train"]["TransactionDT"].to_numpy(dtype="float64") / 86_400.0
    y = prepared["y"].to_numpy(dtype="int8")
    rows = []
    for fold, train_end, valid_start, valid_end in FOLDS:
        if fold not in args.folds:
            continue
        fit_index = np.flatnonzero(day < train_end)
        valid_index = np.flatnonzero((day >= valid_start) & (day < valid_end))
        model_path = CACHE_DIR / f"fold_{fold}.cbm"
        prediction_path = CACHE_DIR / f"fold_{fold}_prediction.npy"
        fold_started = time.time()
        if model_path.exists() and prediction_path.exists() and not args.force:
            prediction = np.load(prediction_path)
            model = CatBoostClassifier()
            model.load_model(model_path)
            iteration = int(model.tree_count_)
            if prediction.dtype != np.float64:
                print(
                    f"Refreshing fold {fold} predictions as float64",
                    flush=True,
                )
                prediction = model.predict_proba(
                    prepared["train"].iloc[valid_index][features]
                )[:, 1]
                np.save(prediction_path, prediction)
            del model
            cached = True
        else:
            print(
                f"User-means CatBoost fold {fold}: {len(fit_index):,} -> "
                f"{len(valid_index):,}; {len(features)} features",
                flush=True,
            )
            model = CatBoostClassifier(
                **{
                    **CAT_PARAMS,
                    "iterations": MAX_ITERATIONS,
                    "random_seed": SEED + fold,
                    "verbose": 200,
                }
            )
            model.fit(
                prepared["train"].iloc[fit_index][features],
                y[fit_index],
                cat_features=categorical,
                eval_set=(
                    prepared["train"].iloc[valid_index][features],
                    y[valid_index],
                ),
                early_stopping_rounds=140,
                use_best_model=True,
            )
            prediction = model.predict_proba(
                prepared["train"].iloc[valid_index][features]
            )[:, 1]
            iteration = int(model.tree_count_)
            model.save_model(model_path)
            np.save(prediction_path, prediction)
            del model
            gc.collect()
            cached = False
        score = float(roc_auc_score(y[valid_index], prediction))
        pd.DataFrame(
            {
                "row_index": valid_index,
                "TransactionID": prepared["train_ids"].iloc[valid_index],
                "fold": fold,
                TARGET: y[valid_index],
                SOURCE: prediction,
            }
        ).to_csv(CACHE_DIR / f"fold_{fold}_oof.csv", index=False)
        row = {
            "fold": fold,
            "train_rows": int(len(fit_index)),
            "valid_rows": int(len(valid_index)),
            "auc": score,
            "best_iteration": iteration,
            "cached": cached,
            "minutes": (time.time() - fold_started) / 60.0,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    report = {
        "data_policy": "official train/test only; no gap/test labels",
        "folds_requested": list(args.folds),
        "base_features": len(features) - len(mean_features),
        "user_mean_features": mean_features,
        "total_features": len(features),
        "categorical": len(categorical),
        "vblock_stats": vblock_stats,
        "models": rows,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Train and evaluate an advanced-UID CatBoost source on clean temporal OOF."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from build_honest_no_gap_meta import BASE_SOURCES, evaluate_source_set, make_uid
from fraud_honest_advanced_data import (
    ADVANCED_CAT_PARAMS as CAT_PARAMS,
    prepare_advanced_catboost_data,
)


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "honest_advanced_catboost"
OOF_PATH = CACHE_DIR / "oof.csv"
REPORT_PATH = CACHE_DIR / "oof_report.json"
SOURCE = "advanced_catboost"
TARGET = "isFraud"
MAX_ITERATIONS = 1_700
SEED = 1729
FOLDS = (
    (0, 30, 60, 75),
    (1, 45, 75, 90),
    (2, 60, 90, 106),
)


def prepare() -> tuple[dict, list[str], list[str]]:
    return prepare_advanced_catboost_data(ROOT)


def train_oof(prepared: dict, features: list[str], categorical: list[str], force: bool) -> tuple[pd.DataFrame, list[dict]]:
    CACHE_DIR.mkdir(exist_ok=True)
    day = prepared["train"]["TransactionDT"].to_numpy(dtype="float64") / 86_400.0
    y = prepared["y"].to_numpy(dtype="int8")
    parts = []
    metrics = []
    for fold, train_end, valid_start, valid_end in FOLDS:
        fit_index = np.flatnonzero(day < train_end)
        valid_index = np.flatnonzero((day >= valid_start) & (day < valid_end))
        prediction_path = CACHE_DIR / f"fold_{fold}_prediction.npy"
        model_path = CACHE_DIR / f"fold_{fold}.cbm"
        metadata_path = CACHE_DIR / f"fold_{fold}.json"
        started = time.time()
        if prediction_path.exists() and metadata_path.exists() and not force:
            prediction = np.load(prediction_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if len(prediction) != len(valid_index):
                raise ValueError(f"Cached fold {fold} has the wrong length")
            if prediction.dtype != np.float64:
                print(
                    f"Refreshing fold {fold} predictions as float64",
                    flush=True,
                )
                model = CatBoostClassifier()
                model.load_model(model_path)
                prediction = model.predict_proba(
                    prepared["train"].iloc[valid_index][features]
                )[:, 1]
                np.save(prediction_path, prediction)
                del model
                gc.collect()
            metadata["cached"] = True
        else:
            print(
                f"Advanced CatBoost fold {fold}: {len(fit_index):,} -> "
                f"{len(valid_index):,}",
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
            model.save_model(model_path)
            np.save(prediction_path, prediction)
            metadata = {
                "fold": fold,
                "train_rows": int(len(fit_index)),
                "valid_rows": int(len(valid_index)),
                "best_iteration": int(model.tree_count_),
                "auc": float(roc_auc_score(y[valid_index], prediction)),
                "minutes": (time.time() - started) / 60.0,
                "cached": False,
                "model": model_path.name,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            del model
            gc.collect()
        metadata["auc"] = float(roc_auc_score(y[valid_index], prediction))
        metrics.append(metadata)
        parts.append(
            pd.DataFrame(
                {
                    "row_index": valid_index,
                    "TransactionID": prepared["train_ids"].iloc[valid_index].to_numpy(),
                    "fold": fold,
                    TARGET: y[valid_index],
                    SOURCE: prediction,
                }
            )
        )
        print(f"fold {fold} AUC={metadata['auc']:.9f}", flush=True)
    oof = pd.concat(parts, ignore_index=True).sort_values(["fold", "row_index"])
    oof.to_csv(OOF_PATH, index=False)
    return oof, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.time()
    prepared, features, categorical = prepare()
    source_oof, fold_metrics = train_oof(
        prepared, features, categorical, args.force
    )

    base = pd.read_csv(ROOT / "boost3_oof_predictions.csv")
    oof = base.merge(
        source_oof[["row_index", SOURCE]],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    raw = pd.read_csv(
        ROOT / "train_transaction.csv",
        usecols=["TransactionDT", "card1", "addr1", "D1", "P_emaildomain"],
    )
    uid = make_uid(raw)
    baseline = evaluate_source_set(oof, uid, BASE_SOURCES)
    candidate = evaluate_source_set(oof, uid, (*BASE_SOURCES, SOURCE))
    gain = float(candidate["uid_recipe"]["auc"] - baseline["uid_recipe"]["auc"])
    report = {
        "data_policy": "official train/test only; train labels only for temporal OOF",
        "features": len(features),
        "categorical": len(categorical),
        "params": {**CAT_PARAMS, "iterations": MAX_ITERATIONS, "random_seed": SEED},
        "folds": fold_metrics,
        "baseline": baseline,
        "candidate": candidate,
        "holdout_gain": gain,
        "accepted": gain > 0,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

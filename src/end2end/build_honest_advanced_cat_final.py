"""Train the locked three-seed advanced CatBoost and build its clean stack."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from build_honest_no_gap_meta import evaluate_source_set, fit_final, make_uid, rank_prediction
from fraud_honest_advanced_data import ADVANCED_CAT_PARAMS as CAT_PARAMS
from train_honest_advanced_catboost import CACHE_DIR, SOURCE, prepare


ROOT = Path(__file__).resolve().parent
TARGET = "isFraud"
ENHANCED_SOURCE = "cat_enhanced"
SOURCES = ("catboost", ENHANCED_SOURCE, "lightgbm", "xgboost")
SEEDS = (1729, 2026, 3407)
FINAL_ITERATIONS = 1_900
ADVANCED_WEIGHT = 0.25
OUTPUT_PATH = ROOT / "submission_honest_advanced_cat_multiseed.csv"
REPORT_PATH = CACHE_DIR / "final_report.json"


def train_final_source(
    prepared: dict,
    features: list[str],
    categorical: list[str],
    force: bool,
) -> tuple[np.ndarray, list[dict]]:
    rank_parts = []
    rows = []
    for seed in SEEDS:
        model_path = CACHE_DIR / f"final_seed_{seed}.cbm"
        prediction_path = CACHE_DIR / f"final_seed_{seed}_test.npy"
        started = time.time()
        if model_path.exists() and prediction_path.exists() and not force:
            prediction = np.load(prediction_path)
            if prediction.dtype != np.float64:
                print(
                    f"Refreshing final seed {seed} predictions as float64",
                    flush=True,
                )
                model = CatBoostClassifier()
                model.load_model(model_path)
                prediction = model.predict_proba(
                    prepared["inference"][features]
                )[:, 1]
                np.save(prediction_path, prediction)
                del model
                gc.collect()
            cached = True
        else:
            print(
                f"Final advanced CatBoost seed {seed}: "
                f"{FINAL_ITERATIONS} trees",
                flush=True,
            )
            model = CatBoostClassifier(
                **{
                    **CAT_PARAMS,
                    "iterations": FINAL_ITERATIONS,
                    "random_seed": seed,
                    "verbose": 200,
                }
            )
            model.fit(
                prepared["train"][features],
                prepared["y"],
                cat_features=categorical,
            )
            prediction = model.predict_proba(
                prepared["inference"][features]
            )[:, 1]
            model.save_model(model_path)
            np.save(prediction_path, prediction)
            del model
            gc.collect()
            cached = False
        rank_parts.append(rank_prediction(prediction))
        rows.append(
            {
                "seed": seed,
                "iterations": FINAL_ITERATIONS,
                "cached": cached,
                "model": model_path.name,
                "minutes": (time.time() - started) / 60.0,
            }
        )
    return np.mean(rank_parts, axis=0), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.time()

    prepared, features, categorical = prepare()
    advanced_rank, seed_rows = train_final_source(
        prepared, features, categorical, args.force
    )
    raw_test_ids = prepared["inference_ids"].to_numpy()
    del prepared
    gc.collect()

    base_oof = pd.read_csv(ROOT / "boost3_oof_predictions.csv")
    advanced_oof = pd.read_csv(
        CACHE_DIR / "oof.csv", usecols=["row_index", SOURCE]
    )
    oof = base_oof.merge(
        advanced_oof,
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    old_rank_oof = oof.groupby("fold")["catboost"].rank(pct=True)
    advanced_rank_oof = oof.groupby("fold")[SOURCE].rank(pct=True)
    oof[ENHANCED_SOURCE] = (
        (1.0 - ADVANCED_WEIGHT) * old_rank_oof
        + ADVANCED_WEIGHT * advanced_rank_oof
    )

    raw_train = pd.read_csv(
        ROOT / "train_transaction.csv",
        usecols=["TransactionDT", "card1", "addr1", "D1", "P_emaildomain"],
    )
    raw_test = pd.read_csv(
        ROOT / "test_transaction.csv",
        usecols=[
            "TransactionID",
            "TransactionDT",
            "card1",
            "addr1",
            "D1",
            "P_emaildomain",
        ],
    )
    uid_train = make_uid(raw_train)
    uid_test = make_uid(raw_test)
    baseline = evaluate_source_set(
        oof, uid_train, ("catboost", "lightgbm", "xgboost")
    )
    selected = evaluate_source_set(oof, uid_train, SOURCES)
    holdout_gain = float(
        selected["uid_recipe"]["auc"] - baseline["uid_recipe"]["auc"]
    )
    if holdout_gain <= 0:
        raise RuntimeError(f"Locked enhanced CatBoost failed: {holdout_gain}")

    old_cat_test = np.load(ROOT / "stack_test_catboost.npy")
    enhanced_test = (
        (1.0 - ADVANCED_WEIGHT) * rank_prediction(old_cat_test)
        + ADVANCED_WEIGHT * advanced_rank
    )
    test_predictions = pd.DataFrame(
        {
            "catboost": old_cat_test,
            ENHANCED_SOURCE: enhanced_test,
            "lightgbm": np.load(ROOT / "stack_test_lightgbm.npy"),
            "xgboost": np.load(ROOT / "stack_test_xgboost.npy"),
        }
    )
    prediction = fit_final(oof, test_predictions, uid_test, selected)
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    if not np.array_equal(sample["TransactionID"], raw_test_ids):
        raise ValueError("Prepared test and sample TransactionID order differ")
    if not np.array_equal(sample["TransactionID"], raw_test["TransactionID"]):
        raise ValueError("Raw test and sample TransactionID order differ")
    output = sample[["TransactionID"]].copy()
    output[TARGET] = prediction
    output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": "official train/test only; no omitted gap and no test labels",
        "selection": "official-train temporal OOF only",
        "sources": list(SOURCES),
        "advanced_weight": ADVANCED_WEIGHT,
        "seeds": list(SEEDS),
        "iterations": FINAL_ITERATIONS,
        "seed_models": seed_rows,
        "features": len(features),
        "categorical": len(categorical),
        "baseline_holdout_auc": baseline["uid_recipe"]["auc"],
        "selected_holdout_auc": selected["uid_recipe"]["auc"],
        "holdout_gain": holdout_gain,
        "selected_recipe": selected,
        "output": OUTPUT_PATH.name,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

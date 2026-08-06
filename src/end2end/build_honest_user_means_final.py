"""Build the locked conservative user-means CatBoost candidate."""

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

from build_honest_no_gap_meta import evaluate_source_set, fit_final, make_uid, rank_prediction
from fraud_honest_advanced_data import ADVANCED_CAT_PARAMS as CAT_PARAMS
from fraud_vblock_features import add_vblock_user_features
from train_honest_advanced_catboost import CACHE_DIR as ADVANCED_DIR
from train_honest_advanced_catboost import SOURCE as ADVANCED_SOURCE
from train_honest_advanced_catboost import prepare
from train_honest_user_means_catboost import CACHE_DIR, SOURCE


ROOT = Path(__file__).resolve().parent
TARGET = "isFraud"
CAT_ENHANCED = "cat_enhanced"
CAT_MEANS = "cat_means_05"
SOURCES = ("catboost", CAT_ENHANCED, CAT_MEANS, "lightgbm", "xgboost")
ADVANCED_WEIGHT = 0.25
MEANS_WEIGHT = 0.05
DEFAULT_SEEDS = (1729,)
ITERATIONS = 1_900
OUTPUT_PATH = ROOT / "submission_honest_user_means.csv"
MULTISEED_OUTPUT_PATH = ROOT / "submission_honest_user_means_multiseed.csv"
REPORT_PATH = CACHE_DIR / "final_report.json"
MULTISEED_REPORT_PATH = CACHE_DIR / "final_multiseed_report.json"


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("Provide at least one seed")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    started = time.time()

    prepared, features, categorical = prepare()
    (
        prepared["train"],
        prepared["inference"],
        vblock_features,
        _,
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
    model_started = time.time()
    means_ranks = []
    model_rows = []
    for seed in args.seeds:
        model_path = CACHE_DIR / f"final_seed_{seed}.cbm"
        prediction_path = CACHE_DIR / f"final_seed_{seed}_test.npy"
        seed_started = time.time()
        if model_path.exists() and prediction_path.exists() and not args.force:
            seed_prediction = np.load(prediction_path)
            if seed_prediction.dtype != np.float64:
                print(
                    f"Refreshing final seed {seed} predictions as float64",
                    flush=True,
                )
                model = CatBoostClassifier()
                model.load_model(model_path)
                seed_prediction = model.predict_proba(
                    prepared["inference"][features]
                )[:, 1]
                np.save(prediction_path, seed_prediction)
                del model
                gc.collect()
            model_cached = True
        else:
            print(
                f"Final user-means CatBoost: seed={seed}, trees={ITERATIONS}, "
                f"features={len(features)}",
                flush=True,
            )
            model = CatBoostClassifier(
                **{
                    **CAT_PARAMS,
                    "iterations": ITERATIONS,
                    "random_seed": seed,
                    "verbose": 200,
                }
            )
            model.fit(
                prepared["train"][features],
                prepared["y"],
                cat_features=categorical,
            )
            seed_prediction = model.predict_proba(
                prepared["inference"][features]
            )[:, 1]
            model.save_model(model_path)
            np.save(prediction_path, seed_prediction)
            del model
            gc.collect()
            model_cached = False
        means_ranks.append(rank_prediction(seed_prediction))
        model_rows.append(
            {
                "seed": seed,
                "cached": model_cached,
                "minutes": (time.time() - seed_started) / 60.0,
                "model": model_path.name,
            }
        )
    means_test = np.mean(means_ranks, axis=0)
    test_ids = prepared["inference_ids"].to_numpy()
    del prepared
    gc.collect()

    base_oof = pd.read_csv(ROOT / "boost3_oof_predictions.csv")
    advanced_oof = pd.read_csv(
        ADVANCED_DIR / "oof.csv", usecols=["row_index", ADVANCED_SOURCE]
    )
    means_oof = pd.concat(
        [
            pd.read_csv(
                CACHE_DIR / f"fold_{fold}_oof.csv",
                usecols=["row_index", SOURCE],
            )
            for fold in range(3)
        ],
        ignore_index=True,
    )
    oof = base_oof.merge(
        advanced_oof, on="row_index", validate="one_to_one"
    ).merge(means_oof, on="row_index", validate="one_to_one")
    old_oof_rank = oof.groupby("fold")["catboost"].rank(pct=True)
    advanced_oof_rank = oof.groupby("fold")[ADVANCED_SOURCE].rank(pct=True)
    means_oof_rank = oof.groupby("fold")[SOURCE].rank(pct=True)
    oof[CAT_ENHANCED] = (
        (1.0 - ADVANCED_WEIGHT) * old_oof_rank
        + ADVANCED_WEIGHT * advanced_oof_rank
    )
    oof[CAT_MEANS] = (
        (1.0 - MEANS_WEIGHT) * old_oof_rank
        + MEANS_WEIGHT * means_oof_rank
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
    winner_sources = ("catboost", CAT_ENHANCED, "lightgbm", "xgboost")
    baseline = evaluate_source_set(oof, uid_train, winner_sources)
    selected = evaluate_source_set(oof, uid_train, SOURCES)
    holdout_gain = float(
        selected["uid_recipe"]["auc"] - baseline["uid_recipe"]["auc"]
    )
    if holdout_gain <= 0:
        raise RuntimeError(f"User-means lock gain is not positive: {holdout_gain}")

    old_cat_test = np.load(ROOT / "stack_test_catboost.npy")
    # This branch was selected after exporting the advanced source as float32.
    # Keep that rank contract explicit so fresh and cached runs are identical.
    advanced_ranks = np.mean(
        [
            rank_prediction(
                np.load(
                    ADVANCED_DIR / f"final_seed_{seed}_test.npy"
                ).astype("float32")
            )
            for seed in (1729, 2026, 3407)
        ],
        axis=0,
    )
    old_rank = rank_prediction(old_cat_test)
    test_predictions = pd.DataFrame(
        {
            "catboost": old_cat_test,
            CAT_ENHANCED: (
                (1.0 - ADVANCED_WEIGHT) * old_rank
                + ADVANCED_WEIGHT * advanced_ranks
            ),
            CAT_MEANS: (
                (1.0 - MEANS_WEIGHT) * old_rank
                + MEANS_WEIGHT * rank_prediction(means_test)
            ),
            "lightgbm": np.load(ROOT / "stack_test_lightgbm.npy"),
            "xgboost": np.load(ROOT / "stack_test_xgboost.npy"),
        }
    )
    prediction = fit_final(oof, test_predictions, uid_test, selected)
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    if not np.array_equal(sample["TransactionID"], test_ids):
        raise ValueError("Prepared test and sample TransactionID order differ")
    if not np.array_equal(sample["TransactionID"], raw_test["TransactionID"]):
        raise ValueError("Raw test and sample TransactionID order differ")
    output = sample[["TransactionID"]].copy()
    output[TARGET] = prediction
    output_path = OUTPUT_PATH if len(args.seeds) == 1 else MULTISEED_OUTPUT_PATH
    report_path = REPORT_PATH if len(args.seeds) == 1 else MULTISEED_REPORT_PATH
    output.to_csv(output_path, index=False)

    report = {
        "data_policy": "official train/test only",
        "selection": "official-train temporal OOF only",
        "sources": list(SOURCES),
        "advanced_weight": ADVANCED_WEIGHT,
        "means_weight": MEANS_WEIGHT,
        "seeds": list(args.seeds),
        "iterations": ITERATIONS,
        "models": model_rows,
        "model_minutes": (time.time() - model_started) / 60.0,
        "features": len(features),
        "user_mean_features": mean_features,
        "baseline_holdout_auc": baseline["uid_recipe"]["auc"],
        "selected_holdout_auc": selected["uid_recipe"]["auc"],
        "holdout_gain": holdout_gain,
        "selected_recipe": selected,
        "output": output_path.name,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

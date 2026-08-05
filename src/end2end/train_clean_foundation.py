"""Train clean CatBoost and LightGBM foundation models from provided data only."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_features import TARGET, build_features, read_and_merge


ROOT = Path(__file__).resolve().parent
SEED = 42

CAT_PARAMS = {
    "iterations": 1400,
    "depth": 8,
    "learning_rate": 0.06,
    "l2_leaf_reg": 8,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.80,
    "rsm": 0.90,
    "border_count": 128,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": SEED,
    "one_hot_max_size": 10,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": 100,
}

LGB_PARAMS = {
    "n_estimators": 1800,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "min_child_samples": 40,
    "subsample": 0.70,
    "subsample_freq": 1,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "max_bin": 255,
    "max_depth": -1,
    "extra_trees": True,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}


def main() -> None:
    started = time.time()
    print("Reading provided train/test only...", flush=True)
    train = read_and_merge(ROOT, "train")
    test = read_and_merge(ROOT, "test")
    if TARGET in test:
        raise AssertionError("Competition test unexpectedly contains target")

    print("Building target-free train+test features...", flush=True)
    train, test, features, categorical = build_features(
        train,
        test,
        giba_features=True,
    )
    y = train.pop(TARGET).astype("int8")
    train.drop(columns="TransactionID", inplace=True)
    test.drop(columns="TransactionID", inplace=True)
    transaction_dt = train["TransactionDT"].copy()
    cutoff = transaction_dt.quantile(0.80)
    train_index = train.index[transaction_dt < cutoff]
    valid_index = train.index[transaction_dt >= cutoff]

    print("Selecting CatBoost iterations on the final 20% of train...", flush=True)
    cat_validation = CatBoostClassifier(**CAT_PARAMS)
    cat_validation.fit(
        train.loc[train_index, features],
        y.loc[train_index],
        cat_features=categorical,
        eval_set=(train.loc[valid_index, features], y.loc[valid_index]),
        early_stopping_rounds=120,
        use_best_model=True,
    )
    cat_valid_prediction = cat_validation.predict_proba(
        train.loc[valid_index, features]
    )[:, 1]
    cat_auc = float(roc_auc_score(y.loc[valid_index], cat_valid_prediction))
    cat_iterations = int(cat_validation.get_best_iteration() + 1)
    cat_validation.save_model(ROOT / "catboost_giba_validation.cbm")

    print(f"Training full CatBoost with {cat_iterations} iterations...", flush=True)
    cat_final = CatBoostClassifier(
        **{**CAT_PARAMS, "iterations": cat_iterations, "verbose": 100}
    )
    cat_final.fit(train[features], y, cat_features=categorical)
    cat_final.save_model(ROOT / "catboost_giba_final.cbm")
    del cat_validation, cat_final, cat_valid_prediction
    gc.collect()

    print("Converting categories and selecting LightGBM iterations...", flush=True)
    for column in categorical:
        categories = pd.Index(train[column].dropna().unique())
        dtype = pd.CategoricalDtype(categories=categories)
        train[column] = train[column].astype(dtype)
        test[column] = test[column].astype(dtype)

    lgb_validation = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_validation.fit(
        train.loc[train_index, features],
        y.loc[train_index],
        categorical_feature=categorical,
        eval_set=[(train.loc[valid_index, features], y.loc[valid_index])],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(120, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    lgb_valid_prediction = lgb_validation.predict_proba(
        train.loc[valid_index, features],
        num_iteration=lgb_validation.best_iteration_,
    )[:, 1]
    lgb_auc = float(roc_auc_score(y.loc[valid_index], lgb_valid_prediction))
    lgb_iterations = int(lgb_validation.best_iteration_)
    lgb_validation.booster_.save_model(
        ROOT / "lightgbm_giba_validation.txt",
        num_iteration=lgb_iterations,
    )

    print(f"Training full LightGBM with {lgb_iterations} iterations...", flush=True)
    lgb_final = lgb.LGBMClassifier(
        **{
            **LGB_PARAMS,
            "n_estimators": lgb_iterations,
            "random_state": 2026,
        }
    )
    lgb_final.fit(
        train[features],
        y,
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(0)],
    )
    lgb_final.booster_.save_model(ROOT / "lightgbm_giba_final.txt")

    metrics = {
        "data_policy": "provided train/test only",
        "external_gap_used": False,
        "competition_test_labels_used": False,
        "train_rows": len(train),
        "test_rows": len(test),
        "features": len(features),
        "categorical": len(categorical),
        "holdout": {
            "policy": "last 20% by TransactionDT",
            "train_rows": len(train_index),
            "validation_rows": len(valid_index),
            "catboost_auc": cat_auc,
            "catboost_iterations": cat_iterations,
            "lightgbm_auc": lgb_auc,
            "lightgbm_iterations": lgb_iterations,
        },
        "models": [
            "catboost_giba_validation.cbm",
            "catboost_giba_final.cbm",
            "lightgbm_giba_validation.txt",
            "lightgbm_giba_final.txt",
        ],
        "elapsed_minutes": (time.time() - started) / 60,
    }
    (ROOT / "clean_foundation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Clean official-data preparation shared by the honest CatBoost views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_advanced_user_features import add_advanced_user_features
from fraud_features import TARGET, build_features, read_and_merge
from fraud_user_features import add_target_free_user_profile_features


ADVANCED_CAT_PARAMS = {
    "iterations": 2_100,
    "depth": 8,
    "learning_rate": 0.055,
    "l2_leaf_reg": 9,
    "random_strength": 0.45,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.82,
    "rsm": 0.90,
    "border_count": 128,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "one_hot_max_size": 10,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": 100,
}


def add_amount_patterns(frame: pd.DataFrame) -> list[str]:
    amount = frame["TransactionAmt"].astype("float64")
    values: dict[str, np.ndarray] = {}
    for modulus in (50.0, 100.0, 200.0):
        suffix = int(modulus)
        remainder = np.mod(amount, modulus)
        distance = np.minimum(remainder, modulus - remainder)
        values[f"TransactionAmt_mod_{suffix}_is_zero"] = np.isclose(
            remainder, 0.0, atol=0.011
        ).astype("int8")
        values[f"TransactionAmt_mod_{suffix}_distance"] = distance.astype(
            "float32"
        )
    names = list(values)
    frame[names] = pd.DataFrame(values, index=frame.index)
    return names


def prepare_advanced_catboost_data(data_dir: Path) -> tuple[dict, list[str], list[str]]:
    data_dir = Path(data_dir)
    print("Reading official train/test...", flush=True)
    train = read_and_merge(data_dir, "train")
    inference = read_and_merge(data_dir, "test")
    if TARGET in inference:
        raise AssertionError("Competition test unexpectedly contains target")
    y = train[TARGET].astype("int8").reset_index(drop=True)

    print("Building Giba and target-free graph profiles...", flush=True)
    train, inference, base_features, base_categorical = build_features(
        train,
        inference,
        giba_features=True,
    )
    (
        train,
        inference,
        profile_features,
        _,
        _,
        graph_stats,
    ) = add_target_free_user_profile_features(train, inference)

    print("Building advanced UID, rolling and behavior features...", flush=True)
    (
        train,
        inference,
        advanced_features,
        advanced_categorical,
        train_components,
        inference_components,
        advanced_stats,
    ) = add_advanced_user_features(train, inference)
    train_amount = add_amount_patterns(train)
    test_amount = add_amount_patterns(inference)
    if train_amount != test_amount:
        raise ValueError("Train/test amount feature names differ")

    features = list(
        dict.fromkeys(
            [
                *base_features,
                *profile_features,
                *advanced_features,
                *train_amount,
            ]
        )
    )
    categorical = list(
        dict.fromkeys([*base_categorical, *advanced_categorical])
    )
    train_ids = train.pop("TransactionID").reset_index(drop=True)
    inference_ids = inference.pop("TransactionID").reset_index(drop=True)
    train.pop(TARGET)

    # Match the original LightGBM-compatible preparation before converting
    # categories to CatBoost-safe strings. Unseen test categories become missing.
    for column in categorical:
        categories = pd.Index(train[column].dropna().unique())
        dtype = pd.CategoricalDtype(categories=categories)
        train[column] = train[column].astype(dtype).astype("string").fillna(
            "<MISSING>"
        )
        inference[column] = (
            inference[column]
            .astype(dtype)
            .astype("string")
            .fillna("<MISSING>")
        )

    prepared = {
        "train": train,
        "inference": inference,
        "y": y,
        "train_ids": train_ids,
        "inference_ids": inference_ids,
        "train_components": train_components,
        "inference_components": inference_components,
        "graph_stats": graph_stats,
        "advanced_stats": advanced_stats,
    }
    print(
        f"Advanced CatBoost data: {len(features)} features, "
        f"{len(categorical)} categorical",
        flush=True,
    )
    return prepared, features, categorical

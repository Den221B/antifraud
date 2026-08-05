"""Selected target-free feature views for the honest temporal stack."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_advanced_user_features import add_advanced_user_features
from fraud_features import TARGET, build_features, read_and_merge
from fraud_multicounter_features import add_multi_counter_features
from fraud_next_features import (
    add_behavior_distribution_features,
    add_calendar_amount_features,
    add_identity_features,
)
from fraud_temporal_validation import make_four_long_gap_folds
from fraud_transaction_chain_features import add_v307_transaction_chain_features
from fraud_user_features import (
    add_target_free_user_profile_features,
    is_raw_uid_feature,
)
from fraud_vblock_features import add_vblock_user_features


ROOT = Path(__file__).resolve().parent
ADVANCED_CACHE_DIR = ROOT / "advanced_feature_ablation_models"
NEXT_CACHE_DIR = ROOT / "next_feature_ablation_models_v1"
ADVANCED_OOF_PATH = ROOT / "advanced_feature_ablation_oof.csv"
NEXT_OOF_PATH = ROOT / "next_feature_ablation_oof.csv"

LGB_PARAMS = {
    "n_estimators": 750,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 47,
    "learning_rate": 0.035,
    "min_child_samples": 90,
    "subsample": 0.78,
    "subsample_freq": 1,
    "colsample_bytree": 0.78,
    "reg_alpha": 0.75,
    "reg_lambda": 10.0,
    "max_bin": 255,
    "max_depth": -1,
    "extra_trees": True,
    "random_state": 8203,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}


def unique(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(column for group in groups for column in group))


def convert_categories(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        categories = pd.Index(train[column].dropna().unique())
        dtype = pd.CategoricalDtype(categories=categories)
        train[column] = train[column].astype(dtype)
        inference[column] = inference[column].astype(dtype)


def convert_string_categories(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        categories = pd.Index(train[column].dropna().astype("string").unique())
        dtype = pd.CategoricalDtype(categories=categories)
        train[column] = train[column].astype("string").astype(dtype)
        inference[column] = inference[column].astype("string").astype(dtype)


def add_amount_patterns(frame: pd.DataFrame) -> list[str]:
    amount = frame["TransactionAmt"].astype("float64")
    values: dict[str, np.ndarray] = {}
    for modulus in (50.0, 100.0, 200.0):
        suffix = int(modulus)
        remainder = np.mod(amount, modulus)
        values[f"TransactionAmt_mod_{suffix}_is_zero"] = np.isclose(
            remainder, 0.0, atol=0.011
        ).astype("int8")
        values[f"TransactionAmt_mod_{suffix}_distance"] = np.minimum(
            remainder, modulus - remainder
        ).astype("float32")
    names = list(values)
    frame[names] = pd.DataFrame(values, index=frame.index)
    return names


def prepare_base() -> dict:
    print("Reading official train/test...", flush=True)
    train = read_and_merge(ROOT, "train")
    inference = read_and_merge(ROOT, "test")
    if TARGET not in train or TARGET in inference:
        raise RuntimeError("Unexpected target placement in official files")
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
        train_components,
        inference_components,
        graph_stats,
    ) = add_target_free_user_profile_features(train, inference)

    print("Building improved UID, rolling and behavior features...", flush=True)
    (
        train,
        inference,
        advanced_features,
        advanced_categorical,
        train_components,
        inference_components,
        advanced_stats,
    ) = add_advanced_user_features(train, inference)

    features = unique(base_features, profile_features, advanced_features)
    categorical = unique(base_categorical, advanced_categorical)
    train_ids = train.pop("TransactionID").reset_index(drop=True)
    inference_ids = inference.pop("TransactionID").reset_index(drop=True)
    train.pop(TARGET)
    convert_categories(train, inference, categorical)

    advanced_set = set(advanced_features)
    profile_set = set(profile_features)
    dynamics_features = []
    for column in features:
        raw_v = re.fullmatch(r"V\d+", column) is not None
        raw_id = re.fullmatch(r"id_\d+", column) is not None
        if column in advanced_set or column in profile_set:
            dynamics_features.append(column)
        elif not raw_v and not raw_id and not is_raw_uid_feature(column):
            dynamics_features.append(column)

    return {
        "train": train,
        "inference": inference,
        "y": y,
        "train_ids": train_ids,
        "inference_ids": inference_ids,
        "features": features,
        "dynamics_features": dynamics_features,
        "categorical": categorical,
        "train_components": train_components,
        "inference_components": inference_components,
        "graph_stats": graph_stats,
        "advanced_stats": advanced_stats,
    }


def prepare_advanced_views() -> tuple[dict, dict[str, list[str]], dict]:
    prepared = prepare_base()
    train_amount = add_amount_patterns(prepared["train"])
    test_amount = add_amount_patterns(prepared["inference"])
    if train_amount != test_amount:
        raise RuntimeError("Train/test amount feature names differ")

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
    dynamics = prepared["dynamics_features"]
    wide_cd = [
        column
        for column in vblock_features
        if column.startswith("wide_user_C") or column.startswith("wide_user_D")
    ]
    wide_v = [
        column for column in vblock_features if column.startswith("wide_user_V")
    ]
    missing_profiles = [
        column
        for column in vblock_features
        if column.startswith("wide_user_vblock_")
    ]
    views = {
        "vblock_dynamics": unique(dynamics, vblock_features, train_amount),
        "vblock_cd_dynamics": unique(dynamics, wide_cd, train_amount),
        "vblock_aggregates_dynamics": unique(
            dynamics,
            wide_cd,
            wide_v,
            missing_profiles,
            train_amount,
        ),
    }
    metadata = {
        "view_features": {name: len(columns) for name, columns in views.items()},
        "amount_features": train_amount,
        "vblock": vblock_stats,
    }
    return prepared, views, metadata


def prepare_next_views() -> tuple[dict, dict[str, dict], dict]:
    prepared, advanced_views, advanced_metadata = prepare_advanced_views()
    dynamics = prepared["dynamics_features"]
    vblock = advanced_views["vblock_dynamics"]

    prepared["train"].insert(
        0, "TransactionID", prepared["train_ids"].to_numpy()
    )
    prepared["inference"].insert(
        0, "TransactionID", prepared["inference_ids"].to_numpy()
    )

    print("Building target-free transaction chains...", flush=True)
    (
        prepared["train"],
        prepared["inference"],
        v307_features,
        v307_categorical,
        v307_stats,
    ) = add_v307_transaction_chain_features(
        prepared["train"],
        prepared["inference"],
        prepared["y"],
        include_target_history=False,
        max_chain_size=100,
    )
    (
        prepared["train"],
        prepared["inference"],
        multi_features,
        multi_categorical,
        multi_stats,
    ) = add_multi_counter_features(
        prepared["train"],
        prepared["inference"],
        max_chain_size=100,
    )
    chain_features = unique(v307_features, multi_features)

    (
        prepared["train"],
        prepared["inference"],
        calendar_features,
        calendar_stats,
    ) = add_calendar_amount_features(
        prepared["train"], prepared["inference"]
    )
    (
        prepared["train"],
        prepared["inference"],
        identity_features,
        identity_categorical,
        identity_stats,
    ) = add_identity_features(prepared["train"], prepared["inference"])
    (
        prepared["train"],
        prepared["inference"],
        behavior_features,
        behavior_stats,
    ) = add_behavior_distribution_features(
        prepared["train"],
        prepared["inference"],
        prepared["train_components"],
        prepared["inference_components"],
    )

    new_categorical = unique(
        v307_categorical,
        multi_categorical,
        identity_categorical,
    )
    convert_string_categories(
        prepared["train"], prepared["inference"], new_categorical
    )
    prepared["categorical"] = unique(
        prepared["categorical"], new_categorical
    )

    structured = unique(
        calendar_features,
        identity_features,
        behavior_features,
    )
    d_origin = [column for column in calendar_features if column.startswith("D")]
    c_transforms = [
        column for column in calendar_features if column.startswith("C")
    ]
    amount = [
        column for column in calendar_features if column.startswith("amount_")
    ]
    specs = {
        "vblock_d_amount": {
            "features": unique(vblock, d_origin, c_transforms, amount),
            "bayes": False,
            "candidate": True,
        },
        "vblock_structured": {
            "features": unique(vblock, structured),
            "bayes": False,
            "candidate": True,
        },
        "vblock_chains": {
            "features": unique(vblock, chain_features),
            "bayes": False,
            "candidate": True,
        },
    }
    metadata = {
        "view_features": {
            name: len(spec["features"]) for name, spec in specs.items()
        },
        "previous_vblock": advanced_metadata,
        "v307_chain": v307_stats,
        "multi_counter": multi_stats,
        "calendar_amount": calendar_stats,
        "identity": identity_stats,
        "behavior": behavior_stats,
    }
    return prepared, specs, metadata


def train_oof(
    prepared: dict,
    views: dict[str, list[str] | dict],
    cache_dir: Path,
    output_path: Path,
    seed: int,
    force: bool,
) -> tuple[pd.DataFrame, dict]:
    cache_dir.mkdir(exist_ok=True)
    folds = make_four_long_gap_folds(prepared["train"]["TransactionDT"])
    y = prepared["y"].to_numpy(dtype="int8")
    valid_rows = np.unique(
        np.concatenate([fold.validation_index for fold in folds])
    )
    fold_by_row = np.full(len(y), -1, dtype="int8")
    for fold in folds:
        fold_by_row[fold.validation_index] = fold.number
    oof = pd.DataFrame(
        {
            "row_index": valid_rows,
            "TransactionID": prepared["train_ids"].iloc[valid_rows].to_numpy(),
            TARGET: y[valid_rows],
            "fold": fold_by_row[valid_rows],
        }
    )
    metrics = {}

    for view_number, (name, specification) in enumerate(views.items(), start=1):
        features = (
            specification["features"]
            if isinstance(specification, dict)
            else specification
        )
        categorical = [
            column for column in prepared["categorical"] if column in features
        ]
        prediction_all = np.full(len(y), np.nan, dtype="float64")
        fold_rows = []
        print(
            f"View {view_number}/{len(views)}: {name} ({len(features)} features)",
            flush=True,
        )
        for fold in folds:
            prediction_path = cache_dir / f"{name}_fold_{fold.number}_prediction.npy"
            model_path = cache_dir / f"{name}_fold_{fold.number}.txt"
            started = time.time()
            if prediction_path.exists() and model_path.exists() and not force:
                prediction = np.load(prediction_path)
                reused = True
            else:
                model = lgb.LGBMClassifier(
                    **{**LGB_PARAMS, "random_state": seed + fold.number}
                )
                model.fit(
                    prepared["train"].iloc[fold.train_index][features],
                    y[fold.train_index],
                    categorical_feature=categorical,
                    callbacks=[lgb.log_evaluation(0)],
                )
                prediction = model.predict_proba(
                    prepared["train"].iloc[fold.validation_index][features]
                )[:, 1]
                model.booster_.save_model(model_path)
                np.save(prediction_path, prediction)
                del model
                gc.collect()
                reused = False
            if len(prediction) != len(fold.validation_index):
                raise RuntimeError(f"Cached {name} fold {fold.number} is misaligned")
            prediction_all[fold.validation_index] = prediction
            fold_rows.append(
                {
                    "fold": fold.number,
                    "train_rows": int(len(fold.train_index)),
                    "validation_rows": int(len(fold.validation_index)),
                    "auc": float(
                        roc_auc_score(y[fold.validation_index], prediction)
                    ),
                    "reused": reused,
                    "minutes": (time.time() - started) / 60.0,
                }
            )
            print(json.dumps(fold_rows[-1]), flush=True)
        oof[name] = prediction_all[valid_rows]
        metrics[name] = {
            "features": len(features),
            "categorical": len(categorical),
            "folds": fold_rows,
        }
        oof.to_csv(output_path, index=False)
    return oof, metrics


def train_final(
    prepared: dict,
    views: dict[str, list[str] | dict],
    names: tuple[str, ...],
    cache_dir: Path,
    seed: int,
    force: bool,
) -> dict[str, str]:
    outputs = {}
    for name in names:
        specification = views[name]
        features = (
            specification["features"]
            if isinstance(specification, dict)
            else specification
        )
        categorical = [
            column for column in prepared["categorical"] if column in features
        ]
        model_path = cache_dir / f"{name}_final.txt"
        if model_path.exists() and not force:
            print(f"Reusing final {name}", flush=True)
        else:
            print(f"Training final {name} ({len(features)} features)", flush=True)
            model = lgb.LGBMClassifier(
                **{**LGB_PARAMS, "random_state": seed}
            )
            model.fit(
                prepared["train"][features],
                prepared["y"],
                categorical_feature=categorical,
                callbacks=[lgb.log_evaluation(0)],
            )
            model.booster_.save_model(model_path)
            del model
            gc.collect()
        outputs[name] = str(model_path)
    return outputs

"""Train a client-aware LightGBM stack on the locked honest model sources.

Model and post-processing choices are selected on temporal fold 1. Fold 2 is
evaluated once as the untouched train-only lock. Competition-test labels are
never loaded.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import build_honest_no_gap_meta as meta
from build_honest_user_means_final import (
    ADVANCED_WEIGHT,
    CAT_ENHANCED,
    CAT_MEANS,
    MEANS_WEIGHT,
    SOURCES,
)
from train_honest_advanced_catboost import SOURCE as ADVANCED_SOURCE
from train_honest_user_means_catboost import SOURCE as MEANS_SOURCE


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_client_meta"
TARGET = "isFraud"
DAY_SECONDS = 86_400.0
TRAIN_END_BY_FOLD = {0: 30.0, 1: 45.0, 2: 60.0}
META_DEV_FOLD = 1
META_LOCK_FOLD = 2
GENERIC_EMAILS = {"anonymous.com", "mail.com", "<MISSING>"}
OUTPUT_PATH = ROOT / "submission_honest_client_meta.csv"
REPORT_PATH = WORK_DIR / "report.json"

RAW_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "D1",
    "P_emaildomain",
)
GROUP_SPECS = {
    "strict_clean_email": (
        ("card1", "addr1", "origin_day", "clean_email"),
        500,
    ),
    "card_addr_origin": (("card1", "addr1", "origin_day"), 500),
    "card_origin_clean_email": (
        ("card1", "origin_day", "clean_email"),
        250,
    ),
    "card_full_origin": (
        ("card1", "card2", "card3", "card5", "origin_day"),
        250,
    ),
    "card_addr_origin_product": (
        ("card1", "addr1", "origin_day", "ProductCD"),
        500,
    ),
    "origin_clean_email": (("origin_day", "clean_email"), 150),
}
META_VIEWS = {
    "cat_xgb": ("catboost", CAT_ENHANCED, "xgboost"),
    "all_sources": SOURCES,
    "all_sources_clients": SOURCES,
}
LGB_CONFIGS = (
    {"num_leaves": 7, "max_depth": 3, "min_child_samples": 300},
    {"num_leaves": 15, "max_depth": 4, "min_child_samples": 300},
    {"num_leaves": 15, "max_depth": 5, "min_child_samples": 600},
    {"num_leaves": 31, "max_depth": 5, "min_child_samples": 600},
    {"num_leaves": 31, "max_depth": 6, "min_child_samples": 1_000},
    {"num_leaves": 7, "max_depth": 4, "min_child_samples": 1_000},
)
LGB_BASE_PARAMS = {
    "n_estimators": 1_500,
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "random_state": 5309,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
PP_METHODS = ("mean", "q75", "max")
PP_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)


def rank(values: np.ndarray | pd.Series) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(
        method="average", pct=True
    ).to_numpy()


def prepare_raw(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, list(RAW_COLUMNS)].copy().reset_index(drop=True)
    result["origin_day"] = (
        result["TransactionDT"] / DAY_SECONDS - result["D1"]
    ).round()
    email = result["P_emaildomain"].astype("string").str.lower()
    result["clean_email"] = email.mask(email.isin(GENERIC_EMAILS))
    return result


def group_hash(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, list(columns)]
    valid = values.notna().all(axis=1).to_numpy(dtype=bool)
    hashed = pd.util.hash_pandas_object(
        values,
        index=False,
        categorize=True,
    ).to_numpy(dtype="uint64", copy=False)
    return hashed, valid


def membership_features(
    history: pd.DataFrame,
    query: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, int]]]:
    values: dict[str, np.ndarray] = {}
    groups: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    seen_columns = []
    for name, (columns, max_size) in GROUP_SPECS.items():
        history_group, history_valid = group_hash(history, columns)
        query_group, query_valid = group_hash(query, columns)
        history_count = pd.Series(history_group[history_valid]).value_counts(
            sort=False
        )
        query_count = pd.Series(query_group[query_valid]).value_counts(
            sort=False
        )
        mapped_history = np.zeros(len(query), dtype="float64")
        mapped_query = np.zeros(len(query), dtype="float64")
        if query_valid.any():
            mapped_history[query_valid] = (
                pd.Series(query_group[query_valid])
                .map(history_count)
                .fillna(0)
                .to_numpy(dtype="float64")
            )
            mapped_query[query_valid] = (
                pd.Series(query_group[query_valid])
                .map(query_count)
                .fillna(0)
                .to_numpy(dtype="float64")
            )
        seen = query_valid & (mapped_history > 0)
        values[f"client_{name}_valid"] = query_valid.astype("int8")
        values[f"client_{name}_seen"] = seen.astype("int8")
        values[f"client_{name}_history_count"] = np.log1p(
            mapped_history
        ).astype("float32")
        values[f"client_{name}_query_count"] = np.log1p(
            mapped_query
        ).astype("float32")
        seen_columns.append(f"client_{name}_seen")
        groups[name] = (query_group, query_valid, max_size)

    strict = values["client_strict_clean_email_seen"].astype(bool)
    broad = np.maximum.reduce(
        [
            values["client_card_addr_origin_seen"],
            values["client_card_origin_clean_email_seen"],
            values["client_card_full_origin_seen"],
            values["client_card_addr_origin_product_seen"],
            values["client_origin_clean_email_seen"],
        ]
    ).astype(bool)
    partial = ~strict & broad
    cold = ~strict & ~broad
    values["client_segment_strict"] = strict.astype("int8")
    values["client_segment_partial"] = partial.astype("int8")
    values["client_segment_cold"] = cold.astype("int8")
    values["client_seen_key_count"] = np.column_stack(
        [values[column] for column in seen_columns]
    ).sum(axis=1).astype("int8")
    return pd.DataFrame(values), groups


def build_oof_sources() -> pd.DataFrame:
    base = pd.read_csv(ROOT / "boost3_oof_predictions.csv")
    advanced = pd.read_csv(
        ROOT / "honest_advanced_catboost/oof.csv",
        usecols=["row_index", ADVANCED_SOURCE],
    )
    means = pd.concat(
        [
            pd.read_csv(
                ROOT / f"honest_user_means_catboost/fold_{fold}_oof.csv",
                usecols=["row_index", MEANS_SOURCE],
            )
            for fold in range(3)
        ],
        ignore_index=True,
    )
    oof = base.merge(
        advanced, on="row_index", validate="one_to_one"
    ).merge(means, on="row_index", validate="one_to_one")
    old_rank = oof.groupby("fold")["catboost"].rank(pct=True)
    oof[CAT_ENHANCED] = (
        (1.0 - ADVANCED_WEIGHT) * old_rank
        + ADVANCED_WEIGHT
        * oof.groupby("fold")[ADVANCED_SOURCE].rank(pct=True)
    )
    oof[CAT_MEANS] = (
        (1.0 - MEANS_WEIGHT) * old_rank
        + MEANS_WEIGHT * oof.groupby("fold")[MEANS_SOURCE].rank(pct=True)
    )
    return oof.reset_index(drop=True)


def build_test_sources() -> pd.DataFrame:
    old_cat = np.load(ROOT / "stack_test_catboost.npy")
    old_rank = rank(old_cat)
    advanced_rank = np.mean(
        [
            rank(
                np.load(
                    ROOT
                    / f"honest_advanced_catboost/final_seed_{seed}_test.npy"
                ).astype("float32")
            )
            for seed in (1729, 2026, 3407)
        ],
        axis=0,
    )
    means_rank = rank(
        np.load(ROOT / "honest_user_means_catboost/final_seed_1729_test.npy")
    )
    return pd.DataFrame(
        {
            "catboost": old_cat,
            CAT_ENHANCED: (
                (1.0 - ADVANCED_WEIGHT) * old_rank
                + ADVANCED_WEIGHT * advanced_rank
            ),
            CAT_MEANS: (
                (1.0 - MEANS_WEIGHT) * old_rank
                + MEANS_WEIGHT * means_rank
            ),
            "lightgbm": np.load(ROOT / "stack_test_lightgbm.npy"),
            "xgboost": np.load(ROOT / "stack_test_xgboost.npy"),
        }
    )


def build_client_features(
    oof: pd.DataFrame,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, dict[str, tuple[np.ndarray, np.ndarray, int]]],
    dict[str, tuple[np.ndarray, np.ndarray, int]],
]:
    parts = []
    fold_groups = {}
    day = raw_train["TransactionDT"].to_numpy(dtype="float64") / DAY_SECONDS
    for fold in sorted(TRAIN_END_BY_FOLD):
        mask = oof["fold"].eq(fold).to_numpy()
        positions = np.flatnonzero(mask)
        row_index = oof.loc[mask, "row_index"].to_numpy(dtype="int64")
        history = raw_train.loc[day < TRAIN_END_BY_FOLD[fold]].reset_index(
            drop=True
        )
        query = raw_train.iloc[row_index].reset_index(drop=True)
        features, groups = membership_features(history, query)
        features["_position"] = positions
        parts.append(features)
        fold_groups[fold] = groups
    oof_features = (
        pd.concat(parts, ignore_index=True)
        .sort_values("_position")
        .drop(columns="_position")
        .reset_index(drop=True)
    )
    test_features, test_groups = membership_features(raw_train, raw_test)
    return oof_features, test_features, fold_groups, test_groups


def make_meta_view(
    predictions: pd.DataFrame,
    client_features: pd.DataFrame,
    view: str,
    fold: pd.Series | None,
) -> pd.DataFrame:
    result = meta.build_meta_features(
        predictions,
        META_VIEWS[view],
        fold=fold,
    ).reset_index(drop=True)
    if view == "all_sources_clients":
        result = pd.concat(
            [result, client_features.reset_index(drop=True)],
            axis=1,
        )
    return result.astype("float32")


def current_meta_prediction(
    oof: pd.DataFrame,
    uid: pd.Series,
    valid_fold: int,
) -> tuple[np.ndarray, dict]:
    features = meta.build_meta_features(oof, SOURCES, fold=oof["fold"])
    train_mask = oof["fold"].lt(valid_fold)
    valid_mask = oof["fold"].eq(valid_fold)
    y_train = oof.loc[train_mask, TARGET].to_numpy(dtype="float32")
    y_valid = oof.loc[valid_mask, TARGET].to_numpy(dtype="float32")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features.loc[train_mask]).astype("float32")
    X_valid = scaler.transform(features.loc[valid_mask]).astype("float32")
    linear = LogisticRegression(
        C=meta.LOGISTIC_C,
        max_iter=2_000,
        class_weight="balanced",
        random_state=42,
    )
    linear.fit(X_train, y_train)
    linear_prediction = linear.predict_proba(X_valid)[:, 1]
    neural_parts = []
    epochs = {}
    for seed in meta.META_SEEDS:
        epoch, prediction, _ = meta.train_mlp_with_validation(
            X_train,
            y_train,
            X_valid,
            y_valid,
            seed,
        )
        epochs[str(seed)] = int(epoch)
        neural_parts.append(prediction)
    neural_prediction = np.mean(neural_parts, axis=0)
    blend = meta.best_rank_blend(
        y_valid,
        neural_prediction,
        linear_prediction,
    )
    prediction = (
        float(blend["neural_weight"]) * rank(neural_prediction)
        + float(blend["linear_weight"]) * rank(linear_prediction)
    )
    rows = oof.loc[valid_mask, "row_index"].to_numpy(dtype="int64")
    valid_uid = uid.iloc[rows].reset_index(drop=True)
    uid_rows = []
    for weight in (0.0, 0.10, 0.25, 0.50):
        processed = meta.apply_uid_max(prediction, valid_uid, weight)
        uid_rows.append(
            {
                "weight": weight,
                "auc": float(roc_auc_score(y_valid, processed)),
            }
        )
    selected_uid = max(uid_rows, key=lambda row: row["auc"])
    prediction = meta.apply_uid_max(
        prediction,
        valid_uid,
        float(selected_uid["weight"]),
    )
    report = {
        "fold": valid_fold,
        "rows": int(valid_mask.sum()),
        "blend": blend,
        "uid": selected_uid,
        "epochs": epochs,
        "auc": float(roc_auc_score(y_valid, prediction)),
    }
    return prediction, report


def search_lgb_meta(
    oof: pd.DataFrame,
    client_features: pd.DataFrame,
) -> tuple[dict, list[dict], np.ndarray]:
    train_mask = oof["fold"].lt(META_DEV_FOLD).to_numpy()
    valid_mask = oof["fold"].eq(META_DEV_FOLD).to_numpy()
    y_train = oof.loc[train_mask, TARGET].to_numpy(dtype="int8")
    y_valid = oof.loc[valid_mask, TARGET].to_numpy(dtype="int8")
    rows = []
    predictions = {}
    for view in META_VIEWS:
        features = make_meta_view(
            oof,
            client_features,
            view,
            fold=oof["fold"],
        )
        for config_index, config in enumerate(LGB_CONFIGS):
            model = lgb.LGBMClassifier(**LGB_BASE_PARAMS, **config)
            model.fit(
                features.loc[train_mask],
                y_train,
                eval_set=[(features.loc[valid_mask], y_valid)],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            prediction = model.predict_proba(
                features.loc[valid_mask]
            )[:, 1]
            row = {
                "view": view,
                "config_index": config_index,
                **config,
                "best_iteration": int(model.best_iteration_),
                "dev_auc": float(roc_auc_score(y_valid, prediction)),
                "feature_count": int(features.shape[1]),
            }
            key = (view, config_index)
            predictions[key] = prediction
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model
            gc.collect()
    rows.sort(
        key=lambda row: (
            row["dev_auc"],
            -row["num_leaves"],
            -row["feature_count"],
        ),
        reverse=True,
    )
    selected = rows[0]
    selected_prediction = predictions[
        (selected["view"], selected["config_index"])
    ]
    return selected, rows, selected_prediction


def train_lock_lgb(
    oof: pd.DataFrame,
    client_features: pd.DataFrame,
    selected: dict,
) -> tuple[lgb.LGBMClassifier, np.ndarray, int]:
    features = make_meta_view(
        oof,
        client_features,
        selected["view"],
        fold=oof["fold"],
    )
    train_mask = oof["fold"].lt(META_LOCK_FOLD).to_numpy()
    valid_mask = oof["fold"].eq(META_LOCK_FOLD).to_numpy()
    iterations = max(25, int(np.ceil(selected["best_iteration"] * 1.15)))
    config = LGB_CONFIGS[int(selected["config_index"])]
    model = lgb.LGBMClassifier(
        **{
            **LGB_BASE_PARAMS,
            **config,
            "n_estimators": iterations,
        }
    )
    model.fit(
        features.loc[train_mask],
        oof.loc[train_mask, TARGET],
        callbacks=[lgb.log_evaluation(0)],
    )
    prediction = model.predict_proba(features.loc[valid_mask])[:, 1]
    return model, prediction, iterations


def select_stack_weight(
    y: np.ndarray,
    baseline: np.ndarray,
    stacked: np.ndarray,
) -> tuple[dict, list[dict]]:
    rows = []
    for weight in np.linspace(0.0, 1.0, 21):
        prediction = (1.0 - weight) * rank(baseline) + weight * rank(stacked)
        rows.append(
            {
                "lgb_weight": float(weight),
                "auc": float(roc_auc_score(y, prediction)),
            }
        )
    return max(rows, key=lambda row: row["auc"]), rows


def aggregate_prediction(
    prediction: np.ndarray,
    group: np.ndarray,
    valid: np.ndarray,
    max_size: int,
    method: str,
) -> np.ndarray:
    output = prediction.copy()
    work = pd.DataFrame(
        {"group": group[valid], "prediction": prediction[valid]}
    )
    grouped = work.groupby("group", sort=False)["prediction"]
    size = grouped.size()
    eligible = size.index[(size >= 2) & (size <= max_size)]
    eligible_mask = work["group"].isin(eligible).to_numpy()
    if method == "mean":
        statistic = grouped.mean()
    elif method == "q75":
        statistic = grouped.quantile(0.75)
    elif method == "max":
        statistic = grouped.max()
    else:
        raise ValueError(method)
    rows = np.flatnonzero(valid)[eligible_mask]
    output[rows] = work.loc[eligible_mask, "group"].map(statistic).to_numpy()
    return output


def select_postprocess(
    y: np.ndarray,
    prediction: np.ndarray,
    dt: np.ndarray,
    groups: dict[str, tuple[np.ndarray, np.ndarray, int]],
) -> tuple[dict, list[dict]]:
    baseline_auc = float(roc_auc_score(y, prediction))
    midpoint = np.median(dt)
    halves = (dt <= midpoint, dt > midpoint)
    baseline_half = [
        float(roc_auc_score(y[mask], prediction[mask])) for mask in halves
    ]
    rows = [
        {
            "group": "none",
            "method": "none",
            "weight": 0.0,
            "auc": baseline_auc,
            "gain": 0.0,
            "half_gains": [0.0, 0.0],
            "min_half_gain": 0.0,
        }
    ]
    for name, (group, valid, max_size) in groups.items():
        for method in PP_METHODS:
            grouped = aggregate_prediction(
                prediction,
                group,
                valid,
                max_size,
                method,
            )
            for weight in PP_WEIGHTS:
                candidate = (1.0 - weight) * prediction + weight * grouped
                half_auc = [
                    float(roc_auc_score(y[mask], candidate[mask]))
                    for mask in halves
                ]
                half_gains = [
                    score - base
                    for score, base in zip(half_auc, baseline_half)
                ]
                score = float(roc_auc_score(y, candidate))
                rows.append(
                    {
                        "group": name,
                        "method": method,
                        "weight": weight,
                        "auc": score,
                        "gain": score - baseline_auc,
                        "half_gains": half_gains,
                        "min_half_gain": float(min(half_gains)),
                    }
                )
    stable = [row for row in rows if row["min_half_gain"] >= 0.0]
    selected = max(stable, key=lambda row: (row["gain"], row["min_half_gain"]))
    return selected, rows


def apply_postprocess(
    prediction: np.ndarray,
    groups: dict[str, tuple[np.ndarray, np.ndarray, int]],
    recipe: dict,
) -> np.ndarray:
    if recipe["group"] == "none":
        return prediction
    group, valid, max_size = groups[recipe["group"]]
    grouped = aggregate_prediction(
        prediction,
        group,
        valid,
        max_size,
        recipe["method"],
    )
    weight = float(recipe["weight"])
    return (1.0 - weight) * prediction + weight * grouped


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    oof = build_oof_sources()
    test_sources = build_test_sources()
    raw_train = prepare_raw(
        pd.read_csv(ROOT / "train_transaction.csv", usecols=list(RAW_COLUMNS))
    )
    raw_test = prepare_raw(
        pd.read_csv(ROOT / "test_transaction.csv", usecols=list(RAW_COLUMNS))
    )
    (
        client_oof,
        client_test,
        fold_groups,
        test_groups,
    ) = build_client_features(oof, raw_train, raw_test)
    client_oof.to_csv(WORK_DIR / "oof_client_features.csv", index=False)
    client_test.to_csv(WORK_DIR / "test_client_features.csv", index=False)

    raw_uid = pd.read_csv(
        ROOT / "train_transaction.csv",
        usecols=["TransactionDT", "card1", "addr1", "D1", "P_emaildomain"],
    )
    uid = meta.make_uid(raw_uid)
    dev_current, dev_current_report = current_meta_prediction(
        oof, uid, META_DEV_FOLD
    )
    lock_current, lock_current_report = current_meta_prediction(
        oof, uid, META_LOCK_FOLD
    )

    selected_lgb, lgb_search, dev_lgb = search_lgb_meta(oof, client_oof)
    lgb_model, lock_lgb, lock_iterations = train_lock_lgb(
        oof, client_oof, selected_lgb
    )
    lock_model_path = WORK_DIR / "lock_lgb.txt"
    lgb_model.booster_.save_model(lock_model_path)

    dev_mask = oof["fold"].eq(META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(META_LOCK_FOLD).to_numpy()
    y_dev = oof.loc[dev_mask, TARGET].to_numpy(dtype="int8")
    y_lock = oof.loc[lock_mask, TARGET].to_numpy(dtype="int8")
    selected_weight, weight_search = select_stack_weight(
        y_dev,
        dev_current,
        dev_lgb,
    )
    weight = float(selected_weight["lgb_weight"])
    dev_blend = (1.0 - weight) * rank(dev_current) + weight * rank(dev_lgb)
    lock_blend = (1.0 - weight) * rank(lock_current) + weight * rank(lock_lgb)

    dev_rows = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    dev_dt = raw_train.iloc[dev_rows]["TransactionDT"].to_numpy()
    selected_pp, pp_search = select_postprocess(
        y_dev,
        dev_blend,
        dev_dt,
        fold_groups[META_DEV_FOLD],
    )
    dev_final = apply_postprocess(
        dev_blend,
        fold_groups[META_DEV_FOLD],
        selected_pp,
    )
    lock_final = apply_postprocess(
        lock_blend,
        fold_groups[META_LOCK_FOLD],
        selected_pp,
    )
    lock_baseline_auc = float(roc_auc_score(y_lock, lock_current))
    lock_blend_auc = float(roc_auc_score(y_lock, lock_blend))
    lock_final_auc = float(roc_auc_score(y_lock, lock_final))
    accepted = bool(
        selected_weight["auc"] > dev_current_report["auc"]
        and float(roc_auc_score(y_dev, dev_final)) > dev_current_report["auc"]
        and lock_final_auc > lock_baseline_auc
    )

    final_iterations = max(25, int(np.ceil(lock_iterations * 1.15)))
    selected_view = selected_lgb["view"]
    all_features = make_meta_view(
        oof,
        client_oof,
        selected_view,
        fold=oof["fold"],
    )
    test_features = make_meta_view(
        test_sources,
        client_test,
        selected_view,
        fold=None,
    )
    config = LGB_CONFIGS[int(selected_lgb["config_index"])]
    final_model = lgb.LGBMClassifier(
        **{
            **LGB_BASE_PARAMS,
            **config,
            "n_estimators": final_iterations,
        }
    )
    final_model.fit(
        all_features,
        oof[TARGET],
        callbacks=[lgb.log_evaluation(0)],
    )
    final_model.booster_.save_model(WORK_DIR / "final_lgb.txt")
    lgb_test = final_model.predict_proba(test_features)[:, 1]
    current_test = pd.read_csv(ROOT / "submission_honest_user_means.csv")
    prediction = (
        (1.0 - weight) * rank(current_test[TARGET])
        + weight * rank(lgb_test)
    )
    prediction = apply_postprocess(prediction, test_groups, selected_pp)
    output = current_test[["TransactionID"]].copy()
    output[TARGET] = prediction
    output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": "official train/test only; train labels only in temporal OOF",
        "selection": "LGB/weight/postprocess on fold 1; fold 2 one-time lock",
        "sources": list(SOURCES),
        "client_feature_count": int(client_oof.shape[1]),
        "client_segment_counts": {
            "dev": {
                name.removeprefix("client_segment_"): int(
                    client_oof.loc[dev_mask, name].sum()
                )
                for name in (
                    "client_segment_strict",
                    "client_segment_partial",
                    "client_segment_cold",
                )
            },
            "lock": {
                name.removeprefix("client_segment_"): int(
                    client_oof.loc[lock_mask, name].sum()
                )
                for name in (
                    "client_segment_strict",
                    "client_segment_partial",
                    "client_segment_cold",
                )
            },
            "test": {
                name.removeprefix("client_segment_"): int(client_test[name].sum())
                for name in (
                    "client_segment_strict",
                    "client_segment_partial",
                    "client_segment_cold",
                )
            },
        },
        "current_meta": {
            "dev": dev_current_report,
            "lock": lock_current_report,
        },
        "selected_lgb": selected_lgb,
        "lgb_search": lgb_search,
        "lock_iterations": lock_iterations,
        "final_iterations": final_iterations,
        "selected_weight": selected_weight,
        "weight_search": weight_search,
        "selected_postprocess": selected_pp,
        "postprocess_search": pp_search,
        "scores": {
            "dev_current_auc": dev_current_report["auc"],
            "dev_lgb_auc": float(roc_auc_score(y_dev, dev_lgb)),
            "dev_blend_auc": float(roc_auc_score(y_dev, dev_blend)),
            "dev_final_auc": float(roc_auc_score(y_dev, dev_final)),
            "lock_current_auc": lock_baseline_auc,
            "lock_lgb_auc": float(roc_auc_score(y_lock, lock_lgb)),
            "lock_blend_auc": lock_blend_auc,
            "lock_final_auc": lock_final_auc,
            "lock_gain": lock_final_auc - lock_baseline_auc,
        },
        "accepted": accepted,
        "output": OUTPUT_PATH.name,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_lgb": report["selected_lgb"],
                "selected_weight": report["selected_weight"],
                "selected_postprocess": report["selected_postprocess"],
                "scores": report["scores"],
                "accepted": report["accepted"],
                "output": report["output"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

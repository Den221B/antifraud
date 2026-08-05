"""Search a heavier raw-feature meta model on nested temporal OOF only."""

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

import build_honest_no_gap_meta as meta
from fraud_features import build_features, read_and_merge
from fraud_vblock_features import TOP_V_COLUMNS
import refine_honest_client_segments as segments
import search_honest_featureview_meta as featureview
import train_honest_client_meta as client
import train_honest_heavy_temporal_client as heavy


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_raw_feature_meta"
REPORT_PATH = WORK_DIR / "search_report.json"
RAW_FEATURES_PATH = WORK_DIR / "raw_features.json"

META_SOURCES = (
    *client.SOURCES,
    "fv_vblock_dynamics",
    "fv_vblock_cd_dynamics",
    "fv_vblock_structured",
)
LGB_CONFIGS = (
    {"num_leaves": 15, "max_depth": 4, "min_child_samples": 300},
    {"num_leaves": 31, "max_depth": 5, "min_child_samples": 300},
    {"num_leaves": 31, "max_depth": 6, "min_child_samples": 600},
    {"num_leaves": 63, "max_depth": 7, "min_child_samples": 800},
    {"num_leaves": 63, "max_depth": 8, "min_child_samples": 1_200},
)
LGB_PARAMS = {
    **heavy.LGB_PARAMS,
    "n_estimators": 3_000,
    "learning_rate": 0.0125,
    "reg_alpha": 2.0,
    "reg_lambda": 20.0,
    "random_state": 9109,
}


def select_raw_features(columns: list[str]) -> list[str]:
    explicit = {
        "TransactionAmt",
        "TransactionAmt_log1p",
        "TransactionAmt_cents",
        "TransactionAmt_is_integer",
        "TransactionAmt_is_round_10",
        "ProductCD",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
        "P_R_email_match",
        "P_email_suffix",
        "R_email_suffix",
        "DeviceType",
        "DeviceInfo",
        "DeviceInfo_family",
        "browser_family",
        "DT_hour",
        "DT_dayofweek",
        "D1_origin_day",
        "row_missing_count",
        "C_mean",
        "C_std",
        "C_min",
        "C_max",
        "D_missing_count",
        "D_mean",
        "D_std",
        "D_min",
        "D_max",
        "V_missing_count",
        "V_mean",
        "V_std",
        "V_min",
        "V_max",
        "id_numeric_missing_count",
        "id_numeric_mean",
        "id_numeric_std",
        "id_numeric_min",
        "id_numeric_max",
    }
    result = []
    top_v = set(TOP_V_COLUMNS)
    for column in columns:
        raw_family = bool(
            re.fullmatch(r"card[1-6]", column)
            or re.fullmatch(r"C(?:[1-9]|1[0-4])", column)
            or re.fullmatch(r"D(?:[1-9]|1[0-5])", column)
            or re.fullmatch(r"D(?:[1-9]|1[0-5])_minus_day", column)
            or re.fullmatch(r"M[1-9]", column)
            or re.fullmatch(r"id_(?:0[1-9]|[12][0-9]|3[0-8])", column)
        )
        engineered = bool(
            column.startswith("uid_d1_email_seq_")
            or column.startswith("uid_card_addr_d1_email_seq_")
            or (
                column.endswith("_freq")
                and column.startswith(
                    (
                        "card",
                        "addr",
                        "P_email",
                        "R_email",
                        "Device",
                        "uid_",
                    )
                )
            )
        )
        if column in explicit or column in top_v or raw_family or engineered:
            result.append(column)
    # This block was explicitly reported as time-inconsistent by the winning
    # solution and is removed before any local-label model selection.
    result = [
        column
        for column in result
        if not (
            re.fullmatch(r"V\d+", column)
            and 322 <= int(column[1:]) <= 339
        )
    ]
    return list(dict.fromkeys(result))


def encode_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
) -> None:
    for column in categorical:
        combined = pd.concat(
            [train[column], test[column]], ignore_index=True, copy=False
        ).astype("string")
        codes, _ = pd.factorize(combined, sort=False)
        train[column] = codes[: len(train)].astype("int32")
        test[column] = codes[len(train) :].astype("int32")


def prepare_raw_matrix(
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    print("Building target-free raw meta features...", flush=True)
    train = read_and_merge(ROOT, "train")
    test = read_and_merge(ROOT, "test")
    if client.TARGET in test:
        raise RuntimeError("Target reached official test")
    train, test, all_features, categorical = build_features(
        train,
        test,
        giba_features=True,
        frequency_mode="selected",
        v307_chain_features=True,
    )
    selected = select_raw_features(all_features)
    selected_categorical = [column for column in categorical if column in selected]
    encode_categories(train, test, selected_categorical)
    oof_rows = oof["row_index"].to_numpy(dtype="int64")
    oof_raw = train.iloc[oof_rows][selected].reset_index(drop=True)
    test_raw = test[selected].reset_index(drop=True)
    for frame in (oof_raw, test_raw):
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                "float32"
            )
    return oof_raw, test_raw, selected


def make_features(
    predictions: pd.DataFrame,
    membership: pd.DataFrame,
    raw: pd.DataFrame,
    fold: pd.Series | None,
) -> pd.DataFrame:
    source_features = meta.build_meta_features(
        predictions, META_SOURCES, fold=fold
    ).reset_index(drop=True)
    return pd.concat(
        [
            source_features,
            membership.reset_index(drop=True),
            raw.reset_index(drop=True),
        ],
        axis=1,
    ).astype("float32")


def search_lgb(
    oof: pd.DataFrame,
    membership: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[dict, list[dict], np.ndarray]:
    features = make_features(oof, membership, raw, fold=oof["fold"])
    train_mask = oof["fold"].lt(client.META_DEV_FOLD).to_numpy()
    valid_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    y_train = oof.loc[train_mask, client.TARGET].to_numpy(dtype="int8")
    y_valid = oof.loc[valid_mask, client.TARGET].to_numpy(dtype="int8")
    rows = []
    predictions = {}
    for index, config in enumerate(LGB_CONFIGS):
        model = lgb.LGBMClassifier(**LGB_PARAMS, **config)
        model.fit(
            features.loc[train_mask],
            y_train,
            eval_set=[(features.loc[valid_mask], y_valid)],
            callbacks=[
                lgb.early_stopping(180, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        prediction = model.predict_proba(features.loc[valid_mask])[:, 1]
        row = {
            "config_index": index,
            **config,
            "features": int(features.shape[1]),
            "best_iteration": int(model.best_iteration_),
            "dev_auc": float(roc_auc_score(y_valid, prediction)),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
        predictions[index] = prediction
        del model
        gc.collect()
    rows.sort(key=lambda row: (row["dev_auc"], -row["num_leaves"]), reverse=True)
    selected = rows[0]
    return selected, rows, predictions[int(selected["config_index"])]


def train_lock(
    oof: pd.DataFrame,
    membership: pd.DataFrame,
    raw: pd.DataFrame,
    selected: dict,
) -> tuple[lgb.LGBMClassifier, np.ndarray, int]:
    features = make_features(oof, membership, raw, fold=oof["fold"])
    train_mask = oof["fold"].lt(client.META_LOCK_FOLD).to_numpy()
    valid_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    iterations = max(30, int(np.ceil(selected["best_iteration"] * 1.15)))
    model = lgb.LGBMClassifier(
        **{
            **LGB_PARAMS,
            **LGB_CONFIGS[int(selected["config_index"])],
            "n_estimators": iterations,
        }
    )
    model.fit(
        features.loc[train_mask],
        oof.loc[train_mask, client.TARGET],
        callbacks=[lgb.log_evaluation(0)],
    )
    return model, model.predict_proba(features.loc[valid_mask])[:, 1], iterations


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    oof = featureview.build_oof()
    membership = pd.read_csv(ROOT / "honest_client_meta/oof_client_features.csv")
    raw_oof, raw_test_unused, selected_raw = prepare_raw_matrix(oof)
    del raw_test_unused
    gc.collect()
    RAW_FEATURES_PATH.write_text(json.dumps(selected_raw, indent=2), encoding="utf-8")

    raw_train = client.prepare_raw(
        pd.read_csv(ROOT / "train_transaction.csv", usecols=list(client.RAW_COLUMNS))
    )
    raw_test = client.prepare_raw(
        pd.read_csv(ROOT / "test_transaction.csv", usecols=list(client.RAW_COLUMNS))
    )
    _, _, fold_groups, _ = client.build_client_features(oof, raw_train, raw_test)
    uid_raw = pd.read_csv(
        ROOT / "train_transaction.csv",
        usecols=["TransactionDT", "card1", "addr1", "D1", "P_emaildomain"],
    )
    uid = meta.make_uid(uid_raw)
    dev_current, dev_report = client.current_meta_prediction(oof, uid, client.META_DEV_FOLD)
    lock_current, lock_report = client.current_meta_prediction(oof, uid, client.META_LOCK_FOLD)

    selected_model, model_search, dev_raw = search_lgb(oof, membership, raw_oof)
    lock_model, lock_raw, lock_iterations = train_lock(
        oof, membership, raw_oof, selected_model
    )
    lock_model.booster_.save_model(WORK_DIR / "lock_lgb.txt")
    feature_report = json.loads(
        (ROOT / "honest_featureview_meta/search_report.json").read_text(encoding="utf-8")
    )
    postprocess = feature_report["postprocess_locked_from_dev"]
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_rows = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    selected_weights, weight_search = heavy.search_segment_weights(
        oof.loc[dev_mask, client.TARGET].to_numpy(dtype="int8"),
        dev_current,
        dev_raw,
        segments.segment_labels(membership.loc[dev_mask].reset_index(drop=True)),
        raw_train.iloc[dev_rows]["TransactionDT"].to_numpy(),
        fold_groups[client.META_DEV_FOLD],
        postprocess,
    )
    lock_prediction = segments.segmented_blend(
        lock_current,
        lock_raw,
        segments.segment_labels(membership.loc[lock_mask].reset_index(drop=True)),
        selected_weights["weights"],
    )
    lock_prediction = client.apply_postprocess(
        lock_prediction, fold_groups[client.META_LOCK_FOLD], postprocess
    )
    y_lock = oof.loc[lock_mask, client.TARGET].to_numpy(dtype="int8")
    lock_auc = float(roc_auc_score(y_lock, lock_prediction))
    previous = float(feature_report["candidate_lock_auc"])
    report = {
        "data_policy": "official train/test covariates; train OOF labels only",
        "feature_policy": "family rules fixed before nested validation",
        "raw_feature_count": len(selected_raw),
        "raw_features": selected_raw,
        "meta_sources": list(META_SOURCES),
        "selected_model": selected_model,
        "model_search": model_search,
        "selected_segment_weights": selected_weights,
        "weight_search_top": weight_search[:30],
        "postprocess_locked_from_dev": postprocess,
        "current_meta": {"dev": dev_report, "lock": lock_report},
        "lock_iterations": lock_iterations,
        "previous_featureview_lock_auc": previous,
        "candidate_lock_auc": lock_auc,
        "lock_gain": lock_auc - previous,
        "accepted": bool(selected_weights["gain"] > 0 and lock_auc > previous),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_feature_count": len(selected_raw),
                "selected_model": selected_model,
                "selected_segment_weights": selected_weights,
                "previous_featureview_lock_auc": previous,
                "candidate_lock_auc": lock_auc,
                "lock_gain": lock_auc - previous,
                "accepted": report["accepted"],
                "elapsed_minutes": report["elapsed_minutes"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

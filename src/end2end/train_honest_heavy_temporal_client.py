"""Train a heavy client-aware temporal stack using official data only.

The additional seven CatBoost/LightGBM sources are out-of-fold predictions
from purged forward-time splits. Their final test counterparts come from the
clean temporal run. Model choices are made on fold 1; fold 2 stays locked.
"""

from __future__ import annotations

import gc
from itertools import product
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import build_honest_no_gap_meta as meta
import refine_honest_client_segments as segments
import train_honest_client_meta as client


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_heavy_temporal_client"
OUTPUT_PATH = ROOT / "submission_honest_heavy_temporal_client.csv"
REPORT_PATH = WORK_DIR / "report.json"

TEMPORAL_BASE_SOURCES = (
    "cat_giba",
    "cat_multi_full",
    "cat_recent45",
    "cat_recent30",
    "cat_plain",
    "cat_time_plain",
    "lgb_giba",
)
TEMPORAL_SOURCES = tuple(f"t7_{source}" for source in TEMPORAL_BASE_SOURCES)
TEMPORAL_STACK = "t7_stack"
ALL_HEAVY_SOURCES = (*client.SOURCES, *TEMPORAL_SOURCES, TEMPORAL_STACK)

VIEWS = {
    "diverse_clients": (
        (*client.SOURCES, "t7_cat_giba", "t7_cat_recent30", "t7_lgb_giba", TEMPORAL_STACK),
        True,
    ),
    "temporal_core_clients": (
        (*client.SOURCES, "t7_cat_giba", "t7_cat_multi_full", "t7_cat_recent45",
         "t7_cat_recent30", "t7_lgb_giba", TEMPORAL_STACK),
        True,
    ),
    "all_heavy_clients": (ALL_HEAVY_SOURCES, True),
    "all_heavy": (ALL_HEAVY_SOURCES, False),
}

LGB_CONFIGS = (
    {"num_leaves": 15, "max_depth": 4, "min_child_samples": 300},
    {"num_leaves": 31, "max_depth": 5, "min_child_samples": 300},
    {"num_leaves": 31, "max_depth": 6, "min_child_samples": 600},
    {"num_leaves": 63, "max_depth": 6, "min_child_samples": 600},
    {"num_leaves": 63, "max_depth": 7, "min_child_samples": 1_000},
    {"num_leaves": 127, "max_depth": 8, "min_child_samples": 1_000},
)
LGB_PARAMS = {
    "n_estimators": 2_500,
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.015,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_alpha": 1.5,
    "reg_lambda": 15.0,
    "random_state": 7907,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
SEGMENT_WEIGHTS = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))


def verify_no_gap_artifacts() -> dict:
    metrics_path = ROOT / "temporal7_no_gap_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    training = metrics["training"]
    required = {
        "unlabeled_bridge_rows": 0,
        "uses_external_labels": False,
        "uses_competition_test_labels": False,
    }
    observed = {key: training.get(key) for key in required}
    if observed != required:
        raise RuntimeError(f"Temporal source is not the clean no-gap run: {observed}")
    test_header = pd.read_csv(ROOT / "test_transaction.csv", nrows=1)
    if client.TARGET in test_header.columns:
        raise RuntimeError("Target leaked into official test_transaction.csv")
    return training


def build_heavy_oof() -> pd.DataFrame:
    oof = client.build_oof_sources()
    temporal_columns = [
        "row_index",
        "fold",
        *TEMPORAL_BASE_SOURCES,
        "stack_prediction",
    ]
    temporal = pd.read_csv(
        ROOT / "temporal7_oof_predictions.csv",
        usecols=temporal_columns,
    ).rename(
        columns={
            "fold": "temporal_fold",
            "stack_prediction": TEMPORAL_STACK,
            **{source: f"t7_{source}" for source in TEMPORAL_BASE_SOURCES},
        }
    )
    merged = oof.merge(temporal, on="row_index", how="left", validate="one_to_one")
    if merged[list(TEMPORAL_SOURCES) + [TEMPORAL_STACK]].isna().any().any():
        raise RuntimeError("Temporal7 OOF does not cover every current OOF row")
    if not np.array_equal(
        merged["temporal_fold"].to_numpy(),
        merged["fold"].to_numpy() + 1,
    ):
        raise RuntimeError("Temporal7 and current purged folds are misaligned")
    return merged.drop(columns="temporal_fold").reset_index(drop=True)


def build_heavy_test() -> pd.DataFrame:
    result = client.build_test_sources()
    temporal = pd.read_csv(ROOT / "temporal7_no_gap_source_predictions.csv")
    stack = pd.read_csv(ROOT / "submission_temporal7_no_gap.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv", usecols=["TransactionID"])
    for frame, name in ((temporal, "temporal sources"), (stack, "temporal stack")):
        if not np.array_equal(frame["TransactionID"].to_numpy(), sample["TransactionID"].to_numpy()):
            raise RuntimeError(f"TransactionID order differs for {name}")
    for source in TEMPORAL_BASE_SOURCES:
        result[f"t7_{source}"] = temporal[source].to_numpy(dtype="float64")
    result[TEMPORAL_STACK] = stack[client.TARGET].to_numpy(dtype="float64")
    return result


def make_view(
    predictions: pd.DataFrame,
    membership: pd.DataFrame,
    view: str,
    fold: pd.Series | None,
) -> pd.DataFrame:
    sources, include_clients = VIEWS[view]
    features = meta.build_meta_features(predictions, sources, fold=fold).reset_index(drop=True)
    if include_clients:
        features = pd.concat([features, membership.reset_index(drop=True)], axis=1)
    return features.astype("float32")


def search_models(
    oof: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[dict, list[dict], np.ndarray]:
    train_mask = oof["fold"].lt(client.META_DEV_FOLD).to_numpy()
    valid_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    y_train = oof.loc[train_mask, client.TARGET].to_numpy(dtype="int8")
    y_valid = oof.loc[valid_mask, client.TARGET].to_numpy(dtype="int8")
    rows: list[dict] = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for view in VIEWS:
        features = make_view(oof, membership, view, fold=oof["fold"])
        for config_index, config in enumerate(LGB_CONFIGS):
            model = lgb.LGBMClassifier(**LGB_PARAMS, **config)
            model.fit(
                features.loc[train_mask],
                y_train,
                eval_set=[(features.loc[valid_mask], y_valid)],
                callbacks=[
                    lgb.early_stopping(150, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            prediction = model.predict_proba(features.loc[valid_mask])[:, 1]
            row = {
                "view": view,
                "config_index": config_index,
                **config,
                "features": int(features.shape[1]),
                "best_iteration": int(model.best_iteration_),
                "dev_auc": float(roc_auc_score(y_valid, prediction)),
            }
            rows.append(row)
            predictions[(view, config_index)] = prediction
            print(json.dumps(row), flush=True)
            del model
            gc.collect()
        del features
        gc.collect()
    rows.sort(key=lambda row: (row["dev_auc"], -row["num_leaves"]), reverse=True)
    selected = rows[0]
    return selected, rows, predictions[(selected["view"], selected["config_index"])]


def train_model(
    oof: pd.DataFrame,
    membership: pd.DataFrame,
    selected: dict,
    valid_fold: int,
    iteration_scale: float,
) -> tuple[lgb.LGBMClassifier, np.ndarray, int]:
    features = make_view(oof, membership, selected["view"], fold=oof["fold"])
    train_mask = oof["fold"].lt(valid_fold).to_numpy()
    valid_mask = oof["fold"].eq(valid_fold).to_numpy()
    iterations = max(30, int(np.ceil(selected["best_iteration"] * iteration_scale)))
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
    prediction = model.predict_proba(features.loc[valid_mask])[:, 1]
    return model, prediction, iterations


def search_segment_weights(
    y: np.ndarray,
    baseline: np.ndarray,
    stacked: np.ndarray,
    labels: np.ndarray,
    dt: np.ndarray,
    groups: dict[str, tuple[np.ndarray, np.ndarray, int]],
    postprocess: dict,
) -> tuple[dict, list[dict]]:
    midpoint = np.median(dt)
    halves = (dt <= midpoint, dt > midpoint)
    baseline_final = client.apply_postprocess(baseline, groups, postprocess)
    baseline_auc = float(roc_auc_score(y, baseline_final))
    baseline_halves = [float(roc_auc_score(y[mask], baseline_final[mask])) for mask in halves]
    rows = []
    for values in product(SEGMENT_WEIGHTS, repeat=len(segments.SEGMENTS)):
        weights = dict(zip(segments.SEGMENTS, values))
        prediction = segments.segmented_blend(baseline, stacked, labels, weights)
        prediction = client.apply_postprocess(prediction, groups, postprocess)
        half_auc = [float(roc_auc_score(y[mask], prediction[mask])) for mask in halves]
        half_gains = [score - base for score, base in zip(half_auc, baseline_halves)]
        score = float(roc_auc_score(y, prediction))
        rows.append(
            {
                "weights": weights,
                "auc": score,
                "gain": score - baseline_auc,
                "half_gains": half_gains,
                "min_half_gain": float(min(half_gains)),
            }
        )
    stable = [row for row in rows if row["min_half_gain"] >= 0.0]
    selected = max(stable, key=lambda row: (row["gain"], row["min_half_gain"]))
    rows.sort(key=lambda row: (row["gain"], row["min_half_gain"]), reverse=True)
    return selected, rows


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    clean_training = verify_no_gap_artifacts()
    oof = build_heavy_oof()
    test_sources = build_heavy_test()
    membership_oof = pd.read_csv(ROOT / "honest_client_meta/oof_client_features.csv")
    membership_test = pd.read_csv(ROOT / "honest_client_meta/test_client_features.csv")

    raw_train = client.prepare_raw(
        pd.read_csv(ROOT / "train_transaction.csv", usecols=list(client.RAW_COLUMNS))
    )
    raw_test = client.prepare_raw(
        pd.read_csv(ROOT / "test_transaction.csv", usecols=list(client.RAW_COLUMNS))
    )
    _, _, fold_groups, test_groups = client.build_client_features(oof, raw_train, raw_test)
    raw_uid = pd.read_csv(
        ROOT / "train_transaction.csv",
        usecols=["TransactionDT", "card1", "addr1", "D1", "P_emaildomain"],
    )
    uid = meta.make_uid(raw_uid)
    dev_current, dev_current_report = client.current_meta_prediction(oof, uid, client.META_DEV_FOLD)
    lock_current, lock_current_report = client.current_meta_prediction(oof, uid, client.META_LOCK_FOLD)

    selected_model, model_search, dev_heavy = search_models(oof, membership_oof)
    lock_model, lock_heavy, lock_iterations = train_model(
        oof, membership_oof, selected_model, client.META_LOCK_FOLD, 1.15
    )
    lock_model.booster_.save_model(WORK_DIR / "lock_lgb.txt")

    base_report = json.loads(
        (ROOT / "honest_client_meta/report.json").read_text(encoding="utf-8")
    )
    postprocess = base_report["selected_postprocess"]
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_rows = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    selected_weights, weight_search = search_segment_weights(
        oof.loc[dev_mask, client.TARGET].to_numpy(dtype="int8"),
        dev_current,
        dev_heavy,
        segments.segment_labels(membership_oof.loc[dev_mask].reset_index(drop=True)),
        raw_train.iloc[dev_rows]["TransactionDT"].to_numpy(),
        fold_groups[client.META_DEV_FOLD],
        postprocess,
    )
    lock_prediction = segments.segmented_blend(
        lock_current,
        lock_heavy,
        segments.segment_labels(membership_oof.loc[lock_mask].reset_index(drop=True)),
        selected_weights["weights"],
    )
    lock_prediction = client.apply_postprocess(
        lock_prediction, fold_groups[client.META_LOCK_FOLD], postprocess
    )
    y_lock = oof.loc[lock_mask, client.TARGET].to_numpy(dtype="int8")
    lock_auc = float(roc_auc_score(y_lock, lock_prediction))
    previous_lock_auc = float(
        json.loads((ROOT / "honest_client_segments/report.json").read_text(encoding="utf-8"))[
            "segment_candidate_lock_auc"
        ]
    )
    accepted = bool(selected_weights["gain"] > 0.0 and lock_auc > previous_lock_auc)

    final_iterations = max(30, int(np.ceil(lock_iterations * 1.15)))
    all_features = make_view(oof, membership_oof, selected_model["view"], fold=oof["fold"])
    test_features = make_view(test_sources, membership_test, selected_model["view"], fold=None)
    final_model = lgb.LGBMClassifier(
        **{
            **LGB_PARAMS,
            **LGB_CONFIGS[int(selected_model["config_index"])],
            "n_estimators": final_iterations,
        }
    )
    final_model.fit(all_features, oof[client.TARGET], callbacks=[lgb.log_evaluation(0)])
    final_model.booster_.save_model(WORK_DIR / "final_lgb.txt")
    heavy_test = final_model.predict_proba(test_features)[:, 1]
    current_test = pd.read_csv(ROOT / "submission_honest_user_means.csv")
    test_prediction = segments.segmented_blend(
        current_test[client.TARGET].to_numpy(dtype="float64"),
        heavy_test,
        segments.segment_labels(membership_test),
        selected_weights["weights"],
    )
    test_prediction = client.apply_postprocess(test_prediction, test_groups, postprocess)
    output = current_test[["TransactionID"]].copy()
    output[client.TARGET] = test_prediction
    output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": "official train/test only",
        "no_gap_artifact_training": clean_training,
        "fold_contract": "temporal7_fold == current_fold + 1",
        "sources": list(ALL_HEAVY_SOURCES),
        "selected_model": selected_model,
        "model_search": model_search,
        "selected_segment_weights": selected_weights,
        "weight_search_top": weight_search[:30],
        "postprocess_locked_from_dev": postprocess,
        "current_meta": {"dev": dev_current_report, "lock": lock_current_report},
        "lock_iterations": lock_iterations,
        "final_iterations": final_iterations,
        "previous_best_lock_auc": previous_lock_auc,
        "candidate_lock_auc": lock_auc,
        "lock_gain": lock_auc - previous_lock_auc,
        "accepted": accepted,
        "output": OUTPUT_PATH.name,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_model": selected_model,
                "selected_segment_weights": selected_weights,
                "previous_best_lock_auc": previous_lock_auc,
                "candidate_lock_auc": lock_auc,
                "lock_gain": lock_auc - previous_lock_auc,
                "accepted": accepted,
                "output": OUTPUT_PATH.name,
                "elapsed_minutes": report["elapsed_minutes"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

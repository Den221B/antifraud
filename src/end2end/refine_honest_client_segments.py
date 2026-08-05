"""Try client-segment-specific meta weights using train-only temporal folds."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import build_honest_no_gap_meta as meta
import train_honest_client_meta as client


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_client_segments"
REPORT_PATH = WORK_DIR / "report.json"
OUTPUT_PATH = ROOT / "submission_honest_client_segments.csv"
SEGMENTS = ("strict", "partial", "cold")
WEIGHTS = (0.0, 0.20, 0.40, 0.60, 0.80, 1.00)


def rank_match(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    order = np.argsort(signal, kind="mergesort")
    result = np.empty(len(signal), dtype="float64")
    result[order] = np.sort(reference, kind="mergesort")
    return result


def segment_labels(features: pd.DataFrame) -> np.ndarray:
    result = np.full(len(features), "cold", dtype=object)
    partial = features["client_segment_partial"].to_numpy(dtype=bool)
    strict = features["client_segment_strict"].to_numpy(dtype=bool)
    result[partial] = "partial"
    result[strict] = "strict"
    return result


def segmented_blend(
    current: np.ndarray,
    stacked: np.ndarray,
    segments: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    current_rank = client.rank(current)
    stacked_rank = client.rank(stacked)
    prediction = current_rank.copy()
    for segment in SEGMENTS:
        mask = segments == segment
        weight = float(weights[segment])
        signal = (
            (1.0 - weight) * client.rank(current_rank[mask])
            + weight * client.rank(stacked_rank[mask])
        )
        prediction[mask] = rank_match(signal, current_rank[mask])
    return prediction


def search_weights(
    y: np.ndarray,
    current: np.ndarray,
    stacked: np.ndarray,
    segments: np.ndarray,
    dt: np.ndarray,
    groups: dict[str, tuple[np.ndarray, np.ndarray, int]],
    postprocess: dict,
) -> tuple[dict, list[dict]]:
    midpoint = np.median(dt)
    halves = (dt <= midpoint, dt > midpoint)
    baseline = client.apply_postprocess(
        current,
        groups,
        postprocess,
    )
    baseline_auc = float(roc_auc_score(y, baseline))
    baseline_half = [
        float(roc_auc_score(y[mask], baseline[mask])) for mask in halves
    ]
    rows = []
    for values in product(WEIGHTS, repeat=len(SEGMENTS)):
        weights = dict(zip(SEGMENTS, values))
        prediction = segmented_blend(current, stacked, segments, weights)
        prediction = client.apply_postprocess(
            prediction,
            groups,
            postprocess,
        )
        half_auc = [
            float(roc_auc_score(y[mask], prediction[mask])) for mask in halves
        ]
        half_gains = [
            score - base for score, base in zip(half_auc, baseline_half)
        ]
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
    WORK_DIR.mkdir(exist_ok=True)
    base_report = json.loads(
        (ROOT / "honest_client_meta/report.json").read_text(encoding="utf-8")
    )
    selected_lgb = base_report["selected_lgb"]
    postprocess = base_report["selected_postprocess"]

    oof = client.build_oof_sources()
    test_sources = client.build_test_sources()
    client_oof = pd.read_csv(
        ROOT / "honest_client_meta/oof_client_features.csv"
    )
    client_test = pd.read_csv(
        ROOT / "honest_client_meta/test_client_features.csv"
    )
    raw_train = client.prepare_raw(
        pd.read_csv(
            ROOT / "train_transaction.csv",
            usecols=list(client.RAW_COLUMNS),
        )
    )
    raw_test = client.prepare_raw(
        pd.read_csv(
            ROOT / "test_transaction.csv",
            usecols=list(client.RAW_COLUMNS),
        )
    )
    _, _, fold_groups, test_groups = client.build_client_features(
        oof,
        raw_train,
        raw_test,
    )
    raw_uid = pd.read_csv(
        ROOT / "train_transaction.csv",
        usecols=["TransactionDT", "card1", "addr1", "D1", "P_emaildomain"],
    )
    uid = meta.make_uid(raw_uid)
    dev_current, _ = client.current_meta_prediction(
        oof, uid, client.META_DEV_FOLD
    )
    lock_current, _ = client.current_meta_prediction(
        oof, uid, client.META_LOCK_FOLD
    )

    view = selected_lgb["view"]
    features = client.make_meta_view(
        oof,
        client_oof,
        view,
        fold=oof["fold"],
    )
    dev_train = oof["fold"].lt(client.META_DEV_FOLD).to_numpy()
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    config = client.LGB_CONFIGS[int(selected_lgb["config_index"])]
    dev_model = lgb.LGBMClassifier(
        **{
            **client.LGB_BASE_PARAMS,
            **config,
            "n_estimators": int(selected_lgb["best_iteration"]),
        }
    )
    dev_model.fit(
        features.loc[dev_train],
        oof.loc[dev_train, client.TARGET],
        callbacks=[lgb.log_evaluation(0)],
    )
    dev_lgb = dev_model.predict_proba(features.loc[dev_mask])[:, 1]
    lock_model = lgb.Booster(
        model_file=str(ROOT / "honest_client_meta/lock_lgb.txt")
    )
    lock_lgb = lock_model.predict(features.loc[lock_mask])

    dev_rows = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    selected, search = search_weights(
        oof.loc[dev_mask, client.TARGET].to_numpy(dtype="int8"),
        dev_current,
        dev_lgb,
        segment_labels(client_oof.loc[dev_mask].reset_index(drop=True)),
        raw_train.iloc[dev_rows]["TransactionDT"].to_numpy(),
        fold_groups[client.META_DEV_FOLD],
        postprocess,
    )
    lock_prediction = segmented_blend(
        lock_current,
        lock_lgb,
        segment_labels(client_oof.loc[lock_mask].reset_index(drop=True)),
        selected["weights"],
    )
    lock_prediction = client.apply_postprocess(
        lock_prediction,
        fold_groups[client.META_LOCK_FOLD],
        postprocess,
    )
    y_lock = oof.loc[lock_mask, client.TARGET].to_numpy(dtype="int8")
    lock_auc = float(roc_auc_score(y_lock, lock_prediction))
    base_lock_auc = float(base_report["scores"]["lock_final_auc"])
    accepted = bool(selected["gain"] > 0 and lock_auc > base_lock_auc)

    test_features = client.make_meta_view(
        test_sources,
        client_test,
        view,
        fold=None,
    )
    final_model = lgb.Booster(
        model_file=str(ROOT / "honest_client_meta/final_lgb.txt")
    )
    test_lgb = final_model.predict(test_features)
    current_test = pd.read_csv(ROOT / "submission_honest_user_means.csv")
    test_prediction = segmented_blend(
        current_test[client.TARGET].to_numpy(),
        test_lgb,
        segment_labels(client_test),
        selected["weights"],
    )
    test_prediction = client.apply_postprocess(
        test_prediction,
        test_groups,
        postprocess,
    )
    output = current_test[["TransactionID"]].copy()
    output[client.TARGET] = test_prediction
    output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "selection": "segment LGB weights on fold 1 halves; fold 2 lock",
        "data_policy": "official train/test only; no competition-test labels",
        "selected": selected,
        "search_top": search[:20],
        "base_client_meta_lock_auc": base_lock_auc,
        "segment_candidate_lock_auc": lock_auc,
        "lock_gain": lock_auc - base_lock_auc,
        "accepted": accepted,
        "output": OUTPUT_PATH.name,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "base_client_meta_lock_auc": report[
                    "base_client_meta_lock_auc"
                ],
                "segment_candidate_lock_auc": report[
                    "segment_candidate_lock_auc"
                ],
                "lock_gain": report["lock_gain"],
                "accepted": report["accepted"],
                "output": report["output"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

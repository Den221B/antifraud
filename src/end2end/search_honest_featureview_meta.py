"""Search clean feature-view residual sources above the honest client stack."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import build_honest_no_gap_meta as meta
import refine_honest_client_segments as segments
import train_honest_client_meta as client
import train_honest_heavy_temporal_client as heavy


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_featureview_meta"
REPORT_PATH = WORK_DIR / "search_report.json"

FEATURE_SOURCES = (
    "fv_vblock_dynamics",
    "fv_vblock_cd_dynamics",
    "fv_vblock_aggregates_dynamics",
    "fv_vblock_d_amount",
    "fv_vblock_structured",
    "fv_vblock_chains",
)
CURRENT = tuple(client.SOURCES)
VIEWS = {
    "stable_three_clients": (
        (*CURRENT, "fv_vblock_dynamics", "fv_vblock_cd_dynamics", "fv_vblock_structured"),
        True,
    ),
    "structured_clients": (
        (*CURRENT, "fv_vblock_d_amount", "fv_vblock_structured", "fv_vblock_chains"),
        True,
    ),
    "all_featureviews_clients": ((*CURRENT, *FEATURE_SOURCES), True),
    "all_featureviews": ((*CURRENT, *FEATURE_SOURCES), False),
}


def build_oof() -> pd.DataFrame:
    oof = client.build_oof_sources()
    advanced = pd.read_csv(
        ROOT / "advanced_feature_ablation_oof.csv",
        usecols=[
            "row_index",
            "fold",
            "vblock_dynamics",
            "vblock_cd_dynamics",
            "vblock_aggregates_dynamics",
        ],
    ).rename(
        columns={
            "fold": "feature_fold",
            "vblock_dynamics": "fv_vblock_dynamics",
            "vblock_cd_dynamics": "fv_vblock_cd_dynamics",
            "vblock_aggregates_dynamics": "fv_vblock_aggregates_dynamics",
        }
    )
    next_views = pd.read_csv(
        ROOT / "next_feature_ablation_oof.csv",
        usecols=[
            "row_index",
            "vblock_d_amount",
            "vblock_structured",
            "vblock_chains",
        ],
    ).rename(
        columns={
            "vblock_d_amount": "fv_vblock_d_amount",
            "vblock_structured": "fv_vblock_structured",
            "vblock_chains": "fv_vblock_chains",
        }
    )
    merged = oof.merge(advanced, on="row_index", how="left", validate="one_to_one")
    merged = merged.merge(next_views, on="row_index", how="left", validate="one_to_one")
    if merged[list(FEATURE_SOURCES)].isna().any().any():
        raise RuntimeError("Feature-view OOF does not cover current OOF")
    if not np.array_equal(
        merged["feature_fold"].to_numpy(), merged["fold"].to_numpy() + 1
    ):
        raise RuntimeError("Feature-view and current temporal folds are misaligned")
    return merged.drop(columns="feature_fold").reset_index(drop=True)


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    oof = build_oof()
    membership = pd.read_csv(ROOT / "honest_client_meta/oof_client_features.csv")
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

    original_views = heavy.VIEWS
    heavy.VIEWS = VIEWS
    try:
        selected_model, model_search, dev_feature = heavy.search_models(oof, membership)
        lock_model, lock_feature, lock_iterations = heavy.train_model(
            oof, membership, selected_model, client.META_LOCK_FOLD, 1.15
        )
    finally:
        heavy.VIEWS = original_views
    lock_model.booster_.save_model(WORK_DIR / "lock_lgb.txt")

    base_report = json.loads(
        (ROOT / "honest_client_meta/report.json").read_text(encoding="utf-8")
    )
    postprocess = base_report["selected_postprocess"]
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_rows = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    selected_weights, weight_search = heavy.search_segment_weights(
        oof.loc[dev_mask, client.TARGET].to_numpy(dtype="int8"),
        dev_current,
        dev_feature,
        segments.segment_labels(membership.loc[dev_mask].reset_index(drop=True)),
        raw_train.iloc[dev_rows]["TransactionDT"].to_numpy(),
        fold_groups[client.META_DEV_FOLD],
        postprocess,
    )
    lock_prediction = segments.segmented_blend(
        lock_current,
        lock_feature,
        segments.segment_labels(membership.loc[lock_mask].reset_index(drop=True)),
        selected_weights["weights"],
    )
    lock_prediction = client.apply_postprocess(
        lock_prediction, fold_groups[client.META_LOCK_FOLD], postprocess
    )
    y_lock = oof.loc[lock_mask, client.TARGET].to_numpy(dtype="int8")
    lock_auc = float(roc_auc_score(y_lock, lock_prediction))
    previous = float(
        json.loads((ROOT / "honest_client_segments/report.json").read_text(encoding="utf-8"))[
            "segment_candidate_lock_auc"
        ]
    )
    report = {
        "data_policy": "official train OOF only",
        "sources": list(FEATURE_SOURCES),
        "selected_model": selected_model,
        "model_search": model_search,
        "selected_segment_weights": selected_weights,
        "weight_search_top": weight_search[:30],
        "postprocess_locked_from_dev": postprocess,
        "current_meta": {"dev": dev_report, "lock": lock_report},
        "lock_iterations": lock_iterations,
        "previous_best_lock_auc": previous,
        "candidate_lock_auc": lock_auc,
        "lock_gain": lock_auc - previous,
        "accepted": bool(selected_weights["gain"] > 0 and lock_auc > previous),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_model": selected_model,
                "selected_segment_weights": selected_weights,
                "previous_best_lock_auc": previous,
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

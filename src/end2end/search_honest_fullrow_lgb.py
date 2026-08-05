"""Train a full-row raw LightGBM with purged temporal model selection.

Only official train labels are used. Official test covariates participate in
target-free category and sequence construction, matching inference exactly.
"""

from __future__ import annotations

import gc
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import build_honest_no_gap_meta as meta
from fraud_features import build_features, read_and_merge
import refine_honest_client_segments as segments
import search_honest_featureview_meta as featureview
import search_honest_raw_feature_meta as raw_meta
import train_honest_client_meta as client
import train_honest_heavy_temporal_client as heavy


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_fullrow_lgb"
CACHE_DIR = WORK_DIR / "matrix"
REPORT_PATH = WORK_DIR / "report.json"
OUTPUT_PATH = ROOT / "submission_honest_fullrow_lgb.csv"
TARGET = client.TARGET
DAY_SECONDS = 86_400.0
DEV_TRAIN_END = 45.0
LOCK_TRAIN_END = 60.0
DEV_FOLD = client.META_DEV_FOLD
LOCK_FOLD = client.META_LOCK_FOLD
FEATURE_RECIPE_VERSION = "raw233_joint_sequence_v1"

LGB_PARAMS = {
    "n_estimators": 3_500,
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.015,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 2.0,
    "reg_lambda": 20.0,
    "random_state": 12011,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
LGB_CONFIGS = (
    {"num_leaves": 31, "max_depth": 6, "min_child_samples": 300},
    {"num_leaves": 63, "max_depth": 7, "min_child_samples": 500},
    {"num_leaves": 127, "max_depth": 8, "min_child_samples": 800},
    {
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 1_000,
        "extra_trees": True,
    },
    {
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 1_000,
        "extra_trees": True,
    },
)
SEGMENT_WEIGHTS = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def official_hashes() -> dict[str, str]:
    return {
        name: file_sha256(ROOT / name)
        for name in (
            "train_transaction.csv",
            "train_identity.csv",
            "test_transaction.csv",
            "test_identity.csv",
            "sample_submission.csv",
        )
    }


def matrix_paths() -> dict[str, Path]:
    return {
        name: CACHE_DIR / f"{name}.npy"
        for name in ("train", "test", "target", "day", "train_id", "test_id")
    }


def build_matrix_cache(hashes: dict[str, str]) -> dict:
    print("Building the full official-data feature matrix...", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train = read_and_merge(ROOT, "train")
    test = read_and_merge(ROOT, "test")
    if TARGET not in train or TARGET in test:
        raise RuntimeError("Unexpected target placement in official files")

    target = train[TARGET].to_numpy(dtype="int8")
    day = (train["TransactionDT"].to_numpy(dtype="float64") / DAY_SECONDS).astype(
        "float32"
    )
    train_id = train["TransactionID"].to_numpy(dtype="int32")
    test_id = test["TransactionID"].to_numpy(dtype="int32")
    train, test, all_features, categorical = build_features(
        train,
        test,
        giba_features=True,
        frequency_mode="selected",
        v307_chain_features=True,
    )
    selected = raw_meta.select_raw_features(all_features)
    selected_categorical = [name for name in categorical if name in selected]
    raw_meta.encode_categories(train, test, selected_categorical)
    train_matrix = train[selected].astype("float32").to_numpy(copy=True)
    test_matrix = test[selected].astype("float32").to_numpy(copy=True)

    paths = matrix_paths()
    np.save(paths["train"], train_matrix)
    np.save(paths["test"], test_matrix)
    np.save(paths["target"], target)
    np.save(paths["day"], day)
    np.save(paths["train_id"], train_id)
    np.save(paths["test_id"], test_id)
    manifest = {
        "feature_recipe_version": FEATURE_RECIPE_VERSION,
        "official_hashes": hashes,
        "features": selected,
        "categorical_features": selected_categorical,
        "train_shape": list(train_matrix.shape),
        "test_shape": list(test_matrix.shape),
        "target_sum": int(target.sum()),
    }
    (CACHE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    del train, test, train_matrix, test_matrix
    gc.collect()
    return manifest


def load_matrix() -> tuple[dict, dict[str, np.ndarray]]:
    hashes = official_hashes()
    manifest_path = CACHE_DIR / "manifest.json"
    paths = matrix_paths()
    manifest = None
    if manifest_path.exists() and all(path.exists() for path in paths.values()):
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            candidate.get("feature_recipe_version") == FEATURE_RECIPE_VERSION
            and candidate.get("official_hashes") == hashes
        ):
            manifest = candidate
            print("Loading verified full-row matrix cache", flush=True)
    if manifest is None:
        manifest = build_matrix_cache(hashes)
    arrays = {
        name: np.load(path, mmap_mode="r") for name, path in paths.items()
    }
    if list(arrays["train"].shape) != manifest["train_shape"]:
        raise RuntimeError("Cached train matrix shape differs from manifest")
    if list(arrays["test"].shape) != manifest["test_shape"]:
        raise RuntimeError("Cached test matrix shape differs from manifest")
    return manifest, arrays


def make_featureview_matrix(
    oof: pd.DataFrame,
    membership: pd.DataFrame,
    view: str,
) -> pd.DataFrame:
    sources, include_clients = featureview.VIEWS[view]
    features = meta.build_meta_features(
        oof, sources, fold=oof["fold"]
    ).reset_index(drop=True)
    if include_clients:
        features = pd.concat(
            [features, membership.reset_index(drop=True)], axis=1
        )
    return features.astype("float32")


def build_reference_oof(
    oof: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], dict[int, dict], dict]:
    recipe = json.loads(
        (ROOT / "honest_featureview_meta/search_report.json").read_text(
            encoding="utf-8"
        )
    )
    selected = recipe["selected_model"]
    features = make_featureview_matrix(oof, membership, selected["view"])
    raw_train = client.prepare_raw(
        pd.read_csv(
            ROOT / "train_transaction.csv", usecols=list(client.RAW_COLUMNS)
        )
    )
    raw_test = client.prepare_raw(
        pd.read_csv(
            ROOT / "test_transaction.csv", usecols=list(client.RAW_COLUMNS)
        )
    )
    _, _, fold_groups, test_groups = client.build_client_features(
        oof, raw_train, raw_test
    )
    uid_frame = pd.read_csv(
        ROOT / "train_transaction.csv",
        usecols=["TransactionDT", "card1", "addr1", "D1", "P_emaildomain"],
    )
    uid = meta.make_uid(uid_frame)

    predictions = {}
    metrics = {}
    for fold, iterations in (
        (DEV_FOLD, int(selected["best_iteration"])),
        (LOCK_FOLD, int(recipe["lock_iterations"])),
    ):
        train_mask = oof["fold"].lt(fold).to_numpy()
        valid_mask = oof["fold"].eq(fold).to_numpy()
        model = lgb.LGBMClassifier(
            **{
                **heavy.LGB_PARAMS,
                **heavy.LGB_CONFIGS[int(selected["config_index"])],
                "n_estimators": iterations,
            }
        )
        model.fit(
            features.loc[train_mask],
            oof.loc[train_mask, TARGET],
            callbacks=[lgb.log_evaluation(0)],
        )
        feature_prediction = model.predict_proba(features.loc[valid_mask])[:, 1]
        current, _ = client.current_meta_prediction(oof, uid, fold)
        prediction = segments.segmented_blend(
            current,
            feature_prediction,
            segments.segment_labels(
                membership.loc[valid_mask].reset_index(drop=True)
            ),
            recipe["selected_segment_weights"]["weights"],
        )
        prediction = client.apply_postprocess(
            prediction,
            fold_groups[fold],
            recipe["postprocess_locked_from_dev"],
        )
        y_valid = oof.loc[valid_mask, TARGET].to_numpy(dtype="int8")
        predictions[fold] = prediction
        metrics[fold] = {
            "rows": int(valid_mask.sum()),
            "auc": float(roc_auc_score(y_valid, prediction)),
        }
        del model
        gc.collect()

    expected_lock = float(recipe["candidate_lock_auc"])
    if abs(metrics[LOCK_FOLD]["auc"] - expected_lock) > 1e-10:
        raise RuntimeError(
            "Feature-view reference was not reproduced: "
            f"{metrics[LOCK_FOLD]['auc']} != {expected_lock}"
        )
    return predictions, fold_groups, {
        "recipe": recipe,
        "metrics": metrics,
        "test_groups": test_groups,
    }


def search_lgb(
    train: np.ndarray,
    target: np.ndarray,
    day: np.ndarray,
    valid_index: np.ndarray,
) -> tuple[dict, list[dict], np.ndarray]:
    train_index = np.flatnonzero(day < DEV_TRAIN_END)
    rows = []
    predictions = {}
    y_valid = np.asarray(target[valid_index], dtype="int8")
    for config_index, config in enumerate(LGB_CONFIGS):
        model = lgb.LGBMClassifier(**LGB_PARAMS, **config)
        model.fit(
            train[train_index],
            target[train_index],
            eval_set=[(train[valid_index], y_valid)],
            callbacks=[
                lgb.early_stopping(220, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        prediction = model.predict_proba(train[valid_index])[:, 1]
        row = {
            "config_index": config_index,
            **config,
            "best_iteration": int(model.best_iteration_),
            "dev_auc": float(roc_auc_score(y_valid, prediction)),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
        predictions[config_index] = prediction
        del model
        gc.collect()
    rows.sort(
        key=lambda row: (row["dev_auc"], -row["num_leaves"]), reverse=True
    )
    selected = rows[0]
    return selected, rows, predictions[int(selected["config_index"])]


def train_lock(
    train: np.ndarray,
    target: np.ndarray,
    day: np.ndarray,
    valid_index: np.ndarray,
    selected: dict,
) -> tuple[lgb.LGBMClassifier, np.ndarray, int]:
    train_index = np.flatnonzero(day < LOCK_TRAIN_END)
    iterations = max(50, int(np.ceil(selected["best_iteration"] * 1.15)))
    model = lgb.LGBMClassifier(
        **{
            **LGB_PARAMS,
            **LGB_CONFIGS[int(selected["config_index"])],
            "n_estimators": iterations,
        }
    )
    model.fit(
        train[train_index],
        target[train_index],
        callbacks=[lgb.log_evaluation(0)],
    )
    return model, model.predict_proba(train[valid_index])[:, 1], iterations


def prediction_variants(
    prediction: np.ndarray,
    groups: dict,
    postprocess: dict,
) -> dict[str, np.ndarray]:
    ranked = client.rank(prediction)
    postprocessed = client.apply_postprocess(prediction, groups, postprocess)
    return {
        "probability": prediction,
        "rank": ranked,
        "postprocessed_probability": postprocessed,
        "postprocessed_rank": client.apply_postprocess(
            ranked, groups, postprocess
        ),
    }


def search_blend(
    target: np.ndarray,
    baseline: np.ndarray,
    raw_variants: dict[str, np.ndarray],
    segment_labels: np.ndarray,
    dt: np.ndarray,
) -> tuple[dict, list[dict]]:
    midpoint = np.median(dt)
    halves = (dt <= midpoint, dt > midpoint)
    baseline_auc = float(roc_auc_score(target, baseline))
    baseline_halves = [
        float(roc_auc_score(target[mask], baseline[mask])) for mask in halves
    ]
    rows = []
    for variant, raw_prediction in raw_variants.items():
        for weights_tuple in product(
            SEGMENT_WEIGHTS, repeat=len(segments.SEGMENTS)
        ):
            weights = dict(zip(segments.SEGMENTS, weights_tuple))
            prediction = segments.segmented_blend(
                baseline, raw_prediction, segment_labels, weights
            )
            auc = float(roc_auc_score(target, prediction))
            half_auc = [
                float(roc_auc_score(target[mask], prediction[mask]))
                for mask in halves
            ]
            half_gains = [
                score - base for score, base in zip(half_auc, baseline_halves)
            ]
            rows.append(
                {
                    "variant": variant,
                    "weights": weights,
                    "auc": auc,
                    "gain": auc - baseline_auc,
                    "half_gains": half_gains,
                    "min_half_gain": float(min(half_gains)),
                }
            )
    stable = [row for row in rows if row["min_half_gain"] >= 0.0]
    selected = max(
        stable, key=lambda row: (row["gain"], row["min_half_gain"])
    )
    rows.sort(
        key=lambda row: (row["gain"], row["min_half_gain"]), reverse=True
    )
    return selected, rows


def transform_variant(
    prediction: np.ndarray,
    variant: str,
    groups: dict,
    postprocess: dict,
) -> np.ndarray:
    variants = prediction_variants(prediction, groups, postprocess)
    return variants[variant]


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    manifest, arrays = load_matrix()
    train = arrays["train"]
    test = arrays["test"]
    target = arrays["target"]
    day = arrays["day"]

    oof = featureview.build_oof()
    membership_oof = pd.read_csv(
        ROOT / "honest_client_meta/oof_client_features.csv"
    )
    membership_test = pd.read_csv(
        ROOT / "honest_client_meta/test_client_features.csv"
    )
    reference, fold_groups, reference_report = build_reference_oof(
        oof, membership_oof
    )
    dev_mask = oof["fold"].eq(DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(LOCK_FOLD).to_numpy()
    dev_index = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_index = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
    if not np.all((day[dev_index] >= 75.0) & (day[dev_index] < 90.0)):
        raise RuntimeError("Dev rows do not match the purged temporal contract")
    if not np.all((day[lock_index] >= 90.0) & (day[lock_index] < 106.0)):
        raise RuntimeError("Lock rows do not match the purged temporal contract")

    selected, model_search, dev_raw = search_lgb(
        train, target, day, dev_index
    )
    lock_model, lock_raw, lock_iterations = train_lock(
        train, target, day, lock_index, selected
    )
    lock_model_path = WORK_DIR / "lock_lgb.txt"
    lock_model.booster_.save_model(lock_model_path)
    np.save(WORK_DIR / "dev_raw.npy", dev_raw.astype("float32"))
    np.save(WORK_DIR / "lock_raw.npy", lock_raw.astype("float32"))

    postprocess = reference_report["recipe"]["postprocess_locked_from_dev"]
    dev_variants = prediction_variants(
        dev_raw, fold_groups[DEV_FOLD], postprocess
    )
    selected_blend, blend_search = search_blend(
        np.asarray(target[dev_index], dtype="int8"),
        reference[DEV_FOLD],
        dev_variants,
        segments.segment_labels(
            membership_oof.loc[dev_mask].reset_index(drop=True)
        ),
        np.asarray(day[dev_index]),
    )
    lock_variant = transform_variant(
        lock_raw,
        selected_blend["variant"],
        fold_groups[LOCK_FOLD],
        postprocess,
    )
    lock_prediction = segments.segmented_blend(
        reference[LOCK_FOLD],
        lock_variant,
        segments.segment_labels(
            membership_oof.loc[lock_mask].reset_index(drop=True)
        ),
        selected_blend["weights"],
    )
    y_lock = np.asarray(target[lock_index], dtype="int8")
    lock_raw_auc = float(roc_auc_score(y_lock, lock_raw))
    lock_auc = float(roc_auc_score(y_lock, lock_prediction))
    reference_lock_auc = float(reference_report["metrics"][LOCK_FOLD]["auc"])
    accepted = bool(selected_blend["gain"] > 0 and lock_auc > reference_lock_auc)

    final_iterations = max(50, int(np.ceil(lock_iterations * 1.15)))
    final_model = lgb.LGBMClassifier(
        **{
            **LGB_PARAMS,
            **LGB_CONFIGS[int(selected["config_index"])],
            "n_estimators": final_iterations,
        }
    )
    final_model.fit(train, target, callbacks=[lgb.log_evaluation(0)])
    final_model_path = WORK_DIR / "final_lgb.txt"
    final_model.booster_.save_model(final_model_path)
    test_raw = final_model.predict_proba(test)[:, 1]
    np.save(WORK_DIR / "test_raw.npy", test_raw.astype("float32"))
    test_variant = transform_variant(
        test_raw,
        selected_blend["variant"],
        reference_report["test_groups"],
        postprocess,
    )
    baseline_test = pd.read_csv(ROOT / "submission_honest_featureview_client.csv")
    if not np.array_equal(
        baseline_test["TransactionID"].to_numpy(), arrays["test_id"]
    ):
        raise RuntimeError("Full-row test matrix and reference submission differ")
    test_prediction = segments.segmented_blend(
        baseline_test[TARGET].to_numpy(dtype="float64"),
        test_variant,
        segments.segment_labels(membership_test),
        selected_blend["weights"],
    )
    output = baseline_test[["TransactionID"]].copy()
    output[TARGET] = test_prediction
    output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": "official train/test covariates; official train labels only",
        "selection": "model and blend on days 75-90; days 90-105 one-time lock",
        "feature_recipe_version": FEATURE_RECIPE_VERSION,
        "feature_count": len(manifest["features"]),
        "selected_model": selected,
        "model_search": model_search,
        "lock_iterations": lock_iterations,
        "final_iterations": final_iterations,
        "selected_blend": selected_blend,
        "blend_search_top": blend_search[:40],
        "reference": {
            "dev_auc": reference_report["metrics"][DEV_FOLD]["auc"],
            "lock_auc": reference_lock_auc,
        },
        "candidate": {
            "dev_raw_auc": selected["dev_auc"],
            "lock_raw_auc": lock_raw_auc,
            "lock_blend_auc": lock_auc,
            "lock_gain": lock_auc - reference_lock_auc,
        },
        "accepted": accepted,
        "models": {
            "lock": {
                "path": str(lock_model_path),
                "sha256": file_sha256(lock_model_path),
            },
            "final": {
                "path": str(final_model_path),
                "sha256": file_sha256(final_model_path),
            },
        },
        "output": OUTPUT_PATH.name,
        "output_sha256": file_sha256(OUTPUT_PATH),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_model": selected,
                "selected_blend": selected_blend,
                "reference": report["reference"],
                "candidate": report["candidate"],
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

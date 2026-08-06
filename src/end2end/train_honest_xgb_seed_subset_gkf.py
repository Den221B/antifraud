"""Select an XGB seed subset on temporal dev and preserve month-GroupKFold."""

from __future__ import annotations

import gc
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

import finalize_honest_xgb_magic_blend as xgb_final
import refine_honest_client_segments as segments
import search_honest_featureview_meta as featureview
import search_honest_fullrow_lgb as fullrow
import train_honest_client_meta as client
import train_honest_magic_heavy_stack as heavy
import train_honest_xgb_magic as magic


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_xgb_seed_subset_gkf"
MODEL_DIR = WORK_DIR / "models"
REPORT_PATH = WORK_DIR / "report.json"
OUTPUT_PATH = ROOT / "submission_honest_xgb_seed_subset_gkf.csv"
SEEDS = (2027, 3407, 7907, 12011)
PREDICTION_CACHE_VERSION = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(roc_auc_score(target, prediction))


def load_xgb_prediction(
    model_path: Path,
    matrix: np.ndarray,
    row_index: np.ndarray,
) -> np.ndarray:
    # ``missing`` is a sklearn-wrapper inference parameter and is not restored
    # by load_model, so construct the wrapper with the original recipe first.
    model = xgb.XGBClassifier(**magic.XGB_PARAMS)
    model.load_model(model_path)
    prediction = model.predict_proba(matrix[row_index])[:, 1]
    del model
    gc.collect()
    return prediction


def load_seed_predictions(
    matrix: np.ndarray,
    dev_index: np.ndarray,
    lock_index: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    dev = {
        2027: np.load(ROOT / "honest_xgb_magic/dev_prediction.npy").astype(
            "float64"
        )
    }
    lock = {
        2027: np.load(ROOT / "honest_xgb_magic/lock_prediction.npy").astype(
            "float64"
        )
    }
    for seed in SEEDS[1:]:
        dev_path = (
            WORK_DIR / f"dev_seed_{seed}_v{PREDICTION_CACHE_VERSION}.npy"
        )
        lock_path = (
            WORK_DIR / f"lock_seed_{seed}_v{PREDICTION_CACHE_VERSION}.npy"
        )
        if dev_path.exists() and lock_path.exists():
            dev[seed] = np.load(dev_path).astype("float64")
            lock[seed] = np.load(lock_path).astype("float64")
            continue
        dev[seed] = load_xgb_prediction(
            ROOT / f"honest_xgb_magic_multiseed/models/dev_seed_{seed}.json",
            matrix,
            dev_index,
        )
        lock[seed] = load_xgb_prediction(
            ROOT / f"honest_xgb_magic_multiseed/models/lock_seed_{seed}.json",
            matrix,
            lock_index,
        )
        np.save(dev_path, dev[seed].astype("float32"))
        np.save(lock_path, lock[seed].astype("float32"))
    return dev, lock


def apply_xgb_layer(
    clean_prediction: np.ndarray,
    source_prediction: np.ndarray,
    groups: dict,
    postprocess: dict,
    membership: pd.DataFrame,
    recipe: dict,
) -> np.ndarray:
    transformed = fullrow.transform_variant(
        source_prediction, recipe["variant"], groups, postprocess
    )
    return segments.segmented_blend(
        clean_prediction,
        transformed,
        segments.segment_labels(membership),
        recipe["weights"],
    )


def apply_heavy_layer(
    xgb_prediction: np.ndarray,
    cat_prediction: np.ndarray,
    lgb_prediction: np.ndarray,
    groups: dict,
    postprocess: dict,
    membership: pd.DataFrame,
    recipe: dict,
) -> np.ndarray:
    signals = heavy.source_signals(
        cat_prediction, lgb_prediction, groups, postprocess
    )
    return segments.segmented_blend(
        xgb_prediction,
        signals[recipe["variant"]],
        segments.segment_labels(membership),
        recipe["weights"],
    )


def train_seed_group_models(
    train: np.ndarray,
    test: np.ndarray,
    target: np.ndarray,
    month: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, list[dict]]:
    cache_path = WORK_DIR / f"test_seed_{seed}.npy"
    report_path = WORK_DIR / f"test_seed_{seed}.json"
    if cache_path.exists() and report_path.exists():
        return (
            np.load(cache_path).astype("float64"),
            json.loads(report_path.read_text(encoding="utf-8")),
        )
    unique_months = np.unique(month)
    splitter = GroupKFold(n_splits=len(unique_months))
    predictions = []
    rows = []
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(target)), target, groups=month)
    ):
        params = {
            **magic.XGB_PARAMS,
            "n_estimators": iterations,
            "random_state": seed + fold,
        }
        model = xgb.XGBClassifier(**params)
        started = time.time()
        model.fit(train[train_index], target[train_index], verbose=False)
        predictions.append(model.predict_proba(test)[:, 1])
        model_path = MODEL_DIR / f"seed_{seed}_fold_{fold}.json"
        model.save_model(model_path)
        valid_prediction = model.predict_proba(train[valid_index])[:, 1]
        row = {
            "seed": seed,
            "fold": fold,
            "held_month": int(np.unique(month[valid_index])[0]),
            "train_rows": int(len(train_index)),
            "valid_rows": int(len(valid_index)),
            "iterations": iterations,
            "valid_auc": auc(target[valid_index], valid_prediction),
            "minutes": (time.time() - started) / 60.0,
            "model": str(model_path.relative_to(ROOT)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        gc.collect()
    prediction = np.mean(predictions, axis=0)
    np.save(cache_path, prediction.astype("float32"))
    report_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return prediction, rows


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    manifest, arrays = magic.load_matrix()
    oof = featureview.build_oof()
    membership_oof = pd.read_csv(
        ROOT / "honest_client_meta/oof_client_features.csv"
    )
    membership_test = pd.read_csv(
        ROOT / "honest_client_meta/test_client_features.csv"
    )
    reference, fold_groups, reference_report = fullrow.build_reference_oof(
        oof, membership_oof
    )
    clean_oof, clean_report = xgb_final.reconstruct_clean_oof(
        oof, membership_oof, reference, fold_groups, reference_report
    )

    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_index = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_index = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
    y_dev = np.asarray(arrays["target"][dev_index], dtype="int8")
    y_lock = np.asarray(arrays["target"][lock_index], dtype="int8")
    day_dev = np.asarray(arrays["day"][dev_index])
    dev_membership = membership_oof.loc[dev_mask].reset_index(drop=True)
    lock_membership = membership_oof.loc[lock_mask].reset_index(drop=True)

    xgb_report = json.loads(
        (ROOT / "honest_xgb_magic/blend_report.json").read_text(
            encoding="utf-8"
        )
    )
    heavy_report = json.loads(
        (ROOT / "honest_magic_heavy_stack/report.json").read_text(
            encoding="utf-8"
        )
    )
    xgb_recipe = xgb_report["selected"]
    heavy_recipe = heavy_report["selected_blend"]
    cat_name = heavy_report["selected_cat"]["name"]
    lgb_name = heavy_report["selected_lgb"]["name"]
    dev_cat = np.load(WORK_DIR.parent / f"honest_magic_heavy_stack/dev_cat_{cat_name}.npy")
    dev_lgb = np.load(WORK_DIR.parent / f"honest_magic_heavy_stack/dev_lgb_{lgb_name}.npy")
    lock_cat = np.load(ROOT / "honest_magic_heavy_stack/lock_cat.npy")
    lock_lgb = np.load(ROOT / "honest_magic_heavy_stack/lock_lgb.npy")

    dev_seed, lock_seed = load_seed_predictions(
        arrays["train"], dev_index, lock_index
    )
    current_dev_xgb = apply_xgb_layer(
        clean_oof[client.META_DEV_FOLD],
        dev_seed[2027],
        fold_groups[client.META_DEV_FOLD],
        clean_report["postprocess"],
        dev_membership,
        xgb_recipe,
    )
    current_lock_xgb = apply_xgb_layer(
        clean_oof[client.META_LOCK_FOLD],
        lock_seed[2027],
        fold_groups[client.META_LOCK_FOLD],
        clean_report["postprocess"],
        lock_membership,
        xgb_recipe,
    )
    current_dev = apply_heavy_layer(
        current_dev_xgb,
        dev_cat,
        dev_lgb,
        fold_groups[client.META_DEV_FOLD],
        clean_report["postprocess"],
        dev_membership,
        heavy_recipe,
    )
    current_lock = apply_heavy_layer(
        current_lock_xgb,
        lock_cat,
        lock_lgb,
        fold_groups[client.META_LOCK_FOLD],
        clean_report["postprocess"],
        lock_membership,
        heavy_recipe,
    )
    current_dev_auc = auc(y_dev, current_dev)
    current_lock_auc = auc(y_lock, current_lock)
    if abs(current_dev_auc - float(heavy_report["dev_auc"])) > 1e-12:
        raise RuntimeError("Could not reproduce heavy-stack dev prediction")
    if abs(current_lock_auc - float(heavy_report["lock_auc"])) > 1e-12:
        raise RuntimeError("Could not reproduce heavy-stack lock prediction")

    midpoint = np.median(day_dev)
    halves = (day_dev <= midpoint, day_dev > midpoint)
    current_half_auc = [auc(y_dev[mask], current_dev[mask]) for mask in halves]
    subset_rows = []
    subset_predictions = {}
    for size in range(1, len(SEEDS) + 1):
        for subset in combinations(SEEDS, size):
            source = np.mean([dev_seed[seed] for seed in subset], axis=0)
            candidate_xgb = apply_xgb_layer(
                clean_oof[client.META_DEV_FOLD],
                source,
                fold_groups[client.META_DEV_FOLD],
                clean_report["postprocess"],
                dev_membership,
                xgb_recipe,
            )
            candidate = apply_heavy_layer(
                candidate_xgb,
                dev_cat,
                dev_lgb,
                fold_groups[client.META_DEV_FOLD],
                clean_report["postprocess"],
                dev_membership,
                heavy_recipe,
            )
            score = auc(y_dev, candidate)
            half_auc = [auc(y_dev[mask], candidate[mask]) for mask in halves]
            half_gains = [
                value - base for value, base in zip(half_auc, current_half_auc)
            ]
            row = {
                "seeds": list(subset),
                "source_auc": auc(y_dev, source),
                "auc": score,
                "gain": score - current_dev_auc,
                "half_gains": half_gains,
                "min_half_gain": float(min(half_gains)),
            }
            subset_rows.append(row)
            subset_predictions[subset] = candidate
    stable = [row for row in subset_rows if row["min_half_gain"] >= 0.0]
    selected = max(
        stable, key=lambda row: (row["gain"], row["min_half_gain"])
    )
    subset_rows.sort(
        key=lambda row: (row["gain"], row["min_half_gain"]), reverse=True
    )
    selected_seeds = tuple(selected["seeds"])

    lock_source = np.mean([lock_seed[seed] for seed in selected_seeds], axis=0)
    lock_xgb = apply_xgb_layer(
        clean_oof[client.META_LOCK_FOLD],
        lock_source,
        fold_groups[client.META_LOCK_FOLD],
        clean_report["postprocess"],
        lock_membership,
        xgb_recipe,
    )
    lock_candidate = apply_heavy_layer(
        lock_xgb,
        lock_cat,
        lock_lgb,
        fold_groups[client.META_LOCK_FOLD],
        clean_report["postprocess"],
        lock_membership,
        heavy_recipe,
    )
    lock_auc = auc(y_lock, lock_candidate)
    accepted = bool(selected["gain"] > 0.0 and lock_auc > current_lock_auc)

    final_rows = []
    if accepted:
        iterations = int(xgb_report["final_group_iterations"])
        test_members = []
        for seed in selected_seeds:
            if seed == 2027:
                test_prediction = np.load(
                    ROOT / "honest_xgb_magic/test_prediction.npy"
                ).astype("float64")
                rows = [{"seed": seed, "cached_champion": True}]
            else:
                test_prediction, rows = train_seed_group_models(
                    arrays["train"],
                    arrays["test"],
                    arrays["target"],
                    arrays["month"],
                    seed,
                    iterations,
                )
            test_members.append(test_prediction)
            final_rows.extend(rows)
        test_source = np.mean(test_members, axis=0)
        np.save(WORK_DIR / "test_prediction.npy", test_source.astype("float32"))

        clean_submission = pd.read_csv(
            ROOT / "submission_honest_cleanv2_blend.csv"
        )
        if not np.array_equal(
            clean_submission["TransactionID"].to_numpy(), arrays["test_id"]
        ):
            raise RuntimeError("Seed-subset test rows differ from clean blend")
        test_xgb = apply_xgb_layer(
            clean_submission[client.TARGET].to_numpy(dtype="float64"),
            test_source,
            reference_report["test_groups"],
            clean_report["postprocess"],
            membership_test,
            xgb_recipe,
        )
        test_cat = np.load(ROOT / "honest_magic_heavy_stack/test_cat.npy")
        test_lgb = np.load(ROOT / "honest_magic_heavy_stack/test_lgb.npy")
        prediction = apply_heavy_layer(
            test_xgb,
            test_cat,
            test_lgb,
            reference_report["test_groups"],
            clean_report["postprocess"],
            membership_test,
            heavy_recipe,
        )
        output = clean_submission[["TransactionID"]].copy()
        output[client.TARGET] = prediction
        output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": (
            "official train/test covariates; official train labels only; "
            "official train labels only"
        ),
        "official_hashes": manifest["official_hashes"],
        "selection": (
            "15 fixed seed subsets on dev; unchanged XGB/heavy recipes; "
            "one-time lock"
        ),
        "seeds": list(SEEDS),
        "prediction_cache_version": PREDICTION_CACHE_VERSION,
        "champion_auc": {
            "dev": current_dev_auc,
            "lock": current_lock_auc,
        },
        "subset_search": subset_rows,
        "selected": selected,
        "candidate_lock_auc": lock_auc,
        "lock_gain": lock_auc - current_lock_auc,
        "accepted": accepted,
        "final_models": final_rows,
        "output": OUTPUT_PATH.name if accepted else None,
        "output_sha256": file_sha256(OUTPUT_PATH) if accepted else None,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Rebuild Chris Deotte's XGB magic recipe on the official local data only."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from fraud_features import read_and_merge
import refine_honest_client_segments as segments
import search_honest_featureview_meta as featureview
import search_honest_fullrow_lgb as fullrow
import train_honest_client_meta as client


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_xgb_magic"
CACHE_DIR = WORK_DIR / "matrix"
MODEL_DIR = WORK_DIR / "models"
REPORT_PATH = WORK_DIR / "report.json"
OUTPUT_PATH = ROOT / "submission_honest_xgb_magic.csv"
TARGET = client.TARGET
DAY_SECONDS = 86_400.0
RECIPE_VERSION = "deotte_xgb_magic_v1"

V_NUMBERS = (
    1, 3, 4, 6, 8, 11,
    13, 14, 17, 20, 23, 26, 27, 30,
    36, 37, 40, 41, 44, 47, 48,
    54, 56, 59, 62, 65, 67, 68, 70,
    76, 78, 80, 82, 86, 88, 89, 91,
    107, 108, 111, 115, 117, 120, 121, 123,
    124, 127, 129, 130, 136,
    138, 139, 142, 147, 156, 162,
    165, 160, 166,
    178, 176, 173, 182,
    187, 203, 205, 207, 215,
    169, 171, 175, 180, 185, 188, 198, 210, 209,
    218, 223, 224, 226, 228, 229, 235,
    240, 258, 257, 253, 252, 260, 261,
    264, 266, 267, 274, 277,
    220, 221, 234, 238, 250, 271,
    294, 284, 285, 286, 291, 297,
    303, 305, 307, 309, 310, 320,
    281, 283, 289, 296, 301, 314,
)
TRANSACTION_COLUMNS = (
    "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
    *(f"M{i}" for i in range(1, 10)),
    *(f"V{i}" for i in V_NUMBERS),
)
FAILED_TIME_FEATURES = {
    "C3", "M5", "id_08", "id_33", "card4", "id_07", "id_14",
    "id_21", "id_30", "id_32", "id_34",
    *(f"id_{i:02d}" for i in range(22, 28)),
}
REMOVED_D = {"D6", "D7", "D8", "D9", "D12", "D13", "D14"}
XGB_PARAMS = {
    "n_estimators": 3_500,
    "max_depth": 12,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "missing": -1,
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_bin": 256,
    "objective": "binary:logistic",
    "random_state": 2027,
    "n_jobs": -1,
}


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


def cache_paths() -> dict[str, Path]:
    return {
        name: CACHE_DIR / f"{name}.npy"
        for name in (
            "train", "test", "target", "day", "month", "train_id", "test_id"
        )
    }


def joint_factorize(train: pd.DataFrame, test: pd.DataFrame, column: str) -> None:
    combined = pd.concat(
        [train[column], test[column]], ignore_index=True, copy=False
    )
    codes, _ = pd.factorize(combined, sort=True)
    train[column] = codes[: len(train)].astype("int32")
    test[column] = codes[len(train) :].astype("int32")


def frequency_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    names = []
    for column in columns:
        combined = pd.concat(
            [train[column], test[column]], ignore_index=True, copy=False
        )
        frequency = combined.value_counts(dropna=True, normalize=True)
        name = f"{column}_FE"
        train[name] = train[column].map(frequency).fillna(-1).astype("float32")
        test[name] = test[column].map(frequency).fillna(-1).astype("float32")
        names.append(name)
    return names


def combine_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    left: str,
    right: str,
) -> str:
    name = f"{left}_{right}"
    train[name] = train[left].astype(str).str.cat(train[right].astype(str), sep="_")
    test[name] = test[left].astype(str).str.cat(test[right].astype(str), sep="_")
    joint_factorize(train, test, name)
    return name


def aggregate_values(
    train: pd.DataFrame,
    test: pd.DataFrame,
    main_columns: list[str],
    group_columns: list[str],
    aggregations: tuple[str, ...],
    use_na: bool,
) -> list[str]:
    names = []
    for main_column in main_columns:
        for group_column in group_columns:
            combined = pd.concat(
                [
                    train[[group_column, main_column]],
                    test[[group_column, main_column]],
                ],
                ignore_index=True,
                copy=False,
            ).copy()
            if use_na:
                combined.loc[combined[main_column].eq(-1), main_column] = np.nan
            grouped = combined.groupby(group_column, dropna=False)[main_column]
            for aggregation in aggregations:
                name = f"{main_column}_{group_column}_{aggregation}"
                mapping = grouped.agg(aggregation)
                train[name] = (
                    train[group_column].map(mapping).fillna(-1).astype("float32")
                )
                test[name] = (
                    test[group_column].map(mapping).fillna(-1).astype("float32")
                )
                names.append(name)
    return names


def aggregate_nunique(
    train: pd.DataFrame,
    test: pd.DataFrame,
    main_columns: list[str],
    group_columns: list[str],
) -> list[str]:
    names = []
    for main_column in main_columns:
        for group_column in group_columns:
            combined = pd.concat(
                [
                    train[[group_column, main_column]],
                    test[[group_column, main_column]],
                ],
                ignore_index=True,
                copy=False,
            )
            mapping = combined.groupby(group_column, dropna=False)[
                main_column
            ].nunique(dropna=True)
            name = f"{group_column}_{main_column}_ct"
            train[name] = train[group_column].map(mapping).fillna(0).astype("float32")
            test[name] = test[group_column].map(mapping).fillna(0).astype("float32")
            names.append(name)
    return names


def transaction_month(transaction_dt: pd.Series) -> np.ndarray:
    timestamp = pd.Timestamp("2017-11-30") + pd.to_timedelta(
        transaction_dt, unit="s"
    )
    return ((timestamp.dt.year - 2017) * 12 + timestamp.dt.month).to_numpy(
        dtype="int16"
    )


def build_cache(hashes: dict[str, str]) -> dict:
    print("Building the exact XGB-magic feature matrix...", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_all = read_and_merge(ROOT, "train").reset_index(drop=True)
    test_all = read_and_merge(ROOT, "test").reset_index(drop=True)
    if TARGET not in train_all or TARGET in test_all:
        raise RuntimeError("Unexpected target placement in official files")
    identity_columns = [
        column for column in train_all.columns if column.startswith("id_")
    ]
    selected_input = [
        column
        for column in (*TRANSACTION_COLUMNS, *identity_columns)
        if column in train_all and column in test_all
    ]
    train = train_all[selected_input].copy()
    test = test_all[selected_input].copy()
    target = train_all[TARGET].to_numpy(dtype="int8")
    day = (train["TransactionDT"].to_numpy(dtype="float64") / DAY_SECONDS).astype(
        "float32"
    )
    month = transaction_month(train["TransactionDT"])
    test_month = transaction_month(test["TransactionDT"])
    train_id = train_all["TransactionID"].to_numpy(dtype="int32")
    test_id = test_all["TransactionID"].to_numpy(dtype="int32")
    del train_all, test_all
    gc.collect()

    for number in range(1, 16):
        if number in {1, 2, 3, 5, 9}:
            continue
        column = f"D{number}"
        train[column] = train[column] - train["TransactionDT"] / DAY_SECONDS
        test[column] = test[column] - test["TransactionDT"] / DAY_SECONDS

    for column in list(train.columns):
        if not pd.api.types.is_numeric_dtype(train[column]):
            joint_factorize(train, test, column)
        elif column not in {"TransactionAmt", "TransactionDT"}:
            minimum = min(train[column].min(skipna=True), test[column].min(skipna=True))
            if pd.isna(minimum):
                minimum = 0.0
            train[column] = (train[column] - minimum).fillna(-1)
            test[column] = (test[column] - minimum).fillna(-1)

    train["cents"] = (
        train["TransactionAmt"] - np.floor(train["TransactionAmt"])
    ).astype("float32")
    test["cents"] = (
        test["TransactionAmt"] - np.floor(test["TransactionAmt"])
    ).astype("float32")
    frequency_encode(
        train, test, ["addr1", "card1", "card2", "card3", "P_emaildomain"]
    )
    card_addr = combine_columns(train, test, "card1", "addr1")
    card_addr_email = combine_columns(train, test, card_addr, "P_emaildomain")
    frequency_encode(train, test, [card_addr, card_addr_email])
    aggregate_values(
        train,
        test,
        ["TransactionAmt", "D9", "D11"],
        ["card1", card_addr, card_addr_email],
        ("mean", "std"),
        use_na=True,
    )

    train["DT_M"] = month
    test["DT_M"] = test_month
    train["day"] = day
    test["day"] = (
        test["TransactionDT"].to_numpy(dtype="float64") / DAY_SECONDS
    ).astype("float32")
    train["uid"] = train[card_addr].astype(str).str.cat(
        np.floor(train["day"] - train["D1"]).astype(str), sep="_"
    )
    test["uid"] = test[card_addr].astype(str).str.cat(
        np.floor(test["day"] - test["D1"]).astype(str), sep="_"
    )
    frequency_encode(train, test, ["uid"])
    aggregate_values(
        train,
        test,
        ["TransactionAmt", "D4", "D9", "D10", "D15"],
        ["uid"],
        ("mean", "std"),
        use_na=True,
    )
    aggregate_values(
        train,
        test,
        [f"C{i}" for i in range(1, 15) if i != 3],
        ["uid"],
        ("mean",),
        use_na=True,
    )
    aggregate_values(
        train,
        test,
        [f"M{i}" for i in range(1, 10)],
        ["uid"],
        ("mean",),
        use_na=True,
    )
    aggregate_nunique(
        train,
        test,
        ["P_emaildomain", "dist1", "DT_M", "id_02", "cents"],
        ["uid"],
    )
    aggregate_values(
        train, test, ["C14"], ["uid"], ("std",), use_na=True
    )
    aggregate_nunique(train, test, ["C13", "V314"], ["uid"])
    aggregate_nunique(
        train, test, ["V127", "V136", "V309", "V307", "V320"], ["uid"]
    )
    train["outsider15"] = (train["D1"] - train["D15"]).abs().gt(3).astype("int8")
    test["outsider15"] = (test["D1"] - test["D15"]).abs().gt(3).astype("int8")

    excluded = {
        "TransactionDT", "DT_M", "day", "uid", *REMOVED_D, *FAILED_TIME_FEATURES
    }
    features = [column for column in train.columns if column not in excluded]
    missing_numeric = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(train[column])
    ]
    if missing_numeric:
        raise RuntimeError(f"Non-numeric XGB features remain: {missing_numeric[:10]}")
    train_matrix = train[features].astype("float32").to_numpy(copy=True)
    test_matrix = test[features].astype("float32").to_numpy(copy=True)
    paths = cache_paths()
    for name, values in (
        ("train", train_matrix),
        ("test", test_matrix),
        ("target", target),
        ("day", day),
        ("month", month),
        ("train_id", train_id),
        ("test_id", test_id),
    ):
        np.save(paths[name], values)
    manifest = {
        "recipe_version": RECIPE_VERSION,
        "official_hashes": hashes,
        "features": features,
        "train_shape": list(train_matrix.shape),
        "test_shape": list(test_matrix.shape),
        "target_sum": int(target.sum()),
        "source": "https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600",
    }
    (CACHE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    del train, test, train_matrix, test_matrix
    gc.collect()
    return manifest


def load_matrix() -> tuple[dict, dict[str, np.ndarray]]:
    hashes = official_hashes()
    paths = cache_paths()
    manifest_path = CACHE_DIR / "manifest.json"
    manifest = None
    if manifest_path.exists() and all(path.exists() for path in paths.values()):
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            candidate.get("recipe_version") == RECIPE_VERSION
            and candidate.get("official_hashes") == hashes
        ):
            manifest = candidate
            print("Loading verified XGB-magic matrix cache", flush=True)
    if manifest is None:
        manifest = build_cache(hashes)
    arrays = {
        name: np.load(path, mmap_mode="r") for name, path in paths.items()
    }
    return manifest, arrays


def fit_temporal_source(
    train: np.ndarray,
    target: np.ndarray,
    day: np.ndarray,
    train_end: float,
    valid_index: np.ndarray,
    model_path: Path,
    iterations: int | None,
) -> tuple[np.ndarray, int, float]:
    train_index = np.flatnonzero(day < train_end)
    params = {**XGB_PARAMS}
    if iterations is None:
        params["early_stopping_rounds"] = 150
    else:
        params["n_estimators"] = iterations
    model = xgb.XGBClassifier(**params)
    fit_kwargs = {
        "X": train[train_index],
        "y": target[train_index],
        "verbose": False,
    }
    if iterations is None:
        fit_kwargs["eval_set"] = [(train[valid_index], target[valid_index])]
    started = time.time()
    model.fit(**fit_kwargs)
    prediction = model.predict_proba(train[valid_index])[:, 1]
    best = int(getattr(model, "best_iteration", params["n_estimators"] - 1) + 1)
    model.save_model(model_path)
    minutes = (time.time() - started) / 60.0
    del model
    gc.collect()
    return prediction, best, minutes


def train_group_models(
    train: np.ndarray,
    test: np.ndarray,
    target: np.ndarray,
    month: np.ndarray,
    fixed_iterations: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    unique_months = np.unique(month)
    splitter = GroupKFold(n_splits=len(unique_months))
    predictions = []
    rows = []
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(target)), target, groups=month)
    ):
        held_month = int(np.unique(month[valid_index])[0])
        params = {
            **XGB_PARAMS,
            "random_state": XGB_PARAMS["random_state"] + fold,
        }
        if fixed_iterations is None:
            params["early_stopping_rounds"] = 180
        else:
            params["n_estimators"] = fixed_iterations
        model = xgb.XGBClassifier(**params)
        started = time.time()
        fit_kwargs = {
            "X": train[train_index],
            "y": target[train_index],
            "verbose": False,
        }
        if fixed_iterations is None:
            fit_kwargs["eval_set"] = [
                (train[valid_index], target[valid_index])
            ]
        model.fit(**fit_kwargs)
        prediction = model.predict_proba(test)[:, 1]
        predictions.append(prediction)
        model_path = MODEL_DIR / f"month_fold_{fold}.json"
        model.save_model(model_path)
        rows.append(
            {
                "fold": fold,
                "held_month": held_month,
                "train_rows": int(len(train_index)),
                "valid_rows": int(len(valid_index)),
                "best_iteration": int(
                    model.best_iteration + 1
                    if fixed_iterations is None
                    else fixed_iterations
                ),
                "valid_auc": float(
                    roc_auc_score(
                        target[valid_index],
                        model.predict_proba(train[valid_index])[:, 1],
                    )
                ),
                "minutes": (time.time() - started) / 60.0,
                "model": str(model_path),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        del model
        gc.collect()
    return np.mean(predictions, axis=0), rows


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    manifest, arrays = load_matrix()
    train = arrays["train"]
    target = arrays["target"]
    day = arrays["day"]
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
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_index = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_index = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")

    dev_prediction, dev_iterations, dev_minutes = fit_temporal_source(
        train,
        target,
        day,
        45.0,
        dev_index,
        MODEL_DIR / "dev.json",
        iterations=None,
    )
    lock_iterations = max(50, int(np.ceil(dev_iterations * 1.15)))
    lock_prediction, _, lock_minutes = fit_temporal_source(
        train,
        target,
        day,
        60.0,
        lock_index,
        MODEL_DIR / "lock.json",
        iterations=lock_iterations,
    )
    np.save(WORK_DIR / "dev_prediction.npy", dev_prediction.astype("float32"))
    np.save(WORK_DIR / "lock_prediction.npy", lock_prediction.astype("float32"))

    postprocess = reference_report["recipe"]["postprocess_locked_from_dev"]
    dev_variants = fullrow.prediction_variants(
        dev_prediction, fold_groups[client.META_DEV_FOLD], postprocess
    )
    selected_blend, blend_search = fullrow.search_blend(
        np.asarray(target[dev_index], dtype="int8"),
        reference[client.META_DEV_FOLD],
        dev_variants,
        segments.segment_labels(
            membership_oof.loc[dev_mask].reset_index(drop=True)
        ),
        np.asarray(day[dev_index]),
    )
    lock_variant = fullrow.transform_variant(
        lock_prediction,
        selected_blend["variant"],
        fold_groups[client.META_LOCK_FOLD],
        postprocess,
    )
    lock_blend = segments.segmented_blend(
        reference[client.META_LOCK_FOLD],
        lock_variant,
        segments.segment_labels(
            membership_oof.loc[lock_mask].reset_index(drop=True)
        ),
        selected_blend["weights"],
    )
    y_dev = np.asarray(target[dev_index], dtype="int8")
    y_lock = np.asarray(target[lock_index], dtype="int8")
    source_metrics = {
        "dev_auc": float(roc_auc_score(y_dev, dev_prediction)),
        "lock_auc": float(roc_auc_score(y_lock, lock_prediction)),
        "lock_blend_auc": float(roc_auc_score(y_lock, lock_blend)),
    }
    previous_lock = float(
        json.loads(
            (ROOT / "honest_cleanv2_blend/report.json").read_text(encoding="utf-8")
        )["candidate_lock_auc"]
    )
    accepted = bool(
        selected_blend["gain"] > 0
        and source_metrics["lock_blend_auc"] > previous_lock
    )

    group_models = []
    if accepted:
        test_prediction, group_models = train_group_models(
            train, arrays["test"], target, arrays["month"]
        )
        np.save(WORK_DIR / "test_prediction.npy", test_prediction.astype("float32"))
        test_variant = fullrow.transform_variant(
            test_prediction,
            selected_blend["variant"],
            reference_report["test_groups"],
            postprocess,
        )
        baseline_test = pd.read_csv(
            ROOT / "submission_honest_featureview_client.csv"
        )
        if not np.array_equal(
            baseline_test["TransactionID"].to_numpy(), arrays["test_id"]
        ):
            raise RuntimeError("XGB test rows differ from the reference submission")
        output_prediction = segments.segmented_blend(
            baseline_test[TARGET].to_numpy(dtype="float64"),
            test_variant,
            segments.segment_labels(membership_test),
            selected_blend["weights"],
        )
        output = baseline_test[["TransactionID"]].copy()
        output[TARGET] = output_prediction
        output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": "official train/test covariates; official train labels only",
        "source_recipe": manifest["source"],
        "selection": "published XGB recipe; blend on days 75-90; days 90-105 lock",
        "features": len(manifest["features"]),
        "params": XGB_PARAMS,
        "dev_iterations": dev_iterations,
        "lock_iterations": lock_iterations,
        "dev_minutes": dev_minutes,
        "lock_minutes": lock_minutes,
        "source_metrics": source_metrics,
        "selected_blend": selected_blend,
        "blend_search_top": blend_search[:30],
        "previous_cleanv2_lock_auc": previous_lock,
        "lock_gain": source_metrics["lock_blend_auc"] - previous_lock,
        "accepted": accepted,
        "group_models": group_models,
        "output": OUTPUT_PATH.name if accepted else None,
        "output_sha256": file_sha256(OUTPUT_PATH) if accepted else None,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

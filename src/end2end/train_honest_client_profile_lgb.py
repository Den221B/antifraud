"""Client-profile LightGBM trained with purged forward-time validation."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import finalize_honest_xgb_magic_blend as xgb_final
import refine_honest_client_segments as segments
import search_honest_client_pooling as pooling
import search_honest_featureview_meta as featureview
import search_honest_fullrow_lgb as fullrow
import train_honest_client_meta as client
import train_honest_xgb_magic as magic
import train_honest_xgb_seed_subset_gkf as seed_subset


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_client_profile_lgb"
MODEL_DIR = WORK_DIR / "models"
REPORT_PATH = WORK_DIR / "report.json"
OUTPUT_PATH = ROOT / "submission_honest_client_profile_lgb.csv"
DEV_TRAIN_END = 45.0
LOCK_TRAIN_END = 60.0
ITERATION_SCALE = 1.15
MIN_REQUIRED_GAIN = 0.0001
MEAN_FEATURES = 96
DETAIL_FEATURES = 32
SEED = 5903

LGB_BASE = {
    "n_estimators": 2_500,
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "subsample": 0.84,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_alpha": 1.0,
    "reg_lambda": 12.0,
    "max_bin": 255,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
LGB_CONFIGS = (
    {
        "name": "profile_leaf31",
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 160,
    },
    {
        "name": "profile_leaf63",
        "num_leaves": 63,
        "max_depth": 9,
        "min_child_samples": 260,
    },
    {
        "name": "profile_extra63",
        "num_leaves": 63,
        "max_depth": 9,
        "min_child_samples": 220,
        "extra_trees": True,
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(roc_auc_score(target, prediction))


def select_feature_indices(feature_names: list[str]) -> tuple[list[int], list[int]]:
    model = xgb.XGBClassifier(**magic.XGB_PARAMS)
    model.load_model(ROOT / "honest_xgb_magic/models/dev.json")
    importance = model.get_booster().get_score(importance_type="gain")
    ranked = [
        index
        for index, _ in sorted(
            (
                (int(name[1:]), float(value))
                for name, value in importance.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    manual_names = (
        "TransactionAmt",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "uid_FE",
        "TransactionAmt_uid_mean",
        "TransactionAmt_uid_std",
        "D4_uid_mean",
        "D10_uid_mean",
        "D15_uid_mean",
        "C13_uid_mean",
        "C14_uid_std",
    )
    manual = [
        feature_names.index(name) for name in manual_names if name in feature_names
    ]
    mean_indices = list(dict.fromkeys([*manual, *ranked]))[:MEAN_FEATURES]
    detail_indices = mean_indices[:DETAIL_FEATURES]
    del model
    gc.collect()
    return mean_indices, detail_indices


def build_profile_matrix(
    matrix: np.ndarray,
    row_index: np.ndarray | None,
    uid: pd.Series,
    day: np.ndarray,
    feature_names: list[str],
    mean_indices: list[int],
    detail_indices: list[int],
) -> tuple[np.ndarray, list[str], dict]:
    if row_index is None:
        values = np.asarray(matrix[:, mean_indices], dtype="float32").copy()
        uid_rows = uid.reset_index(drop=True)
        day_rows = np.asarray(day, dtype="float32")
    else:
        values = np.asarray(
            matrix[np.ix_(row_index, mean_indices)], dtype="float32"
        ).copy()
        uid_rows = uid.iloc[row_index].reset_index(drop=True)
        day_rows = np.asarray(day[row_index], dtype="float32")
    values[values == -1.0] = np.nan
    codes, unique_uid = pd.factorize(uid_rows, sort=False)
    if np.any(codes < 0):
        raise RuntimeError("Unexpected missing UID in client profile")
    mean_names = [feature_names[index] for index in mean_indices]
    detail_names = [feature_names[index] for index in detail_indices]
    frame = pd.DataFrame(values, columns=mean_names, copy=False)
    grouped = frame.groupby(codes, sort=False, observed=True)
    group_mean = grouped.mean().to_numpy(dtype="float32")

    detail_positions = [mean_indices.index(index) for index in detail_indices]
    detail = frame.iloc[:, detail_positions]
    detail_grouped = detail.groupby(codes, sort=False, observed=True)
    group_std = detail_grouped.std().to_numpy(dtype="float32")
    group_min = detail_grouped.min().to_numpy(dtype="float32")
    group_max = detail_grouped.max().to_numpy(dtype="float32")
    group_first = detail_grouped.first().to_numpy(dtype="float32")
    group_last = detail_grouped.last().to_numpy(dtype="float32")

    counts = np.bincount(codes, minlength=len(unique_uid)).astype("float32")
    day_frame = pd.DataFrame({"group": codes, "day": day_rows})
    day_stats = day_frame.groupby("group", sort=False, observed=True)["day"].agg(
        ["min", "max", "std"]
    )
    span = (day_stats["max"] - day_stats["min"]).to_numpy(dtype="float32")
    day_std = day_stats["std"].to_numpy(dtype="float32")
    group_meta = np.column_stack(
        [
            counts,
            np.log1p(counts),
            span,
            day_std,
            counts / (span + 1.0),
        ]
    ).astype("float32")
    group_matrix = np.column_stack(
        [
            group_mean,
            group_std,
            group_min,
            group_max,
            group_first,
            group_last,
            group_meta,
        ]
    ).astype("float32")
    row_matrix = group_matrix[codes]
    names = [
        *(f"mean_{name}" for name in mean_names),
        *(f"std_{name}" for name in detail_names),
        *(f"min_{name}" for name in detail_names),
        *(f"max_{name}" for name in detail_names),
        *(f"first_{name}" for name in detail_names),
        *(f"last_{name}" for name in detail_names),
        "client_count",
        "client_log_count",
        "client_day_span",
        "client_day_std",
        "client_transactions_per_day",
    ]
    report = {
        "rows": len(row_matrix),
        "clients": len(unique_uid),
        "features": len(names),
        "repeated_row_rate": float(np.mean(counts[codes] > 1)),
        "max_client_rows": int(counts.max()),
    }
    del frame, detail, grouped, detail_grouped, group_matrix, values
    gc.collect()
    return row_matrix, names, report


def fit_lgb(
    train: np.ndarray,
    target: np.ndarray,
    predict: np.ndarray,
    predict_target: np.ndarray | None,
    config: dict,
    model_path: Path,
    fixed_iterations: int | None,
) -> tuple[np.ndarray, int, float]:
    params = {
        **LGB_BASE,
        **{key: value for key, value in config.items() if key != "name"},
    }
    if fixed_iterations is not None:
        params["n_estimators"] = fixed_iterations
    model = lgb.LGBMClassifier(**params)
    fit_kwargs = {"X": train, "y": target}
    if fixed_iterations is None:
        fit_kwargs.update(
            {
                "eval_set": [(predict, predict_target)],
                "eval_metric": "auc",
                "callbacks": [
                    lgb.early_stopping(180, verbose=False),
                    lgb.log_evaluation(200),
                ],
            }
        )
    else:
        fit_kwargs["callbacks"] = [lgb.log_evaluation(0)]
    started = time.time()
    model.fit(**fit_kwargs)
    best = int(model.best_iteration_ or params["n_estimators"])
    prediction = model.predict_proba(predict, num_iteration=best)[:, 1]
    model.booster_.save_model(model_path, num_iteration=best)
    minutes = (time.time() - started) / 60.0
    del model
    gc.collect()
    return prediction, best, minutes


def model_signals(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    signals = dict(predictions)
    names = list(predictions)
    ranks = {name: client.rank(predictions[name]) for name in names}
    signals["all_probability_mean"] = np.mean(
        [predictions[name] for name in names], axis=0
    )
    signals["all_rank_mean"] = np.mean([ranks[name] for name in names], axis=0)
    signals["all_rank_max"] = np.max([ranks[name] for name in names], axis=0)
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            left = names[left_index]
            right = names[right_index]
            signals[f"rank_mean_{left}_{right}"] = 0.5 * (
                ranks[left] + ranks[right]
            )
    return signals


def reconstruct_champion(
    arrays: dict[str, np.ndarray],
    oof: pd.DataFrame,
    membership_oof: pd.DataFrame,
    clean_oof: dict[int, np.ndarray],
    fold_groups: dict,
    postprocess: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
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
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_membership = membership_oof.loc[dev_mask].reset_index(drop=True)
    lock_membership = membership_oof.loc[lock_mask].reset_index(drop=True)
    dev_xgb = seed_subset.apply_xgb_layer(
        clean_oof[client.META_DEV_FOLD],
        np.load(ROOT / "honest_xgb_magic/dev_prediction.npy"),
        fold_groups[client.META_DEV_FOLD],
        postprocess,
        dev_membership,
        xgb_report["selected"],
    )
    lock_xgb = seed_subset.apply_xgb_layer(
        clean_oof[client.META_LOCK_FOLD],
        np.load(ROOT / "honest_xgb_magic/lock_prediction.npy"),
        fold_groups[client.META_LOCK_FOLD],
        postprocess,
        lock_membership,
        xgb_report["selected"],
    )
    cat_name = heavy_report["selected_cat"]["name"]
    lgb_name = heavy_report["selected_lgb"]["name"]
    dev = seed_subset.apply_heavy_layer(
        dev_xgb,
        np.load(ROOT / f"honest_magic_heavy_stack/dev_cat_{cat_name}.npy"),
        np.load(ROOT / f"honest_magic_heavy_stack/dev_lgb_{lgb_name}.npy"),
        fold_groups[client.META_DEV_FOLD],
        postprocess,
        dev_membership,
        heavy_report["selected_blend"],
    )
    lock = seed_subset.apply_heavy_layer(
        lock_xgb,
        np.load(ROOT / "honest_magic_heavy_stack/lock_cat.npy"),
        np.load(ROOT / "honest_magic_heavy_stack/lock_lgb.npy"),
        fold_groups[client.META_LOCK_FOLD],
        postprocess,
        lock_membership,
        heavy_report["selected_blend"],
    )
    dev_index = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_index = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
    metrics = {
        "dev": auc(arrays["target"][dev_index], dev),
        "lock": auc(arrays["target"][lock_index], lock),
    }
    if abs(metrics["dev"] - float(heavy_report["dev_auc"])) > 1e-12:
        raise RuntimeError("Could not reproduce heavy champion on dev")
    if abs(metrics["lock"] - float(heavy_report["lock_auc"])) > 1e-12:
        raise RuntimeError("Could not reproduce heavy champion on lock")
    return dev, lock, metrics


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    manifest, arrays = magic.load_matrix()
    feature_names = list(manifest["features"])
    mean_indices, detail_indices = select_feature_indices(feature_names)
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
    champion_dev, champion_lock, champion_metrics = reconstruct_champion(
        arrays,
        oof,
        membership_oof,
        clean_oof,
        fold_groups,
        clean_report["postprocess"],
    )

    raw_train = pd.read_csv(
        ROOT / "train_transaction.csv", usecols=pooling.RAW_COLUMNS
    )
    raw_test = pd.read_csv(
        ROOT / "test_transaction.csv", usecols=pooling.RAW_COLUMNS
    )
    train_uid = pooling.build_uid_frame(raw_train)["floor_card_addr_email"]
    test_uid = pooling.build_uid_frame(raw_test)["floor_card_addr_email"]
    day = arrays["day"]
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_index = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_index = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
    dev_train_index = np.flatnonzero(day < DEV_TRAIN_END)
    lock_train_index = np.flatnonzero(day < LOCK_TRAIN_END)
    y_dev = np.asarray(arrays["target"][dev_index], dtype="int8")
    y_lock = np.asarray(arrays["target"][lock_index], dtype="int8")
    dev_labels = segments.segment_labels(
        membership_oof.loc[dev_mask].reset_index(drop=True)
    )
    lock_labels = segments.segment_labels(
        membership_oof.loc[lock_mask].reset_index(drop=True)
    )

    dev_history, profile_names, dev_history_report = build_profile_matrix(
        arrays["train"],
        dev_train_index,
        train_uid,
        day,
        feature_names,
        mean_indices,
        detail_indices,
    )
    dev_query, query_names, dev_query_report = build_profile_matrix(
        arrays["train"],
        dev_index,
        train_uid,
        day,
        feature_names,
        mean_indices,
        detail_indices,
    )
    if query_names != profile_names:
        raise RuntimeError("Dev profile feature names differ")
    dev_predictions = {}
    dev_models = []
    for config in LGB_CONFIGS:
        name = config["name"]
        prediction, iterations, minutes = fit_lgb(
            dev_history,
            np.asarray(arrays["target"][dev_train_index], dtype="int8"),
            dev_query,
            y_dev,
            config,
            MODEL_DIR / f"dev_{name}.txt",
            None,
        )
        dev_predictions[name] = prediction
        np.save(WORK_DIR / f"dev_{name}.npy", prediction.astype("float32"))
        row = {
            "name": name,
            "config": config,
            "source_auc": auc(y_dev, prediction),
            "best_iteration": iterations,
            "minutes": minutes,
        }
        dev_models.append(row)
        print(json.dumps(row), flush=True)
    dev_signals = model_signals(dev_predictions)
    selected, search_rows = pooling.search_pooling(
        y_dev,
        champion_dev,
        dev_signals,
        dev_labels,
        np.asarray(day[dev_index]),
    )
    dev_candidate = pooling.direct_segment_blend(
        champion_dev,
        dev_signals[selected["variant"]],
        dev_labels,
        selected["weights"],
    )
    del dev_history, dev_query, dev_predictions, dev_signals
    gc.collect()

    lock_history, lock_names, lock_history_report = build_profile_matrix(
        arrays["train"],
        lock_train_index,
        train_uid,
        day,
        feature_names,
        mean_indices,
        detail_indices,
    )
    lock_query, lock_query_names, lock_query_report = build_profile_matrix(
        arrays["train"],
        lock_index,
        train_uid,
        day,
        feature_names,
        mean_indices,
        detail_indices,
    )
    if lock_names != profile_names or lock_query_names != profile_names:
        raise RuntimeError("Lock profile feature names differ")
    lock_predictions = {}
    lock_models = []
    for config, dev_row in zip(LGB_CONFIGS, dev_models):
        iterations = max(
            50, int(np.ceil(dev_row["best_iteration"] * ITERATION_SCALE))
        )
        prediction, _, minutes = fit_lgb(
            lock_history,
            np.asarray(arrays["target"][lock_train_index], dtype="int8"),
            lock_query,
            None,
            config,
            MODEL_DIR / f"lock_{config['name']}.txt",
            iterations,
        )
        lock_predictions[config["name"]] = prediction
        np.save(
            WORK_DIR / f"lock_{config['name']}.npy",
            prediction.astype("float32"),
        )
        lock_models.append(
            {"name": config["name"], "iterations": iterations, "minutes": minutes}
        )
    lock_signals = model_signals(lock_predictions)
    lock_candidate = pooling.direct_segment_blend(
        champion_lock,
        lock_signals[selected["variant"]],
        lock_labels,
        selected["weights"],
    )
    dev_auc = auc(y_dev, dev_candidate)
    lock_auc = auc(y_lock, lock_candidate)
    dev_gain = dev_auc - champion_metrics["dev"]
    lock_gain = lock_auc - champion_metrics["lock"]
    accepted = bool(
        dev_gain >= MIN_REQUIRED_GAIN and lock_gain >= MIN_REQUIRED_GAIN
    )
    del lock_history, lock_query, lock_predictions, lock_signals
    gc.collect()

    final_models = []
    if accepted:
        final_history, final_names, final_history_report = build_profile_matrix(
            arrays["train"],
            None,
            train_uid,
            day,
            feature_names,
            mean_indices,
            detail_indices,
        )
        test_query, test_names, test_query_report = build_profile_matrix(
            arrays["test"],
            None,
            test_uid,
            np.asarray(
                raw_test["TransactionDT"].to_numpy(dtype="float64") / 86_400.0,
                dtype="float32",
            ),
            feature_names,
            mean_indices,
            detail_indices,
        )
        if final_names != profile_names or test_names != profile_names:
            raise RuntimeError("Final profile feature names differ")
        test_predictions = {}
        for config, lock_row in zip(LGB_CONFIGS, lock_models):
            iterations = max(
                50, int(np.ceil(lock_row["iterations"] * ITERATION_SCALE))
            )
            prediction, _, minutes = fit_lgb(
                final_history,
                np.asarray(arrays["target"], dtype="int8"),
                test_query,
                None,
                config,
                MODEL_DIR / f"final_{config['name']}.txt",
                iterations,
            )
            test_predictions[config["name"]] = prediction
            final_models.append(
                {"name": config["name"], "iterations": iterations, "minutes": minutes}
            )
        test_signals = model_signals(test_predictions)
        baseline = pd.read_csv(ROOT / "submission_honest_magic_heavy_stack.csv")
        if not np.array_equal(
            baseline["TransactionID"].to_numpy(), arrays["test_id"]
        ):
            raise RuntimeError("Client-profile test rows differ from champion")
        prediction = pooling.direct_segment_blend(
            baseline[client.TARGET].to_numpy(dtype="float64"),
            test_signals[selected["variant"]],
            segments.segment_labels(membership_test),
            selected["weights"],
        )
        output = baseline[["TransactionID"]].copy()
        output[client.TARGET] = prediction
        output.to_csv(OUTPUT_PATH, index=False)
    else:
        final_history_report = None
        test_query_report = None

    report = {
        "data_policy": (
            "official train/test covariates; official train labels only; "
            "history/query profiles built separately"
        ),
        "official_hashes": manifest["official_hashes"],
        "uid": "card1|addr1|floor(day-D1)|P_emaildomain",
        "profile": {
            "mean_features": [feature_names[index] for index in mean_indices],
            "detail_features": [feature_names[index] for index in detail_indices],
            "feature_count": len(profile_names),
        },
        "profile_reports": {
            "dev_history": dev_history_report,
            "dev_query": dev_query_report,
            "lock_history": lock_history_report,
            "lock_query": lock_query_report,
            "final_history": final_history_report,
            "test_query": test_query_report,
        },
        "dev_models": dev_models,
        "selected": selected,
        "search_top": search_rows[:40],
        "champion_auc": champion_metrics,
        "dev_auc": dev_auc,
        "dev_gain": dev_gain,
        "lock_models": lock_models,
        "lock_auc": lock_auc,
        "lock_gain": lock_gain,
        "minimum_required_gain": MIN_REQUIRED_GAIN,
        "accepted": accepted,
        "final_models": final_models,
        "output": OUTPUT_PATH.name if accepted else None,
        "output_sha256": file_sha256(OUTPUT_PATH) if accepted else None,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

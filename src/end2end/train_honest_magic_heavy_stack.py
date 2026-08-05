"""Heavy CatBoost/LightGBM stack on the honest XGB-magic feature view.

Model and blend selection use two purged forward-time windows from official
train. Official test covariates are used only by target-free feature builders.
"""

from __future__ import annotations

import gc
import hashlib
from itertools import product
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import finalize_honest_xgb_magic_blend as xgb_final
import refine_honest_client_segments as segments
import search_honest_featureview_meta as featureview
import search_honest_fullrow_lgb as fullrow
import train_honest_client_meta as client
import train_honest_xgb_magic as magic


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_magic_heavy_stack"
MODEL_DIR = WORK_DIR / "models"
REPORT_PATH = WORK_DIR / "report.json"
OUTPUT_PATH = ROOT / "submission_honest_magic_heavy_stack.csv"

DEV_TRAIN_END = 45.0
LOCK_TRAIN_END = 60.0
ITERATION_SCALE = 1.15
SEED = 4513

CAT_BASE = {
    "iterations": 2_400,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.82,
    "border_count": 128,
    "one_hot_max_size": 16,
    "thread_count": -1,
    "allow_writing_files": False,
    "random_seed": SEED,
    "verbose": 200,
}
CAT_CONFIGS = (
    {
        "name": "ctr_d8",
        "depth": 8,
        "learning_rate": 0.045,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.35,
        "rsm": 0.85,
        "max_ctr_complexity": 2,
    },
    {
        "name": "ctr_d9",
        "depth": 9,
        "learning_rate": 0.035,
        "l2_leaf_reg": 12.0,
        "random_strength": 0.50,
        "rsm": 0.75,
        "max_ctr_complexity": 1,
    },
)

LGB_BASE = {
    "n_estimators": 3_500,
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.015,
    "subsample": 0.84,
    "subsample_freq": 1,
    "colsample_bytree": 0.78,
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
        "name": "leaf63",
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 120,
    },
    {
        "name": "leaf127",
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 240,
    },
    {
        "name": "extra127",
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 300,
        "extra_trees": True,
    },
)

CATEGORICAL_NAMES = {
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "card1_addr1",
    "card1_addr1_P_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(roc_auc_score(target, prediction))


def make_cat_frame(
    matrix: np.ndarray,
    row_index: np.ndarray | None,
    feature_names: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    values = matrix if row_index is None else matrix[row_index]
    frame = pd.DataFrame(values, columns=feature_names, copy=False)
    for column in categorical:
        frame[column] = np.rint(frame[column]).astype("int32")
    return frame


def fit_cat(
    matrix: np.ndarray,
    target: np.ndarray,
    train_index: np.ndarray | None,
    predict_matrix: np.ndarray,
    predict_index: np.ndarray | None,
    feature_names: list[str],
    categorical: list[str],
    config: dict,
    model_path: Path,
    fixed_iterations: int | None,
) -> tuple[np.ndarray, int, float]:
    params = {
        **CAT_BASE,
        **{key: value for key, value in config.items() if key != "name"},
    }
    if fixed_iterations is not None:
        params["iterations"] = fixed_iterations
    fit_x = make_cat_frame(matrix, train_index, feature_names, categorical)
    fit_y = target if train_index is None else target[train_index]
    predict_x = make_cat_frame(
        predict_matrix, predict_index, feature_names, categorical
    )
    model = CatBoostClassifier(**params)
    fit_kwargs = {
        "X": fit_x,
        "y": fit_y,
        "cat_features": categorical,
    }
    if fixed_iterations is None:
        fit_kwargs.update(
            {
                "eval_set": (predict_x, target[predict_index]),
                "early_stopping_rounds": 180,
                "use_best_model": True,
            }
        )
    started = time.time()
    model.fit(**fit_kwargs)
    prediction = model.predict_proba(predict_x)[:, 1]
    iterations = int(model.tree_count_)
    model.save_model(model_path)
    minutes = (time.time() - started) / 60.0
    del model, fit_x, predict_x
    gc.collect()
    return prediction, iterations, minutes


def fit_lgb(
    matrix: np.ndarray,
    target: np.ndarray,
    train_index: np.ndarray | None,
    predict_matrix: np.ndarray,
    predict_index: np.ndarray | None,
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
    fit_x = matrix if train_index is None else matrix[train_index]
    fit_y = target if train_index is None else target[train_index]
    predict_x = (
        predict_matrix
        if predict_index is None
        else predict_matrix[predict_index]
    )
    model = lgb.LGBMClassifier(**params)
    fit_kwargs = {"X": fit_x, "y": fit_y}
    if fixed_iterations is None:
        fit_kwargs.update(
            {
                "eval_set": [(predict_x, target[predict_index])],
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
    prediction = model.predict_proba(predict_x, num_iteration=best)[:, 1]
    model.booster_.save_model(model_path, num_iteration=best)
    minutes = (time.time() - started) / 60.0
    del model, fit_x, predict_x
    gc.collect()
    return prediction, best, minutes


def reconstruct_champion(
    arrays: dict[str, np.ndarray],
    oof: pd.DataFrame,
    membership: pd.DataFrame,
    clean_oof: dict[int, np.ndarray],
    fold_groups: dict,
    postprocess: dict,
) -> tuple[dict[int, np.ndarray], dict]:
    report = json.loads(
        (ROOT / "honest_xgb_magic/blend_report.json").read_text(
            encoding="utf-8"
        )
    )
    recipe = report["selected"]
    predictions = {}
    for fold, filename in (
        (client.META_DEV_FOLD, "dev_prediction.npy"),
        (client.META_LOCK_FOLD, "lock_prediction.npy"),
    ):
        mask = oof["fold"].eq(fold).to_numpy()
        source = np.load(ROOT / "honest_xgb_magic" / filename)
        transformed = fullrow.transform_variant(
            source, recipe["variant"], fold_groups[fold], postprocess
        )
        predictions[fold] = segments.segmented_blend(
            clean_oof[fold],
            transformed,
            segments.segment_labels(
                membership.loc[mask].reset_index(drop=True)
            ),
            recipe["weights"],
        )
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    metrics = {
        "dev": auc(
            np.asarray(arrays["target"])[
                oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
            ],
            predictions[client.META_DEV_FOLD],
        ),
        "lock": auc(
            np.asarray(arrays["target"])[
                oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
            ],
            predictions[client.META_LOCK_FOLD],
        ),
    }
    if abs(metrics["dev"] - float(recipe["auc"])) > 1e-12:
        raise RuntimeError("Could not reproduce champion dev prediction")
    if abs(metrics["lock"] - float(report["candidate_lock_auc"])) > 1e-12:
        raise RuntimeError("Could not reproduce champion lock prediction")
    return predictions, {"recipe": recipe, "metrics": metrics}


def screen_source(
    target: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    groups: dict,
    postprocess: dict,
    segment_labels: np.ndarray,
    day: np.ndarray,
) -> tuple[dict, list[dict]]:
    variants = fullrow.prediction_variants(prediction, groups, postprocess)
    return fast_search_blend(
        target,
        baseline,
        variants,
        segment_labels,
        day,
        starts=((0.0, 0.0, 0.0), (0.2, 0.2, 0.4)),
    )


def fast_search_blend(
    target: np.ndarray,
    baseline: np.ndarray,
    raw_variants: dict[str, np.ndarray],
    segment_labels: np.ndarray,
    day: np.ndarray,
    starts: tuple[tuple[float, float, float], ...],
) -> tuple[dict, list[dict]]:
    """Coordinate search with cached rank transforms for segment weights."""
    weight_grid = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))
    midpoint = np.median(day)
    halves = (day <= midpoint, day > midpoint)
    baseline_rank = client.rank(baseline)
    baseline_auc = auc(target, baseline_rank)
    baseline_halves = [
        auc(target[mask], baseline_rank[mask]) for mask in halves
    ]
    masks = {
        name: segment_labels == name for name in segments.SEGMENTS
    }
    baseline_segment_ranks = {
        name: client.rank(baseline_rank[mask]) for name, mask in masks.items()
    }
    rows = []
    seen = set()

    for variant, raw_prediction in raw_variants.items():
        stacked_rank = client.rank(raw_prediction)
        transformed = {}
        for name, mask in masks.items():
            current = baseline_segment_ranks[name]
            stacked = client.rank(stacked_rank[mask])
            transformed[name] = {}
            for weight in weight_grid:
                signal = (1.0 - weight) * current + weight * stacked
                transformed[name][weight] = segments.rank_match(
                    signal, baseline_rank[mask]
                )

        def evaluate(weights: dict[str, float]) -> dict:
            key = (
                variant,
                *(float(weights[name]) for name in segments.SEGMENTS),
            )
            if key in seen:
                return next(row for row in rows if row["_key"] == key)
            prediction = baseline_rank.copy()
            for name, mask in masks.items():
                prediction[mask] = transformed[name][float(weights[name])]
            score = auc(target, prediction)
            half_auc = [
                auc(target[mask], prediction[mask]) for mask in halves
            ]
            half_gains = [
                value - base
                for value, base in zip(half_auc, baseline_halves)
            ]
            row = {
                "variant": variant,
                "weights": {
                    name: float(weights[name]) for name in segments.SEGMENTS
                },
                "auc": score,
                "gain": score - baseline_auc,
                "half_gains": half_gains,
                "min_half_gain": float(min(half_gains)),
                "_key": key,
            }
            rows.append(row)
            seen.add(key)
            return row

        for start in starts:
            weights = dict(zip(segments.SEGMENTS, start))
            evaluate(weights)
            for _ in range(2):
                for name in segments.SEGMENTS:
                    local = []
                    for weight in weight_grid:
                        candidate = dict(weights)
                        candidate[name] = float(weight)
                        local.append(evaluate(candidate))
                    best = max(
                        local,
                        key=lambda row: (row["auc"], row["min_half_gain"]),
                    )
                    weights = dict(best["weights"])
            neighborhoods = []
            for name in segments.SEGMENTS:
                center = weight_grid.index(float(weights[name]))
                neighborhoods.append(
                    weight_grid[max(0, center - 1) : center + 2]
                )
            for values in product(*neighborhoods):
                evaluate(dict(zip(segments.SEGMENTS, values)))

    stable = [row for row in rows if row["min_half_gain"] >= 0.0]
    if not stable:
        raise RuntimeError("No temporally stable blend candidate")
    selected = max(
        stable, key=lambda row: (row["gain"], row["min_half_gain"])
    )
    rows.sort(
        key=lambda row: (row["gain"], row["min_half_gain"]), reverse=True
    )
    for row in rows:
        row.pop("_key", None)
    return selected, rows


def source_signals(
    cat_prediction: np.ndarray,
    lgb_prediction: np.ndarray,
    groups: dict,
    postprocess: dict,
) -> dict[str, np.ndarray]:
    cat_rank = client.rank(cat_prediction)
    lgb_rank = client.rank(lgb_prediction)
    cat_post = client.apply_postprocess(cat_rank, groups, postprocess)
    lgb_post = client.apply_postprocess(lgb_rank, groups, postprocess)
    signals = {}
    for mode, left, right in (
        ("rank", cat_rank, lgb_rank),
        ("postprocessed_rank", cat_post, lgb_post),
    ):
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            signals[f"{mode}_cat{alpha:.2f}"] = (
                alpha * left + (1.0 - alpha) * right
            )
        signals[f"{mode}_max"] = np.maximum(left, right)
        signals[f"{mode}_min"] = np.minimum(left, right)
        signals[f"{mode}_geo"] = np.sqrt(
            np.clip(left, 0.0, None) * np.clip(right, 0.0, None)
        )
    return signals


def search_signals(
    target: np.ndarray,
    baseline: np.ndarray,
    signals: dict[str, np.ndarray],
    segment_labels: np.ndarray,
    day: np.ndarray,
) -> tuple[dict, list[dict]]:
    return fast_search_blend(
        target,
        baseline,
        signals,
        segment_labels,
        day,
        starts=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.4),
            (0.2, 0.2, 0.4),
            (0.5, 0.5, 0.5),
        ),
    )


def segment_metrics(
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name in segments.SEGMENTS:
        mask = labels == name
        result[name] = {
            "rows": int(mask.sum()),
            "frauds": int(target[mask].sum()),
            "baseline_auc": auc(target[mask], baseline[mask]),
            "candidate_auc": auc(target[mask], candidate[mask]),
        }
        result[name]["gain"] = (
            result[name]["candidate_auc"] - result[name]["baseline_auc"]
        )
    return result


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    manifest, arrays = magic.load_matrix()
    train = arrays["train"]
    target = arrays["target"]
    day = arrays["day"]
    feature_names = list(manifest["features"])
    categorical = sorted(CATEGORICAL_NAMES.intersection(feature_names))

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
    champion, champion_report = reconstruct_champion(
        arrays,
        oof,
        membership_oof,
        clean_oof,
        fold_groups,
        clean_report["postprocess"],
    )

    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_index = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_index = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
    dev_train_index = np.flatnonzero(day < DEV_TRAIN_END)
    lock_train_index = np.flatnonzero(day < LOCK_TRAIN_END)
    y_dev = np.asarray(target[dev_index], dtype="int8")
    y_lock = np.asarray(target[lock_index], dtype="int8")
    dev_segments = segments.segment_labels(
        membership_oof.loc[dev_mask].reset_index(drop=True)
    )
    lock_segments = segments.segment_labels(
        membership_oof.loc[lock_mask].reset_index(drop=True)
    )

    dev_rows = []
    dev_predictions: dict[str, np.ndarray] = {}
    for config in CAT_CONFIGS:
        name = config["name"]
        prediction_path = WORK_DIR / f"dev_cat_{name}.npy"
        model_path = MODEL_DIR / f"dev_cat_{name}.cbm"
        cached = prediction_path.exists() and model_path.exists()
        if cached:
            prediction = np.load(prediction_path).astype("float64")
            cached_model = CatBoostClassifier()
            cached_model.load_model(model_path)
            iterations = int(cached_model.tree_count_)
            minutes = 0.0
            del cached_model
        else:
            prediction, iterations, minutes = fit_cat(
                train,
                target,
                dev_train_index,
                train,
                dev_index,
                feature_names,
                categorical,
                config,
                model_path,
                None,
            )
            np.save(prediction_path, prediction.astype("float32"))
        selected, search = screen_source(
            y_dev,
            champion[client.META_DEV_FOLD],
            prediction,
            fold_groups[client.META_DEV_FOLD],
            clean_report["postprocess"],
            dev_segments,
            np.asarray(day[dev_index]),
        )
        row = {
            "family": "catboost",
            "name": name,
            "config": config,
            "source_auc": auc(y_dev, prediction),
            "best_iteration": iterations,
            "minutes": minutes,
            "cached": cached,
            "best_blend": selected,
            "search_top": search[:5],
        }
        dev_rows.append(row)
        dev_predictions[f"catboost:{name}"] = prediction
        print(json.dumps(row), flush=True)

    for config in LGB_CONFIGS:
        name = config["name"]
        prediction_path = WORK_DIR / f"dev_lgb_{name}.npy"
        model_path = MODEL_DIR / f"dev_lgb_{name}.txt"
        cached = prediction_path.exists() and model_path.exists()
        if cached:
            prediction = np.load(prediction_path).astype("float64")
            cached_model = lgb.Booster(model_file=str(model_path))
            iterations = int(cached_model.num_trees())
            minutes = 0.0
            del cached_model
        else:
            prediction, iterations, minutes = fit_lgb(
                train,
                target,
                dev_train_index,
                train,
                dev_index,
                config,
                model_path,
                None,
            )
            np.save(prediction_path, prediction.astype("float32"))
        selected, search = screen_source(
            y_dev,
            champion[client.META_DEV_FOLD],
            prediction,
            fold_groups[client.META_DEV_FOLD],
            clean_report["postprocess"],
            dev_segments,
            np.asarray(day[dev_index]),
        )
        row = {
            "family": "lightgbm",
            "name": name,
            "config": config,
            "source_auc": auc(y_dev, prediction),
            "best_iteration": iterations,
            "minutes": minutes,
            "cached": cached,
            "best_blend": selected,
            "search_top": search[:5],
        }
        dev_rows.append(row)
        dev_predictions[f"lightgbm:{name}"] = prediction
        print(json.dumps(row), flush=True)

    best_cat = max(
        (row for row in dev_rows if row["family"] == "catboost"),
        key=lambda row: (
            row["best_blend"]["gain"],
            row["best_blend"]["min_half_gain"],
        ),
    )
    best_lgb = max(
        (row for row in dev_rows if row["family"] == "lightgbm"),
        key=lambda row: (
            row["best_blend"]["gain"],
            row["best_blend"]["min_half_gain"],
        ),
    )
    dev_cat = dev_predictions[f"catboost:{best_cat['name']}"]
    dev_lgb = dev_predictions[f"lightgbm:{best_lgb['name']}"]
    dev_signals = source_signals(
        dev_cat,
        dev_lgb,
        fold_groups[client.META_DEV_FOLD],
        clean_report["postprocess"],
    )
    selected, search_rows = search_signals(
        y_dev,
        champion[client.META_DEV_FOLD],
        dev_signals,
        dev_segments,
        np.asarray(day[dev_index]),
    )
    dev_candidate = segments.segmented_blend(
        champion[client.META_DEV_FOLD],
        dev_signals[selected["variant"]],
        dev_segments,
        selected["weights"],
    )

    cat_config = next(
        config for config in CAT_CONFIGS if config["name"] == best_cat["name"]
    )
    lgb_config = next(
        config for config in LGB_CONFIGS if config["name"] == best_lgb["name"]
    )
    cat_lock_iterations = max(
        50, int(np.ceil(best_cat["best_iteration"] * ITERATION_SCALE))
    )
    lgb_lock_iterations = max(
        50, int(np.ceil(best_lgb["best_iteration"] * ITERATION_SCALE))
    )
    lock_cat, _, cat_lock_minutes = fit_cat(
        train,
        target,
        lock_train_index,
        train,
        lock_index,
        feature_names,
        categorical,
        cat_config,
        MODEL_DIR / "lock_cat.cbm",
        cat_lock_iterations,
    )
    lock_lgb, _, lgb_lock_minutes = fit_lgb(
        train,
        target,
        lock_train_index,
        train,
        lock_index,
        lgb_config,
        MODEL_DIR / "lock_lgb.txt",
        lgb_lock_iterations,
    )
    np.save(WORK_DIR / "lock_cat.npy", lock_cat.astype("float32"))
    np.save(WORK_DIR / "lock_lgb.npy", lock_lgb.astype("float32"))
    lock_signals = source_signals(
        lock_cat,
        lock_lgb,
        fold_groups[client.META_LOCK_FOLD],
        clean_report["postprocess"],
    )
    lock_candidate = segments.segmented_blend(
        champion[client.META_LOCK_FOLD],
        lock_signals[selected["variant"]],
        lock_segments,
        selected["weights"],
    )
    dev_auc = auc(y_dev, dev_candidate)
    lock_auc = auc(y_lock, lock_candidate)
    accepted = bool(
        dev_auc > champion_report["metrics"]["dev"]
        and lock_auc > champion_report["metrics"]["lock"]
        and selected["min_half_gain"] >= 0.0
    )

    final_rows = []
    if accepted:
        cat_final_iterations = max(
            50, int(np.ceil(cat_lock_iterations * ITERATION_SCALE))
        )
        lgb_final_iterations = max(
            50, int(np.ceil(lgb_lock_iterations * ITERATION_SCALE))
        )
        test_cat, _, cat_minutes = fit_cat(
            train,
            target,
            None,
            arrays["test"],
            None,
            feature_names,
            categorical,
            cat_config,
            MODEL_DIR / "final_cat.cbm",
            cat_final_iterations,
        )
        final_rows.append(
            {
                "family": "catboost",
                "iterations": cat_final_iterations,
                "minutes": cat_minutes,
            }
        )
        test_lgb, _, lgb_minutes = fit_lgb(
            train,
            target,
            None,
            arrays["test"],
            None,
            lgb_config,
            MODEL_DIR / "final_lgb.txt",
            lgb_final_iterations,
        )
        final_rows.append(
            {
                "family": "lightgbm",
                "iterations": lgb_final_iterations,
                "minutes": lgb_minutes,
            }
        )
        np.save(WORK_DIR / "test_cat.npy", test_cat.astype("float32"))
        np.save(WORK_DIR / "test_lgb.npy", test_lgb.astype("float32"))
        test_signals = source_signals(
            test_cat,
            test_lgb,
            reference_report["test_groups"],
            clean_report["postprocess"],
        )
        baseline = pd.read_csv(ROOT / "submission_honest_xgb_magic_blend.csv")
        if not np.array_equal(
            baseline["TransactionID"].to_numpy(), arrays["test_id"]
        ):
            raise RuntimeError("Heavy-stack test rows differ from champion")
        prediction = segments.segmented_blend(
            baseline[client.TARGET].to_numpy(dtype="float64"),
            test_signals[selected["variant"]],
            segments.segment_labels(membership_test),
            selected["weights"],
        )
        output = baseline[["TransactionID"]].copy()
        output[client.TARGET] = prediction
        output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": (
            "official train/test covariates; official train labels only; "
            "no gap or audit labels"
        ),
        "official_hashes": manifest["official_hashes"],
        "feature_recipe": manifest["source"],
        "feature_count": len(feature_names),
        "categorical_count": len(categorical),
        "categorical": categorical,
        "fold_contract": {
            "dev_train": f"day < {DEV_TRAIN_END:g}",
            "dev_valid": "day 75-90",
            "lock_train": f"day < {LOCK_TRAIN_END:g}",
            "lock_valid": "day 90-105",
        },
        "champion": champion_report,
        "development_screen": dev_rows,
        "selected_cat": best_cat,
        "selected_lgb": best_lgb,
        "selected_blend": selected,
        "search_top": search_rows[:40],
        "dev_auc": dev_auc,
        "dev_gain": dev_auc - champion_report["metrics"]["dev"],
        "lock_auc": lock_auc,
        "lock_gain": lock_auc - champion_report["metrics"]["lock"],
        "lock_source_auc": {
            "catboost": auc(y_lock, lock_cat),
            "lightgbm": auc(y_lock, lock_lgb),
        },
        "segment_metrics": {
            "dev": segment_metrics(
                y_dev,
                champion[client.META_DEV_FOLD],
                dev_candidate,
                dev_segments,
            ),
            "lock": segment_metrics(
                y_lock,
                champion[client.META_LOCK_FOLD],
                lock_candidate,
                lock_segments,
            ),
        },
        "lock_iterations": {
            "catboost": cat_lock_iterations,
            "lightgbm": lgb_lock_iterations,
        },
        "lock_minutes": {
            "catboost": cat_lock_minutes,
            "lightgbm": lgb_lock_minutes,
        },
        "acceptance_rule": "strictly improve both dev and untouched lock",
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

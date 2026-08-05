"""Strict no-bridge temporal ensemble for the local fraud dataset.

Development deliberately rebuilds every feature matrix from only two pieces:
the labeled history available before a cutoff and the future validation block.
Rows inside the embargo are not passed to feature engineering. This mirrors the
competition setting where no middle bridge is available.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import gc
from itertools import product
import json
from pathlib import Path
import re
import time

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from fraud_features import TARGET, build_features, read_and_merge
from fraud_multicounter_features import add_multi_counter_features
from fraud_next_features import (
    add_behavior_distribution_features,
    add_calendar_amount_features,
    add_identity_features,
    add_velocity_features,
)
from fraud_overlap_recipe import assign_segments, uid_metadata
from fraud_user_features import CAUSAL_HISTORY_COLUMNS, add_user_profile_features
from fraud_vblock_features import TOP_V_COLUMNS, add_vblock_user_features


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "clean_v2"
MODEL_DIR = WORK_DIR / "models"
PREDICTION_DIR = WORK_DIR / "predictions"
RECIPE_PATH = WORK_DIR / "recipe.json"
REPORT_PATH = WORK_DIR / "backtest_report.json"
SOURCE_PATH = WORK_DIR / "test_source_predictions.csv"
SUBMISSION_PATH = ROOT / "submission_clean_v2.csv"

EXPECTED_TRAIN_ROWS = 365_365
EXPECTED_TEST_ROWS = 129_736
DAY_SECONDS = 86_400.0
SEEDS = (42, 2026, 3407)
SOURCE_NAMES = (
    "cat_identity",
    "cat_history",
    "cat_weighted",
    "lgb_giba",
    "lgb_cold",
    "xgb_giba",
)
SEGMENTS = ("strict", "partial", "cold")
HORIZONS = (30, 45, 60, 75)


@dataclass(frozen=True)
class PairSpec:
    name: str
    stage: str
    horizon: int
    train_end_day: int
    valid_start_day: int
    valid_end_day: int


DEV_SPECS = (
    PairSpec("dev_h30_a", "dev", 30, 15, 45, 60),
    PairSpec("dev_h30_b", "dev", 30, 30, 60, 75),
    PairSpec("dev_h30_c", "dev", 30, 45, 75, 90),
    PairSpec("dev_h45_a", "dev", 45, 15, 60, 75),
    PairSpec("dev_h45_b", "dev", 45, 30, 75, 90),
    PairSpec("dev_h60", "dev", 60, 15, 75, 90),
)
LOCK_SPECS = (
    PairSpec("lock_h30", "lock", 30, 60, 90, 106),
    PairSpec("lock_h45", "lock", 45, 45, 90, 106),
    PairSpec("lock_h60", "lock", 60, 30, 90, 106),
    PairSpec("lock_h75", "lock", 75, 15, 90, 106),
)


CAT_PARAMS = {
    "iterations": 1_500,
    "depth": 8,
    "learning_rate": 0.06,
    "l2_leaf_reg": 8,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.80,
    "rsm": 0.90,
    "border_count": 128,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "one_hot_max_size": 10,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": 200,
}

LGB_GIBA_PARAMS = {
    "n_estimators": 1_500,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "min_child_samples": 40,
    "subsample": 0.70,
    "subsample_freq": 1,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "max_bin": 255,
    "extra_trees": True,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}

LGB_COLD_PARAMS = {
    "n_estimators": 1_000,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 47,
    "learning_rate": 0.035,
    "min_child_samples": 90,
    "subsample": 0.78,
    "subsample_freq": 1,
    "colsample_bytree": 0.78,
    "reg_alpha": 0.75,
    "reg_lambda": 10.0,
    "max_bin": 255,
    "extra_trees": True,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}

XGB_PARAMS = {
    "n_estimators": 1_500,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 20,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "gamma": 0.05,
    "max_bin": 256,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
}


def unique(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(column for group in groups for column in group))


def rank_prediction(values: np.ndarray | pd.Series) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy()


def auc(y: np.ndarray | pd.Series, prediction: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(prediction)))


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )


def read_official_train() -> pd.DataFrame:
    frame = read_and_merge(ROOT, "train").reset_index(drop=True)
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_ROWS:,} train rows, got {len(frame):,}"
        )
    if TARGET not in frame:
        raise ValueError(f"Official train is missing {TARGET}")
    if not frame["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Training rows are not sorted by TransactionDT")
    return frame


def read_official_test() -> pd.DataFrame:
    frame = read_and_merge(ROOT, "test").reset_index(drop=True)
    if len(frame) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS:,} test rows, got {len(frame):,}"
        )
    if TARGET in frame:
        raise AssertionError("Target unexpectedly exists in competition test")
    return frame


def normalize_categories(
    history: pd.DataFrame,
    future: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        history[column] = (
            history[column].astype("string").fillna("<MISSING>").astype(str)
        )
        future[column] = (
            future[column].astype("string").fillna("<MISSING>").astype(str)
        )


def prepare_pair(
    history_raw: pd.DataFrame,
    future_raw: pd.DataFrame,
) -> dict:
    """Build target-free pair features with no rows inside the time gap."""
    y = history_raw[TARGET].astype("int8").reset_index(drop=True)
    history = history_raw.copy().reset_index(drop=True)
    future = future_raw.drop(columns=TARGET, errors="ignore").copy().reset_index(
        drop=True
    )

    history, future, base_features, base_categorical = build_features(
        history,
        future,
        giba_features=True,
        frequency_mode="selected",
        v307_chain_features=True,
    )
    metadata_history = uid_metadata(history).reset_index(drop=True)
    metadata_future = uid_metadata(future).reset_index(drop=True)

    (
        history,
        future,
        user_features,
        history_components,
        future_components,
        graph_stats,
    ) = add_user_profile_features(history, future, y)
    causal_history = [
        column for column in CAUSAL_HISTORY_COLUMNS if column in history
    ]
    target_free_profile = [
        column for column in user_features if column not in causal_history
    ]

    history, future, vblock_features, vblock_stats = add_vblock_user_features(
        history,
        future,
        history_components,
        future_components,
    )
    history, future, multi_features, multi_categorical, multi_stats = (
        add_multi_counter_features(history, future, max_chain_size=100)
    )
    history, future, velocity_features, velocity_stats = add_velocity_features(
        history, future
    )
    history, future, calendar_features, calendar_stats = (
        add_calendar_amount_features(history, future)
    )
    (
        history,
        future,
        identity_features,
        identity_categorical,
        identity_stats,
    ) = add_identity_features(history, future)
    history, future, behavior_features, behavior_stats = (
        add_behavior_distribution_features(
            history,
            future,
            history_components,
            future_components,
        )
    )

    categorical = unique(
        base_categorical,
        multi_categorical,
        identity_categorical,
    )
    normalize_categories(history, future, categorical)

    identity_view = unique(base_features, multi_features)
    history_view = unique(
        base_features,
        target_free_profile,
        causal_history,
        multi_features,
        velocity_features,
        identity_features,
    )

    generic_core = []
    for column in base_features:
        if column.startswith("uid_"):
            continue
        if re.fullmatch(r"V\d+", column) or re.fullmatch(r"id_\d+", column):
            continue
        generic_core.append(column)
    selected_raw_v = [column for column in TOP_V_COLUMNS if column in history]
    cold_view = unique(
        generic_core,
        selected_raw_v,
        target_free_profile,
        vblock_features,
        velocity_features,
        calendar_features,
        identity_features,
        behavior_features,
    )

    domain_view = [
        column
        for column in unique(
            generic_core,
            selected_raw_v,
            identity_features,
            [name for name in vblock_features if "wide_user_" not in name],
        )
        if column != "TransactionDT"
        and not column.startswith("DT_")
        and not column.startswith("calendar_")
        and not column.endswith("_minus_day")
        and column != "D1_origin_day"
    ]

    views = {
        "identity": identity_view,
        "history": history_view,
        "cold": cold_view,
        "domain": domain_view,
    }
    for name, features in views.items():
        missing = [column for column in features if column not in history]
        if missing:
            raise ValueError(f"{name} view has missing columns: {missing[:5]}")

    return {
        "history": history,
        "future": future,
        "y": y,
        "views": views,
        "categorical": categorical,
        "metadata_history": metadata_history,
        "metadata_future": metadata_future,
        "stats": {
            "rows": {"history": len(history), "future": len(future)},
            "features": {name: len(value) for name, value in views.items()},
            "categorical": len(categorical),
            "graph": graph_stats,
            "vblock": vblock_stats,
            "multi": multi_stats,
            "velocity": velocity_stats,
            "calendar": calendar_stats,
            "identity": identity_stats,
            "behavior": behavior_stats,
        },
    }


def encode_pair(
    history: pd.DataFrame,
    future: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [history[features], future[features]], ignore_index=True, copy=False
    ).copy()
    for column in categorical:
        if column in combined:
            codes, _ = pd.factorize(combined[column], sort=False)
            combined[column] = codes.astype("int32", copy=False)
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    history_rows = len(history)
    return combined.iloc[:history_rows], combined.iloc[history_rows:]


def domain_weights(prepared: dict, seed: int) -> tuple[np.ndarray, dict]:
    history = prepared["history"]
    future = prepared["future"]
    features = prepared["views"]["domain"]
    categorical = [
        column for column in prepared["categorical"] if column in features
    ]
    X_history, X_future = encode_pair(
        history, future, features, categorical
    )
    rng = np.random.default_rng(seed)
    max_history = min(len(X_history), max(len(X_future) * 2, 20_000))
    history_index = np.sort(
        rng.choice(len(X_history), size=max_history, replace=False)
    )
    X_domain = pd.concat(
        [X_history.iloc[history_index], X_future], ignore_index=True
    )
    y_domain = np.concatenate(
        [np.zeros(len(history_index), dtype="int8"), np.ones(len(X_future), dtype="int8")]
    )
    model = lgb.LGBMClassifier(
        n_estimators=250,
        objective="binary",
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_child_samples=120,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(X_domain, y_domain, callbacks=[lgb.log_evaluation(0)])
    domain_train_prediction = model.predict_proba(X_domain)[:, 1]
    history_probability = np.clip(model.predict_proba(X_history)[:, 1], 0.02, 0.98)
    prior_ratio = len(history_index) / max(len(X_future), 1)
    weights = prior_ratio * history_probability / (1.0 - history_probability)
    weights /= max(float(np.mean(weights)), 1e-9)
    weights = np.clip(weights, 0.25, 4.0)
    weights /= float(np.mean(weights))
    report = {
        "features": len(features),
        "sample_rows": len(X_domain),
        "train_auc": auc(y_domain, domain_train_prediction),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "weight_q05": float(np.quantile(weights, 0.05)),
        "weight_q95": float(np.quantile(weights, 0.95)),
    }
    del X_history, X_future, X_domain, model
    gc.collect()
    return weights.astype("float32"), report


def fit_cat_source(
    prepared: dict,
    features: list[str],
    seeds: tuple[int, ...],
    model_stem: str,
    fixed_iterations: int | None,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    categorical = [
        column for column in prepared["categorical"] if column in features
    ]
    predictions = []
    iterations = []
    for seed in seeds:
        model_path = MODEL_DIR / f"{model_stem}_s{seed}.cbm"
        params = {
            **CAT_PARAMS,
            "random_seed": seed,
            "iterations": fixed_iterations or CAT_PARAMS["iterations"],
        }
        model = CatBoostClassifier(**params)
        fit_kwargs = {
            "X": prepared["history"][features],
            "y": prepared["y"],
            "cat_features": categorical,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if fixed_iterations is None:
            fit_kwargs.update(
                {
                    "eval_set": (
                        prepared["future"][features],
                        prepared["future_y"],
                    ),
                    "early_stopping_rounds": 120,
                    "use_best_model": True,
                }
            )
        model.fit(**fit_kwargs)
        prediction = model.predict_proba(prepared["future"][features])[:, 1]
        predictions.append(prediction)
        best = model.tree_count_
        iterations.append(int(best))
        model.save_model(model_path)
        del model
        gc.collect()
    return np.mean(predictions, axis=0), iterations


def fit_lgb_source(
    prepared: dict,
    features: list[str],
    source: str,
    seed: int,
    model_stem: str,
    fixed_iterations: int | None,
) -> tuple[np.ndarray, int]:
    categorical = [
        column for column in prepared["categorical"] if column in features
    ]
    X_history, X_future = encode_pair(
        prepared["history"], prepared["future"], features, categorical
    )
    base = LGB_GIBA_PARAMS if source == "lgb_giba" else LGB_COLD_PARAMS
    params = {
        **base,
        "random_state": seed,
        "n_estimators": fixed_iterations or base["n_estimators"],
    }
    model = lgb.LGBMClassifier(**params)
    fit_kwargs = {
        "X": X_history,
        "y": prepared["y"],
        "categorical_feature": categorical,
    }
    if fixed_iterations is None:
        fit_kwargs.update(
            {
                "eval_set": [(X_future, prepared["future_y"])],
                "eval_metric": "auc",
                "callbacks": [
                    lgb.early_stopping(120, verbose=False),
                    lgb.log_evaluation(200),
                ],
            }
        )
    else:
        fit_kwargs["callbacks"] = [lgb.log_evaluation(0)]
    model.fit(**fit_kwargs)
    best = int(model.best_iteration_ or params["n_estimators"])
    prediction = model.predict_proba(X_future, num_iteration=best)[:, 1]
    model.booster_.save_model(MODEL_DIR / f"{model_stem}.txt", num_iteration=best)
    del model, X_history, X_future
    gc.collect()
    return prediction, best


def fit_xgb_source(
    prepared: dict,
    features: list[str],
    seed: int,
    model_stem: str,
    fixed_iterations: int | None,
) -> tuple[np.ndarray, int]:
    categorical = [
        column for column in prepared["categorical"] if column in features
    ]
    X_history, X_future = encode_pair(
        prepared["history"], prepared["future"], features, categorical
    )
    params = {
        **XGB_PARAMS,
        "random_state": seed,
        "n_estimators": fixed_iterations or XGB_PARAMS["n_estimators"],
    }
    if fixed_iterations is None:
        params["early_stopping_rounds"] = 120
    model = xgb.XGBClassifier(**params)
    fit_kwargs = {"X": X_history, "y": prepared["y"], "verbose": False}
    if fixed_iterations is None:
        fit_kwargs["eval_set"] = [(X_future, prepared["future_y"])]
    model.fit(**fit_kwargs)
    best = int(
        getattr(model, "best_iteration", params["n_estimators"] - 1) + 1
    )
    prediction = model.predict_proba(X_future)[:, 1]
    model.save_model(MODEL_DIR / f"{model_stem}.json")
    del model, X_history, X_future
    gc.collect()
    return prediction, best


def prediction_cache(spec: PairSpec) -> Path:
    return PREDICTION_DIR / f"{spec.name}.csv"


def cached_fit_report(spec: PairSpec, raw: pd.DataFrame) -> dict:
    """Recover iteration counts from model files after an interrupted run."""
    cat_iterations: dict[str, list[int]] = {}
    cat_layout = {
        "cat_identity": (42, 2026),
        "cat_history": (42,),
        "cat_weighted": (3407,),
    }
    for source, seeds in cat_layout.items():
        values = []
        for seed in seeds:
            path = MODEL_DIR / f"{spec.name}_{source}_s{seed}.cbm"
            if not path.exists():
                continue
            model = CatBoostClassifier()
            model.load_model(path)
            values.append(int(model.tree_count_))
        cat_iterations[source] = values

    lgb_iterations = {}
    for source in ("lgb_giba", "lgb_cold"):
        path = MODEL_DIR / f"{spec.name}_{source}.txt"
        lgb_iterations[source] = (
            [int(lgb.Booster(model_file=str(path)).num_trees())]
            if path.exists()
            else []
        )
    xgb_path = MODEL_DIR / f"{spec.name}_xgb_giba.json"
    xgb_iterations = []
    if xgb_path.exists():
        model = xgb.XGBClassifier()
        model.load_model(xgb_path)
        xgb_iterations = [int(model.get_booster().num_boosted_rounds())]

    day = raw["TransactionDT"].to_numpy(dtype="float64") / DAY_SECONDS
    return {
        "cached": True,
        "spec": asdict(spec),
        "history_rows": int(np.sum(day < spec.train_end_day)),
        "future_rows": int(
            np.sum((day >= spec.valid_start_day) & (day < spec.valid_end_day))
        ),
        "best_iterations": {
            **cat_iterations,
            **lgb_iterations,
            "xgb_giba": xgb_iterations,
        },
    }


def train_pair(
    raw: pd.DataFrame,
    spec: PairSpec,
    fixed_iterations: dict[str, int] | None,
    force: bool,
) -> tuple[pd.DataFrame, dict]:
    cache = prediction_cache(spec)
    if cache.exists() and not force:
        frame = pd.read_csv(cache)
        required = {"row_index", "TransactionID", TARGET, "segment", *SOURCE_NAMES}
        if required.issubset(frame.columns):
            print(f"Loading cached {spec.name}", flush=True)
            return frame, cached_fit_report(spec, raw)

    day = raw["TransactionDT"].to_numpy(dtype="float64") / DAY_SECONDS
    history_index = np.flatnonzero(day < spec.train_end_day)
    future_index = np.flatnonzero(
        (day >= spec.valid_start_day) & (day < spec.valid_end_day)
    )
    if not len(history_index) or not len(future_index):
        raise ValueError(f"Empty history/future for {spec.name}")
    skipped = int(
        np.sum((day >= spec.train_end_day) & (day < spec.valid_start_day))
    )
    print(
        f"\n{spec.name}: history={len(history_index):,}, "
        f"future={len(future_index):,}, physically skipped={skipped:,}",
        flush=True,
    )
    started = time.time()
    prepared = prepare_pair(raw.iloc[history_index], raw.iloc[future_index])
    prepared["future_y"] = raw.iloc[future_index][TARGET].astype("int8").reset_index(
        drop=True
    )
    combined_metadata = pd.concat(
        [prepared["metadata_history"], prepared["metadata_future"]],
        ignore_index=True,
    )
    segments = assign_segments(
        combined_metadata,
        np.arange(len(history_index), dtype="int64"),
        np.arange(
            len(history_index), len(history_index) + len(future_index), dtype="int64"
        ),
    )

    tuned = fixed_iterations is None
    identity_iterations = None if tuned else fixed_iterations["cat_identity"]
    identity_prediction, identity_trees = fit_cat_source(
        prepared,
        prepared["views"]["identity"],
        SEEDS[:2],
        f"{spec.name}_cat_identity",
        identity_iterations,
    )
    history_prediction, history_trees = fit_cat_source(
        prepared,
        prepared["views"]["history"],
        (42,),
        f"{spec.name}_cat_history",
        None if tuned else fixed_iterations["cat_history"],
    )
    weights, domain_report = domain_weights(prepared, seed=7300 + spec.train_end_day)
    weighted_prediction, weighted_trees = fit_cat_source(
        prepared,
        prepared["views"]["identity"],
        (3407,),
        f"{spec.name}_cat_weighted",
        None if tuned else fixed_iterations["cat_weighted"],
        sample_weight=weights,
    )
    lgb_giba_prediction, lgb_giba_trees = fit_lgb_source(
        prepared,
        prepared["views"]["identity"],
        "lgb_giba",
        9203,
        f"{spec.name}_lgb_giba",
        None if tuned else fixed_iterations["lgb_giba"],
    )
    lgb_cold_prediction, lgb_cold_trees = fit_lgb_source(
        prepared,
        prepared["views"]["cold"],
        "lgb_cold",
        13303,
        f"{spec.name}_lgb_cold",
        None if tuned else fixed_iterations["lgb_cold"],
    )
    xgb_prediction, xgb_trees = fit_xgb_source(
        prepared,
        prepared["views"]["identity"],
        2026,
        f"{spec.name}_xgb_giba",
        None if tuned else fixed_iterations["xgb_giba"],
    )

    frame = pd.DataFrame(
        {
            "row_index": future_index,
            "TransactionID": raw.iloc[future_index]["TransactionID"].to_numpy(),
            TARGET: prepared["future_y"].to_numpy(),
            "fold": spec.name,
            "stage": spec.stage,
            "horizon": spec.horizon,
            "segment": segments,
            "strict_uid": prepared["metadata_future"]["strict_uid"]
            .astype("string")
            .fillna("<INVALID>"),
            "strict_valid": prepared["metadata_future"]["strict_valid"].to_numpy(),
            "cat_identity": identity_prediction,
            "cat_history": history_prediction,
            "cat_weighted": weighted_prediction,
            "lgb_giba": lgb_giba_prediction,
            "lgb_cold": lgb_cold_prediction,
            "xgb_giba": xgb_prediction,
        }
    )
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    source_auc = {source: auc(frame[TARGET], frame[source]) for source in SOURCE_NAMES}
    report = {
        "cached": False,
        "spec": asdict(spec),
        "history_rows": len(history_index),
        "future_rows": len(future_index),
        "physically_skipped_rows": skipped,
        "segments": {segment: int(np.sum(segments == segment)) for segment in SEGMENTS},
        "source_auc": source_auc,
        "best_iterations": {
            "cat_identity": identity_trees,
            "cat_history": history_trees,
            "cat_weighted": weighted_trees,
            "lgb_giba": [lgb_giba_trees],
            "lgb_cold": [lgb_cold_trees],
            "xgb_giba": [xgb_trees],
        },
        "domain": domain_report,
        "feature_stats": prepared["stats"],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    print(json.dumps({"source_auc": source_auc, "minutes": report["elapsed_minutes"]}, indent=2), flush=True)
    del prepared, combined_metadata, weights
    gc.collect()
    return frame, report


def derive_fixed_iterations(reports: list[dict]) -> dict[str, int]:
    limits = {
        "cat_identity": CAT_PARAMS["iterations"],
        "cat_history": CAT_PARAMS["iterations"],
        "cat_weighted": CAT_PARAMS["iterations"],
        "lgb_giba": LGB_GIBA_PARAMS["n_estimators"],
        "lgb_cold": LGB_COLD_PARAMS["n_estimators"],
        "xgb_giba": XGB_PARAMS["n_estimators"],
    }
    result = {}
    for source in SOURCE_NAMES:
        scaled_values = []
        for report in reports:
            history_rows = max(int(report.get("history_rows", 0)), 1)
            scale = np.sqrt(EXPECTED_TRAIN_ROWS / history_rows)
            scaled_values.extend(
                value * scale
                for value in report.get("best_iterations", {}).get(source, [])
            )
        if not scaled_values:
            result[source] = int(limits[source])
            continue
        # A mild upper quantile compensates for the larger final training set,
        # while several horizons keep one anomalously long fold from dominating.
        estimate = int(np.ceil(np.quantile(scaled_values, 0.65)))
        result[source] = int(np.clip(estimate, 25, limits[source]))
    return result


def derive_spec_iterations(
    reports: list[dict],
    spec: PairSpec,
    raw: pd.DataFrame,
) -> dict[str, int]:
    """Scale development tree counts to a particular untouched lock history."""
    limits = {
        "cat_identity": CAT_PARAMS["iterations"],
        "cat_history": CAT_PARAMS["iterations"],
        "cat_weighted": CAT_PARAMS["iterations"],
        "lgb_giba": LGB_GIBA_PARAMS["n_estimators"],
        "lgb_cold": LGB_COLD_PARAMS["n_estimators"],
        "xgb_giba": XGB_PARAMS["n_estimators"],
    }
    source_horizon = 60 if spec.horizon == 75 else spec.horizon
    selected = [
        report
        for report in reports
        if int(report["spec"]["horizon"]) == source_horizon
    ]
    target_rows = int(
        np.sum(
            raw["TransactionDT"].to_numpy(dtype="float64") / DAY_SECONDS
            < spec.train_end_day
        )
    )
    output = {}
    for source in SOURCE_NAMES:
        scaled = []
        for report in selected:
            source_rows = max(int(report["history_rows"]), 1)
            factor = np.sqrt(target_rows / source_rows)
            scaled.extend(
                value * factor
                for value in report["best_iterations"].get(source, [])
            )
        if not scaled:
            output[source] = limits[source]
        else:
            output[source] = int(
                np.clip(np.ceil(np.quantile(scaled, 0.55)), 25, limits[source])
            )
    return output


def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source in SOURCE_NAMES:
        result[f"{source}_rank"] = rank_prediction(result[source])
    return result


def simplex_weights(size: int, denominator: int = 10):
    for values in product(range(denominator + 1), repeat=size):
        if sum(values) == denominator:
            yield np.asarray(values, dtype="float64") / denominator


def score_weights(
    frames: list[pd.DataFrame],
    segment: str,
    weights: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    columns = [f"{source}_rank" for source in SOURCE_NAMES]
    scores = []
    gains = []
    anchor = np.zeros(len(SOURCE_NAMES), dtype="float64")
    anchor[0] = 1.0
    for frame in frames:
        mask = frame["segment"].eq(segment).to_numpy()
        if np.unique(frame.loc[mask, TARGET]).size < 2:
            continue
        matrix = frame.loc[mask, columns].to_numpy()
        y = frame.loc[mask, TARGET].to_numpy()
        score = auc(y, matrix @ weights)
        base = auc(y, matrix @ anchor)
        scores.append(score)
        gains.append(score - base)
    return float(np.mean(scores)), scores, gains


def learn_weights(dev_frames: list[pd.DataFrame]) -> tuple[dict, dict]:
    ranked = [add_ranks(frame) for frame in dev_frames]
    global_weights = {}
    report = {"global": {}, "local": {}}
    anchor = np.zeros(len(SOURCE_NAMES), dtype="float64")
    anchor[0] = 1.0

    for segment in SEGMENTS:
        candidates = []
        for weights in simplex_weights(len(SOURCE_NAMES), denominator=5):
            mean_score, scores, gains = score_weights(ranked, segment, weights)
            positive = int(np.sum(np.asarray(gains) > 0))
            candidates.append((mean_score, positive, float(np.mean(gains)), weights, scores, gains))
        candidates.sort(key=lambda row: (row[1], row[2], row[0]), reverse=True)
        best = candidates[0]
        required = max(1, int(np.ceil(len(best[5]) * 0.60)))
        weights = best[3] if best[1] >= required and best[2] > 0 else anchor.copy()
        global_weights[segment] = weights
        report["global"][segment] = {
            "weights": dict(zip(SOURCE_NAMES, weights.tolist())),
            "mean_auc": best[0],
            "positive_folds": best[1],
            "mean_gain": best[2],
            "fold_auc": best[4],
            "fold_gain": best[5],
        }

    horizon_weights = {}
    for horizon in (30, 45, 60):
        local_frames = [
            frame for frame in ranked if int(frame["horizon"].iloc[0]) == horizon
        ]
        horizon_weights[str(horizon)] = {}
        report["local"][str(horizon)] = {}
        for segment in SEGMENTS:
            best = None
            for weights in simplex_weights(len(SOURCE_NAMES), denominator=5):
                values = score_weights(local_frames, segment, weights)
                row = (values[0], float(np.mean(values[2])), weights, values)
                if best is None or row[:2] > best[:2]:
                    best = row
            assert best is not None
            alpha = len(local_frames) / (len(local_frames) + 3.0)
            shrunk = alpha * best[2] + (1.0 - alpha) * global_weights[segment]
            shrunk /= shrunk.sum()
            horizon_weights[str(horizon)][segment] = dict(
                zip(SOURCE_NAMES, shrunk.tolist())
            )
            report["local"][str(horizon)][segment] = {
                "raw_weights": dict(zip(SOURCE_NAMES, best[2].tolist())),
                "weights": horizon_weights[str(horizon)][segment],
                "alpha": alpha,
                "mean_auc": best[0],
                "mean_gain": best[1],
                "fold_auc": best[3][1],
                "fold_gain": best[3][2],
            }
    horizon_weights["75"] = {
        segment: dict(horizon_weights["60"][segment]) for segment in SEGMENTS
    }
    report["local"]["75"] = {"source": "h60"}
    return horizon_weights, report


def blend_frame(frame: pd.DataFrame, horizon_weights: dict) -> np.ndarray:
    ranked = add_ranks(frame)
    prediction = np.zeros(len(frame), dtype="float64")
    columns = [f"{source}_rank" for source in SOURCE_NAMES]
    weights = horizon_weights[str(int(frame["horizon"].iloc[0]))]
    for segment in SEGMENTS:
        mask = ranked["segment"].eq(segment).to_numpy()
        vector = np.asarray([weights[segment][source] for source in SOURCE_NAMES])
        prediction[mask] = ranked.loc[mask, columns].to_numpy() @ vector
    return prediction


def aggregate_group(
    prediction: np.ndarray,
    group: pd.Series,
    valid: np.ndarray,
    method: str,
) -> np.ndarray:
    result = prediction.copy()
    values = pd.DataFrame(
        {"group": group.loc[valid].to_numpy(), "prediction": prediction[valid]}
    )
    grouped = values.groupby("group", sort=False)["prediction"]
    if method == "mean":
        aggregate = grouped.mean()
    elif method == "q75":
        aggregate = grouped.quantile(0.75)
    elif method == "max":
        aggregate = grouped.max()
    else:
        raise ValueError(method)
    result[valid] = values["group"].map(aggregate).to_numpy()
    return result


def choose_uid_recipe(dev_frames: list[pd.DataFrame], horizon_weights: dict) -> tuple[dict, list[dict]]:
    rows = []
    candidates = [("none", 0.0)] + [
        (method, weight)
        for method in ("mean", "q75", "max")
        for weight in (0.05, 0.10, 0.20, 0.30)
    ]
    for method, weight in candidates:
        gains = []
        scores = []
        for frame in dev_frames:
            base = blend_frame(frame, horizon_weights)
            if method == "none":
                prediction = base
            else:
                valid = frame["strict_valid"].to_numpy(dtype=bool)
                grouped = aggregate_group(base, frame["strict_uid"], valid, method)
                prediction = (1.0 - weight) * base + weight * grouped
            scores.append(auc(frame[TARGET], prediction))
            gains.append(scores[-1] - auc(frame[TARGET], base))
        rows.append(
            {
                "method": method,
                "weight": weight,
                "mean_auc": float(np.mean(scores)),
                "mean_gain": float(np.mean(gains)),
                "min_gain": float(np.min(gains)),
                "positive_folds": int(np.sum(np.asarray(gains) > 0)),
                "fold_auc": scores,
                "fold_gain": gains,
            }
        )
    rows.sort(
        key=lambda row: (row["positive_folds"], row["min_gain"], row["mean_gain"]),
        reverse=True,
    )
    best = rows[0]
    if best["mean_gain"] <= 0 or best["positive_folds"] < 4:
        best = next(row for row in rows if row["method"] == "none")
    return {"method": best["method"], "weight": best["weight"]}, rows


def apply_uid_recipe(frame: pd.DataFrame, prediction: np.ndarray, recipe: dict) -> np.ndarray:
    if recipe["method"] == "none" or recipe["weight"] == 0:
        return prediction
    valid = frame["strict_valid"].to_numpy(dtype=bool)
    grouped = aggregate_group(
        prediction,
        frame["strict_uid"],
        valid,
        recipe["method"],
    )
    return (1.0 - recipe["weight"]) * prediction + recipe["weight"] * grouped


def evaluate_frame(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    result = {
        "overall": auc(frame[TARGET], prediction),
        "anchor": auc(frame[TARGET], frame["cat_identity"]),
        "gain": auc(frame[TARGET], prediction) - auc(frame[TARGET], frame["cat_identity"]),
        "segments": {},
    }
    for segment in SEGMENTS:
        mask = frame["segment"].eq(segment).to_numpy()
        result["segments"][segment] = {
            "rows": int(mask.sum()),
            "auc": auc(frame.loc[mask, TARGET], prediction[mask]),
            "anchor_auc": auc(frame.loc[mask, TARGET], frame.loc[mask, "cat_identity"]),
        }
    return result


def apply_lock_fallbacks(
    lock_frames: list[pd.DataFrame],
    horizon_weights: dict,
) -> tuple[dict, dict]:
    """Use train-only lock folds for conservative segment fallback decisions."""
    adjusted = copy.deepcopy(horizon_weights)
    anchor = {source: float(source == "cat_identity") for source in SOURCE_NAMES}
    segment_gains: dict[str, dict[str, float]] = {segment: {} for segment in SEGMENTS}
    for frame in lock_frames:
        horizon = str(int(frame["horizon"].iloc[0]))
        candidate = blend_frame(frame, horizon_weights)
        for segment in SEGMENTS:
            mask = frame["segment"].eq(segment).to_numpy()
            segment_gains[segment][horizon] = (
                auc(frame.loc[mask, TARGET], candidate[mask])
                - auc(frame.loc[mask, TARGET], frame.loc[mask, "cat_identity"])
            )

    fallbacks = []
    for segment in SEGMENTS:
        gains = segment_gains[segment]
        positive = sum(value > 0 for value in gains.values())
        if positive < 3 or float(np.mean(list(gains.values()))) <= 0:
            for horizon in HORIZONS:
                adjusted[str(horizon)][segment] = dict(anchor)
            fallbacks.append(
                {
                    "segment": segment,
                    "horizons": list(HORIZONS),
                    "reason": "fewer than three positive lock horizons",
                }
            )
            continue
        # Horizon 75 has no development fold. A non-positive lock result there
        # therefore falls back instead of extrapolating the h60 recipe blindly.
        if gains.get("75", 0.0) <= 0:
            adjusted["75"][segment] = dict(anchor)
            fallbacks.append(
                {
                    "segment": segment,
                    "horizons": [75],
                    "reason": "no h75 development fold and non-positive h75 lock gain",
                }
            )
    return adjusted, {"segment_lock_gain": segment_gains, "fallbacks": fallbacks}


def run_backtest(force: bool, only_spec: str | None) -> dict:
    WORK_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    PREDICTION_DIR.mkdir(exist_ok=True)
    raw = read_official_train()
    dev_frames = []
    dev_reports = []
    for spec in DEV_SPECS:
        if only_spec and spec.name != only_spec:
            continue
        frame, report = train_pair(raw, spec, None, force)
        dev_frames.append(frame)
        dev_reports.append(report)
    if only_spec:
        payload = {"spec": only_spec, "reports": dev_reports}
        save_json(WORK_DIR / f"{only_spec}_report.json", payload)
        return payload
    if len(dev_frames) != len(DEV_SPECS):
        raise RuntimeError("Development fold set is incomplete")

    fixed_iterations = derive_fixed_iterations(dev_reports)
    horizon_weights, weight_report = learn_weights(dev_frames)
    uid_recipe, uid_search = choose_uid_recipe(dev_frames, horizon_weights)
    recipe = {
        "data_policy": "official train/test only; embargo rows excluded from feature building",
        "sources": list(SOURCE_NAMES),
        "segments": list(SEGMENTS),
        "horizons": list(HORIZONS),
        "fixed_iterations": fixed_iterations,
        "horizon_weights": horizon_weights,
        "uid_postprocess": uid_recipe,
        "seeds": list(SEEDS),
    }
    save_json(RECIPE_PATH, recipe)

    lock_frames = []
    lock_reports = []
    lock_iterations = {}
    for spec in LOCK_SPECS:
        spec_iterations = derive_spec_iterations(dev_reports, spec, raw)
        lock_iterations[spec.name] = spec_iterations
        frame, report = train_pair(raw, spec, spec_iterations, force)
        lock_frames.append(frame)
        lock_reports.append(report)

    adjusted_weights, confirmation = apply_lock_fallbacks(
        lock_frames, horizon_weights
    )
    recipe["horizon_weights_before_lock_fallback"] = horizon_weights
    recipe["horizon_weights"] = adjusted_weights
    recipe["lock_fallback_policy"] = confirmation
    save_json(RECIPE_PATH, recipe)

    dev_evaluation = {}
    for frame in dev_frames:
        prediction = apply_uid_recipe(
            frame, blend_frame(frame, adjusted_weights), uid_recipe
        )
        dev_evaluation[str(frame["fold"].iloc[0])] = evaluate_frame(frame, prediction)
    lock_evaluation = {}
    for frame in lock_frames:
        prediction = apply_uid_recipe(
            frame, blend_frame(frame, adjusted_weights), uid_recipe
        )
        lock_evaluation[str(frame["fold"].iloc[0])] = evaluate_frame(frame, prediction)

    mean_lock_gain = float(
        np.mean([value["gain"] for value in lock_evaluation.values()])
    )
    report = {
        "data_policy": recipe["data_policy"],
        "development_specs": [asdict(spec) for spec in DEV_SPECS],
        "lock_specs": [asdict(spec) for spec in LOCK_SPECS],
        "development_fit": dev_reports,
        "fixed_iterations": fixed_iterations,
        "lock_iterations": lock_iterations,
        "weight_search": weight_report,
        "uid_search": uid_search,
        "lock_fallback_confirmation": confirmation,
        "recipe": recipe,
        "development_evaluation": dev_evaluation,
        "lock_fit": lock_reports,
        "lock_evaluation": lock_evaluation,
        "mean_lock_gain_over_multiseed_cat": mean_lock_gain,
        "accepted": bool(mean_lock_gain > 0),
    }
    save_json(REPORT_PATH, report)
    print("\nLOCK RESULTS", flush=True)
    print(json.dumps(lock_evaluation, indent=2), flush=True)
    print(f"Mean lock gain: {mean_lock_gain:+.9f}", flush=True)
    return report


def interpolated_source_weights(
    horizon: np.ndarray,
    segments: np.ndarray,
    recipe: dict,
) -> np.ndarray:
    grid = np.asarray(HORIZONS, dtype="float64")
    output = np.zeros((len(horizon), len(SOURCE_NAMES)), dtype="float64")
    for segment in SEGMENTS:
        mask = segments == segment
        if not mask.any():
            continue
        for source_number, source in enumerate(SOURCE_NAMES):
            values = [
                recipe["horizon_weights"][str(value)][segment][source]
                for value in HORIZONS
            ]
            output[mask, source_number] = np.interp(
                horizon[mask], grid, values
            )
    output /= output.sum(axis=1, keepdims=True)
    return output


def train_final_source_models(prepared: dict, recipe: dict, force: bool) -> pd.DataFrame:
    fixed = recipe["fixed_iterations"]
    output_cache = WORK_DIR / "raw_test_sources.csv"
    if output_cache.exists() and not force:
        frame = pd.read_csv(output_cache)
        if set(SOURCE_NAMES).issubset(frame.columns) and len(frame) == len(prepared["future"]):
            return frame

    prepared["future_y"] = None
    active_sources = {
        source
        for horizon in recipe["horizon_weights"].values()
        for segment in horizon.values()
        for source, weight in segment.items()
        if float(weight) > 0
    }
    weights, domain_report = domain_weights(prepared, seed=9307)
    identity, _ = fit_cat_source(
        prepared,
        prepared["views"]["identity"],
        SEEDS,
        "final_cat_identity",
        fixed["cat_identity"],
    )
    if "cat_history" in active_sources:
        history, _ = fit_cat_source(
            prepared,
            prepared["views"]["history"],
            SEEDS[:2],
            "final_cat_history",
            fixed["cat_history"],
        )
    else:
        print("Skipping inactive final source: cat_history", flush=True)
        history = identity.copy()
    weighted, _ = fit_cat_source(
        prepared,
        prepared["views"]["identity"],
        (3407,),
        "final_cat_weighted",
        fixed["cat_weighted"],
        sample_weight=weights,
    )
    lgb_giba_predictions = []
    lgb_cold_predictions = []
    for seed in (9203, 13303):
        value, _ = fit_lgb_source(
            prepared,
            prepared["views"]["identity"],
            "lgb_giba",
            seed,
            f"final_lgb_giba_s{seed}",
            fixed["lgb_giba"],
        )
        lgb_giba_predictions.append(value)
        value, _ = fit_lgb_source(
            prepared,
            prepared["views"]["cold"],
            "lgb_cold",
            seed,
            f"final_lgb_cold_s{seed}",
            fixed["lgb_cold"],
        )
        lgb_cold_predictions.append(value)
    xgb_prediction, _ = fit_xgb_source(
        prepared,
        prepared["views"]["identity"],
        2026,
        "final_xgb_giba",
        fixed["xgb_giba"],
    )
    frame = pd.DataFrame(
        {
            "TransactionID": prepared["future"]["TransactionID"].to_numpy(),
            "cat_identity": identity,
            "cat_history": history,
            "cat_weighted": weighted,
            "lgb_giba": np.mean(lgb_giba_predictions, axis=0),
            "lgb_cold": np.mean(lgb_cold_predictions, axis=0),
            "xgb_giba": xgb_prediction,
        }
    )
    frame.to_csv(output_cache, index=False)
    save_json(
        WORK_DIR / "final_domain_report.json",
        {"active_sources": sorted(active_sources), **domain_report},
    )
    return frame


def run_final(force: bool) -> dict:
    if not RECIPE_PATH.exists():
        raise FileNotFoundError("Run backtest before final training")
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    train_raw = read_official_train()
    test_raw = read_official_test()
    y = train_raw[TARGET].astype("int8").reset_index(drop=True)
    prepared = prepare_pair(train_raw, test_raw)
    if not np.array_equal(
        prepared["future"]["TransactionID"].to_numpy(),
        test_raw["TransactionID"].to_numpy(),
    ):
        raise ValueError("Test rows moved during feature preparation")
    sources = train_final_source_models(prepared, recipe, force)

    combined_metadata = pd.concat(
        [prepared["metadata_history"], prepared["metadata_future"]],
        ignore_index=True,
    )
    segments = assign_segments(
        combined_metadata,
        np.arange(len(train_raw), dtype="int64"),
        np.arange(len(train_raw), len(train_raw) + len(test_raw), dtype="int64"),
    )
    train_end = float(train_raw["TransactionDT"].max() / DAY_SECONDS)
    horizon = test_raw["TransactionDT"].to_numpy(dtype="float64") / DAY_SECONDS - train_end
    rank_matrix = np.column_stack(
        [rank_prediction(sources[source]) for source in SOURCE_NAMES]
    )
    row_weights = interpolated_source_weights(horizon, segments, recipe)
    prediction = np.sum(rank_matrix * row_weights, axis=1)
    post_frame = pd.DataFrame(
        {
            "strict_uid": prepared["metadata_future"]["strict_uid"]
            .astype("string")
            .fillna("<INVALID>"),
            "strict_valid": prepared["metadata_future"]["strict_valid"].to_numpy(),
        }
    )
    prediction = apply_uid_recipe(post_frame, prediction, recipe["uid_postprocess"])
    prediction = rank_prediction(prediction)

    source_output = sources.copy()
    source_output["segment"] = segments
    source_output["forecast_horizon"] = horizon
    source_output["prediction"] = prediction
    source_output.to_csv(SOURCE_PATH, index=False)

    sample = pd.read_csv(ROOT / "sample_submission.csv").drop(
        columns="Unnamed: 0", errors="ignore"
    )
    if not np.array_equal(
        sample["TransactionID"].to_numpy(), test_raw["TransactionID"].to_numpy()
    ):
        raise ValueError("Sample submission IDs differ from official test")
    submission = sample[["TransactionID"]].copy()
    submission[TARGET] = prediction
    if not np.isfinite(submission[TARGET]).all():
        raise ValueError("Non-finite final predictions")
    submission.to_csv(SUBMISSION_PATH, index=False)
    report = {
        "data_policy": recipe["data_policy"],
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "sources": list(SOURCE_NAMES),
        "segments": {segment: int(np.sum(segments == segment)) for segment in SEGMENTS},
        "forecast_horizon": {"min": float(horizon.min()), "max": float(horizon.max())},
        "submission": SUBMISSION_PATH.name,
        "prediction": {
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "mean": float(prediction.mean()),
        },
    }
    save_json(WORK_DIR / "final_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    del prepared, combined_metadata, y
    gc.collect()
    return report


def self_test() -> None:
    values = list(simplex_weights(3, denominator=5))
    assert len(values) == 21
    assert all(np.isclose(value.sum(), 1.0) for value in values)
    horizon = np.asarray([30.0, 37.5, 75.0])
    segments = np.asarray(["strict", "partial", "cold"])
    recipe = {
        "horizon_weights": {
            str(h): {
                segment: {
                    source: float(source == "cat_identity")
                    for source in SOURCE_NAMES
                }
                for segment in SEGMENTS
            }
            for h in HORIZONS
        }
    }
    weights = interpolated_source_weights(horizon, segments, recipe)
    assert np.allclose(weights[:, 0], 1.0)
    assert np.allclose(weights.sum(axis=1), 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("backtest", "final", "all", "inspect"), default="all"
    )
    parser.add_argument("--spec", choices=[spec.name for spec in DEV_SPECS])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("Self-test: OK")
        return
    if args.mode == "inspect":
        train = read_official_train()
        test = read_official_test()
        print(
            json.dumps(
                {
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_day_max": float(train["TransactionDT"].max() / DAY_SECONDS),
                    "test_day_min": float(test["TransactionDT"].min() / DAY_SECONDS),
                    "test_day_max": float(test["TransactionDT"].max() / DAY_SECONDS),
                    "development": [asdict(spec) for spec in DEV_SPECS],
                    "lock": [asdict(spec) for spec in LOCK_SPECS],
                },
                indent=2,
            )
        )
        return

    started = time.time()
    if args.mode in {"backtest", "all"}:
        run_backtest(args.force, args.spec)
    if args.mode in {"final", "all"} and not args.spec:
        run_final(args.force)
    print(f"Total elapsed: {(time.time() - started) / 60.0:.2f} min", flush=True)


if __name__ == "__main__":
    main()

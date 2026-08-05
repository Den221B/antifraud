from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_features import TARGET


SOURCE_NAMES = (
    "identity_cat",
    "repeat_cat",
    "generic_cat",
    "generic_lgb",
)
SEGMENT_NAMES = ("strict", "partial", "cold")
HORIZONS = (30, 45, 60, 75)


@dataclass(frozen=True)
class FoldSpec:
    name: str
    stage: str
    horizon: int
    train_end_day: int
    valid_start_day: int
    valid_end_day: int

    def to_dict(self) -> dict:
        return asdict(self)


DEV_FOLDS = (
    FoldSpec("dev_h30_a", "dev", 30, 15, 45, 60),
    FoldSpec("dev_h30_b", "dev", 30, 30, 60, 75),
    FoldSpec("dev_h45_a", "dev", 45, 15, 60, 75),
    FoldSpec("dev_h45_b", "dev", 45, 30, 75, 90),
    FoldSpec("dev_h60", "dev", 60, 15, 75, 90),
)

LOCK_FOLDS = (
    FoldSpec("lock_h30", "lock", 30, 60, 90, 106),
    FoldSpec("lock_h45", "lock", 45, 45, 90, 106),
    FoldSpec("lock_h60", "lock", 60, 30, 90, 106),
    FoldSpec("lock_h75", "lock", 75, 15, 90, 106),
)


def read_labeled_split(data_dir: Path, split: str = "train") -> pd.DataFrame:
    transaction = pd.read_csv(data_dir / f"{split}_transaction.csv")
    identity = pd.read_csv(data_dir / f"{split}_identity.csv")
    transaction.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    identity.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    frame = transaction.merge(identity, on="TransactionID", how="left")
    if TARGET not in frame:
        raise ValueError(f"{split} does not contain {TARGET}")
    return frame


def read_unlabeled_split(data_dir: Path, split: str) -> pd.DataFrame:
    excluded = {TARGET, "Unnamed: 0"}
    transaction = pd.read_csv(
        data_dir / f"{split}_transaction.csv",
        usecols=lambda column: column not in excluded,
    )
    identity = pd.read_csv(
        data_dir / f"{split}_identity.csv",
        usecols=lambda column: column not in excluded,
    )
    frame = transaction.merge(identity, on="TransactionID", how="left")
    if TARGET in frame:
        raise AssertionError(f"Unexpected {TARGET} in unlabeled split")
    return frame


def make_fold_indices(day: pd.Series, spec: FoldSpec) -> tuple[np.ndarray, np.ndarray]:
    values = day.to_numpy()
    train_index = np.flatnonzero(values < spec.train_end_day)
    valid_index = np.flatnonzero(
        (values >= spec.valid_start_day) & (values < spec.valid_end_day)
    )
    if not len(train_index) or not len(valid_index):
        raise ValueError(f"Empty train or validation window for {spec.name}")
    return train_index, valid_index


def uid_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "uid_card_addr_d1_email",
        "uid_card_addr_d1",
        "uid_d1_email",
        "card1",
        "addr1",
        "D1_origin_day",
        "P_emaildomain",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing UID columns: {sorted(missing)}")

    strict_valid = (
        frame["card1"].notna()
        & frame["addr1"].notna()
        & frame["D1_origin_day"].notna()
    )
    d1_email_valid = (
        frame["D1_origin_day"].notna()
        & frame["P_emaildomain"].notna()
    )
    return pd.DataFrame(
        {
            "strict_uid": frame["uid_card_addr_d1_email"].astype("string"),
            "card_d1_uid": frame["uid_card_addr_d1"].astype("string"),
            "d1_email_uid": frame["uid_d1_email"].astype("string"),
            "strict_valid": strict_valid.to_numpy(dtype=bool),
            "card_d1_valid": strict_valid.to_numpy(dtype=bool),
            "d1_email_valid": d1_email_valid.to_numpy(dtype=bool),
        },
        index=frame.index,
    )


def _seen_mask(
    train_values: pd.Series,
    train_valid: pd.Series,
    query_values: pd.Series,
    query_valid: pd.Series,
) -> np.ndarray:
    seen = pd.Index(train_values.loc[train_valid].dropna().unique())
    return query_valid.to_numpy(dtype=bool) & query_values.isin(seen).to_numpy()


def assign_segments(
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    query_index: np.ndarray,
) -> np.ndarray:
    train = metadata.iloc[train_index]
    query = metadata.iloc[query_index]
    strict = _seen_mask(
        train["strict_uid"],
        train["strict_valid"],
        query["strict_uid"],
        query["strict_valid"],
    )
    card_d1 = _seen_mask(
        train["card_d1_uid"],
        train["card_d1_valid"],
        query["card_d1_uid"],
        query["card_d1_valid"],
    )
    d1_email = _seen_mask(
        train["d1_email_uid"],
        train["d1_email_valid"],
        query["d1_email_uid"],
        query["d1_email_valid"],
    )
    result = np.full(len(query), "cold", dtype=object)
    result[~strict & (card_d1 | d1_email)] = "partial"
    result[strict] = "strict"
    return result


def repeated_training_indices(
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    minimum_count: int = 2,
) -> np.ndarray:
    train = metadata.iloc[train_index]
    valid_uid = train.loc[train["strict_valid"], "strict_uid"]
    counts = valid_uid.value_counts(dropna=False)
    repeated = pd.Index(counts.index[counts >= minimum_count])
    keep = train["strict_valid"].to_numpy(dtype=bool) & train[
        "strict_uid"
    ].isin(repeated).to_numpy()
    return train_index[keep]


def rank_prediction(values: Iterable[float]) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy()


def add_source_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source in SOURCE_NAMES:
        result[f"{source}_rank"] = rank_prediction(result[source].to_numpy())
    return result


def safe_auc(y_true: Iterable[int], prediction: Iterable[float]) -> float | None:
    y = np.asarray(y_true)
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, np.asarray(prediction)))


def simplex_weights(n_sources: int, denominator: int = 10) -> Iterable[np.ndarray]:
    for values in product(range(denominator + 1), repeat=n_sources):
        if sum(values) == denominator:
            yield np.asarray(values, dtype="float64") / denominator


def _mean_fold_auc(
    fold_values: list[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
) -> float:
    scores = []
    for y_true, source_values in fold_values:
        score = safe_auc(y_true, source_values @ weights)
        if score is not None:
            scores.append(score)
    return float(np.mean(scores)) if scores else float("-inf")


def search_segment_weights(
    frames: list[pd.DataFrame],
    segment: str,
    denominator: int = 10,
) -> tuple[np.ndarray, float]:
    columns = [f"{source}_rank" for source in SOURCE_NAMES]
    fold_values = []
    for frame in frames:
        mask = frame["segment"].eq(segment).to_numpy()
        if mask.any():
            fold_values.append(
                (
                    frame.loc[mask, TARGET].to_numpy(),
                    frame.loc[mask, columns].to_numpy(),
                )
            )
    best_weights = np.full(len(SOURCE_NAMES), 1 / len(SOURCE_NAMES))
    best_score = float("-inf")
    coarse_denominator = min(5, denominator)
    for weights in simplex_weights(len(SOURCE_NAMES), coarse_denominator):
        score = _mean_fold_auc(fold_values, weights)
        if score > best_score + 1e-12:
            best_score = score
            best_weights = weights
    if denominator > coarse_denominator:
        radius = 2.0 / coarse_denominator
        for weights in simplex_weights(len(SOURCE_NAMES), denominator):
            if np.abs(weights - best_weights).sum() > radius + 1e-12:
                continue
            score = _mean_fold_auc(fold_values, weights)
            if score > best_score + 1e-12:
                best_score = score
                best_weights = weights
    return best_weights, best_score


def learn_horizon_weights(
    dev_frames: list[pd.DataFrame],
    denominator: int = 10,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict]:
    global_weights = {}
    search_report = {"global": {}, "local": {}}
    equal = np.full(len(SOURCE_NAMES), 1 / len(SOURCE_NAMES))
    for segment in SEGMENT_NAMES:
        weights, score = search_segment_weights(dev_frames, segment, denominator)
        if not np.isfinite(score):
            weights = equal.copy()
        global_weights[segment] = weights
        search_report["global"][segment] = {
            "auc": None if not np.isfinite(score) else score,
            "weights": dict(zip(SOURCE_NAMES, weights.tolist())),
        }

    result: dict[str, dict[str, dict[str, float]]] = {}
    for horizon in (30, 45, 60):
        local_frames = [
            frame for frame in dev_frames if int(frame["horizon"].iloc[0]) == horizon
        ]
        result[str(horizon)] = {}
        search_report["local"][str(horizon)] = {}
        for segment in SEGMENT_NAMES:
            local, score = search_segment_weights(
                local_frames, segment, denominator
            )
            if not np.isfinite(score):
                local = global_weights[segment]
            alpha = len(local_frames) / (len(local_frames) + 2.0)
            shrunk = alpha * local + (1.0 - alpha) * global_weights[segment]
            shrunk /= shrunk.sum()
            result[str(horizon)][segment] = dict(
                zip(SOURCE_NAMES, shrunk.tolist())
            )
            search_report["local"][str(horizon)][segment] = {
                "auc": None if not np.isfinite(score) else score,
                "folds": len(local_frames),
                "shrinkage": alpha,
                "raw_weights": dict(zip(SOURCE_NAMES, local.tolist())),
                "weights": dict(zip(SOURCE_NAMES, shrunk.tolist())),
            }

    result["75"] = {
        segment: dict(result["60"][segment]) for segment in SEGMENT_NAMES
    }
    search_report["local"]["75"] = {
        segment: {
            "auc": None,
            "folds": 0,
            "shrinkage": 0.0,
            "source": "h60",
            "weights": dict(result["75"][segment]),
        }
        for segment in SEGMENT_NAMES
    }
    return result, search_report


def blend_with_weights(
    frame: pd.DataFrame,
    weights: dict[str, dict[str, float]],
) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype="float64")
    for segment in SEGMENT_NAMES:
        mask = frame["segment"].eq(segment).to_numpy()
        if not mask.any():
            continue
        columns = [f"{source}_rank" for source in SOURCE_NAMES]
        vector = np.asarray([weights[segment][source] for source in SOURCE_NAMES])
        prediction[mask] = frame.loc[mask, columns].to_numpy() @ vector
    return prediction


def interpolated_blend(
    frame: pd.DataFrame,
    horizon_weights: dict[str, dict[str, dict[str, float]]],
) -> np.ndarray:
    horizons = np.asarray(HORIZONS, dtype="float64")
    prediction = np.zeros(len(frame), dtype="float64")
    source_values = frame[
        [f"{source}_rank" for source in SOURCE_NAMES]
    ].to_numpy()
    row_horizon = frame["forecast_horizon"].to_numpy(dtype="float64")
    for segment in SEGMENT_NAMES:
        mask = frame["segment"].eq(segment).to_numpy()
        if not mask.any():
            continue
        row_weights = np.column_stack(
            [
                np.interp(
                    row_horizon[mask],
                    horizons,
                    [
                        horizon_weights[str(h)][segment][source]
                        for h in HORIZONS
                    ],
                )
                for source in SOURCE_NAMES
            ]
        )
        row_weights /= row_weights.sum(axis=1, keepdims=True)
        prediction[mask] = np.sum(source_values[mask] * row_weights, axis=1)
    return prediction


def uid_aggregate(
    prediction: np.ndarray,
    uid: pd.Series,
    valid: pd.Series | np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "none":
        return np.asarray(prediction, dtype="float64").copy()
    valid_array = np.asarray(valid, dtype=bool)
    result = np.asarray(prediction, dtype="float64").copy()
    values = pd.DataFrame(
        {
            "prediction": result[valid_array],
            "uid": uid.iloc[np.flatnonzero(valid_array)].to_numpy(),
        }
    )
    grouped = values.groupby("uid", sort=False, dropna=False)["prediction"]
    if method == "mean":
        aggregate_by_uid = grouped.mean()
    elif method == "max":
        aggregate_by_uid = grouped.max()
    elif method == "q75":
        aggregate_by_uid = grouped.quantile(0.75)
    elif method == "q90":
        aggregate_by_uid = grouped.quantile(0.90)
    else:
        raise ValueError(f"Unknown UID aggregate: {method}")
    result[valid_array] = values["uid"].map(aggregate_by_uid).to_numpy()
    return result


def apply_uid_postprocess(
    prediction: np.ndarray,
    uid: pd.Series,
    valid: pd.Series | np.ndarray,
    method: str,
    weight: float,
) -> np.ndarray:
    if method == "none" or weight == 0:
        return np.asarray(prediction, dtype="float64").copy()
    aggregate = uid_aggregate(prediction, uid, valid, method)
    return (1.0 - weight) * np.asarray(prediction) + weight * aggregate


def choose_uid_postprocess(
    dev_frames: list[pd.DataFrame],
) -> tuple[dict, list[dict]]:
    candidates = [("none", 0.0)]
    for method in ("mean", "q75", "q90", "max"):
        for weight in (0.25, 0.50, 0.75, 1.0):
            candidates.append((method, weight))

    aggregates = []
    for frame in dev_frames:
        fold_aggregates = {"none": frame["gated"].to_numpy()}
        for method in ("mean", "q75", "q90", "max"):
            fold_aggregates[method] = uid_aggregate(
                frame["gated"].to_numpy(),
                frame["strict_uid"],
                frame["strict_valid"],
                method,
            )
        aggregates.append(fold_aggregates)

    rows = []
    for method, weight in candidates:
        fold_scores = []
        for frame, fold_aggregates in zip(dev_frames, aggregates):
            base = frame["gated"].to_numpy()
            prediction = (
                (1.0 - weight) * base + weight * fold_aggregates[method]
            )
            score = safe_auc(frame[TARGET], prediction)
            if score is not None:
                fold_scores.append(score)
        rows.append(
            {
                "method": method,
                "weight": weight,
                "mean_auc": float(np.mean(fold_scores)),
                "fold_auc": fold_scores,
            }
        )
    best = max(rows, key=lambda row: (row["mean_auc"], -row["weight"]))
    return dict(best), rows


def evaluate_prediction(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict:
    result = {"overall": safe_auc(frame[TARGET], prediction), "segments": {}}
    for segment in SEGMENT_NAMES:
        mask = frame["segment"].eq(segment).to_numpy()
        result["segments"][segment] = {
            "rows": int(mask.sum()),
            "rate": float(mask.mean()),
            "auc": safe_auc(frame.loc[mask, TARGET], prediction[mask]),
        }
    return result


def self_test() -> None:
    frame = pd.DataFrame(
        {
            "uid_card_addr_d1_email": ["a", "a", "b", "c", "a", "x"],
            "uid_card_addr_d1": ["aa", "aa", "bb", "cc", "aa", "xx"],
            "uid_d1_email": ["da", "da", "db", "dc", "da", "dx"],
            "card1": [1, 1, 2, 3, 1, np.nan],
            "addr1": [1, 1, 2, 3, 1, np.nan],
            "D1_origin_day": [1, 1, 2, 3, 1, np.nan],
            "P_emaildomain": ["e", "e", "e", "e", "e", None],
        }
    )
    metadata = uid_metadata(frame)
    train_index = np.asarray([0, 1, 2])
    query_index = np.asarray([3, 4, 5])
    assert assign_segments(metadata, train_index, query_index).tolist() == [
        "cold",
        "strict",
        "cold",
    ]
    assert repeated_training_indices(metadata, train_index).tolist() == [0, 1]
    prediction = np.asarray([0.1, 0.9, 0.4])
    grouped = uid_aggregate(
        prediction,
        pd.Series(["u", "u", "v"]),
        np.asarray([True, True, True]),
        "q90",
    )
    assert np.allclose(grouped, [0.82, 0.82, 0.4])
    weights = list(simplex_weights(4, 10))
    assert len(weights) == 286
    assert all(np.isclose(value.sum(), 1.0) for value in weights)

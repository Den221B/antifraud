"""Blend the clean-v2 six-model source into the honest feature-view stack."""

from __future__ import annotations

from itertools import product
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import clean_v2_pipeline as clean_v2
import refine_honest_client_segments as segments
import search_honest_featureview_meta as featureview
import search_honest_fullrow_lgb as fullrow
import train_honest_client_meta as client


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_cleanv2_blend"
REPORT_PATH = WORK_DIR / "report.json"
OUTPUT_PATH = ROOT / "submission_honest_cleanv2_blend.csv"
SOURCES = ("featureview", "clean_v2", "fullrow_lgb")
DENOMINATOR = 10
TOP_PER_SEGMENT = 12


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def simplex(size: int, denominator: int = DENOMINATOR):
    for values in product(range(denominator + 1), repeat=size):
        if sum(values) == denominator:
            yield np.asarray(values, dtype="float64") / denominator


def load_clean_prediction(path: Path, recipe: dict) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    prediction = clean_v2.blend_frame(frame, recipe["horizon_weights"])
    prediction = clean_v2.apply_uid_recipe(
        frame, prediction, recipe["uid_postprocess"]
    )
    return frame, prediction


def transform_sources(values: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    if mode == "probability":
        return values
    if mode == "rank":
        return {name: client.rank(prediction) for name, prediction in values.items()}
    raise ValueError(mode)


def safe_auc(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.unique(target).size < 2:
        return float("nan")
    return float(roc_auc_score(target, prediction))


def segment_candidates(
    target: np.ndarray,
    matrix: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    dt: np.ndarray,
    segment: str,
) -> list[dict]:
    segment_mask = labels == segment
    midpoint = np.median(dt)
    halves = (
        segment_mask & (dt <= midpoint),
        segment_mask & (dt > midpoint),
    )
    baseline_auc = safe_auc(target[segment_mask], baseline[segment_mask])
    baseline_halves = [safe_auc(target[mask], baseline[mask]) for mask in halves]
    rows = []
    for weights in simplex(len(SOURCES)):
        prediction = matrix @ weights
        half_auc = [safe_auc(target[mask], prediction[mask]) for mask in halves]
        half_gains = [
            score - base for score, base in zip(half_auc, baseline_halves)
        ]
        score = safe_auc(target[segment_mask], prediction[segment_mask])
        rows.append(
            {
                "weights": dict(zip(SOURCES, weights.tolist())),
                "auc": score,
                "gain": score - baseline_auc,
                "half_gains": half_gains,
                "min_half_gain": float(np.nanmin(half_gains)),
            }
        )
    stable = [row for row in rows if row["min_half_gain"] >= 0.0]
    if not stable:
        stable = [
            row
            for row in rows
            if row["weights"]["featureview"] == 1.0
        ]
    stable.sort(
        key=lambda row: (row["gain"], row["min_half_gain"]), reverse=True
    )
    return stable[:TOP_PER_SEGMENT]


def apply_segment_recipe(
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    weights: dict[str, dict[str, float]],
) -> np.ndarray:
    output = np.empty(len(labels), dtype="float64")
    for segment in segments.SEGMENTS:
        mask = labels == segment
        matrix = np.column_stack([values[source][mask] for source in SOURCES])
        vector = np.asarray([weights[segment][source] for source in SOURCES])
        output[mask] = matrix @ vector
    return output


def search_recipe(
    target: np.ndarray,
    raw_values: dict[str, np.ndarray],
    labels: np.ndarray,
    dt: np.ndarray,
) -> tuple[dict, list[dict], dict]:
    rows = []
    segment_search = {}
    midpoint = np.median(dt)
    halves = (dt <= midpoint, dt > midpoint)
    for mode in ("probability", "rank"):
        values = transform_sources(raw_values, mode)
        baseline = values["featureview"]
        baseline_auc = float(roc_auc_score(target, baseline))
        baseline_halves = [
            float(roc_auc_score(target[mask], baseline[mask])) for mask in halves
        ]
        per_segment = {
            segment: segment_candidates(
                target,
                np.column_stack([values[source] for source in SOURCES]),
                baseline,
                labels,
                dt,
                segment,
            )
            for segment in segments.SEGMENTS
        }
        segment_search[mode] = per_segment
        for combination in product(
            *(per_segment[segment] for segment in segments.SEGMENTS)
        ):
            weights = {
                segment: row["weights"]
                for segment, row in zip(segments.SEGMENTS, combination)
            }
            prediction = apply_segment_recipe(values, labels, weights)
            score = float(roc_auc_score(target, prediction))
            half_auc = [
                float(roc_auc_score(target[mask], prediction[mask]))
                for mask in halves
            ]
            half_gains = [
                value - base for value, base in zip(half_auc, baseline_halves)
            ]
            rows.append(
                {
                    "mode": mode,
                    "weights": weights,
                    "auc": score,
                    "gain": score - baseline_auc,
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
    return selected, rows, segment_search


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
    recipe = json.loads(
        (ROOT / "clean_v2/recipe.json").read_text(encoding="utf-8")
    )
    if "official train/test only" not in recipe.get("data_policy", ""):
        raise RuntimeError("clean_v2 recipe does not declare the official-data policy")

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
    feature_recipe = reference_report["recipe"]
    postprocess = feature_recipe["postprocess_locked_from_dev"]

    clean_dev_frame, clean_dev = load_clean_prediction(
        ROOT / "clean_v2/predictions/dev_h30_c.csv", recipe
    )
    clean_lock_frame, clean_lock = load_clean_prediction(
        ROOT / "clean_v2/predictions/lock_h30.csv", recipe
    )
    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_rows = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_rows = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
    if not np.array_equal(clean_dev_frame["row_index"].to_numpy(), dev_rows):
        raise RuntimeError("clean_v2 dev rows differ from the feature-view dev fold")
    if not np.array_equal(clean_lock_frame["row_index"].to_numpy(), lock_rows):
        raise RuntimeError("clean_v2 lock rows differ from the feature-view lock fold")
    if not np.array_equal(
        clean_dev_frame[client.TARGET].to_numpy(),
        oof.loc[dev_mask, client.TARGET].to_numpy(),
    ):
        raise RuntimeError("clean_v2 dev labels are not official-train labels")

    fullrow_recipe = json.loads(
        (ROOT / "honest_fullrow_lgb/report.json").read_text(encoding="utf-8")
    )
    dev_lgb_raw = np.load(ROOT / "honest_fullrow_lgb/dev_raw.npy")
    lock_lgb_raw = np.load(ROOT / "honest_fullrow_lgb/lock_raw.npy")
    dev_lgb = fullrow.transform_variant(
        dev_lgb_raw,
        fullrow_recipe["selected_blend"]["variant"],
        fold_groups[client.META_DEV_FOLD],
        postprocess,
    )
    lock_lgb = fullrow.transform_variant(
        lock_lgb_raw,
        fullrow_recipe["selected_blend"]["variant"],
        fold_groups[client.META_LOCK_FOLD],
        postprocess,
    )

    raw_dev = {
        "featureview": reference[client.META_DEV_FOLD],
        "clean_v2": clean_dev,
        "fullrow_lgb": dev_lgb,
    }
    y_dev = oof.loc[dev_mask, client.TARGET].to_numpy(dtype="int8")
    dt_train = pd.read_csv(
        ROOT / "train_transaction.csv", usecols=["TransactionDT"]
    )["TransactionDT"].to_numpy(dtype="float64")
    labels_dev = segments.segment_labels(
        membership_oof.loc[dev_mask].reset_index(drop=True)
    )
    selected, search_rows, segment_search = search_recipe(
        y_dev,
        raw_dev,
        labels_dev,
        dt_train[dev_rows],
    )

    raw_lock = transform_sources(
        {
            "featureview": reference[client.META_LOCK_FOLD],
            "clean_v2": clean_lock,
            "fullrow_lgb": lock_lgb,
        },
        selected["mode"],
    )
    labels_lock = segments.segment_labels(
        membership_oof.loc[lock_mask].reset_index(drop=True)
    )
    lock_prediction = apply_segment_recipe(
        raw_lock, labels_lock, selected["weights"]
    )
    y_lock = oof.loc[lock_mask, client.TARGET].to_numpy(dtype="int8")
    lock_auc = float(roc_auc_score(y_lock, lock_prediction))
    reference_lock_auc = float(
        reference_report["metrics"][client.META_LOCK_FOLD]["auc"]
    )
    accepted = bool(selected["gain"] > 0.0 and lock_auc > reference_lock_auc)

    baseline_test = pd.read_csv(ROOT / "submission_honest_featureview_client.csv")
    clean_test = pd.read_csv(ROOT / "submission_clean_v2.csv")
    clean_sources = pd.read_csv(
        ROOT / "clean_v2/test_source_predictions.csv",
        usecols=["TransactionID"],
    )
    for frame, name in (
        (clean_test, "clean_v2 submission"),
        (clean_sources, "clean_v2 sources"),
    ):
        if not np.array_equal(
            frame["TransactionID"].to_numpy(),
            baseline_test["TransactionID"].to_numpy(),
        ):
            raise RuntimeError(f"{name} differs from sample order")
    test_lgb_raw = np.load(ROOT / "honest_fullrow_lgb/test_raw.npy")
    test_lgb = fullrow.transform_variant(
        test_lgb_raw,
        fullrow_recipe["selected_blend"]["variant"],
        reference_report["test_groups"],
        postprocess,
    )
    raw_test = transform_sources(
        {
            "featureview": baseline_test[client.TARGET].to_numpy(dtype="float64"),
            "clean_v2": clean_test[client.TARGET].to_numpy(dtype="float64"),
            "fullrow_lgb": test_lgb,
        },
        selected["mode"],
    )
    test_prediction = apply_segment_recipe(
        raw_test,
        segments.segment_labels(membership_test),
        selected["weights"],
    )
    output = baseline_test[["TransactionID"]].copy()
    output[client.TARGET] = test_prediction
    output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": "official train/test only; official train labels for dev/lock",
        "selection": "simplex and mode on days 75-90; days 90-105 one-time lock",
        "sources": list(SOURCES),
        "selected": selected,
        "search_top": search_rows[:40],
        "segment_search_top": {
            mode: {
                segment: rows[:8] for segment, rows in per_segment.items()
            }
            for mode, per_segment in segment_search.items()
        },
        "source_auc": {
            "dev": {
                source: float(roc_auc_score(y_dev, values))
                for source, values in raw_dev.items()
            },
            "lock": {
                source: float(roc_auc_score(y_lock, values))
                for source, values in {
                    "featureview": reference[client.META_LOCK_FOLD],
                    "clean_v2": clean_lock,
                    "fullrow_lgb": lock_lgb,
                }.items()
            },
        },
        "reference_lock_auc": reference_lock_auc,
        "candidate_lock_auc": lock_auc,
        "lock_gain": lock_auc - reference_lock_auc,
        "accepted": accepted,
        "output": OUTPUT_PATH.name,
        "output_sha256": file_sha256(OUTPUT_PATH),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected,
                "source_auc": report["source_auc"],
                "reference_lock_auc": reference_lock_auc,
                "candidate_lock_auc": lock_auc,
                "lock_gain": lock_auc - reference_lock_auc,
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

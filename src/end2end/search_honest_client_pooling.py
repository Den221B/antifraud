"""Target-free client pooling selected on purged temporal validation."""

from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import finalize_honest_xgb_magic_blend as xgb_final
import refine_honest_client_segments as segments
import search_honest_featureview_meta as featureview
import search_honest_fullrow_lgb as fullrow
import train_honest_client_meta as client
import train_honest_magic_heavy_stack as heavy
import train_honest_xgb_magic as magic
import train_honest_xgb_seed_subset_gkf as seed_subset


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_client_pooling"
REPORT_PATH = WORK_DIR / "report.json"
OUTPUT_PATH = ROOT / "submission_honest_client_pooling.csv"
RAW_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "ProductCD",
    "card1",
    "addr1",
    "D1",
    "P_emaildomain",
)
METHODS = ("mean", "max", "q75")
WEIGHTS = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(roc_auc_score(target, prediction))


def token(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return values.round().astype("Int64").astype("string").fillna("NA")
    return values.astype("string").fillna("NA")


def build_uid_frame(raw: pd.DataFrame) -> pd.DataFrame:
    day = raw["TransactionDT"] / 86_400.0
    origin = day - raw["D1"]
    floor_origin = token(np.floor(origin))
    round_origin = token(np.round(origin))
    card = token(raw["card1"])
    address = token(raw["addr1"])
    email = token(raw["P_emaildomain"])
    product_code = token(raw["ProductCD"])
    card_addr = card.str.cat(address, sep="|")
    return pd.DataFrame(
        {
            "floor_card_addr": card_addr.str.cat(floor_origin, sep="|"),
            "floor_card_addr_email": card_addr.str.cat(
                floor_origin, sep="|"
            ).str.cat(email, sep="|"),
            "round_card_addr_email": card_addr.str.cat(
                round_origin, sep="|"
            ).str.cat(email, sep="|"),
            "floor_card_origin_email": card.str.cat(
                floor_origin, sep="|"
            ).str.cat(email, sep="|"),
            "floor_card_addr_product": card_addr.str.cat(
                floor_origin, sep="|"
            ).str.cat(product_code, sep="|"),
            "round_card_addr_product_email": card_addr.str.cat(
                round_origin, sep="|"
            ).str.cat(product_code, sep="|").str.cat(email, sep="|"),
        }
    )


def group_pool(
    prediction: np.ndarray,
    uid: pd.Series,
    method: str,
) -> np.ndarray:
    work = pd.DataFrame(
        {"uid": uid.to_numpy(), "prediction": np.asarray(prediction)}
    )
    grouped = work.groupby("uid", sort=False, dropna=False)["prediction"]
    if method == "q75":
        pooled = grouped.transform(lambda values: values.quantile(0.75))
    else:
        pooled = grouped.transform(method)
    return pooled.to_numpy(dtype="float64")


def pooling_signals(
    prediction: np.ndarray,
    uid_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    signals = {}
    by_method: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    for uid_name in uid_frame.columns:
        for method in METHODS:
            values = group_pool(prediction, uid_frame[uid_name], method)
            signals[f"{uid_name}_{method}"] = values
            by_method[method].append(values)
    for method, values in by_method.items():
        ranks = np.column_stack([client.rank(value) for value in values])
        signals[f"multi_{method}_meanrank"] = ranks.mean(axis=1)
        signals[f"multi_{method}_maxrank"] = ranks.max(axis=1)
    primary = "floor_card_addr"
    primary_ranks = np.column_stack(
        [
            client.rank(signals[f"{primary}_{method}"])
            for method in METHODS
        ]
    )
    signals["floor_card_addr_method_meanrank"] = primary_ranks.mean(axis=1)
    return signals


def direct_segment_blend(
    baseline: np.ndarray,
    signal: np.ndarray,
    labels: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    prediction = np.asarray(baseline, dtype="float64").copy()
    for name in segments.SEGMENTS:
        mask = labels == name
        weight = float(weights[name])
        prediction[mask] = (
            (1.0 - weight) * prediction[mask] + weight * signal[mask]
        )
    return prediction


def search_pooling(
    target: np.ndarray,
    baseline: np.ndarray,
    signals: dict[str, np.ndarray],
    labels: np.ndarray,
    day: np.ndarray,
) -> tuple[dict, list[dict]]:
    midpoint = np.median(day)
    halves = (day <= midpoint, day > midpoint)
    baseline_auc = auc(target, baseline)
    baseline_halves = [auc(target[mask], baseline[mask]) for mask in halves]
    masks = {name: labels == name for name in segments.SEGMENTS}
    rows = []
    row_map = {}

    for variant, signal in signals.items():
        transformed = {
            name: {
                weight: (
                    (1.0 - weight) * baseline[mask]
                    + weight * signal[mask]
                )
                for weight in WEIGHTS
            }
            for name, mask in masks.items()
        }

        def evaluate(weights: dict[str, float]) -> dict:
            key = (
                variant,
                *(float(weights[name]) for name in segments.SEGMENTS),
            )
            if key in row_map:
                return row_map[key]
            prediction = np.asarray(baseline, dtype="float64").copy()
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
            }
            rows.append(row)
            row_map[key] = row
            return row

        for start in (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.3),
            (0.2, 0.2, 0.2),
            (0.5, 0.5, 0.5),
        ):
            weights = dict(zip(segments.SEGMENTS, start))
            evaluate(weights)
            for _ in range(2):
                for name in segments.SEGMENTS:
                    local = []
                    for weight in WEIGHTS:
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
                center = WEIGHTS.index(float(weights[name]))
                neighborhoods.append(WEIGHTS[max(0, center - 1) : center + 2])
            for values in product(*neighborhoods):
                evaluate(dict(zip(segments.SEGMENTS, values)))

    stable = [row for row in rows if row["min_half_gain"] >= 0.0]
    selected = max(
        stable, key=lambda row: (row["gain"], row["min_half_gain"])
    )
    rows.sort(
        key=lambda row: (row["gain"], row["min_half_gain"]), reverse=True
    )
    return selected, rows


def main() -> None:
    started = time.time()
    WORK_DIR.mkdir(exist_ok=True)
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

    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_index = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_index = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
    dev_membership = membership_oof.loc[dev_mask].reset_index(drop=True)
    lock_membership = membership_oof.loc[lock_mask].reset_index(drop=True)
    dev_xgb = seed_subset.apply_xgb_layer(
        clean_oof[client.META_DEV_FOLD],
        np.load(ROOT / "honest_xgb_magic/dev_prediction.npy"),
        fold_groups[client.META_DEV_FOLD],
        clean_report["postprocess"],
        dev_membership,
        xgb_recipe,
    )
    lock_xgb = seed_subset.apply_xgb_layer(
        clean_oof[client.META_LOCK_FOLD],
        np.load(ROOT / "honest_xgb_magic/lock_prediction.npy"),
        fold_groups[client.META_LOCK_FOLD],
        clean_report["postprocess"],
        lock_membership,
        xgb_recipe,
    )
    dev_current = seed_subset.apply_heavy_layer(
        dev_xgb,
        np.load(ROOT / f"honest_magic_heavy_stack/dev_cat_{cat_name}.npy"),
        np.load(ROOT / f"honest_magic_heavy_stack/dev_lgb_{lgb_name}.npy"),
        fold_groups[client.META_DEV_FOLD],
        clean_report["postprocess"],
        dev_membership,
        heavy_recipe,
    )
    lock_current = seed_subset.apply_heavy_layer(
        lock_xgb,
        np.load(ROOT / "honest_magic_heavy_stack/lock_cat.npy"),
        np.load(ROOT / "honest_magic_heavy_stack/lock_lgb.npy"),
        fold_groups[client.META_LOCK_FOLD],
        clean_report["postprocess"],
        lock_membership,
        heavy_recipe,
    )
    y_dev = np.asarray(arrays["target"][dev_index], dtype="int8")
    y_lock = np.asarray(arrays["target"][lock_index], dtype="int8")
    current_dev_auc = auc(y_dev, dev_current)
    current_lock_auc = auc(y_lock, lock_current)
    if abs(current_dev_auc - float(heavy_report["dev_auc"])) > 1e-12:
        raise RuntimeError("Could not reproduce heavy-stack dev prediction")
    if abs(current_lock_auc - float(heavy_report["lock_auc"])) > 1e-12:
        raise RuntimeError("Could not reproduce heavy-stack lock prediction")

    raw_train = pd.read_csv(ROOT / "train_transaction.csv", usecols=RAW_COLUMNS)
    raw_test = pd.read_csv(ROOT / "test_transaction.csv", usecols=RAW_COLUMNS)
    if not np.array_equal(
        raw_train["TransactionID"].to_numpy(), arrays["train_id"]
    ):
        raise RuntimeError("Raw train rows differ from the magic matrix")
    if not np.array_equal(
        raw_test["TransactionID"].to_numpy(), arrays["test_id"]
    ):
        raise RuntimeError("Raw test rows differ from the magic matrix")
    train_uid = build_uid_frame(raw_train)
    test_uid = build_uid_frame(raw_test)
    dev_uid = train_uid.iloc[dev_index].reset_index(drop=True)
    lock_uid = train_uid.iloc[lock_index].reset_index(drop=True)
    dev_signals = pooling_signals(dev_current, dev_uid)
    selected, search_rows = search_pooling(
        y_dev,
        dev_current,
        dev_signals,
        segments.segment_labels(dev_membership),
        np.asarray(arrays["day"][dev_index]),
    )
    dev_candidate = direct_segment_blend(
        dev_current,
        dev_signals[selected["variant"]],
        segments.segment_labels(dev_membership),
        selected["weights"],
    )
    lock_signals = pooling_signals(lock_current, lock_uid)
    lock_candidate = direct_segment_blend(
        lock_current,
        lock_signals[selected["variant"]],
        segments.segment_labels(lock_membership),
        selected["weights"],
    )
    dev_auc = auc(y_dev, dev_candidate)
    lock_auc = auc(y_lock, lock_candidate)
    accepted = bool(
        selected["gain"] > 0.0
        and dev_auc > current_dev_auc
        and lock_auc > current_lock_auc
    )

    if accepted:
        baseline = pd.read_csv(ROOT / "submission_honest_magic_heavy_stack.csv")
        if not np.array_equal(
            baseline["TransactionID"].to_numpy(), arrays["test_id"]
        ):
            raise RuntimeError("Pooling test rows differ from heavy stack")
        test_current = baseline[client.TARGET].to_numpy(dtype="float64")
        test_signals = pooling_signals(test_current, test_uid)
        prediction = direct_segment_blend(
            test_current,
            test_signals[selected["variant"]],
            segments.segment_labels(membership_test),
            selected["weights"],
        )
        output = baseline[["TransactionID"]].copy()
        output[client.TARGET] = prediction
        output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": (
            "official train/test covariates and train labels only; "
            "pooling uses query covariates/predictions without labels"
        ),
        "official_hashes": manifest["official_hashes"],
        "uid_definitions": list(dev_uid.columns),
        "methods": list(METHODS),
        "selection": "pooling recipe on dev; one-time unchanged lock",
        "champion_auc": {"dev": current_dev_auc, "lock": current_lock_auc},
        "selected": selected,
        "search_top": search_rows[:50],
        "dev_auc": dev_auc,
        "dev_gain": dev_auc - current_dev_auc,
        "lock_auc": lock_auc,
        "lock_gain": lock_auc - current_lock_auc,
        "accepted": accepted,
        "output": OUTPUT_PATH.name if accepted else None,
        "output_sha256": file_sha256(OUTPUT_PATH) if accepted else None,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

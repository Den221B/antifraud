"""Add the published XGB-magic source above the locked clean-v2 blend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import refine_honest_client_segments as segments
import search_honest_cleanv2_blend as clean_blend
import search_honest_featureview_meta as featureview
import search_honest_fullrow_lgb as fullrow
import train_honest_client_meta as client
import train_honest_xgb_magic as magic


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_xgb_magic"
REPORT_PATH = WORK_DIR / "blend_report.json"
OUTPUT_PATH = ROOT / "submission_honest_xgb_magic_blend.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_clean_oof(
    oof: pd.DataFrame,
    membership: pd.DataFrame,
    reference: dict[int, np.ndarray],
    fold_groups: dict,
    reference_report: dict,
) -> tuple[dict[int, np.ndarray], dict]:
    clean_recipe = json.loads(
        (ROOT / "clean_v2/recipe.json").read_text(encoding="utf-8")
    )
    blend_report = json.loads(
        (ROOT / "honest_cleanv2_blend/report.json").read_text(encoding="utf-8")
    )
    selected = blend_report["selected"]
    fullrow_report = json.loads(
        (ROOT / "honest_fullrow_lgb/report.json").read_text(encoding="utf-8")
    )
    postprocess = reference_report["recipe"]["postprocess_locked_from_dev"]
    specs = {
        client.META_DEV_FOLD: (
            ROOT / "clean_v2/predictions/dev_h30_c.csv",
            ROOT / "honest_fullrow_lgb/dev_raw.npy",
        ),
        client.META_LOCK_FOLD: (
            ROOT / "clean_v2/predictions/lock_h30.csv",
            ROOT / "honest_fullrow_lgb/lock_raw.npy",
        ),
    }
    predictions = {}
    metrics = {}
    for fold, (clean_path, lgb_path) in specs.items():
        fold_mask = oof["fold"].eq(fold).to_numpy()
        frame, clean_prediction = clean_blend.load_clean_prediction(
            clean_path, clean_recipe
        )
        expected_rows = oof.loc[fold_mask, "row_index"].to_numpy(dtype="int64")
        if not np.array_equal(frame["row_index"].to_numpy(), expected_rows):
            raise RuntimeError(f"clean_v2 rows differ on fold {fold}")
        lgb_prediction = fullrow.transform_variant(
            np.load(lgb_path),
            fullrow_report["selected_blend"]["variant"],
            fold_groups[fold],
            postprocess,
        )
        values = clean_blend.transform_sources(
            {
                "featureview": reference[fold],
                "clean_v2": clean_prediction,
                "fullrow_lgb": lgb_prediction,
            },
            selected["mode"],
        )
        prediction = clean_blend.apply_segment_recipe(
            values,
            segments.segment_labels(
                membership.loc[fold_mask].reset_index(drop=True)
            ),
            selected["weights"],
        )
        y = oof.loc[fold_mask, client.TARGET].to_numpy(dtype="int8")
        predictions[fold] = prediction
        metrics[fold] = float(roc_auc_score(y, prediction))
    expected = float(blend_report["candidate_lock_auc"])
    if abs(metrics[client.META_LOCK_FOLD] - expected) > 1e-12:
        raise RuntimeError("The locked clean-v2 blend was not reproduced")
    return predictions, {
        "recipe": selected,
        "metrics": metrics,
        "fullrow_report": fullrow_report,
        "postprocess": postprocess,
    }


def main() -> None:
    started = time.time()
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
    clean_oof, clean_report = reconstruct_clean_oof(
        oof, membership_oof, reference, fold_groups, reference_report
    )

    dev_mask = oof["fold"].eq(client.META_DEV_FOLD).to_numpy()
    lock_mask = oof["fold"].eq(client.META_LOCK_FOLD).to_numpy()
    dev_index = oof.loc[dev_mask, "row_index"].to_numpy(dtype="int64")
    lock_index = oof.loc[lock_mask, "row_index"].to_numpy(dtype="int64")
    dev_magic = np.load(WORK_DIR / "dev_prediction.npy")
    lock_magic = np.load(WORK_DIR / "lock_prediction.npy")
    dev_variants = fullrow.prediction_variants(
        dev_magic,
        fold_groups[client.META_DEV_FOLD],
        clean_report["postprocess"],
    )
    selected, search_rows = fullrow.search_blend(
        np.asarray(arrays["target"][dev_index], dtype="int8"),
        clean_oof[client.META_DEV_FOLD],
        dev_variants,
        segments.segment_labels(
            membership_oof.loc[dev_mask].reset_index(drop=True)
        ),
        np.asarray(arrays["day"][dev_index]),
    )
    lock_variant = fullrow.transform_variant(
        lock_magic,
        selected["variant"],
        fold_groups[client.META_LOCK_FOLD],
        clean_report["postprocess"],
    )
    lock_prediction = segments.segmented_blend(
        clean_oof[client.META_LOCK_FOLD],
        lock_variant,
        segments.segment_labels(
            membership_oof.loc[lock_mask].reset_index(drop=True)
        ),
        selected["weights"],
    )
    y_lock = np.asarray(arrays["target"][lock_index], dtype="int8")
    lock_auc = float(roc_auc_score(y_lock, lock_prediction))
    previous_lock = clean_report["metrics"][client.META_LOCK_FOLD]
    accepted = bool(selected["gain"] > 0.0 and lock_auc > previous_lock)

    group_models = []
    if accepted:
        source_report = json.loads(
            (WORK_DIR / "report.json").read_text(encoding="utf-8")
        )
        final_iterations = max(
            50, int(np.ceil(source_report["lock_iterations"] * 1.15))
        )
        test_magic, group_models = magic.train_group_models(
            arrays["train"],
            arrays["test"],
            arrays["target"],
            arrays["month"],
            fixed_iterations=final_iterations,
        )
        np.save(WORK_DIR / "test_prediction.npy", test_magic.astype("float32"))
        test_variant = fullrow.transform_variant(
            test_magic,
            selected["variant"],
            reference_report["test_groups"],
            clean_report["postprocess"],
        )
        baseline = pd.read_csv(ROOT / "submission_honest_cleanv2_blend.csv")
        if not np.array_equal(
            baseline["TransactionID"].to_numpy(), arrays["test_id"]
        ):
            raise RuntimeError("Magic test rows differ from the clean baseline")
        prediction = segments.segmented_blend(
            baseline[client.TARGET].to_numpy(dtype="float64"),
            test_variant,
            segments.segment_labels(membership_test),
            selected["weights"],
        )
        output = baseline[["TransactionID"]].copy()
        output[client.TARGET] = prediction
        output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": "official train/test covariates; official train labels only",
        "feature_recipe": manifest["source"],
        "selection": "blend on days 75-90; days 90-105 one-time lock",
        "clean_baseline": clean_report["metrics"],
        "selected": selected,
        "search_top": search_rows[:30],
        "previous_lock_auc": previous_lock,
        "candidate_lock_auc": lock_auc,
        "lock_gain": lock_auc - previous_lock,
        "accepted": accepted,
        "final_group_iterations": (
            final_iterations if accepted else None
        ),
        "group_models": group_models,
        "output": OUTPUT_PATH.name if accepted else None,
        "output_sha256": file_sha256(OUTPUT_PATH) if accepted else None,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Finalize the train-only selected feature-view client stack."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import refine_honest_client_segments as segments
import honest_featureview_sources as next_features
import search_honest_featureview_meta as search
import train_honest_client_meta as client
import train_honest_heavy_temporal_client as heavy


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "honest_featureview_meta"
SOURCE_CACHE = WORK_DIR / "test_featureview_sources.csv"
OUTPUT_PATH = ROOT / "submission_honest_featureview_client.csv"
FINAL_REPORT_PATH = WORK_DIR / "final_report.json"

MODEL_PATHS = {
    "fv_vblock_dynamics": ROOT
    / "advanced_feature_ablation_models/vblock_dynamics_final.txt",
    "fv_vblock_cd_dynamics": ROOT
    / "advanced_feature_ablation_models/vblock_cd_dynamics_final.txt",
    "fv_vblock_structured": ROOT
    / "next_feature_ablation_models_v1/vblock_structured_final.txt",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_clean_sources() -> pd.DataFrame:
    sample = pd.read_csv(ROOT / "sample_submission.csv", usecols=["TransactionID"])
    if SOURCE_CACHE.exists():
        cached = pd.read_csv(SOURCE_CACHE)
        required = {"TransactionID", *MODEL_PATHS}
        if required.issubset(cached.columns) and np.array_equal(
            cached["TransactionID"].to_numpy(), sample["TransactionID"].to_numpy()
        ):
            print("Loading cached clean feature-view predictions", flush=True)
            return cached

    print("Preparing official train/test feature views...", flush=True)
    prepared, _, _ = next_features.prepare_next_views()
    if client.TARGET in prepared["inference"]:
        raise RuntimeError("Target reached feature-view inference frame")
    if not np.array_equal(
        prepared["inference_ids"].to_numpy(), sample["TransactionID"].to_numpy()
    ):
        raise RuntimeError("Prepared test rows differ from sample_submission")

    output = sample.copy()
    for source, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = lgb.Booster(model_file=str(model_path))
        features = model.feature_name()
        missing = sorted(set(features).difference(prepared["inference"].columns))
        if missing:
            raise RuntimeError(f"Missing {source} features: {missing[:10]}")
        print(f"Predicting {source}: {len(features)} features", flush=True)
        output[source] = model.predict(prepared["inference"][features])
        del model
        gc.collect()
    output.to_csv(SOURCE_CACHE, index=False)
    del prepared
    gc.collect()
    return output


def main() -> None:
    started = time.time()
    recipe = json.loads(
        (WORK_DIR / "search_report.json").read_text(encoding="utf-8")
    )
    if not recipe["accepted"]:
        raise RuntimeError("Feature-view candidate did not pass temporal lock")
    selected = recipe["selected_model"]
    selected_sources, _ = search.VIEWS[selected["view"]]
    needed_feature_sources = [
        source for source in selected_sources if source.startswith("fv_")
    ]
    if set(needed_feature_sources).difference(MODEL_PATHS):
        raise RuntimeError("Selected source has no clean final model")

    clean_sources = predict_clean_sources()
    oof = search.build_oof()
    test_predictions = client.build_test_sources()
    for source in needed_feature_sources:
        test_predictions[source] = clean_sources[source].to_numpy(dtype="float64")
    membership_oof = pd.read_csv(ROOT / "honest_client_meta/oof_client_features.csv")
    membership_test = pd.read_csv(ROOT / "honest_client_meta/test_client_features.csv")

    original_views = heavy.VIEWS
    heavy.VIEWS = search.VIEWS
    try:
        train_features = heavy.make_view(
            oof, membership_oof, selected["view"], fold=oof["fold"]
        )
        test_features = heavy.make_view(
            test_predictions, membership_test, selected["view"], fold=None
        )
    finally:
        heavy.VIEWS = original_views
    final_iterations = max(
        30, int(np.ceil(float(recipe["lock_iterations"]) * 1.15))
    )
    model = lgb.LGBMClassifier(
        **{
            **heavy.LGB_PARAMS,
            **heavy.LGB_CONFIGS[int(selected["config_index"])],
            "n_estimators": final_iterations,
        }
    )
    model.fit(
        train_features,
        oof[client.TARGET],
        callbacks=[lgb.log_evaluation(0)],
    )
    model_path = WORK_DIR / "final_lgb.txt"
    model.booster_.save_model(model_path)
    feature_prediction = model.predict_proba(test_features)[:, 1]

    raw_train = client.prepare_raw(
        pd.read_csv(ROOT / "train_transaction.csv", usecols=list(client.RAW_COLUMNS))
    )
    raw_test = client.prepare_raw(
        pd.read_csv(ROOT / "test_transaction.csv", usecols=list(client.RAW_COLUMNS))
    )
    _, _, _, test_groups = client.build_client_features(
        oof, raw_train, raw_test
    )
    current = pd.read_csv(ROOT / "submission_honest_user_means.csv")
    if not np.array_equal(
        current["TransactionID"].to_numpy(), clean_sources["TransactionID"].to_numpy()
    ):
        raise RuntimeError("Current baseline and feature sources are misaligned")
    prediction = segments.segmented_blend(
        current[client.TARGET].to_numpy(dtype="float64"),
        feature_prediction,
        segments.segment_labels(membership_test),
        recipe["selected_segment_weights"]["weights"],
    )
    prediction = client.apply_postprocess(
        prediction, test_groups, recipe["postprocess_locked_from_dev"]
    )
    output = current[["TransactionID"]].copy()
    output[client.TARGET] = prediction
    output.to_csv(OUTPUT_PATH, index=False)

    report = {
        "data_policy": "official train/test only; no gap/test/audit labels",
        "selection_report": str(WORK_DIR / "search_report.json"),
        "selected_sources": list(selected_sources),
        "clean_feature_sources": needed_feature_sources,
        "source_models": {
            source: {"path": str(path), "sha256": file_sha256(path)}
            for source, path in MODEL_PATHS.items()
            if source in needed_feature_sources
        },
        "meta_model": {
            "path": str(model_path),
            "iterations": final_iterations,
            "features": int(train_features.shape[1]),
            "seed": heavy.LGB_PARAMS["random_state"],
        },
        "output": OUTPUT_PATH.name,
        "output_sha256": file_sha256(OUTPUT_PATH),
        "rows": int(len(output)),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    FINAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

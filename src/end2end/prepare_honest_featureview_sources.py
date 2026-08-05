"""Build the train-only OOF and target-free test feature-view sources.

This is the executable subset of the earlier feature-ablation experiments.
It trains every OOF view required by the final stack, then fits only the three
full-data LightGBM sources selected by temporal validation.  No submission,
external bridge, or test-label audit path from the exploratory scripts is run.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import honest_featureview_sources as sources


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "honest_featureview_source_report.json"

ADVANCED_FINAL_VIEWS = ("vblock_dynamics", "vblock_cd_dynamics")
NEXT_FINAL_VIEWS = ("vblock_structured",)
REQUIRED_ADVANCED_OOF = (
    "vblock_dynamics",
    "vblock_cd_dynamics",
    "vblock_aggregates_dynamics",
)
REQUIRED_NEXT_OOF = (
    "vblock_d_amount",
    "vblock_structured",
    "vblock_chains",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.time()

    print("Building official-data advanced feature views...", flush=True)
    prepared, views, advanced_metadata = sources.prepare_advanced_views()
    missing = sorted(set(REQUIRED_ADVANCED_OOF).difference(views))
    if missing:
        raise RuntimeError(f"Missing advanced views: {missing}")
    advanced_oof, advanced_metrics = sources.train_oof(
        prepared,
        views,
        sources.ADVANCED_CACHE_DIR,
        sources.ADVANCED_OOF_PATH,
        seed=8203,
        force=args.force,
    )
    advanced_models = sources.train_final(
        prepared,
        views,
        ADVANCED_FINAL_VIEWS,
        sources.ADVANCED_CACHE_DIR,
        seed=9203,
        force=args.force,
    )
    del prepared
    gc.collect()

    print("Building official-data structured feature views...", flush=True)
    prepared, specs, next_metadata = sources.prepare_next_views()
    missing = sorted(set(REQUIRED_NEXT_OOF).difference(specs))
    if missing:
        raise RuntimeError(f"Missing structured views: {missing}")

    next_oof, next_metrics = sources.train_oof(
        prepared,
        specs,
        sources.NEXT_CACHE_DIR,
        sources.NEXT_OOF_PATH,
        seed=12303,
        force=args.force,
    )
    next_models = sources.train_final(
        prepared,
        specs,
        NEXT_FINAL_VIEWS,
        sources.NEXT_CACHE_DIR,
        seed=13303,
        force=args.force,
    )
    del prepared
    gc.collect()

    required_models = (
        sources.ADVANCED_CACHE_DIR / "vblock_dynamics_final.txt",
        sources.ADVANCED_CACHE_DIR / "vblock_cd_dynamics_final.txt",
        sources.NEXT_CACHE_DIR / "vblock_structured_final.txt",
    )
    missing_models = [str(path) for path in required_models if not path.exists()]
    if missing_models:
        raise RuntimeError(f"Missing final feature-view models: {missing_models}")

    report = {
        "data_policy": (
            "official train/test covariates and official train labels only; "
            "no bridge, external labels, previous submission, or audit"
        ),
        "selection": "frozen from purged forward-time train folds",
        "advanced_oof_rows": int(len(advanced_oof)),
        "next_oof_rows": int(len(next_oof)),
        "required_advanced_oof": list(REQUIRED_ADVANCED_OOF),
        "required_next_oof": list(REQUIRED_NEXT_OOF),
        "advanced_final_views": list(ADVANCED_FINAL_VIEWS),
        "next_final_views": list(NEXT_FINAL_VIEWS),
        "advanced_metadata": advanced_metadata,
        "next_metadata": next_metadata,
        "advanced_metrics": advanced_metrics,
        "next_metrics": next_metrics,
        "advanced_models": advanced_models,
        "next_models": next_models,
        "models": [str(path.relative_to(ROOT)) for path in required_models],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd


V_BLOCKS = (
    (1, 11),
    (12, 34),
    (35, 52),
    (53, 74),
    (75, 94),
    (95, 137),
    (138, 166),
    (167, 216),
    (217, 278),
    (279, 321),
    (322, 339),
)

TOP_V_COLUMNS = (
    "V242",
    "V243",
    "V258",
    "V265",
    "V199",
    "V201",
    "V257",
    "V274",
    "V86",
    "V156",
    "V44",
    "V87",
    "V62",
    "V217",
    "V91",
    "V67",
    "V275",
    "V189",
    "V212",
    "V70",
    "V61",
    "V69",
    "V45",
    "V149",
    "V256",
    "V90",
    "V266",
    "V94",
    "V308",
    "V200",
)


def _row_vblock_features(frame: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for start, end in V_BLOCKS:
        columns = [
            f"V{number}"
            for number in range(start, end + 1)
            if f"V{number}" in frame
        ]
        if not columns:
            continue
        values = frame[columns]
        prefix = f"vblock_{start}_{end}"
        features[f"{prefix}_missing_rate"] = values.isna().mean(axis=1).astype(
            "float32"
        )
        features[f"{prefix}_mean"] = values.mean(axis=1).astype("float32")
        features[f"{prefix}_std"] = values.std(axis=1).astype("float32")
        block_min = values.min(axis=1)
        block_max = values.max(axis=1)
        features[f"{prefix}_min"] = block_min.astype("float32")
        features[f"{prefix}_max"] = block_max.astype("float32")
        features[f"{prefix}_range"] = (block_max - block_min).astype("float32")
        features[f"{prefix}_nunique"] = values.nunique(axis=1).astype("int16")
        features[f"{prefix}_zero_count"] = values.eq(0).sum(axis=1).astype(
            "int16"
        )
        features[f"{prefix}_one_count"] = values.eq(1).sum(axis=1).astype(
            "int16"
        )
        features[f"{prefix}_two_count"] = values.eq(2).sum(axis=1).astype(
            "int16"
        )
    return pd.DataFrame(features, index=frame.index)


def add_vblock_user_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    train_components: np.ndarray,
    inference_components: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    print("Building row-level V-block encodings...", flush=True)
    train_blocks = _row_vblock_features(train).reset_index(drop=True)
    inference_blocks = _row_vblock_features(inference).reset_index(drop=True)
    block_features = train_blocks.columns.tolist()

    c_columns = [f"C{i}" for i in range(1, 15) if f"C{i}" in train]
    d_columns = [f"D{i}" for i in range(1, 16) if f"D{i}" in train]
    v_columns = [column for column in TOP_V_COLUMNS if column in train]
    aggregate_columns = [*c_columns, *d_columns, *v_columns]
    combined = pd.concat(
        [train[aggregate_columns], inference[aggregate_columns]],
        ignore_index=True,
        copy=False,
    )
    combined["_component"] = np.concatenate(
        [train_components, inference_components]
    )
    group = combined.groupby("_component", sort=False, observed=True)
    component_count = group["_component"].transform("size")
    aggregate_features: dict[str, pd.Series] = {}

    print("Building broad user C/D aggregates...", flush=True)
    for column in [*c_columns, *d_columns]:
        values = combined[column]
        column_group = group[column]
        mean = column_group.transform("mean")
        std = column_group.transform("std")
        minimum = column_group.transform("min")
        maximum = column_group.transform("max")
        nunique = column_group.transform("nunique")
        prefix = f"wide_user_{column}"
        aggregate_features[f"{prefix}_mean"] = mean.astype("float32")
        aggregate_features[f"{prefix}_std"] = std.astype("float32")
        aggregate_features[f"{prefix}_range"] = (maximum - minimum).astype(
            "float32"
        )
        aggregate_features[f"{prefix}_nunique_rate"] = (
            nunique / component_count
        ).astype("float32")
        aggregate_features[f"{prefix}_diff_mean"] = (values - mean).astype(
            "float32"
        )
        aggregate_features[f"{prefix}_zscore"] = (
            (values - mean) / std.replace(0, np.nan)
        ).astype("float32")

    print("Building selected user V aggregates...", flush=True)
    for column in v_columns:
        values = combined[column]
        column_group = group[column]
        mean = column_group.transform("mean")
        std = column_group.transform("std")
        prefix = f"wide_user_{column}"
        aggregate_features[f"{prefix}_mean"] = mean.astype("float32")
        aggregate_features[f"{prefix}_std"] = std.astype("float32")
        aggregate_features[f"{prefix}_zscore"] = (
            (values - mean) / std.replace(0, np.nan)
        ).astype("float32")

    aggregate_frame = pd.DataFrame(aggregate_features)
    train_rows = len(train)
    train_aggregate = aggregate_frame.iloc[:train_rows].reset_index(drop=True)
    inference_aggregate = aggregate_frame.iloc[train_rows:].reset_index(
        drop=True
    )

    print("Building user-level V-block missingness profiles...", flush=True)
    combined_blocks = pd.concat(
        [train_blocks, inference_blocks], ignore_index=True, copy=False
    )
    combined_blocks["_component"] = np.concatenate(
        [train_components, inference_components]
    )
    missing_columns = [
        column for column in block_features if column.endswith("_missing_rate")
    ]
    missing_features: dict[str, pd.Series] = {}
    missing_group = combined_blocks.groupby(
        "_component", sort=False, observed=True
    )
    for column in missing_columns:
        mean = missing_group[column].transform("mean")
        missing_features[f"wide_user_{column}_mean"] = mean.astype("float32")
        missing_features[f"wide_user_{column}_diff"] = (
            combined_blocks[column] - mean
        ).astype("float32")
    missing_frame = pd.DataFrame(missing_features)
    train_missing = missing_frame.iloc[:train_rows].reset_index(drop=True)
    inference_missing = missing_frame.iloc[train_rows:].reset_index(drop=True)

    train_new = pd.concat(
        [train_blocks, train_aggregate, train_missing], axis=1
    )
    inference_new = pd.concat(
        [inference_blocks, inference_aggregate, inference_missing], axis=1
    )
    feature_names = train_new.columns.tolist()
    stats = {
        "v_blocks": len(V_BLOCKS),
        "block_features": len(block_features),
        "c_columns": len(c_columns),
        "d_columns": len(d_columns),
        "selected_v_columns": list(v_columns),
        "aggregate_features": len(train_aggregate.columns),
        "missing_profile_features": len(train_missing.columns),
        "total_features": len(feature_names),
    }
    return (
        pd.concat([train.reset_index(drop=True), train_new], axis=1),
        pd.concat([inference.reset_index(drop=True), inference_new], axis=1),
        feature_names,
        stats,
    )

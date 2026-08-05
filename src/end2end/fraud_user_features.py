from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RAW_UID_COLUMNS = {
    "uid_card_addr",
    "uid_card_email",
    "uid_card_full",
    "uid_card_device",
    "uid_email_pair",
    "uid_d1_email",
    "uid_card_addr_d1",
    "uid_card_addr_d1_email",
}


@dataclass(frozen=True)
class GraphKey:
    name: str
    columns: tuple[str, ...]
    max_group_size: int


GRAPH_KEYS = (
    GraphKey(
        "card_addr_origin",
        ("card1", "addr1", "D1_origin_day"),
        200,
    ),
    GraphKey(
        "card_origin_email",
        ("card1", "D1_origin_day", "P_emaildomain"),
        200,
    ),
    GraphKey(
        "card_full_origin",
        ("card1", "card2", "card3", "card5", "D1_origin_day"),
        150,
    ),
    GraphKey(
        "card_addr_origin_device",
        ("card1", "addr1", "D1_origin_day", "DeviceInfo"),
        100,
    ),
)

PROFILE_NUMERIC_COLUMNS = (
    "dist1",
    "dist2",
    "D1",
    "D3",
    "C1",
    "C2",
    "C13",
    "C14",
    "V279",
    "V280",
    "V306",
    "V307",
    "V308",
)

PROFILE_ID_COLUMNS = (
    "card1",
    "card2",
    "card5",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "ProductCD",
)

CAUSAL_HISTORY_COLUMNS = (
    "user_known_txn_count",
    "user_known_fraud_count",
    "user_known_nonfraud_count",
    "user_known_fraud_rate",
    "user_known_fraud_rate_smoothed",
    "user_known_any_fraud",
    "user_seconds_since_known_fraud",
)


def is_raw_uid_feature(column: str) -> bool:
    """Return True only for direct identifiers, not their numeric aggregates."""
    return column in RAW_UID_COLUMNS


def _valid_key_value(series: pd.Series) -> pd.Series:
    valid = series.notna()
    if isinstance(series.dtype, pd.StringDtype) or pd.api.types.is_object_dtype(
        series.dtype
    ):
        valid &= series.astype("string").ne("<MISSING>")
    return valid


def _normalized_key_frame(frame: pd.DataFrame, key: GraphKey) -> pd.DataFrame:
    values = frame.loc[:, key.columns].copy()
    if "D1_origin_day" in values:
        values["D1_origin_day"] = values["D1_origin_day"].round()
    return values


def _find_root(parent: np.ndarray, node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = int(parent[node])
    return node


def _union_nodes(
    parent: np.ndarray,
    component_size: np.ndarray,
    first: int,
    second: int,
    max_component_size: int,
) -> bool:
    first_root = _find_root(parent, first)
    second_root = _find_root(parent, second)
    if first_root == second_root:
        return False
    if (
        int(component_size[first_root]) + int(component_size[second_root])
        > max_component_size
    ):
        return False
    if component_size[first_root] < component_size[second_root]:
        first_root, second_root = second_root, first_root
    parent[second_root] = first_root
    component_size[first_root] += component_size[second_root]
    return True


def build_user_components(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    max_component_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Join strict UID variants into graph components without exposing their IDs."""
    required = {column for key in GRAPH_KEYS for column in key.columns}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Cannot build user graph; missing columns: {missing}")

    combined = pd.concat(
        [train[list(required)], inference[list(required)]],
        ignore_index=True,
        copy=False,
    )
    row_count = len(combined)
    parent = np.arange(row_count, dtype="int32")
    component_size = np.ones(row_count, dtype="int32")
    key_support = np.zeros(row_count, dtype="int8")
    key_metrics = []

    for key in GRAPH_KEYS:
        key_values = _normalized_key_frame(combined, key)
        valid = np.ones(row_count, dtype=bool)
        for column in key.columns:
            valid &= _valid_key_value(key_values[column]).to_numpy()
        valid_rows = np.flatnonzero(valid).astype("int32", copy=False)
        key_support[valid_rows] += 1

        if len(valid_rows) == 0:
            key_metrics.append(
                {
                    "key": key.name,
                    "valid_rows": 0,
                    "linked_groups": 0,
                    "oversized_groups": 0,
                    "union_edges": 0,
                }
            )
            continue

        hashes = pd.util.hash_pandas_object(
            key_values.loc[valid, list(key.columns)],
            index=False,
            categorize=True,
        ).to_numpy(dtype="uint64", copy=False)
        order = np.argsort(hashes, kind="stable")
        sorted_hashes = hashes[order]
        boundaries = np.flatnonzero(
            np.r_[True, sorted_hashes[1:] != sorted_hashes[:-1], True]
        )

        linked_groups = 0
        oversized_groups = 0
        union_edges = 0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            group_size = int(end - start)
            if group_size < 2:
                continue
            if group_size > key.max_group_size:
                oversized_groups += 1
                continue
            members = valid_rows[order[start:end]]
            anchor = int(members[0])
            linked_groups += 1
            for member in members[1:]:
                union_edges += int(
                    _union_nodes(
                        parent,
                        component_size,
                        anchor,
                        int(member),
                        max_component_size,
                    )
                )

        key_metrics.append(
            {
                "key": key.name,
                "valid_rows": int(len(valid_rows)),
                "linked_groups": linked_groups,
                "oversized_groups": oversized_groups,
                "union_edges": union_edges,
            }
        )

    roots = np.empty(row_count, dtype="int32")
    for row in range(row_count):
        roots[row] = _find_root(parent, row)
    components, _ = pd.factorize(roots, sort=False)
    components = components.astype("int32", copy=False)
    component_counts = np.bincount(components)
    train_rows = len(train)
    stats = {
        "rows": row_count,
        "components": int(len(component_counts)),
        "singleton_components": int(np.sum(component_counts == 1)),
        "multirow_components": int(np.sum(component_counts > 1)),
        "rows_in_multirow_components": int(
            component_counts[component_counts > 1].sum()
        ),
        "max_component_size": int(component_counts.max()),
        "keys": key_metrics,
    }
    return (
        components[:train_rows],
        components[train_rows:],
        key_support,
        stats,
    )


def _profile_aggregates(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    train_components: np.ndarray,
    inference_components: np.ndarray,
    key_support: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    source_columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        *[column for column in PROFILE_NUMERIC_COLUMNS if column in train],
        *[column for column in PROFILE_ID_COLUMNS if column in train],
    ]
    source_columns = list(dict.fromkeys(source_columns))
    sequence = pd.concat(
        [train[source_columns], inference[source_columns]],
        ignore_index=True,
        copy=False,
    )
    sequence["_component"] = np.concatenate(
        [train_components, inference_components]
    )
    sequence["_row_order"] = np.arange(len(sequence), dtype="int32")
    sequence = sequence.sort_values(
        ["TransactionDT", "TransactionID"],
        kind="stable",
    ).reset_index(drop=True)
    group = sequence.groupby("_component", sort=False, observed=True)

    feature_values: dict[str, pd.Series | np.ndarray] = {}
    count = group["TransactionID"].transform("size").astype("int32")
    order = group.cumcount().astype("int32")
    feature_values["user_txn_count"] = count
    feature_values["user_order"] = order
    feature_values["user_order_from_end"] = (count - order - 1).astype("int32")
    feature_values["user_order_fraction"] = (
        order / (count - 1).replace(0, np.nan)
    ).astype("float32")

    previous_dt = group["TransactionDT"].diff()
    next_dt = group["TransactionDT"].shift(-1) - sequence["TransactionDT"]
    first_dt = group["TransactionDT"].transform("min")
    last_dt = group["TransactionDT"].transform("max")
    feature_values["user_previous_dt"] = previous_dt.astype("float32")
    feature_values["user_next_dt"] = next_dt.astype("float32")
    feature_values["user_time_span"] = (last_dt - first_dt).astype("float32")
    feature_values["user_time_since_first"] = (
        sequence["TransactionDT"] - first_dt
    ).astype("float32")
    feature_values["user_time_to_last"] = (
        last_dt - sequence["TransactionDT"]
    ).astype("float32")

    sequence["_previous_dt"] = previous_dt
    gap_group = sequence.groupby(
        "_component", sort=False, observed=True
    )["_previous_dt"]
    feature_values["user_mean_dt"] = gap_group.transform("mean").astype(
        "float32"
    )
    feature_values["user_std_dt"] = gap_group.transform("std").astype(
        "float32"
    )
    feature_values["user_median_dt"] = gap_group.transform("median").astype(
        "float32"
    )

    amount = sequence["TransactionAmt"]
    amount_mean = group["TransactionAmt"].transform("mean")
    amount_std = group["TransactionAmt"].transform("std")
    feature_values["user_amount_mean"] = amount_mean.astype("float32")
    feature_values["user_amount_std"] = amount_std.astype("float32")
    feature_values["user_amount_median"] = group["TransactionAmt"].transform(
        "median"
    ).astype("float32")
    feature_values["user_amount_min"] = group["TransactionAmt"].transform(
        "min"
    ).astype("float32")
    feature_values["user_amount_max"] = group["TransactionAmt"].transform(
        "max"
    ).astype("float32")
    feature_values["user_previous_amount"] = group["TransactionAmt"].shift(
        1
    ).astype("float32")
    feature_values["user_next_amount"] = group["TransactionAmt"].shift(-1).astype(
        "float32"
    )
    feature_values["user_amount_to_mean"] = (
        amount / amount_mean.replace(0, np.nan)
    ).astype("float32")
    feature_values["user_amount_zscore"] = (
        (amount - amount_mean) / amount_std.replace(0, np.nan)
    ).astype("float32")

    amount_pair = sequence.groupby(
        ["_component", "TransactionAmt"],
        sort=False,
        observed=True,
        dropna=False,
    )["TransactionID"].transform("size")
    feature_values["user_same_amount_count"] = amount_pair.astype("int32")
    feature_values["user_same_amount_fraction"] = (amount_pair / count).astype(
        "float32"
    )

    day = (sequence["TransactionDT"] // 86400).astype("int32")
    day_frame = pd.DataFrame(
        {"component": sequence["_component"], "day": day}
    )
    feature_values["user_active_day_count"] = day_frame.groupby(
        "component", sort=False, observed=True
    )["day"].transform("nunique").astype("int16")

    for column in PROFILE_ID_COLUMNS:
        if column not in sequence:
            continue
        value_frame = pd.DataFrame(
            {
                "component": sequence["_component"],
                "value": sequence[column],
            }
        )
        unique_count = value_frame.groupby(
            "component", sort=False, observed=True
        )["value"].transform("nunique")
        same_count = value_frame.groupby(
            ["component", "value"],
            sort=False,
            observed=True,
            dropna=False,
        )["value"].transform("size")
        prefix = f"user_{column}"
        feature_values[f"{prefix}_nunique"] = unique_count.astype("int16")
        feature_values[f"{prefix}_same_fraction"] = (same_count / count).astype(
            "float32"
        )

    numeric_columns = [
        column for column in PROFILE_NUMERIC_COLUMNS if column in sequence
    ]
    for column in numeric_columns:
        column_group = group[column]
        prefix = f"user_{column}"
        feature_values[f"{prefix}_mean"] = column_group.transform("mean").astype(
            "float32"
        )
        feature_values[f"{prefix}_std"] = column_group.transform("std").astype(
            "float32"
        )
        if column in {"V279", "V280", "V306", "V307", "V308"}:
            feature_values[f"{prefix}_min"] = column_group.transform("min").astype(
                "float32"
            )
            feature_values[f"{prefix}_max"] = column_group.transform("max").astype(
                "float32"
            )

    sorted_support = key_support[sequence["_row_order"].to_numpy()]
    feature_values["user_valid_link_key_count"] = sorted_support.astype("int8")
    feature_values["user_is_multirow"] = count.gt(1).astype("int8")

    features = pd.DataFrame(feature_values)
    features["_row_order"] = sequence["_row_order"].to_numpy()
    features = features.sort_values("_row_order", kind="stable").drop(
        columns="_row_order"
    )
    features.reset_index(drop=True, inplace=True)
    train_rows = len(train)
    return (
        features.iloc[:train_rows].copy(),
        features.iloc[train_rows:].reset_index(drop=True).copy(),
        features.columns.tolist(),
    )


def _strictly_causal_history(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    y: pd.Series,
    train_components: np.ndarray,
    inference_components: np.ndarray,
    smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    global_rate = float(np.asarray(y).mean())
    history = pd.DataFrame(
        {
            "_component": train_components,
            "_dt": train["TransactionDT"].to_numpy(),
            "_target": np.asarray(y, dtype="int8"),
        }
    )

    buckets = (
        history.groupby(["_component", "_dt"], sort=False, observed=True)
        .agg(
            _bucket_txn_count=("_target", "size"),
            _bucket_fraud_count=("_target", "sum"),
        )
        .reset_index()
        .sort_values(["_component", "_dt"], kind="stable")
    )
    bucket_group = buckets.groupby("_component", sort=False, observed=True)
    buckets["user_known_txn_count"] = (
        bucket_group["_bucket_txn_count"].cumsum()
        - buckets["_bucket_txn_count"]
    ).astype("int32")
    buckets["user_known_fraud_count"] = (
        bucket_group["_bucket_fraud_count"].cumsum()
        - buckets["_bucket_fraud_count"]
    ).astype("int32")
    buckets["_fraud_dt"] = buckets["_dt"].where(
        buckets["_bucket_fraud_count"].gt(0)
    )
    buckets["_last_fraud_including_bucket"] = bucket_group["_fraud_dt"].ffill()
    buckets["_last_known_fraud_dt"] = buckets.groupby(
        "_component", sort=False, observed=True
    )["_last_fraud_including_bucket"].shift(1)

    history_columns = [
        "user_known_txn_count",
        "user_known_fraud_count",
        "_last_known_fraud_dt",
    ]
    train_history = history.merge(
        buckets[["_component", "_dt", *history_columns]],
        on=["_component", "_dt"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    component_totals = (
        history.groupby("_component", sort=False, observed=True)
        .agg(
            user_known_txn_count=("_target", "size"),
            user_known_fraud_count=("_target", "sum"),
        )
        .reset_index()
    )
    fraud_rows = history.loc[history["_target"].eq(1)]
    last_fraud = fraud_rows.groupby(
        "_component", sort=False, observed=True
    )["_dt"].max()

    inference_history = pd.DataFrame(
        {
            "_component": inference_components,
            "_dt": inference["TransactionDT"].to_numpy(),
        }
    ).merge(
        component_totals,
        on="_component",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    inference_history["_last_known_fraud_dt"] = inference_history[
        "_component"
    ].map(last_fraud)
    inference_history["user_known_txn_count"] = inference_history[
        "user_known_txn_count"
    ].fillna(0).astype("int32")
    inference_history["user_known_fraud_count"] = inference_history[
        "user_known_fraud_count"
    ].fillna(0).astype("int32")

    feature_names = list(CAUSAL_HISTORY_COLUMNS)

    def finish(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[
            ["user_known_txn_count", "user_known_fraud_count"]
        ].copy()
        result["user_known_nonfraud_count"] = (
            result["user_known_txn_count"]
            - result["user_known_fraud_count"]
        ).astype("int32")
        result["user_known_fraud_rate"] = (
            result["user_known_fraud_count"]
            / result["user_known_txn_count"].replace(0, np.nan)
        ).astype("float32")
        result["user_known_fraud_rate_smoothed"] = (
            (result["user_known_fraud_count"] + smoothing * global_rate)
            / (result["user_known_txn_count"] + smoothing)
        ).astype("float32")
        result["user_known_any_fraud"] = result[
            "user_known_fraud_count"
        ].gt(0).astype("int8")
        result["user_seconds_since_known_fraud"] = (
            frame["_dt"] - frame["_last_known_fraud_dt"]
        ).astype("float32")
        return result[feature_names]

    return finish(train_history), finish(inference_history), feature_names


def add_user_profile_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    y: pd.Series,
    smoothing: float = 20.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Add graph-user aggregates and causal history, but never the UID itself."""
    (
        train_components,
        inference_components,
        key_support,
        graph_stats,
    ) = build_user_components(train, inference)
    train_profile, inference_profile, profile_names = _profile_aggregates(
        train,
        inference,
        train_components,
        inference_components,
        key_support,
    )
    train_history, inference_history, history_names = _strictly_causal_history(
        train,
        inference,
        y,
        train_components,
        inference_components,
        smoothing,
    )

    train_result = pd.concat(
        [
            train.reset_index(drop=True),
            train_profile,
            train_history.reset_index(drop=True),
        ],
        axis=1,
    )
    inference_result = pd.concat(
        [
            inference.reset_index(drop=True),
            inference_profile,
            inference_history.reset_index(drop=True),
        ],
        axis=1,
    )
    graph_stats["profile_features"] = len(profile_names)
    graph_stats["causal_history_features"] = len(history_names)
    graph_stats["history_smoothing"] = smoothing
    graph_stats["train_target_rate"] = float(np.asarray(y).mean())
    return (
        train_result,
        inference_result,
        [*profile_names, *history_names],
        train_components,
        inference_components,
        graph_stats,
    )


def add_target_free_user_profile_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Add graph-user aggregates without accepting or deriving target data."""
    (
        train_components,
        inference_components,
        key_support,
        graph_stats,
    ) = build_user_components(train, inference)
    train_profile, inference_profile, profile_names = _profile_aggregates(
        train,
        inference,
        train_components,
        inference_components,
        key_support,
    )
    graph_stats["profile_features"] = len(profile_names)
    graph_stats["causal_history_features"] = 0
    return (
        pd.concat(
            [train.reset_index(drop=True), train_profile],
            axis=1,
        ),
        pd.concat(
            [inference.reset_index(drop=True), inference_profile],
            axis=1,
        ),
        profile_names,
        train_components,
        inference_components,
        graph_stats,
    )

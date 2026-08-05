from __future__ import annotations

import numpy as np
import pandas as pd


GENERIC_EMAIL_DOMAINS = {
    "<MISSING>",
    "anonymous.com",
    "mail.com",
}
ROLLING_WINDOWS = (2, 3, 4, 5, 10, 20)
BEHAVIOR_COLUMNS = (
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "ProductCD",
    "card2",
    "card5",
)
ADVANCED_UID_COLUMNS = (
    "uid_adv_component",
    "uid_adv_clean_email",
    "uid_adv_product",
)


def _tokens(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>")


def _integer_tokens(series: pd.Series) -> pd.Series:
    return series.round().astype("Int32").astype("string").fillna("<MISSING>")


def _join_tokens(*values: pd.Series) -> pd.Series:
    result = _tokens(values[0])
    for value in values[1:]:
        result = result.str.cat(_tokens(value), sep="|")
    return result


def _clean_email(series: pd.Series) -> pd.Series:
    result = _tokens(series).str.lower()
    return result.mask(result.isin(GENERIC_EMAIL_DOMAINS), "<MISSING>")


def _find(parent: np.ndarray, node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = int(parent[node])
    return node


def _union(
    parent: np.ndarray,
    sizes: np.ndarray,
    first: int,
    second: int,
    max_component_size: int,
) -> bool:
    first_root = _find(parent, first)
    second_root = _find(parent, second)
    if first_root == second_root:
        return False
    if int(sizes[first_root]) + int(sizes[second_root]) > max_component_size:
        return False
    if sizes[first_root] < sizes[second_root]:
        first_root, second_root = second_root, first_root
    parent[second_root] = first_root
    sizes[first_root] += sizes[second_root]
    return True


def _union_hash_groups(
    frame: pd.DataFrame,
    columns: list[str],
    valid: np.ndarray,
    parent: np.ndarray,
    sizes: np.ndarray,
    max_group_size: int,
    max_component_size: int,
) -> dict:
    rows = np.flatnonzero(valid).astype("int32", copy=False)
    if len(rows) == 0:
        return {
            "valid_rows": 0,
            "linked_groups": 0,
            "oversized_groups": 0,
            "union_edges": 0,
        }
    hashes = pd.util.hash_pandas_object(
        frame.loc[valid, columns],
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
    edges = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_size = int(end - start)
        if group_size < 2:
            continue
        if group_size > max_group_size:
            oversized_groups += 1
            continue
        members = rows[order[start:end]]
        anchor = int(members[0])
        linked_groups += 1
        for member in members[1:]:
            edges += int(
                _union(
                    parent,
                    sizes,
                    anchor,
                    int(member),
                    max_component_size,
                )
            )
    return {
        "valid_rows": int(len(rows)),
        "linked_groups": linked_groups,
        "oversized_groups": oversized_groups,
        "union_edges": edges,
    }


def build_advanced_components(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    max_component_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    columns = [
        "TransactionDT",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "D1_origin_day",
        "D3",
        "C13",
        "P_emaildomain",
        "ProductCD",
    ]
    missing = sorted(set(columns).difference(train.columns))
    if missing:
        raise ValueError(f"Missing columns for advanced UID: {missing}")
    combined = pd.concat(
        [train[columns], inference[columns]],
        ignore_index=True,
        copy=False,
    )
    combined["_origin"] = combined["D1_origin_day"].round()
    combined["_clean_email"] = _clean_email(combined["P_emaildomain"])
    row_count = len(combined)
    parent = np.arange(row_count, dtype="int32")
    sizes = np.ones(row_count, dtype="int32")
    key_metrics = {}

    exact_keys = {
        "card_addr_origin": ["card1", "addr1", "_origin"],
        "card_origin_clean_email": [
            "card1",
            "_origin",
            "_clean_email",
        ],
        "card_full_origin": [
            "card1",
            "card2",
            "card3",
            "card5",
            "_origin",
        ],
    }
    max_group_sizes = {
        "card_addr_origin": 200,
        "card_origin_clean_email": 150,
        "card_full_origin": 150,
    }
    for name, key_columns in exact_keys.items():
        valid = combined[key_columns].notna().all(axis=1).to_numpy()
        if "_clean_email" in key_columns:
            valid &= combined["_clean_email"].ne("<MISSING>").to_numpy()
        key_metrics[name] = _union_hash_groups(
            combined,
            key_columns,
            valid,
            parent,
            sizes,
            max_group_sizes[name],
            max_component_size,
        )

    ordered = combined.reset_index(names="_row").sort_values(
        ["card1", "addr1", "TransactionDT", "_row"],
        kind="stable",
    )
    pair_group = ordered.groupby(
        ["card1", "addr1"],
        sort=False,
        observed=True,
        dropna=False,
    )
    previous_row = pair_group["_row"].shift(1)
    gap_days = pair_group["TransactionDT"].diff() / 86400.0
    d3_error = (gap_days - ordered["D3"]).abs()
    same_product = _tokens(ordered["ProductCD"]).eq(
        _tokens(pair_group["ProductCD"].shift(1))
    )
    current_email = ordered["_clean_email"]
    previous_email = pair_group["_clean_email"].shift(1)
    same_email = (
        current_email.eq(previous_email)
        & current_email.ne("<MISSING>")
    )
    c13_diff = ordered["C13"] - pair_group["C13"].shift(1)
    c13_consistent = c13_diff.between(-1, 10, inclusive="both")
    origin_diff = (
        ordered["_origin"] - pair_group["_origin"].shift(1)
    ).abs()
    fuzzy_valid = (
        previous_row.notna()
        & gap_days.between(0, 120, inclusive="both")
        & d3_error.le(1.5)
        & origin_diff.le(35)
        & (same_product | same_email | c13_consistent)
    )
    fuzzy_edges = 0
    fuzzy_rejected_by_cap = 0
    for current, previous in zip(
        ordered.loc[fuzzy_valid, "_row"].to_numpy(dtype="int32"),
        previous_row.loc[fuzzy_valid].to_numpy(dtype="int32"),
    ):
        if _union(
            parent,
            sizes,
            int(current),
            int(previous),
            max_component_size,
        ):
            fuzzy_edges += 1
        else:
            fuzzy_rejected_by_cap += 1

    roots = np.empty(row_count, dtype="int32")
    for row in range(row_count):
        roots[row] = _find(parent, row)
    components, _ = pd.factorize(roots, sort=False)
    components = components.astype("int32", copy=False)
    component_counts = np.bincount(components)

    origin_token = _integer_tokens(combined["_origin"])
    clean_uid = _join_tokens(
        combined["card1"],
        combined["addr1"],
        origin_token,
        combined["_clean_email"],
    )
    product_uid = _join_tokens(
        combined["card1"],
        combined["addr1"],
        origin_token,
        combined["ProductCD"],
    )
    uid_frame = pd.DataFrame(
        {
            "uid_adv_component": pd.Series(components).map(
                lambda value: f"u{value}"
            ).astype("string"),
            "uid_adv_clean_email": clean_uid,
            "uid_adv_product": product_uid,
        }
    )
    train_rows = len(train)
    stats = {
        "rows": row_count,
        "components": int(len(component_counts)),
        "singleton_components": int(np.sum(component_counts == 1)),
        "rows_in_multirow_components": int(
            component_counts[component_counts > 1].sum()
        ),
        "max_component_size": int(component_counts.max()),
        "exact_keys": key_metrics,
        "fuzzy_d3_candidates": int(fuzzy_valid.sum()),
        "fuzzy_d3_union_edges": fuzzy_edges,
        "fuzzy_d3_rejected_or_existing": fuzzy_rejected_by_cap,
    }
    return (
        components[:train_rows],
        components[train_rows:],
        uid_frame,
        stats,
    )


def _rolling_group_feature(
    sequence: pd.DataFrame,
    value_column: str,
    window: int,
    statistic: str,
) -> pd.Series:
    rolled = (
        sequence.groupby("_component", sort=False, observed=True)[value_column]
        .rolling(window=window, min_periods=1)
        .agg(statistic)
        .reset_index(level=0, drop=True)
    )
    return rolled.reindex(sequence.index).astype("float32")


def add_advanced_user_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    dict,
]:
    (
        train_components,
        inference_components,
        uid_frame,
        stats,
    ) = build_advanced_components(train, inference)
    source_columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "D3",
        "C13",
        *[column for column in BEHAVIOR_COLUMNS if column in train],
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
    for column in ADVANCED_UID_COLUMNS:
        sequence[column] = uid_frame[column]
    sequence = sequence.sort_values(
        ["TransactionDT", "TransactionID"],
        kind="stable",
    ).reset_index(drop=True)
    group = sequence.groupby("_component", sort=False, observed=True)

    feature_values: dict[str, pd.Series | np.ndarray] = {}
    count = group["TransactionID"].transform("size").astype("int32")
    order = group.cumcount().astype("int32")
    feature_values["adv_user_total_count"] = count
    feature_values["adv_user_is_singleton"] = count.eq(1).astype("int8")
    feature_values["adv_user_prior_count"] = order

    day = (sequence["TransactionDT"] // 86400).astype("int32")
    sequence["_day"] = day
    previous_day = group["_day"].shift(1)
    sequence["_new_active_day"] = day.ne(previous_day).astype("int8")
    active_days = group["_new_active_day"].cumsum().astype("int16")
    feature_values["adv_user_active_days_so_far"] = active_days
    feature_values["adv_user_transactions_per_active_day"] = (
        (order + 1) / active_days.replace(0, np.nan)
    ).astype("float32")

    sequence["_dt_gap"] = group["TransactionDT"].diff().astype("float32")
    for window in ROLLING_WINDOWS:
        feature_values[f"adv_user_dt_gap_mean_last_{window}"] = (
            _rolling_group_feature(sequence, "_dt_gap", window, "mean")
        )
        feature_values[f"adv_user_dt_gap_std_last_{window}"] = (
            _rolling_group_feature(sequence, "_dt_gap", window, "std")
        )

    previous_c13 = group["C13"].shift(1)
    feature_values["adv_user_C13_diff_previous"] = (
        sequence["C13"] - previous_c13
    ).astype("float32")
    gap_days = sequence["_dt_gap"] / 86400.0
    feature_values["adv_user_D3_gap_error"] = (
        sequence["D3"] - gap_days
    ).astype("float32")
    feature_values["adv_user_D3_gap_match"] = (
        (sequence["D3"] - gap_days).abs().le(1.5)
        & sequence["D3"].notna()
        & gap_days.notna()
    ).astype("int8")

    for column in BEHAVIOR_COLUMNS:
        if column not in sequence:
            continue
        value = _tokens(sequence[column])
        behavior = pd.DataFrame(
            {
                "component": sequence["_component"],
                "value": value,
                "dt": sequence["TransactionDT"],
            }
        )
        pair_group = behavior.groupby(
            ["component", "value"],
            sort=False,
            observed=True,
            dropna=False,
        )
        current_value_count = pair_group.cumcount().add(1).astype("int32")
        prior_same_count = current_value_count - 1
        running_mode_count = current_value_count.groupby(
            behavior["component"], sort=False
        ).cummax()
        prior_mode_count = running_mode_count.groupby(
            behavior["component"], sort=False
        ).shift(1).fillna(0)
        seconds_since_same = pair_group["dt"].diff()
        previous_value = value.groupby(
            sequence["_component"], sort=False
        ).shift(1)
        prefix = f"adv_user_{column}"
        feature_values[f"{prefix}_prior_count"] = prior_same_count
        feature_values[f"{prefix}_prior_fraction"] = (
            prior_same_count / order.replace(0, np.nan)
        ).astype("float32")
        feature_values[f"{prefix}_is_prior_mode"] = (
            prior_same_count.ge(prior_mode_count) & order.gt(0)
        ).astype("int8")
        feature_values[f"{prefix}_seen_before"] = prior_same_count.gt(0).astype(
            "int8"
        )
        feature_values[f"{prefix}_seconds_since_same"] = (
            seconds_since_same.astype("float32")
        )
        feature_values[f"{prefix}_seen_previous_30d"] = (
            seconds_since_same.between(0, 30 * 86400, inclusive="both")
        ).astype("int8")
        feature_values[f"{prefix}_changed_from_previous"] = (
            previous_value.notna() & value.ne(previous_value)
        ).astype("int8")

    for uid in ADVANCED_UID_COLUMNS:
        feature_values[uid] = sequence[uid].astype("string")
        train_uid = uid_frame[uid].iloc[: len(train)]
        frequencies = train_uid.value_counts(dropna=False) / len(train_uid)
        feature_values[f"{uid}_freq"] = sequence[uid].map(frequencies).fillna(
            0
        ).astype("float32")

    features = pd.DataFrame(feature_values)
    features["_row_order"] = sequence["_row_order"].to_numpy()
    features = features.sort_values("_row_order", kind="stable").drop(
        columns="_row_order"
    )
    features.reset_index(drop=True, inplace=True)
    train_rows = len(train)
    train_features = features.iloc[:train_rows].copy()
    inference_features = features.iloc[train_rows:].reset_index(drop=True).copy()
    numeric_columns = [
        column
        for column in features.columns
        if column not in ADVANCED_UID_COLUMNS
    ]
    for frame in (train_features, inference_features):
        for column in numeric_columns:
            if pd.api.types.is_float_dtype(frame[column]):
                frame[column] = frame[column].astype("float32")
    return (
        pd.concat([train.reset_index(drop=True), train_features], axis=1),
        pd.concat(
            [inference.reset_index(drop=True), inference_features], axis=1
        ),
        features.columns.tolist(),
        list(ADVANCED_UID_COLUMNS),
        train_components,
        inference_components,
        stats,
    )

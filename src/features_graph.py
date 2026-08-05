"""Два графа пользователей и агрегаты по компонентам связности.

Явного user_id нет: строки склеиваются несколькими ключами на основе
origin_day = round(TransactionDT / 86400 - D1). Размер группы и компоненты
ограничен, иначе одна ошибочная склейка утаскивает половину датасета.

Код перенесён без изменений из проверенного прогона: обе модели, собранные
на этих функциях, совпали с эталонными деревом в дерево.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features_row import as_integer_token, as_token, join_tokens



@dataclass(frozen=True)
class GraphKey:
    name: str
    columns: tuple
    max_group_size: int


GRAPH_KEYS = (
    GraphKey("card_addr_origin", ("card1", "addr1", "D1_origin_day"), 200),
    GraphKey("card_origin_email", ("card1", "D1_origin_day", "P_emaildomain"), 200),
    GraphKey("card_full_origin", ("card1", "card2", "card3", "card5", "D1_origin_day"), 150),
    GraphKey("card_addr_origin_device", ("card1", "addr1", "D1_origin_day", "DeviceInfo"), 100),
)

PROFILE_NUMERIC_COLUMNS = ("dist1", "dist2", "D1", "D3", "C1", "C2", "C13", "C14",
                           "V279", "V280", "V306", "V307", "V308")
PROFILE_ID_COLUMNS = ("card1", "card2", "card5", "addr1", "P_emaildomain",
                      "R_emaildomain", "DeviceInfo", "ProductCD")
RAW_UID_COLUMNS = {"uid_card_addr", "uid_card_email", "uid_card_full", "uid_card_device",
                   "uid_email_pair", "uid_d1_email", "uid_card_addr_d1", "uid_card_addr_d1_email"}


def is_raw_uid_feature(column):
    return column in RAW_UID_COLUMNS


def find_root(parent, node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = int(parent[node])
    return node


def union_nodes(parent, sizes, first, second, max_component_size):
    first_root, second_root = find_root(parent, first), find_root(parent, second)
    if first_root == second_root:
        return False
    if int(sizes[first_root]) + int(sizes[second_root]) > max_component_size:
        return False
    if sizes[first_root] < sizes[second_root]:
        first_root, second_root = second_root, first_root
    parent[second_root] = first_root
    sizes[first_root] += sizes[second_root]
    return True


def valid_key_value(series):
    valid = series.notna()
    if isinstance(series.dtype, pd.StringDtype) or pd.api.types.is_object_dtype(series.dtype):
        valid &= series.astype("string").ne("<MISSING>")
    return valid


def union_hash_groups(frame, columns, valid, parent, sizes, max_group_size, max_component_size):
    rows = np.flatnonzero(valid).astype("int32", copy=False)
    if len(rows) == 0:
        return {"valid_rows": 0, "linked_groups": 0, "oversized_groups": 0, "union_edges": 0}
    hashes = pd.util.hash_pandas_object(
        frame.loc[valid, columns], index=False, categorize=True
    ).to_numpy(dtype="uint64", copy=False)
    order = np.argsort(hashes, kind="stable")
    sorted_hashes = hashes[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_hashes[1:] != sorted_hashes[:-1], True])
    linked_groups = oversized_groups = edges = 0
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
            edges += int(union_nodes(parent, sizes, anchor, int(member), max_component_size))
    return {"valid_rows": int(len(rows)), "linked_groups": linked_groups,
            "oversized_groups": oversized_groups, "union_edges": edges}


def build_user_components(train, inference, max_component_size=500):
    required = {column for key in GRAPH_KEYS for column in key.columns}
    combined = pd.concat([train[list(required)], inference[list(required)]], ignore_index=True, copy=False)
    row_count = len(combined)
    parent = np.arange(row_count, dtype="int32")
    sizes = np.ones(row_count, dtype="int32")
    key_support = np.zeros(row_count, dtype="int8")
    key_metrics = []

    for key in GRAPH_KEYS:
        key_values = combined.loc[:, list(key.columns)].copy()
        if "D1_origin_day" in key_values:
            key_values["D1_origin_day"] = key_values["D1_origin_day"].round()
        valid = np.ones(row_count, dtype=bool)
        for column in key.columns:
            valid &= valid_key_value(key_values[column]).to_numpy()
        key_support[np.flatnonzero(valid)] += 1
        metrics = union_hash_groups(key_values, list(key.columns), valid, parent, sizes,
                                    key.max_group_size, max_component_size)
        key_metrics.append({"key": key.name, **metrics})

    roots = np.empty(row_count, dtype="int32")
    for row in range(row_count):
        roots[row] = find_root(parent, row)
    components = pd.factorize(roots, sort=False)[0].astype("int32", copy=False)
    counts = np.bincount(components)
    stats = {
        "rows": row_count, "components": int(len(counts)),
        "singleton_components": int(np.sum(counts == 1)),
        "multirow_components": int(np.sum(counts > 1)),
        "rows_in_multirow_components": int(counts[counts > 1].sum()),
        "max_component_size": int(counts.max()), "keys": key_metrics,
    }
    return components[: len(train)], components[len(train):], key_support, stats


def profile_aggregates(train, inference, train_components, inference_components, key_support):
    source_columns = list(dict.fromkeys(
        ["TransactionID", "TransactionDT", "TransactionAmt"]
        + [c for c in PROFILE_NUMERIC_COLUMNS if c in train]
        + [c for c in PROFILE_ID_COLUMNS if c in train]
    ))
    sequence = pd.concat([train[source_columns], inference[source_columns]], ignore_index=True, copy=False)
    sequence["_component"] = np.concatenate([train_components, inference_components])
    sequence["_row_order"] = np.arange(len(sequence), dtype="int32")
    sequence = sequence.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    group = sequence.groupby("_component", sort=False, observed=True)

    values = {}
    count = group["TransactionID"].transform("size").astype("int32")
    order = group.cumcount().astype("int32")
    values["user_txn_count"] = count
    values["user_order"] = order
    values["user_order_from_end"] = (count - order - 1).astype("int32")
    values["user_order_fraction"] = (order / (count - 1).replace(0, np.nan)).astype("float32")

    previous_dt = group["TransactionDT"].diff()
    first_dt, last_dt = group["TransactionDT"].transform("min"), group["TransactionDT"].transform("max")
    values["user_previous_dt"] = previous_dt.astype("float32")
    values["user_next_dt"] = (group["TransactionDT"].shift(-1) - sequence["TransactionDT"]).astype("float32")
    values["user_time_span"] = (last_dt - first_dt).astype("float32")
    values["user_time_since_first"] = (sequence["TransactionDT"] - first_dt).astype("float32")
    values["user_time_to_last"] = (last_dt - sequence["TransactionDT"]).astype("float32")

    sequence["_previous_dt"] = previous_dt
    gap_group = sequence.groupby("_component", sort=False, observed=True)["_previous_dt"]
    values["user_mean_dt"] = gap_group.transform("mean").astype("float32")
    values["user_std_dt"] = gap_group.transform("std").astype("float32")
    values["user_median_dt"] = gap_group.transform("median").astype("float32")

    amount = sequence["TransactionAmt"]
    amount_mean, amount_std = group["TransactionAmt"].transform("mean"), group["TransactionAmt"].transform("std")
    values["user_amount_mean"] = amount_mean.astype("float32")
    values["user_amount_std"] = amount_std.astype("float32")
    values["user_amount_median"] = group["TransactionAmt"].transform("median").astype("float32")
    values["user_amount_min"] = group["TransactionAmt"].transform("min").astype("float32")
    values["user_amount_max"] = group["TransactionAmt"].transform("max").astype("float32")
    values["user_previous_amount"] = group["TransactionAmt"].shift(1).astype("float32")
    values["user_next_amount"] = group["TransactionAmt"].shift(-1).astype("float32")
    values["user_amount_to_mean"] = (amount / amount_mean.replace(0, np.nan)).astype("float32")
    values["user_amount_zscore"] = ((amount - amount_mean) / amount_std.replace(0, np.nan)).astype("float32")

    amount_pair = sequence.groupby(["_component", "TransactionAmt"], sort=False, observed=True,
                                   dropna=False)["TransactionID"].transform("size")
    values["user_same_amount_count"] = amount_pair.astype("int32")
    values["user_same_amount_fraction"] = (amount_pair / count).astype("float32")

    day_frame = pd.DataFrame({"component": sequence["_component"],
                              "day": (sequence["TransactionDT"] // 86400).astype("int32")})
    values["user_active_day_count"] = day_frame.groupby(
        "component", sort=False, observed=True)["day"].transform("nunique").astype("int16")

    for column in PROFILE_ID_COLUMNS:
        if column not in sequence:
            continue
        value_frame = pd.DataFrame({"component": sequence["_component"], "value": sequence[column]})
        unique_count = value_frame.groupby("component", sort=False, observed=True)["value"].transform("nunique")
        same_count = value_frame.groupby(["component", "value"], sort=False, observed=True,
                                         dropna=False)["value"].transform("size")
        values[f"user_{column}_nunique"] = unique_count.astype("int16")
        values[f"user_{column}_same_fraction"] = (same_count / count).astype("float32")

    for column in [c for c in PROFILE_NUMERIC_COLUMNS if c in sequence]:
        column_group = group[column]
        values[f"user_{column}_mean"] = column_group.transform("mean").astype("float32")
        values[f"user_{column}_std"] = column_group.transform("std").astype("float32")
        if column in {"V279", "V280", "V306", "V307", "V308"}:
            values[f"user_{column}_min"] = column_group.transform("min").astype("float32")
            values[f"user_{column}_max"] = column_group.transform("max").astype("float32")

    values["user_valid_link_key_count"] = key_support[sequence["_row_order"].to_numpy()].astype("int8")
    values["user_is_multirow"] = count.gt(1).astype("int8")

    features = pd.DataFrame(values)
    features["_row_order"] = sequence["_row_order"].to_numpy()
    features = features.sort_values("_row_order", kind="stable").drop(columns="_row_order")
    features.reset_index(drop=True, inplace=True)
    train_rows = len(train)
    return (features.iloc[:train_rows].copy(),
            features.iloc[train_rows:].reset_index(drop=True).copy(),
            features.columns.tolist())


def add_user_profile_features(train, inference):
    train_components, inference_components, key_support, stats = build_user_components(train, inference)
    train_profile, inference_profile, names = profile_aggregates(
        train, inference, train_components, inference_components, key_support
    )
    stats["profile_features"] = len(names)
    return (
        pd.concat([train.reset_index(drop=True), train_profile], axis=1),
        pd.concat([inference.reset_index(drop=True), inference_profile], axis=1),
        names, train_components, inference_components, stats,
    )


GENERIC_EMAIL_DOMAINS = {"<MISSING>", "anonymous.com", "mail.com"}
ROLLING_WINDOWS = (2, 3, 4, 5, 10, 20)
ADV_BEHAVIOR_COLUMNS = ("addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo", "ProductCD", "card2", "card5")
ADVANCED_UID_COLUMNS = ("uid_adv_component", "uid_adv_clean_email", "uid_adv_product")


def clean_email(series):
    result = as_token(series).str.lower()
    return result.mask(result.isin(GENERIC_EMAIL_DOMAINS), "<MISSING>")


def build_advanced_components(train, inference, max_component_size=500):
    columns = ["TransactionDT", "card1", "card2", "card3", "card5", "addr1",
               "D1_origin_day", "D3", "C13", "P_emaildomain", "ProductCD"]
    combined = pd.concat([train[columns], inference[columns]], ignore_index=True, copy=False)
    combined["_origin"] = combined["D1_origin_day"].round()
    combined["_clean_email"] = clean_email(combined["P_emaildomain"])
    row_count = len(combined)
    parent = np.arange(row_count, dtype="int32")
    sizes = np.ones(row_count, dtype="int32")

    exact_keys = {
        "card_addr_origin": ["card1", "addr1", "_origin"],
        "card_origin_clean_email": ["card1", "_origin", "_clean_email"],
        "card_full_origin": ["card1", "card2", "card3", "card5", "_origin"],
    }
    max_group_sizes = {"card_addr_origin": 200, "card_origin_clean_email": 150, "card_full_origin": 150}
    key_metrics = {}
    for name, key_columns in exact_keys.items():
        valid = combined[key_columns].notna().all(axis=1).to_numpy()
        if "_clean_email" in key_columns:
            valid &= combined["_clean_email"].ne("<MISSING>").to_numpy()
        key_metrics[name] = union_hash_groups(combined, key_columns, valid, parent, sizes,
                                              max_group_sizes[name], max_component_size)

    # мягкая склейка по совпадению D3 с реальной паузой
    ordered = combined.reset_index(names="_row").sort_values(
        ["card1", "addr1", "TransactionDT", "_row"], kind="stable")
    pair_group = ordered.groupby(["card1", "addr1"], sort=False, observed=True, dropna=False)
    previous_row = pair_group["_row"].shift(1)
    gap_days = pair_group["TransactionDT"].diff() / 86400.0
    d3_error = (gap_days - ordered["D3"]).abs()
    same_product = as_token(ordered["ProductCD"]).eq(as_token(pair_group["ProductCD"].shift(1)))
    current_email, previous_email = ordered["_clean_email"], pair_group["_clean_email"].shift(1)
    same_email = current_email.eq(previous_email) & current_email.ne("<MISSING>")
    c13_consistent = (ordered["C13"] - pair_group["C13"].shift(1)).between(-1, 10, inclusive="both")
    origin_diff = (ordered["_origin"] - pair_group["_origin"].shift(1)).abs()
    fuzzy_valid = (
        previous_row.notna()
        & gap_days.between(0, 120, inclusive="both")
        & d3_error.le(1.5)
        & origin_diff.le(35)
        & (same_product | same_email | c13_consistent)
    )
    fuzzy_edges = fuzzy_rejected = 0
    for current, previous in zip(ordered.loc[fuzzy_valid, "_row"].to_numpy(dtype="int32"),
                                 previous_row.loc[fuzzy_valid].to_numpy(dtype="int32")):
        if union_nodes(parent, sizes, int(current), int(previous), max_component_size):
            fuzzy_edges += 1
        else:
            fuzzy_rejected += 1

    roots = np.empty(row_count, dtype="int32")
    for row in range(row_count):
        roots[row] = find_root(parent, row)
    components = pd.factorize(roots, sort=False)[0].astype("int32", copy=False)
    counts = np.bincount(components)

    origin_token = as_integer_token(combined["_origin"])
    uid_frame = pd.DataFrame({
        "uid_adv_component": pd.Series(components).map(lambda value: f"u{value}").astype("string"),
        "uid_adv_clean_email": join_tokens(combined["card1"], combined["addr1"], origin_token,
                                           combined["_clean_email"]),
        "uid_adv_product": join_tokens(combined["card1"], combined["addr1"], origin_token,
                                       combined["ProductCD"]),
    })
    stats = {
        "rows": row_count, "components": int(len(counts)),
        "singleton_components": int(np.sum(counts == 1)),
        "rows_in_multirow_components": int(counts[counts > 1].sum()),
        "max_component_size": int(counts.max()), "exact_keys": key_metrics,
        "fuzzy_d3_candidates": int(fuzzy_valid.sum()), "fuzzy_d3_union_edges": fuzzy_edges,
        "fuzzy_d3_rejected_or_existing": fuzzy_rejected,
    }
    return components[: len(train)], components[len(train):], uid_frame, stats


def rolling_group_feature(sequence, value_column, window, statistic):
    rolled = (
        sequence.groupby("_component", sort=False, observed=True)[value_column]
        .rolling(window=window, min_periods=1).agg(statistic).reset_index(level=0, drop=True)
    )
    return rolled.reindex(sequence.index).astype("float32")


def add_advanced_user_features(train, inference):
    train_components, inference_components, uid_frame, stats = build_advanced_components(train, inference)
    source_columns = list(dict.fromkeys(
        ["TransactionID", "TransactionDT", "TransactionAmt", "D3", "C13"]
        + [c for c in ADV_BEHAVIOR_COLUMNS if c in train]
    ))
    sequence = pd.concat([train[source_columns], inference[source_columns]], ignore_index=True, copy=False)
    sequence["_component"] = np.concatenate([train_components, inference_components])
    sequence["_row_order"] = np.arange(len(sequence), dtype="int32")
    for column in ADVANCED_UID_COLUMNS:
        sequence[column] = uid_frame[column]
    sequence = sequence.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    group = sequence.groupby("_component", sort=False, observed=True)

    values = {}
    count = group["TransactionID"].transform("size").astype("int32")
    order = group.cumcount().astype("int32")
    values["adv_user_total_count"] = count
    values["adv_user_is_singleton"] = count.eq(1).astype("int8")
    values["adv_user_prior_count"] = order

    day = (sequence["TransactionDT"] // 86400).astype("int32")
    sequence["_day"] = day
    sequence["_new_active_day"] = day.ne(group["_day"].shift(1)).astype("int8")
    active_days = group["_new_active_day"].cumsum().astype("int16")
    values["adv_user_active_days_so_far"] = active_days
    values["adv_user_transactions_per_active_day"] = (
        (order + 1) / active_days.replace(0, np.nan)
    ).astype("float32")

    sequence["_dt_gap"] = group["TransactionDT"].diff().astype("float32")
    for window in ROLLING_WINDOWS:
        values[f"adv_user_dt_gap_mean_last_{window}"] = rolling_group_feature(sequence, "_dt_gap", window, "mean")
        values[f"adv_user_dt_gap_std_last_{window}"] = rolling_group_feature(sequence, "_dt_gap", window, "std")

    values["adv_user_C13_diff_previous"] = (sequence["C13"] - group["C13"].shift(1)).astype("float32")
    gap_days = sequence["_dt_gap"] / 86400.0
    values["adv_user_D3_gap_error"] = (sequence["D3"] - gap_days).astype("float32")
    values["adv_user_D3_gap_match"] = (
        (sequence["D3"] - gap_days).abs().le(1.5) & sequence["D3"].notna() & gap_days.notna()
    ).astype("int8")

    for column in ADV_BEHAVIOR_COLUMNS:
        if column not in sequence:
            continue
        value = as_token(sequence[column])
        behavior = pd.DataFrame({"component": sequence["_component"], "value": value,
                                 "dt": sequence["TransactionDT"]})
        pair_group = behavior.groupby(["component", "value"], sort=False, observed=True, dropna=False)
        current_value_count = pair_group.cumcount().add(1).astype("int32")
        prior_same_count = current_value_count - 1
        running_mode_count = current_value_count.groupby(behavior["component"], sort=False).cummax()
        prior_mode_count = running_mode_count.groupby(behavior["component"], sort=False).shift(1).fillna(0)
        seconds_since_same = pair_group["dt"].diff()
        previous_value = value.groupby(sequence["_component"], sort=False).shift(1)
        prefix = f"adv_user_{column}"
        values[f"{prefix}_prior_count"] = prior_same_count
        values[f"{prefix}_prior_fraction"] = (prior_same_count / order.replace(0, np.nan)).astype("float32")
        values[f"{prefix}_is_prior_mode"] = (prior_same_count.ge(prior_mode_count) & order.gt(0)).astype("int8")
        values[f"{prefix}_seen_before"] = prior_same_count.gt(0).astype("int8")
        values[f"{prefix}_seconds_since_same"] = seconds_since_same.astype("float32")
        values[f"{prefix}_seen_previous_30d"] = seconds_since_same.between(
            0, 30 * 86400, inclusive="both").astype("int8")
        values[f"{prefix}_changed_from_previous"] = (
            previous_value.notna() & value.ne(previous_value)).astype("int8")

    for uid in ADVANCED_UID_COLUMNS:
        values[uid] = sequence[uid].astype("string")
        train_uid = uid_frame[uid].iloc[: len(train)]
        frequencies = train_uid.value_counts(dropna=False) / len(train_uid)
        values[f"{uid}_freq"] = sequence[uid].map(frequencies).fillna(0).astype("float32")

    features = pd.DataFrame(values)
    features["_row_order"] = sequence["_row_order"].to_numpy()
    features = features.sort_values("_row_order", kind="stable").drop(columns="_row_order")
    features.reset_index(drop=True, inplace=True)
    train_rows = len(train)
    train_features = features.iloc[:train_rows].copy()
    inference_features = features.iloc[train_rows:].reset_index(drop=True).copy()
    for frame in (train_features, inference_features):
        for column in features.columns:
            if column not in ADVANCED_UID_COLUMNS and pd.api.types.is_float_dtype(frame[column]):
                frame[column] = frame[column].astype("float32")
    return (
        pd.concat([train.reset_index(drop=True), train_features], axis=1),
        pd.concat([inference.reset_index(drop=True), inference_features], axis=1),
        features.columns.tolist(), list(ADVANCED_UID_COLUMNS),
        train_components, inference_components, stats,
    )

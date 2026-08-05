from __future__ import annotations

import numpy as np
import pandas as pd


COUNTER_COLUMNS = ("V126", "V127", "V128", "V306", "V307", "V308")
CHAIN_COLUMN = "multi_counter_transaction_chain"


def _build_chain_ids(
    sequence: pd.DataFrame,
    max_chain_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    uid_codes, _ = pd.factorize(
        sequence["uid_card_addr_d1_email"],
        sort=False,
    )
    transaction_dt = sequence["TransactionDT"].to_numpy()
    transaction_id = sequence["TransactionID"].to_numpy()
    amount = sequence["TransactionAmt"].to_numpy(dtype="float64", copy=False)
    counter_values = {
        column: sequence[column].to_numpy(dtype="float64", copy=False)
        for column in COUNTER_COLUMNS
    }
    order = np.lexsort((transaction_id, transaction_dt, uid_codes))

    chain_ids = np.empty(len(sequence), dtype="int32")
    match_votes = np.zeros(len(sequence), dtype="int8")
    chain_sizes = np.zeros(len(sequence), dtype="int32")
    next_chain_id = 0
    current_uid = None
    endpoint_maps: dict[str, dict[int, tuple[int, int]]] = {}
    capped_candidates = 0

    for sequence_position, row in enumerate(order):
        uid = int(uid_codes[row])
        if uid != current_uid:
            endpoint_maps = {column: {} for column in COUNTER_COLUMNS}
            current_uid = uid

        candidates: dict[int, tuple[int, int]] = {}
        if np.isfinite(amount[row]):
            for column in COUNTER_COLUMNS:
                value = counter_values[column][row]
                if not np.isfinite(value):
                    continue
                start = int(np.rint(value * 1000.0))
                matches = [
                    endpoint_maps[column][key]
                    for key in (start - 1, start, start + 1)
                    if key in endpoint_maps[column]
                ]
                if not matches:
                    continue
                chain_id, last_position = max(
                    matches,
                    key=lambda candidate: candidate[1],
                )
                if chain_sizes[chain_id] >= max_chain_size:
                    capped_candidates += 1
                    continue
                votes, latest = candidates.get(chain_id, (0, -1))
                candidates[chain_id] = (
                    votes + 1,
                    max(latest, last_position),
                )

        if candidates:
            chain_id, (votes, _) = max(
                candidates.items(),
                key=lambda item: (item[1][0], item[1][1]),
            )
            match_votes[row] = votes
        else:
            chain_id = next_chain_id
            next_chain_id += 1
        chain_ids[row] = chain_id
        chain_sizes[chain_id] += 1

        if np.isfinite(amount[row]):
            for column in COUNTER_COLUMNS:
                value = counter_values[column][row]
                if not np.isfinite(value):
                    continue
                endpoint = int(np.rint((value + amount[row]) * 1000.0))
                endpoint_maps[column][endpoint] = (
                    chain_id,
                    sequence_position,
                )

    counts = np.bincount(chain_ids)
    stats = {
        "chains": int(len(counts)),
        "linked_rows": int(np.sum(match_votes > 0)),
        "rows_with_two_or_more_votes": int(np.sum(match_votes >= 2)),
        "multirow_chains": int(np.sum(counts > 1)),
        "rows_in_multirow_chains": int(counts[counts > 1].sum()),
        "max_chain_size": int(counts.max()),
        "chain_size_cap": int(max_chain_size),
        "capped_counter_matches": int(capped_candidates),
    }
    return chain_ids, match_votes, stats


def add_multi_counter_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    max_chain_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict]:
    source_columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "uid_card_addr_d1_email",
        *COUNTER_COLUMNS,
    ]
    missing = sorted(set(source_columns).difference(train.columns))
    if missing:
        raise ValueError(f"Missing multi-counter columns: {missing}")
    sequence = pd.concat(
        [train[source_columns], inference[source_columns]],
        ignore_index=True,
        copy=False,
    )
    train_rows = len(train)
    chain_ids, match_votes, stats = _build_chain_ids(
        sequence,
        max_chain_size=max_chain_size,
    )

    sequence["_chain"] = chain_ids
    sequence["_match_votes"] = match_votes
    sequence["_row_order"] = np.arange(len(sequence), dtype="int32")
    sequence = sequence.sort_values(
        ["TransactionDT", "TransactionID"],
        kind="stable",
    ).reset_index(drop=True)
    uid_group = sequence.groupby(
        "uid_card_addr_d1_email",
        sort=False,
        observed=True,
        dropna=False,
    )
    chain_group = sequence.groupby("_chain", sort=False, observed=True)

    feature_values: dict[str, pd.Series | np.ndarray] = {}
    previous_matches = []
    next_matches = []
    previous_unchanged = []
    next_unchanged = []
    previous_amount = uid_group["TransactionAmt"].shift(1)
    for column in COUNTER_COLUMNS:
        previous_value = uid_group[column].shift(1)
        next_value = uid_group[column].shift(-1)
        previous_delta = sequence[column] - previous_value
        next_delta = next_value - sequence[column]
        previous_error = previous_delta - previous_amount
        next_error = next_delta - sequence["TransactionAmt"]
        prefix = f"multi_{column}"
        feature_values[f"{prefix}_previous_delta"] = previous_delta.astype(
            "float32"
        )
        feature_values[f"{prefix}_previous_amount_error"] = (
            previous_error.astype("float32")
        )
        feature_values[f"{prefix}_next_delta"] = next_delta.astype("float32")
        feature_values[f"{prefix}_next_amount_error"] = next_error.astype(
            "float32"
        )
        previous_match = previous_error.abs().le(0.011) & previous_error.notna()
        next_match = next_error.abs().le(0.011) & next_error.notna()
        previous_zero = previous_delta.abs().le(0.011) & previous_delta.notna()
        next_zero = next_delta.abs().le(0.011) & next_delta.notna()
        feature_values[f"{prefix}_previous_amount_match"] = previous_match.astype(
            "int8"
        )
        feature_values[f"{prefix}_next_amount_match"] = next_match.astype("int8")
        feature_values[f"{prefix}_previous_unchanged"] = previous_zero.astype(
            "int8"
        )
        feature_values[f"{prefix}_next_unchanged"] = next_zero.astype("int8")
        previous_matches.append(previous_match.to_numpy(dtype="int8"))
        next_matches.append(next_match.to_numpy(dtype="int8"))
        previous_unchanged.append(previous_zero.to_numpy(dtype="int8"))
        next_unchanged.append(next_zero.to_numpy(dtype="int8"))

    feature_values["multi_counter_previous_match_count"] = np.sum(
        previous_matches,
        axis=0,
        dtype="int8",
    )
    feature_values["multi_counter_next_match_count"] = np.sum(
        next_matches,
        axis=0,
        dtype="int8",
    )
    feature_values["multi_counter_previous_unchanged_count"] = np.sum(
        previous_unchanged,
        axis=0,
        dtype="int8",
    )
    feature_values["multi_counter_next_unchanged_count"] = np.sum(
        next_unchanged,
        axis=0,
        dtype="int8",
    )

    count = chain_group["TransactionID"].transform("size").astype("int32")
    position = chain_group.cumcount().astype("int32")
    first_dt = chain_group["TransactionDT"].transform("min")
    last_dt = chain_group["TransactionDT"].transform("max")
    feature_values[CHAIN_COLUMN] = (
        "mc" + sequence["_chain"].astype("string")
    )
    feature_values["multi_chain_total_count"] = count
    feature_values["multi_chain_position"] = position
    feature_values["multi_chain_position_from_end"] = count - position - 1
    feature_values["multi_chain_previous_dt"] = chain_group[
        "TransactionDT"
    ].diff().astype("float32")
    feature_values["multi_chain_next_dt"] = (
        chain_group["TransactionDT"].shift(-1) - sequence["TransactionDT"]
    ).astype("float32")
    feature_values["multi_chain_time_span"] = (last_dt - first_dt).astype(
        "float32"
    )
    feature_values["multi_chain_previous_amount"] = chain_group[
        "TransactionAmt"
    ].shift(1).astype("float32")
    feature_values["multi_chain_next_amount"] = chain_group[
        "TransactionAmt"
    ].shift(-1).astype("float32")
    feature_values["multi_chain_match_votes"] = sequence["_match_votes"].astype(
        "int8"
    )

    features = pd.DataFrame(feature_values)
    features["_row_order"] = sequence["_row_order"].to_numpy()
    features = features.sort_values("_row_order", kind="stable").drop(
        columns="_row_order"
    )
    features.reset_index(drop=True, inplace=True)
    feature_names = features.columns.tolist()
    stats["rows"] = int(len(sequence))
    stats["features"] = int(len(feature_names))

    train_features = features.iloc[:train_rows].reset_index(drop=True)
    inference_features = features.iloc[train_rows:].reset_index(drop=True)
    return (
        pd.concat([train.reset_index(drop=True), train_features], axis=1),
        pd.concat(
            [inference.reset_index(drop=True), inference_features],
            axis=1,
        ),
        feature_names,
        [CHAIN_COLUMN],
        stats,
    )

from __future__ import annotations

import numpy as np
import pandas as pd


CHAIN_COLUMN = "v307_transaction_chain"
CHAIN_TARGET_COLUMNS = (
    "v307_chain_known_txn_count",
    "v307_chain_known_fraud_count",
    "v307_chain_known_fraud_rate_smoothed_5",
    "v307_chain_known_fraud_rate_smoothed_20",
    "v307_chain_known_any_fraud",
    "v307_chain_seconds_since_known_fraud",
)


def _build_chain_ids(
    sequence: pd.DataFrame,
    max_chain_size: int | None = None,
) -> tuple[np.ndarray, dict]:
    uid_codes, _ = pd.factorize(
        sequence["uid_card_addr_d1_email"],
        sort=False,
    )
    transaction_dt = sequence["TransactionDT"].to_numpy()
    transaction_id = sequence["TransactionID"].to_numpy()
    v307 = sequence["V307"].to_numpy(dtype="float64", copy=False)
    amount = sequence["TransactionAmt"].to_numpy(dtype="float64", copy=False)
    valid = np.isfinite(v307) & np.isfinite(amount)
    order = np.lexsort((transaction_id, transaction_dt, uid_codes))

    chain_ids = np.empty(len(sequence), dtype="int32")
    chain_sizes = np.zeros(len(sequence), dtype="int32")
    next_chain_id = 0
    current_uid = None
    endpoints: dict[int, tuple[int, int]] = {}
    linked_rows = 0
    capped_candidates = 0
    for sequence_position, row in enumerate(order):
        uid = int(uid_codes[row])
        if uid != current_uid:
            endpoints = {}
            current_uid = uid

        chain_id = -1
        if valid[row]:
            start = int(np.rint(v307[row] * 1000.0))
            candidates = [
                endpoints[key]
                for key in (start - 1, start, start + 1)
                if key in endpoints
            ]
            if max_chain_size is not None and candidates:
                uncapped_candidates = candidates
                candidates = [
                    candidate
                    for candidate in candidates
                    if chain_sizes[candidate[0]] < max_chain_size
                ]
                capped_candidates += int(
                    bool(uncapped_candidates) and not candidates
                )
            if candidates:
                chain_id, _ = max(candidates, key=lambda value: value[1])
                linked_rows += 1

        if chain_id < 0:
            chain_id = next_chain_id
            next_chain_id += 1
        chain_ids[row] = chain_id
        chain_sizes[chain_id] += 1

        if valid[row]:
            endpoint = int(np.rint((v307[row] + amount[row]) * 1000.0))
            endpoints[endpoint] = (chain_id, sequence_position)

    counts = np.bincount(chain_ids)
    return chain_ids, {
        "chains": int(len(counts)),
        "linked_rows": int(linked_rows),
        "multirow_chains": int(np.sum(counts > 1)),
        "rows_in_multirow_chains": int(counts[counts > 1].sum()),
        "max_chain_size": int(counts.max()),
        "chain_size_cap": max_chain_size,
        "capped_link_candidates": int(capped_candidates),
    }


def _target_free_features(
    sequence: pd.DataFrame,
    chain_ids: np.ndarray,
    train_rows: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    ordered = sequence.copy()
    ordered["_chain"] = chain_ids
    ordered["_row_order"] = np.arange(len(ordered), dtype="int32")
    ordered = ordered.sort_values(
        ["TransactionDT", "TransactionID"],
        kind="stable",
    ).reset_index(drop=True)
    group = ordered.groupby("_chain", sort=False, observed=True)
    count = group["TransactionID"].transform("size").astype("int32")
    position = group.cumcount().astype("int32")
    first_dt = group["TransactionDT"].transform("min")
    last_dt = group["TransactionDT"].transform("max")

    feature_names = [
        CHAIN_COLUMN,
        "v307_chain_total_count",
        "v307_chain_position",
        "v307_chain_position_from_end",
        "v307_chain_previous_dt",
        "v307_chain_next_dt",
        "v307_chain_time_span",
        "v307_chain_previous_amount",
        "v307_chain_next_amount",
        "v307_chain_has_previous",
        "v307_chain_has_next",
    ]
    features = pd.DataFrame(
        {
            CHAIN_COLUMN: (
                "vc" + ordered["_chain"].astype("string")
            ),
            "v307_chain_total_count": count,
            "v307_chain_position": position,
            "v307_chain_position_from_end": count - position - 1,
            "v307_chain_previous_dt": group["TransactionDT"].diff().astype(
                "float32"
            ),
            "v307_chain_next_dt": (
                group["TransactionDT"].shift(-1) - ordered["TransactionDT"]
            ).astype("float32"),
            "v307_chain_time_span": (last_dt - first_dt).astype("float32"),
            "v307_chain_previous_amount": group["TransactionAmt"].shift(1).astype(
                "float32"
            ),
            "v307_chain_next_amount": group["TransactionAmt"].shift(-1).astype(
                "float32"
            ),
            "v307_chain_has_previous": position.gt(0).astype("int8"),
            "v307_chain_has_next": position.lt(count - 1).astype("int8"),
            "_row_order": ordered["_row_order"],
        }
    )
    features = features.sort_values("_row_order", kind="stable").drop(
        columns="_row_order"
    )
    features.reset_index(drop=True, inplace=True)
    return features, feature_names, [CHAIN_COLUMN]


def _causal_target_features(
    sequence: pd.DataFrame,
    chain_ids: np.ndarray,
    y: pd.Series,
    train_rows: int,
) -> pd.DataFrame:
    global_rate = float(np.asarray(y).mean())
    history = pd.DataFrame(
        {
            "_chain": chain_ids[:train_rows],
            "_dt": sequence["TransactionDT"].to_numpy()[:train_rows],
            "_target": np.asarray(y, dtype="int8"),
        }
    )
    buckets = (
        history.groupby(["_chain", "_dt"], sort=False, observed=True)
        .agg(
            _bucket_txn_count=("_target", "size"),
            _bucket_fraud_count=("_target", "sum"),
        )
        .reset_index()
        .sort_values(["_chain", "_dt"], kind="stable")
    )
    group = buckets.groupby("_chain", sort=False, observed=True)
    buckets["_known_txn"] = (
        group["_bucket_txn_count"].cumsum() - buckets["_bucket_txn_count"]
    ).astype("int32")
    buckets["_known_fraud"] = (
        group["_bucket_fraud_count"].cumsum()
        - buckets["_bucket_fraud_count"]
    ).astype("int32")
    buckets["_fraud_dt"] = buckets["_dt"].where(
        buckets["_bucket_fraud_count"].gt(0)
    )
    buckets["_last_fraud"] = group["_fraud_dt"].ffill()
    buckets["_known_last_fraud"] = buckets.groupby(
        "_chain", sort=False, observed=True
    )["_last_fraud"].shift(1)
    train_history = history.merge(
        buckets[
            [
                "_chain",
                "_dt",
                "_known_txn",
                "_known_fraud",
                "_known_last_fraud",
            ]
        ],
        on=["_chain", "_dt"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    totals = (
        history.groupby("_chain", sort=False, observed=True)
        .agg(
            _known_txn=("_target", "size"),
            _known_fraud=("_target", "sum"),
        )
        .reset_index()
    )
    last_fraud = history.loc[history["_target"].eq(1)].groupby(
        "_chain", sort=False, observed=True
    )["_dt"].max()
    inference_history = pd.DataFrame(
        {
            "_chain": chain_ids[train_rows:],
            "_dt": sequence["TransactionDT"].to_numpy()[train_rows:],
        }
    ).merge(
        totals,
        on="_chain",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    inference_history["_known_txn"] = inference_history["_known_txn"].fillna(0)
    inference_history["_known_fraud"] = inference_history["_known_fraud"].fillna(0)
    inference_history["_known_last_fraud"] = inference_history["_chain"].map(
        last_fraud
    )

    def finish(frame: pd.DataFrame) -> pd.DataFrame:
        known_txn = frame["_known_txn"].astype("int32")
        known_fraud = frame["_known_fraud"].astype("int32")
        return pd.DataFrame(
            {
                "v307_chain_known_txn_count": known_txn,
                "v307_chain_known_fraud_count": known_fraud,
                "v307_chain_known_fraud_rate_smoothed_5": (
                    (known_fraud + 5.0 * global_rate) / (known_txn + 5.0)
                ).astype("float32"),
                "v307_chain_known_fraud_rate_smoothed_20": (
                    (known_fraud + 20.0 * global_rate) / (known_txn + 20.0)
                ).astype("float32"),
                "v307_chain_known_any_fraud": known_fraud.gt(0).astype("int8"),
                "v307_chain_seconds_since_known_fraud": (
                    frame["_dt"] - frame["_known_last_fraud"]
                ).astype("float32"),
            }
        )[list(CHAIN_TARGET_COLUMNS)]

    return pd.concat(
        [finish(train_history), finish(inference_history)],
        ignore_index=True,
    )


def add_v307_transaction_chain_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    y: pd.Series,
    include_target_history: bool = False,
    max_chain_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict]:
    required = {
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "V307",
        "uid_card_addr_d1_email",
    }
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Missing V307 chain columns: {missing}")

    source_columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "V307",
        "uid_card_addr_d1_email",
    ]
    sequence = pd.concat(
        [train[source_columns], inference[source_columns]],
        ignore_index=True,
        copy=False,
    )
    train_rows = len(train)
    chain_ids, stats = _build_chain_ids(
        sequence,
        max_chain_size=max_chain_size,
    )
    features, feature_names, categorical = _target_free_features(
        sequence,
        chain_ids,
        train_rows,
    )
    if include_target_history:
        target_features = _causal_target_features(
            sequence,
            chain_ids,
            y,
            train_rows,
        )
        features = pd.concat([features, target_features], axis=1)
        feature_names.extend(CHAIN_TARGET_COLUMNS)

    train_features = features.iloc[:train_rows].reset_index(drop=True)
    inference_features = features.iloc[train_rows:].reset_index(drop=True)
    reference_counts = np.bincount(
        chain_ids[:train_rows],
        minlength=int(chain_ids.max()) + 1,
    )
    linked_inference = reference_counts[chain_ids[train_rows:]] > 0
    stats.update(
        {
            "rows": int(len(sequence)),
            "inference_rows_linked_to_reference": int(linked_inference.sum()),
            "inference_link_rate": float(linked_inference.mean()),
            "target_history_features": bool(include_target_history),
        }
    )
    return (
        pd.concat([train.reset_index(drop=True), train_features], axis=1),
        pd.concat(
            [inference.reset_index(drop=True), inference_features],
            axis=1,
        ),
        feature_names,
        categorical,
        stats,
    )

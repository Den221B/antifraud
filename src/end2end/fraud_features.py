from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "isFraud"
GIBA_UID_COLUMNS = [
    "uid_d1_email",
    "uid_card_addr_d1",
    "uid_card_addr_d1_email",
]
GIBA_SEQUENCE_UID_COLUMNS = [
    "uid_d1_email",
    "uid_card_addr_d1_email",
]
SELECTED_FREQUENCY_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "DeviceInfo_family",
    "uid_card_addr",
    "uid_card_email",
    "uid_card_full",
    "uid_card_device",
    "uid_email_pair",
    *GIBA_UID_COLUMNS,
]


def read_and_merge(data_dir, split):
    data_dir = Path(data_dir)
    transaction = pd.read_csv(data_dir / f"{split}_transaction.csv")
    identity = pd.read_csv(data_dir / f"{split}_identity.csv")

    transaction = transaction.drop(columns=["Unnamed: 0"], errors="ignore")
    identity = identity.drop(columns=["Unnamed: 0"], errors="ignore")

    return transaction.merge(identity, on="TransactionID", how="left")


def _as_token(series):
    return series.astype("string").fillna("<MISSING>")


def _as_integer_token(series):
    return series.round().astype("Int32").astype("string").fillna("<MISSING>")


def _join_tokens(*series):
    result = _as_token(series[0])
    for value in series[1:]:
        result = result.str.cat(_as_token(value), sep="|")
    return result


def _add_group_aggregates(frame, prefix, columns):
    if not columns:
        return

    values = frame[columns]
    frame[f"{prefix}_missing_count"] = values.isna().sum(axis=1).astype("int16")
    frame[f"{prefix}_mean"] = values.mean(axis=1).astype("float32")
    frame[f"{prefix}_std"] = values.std(axis=1).astype("float32")
    frame[f"{prefix}_min"] = values.min(axis=1).astype("float32")
    frame[f"{prefix}_max"] = values.max(axis=1).astype("float32")


def _add_row_features(frame):
    transaction_dt = frame["TransactionDT"]
    transaction_day = transaction_dt / 86400
    amount = frame["TransactionAmt"]

    frame["DT_day"] = np.floor(transaction_day).astype("int16")
    frame["DT_week"] = np.floor(transaction_day / 7).astype("int16")
    frame["DT_hour"] = ((transaction_dt // 3600) % 24).astype("int8")
    frame["DT_dayofweek"] = (
        (transaction_dt // 86400) % 7
    ).astype("int8")

    frame["TransactionAmt_log1p"] = np.log1p(amount).astype("float32")
    frame["TransactionAmt_cents"] = (
        np.round((amount - np.floor(amount)) * 100) % 100
    ).astype("int8")
    frame["TransactionAmt_is_integer"] = (
        np.isclose(amount % 1, 0)
    ).astype("int8")
    frame["TransactionAmt_is_round_10"] = (
        np.isclose(amount % 10, 0)
    ).astype("int8")

    p_email = _as_token(frame["P_emaildomain"])
    r_email = _as_token(frame["R_emaildomain"])
    frame["P_R_email_match"] = (p_email == r_email).astype("int8")
    frame["P_email_suffix"] = p_email.str.rsplit(".", n=1).str[-1]
    frame["R_email_suffix"] = r_email.str.rsplit(".", n=1).str[-1]

    if "DeviceInfo" in frame:
        device_info = _as_token(frame["DeviceInfo"])
        frame["DeviceInfo_family"] = device_info.str.split("/", n=1).str[0]

    if "id_31" in frame:
        browser = _as_token(frame["id_31"])
        frame["browser_family"] = browser.str.replace(
            r"[\d._-]+$", "", regex=True
        ).str.strip()

    combo_columns = {
        "uid_card_addr": ["card1", "addr1"],
        "uid_card_email": ["card1", "addr1", "P_emaildomain"],
        "uid_card_full": ["card1", "card2", "card3", "card5"],
        "uid_card_device": ["card1", "addr1", "DeviceInfo"],
        "uid_email_pair": ["P_emaildomain", "R_emaildomain"],
    }
    for name, columns in combo_columns.items():
        frame[name] = _as_token(frame[columns[0]])
        for column in columns[1:]:
            frame[name] = frame[name].str.cat(
                _as_token(frame[column]),
                sep="|",
            )

    d_columns = [f"D{i}" for i in range(1, 16) if f"D{i}" in frame]
    for column in d_columns:
        frame[f"{column}_minus_day"] = (
            frame[column] - transaction_day
        ).astype("float32")

    c_columns = [f"C{i}" for i in range(1, 15) if f"C{i}" in frame]
    v_columns = [f"V{i}" for i in range(1, 340) if f"V{i}" in frame]
    id_numeric = [
        column
        for column in frame.columns
        if column.startswith("id_") and pd.api.types.is_numeric_dtype(frame[column])
    ]

    _add_group_aggregates(frame, "C", c_columns)
    _add_group_aggregates(frame, "D", d_columns)
    _add_group_aggregates(frame, "V", v_columns)
    _add_group_aggregates(frame, "id_numeric", id_numeric)

    frame["row_missing_count"] = frame.isna().sum(axis=1).astype("int16")


def _add_giba_uid_features(frame):
    d1_origin_day = frame["DT_day"].astype("float32") - frame["D1"]
    frame["D1_origin_day"] = d1_origin_day.astype("float32")

    origin_token = _as_integer_token(d1_origin_day)
    p_email = _as_token(frame["P_emaildomain"])
    frame["uid_d1_email"] = origin_token.str.cat(p_email, sep="|")
    frame["uid_card_addr_d1"] = _join_tokens(
        frame["card1"],
        frame["addr1"],
        origin_token,
    )
    frame["uid_card_addr_d1_email"] = frame["uid_card_addr_d1"].str.cat(
        p_email,
        sep="|",
    )


def _add_uid_sequence_features(
    train,
    test,
    add_v307_chain=False,
):
    columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        *GIBA_SEQUENCE_UID_COLUMNS,
    ]
    if add_v307_chain and "V307" in train:
        columns.append("V307")
    sequence = pd.concat(
        [train[columns], test[columns]],
        ignore_index=True,
        copy=False,
    )
    sequence["_row_order"] = np.arange(len(sequence), dtype="int32")
    sequence = sequence.sort_values(
        ["TransactionDT", "TransactionID"],
        kind="stable",
    ).reset_index(drop=True)

    feature_names = []
    for uid in GIBA_SEQUENCE_UID_COLUMNS:
        group = sequence.groupby(uid, sort=False, observed=True, dropna=False)
        prefix = f"{uid}_seq"

        sequence[f"{prefix}_count"] = group["TransactionID"].transform(
            "size"
        ).astype("int32")
        sequence[f"{prefix}_order"] = group.cumcount().astype("int32")
        sequence[f"{prefix}_order_from_end"] = (
            sequence[f"{prefix}_count"] - sequence[f"{prefix}_order"] - 1
        ).astype("int32")

        previous_dt = group["TransactionDT"].diff()
        next_dt = group["TransactionDT"].shift(-1) - sequence["TransactionDT"]
        sequence[f"{prefix}_previous_dt"] = previous_dt.astype("float32")
        sequence[f"{prefix}_next_dt"] = next_dt.astype("float32")

        gap_column = f"_{uid}_gap"
        sequence[gap_column] = previous_dt
        gap_group = sequence.groupby(
            uid,
            sort=False,
            observed=True,
            dropna=False,
        )[gap_column]
        sequence[f"{prefix}_mean_dt"] = gap_group.transform("mean").astype(
            "float32"
        )
        sequence[f"{prefix}_std_dt"] = gap_group.transform("std").astype(
            "float32"
        )
        sequence[f"{prefix}_median_dt"] = gap_group.transform("median").astype(
            "float32"
        )
        sequence.drop(columns=[gap_column], inplace=True)

        first_dt = group["TransactionDT"].transform("min")
        last_dt = group["TransactionDT"].transform("max")
        sequence[f"{prefix}_time_span"] = (last_dt - first_dt).astype("float32")

        mean_amount = group["TransactionAmt"].transform("mean")
        std_amount = group["TransactionAmt"].transform("std")
        sequence[f"{prefix}_mean_amt"] = mean_amount.astype("float32")
        sequence[f"{prefix}_std_amt"] = std_amount.astype("float32")
        sequence[f"{prefix}_median_amt"] = group[
            "TransactionAmt"
        ].transform("median").astype("float32")
        sequence[f"{prefix}_previous_amt"] = group[
            "TransactionAmt"
        ].shift(1).astype("float32")
        sequence[f"{prefix}_next_amt"] = group[
            "TransactionAmt"
        ].shift(-1).astype("float32")
        sequence[f"{prefix}_amt_to_mean"] = (
            sequence["TransactionAmt"] / mean_amount.replace(0, np.nan)
        ).astype("float32")
        sequence[f"{prefix}_amt_zscore"] = (
            (sequence["TransactionAmt"] - mean_amount)
            / std_amount.replace(0, np.nan)
        ).astype("float32")

        chain_feature_names = []
        if add_v307_chain and "V307" in sequence:
            previous_v307 = group["V307"].shift(1)
            next_v307 = group["V307"].shift(-1)
            previous_amount = group["TransactionAmt"].shift(1)
            previous_delta = sequence["V307"] - previous_v307
            next_delta = next_v307 - sequence["V307"]
            previous_error = previous_delta - previous_amount
            next_error = next_delta - sequence["TransactionAmt"]
            chain_feature_names = [
                f"{prefix}_V307_previous_delta",
                f"{prefix}_V307_previous_amount_error",
                f"{prefix}_V307_previous_amount_match",
                f"{prefix}_V307_next_delta",
                f"{prefix}_V307_next_amount_error",
                f"{prefix}_V307_next_amount_match",
            ]
            sequence[chain_feature_names[0]] = previous_delta.astype("float32")
            sequence[chain_feature_names[1]] = previous_error.astype("float32")
            sequence[chain_feature_names[2]] = (
                previous_error.abs().le(0.011)
                & previous_error.notna()
            ).astype("int8")
            sequence[chain_feature_names[3]] = next_delta.astype("float32")
            sequence[chain_feature_names[4]] = next_error.astype("float32")
            sequence[chain_feature_names[5]] = (
                next_error.abs().le(0.011) & next_error.notna()
            ).astype("int8")

        feature_names.extend(
            [
                f"{prefix}_count",
                f"{prefix}_order",
                f"{prefix}_order_from_end",
                f"{prefix}_previous_dt",
                f"{prefix}_next_dt",
                f"{prefix}_mean_dt",
                f"{prefix}_std_dt",
                f"{prefix}_median_dt",
                f"{prefix}_time_span",
                f"{prefix}_mean_amt",
                f"{prefix}_std_amt",
                f"{prefix}_median_amt",
                f"{prefix}_previous_amt",
                f"{prefix}_next_amt",
                f"{prefix}_amt_to_mean",
                f"{prefix}_amt_zscore",
                *chain_feature_names,
            ]
        )

    sequence = sequence.sort_values("_row_order", kind="stable")
    train_rows = len(train)
    for feature_name in feature_names:
        values = sequence[feature_name].to_numpy()
        train[feature_name] = values[:train_rows]
        test[feature_name] = values[train_rows:]


def _add_frequency_features(
    train,
    test,
    mode="selected",
    raw_columns=None,
):
    if mode == "selected":
        frequency_columns = SELECTED_FREQUENCY_COLUMNS
    elif mode in {"all-train", "all-joint"}:
        raw_frequency_columns = [
            column for column in (raw_columns or [])
            if column not in {TARGET, "TransactionID"}
        ]
        frequency_columns = list(
            dict.fromkeys(
                [*SELECTED_FREQUENCY_COLUMNS, *raw_frequency_columns]
            )
        )
    else:
        raise ValueError(f"Unknown frequency mode: {mode}")

    train_features = {}
    test_features = {}
    for column in frequency_columns:
        if column not in train:
            continue
        feature_name = f"{column}_freq"
        if mode == "selected":
            train_tokens = _as_token(train[column])
            counts = train_tokens.value_counts(dropna=False)
            frequencies = counts / len(train)
            train_frequency = train_tokens.map(frequencies).fillna(0)
            test_frequency = _as_token(test[column]).map(frequencies).fillna(0)
        else:
            combined = pd.concat(
                [train[column], test[column]],
                ignore_index=True,
                copy=False,
            )
            codes, _ = pd.factorize(combined, sort=False)
            # Reserve code zero for missing values, which factorize marks as -1.
            codes = codes.astype("int32", copy=False) + 1
            train_codes = codes[: len(train)]
            if mode == "all-joint":
                counts = np.bincount(codes)
                denominator = len(combined)
            else:
                counts = np.bincount(train_codes, minlength=codes.max() + 1)
                denominator = len(train)
            train_frequency = counts[train_codes] / denominator
            test_frequency = counts[codes[len(train) :]] / denominator

            # A constant train feature cannot be used by a supervised model.
            if np.unique(train_frequency).size < 2:
                continue

        train_features[feature_name] = np.asarray(
            train_frequency,
            dtype="float32",
        )
        test_features[feature_name] = np.asarray(
            test_frequency,
            dtype="float32",
        )

    return (
        pd.concat([train, pd.DataFrame(train_features, index=train.index)], axis=1),
        pd.concat([test, pd.DataFrame(test_features, index=test.index)], axis=1),
    )


def _downcast(frame):
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype("float32")

    for column in frame.select_dtypes(include=["int64"]).columns:
        if column == "TransactionID":
            frame[column] = frame[column].astype("int32")
        elif column == TARGET:
            frame[column] = frame[column].astype("int8")
        else:
            frame[column] = pd.to_numeric(frame[column], downcast="integer")


def build_features(
    train,
    test,
    giba_features=False,
    frequency_mode="selected",
    v307_chain_features=False,
):
    raw_frequency_columns = train.columns.tolist()
    _add_row_features(train)
    _add_row_features(test)
    if giba_features:
        _add_giba_uid_features(train)
        _add_giba_uid_features(test)
        _add_uid_sequence_features(
            train,
            test,
            add_v307_chain=v307_chain_features,
        )
    train, test = _add_frequency_features(
        train,
        test,
        mode=frequency_mode,
        raw_columns=raw_frequency_columns,
    )
    _downcast(train)
    _downcast(test)

    feature_columns = [
        column
        for column in train.columns
        if column not in {TARGET, "TransactionID"}
    ]
    test = test.reindex(columns=["TransactionID", *feature_columns])

    categorical_columns = train[feature_columns].select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    for column in categorical_columns:
        train[column] = _as_token(train[column])
        test[column] = _as_token(test[column])

    return train, test, feature_columns, categorical_columns

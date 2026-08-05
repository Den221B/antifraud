"""Построчные признаки, UID-цепочки и частоты, посчитанные только по train.

Код перенесён без изменений из проверенного прогона: обе модели, собранные
на этих функциях, совпали с эталонными деревом в дерево.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import TARGET



GIBA_UID_COLUMNS = ["uid_d1_email", "uid_card_addr_d1", "uid_card_addr_d1_email"]
GIBA_SEQUENCE_UID_COLUMNS = ["uid_d1_email", "uid_card_addr_d1_email"]
SELECTED_FREQUENCY_COLUMNS = [
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "DeviceInfo_family",
    "uid_card_addr", "uid_card_email", "uid_card_full", "uid_card_device", "uid_email_pair",
    *GIBA_UID_COLUMNS,
]


def read_and_merge(data_dir, split):
    transaction = pd.read_csv(Path(data_dir) / f"{split}_transaction.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    identity = pd.read_csv(Path(data_dir) / f"{split}_identity.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    return transaction.merge(identity, on="TransactionID", how="left")


def as_token(series):
    return series.astype("string").fillna("<MISSING>")


def as_integer_token(series):
    return series.round().astype("Int32").astype("string").fillna("<MISSING>")


def join_tokens(*series):
    result = as_token(series[0])
    for value in series[1:]:
        result = result.str.cat(as_token(value), sep="|")
    return result


def add_group_aggregates(frame, prefix, columns):
    if not columns:
        return
    values = frame[columns]
    frame[f"{prefix}_missing_count"] = values.isna().sum(axis=1).astype("int16")
    frame[f"{prefix}_mean"] = values.mean(axis=1).astype("float32")
    frame[f"{prefix}_std"] = values.std(axis=1).astype("float32")
    frame[f"{prefix}_min"] = values.min(axis=1).astype("float32")
    frame[f"{prefix}_max"] = values.max(axis=1).astype("float32")


def add_row_features(frame):
    transaction_dt = frame["TransactionDT"]
    transaction_day = transaction_dt / 86400
    amount = frame["TransactionAmt"]

    # календарные срезы
    frame["DT_day"] = np.floor(transaction_day).astype("int16")
    frame["DT_week"] = np.floor(transaction_day / 7).astype("int16")
    frame["DT_hour"] = ((transaction_dt // 3600) % 24).astype("int8")
    frame["DT_dayofweek"] = ((transaction_dt // 86400) % 7).astype("int8")

    # форма суммы: мошенники часто платят «некруглыми» суммами
    frame["TransactionAmt_log1p"] = np.log1p(amount).astype("float32")
    frame["TransactionAmt_cents"] = (np.round((amount - np.floor(amount)) * 100) % 100).astype("int8")
    frame["TransactionAmt_is_integer"] = (np.isclose(amount % 1, 0)).astype("int8")
    frame["TransactionAmt_is_round_10"] = (np.isclose(amount % 10, 0)).astype("int8")

    p_email, r_email = as_token(frame["P_emaildomain"]), as_token(frame["R_emaildomain"])
    frame["P_R_email_match"] = (p_email == r_email).astype("int8")
    frame["P_email_suffix"] = p_email.str.rsplit(".", n=1).str[-1]
    frame["R_email_suffix"] = r_email.str.rsplit(".", n=1).str[-1]

    if "DeviceInfo" in frame:
        frame["DeviceInfo_family"] = as_token(frame["DeviceInfo"]).str.split("/", n=1).str[0]
    if "id_31" in frame:
        frame["browser_family"] = as_token(frame["id_31"]).str.replace(r"[\d._-]+$", "", regex=True).str.strip()

    # грубые составные ключи «карта + адрес + почта + устройство»
    combos = {
        "uid_card_addr": ["card1", "addr1"],
        "uid_card_email": ["card1", "addr1", "P_emaildomain"],
        "uid_card_full": ["card1", "card2", "card3", "card5"],
        "uid_card_device": ["card1", "addr1", "DeviceInfo"],
        "uid_email_pair": ["P_emaildomain", "R_emaildomain"],
    }
    for name, columns in combos.items():
        frame[name] = as_token(frame[columns[0]])
        for column in columns[1:]:
            frame[name] = frame[name].str.cat(as_token(frame[column]), sep="|")

    # D — это дни «назад», поэтому разность с днём транзакции даёт стабильную дату события
    d_columns = [f"D{i}" for i in range(1, 16) if f"D{i}" in frame]
    for column in d_columns:
        frame[f"{column}_minus_day"] = (frame[column] - transaction_day).astype("float32")

    c_columns = [f"C{i}" for i in range(1, 15) if f"C{i}" in frame]
    v_columns = [f"V{i}" for i in range(1, 340) if f"V{i}" in frame]
    id_numeric = [c for c in frame.columns if c.startswith("id_") and pd.api.types.is_numeric_dtype(frame[c])]
    add_group_aggregates(frame, "C", c_columns)
    add_group_aggregates(frame, "D", d_columns)
    add_group_aggregates(frame, "V", v_columns)
    add_group_aggregates(frame, "id_numeric", id_numeric)
    frame["row_missing_count"] = frame.isna().sum(axis=1).astype("int16")


def add_giba_uid_features(frame):
    d1_origin_day = frame["DT_day"].astype("float32") - frame["D1"]
    frame["D1_origin_day"] = d1_origin_day.astype("float32")
    origin_token = as_integer_token(d1_origin_day)
    p_email = as_token(frame["P_emaildomain"])
    frame["uid_d1_email"] = origin_token.str.cat(p_email, sep="|")
    frame["uid_card_addr_d1"] = join_tokens(frame["card1"], frame["addr1"], origin_token)
    frame["uid_card_addr_d1_email"] = frame["uid_card_addr_d1"].str.cat(p_email, sep="|")


def add_uid_sequence_features(train, test):
    columns = ["TransactionID", "TransactionDT", "TransactionAmt", *GIBA_SEQUENCE_UID_COLUMNS]
    sequence = pd.concat([train[columns], test[columns]], ignore_index=True, copy=False)
    sequence["_row_order"] = np.arange(len(sequence), dtype="int32")
    sequence = sequence.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)

    feature_names = []
    for uid in GIBA_SEQUENCE_UID_COLUMNS:
        group = sequence.groupby(uid, sort=False, observed=True, dropna=False)
        prefix = f"{uid}_seq"
        sequence[f"{prefix}_count"] = group["TransactionID"].transform("size").astype("int32")
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
        gap_group = sequence.groupby(uid, sort=False, observed=True, dropna=False)[gap_column]
        sequence[f"{prefix}_mean_dt"] = gap_group.transform("mean").astype("float32")
        sequence[f"{prefix}_std_dt"] = gap_group.transform("std").astype("float32")
        sequence[f"{prefix}_median_dt"] = gap_group.transform("median").astype("float32")
        sequence.drop(columns=[gap_column], inplace=True)

        first_dt = group["TransactionDT"].transform("min")
        last_dt = group["TransactionDT"].transform("max")
        sequence[f"{prefix}_time_span"] = (last_dt - first_dt).astype("float32")

        mean_amount = group["TransactionAmt"].transform("mean")
        std_amount = group["TransactionAmt"].transform("std")
        sequence[f"{prefix}_mean_amt"] = mean_amount.astype("float32")
        sequence[f"{prefix}_std_amt"] = std_amount.astype("float32")
        sequence[f"{prefix}_median_amt"] = group["TransactionAmt"].transform("median").astype("float32")
        sequence[f"{prefix}_previous_amt"] = group["TransactionAmt"].shift(1).astype("float32")
        sequence[f"{prefix}_next_amt"] = group["TransactionAmt"].shift(-1).astype("float32")
        sequence[f"{prefix}_amt_to_mean"] = (
            sequence["TransactionAmt"] / mean_amount.replace(0, np.nan)
        ).astype("float32")
        sequence[f"{prefix}_amt_zscore"] = (
            (sequence["TransactionAmt"] - mean_amount) / std_amount.replace(0, np.nan)
        ).astype("float32")

        feature_names.extend([
            f"{prefix}_count", f"{prefix}_order", f"{prefix}_order_from_end",
            f"{prefix}_previous_dt", f"{prefix}_next_dt", f"{prefix}_mean_dt",
            f"{prefix}_std_dt", f"{prefix}_median_dt", f"{prefix}_time_span",
            f"{prefix}_mean_amt", f"{prefix}_std_amt", f"{prefix}_median_amt",
            f"{prefix}_previous_amt", f"{prefix}_next_amt", f"{prefix}_amt_to_mean",
            f"{prefix}_amt_zscore",
        ])

    sequence = sequence.sort_values("_row_order", kind="stable")
    train_rows = len(train)
    for name in feature_names:
        values = sequence[name].to_numpy()
        train[name] = values[:train_rows]
        test[name] = values[train_rows:]


def add_frequency_features(train, test):
    # частота считается ТОЛЬКО по train — иначе признак подглядывает в распределение теста
    train_features, test_features = {}, {}
    for column in SELECTED_FREQUENCY_COLUMNS:
        if column not in train:
            continue
        train_tokens = as_token(train[column])
        frequencies = train_tokens.value_counts(dropna=False) / len(train)
        train_features[f"{column}_freq"] = np.asarray(train_tokens.map(frequencies).fillna(0), dtype="float32")
        test_features[f"{column}_freq"] = np.asarray(
            as_token(test[column]).map(frequencies).fillna(0), dtype="float32"
        )
    return (
        pd.concat([train, pd.DataFrame(train_features, index=train.index)], axis=1),
        pd.concat([test, pd.DataFrame(test_features, index=test.index)], axis=1),
    )


def downcast(frame):
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype("float32")
    for column in frame.select_dtypes(include=["int64"]).columns:
        if column == "TransactionID":
            frame[column] = frame[column].astype("int32")
        elif column == TARGET:
            frame[column] = frame[column].astype("int8")
        else:
            frame[column] = pd.to_numeric(frame[column], downcast="integer")


def build_features(train, test):
    add_row_features(train)
    add_row_features(test)
    add_giba_uid_features(train)
    add_giba_uid_features(test)
    add_uid_sequence_features(train, test)
    train, test = add_frequency_features(train, test)
    downcast(train)
    downcast(test)

    feature_columns = [c for c in train.columns if c not in {TARGET, "TransactionID"}]
    test = test.reindex(columns=["TransactionID", *feature_columns])
    categorical_columns = train[feature_columns].select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    for column in categorical_columns:
        train[column] = as_token(train[column])
        test[column] = as_token(test[column])
    return train, test, feature_columns, categorical_columns


def convert_categories(train, test, categorical_columns):
    for column in categorical_columns:
        dtype = pd.CategoricalDtype(categories=pd.Index(train[column].dropna().astype("string").unique()))
        train[column] = train[column].astype("string").astype(dtype)
        test[column] = test[column].astype("string").astype(dtype)


def add_amount_patterns(frame):
    amount = frame["TransactionAmt"].astype("float64")
    values = {}
    for modulus in (50.0, 100.0, 200.0):
        remainder = np.mod(amount, modulus)
        values[f"TransactionAmt_mod_{int(modulus)}_is_zero"] = np.isclose(remainder, 0.0, atol=0.011).astype("int8")
        values[f"TransactionAmt_mod_{int(modulus)}_distance"] = np.minimum(remainder, modulus - remainder).astype("float32")
    additions = pd.DataFrame(values, index=frame.index)
    return pd.concat([frame, additions], axis=1), additions.columns.tolist()

"""Свёртка V-блоков, календарь и суммы, разбор устройства, распределения значений.

Сырые V1-V339 в модель не идут: на временном фолде они проигрывают
собственным сводкам по блокам (§2.3 ноутбука).

Код перенесён без изменений из проверенного прогона: обе модели, собранные
на этих функциях, совпали с эталонными деревом в дерево.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from .common import TOP_V_COLUMNS, V_BLOCKS
from .features_row import as_token



def row_vblock_features(frame):
    features = {}
    for start, end in V_BLOCKS:
        columns = [f"V{n}" for n in range(start, end + 1) if f"V{n}" in frame]
        if not columns:
            continue
        values = frame[columns]
        prefix = f"vblock_{start}_{end}"
        block_min, block_max = values.min(axis=1), values.max(axis=1)
        features[f"{prefix}_missing_rate"] = values.isna().mean(axis=1).astype("float32")
        features[f"{prefix}_mean"] = values.mean(axis=1).astype("float32")
        features[f"{prefix}_std"] = values.std(axis=1).astype("float32")
        features[f"{prefix}_min"] = block_min.astype("float32")
        features[f"{prefix}_max"] = block_max.astype("float32")
        features[f"{prefix}_range"] = (block_max - block_min).astype("float32")
        features[f"{prefix}_nunique"] = values.nunique(axis=1).astype("int16")
        features[f"{prefix}_zero_count"] = values.eq(0).sum(axis=1).astype("int16")
        features[f"{prefix}_one_count"] = values.eq(1).sum(axis=1).astype("int16")
        features[f"{prefix}_two_count"] = values.eq(2).sum(axis=1).astype("int16")
    return pd.DataFrame(features, index=frame.index)


def add_vblock_user_features(train, inference, train_components, inference_components):
    train_blocks = row_vblock_features(train).reset_index(drop=True)
    inference_blocks = row_vblock_features(inference).reset_index(drop=True)
    block_features = train_blocks.columns.tolist()

    c_columns = [f"C{i}" for i in range(1, 15) if f"C{i}" in train]
    d_columns = [f"D{i}" for i in range(1, 16) if f"D{i}" in train]
    v_columns = [c for c in TOP_V_COLUMNS if c in train]
    aggregate_columns = [*c_columns, *d_columns, *v_columns]
    combined = pd.concat([train[aggregate_columns], inference[aggregate_columns]],
                         ignore_index=True, copy=False)
    combined["_component"] = np.concatenate([train_components, inference_components])
    group = combined.groupby("_component", sort=False, observed=True)
    component_count = group["_component"].transform("size")

    aggregates = {}
    for column in [*c_columns, *d_columns]:
        values = combined[column]
        column_group = group[column]
        mean, std = column_group.transform("mean"), column_group.transform("std")
        minimum, maximum = column_group.transform("min"), column_group.transform("max")
        nunique = column_group.transform("nunique")
        prefix = f"wide_user_{column}"
        aggregates[f"{prefix}_mean"] = mean.astype("float32")
        aggregates[f"{prefix}_std"] = std.astype("float32")
        aggregates[f"{prefix}_range"] = (maximum - minimum).astype("float32")
        aggregates[f"{prefix}_nunique_rate"] = (nunique / component_count).astype("float32")
        aggregates[f"{prefix}_diff_mean"] = (values - mean).astype("float32")
        aggregates[f"{prefix}_zscore"] = ((values - mean) / std.replace(0, np.nan)).astype("float32")

    for column in v_columns:
        values = combined[column]
        column_group = group[column]
        mean, std = column_group.transform("mean"), column_group.transform("std")
        prefix = f"wide_user_{column}"
        aggregates[f"{prefix}_mean"] = mean.astype("float32")
        aggregates[f"{prefix}_std"] = std.astype("float32")
        aggregates[f"{prefix}_zscore"] = ((values - mean) / std.replace(0, np.nan)).astype("float32")

    aggregate_frame = pd.DataFrame(aggregates)
    train_rows = len(train)
    train_aggregate = aggregate_frame.iloc[:train_rows].reset_index(drop=True)
    inference_aggregate = aggregate_frame.iloc[train_rows:].reset_index(drop=True)

    combined_blocks = pd.concat([train_blocks, inference_blocks], ignore_index=True, copy=False)
    combined_blocks["_component"] = np.concatenate([train_components, inference_components])
    missing_group = combined_blocks.groupby("_component", sort=False, observed=True)
    missing = {}
    for column in [c for c in block_features if c.endswith("_missing_rate")]:
        mean = missing_group[column].transform("mean")
        missing[f"wide_user_{column}_mean"] = mean.astype("float32")
        missing[f"wide_user_{column}_diff"] = (combined_blocks[column] - mean).astype("float32")
    missing_frame = pd.DataFrame(missing)
    train_missing = missing_frame.iloc[:train_rows].reset_index(drop=True)
    inference_missing = missing_frame.iloc[train_rows:].reset_index(drop=True)

    train_new = pd.concat([train_blocks, train_aggregate, train_missing], axis=1)
    inference_new = pd.concat([inference_blocks, inference_aggregate, inference_missing], axis=1)
    stats = {"v_blocks": len(V_BLOCKS), "block_features": len(block_features),
             "c_columns": len(c_columns), "d_columns": len(d_columns),
             "selected_v_columns": list(v_columns),
             "aggregate_features": len(train_aggregate.columns),
             "missing_profile_features": len(train_missing.columns),
             "total_features": len(train_new.columns)}
    return (pd.concat([train.reset_index(drop=True), train_new], axis=1),
            pd.concat([inference.reset_index(drop=True), inference_new], axis=1),
            train_new.columns.tolist(), stats)


START_DATE = pd.Timestamp("2017-12-01")
NEXT_BEHAVIOR_COLUMNS = ("ProductCD", "card2", "card5", "addr1", "P_emaildomain",
                         "R_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33")
COOCCURRENCE_PAIRS = (("card1", "addr1"), ("card1", "P_emaildomain"), ("card1", "DeviceInfo"),
                      ("P_emaildomain", "DeviceInfo"), ("id_30", "id_31"), ("card4", "card6"))


def joined_tokens(frame, columns):
    result = as_token(frame[columns[0]])
    for column in columns[1:]:
        series = frame[column]
        if column.endswith("origin_day"):
            series = series.round()
        result = result.str.cat(as_token(series), sep="|")
    return result


def attach_features(train, inference, features):
    features = features.reset_index(drop=True)
    train_rows = len(train)
    return (
        pd.concat([train.reset_index(drop=True), features.iloc[:train_rows]], axis=1),
        pd.concat([inference.reset_index(drop=True), features.iloc[train_rows:].reset_index(drop=True)], axis=1),
        features.columns.tolist(),
    )


def calendar_amount_rows(frame):
    dt = frame["TransactionDT"].astype("float64")
    day = dt / 86_400.0
    dates = START_DATE + pd.to_timedelta(dt, unit="s")
    amount = frame["TransactionAmt"].astype("float64")
    values = {
        "calendar_month": dates.dt.month.astype("int8"),
        "calendar_dayofmonth": dates.dt.day.astype("int8"),
        "calendar_weekofyear": dates.dt.isocalendar().week.astype("int16"),
        "calendar_is_weekend": dates.dt.dayofweek.ge(5).astype("int8"),
        "calendar_minute": dates.dt.minute.astype("int8"),
        "calendar_seconds_into_day": np.mod(dt, 86_400).astype("float32"),
        "calendar_hour_sin": np.sin(2 * np.pi * dt / 86_400).astype("float32"),
        "calendar_hour_cos": np.cos(2 * np.pi * dt / 86_400).astype("float32"),
        "calendar_week_sin": np.sin(2 * np.pi * day / 7).astype("float32"),
        "calendar_week_cos": np.cos(2 * np.pi * day / 7).astype("float32"),
        "amount_log10": np.log10(amount.clip(lower=0) + 0.01).astype("float32"),
        "amount_magnitude": np.floor(np.log10(amount.clip(lower=0) + 0.01)).astype("int8"),
        "amount_fraction_1000": np.mod(np.rint(amount * 1000), 1000).astype("int16"),
        "amount_nearest_integer_distance": np.abs(amount - np.rint(amount)).astype("float32"),
    }
    cents = np.mod(np.rint(amount * 100), 100).astype("int16")
    values["amount_cent_ending"] = cents
    for ending in (0, 25, 50, 95, 99):
        values[f"amount_cent_is_{ending:02d}"] = cents.eq(ending).astype("int8")
    for modulus in (1.0, 5.0, 10.0, 25.0, 500.0):
        suffix = str(modulus).replace(".", "p")
        remainder = np.mod(amount, modulus)
        values[f"amount_mod_{suffix}_distance"] = np.minimum(remainder, modulus - remainder).astype("float32")

    origins = {}
    for number in range(1, 16):
        column = f"D{number}"
        if column not in frame:
            continue
        origin = day - frame[column].astype("float64")
        rounded = np.rint(origin)
        origins[column] = rounded
        values[f"{column}_origin_round"] = rounded.astype("float32")
        values[f"{column}_origin_week"] = np.floor(rounded / 7).astype("float32")
        values[f"{column}_origin_round_distance"] = np.abs(origin - rounded).astype("float32")
    if origins:
        origin_frame = pd.DataFrame(origins, index=frame.index)
        values["D_origin_mean"] = origin_frame.mean(axis=1).astype("float32")
        values["D_origin_std"] = origin_frame.std(axis=1).astype("float32")
        values["D_origin_range"] = (origin_frame.max(axis=1) - origin_frame.min(axis=1)).astype("float32")
        values["D_origin_nunique"] = origin_frame.nunique(axis=1).astype("int8")
        if "D1" in origin_frame:
            values["D_origin_matches_D1"] = origin_frame.sub(
                origin_frame["D1"], axis=0).abs().le(1).sum(axis=1).astype("int8")

    for column in [f"C{i}" for i in range(1, 15) if f"C{i}" in frame]:
        values[f"{column}_log1p"] = np.log1p(frame[column].clip(lower=0)).astype("float32")
    for first, second in (("C1", "C2"), ("C1", "C14"), ("C2", "C14"), ("C5", "C9"),
                          ("C6", "C8"), ("C10", "C13"), ("C11", "C13"), ("C12", "C13")):
        if first in frame and second in frame:
            values[f"{first}_to_{second}"] = (frame[first] / frame[second].replace(0, np.nan)).astype("float32")
    return pd.DataFrame(values, index=frame.index)


def amount_group_features(combined):
    amount = combined["TransactionAmt"].astype("float64")
    amount_cents = np.rint(amount * 100).astype("int64")
    key_specs = {
        "card1": ("card1",), "card1_addr1": ("card1", "addr1"),
        "card1_product": ("card1", "ProductCD"), "card1_card5": ("card1", "card5"),
        "card1_register": ("card1", "D1_origin_day"), "product_card4": ("ProductCD", "card4"),
    }
    values = {}
    for name, columns in key_specs.items():
        helper = pd.DataFrame({"key": joined_tokens(combined, columns),
                               "amount": amount, "amount_cents": amount_cents})
        group = helper.groupby("key", sort=False, observed=True)
        count = group["amount"].transform("size")
        mean, std = group["amount"].transform("mean"), group["amount"].transform("std")
        median = group["amount"].transform("median")
        same_count = helper.groupby(["key", "amount_cents"], sort=False,
                                    observed=True)["amount"].transform("size")
        prefix = f"amount_group_{name}"
        values[f"{prefix}_log_count"] = np.log1p(count).astype("float32")
        values[f"{prefix}_mean"] = mean.astype("float32")
        values[f"{prefix}_std"] = std.astype("float32")
        values[f"{prefix}_median"] = median.astype("float32")
        values[f"{prefix}_diff_mean"] = (amount - mean).astype("float32")
        values[f"{prefix}_zscore"] = ((amount - mean) / std.replace(0, np.nan)).astype("float32")
        values[f"{prefix}_same_amount_fraction"] = (same_count / count).astype("float32")
    return pd.DataFrame(values, index=combined.index)


def add_calendar_amount_features(train, inference):
    train_rows_frame = calendar_amount_rows(train)
    inference_rows_frame = calendar_amount_rows(inference)
    combined = pd.concat([train, inference], ignore_index=True, copy=False)
    group_features = amount_group_features(combined)
    del combined
    gc.collect()
    features = pd.concat([
        pd.concat([train_rows_frame, inference_rows_frame], ignore_index=True),
        group_features.reset_index(drop=True),
    ], axis=1)
    train, inference, names = attach_features(train, inference, features)
    return train, inference, names, {"features": len(names), "amount_group_keys": 6,
                                     "start_date": str(START_DATE.date())}


def device_brand(device):
    lower = as_token(device).str.lower()
    brand = lower.str.split(r"[/ ]", n=1, regex=True).str[0]
    for pattern, replacement in (
        (r"windows|trident|^rv:", "windows"), (r"ios|iphone|ipad", "apple_ios"),
        (r"macos|mac os", "apple_mac"), (r"^sm-|^gt-|^sch-|^sgh-", "samsung"),
        (r"huawei|^ale-|^ane-|^bla-|^vns-", "huawei"), (r"redmi|xiaomi|^mi ", "xiaomi"),
        (r"^lg-|^lg$", "lg"), (r"moto|motorola", "motorola"), (r"pixel|nexus", "google"),
        (r"htc", "htc"), (r"zte", "zte"),
    ):
        brand = brand.mask(lower.str.contains(pattern, regex=True, na=False), replacement)
    return brand.astype("string")


def identity_rows(frame):
    values, categorical = {}, []
    device = as_token(frame.get("DeviceInfo", pd.Series(index=frame.index))).str.lower()
    values["identity_device_brand"] = device_brand(device)
    categorical.append("identity_device_brand")
    values["identity_device_is_mobile"] = device.str.contains(
        r"android|ios|iphone|ipad|^sm-|moto|huawei|redmi|xiaomi", regex=True, na=False).astype("int8")
    values["identity_device_has_build"] = device.str.contains("build", regex=False, na=False).astype("int8")

    os_value = as_token(frame.get("id_30", pd.Series(index=frame.index))).str.lower()
    values["identity_os_family"] = os_value.str.replace(r"[\d._-]+.*$", "", regex=True).str.strip()
    categorical.append("identity_os_family")
    values["identity_os_major"] = pd.to_numeric(
        os_value.str.extract(r"(\d+)", expand=False), errors="coerce").astype("float32")

    browser = as_token(frame.get("id_31", pd.Series(index=frame.index))).str.lower()
    values["identity_browser_family"] = browser.str.replace(r"[\d._-]+.*$", "", regex=True).str.strip()
    categorical.append("identity_browser_family")
    values["identity_browser_major"] = pd.to_numeric(
        browser.str.extract(r"(\d+)", expand=False), errors="coerce").astype("float32")
    values["identity_browser_is_mobile"] = browser.str.contains(
        r"mobile|android|ios", regex=True, na=False).astype("int8")
    values["identity_browser_is_generic"] = browser.str.contains(
        "generic", regex=False, na=False).astype("int8")

    screen = as_token(frame.get("id_33", pd.Series(index=frame.index))).str.lower()
    dimensions = screen.str.extract(r"^(\d+)x(\d+)$")
    width = pd.to_numeric(dimensions[0], errors="coerce")
    height = pd.to_numeric(dimensions[1], errors="coerce")
    values["identity_screen_width"] = width.astype("float32")
    values["identity_screen_height"] = height.astype("float32")
    values["identity_screen_pixels"] = (width * height).astype("float32")
    values["identity_screen_aspect"] = (width / height.replace(0, np.nan)).astype("float32")
    screen_known = width.notna() & height.notna()
    values["identity_screen_known"] = screen_known.astype("int8")
    values["identity_screen_portrait"] = (width.lt(height) & screen_known).fillna(False).astype("int8")

    match_status = as_token(frame.get("id_34", pd.Series(index=frame.index))).str.extract(
        r"(-?\d+)$", expand=False)
    values["identity_match_status"] = pd.to_numeric(match_status, errors="coerce").astype("float32")

    m_columns = [f"M{i}" for i in range(1, 10) if f"M{i}" in frame]
    if m_columns:
        m_values = frame[m_columns].astype("string")
        values["identity_M_true_count"] = m_values.eq("T").sum(axis=1).astype("int8")
        values["identity_M_false_count"] = m_values.eq("F").sum(axis=1).astype("int8")
        values["identity_M_missing_count"] = m_values.isna().sum(axis=1).astype("int8")
        values["identity_M_pattern"] = m_values.fillna("N").agg("".join, axis=1)
        categorical.append("identity_M_pattern")

    id_columns = [f"id_{i:02d}" for i in range(1, 39) if f"id_{i:02d}" in frame]
    for block_number, start in enumerate((0, 19)):
        block = id_columns[start: start + 19]
        if not block:
            continue
        present = frame[block].notna().to_numpy(dtype="uint32")
        weights = np.left_shift(np.uint32(1), np.arange(len(block), dtype="uint32"))
        values[f"identity_present_mask_{block_number}"] = (present @ weights).astype("uint32")
    if id_columns:
        values["identity_present_count"] = frame[id_columns].notna().sum(axis=1).astype("int8")

    d_columns = [f"D{i}" for i in range(1, 16) if f"D{i}" in frame]
    if d_columns:
        present = frame[d_columns].notna().to_numpy(dtype="uint16")
        weights = np.left_shift(np.uint16(1), np.arange(len(d_columns), dtype="uint16"))
        values["identity_D_present_mask"] = (present @ weights).astype("uint16")

    p_email, r_email = as_token(frame["P_emaildomain"]), as_token(frame["R_emaildomain"])
    email_state = pd.Series("different", index=frame.index, dtype="string")
    both_missing = p_email.eq("<MISSING>") & r_email.eq("<MISSING>")
    one_missing = p_email.eq("<MISSING>") ^ r_email.eq("<MISSING>")
    email_state = email_state.mask(p_email.eq(r_email) & ~both_missing, "same")
    email_state = email_state.mask(one_missing, "one_missing")
    email_state = email_state.mask(both_missing, "both_missing")
    values["identity_email_state"] = email_state
    categorical.append("identity_email_state")
    return pd.DataFrame(values, index=frame.index), categorical


def add_identity_features(train, inference):
    train_features, train_categorical = identity_rows(train)
    inference_features, inference_categorical = identity_rows(inference)
    assert train_categorical == inference_categorical
    features = pd.concat([train_features, inference_features], ignore_index=True)
    train, inference, names = attach_features(train, inference, features)
    return train, inference, names, train_categorical, {"features": len(names)}


def distribution_features(group_codes, values, prefix):
    value_codes = pd.factorize(as_token(values), sort=False)[0]
    helper = pd.DataFrame({"group": group_codes, "value": value_codes.astype("int32", copy=False)})
    group_count = helper.groupby("group", sort=False)["value"].transform("size")
    pair_count = helper.groupby(["group", "value"], sort=False)["value"].transform("size")
    unique_count = helper.groupby("group", sort=False)["value"].transform("nunique")
    mode_count = pair_count.groupby(helper["group"], sort=False).transform("max")
    share = pair_count / group_count

    pair_frame = helper.groupby(["group", "value"], sort=False).size().rename("count").reset_index()
    totals = pair_frame.groupby("group", sort=False)["count"].transform("sum")
    probability = pair_frame["count"] / totals
    pair_frame["entropy_part"] = -probability * np.log(probability)
    entropy = pair_frame.groupby("group", sort=False)["entropy_part"].sum()
    entropy_rows = helper["group"].map(entropy)
    normalized_entropy = entropy_rows / np.log(unique_count.clip(lower=2))
    return {
        f"{prefix}_nunique": unique_count.to_numpy(dtype="int16"),
        f"{prefix}_current_share": share.to_numpy(dtype="float32"),
        f"{prefix}_surprise": (-np.log(share.clip(lower=1e-8))).to_numpy(dtype="float32"),
        f"{prefix}_mode_share": (mode_count / group_count).to_numpy(dtype="float32"),
        f"{prefix}_is_mode": pair_count.eq(mode_count).to_numpy(dtype="int8"),
        f"{prefix}_entropy": entropy_rows.to_numpy(dtype="float32"),
        f"{prefix}_normalized_entropy": normalized_entropy.to_numpy(dtype="float32"),
    }


def cooccurrence_features(combined, first, second):
    first_codes = pd.factorize(as_token(combined[first]), sort=False)[0]
    second_codes = pd.factorize(as_token(combined[second]), sort=False)[0]
    helper = pd.DataFrame({"first": first_codes.astype("int32", copy=False),
                           "second": second_codes.astype("int32", copy=False)})
    first_count = helper.groupby("first", sort=False)["second"].transform("size")
    second_count = helper.groupby("second", sort=False)["first"].transform("size")
    pair_count = helper.groupby(["first", "second"], sort=False)["first"].transform("size")
    first_nunique = helper.groupby("first", sort=False)["second"].transform("nunique")
    second_nunique = helper.groupby("second", sort=False)["first"].transform("nunique")
    prefix = f"cooc_{first}_{second}"
    return {
        f"{prefix}_log_pair_count": np.log1p(pair_count).to_numpy(dtype="float32"),
        f"{prefix}_given_first": (pair_count / first_count).to_numpy(dtype="float32"),
        f"{prefix}_given_second": (pair_count / second_count).to_numpy(dtype="float32"),
        f"{prefix}_first_nunique": first_nunique.to_numpy(dtype="int32"),
        f"{prefix}_second_nunique": second_nunique.to_numpy(dtype="int32"),
    }


def add_behavior_distribution_features(train, inference, train_components, inference_components):
    columns = sorted(set(NEXT_BEHAVIOR_COLUMNS)
                     | {c for pair in COOCCURRENCE_PAIRS for c in pair}
                     | {"uid_card_addr_d1_email"})
    combined = pd.concat([train[columns], inference[columns]], ignore_index=True, copy=False)
    strict_codes = pd.factorize(as_token(combined["uid_card_addr_d1_email"]), sort=False)[0]
    group_specs = {
        "component": np.concatenate([train_components, inference_components]).astype("int32", copy=False),
        "strict_uid": strict_codes.astype("int32", copy=False),
    }
    values = {}
    for group_name, group_codes in group_specs.items():
        for column in NEXT_BEHAVIOR_COLUMNS:
            values.update(distribution_features(group_codes, combined[column],
                                                f"behavior_{group_name}_{column}"))
    for first, second in COOCCURRENCE_PAIRS:
        values.update(cooccurrence_features(combined, first, second))
    features = pd.DataFrame(values)
    del combined
    gc.collect()
    train, inference, names = attach_features(train, inference, features)
    return train, inference, names, {"features": len(names),
                                     "behavior_columns": list(NEXT_BEHAVIOR_COLUMNS),
                                     "cooccurrence_pairs": [list(p) for p in COOCCURRENCE_PAIRS]}

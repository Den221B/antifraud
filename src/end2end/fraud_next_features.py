"""Additional target-free feature families for the fraud temporal ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd


START_DATE = pd.Timestamp("2017-12-01")
VELOCITY_WINDOWS = (
    ("1h", 3_600.0),
    ("6h", 6 * 3_600.0),
    ("1d", 86_400.0),
    ("7d", 7 * 86_400.0),
    ("30d", 30 * 86_400.0),
)
VELOCITY_KEYS = (
    "uid_card_addr_d1_email",
    "uid_d1_email",
    "uid_card_addr",
)
BEHAVIOR_COLUMNS = (
    "ProductCD",
    "card2",
    "card5",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
)
COOCCURRENCE_PAIRS = (
    ("card1", "addr1"),
    ("card1", "P_emaildomain"),
    ("card1", "DeviceInfo"),
    ("P_emaildomain", "DeviceInfo"),
    ("id_30", "id_31"),
    ("card4", "card6"),
)


def _tokens(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>")


def _joined_tokens(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = _tokens(frame[columns[0]])
    for column in columns[1:]:
        values = frame[column]
        if column.endswith("origin_day"):
            values = values.round()
        result = result.str.cat(_tokens(values), sep="|")
    return result


def _attach_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    features = features.reset_index(drop=True)
    train_rows = len(train)
    names = features.columns.tolist()
    return (
        pd.concat(
            [train.reset_index(drop=True), features.iloc[:train_rows]],
            axis=1,
        ),
        pd.concat(
            [
                inference.reset_index(drop=True),
                features.iloc[train_rows:].reset_index(drop=True),
            ],
            axis=1,
        ),
        names,
    )


def _velocity_arrays(
    group_codes: np.ndarray,
    transaction_dt: np.ndarray,
    transaction_id: np.ndarray,
    amount: np.ndarray,
) -> dict[str, np.ndarray]:
    row_count = len(group_codes)
    result: dict[str, np.ndarray] = {
        "prior_count": np.zeros(row_count, dtype="int32"),
        "seconds_since_previous": np.full(row_count, np.nan, dtype="float32"),
        "seconds_since_first": np.full(row_count, np.nan, dtype="float32"),
        "prior_amount_mean": np.full(row_count, np.nan, dtype="float32"),
        "prior_amount_std": np.full(row_count, np.nan, dtype="float32"),
        "amount_zscore_prior": np.full(row_count, np.nan, dtype="float32"),
        "amount_diff_previous": np.full(row_count, np.nan, dtype="float32"),
    }
    for name, _ in VELOCITY_WINDOWS:
        result[f"count_last_{name}"] = np.zeros(row_count, dtype="int32")
    for name in ("1d", "7d"):
        result[f"amount_sum_last_{name}"] = np.zeros(
            row_count, dtype="float32"
        )
        result[f"amount_mean_last_{name}"] = np.full(
            row_count, np.nan, dtype="float32"
        )

    order = np.lexsort((transaction_id, transaction_dt, group_codes))
    sorted_codes = group_codes[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True]
    )

    for group_start, group_end in zip(boundaries[:-1], boundaries[1:]):
        rows = order[group_start:group_end]
        dt = transaction_dt[rows]
        amt = amount[rows]
        size = len(rows)
        amount_valid = np.isfinite(amt)
        amount_zeroed = np.where(amount_valid, amt, 0.0)
        cumulative_amount = np.r_[0.0, np.cumsum(amount_zeroed)]
        cumulative_square = np.r_[0.0, np.cumsum(amount_zeroed**2)]
        cumulative_valid = np.r_[0, np.cumsum(amount_valid.astype("int32"))]
        left = np.zeros(len(VELOCITY_WINDOWS), dtype="int32")

        bucket_start = 0
        while bucket_start < size:
            bucket_end = bucket_start + 1
            while bucket_end < size and dt[bucket_end] == dt[bucket_start]:
                bucket_end += 1
            current_rows = rows[bucket_start:bucket_end]
            prior_count = bucket_start
            result["prior_count"][current_rows] = prior_count

            if prior_count:
                result["seconds_since_previous"][current_rows] = (
                    dt[bucket_start] - dt[bucket_start - 1]
                )
                result["seconds_since_first"][current_rows] = (
                    dt[bucket_start] - dt[0]
                )
                valid_count = cumulative_valid[bucket_start]
                if valid_count:
                    prior_sum = cumulative_amount[bucket_start]
                    prior_mean = prior_sum / valid_count
                    prior_variance = max(
                        cumulative_square[bucket_start] / valid_count
                        - prior_mean**2,
                        0.0,
                    )
                    prior_std = np.sqrt(prior_variance)
                    result["prior_amount_mean"][current_rows] = prior_mean
                    result["prior_amount_std"][current_rows] = prior_std
                    if prior_std > 1e-9:
                        result["amount_zscore_prior"][current_rows] = (
                            amt[bucket_start:bucket_end] - prior_mean
                        ) / prior_std
                if amount_valid[bucket_start - 1]:
                    result["amount_diff_previous"][current_rows] = (
                        amt[bucket_start:bucket_end] - amt[bucket_start - 1]
                    )

            for window_index, (name, seconds) in enumerate(VELOCITY_WINDOWS):
                cutoff = dt[bucket_start] - seconds
                while left[window_index] < bucket_start and (
                    dt[left[window_index]] < cutoff
                ):
                    left[window_index] += 1
                window_start = int(left[window_index])
                count = bucket_start - window_start
                result[f"count_last_{name}"][current_rows] = count
                if name in {"1d", "7d"}:
                    valid_count = (
                        cumulative_valid[bucket_start]
                        - cumulative_valid[window_start]
                    )
                    amount_sum = (
                        cumulative_amount[bucket_start]
                        - cumulative_amount[window_start]
                    )
                    result[f"amount_sum_last_{name}"][current_rows] = amount_sum
                    if valid_count:
                        result[f"amount_mean_last_{name}"][current_rows] = (
                            amount_sum / valid_count
                        )
            bucket_start = bucket_end
    return result


def add_velocity_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    required = {
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        *VELOCITY_KEYS,
    }
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Missing velocity columns: {missing}")
    columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        *VELOCITY_KEYS,
    ]
    combined = pd.concat(
        [train[columns], inference[columns]], ignore_index=True, copy=False
    )
    transaction_dt = combined["TransactionDT"].to_numpy(dtype="float64")
    transaction_id = combined["TransactionID"].to_numpy(dtype="int64")
    amount = combined["TransactionAmt"].to_numpy(dtype="float64")
    feature_values: dict[str, np.ndarray] = {}
    for key in VELOCITY_KEYS:
        print(f"Building causal velocity for {key}...", flush=True)
        codes, _ = pd.factorize(_tokens(combined[key]), sort=False)
        values = _velocity_arrays(
            codes.astype("int32", copy=False),
            transaction_dt,
            transaction_id,
            amount,
        )
        feature_values.update(
            {f"velocity_{key}_{name}": value for name, value in values.items()}
        )
    features = pd.DataFrame(feature_values)
    train, inference, names = _attach_features(train, inference, features)
    return train, inference, names, {
        "keys": list(VELOCITY_KEYS),
        "windows": [name for name, _ in VELOCITY_WINDOWS],
        "features": len(names),
    }


def _calendar_amount_rows(frame: pd.DataFrame) -> pd.DataFrame:
    dt = frame["TransactionDT"].astype("float64")
    day = dt / 86_400.0
    dates = START_DATE + pd.to_timedelta(dt, unit="s")
    amount = frame["TransactionAmt"].astype("float64")
    values: dict[str, pd.Series | np.ndarray] = {
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
        "amount_magnitude": np.floor(
            np.log10(amount.clip(lower=0) + 0.01)
        ).astype("int8"),
        "amount_fraction_1000": np.mod(np.rint(amount * 1000), 1000).astype(
            "int16"
        ),
        "amount_nearest_integer_distance": np.abs(amount - np.rint(amount)).astype(
            "float32"
        ),
    }
    cents = np.mod(np.rint(amount * 100), 100).astype("int16")
    values["amount_cent_ending"] = cents
    for ending in (0, 25, 50, 95, 99):
        values[f"amount_cent_is_{ending:02d}"] = cents.eq(ending).astype("int8")
    for modulus in (1.0, 5.0, 10.0, 25.0, 500.0):
        suffix = str(modulus).replace(".", "p")
        remainder = np.mod(amount, modulus)
        values[f"amount_mod_{suffix}_distance"] = np.minimum(
            remainder, modulus - remainder
        ).astype("float32")

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
        values[f"{column}_origin_round_distance"] = np.abs(
            origin - rounded
        ).astype("float32")
    if origins:
        origin_frame = pd.DataFrame(origins, index=frame.index)
        values["D_origin_mean"] = origin_frame.mean(axis=1).astype("float32")
        values["D_origin_std"] = origin_frame.std(axis=1).astype("float32")
        values["D_origin_range"] = (
            origin_frame.max(axis=1) - origin_frame.min(axis=1)
        ).astype("float32")
        values["D_origin_nunique"] = origin_frame.nunique(axis=1).astype("int8")
        if "D1" in origin_frame:
            values["D_origin_matches_D1"] = origin_frame.sub(
                origin_frame["D1"], axis=0
            ).abs().le(1).sum(axis=1).astype("int8")

    c_columns = [f"C{i}" for i in range(1, 15) if f"C{i}" in frame]
    for column in c_columns:
        values[f"{column}_log1p"] = np.log1p(
            frame[column].clip(lower=0)
        ).astype("float32")
    for first, second in (
        ("C1", "C2"),
        ("C1", "C14"),
        ("C2", "C14"),
        ("C5", "C9"),
        ("C6", "C8"),
        ("C10", "C13"),
        ("C11", "C13"),
        ("C12", "C13"),
    ):
        if first in frame and second in frame:
            values[f"{first}_to_{second}"] = (
                frame[first] / frame[second].replace(0, np.nan)
            ).astype("float32")
    return pd.DataFrame(values, index=frame.index)


def _amount_group_features(combined: pd.DataFrame) -> pd.DataFrame:
    amount = combined["TransactionAmt"].astype("float64")
    amount_cents = np.rint(amount * 100).astype("int64")
    key_specs = {
        "card1": ("card1",),
        "card1_addr1": ("card1", "addr1"),
        "card1_product": ("card1", "ProductCD"),
        "card1_card5": ("card1", "card5"),
        "card1_register": ("card1", "D1_origin_day"),
        "product_card4": ("ProductCD", "card4"),
    }
    values: dict[str, pd.Series] = {}
    for name, columns in key_specs.items():
        key = _joined_tokens(combined, columns)
        helper = pd.DataFrame(
            {"key": key, "amount": amount, "amount_cents": amount_cents}
        )
        group = helper.groupby("key", sort=False, observed=True)
        count = group["amount"].transform("size")
        mean = group["amount"].transform("mean")
        std = group["amount"].transform("std")
        median = group["amount"].transform("median")
        same_count = helper.groupby(
            ["key", "amount_cents"], sort=False, observed=True
        )["amount"].transform("size")
        prefix = f"amount_group_{name}"
        values[f"{prefix}_log_count"] = np.log1p(count).astype("float32")
        values[f"{prefix}_mean"] = mean.astype("float32")
        values[f"{prefix}_std"] = std.astype("float32")
        values[f"{prefix}_median"] = median.astype("float32")
        values[f"{prefix}_diff_mean"] = (amount - mean).astype("float32")
        values[f"{prefix}_zscore"] = (
            (amount - mean) / std.replace(0, np.nan)
        ).astype("float32")
        values[f"{prefix}_same_amount_fraction"] = (
            same_count / count
        ).astype("float32")
    return pd.DataFrame(values, index=combined.index)


def add_calendar_amount_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    print("Building calendar, D-origin and amount fingerprints...", flush=True)
    train_rows = _calendar_amount_rows(train)
    inference_rows = _calendar_amount_rows(inference)
    combined = pd.concat([train, inference], ignore_index=True, copy=False)
    print("Building amount aggregates for stable entity keys...", flush=True)
    group_features = _amount_group_features(combined)
    features = pd.concat(
        [
            pd.concat([train_rows, inference_rows], ignore_index=True),
            group_features.reset_index(drop=True),
        ],
        axis=1,
    )
    train, inference, names = _attach_features(train, inference, features)
    return train, inference, names, {
        "features": len(names),
        "amount_group_keys": 6,
        "start_date": str(START_DATE.date()),
    }


def _device_brand(device: pd.Series) -> pd.Series:
    lower = _tokens(device).str.lower()
    brand = lower.str.split(r"[/ ]", n=1, regex=True).str[0]
    mappings = (
        (r"windows|trident|^rv:", "windows"),
        (r"ios|iphone|ipad", "apple_ios"),
        (r"macos|mac os", "apple_mac"),
        (r"^sm-|^gt-|^sch-|^sgh-", "samsung"),
        (r"huawei|^ale-|^ane-|^bla-|^vns-", "huawei"),
        (r"redmi|xiaomi|^mi ", "xiaomi"),
        (r"^lg-|^lg$", "lg"),
        (r"moto|motorola", "motorola"),
        (r"pixel|nexus", "google"),
        (r"htc", "htc"),
        (r"zte", "zte"),
    )
    for pattern, replacement in mappings:
        brand = brand.mask(lower.str.contains(pattern, regex=True, na=False), replacement)
    return brand.astype("string")


def _identity_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    values: dict[str, pd.Series | np.ndarray] = {}
    categorical = []
    device = _tokens(frame.get("DeviceInfo", pd.Series(index=frame.index))).str.lower()
    device_brand = _device_brand(device)
    values["identity_device_brand"] = device_brand
    categorical.append("identity_device_brand")
    values["identity_device_is_mobile"] = device.str.contains(
        r"android|ios|iphone|ipad|^sm-|moto|huawei|redmi|xiaomi",
        regex=True,
        na=False,
    ).astype("int8")
    values["identity_device_has_build"] = device.str.contains(
        "build", regex=False, na=False
    ).astype("int8")

    os_value = _tokens(frame.get("id_30", pd.Series(index=frame.index))).str.lower()
    os_family = os_value.str.replace(r"[\d._-]+.*$", "", regex=True).str.strip()
    values["identity_os_family"] = os_family
    categorical.append("identity_os_family")
    values["identity_os_major"] = pd.to_numeric(
        os_value.str.extract(r"(\d+)", expand=False), errors="coerce"
    ).astype("float32")

    browser = _tokens(frame.get("id_31", pd.Series(index=frame.index))).str.lower()
    browser_family = browser.str.replace(r"[\d._-]+.*$", "", regex=True).str.strip()
    values["identity_browser_family"] = browser_family
    categorical.append("identity_browser_family")
    values["identity_browser_major"] = pd.to_numeric(
        browser.str.extract(r"(\d+)", expand=False), errors="coerce"
    ).astype("float32")
    values["identity_browser_is_mobile"] = browser.str.contains(
        r"mobile|android|ios", regex=True, na=False
    ).astype("int8")
    values["identity_browser_is_generic"] = browser.str.contains(
        "generic", regex=False, na=False
    ).astype("int8")

    screen = _tokens(frame.get("id_33", pd.Series(index=frame.index))).str.lower()
    dimensions = screen.str.extract(r"^(\d+)x(\d+)$")
    width = pd.to_numeric(dimensions[0], errors="coerce")
    height = pd.to_numeric(dimensions[1], errors="coerce")
    values["identity_screen_width"] = width.astype("float32")
    values["identity_screen_height"] = height.astype("float32")
    values["identity_screen_pixels"] = (width * height).astype("float32")
    values["identity_screen_aspect"] = (
        width / height.replace(0, np.nan)
    ).astype("float32")
    screen_known = width.notna() & height.notna()
    values["identity_screen_known"] = screen_known.astype("int8")
    values["identity_screen_portrait"] = (
        width.lt(height) & screen_known
    ).fillna(False).astype("int8")

    match_status = _tokens(
        frame.get("id_34", pd.Series(index=frame.index))
    ).str.extract(r"(-?\d+)$", expand=False)
    values["identity_match_status"] = pd.to_numeric(
        match_status, errors="coerce"
    ).astype("float32")

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
        block = id_columns[start : start + 19]
        if not block:
            continue
        present = frame[block].notna().to_numpy(dtype="uint32")
        weights = np.left_shift(np.uint32(1), np.arange(len(block), dtype="uint32"))
        values[f"identity_present_mask_{block_number}"] = (
            present @ weights
        ).astype("uint32")
    if id_columns:
        values["identity_present_count"] = frame[id_columns].notna().sum(axis=1).astype(
            "int8"
        )

    d_columns = [f"D{i}" for i in range(1, 16) if f"D{i}" in frame]
    if d_columns:
        present = frame[d_columns].notna().to_numpy(dtype="uint16")
        weights = np.left_shift(np.uint16(1), np.arange(len(d_columns), dtype="uint16"))
        values["identity_D_present_mask"] = (present @ weights).astype("uint16")

    p_email = _tokens(frame["P_emaildomain"])
    r_email = _tokens(frame["R_emaildomain"])
    email_state = pd.Series("different", index=frame.index, dtype="string")
    both_missing = p_email.eq("<MISSING>") & r_email.eq("<MISSING>")
    one_missing = p_email.eq("<MISSING>") ^ r_email.eq("<MISSING>")
    email_state = email_state.mask(p_email.eq(r_email) & ~both_missing, "same")
    email_state = email_state.mask(one_missing, "one_missing")
    email_state = email_state.mask(both_missing, "both_missing")
    values["identity_email_state"] = email_state
    categorical.append("identity_email_state")
    return pd.DataFrame(values, index=frame.index), categorical


def add_identity_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict]:
    print("Parsing device, OS, browser and missingness signatures...", flush=True)
    train_features, train_categorical = _identity_rows(train)
    inference_features, inference_categorical = _identity_rows(inference)
    if train_categorical != inference_categorical:
        raise ValueError("Identity categorical columns differ")
    features = pd.concat([train_features, inference_features], ignore_index=True)
    train, inference, names = _attach_features(train, inference, features)
    return train, inference, names, train_categorical, {"features": len(names)}


def _distribution_features(
    group_codes: np.ndarray,
    values: pd.Series,
    prefix: str,
) -> dict[str, np.ndarray]:
    value_codes, _ = pd.factorize(_tokens(values), sort=False)
    helper = pd.DataFrame(
        {
            "group": group_codes,
            "value": value_codes.astype("int32", copy=False),
        }
    )
    group_count = helper.groupby("group", sort=False)["value"].transform("size")
    pair_count = helper.groupby(["group", "value"], sort=False)["value"].transform(
        "size"
    )
    unique_count = helper.groupby("group", sort=False)["value"].transform("nunique")
    mode_count = pair_count.groupby(helper["group"], sort=False).transform("max")
    share = pair_count / group_count

    pair_table = helper.groupby(["group", "value"], sort=False).size().rename("count")
    pair_frame = pair_table.reset_index()
    totals = pair_frame.groupby("group", sort=False)["count"].transform("sum")
    probability = pair_frame["count"] / totals
    pair_frame["entropy_part"] = -probability * np.log(probability)
    entropy = pair_frame.groupby("group", sort=False)["entropy_part"].sum()
    entropy_rows = helper["group"].map(entropy)
    normalized_entropy = entropy_rows / np.log(unique_count.clip(lower=2))
    return {
        f"{prefix}_nunique": unique_count.to_numpy(dtype="int16"),
        f"{prefix}_current_share": share.to_numpy(dtype="float32"),
        f"{prefix}_surprise": (-np.log(share.clip(lower=1e-8))).to_numpy(
            dtype="float32"
        ),
        f"{prefix}_mode_share": (mode_count / group_count).to_numpy(
            dtype="float32"
        ),
        f"{prefix}_is_mode": pair_count.eq(mode_count).to_numpy(dtype="int8"),
        f"{prefix}_entropy": entropy_rows.to_numpy(dtype="float32"),
        f"{prefix}_normalized_entropy": normalized_entropy.to_numpy(
            dtype="float32"
        ),
    }


def _cooccurrence_features(
    combined: pd.DataFrame,
    first: str,
    second: str,
) -> dict[str, np.ndarray]:
    first_codes, _ = pd.factorize(_tokens(combined[first]), sort=False)
    second_codes, _ = pd.factorize(_tokens(combined[second]), sort=False)
    helper = pd.DataFrame(
        {
            "first": first_codes.astype("int32", copy=False),
            "second": second_codes.astype("int32", copy=False),
        }
    )
    first_count = helper.groupby("first", sort=False)["second"].transform("size")
    second_count = helper.groupby("second", sort=False)["first"].transform("size")
    pair_count = helper.groupby(["first", "second"], sort=False)["first"].transform(
        "size"
    )
    first_nunique = helper.groupby("first", sort=False)["second"].transform("nunique")
    second_nunique = helper.groupby("second", sort=False)["first"].transform("nunique")
    prefix = f"cooc_{first}_{second}"
    return {
        f"{prefix}_log_pair_count": np.log1p(pair_count).to_numpy(dtype="float32"),
        f"{prefix}_given_first": (pair_count / first_count).to_numpy(
            dtype="float32"
        ),
        f"{prefix}_given_second": (pair_count / second_count).to_numpy(
            dtype="float32"
        ),
        f"{prefix}_first_nunique": first_nunique.to_numpy(dtype="int32"),
        f"{prefix}_second_nunique": second_nunique.to_numpy(dtype="int32"),
    }


def add_behavior_distribution_features(
    train: pd.DataFrame,
    inference: pd.DataFrame,
    train_components: np.ndarray,
    inference_components: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    columns = sorted(
        set(BEHAVIOR_COLUMNS)
        | {column for pair in COOCCURRENCE_PAIRS for column in pair}
        | {"uid_card_addr_d1_email"}
    )
    missing = sorted(set(columns).difference(train.columns))
    if missing:
        raise ValueError(f"Missing behavior columns: {missing}")
    combined = pd.concat(
        [train[columns], inference[columns]], ignore_index=True, copy=False
    )
    strict_codes, _ = pd.factorize(
        _tokens(combined["uid_card_addr_d1_email"]), sort=False
    )
    component_codes = np.concatenate([train_components, inference_components])
    group_specs = {
        "component": component_codes.astype("int32", copy=False),
        "strict_uid": strict_codes.astype("int32", copy=False),
    }
    feature_values: dict[str, np.ndarray] = {}
    for group_name, group_codes in group_specs.items():
        print(f"Building behavior distributions for {group_name}...", flush=True)
        for column in BEHAVIOR_COLUMNS:
            feature_values.update(
                _distribution_features(
                    group_codes,
                    combined[column],
                    f"behavior_{group_name}_{column}",
                )
            )
    print("Building categorical co-occurrence features...", flush=True)
    for first, second in COOCCURRENCE_PAIRS:
        feature_values.update(_cooccurrence_features(combined, first, second))
    features = pd.DataFrame(feature_values)
    train, inference, names = _attach_features(train, inference, features)
    return train, inference, names, {
        "features": len(names),
        "behavior_columns": list(BEHAVIOR_COLUMNS),
        "cooccurrence_pairs": [list(pair) for pair in COOCCURRENCE_PAIRS],
    }

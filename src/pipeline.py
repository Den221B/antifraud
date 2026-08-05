"""Сборка признаков, сегментация test и ранговое смешивание.

Здесь собирается всё, что разложено по features_*.py, в два набора признаков:

* ``vblock``     — 712 признаков: построчные, пользователь, поведение, V-блоки;
* ``structured`` — 1046 признаков: то же плюс календарь, суммы, устройство,
  распределения значений.

Две модели на разных взглядах работают лучше одной большой: признаки не
конкурируют за расщепления и слабый полезный сигнал не тонет.
"""
from __future__ import annotations

import gc
import re
import time

import numpy as np
import pandas as pd

from .common import DAY_SECONDS, SEGMENT_COLUMNS, TARGET, rank_prediction, unique_columns
from .features_blocks import (
    add_behavior_distribution_features,
    add_calendar_amount_features,
    add_identity_features,
    add_vblock_user_features,
)
from .features_graph import add_advanced_user_features, add_user_profile_features, is_raw_uid_feature
from .features_row import add_amount_patterns, build_features, convert_categories, read_and_merge

VBLOCK_FEATURE_COUNT = 712
STRUCTURED_FEATURE_COUNT = 1046


# --------------------------------------------------------------------- признаки
def build_all_features(data_dir, verbose: bool = True) -> dict:
    """Читает сырые csv и возвращает обе матрицы признаков со статистикой.

    Ничего не использует таргет: счётчики фрода по истории пользователя не
    пережили временной сдвиг и в наборе отсутствуют.
    """
    started = time.time()
    log = print if verbose else (lambda *a, **k: None)

    log("1/7 чтение и склейка", flush=True)
    train = read_and_merge(data_dir, "train")
    test = read_and_merge(data_dir, "test")
    y = train[TARGET].astype("int8").reset_index(drop=True)

    log("2/7 построчные признаки", flush=True)
    train, test, base_features, base_categorical = build_features(train, test)

    log("3/7 граф пользователей", flush=True)
    train, test, profile_features, _, _, graph_stats = add_user_profile_features(train, test)

    log("4/7 второй граф и поведение", flush=True)
    (train, test, advanced_features, advanced_categorical,
     train_components, test_components, advanced_stats) = add_advanced_user_features(train, test)

    main_features = unique_columns(base_features, profile_features, advanced_features)
    categorical = unique_columns(base_categorical, advanced_categorical)
    advanced_set, profile_set = set(advanced_features), set(profile_features)
    dynamics_features = []
    for column in main_features:
        if column in advanced_set or column in profile_set:
            dynamics_features.append(column)
        elif not (re.fullmatch(r"V\d+", column) or re.fullmatch(r"id_\d+", column)
                  or is_raw_uid_feature(column)):
            dynamics_features.append(column)

    train_ids = train.pop("TransactionID").reset_index(drop=True)
    test_ids = test.pop("TransactionID").reset_index(drop=True)
    train.pop(TARGET)
    convert_categories(train, test, categorical)
    train, amount_features = add_amount_patterns(train)
    test, _ = add_amount_patterns(test)

    log("5/7 V-блоки", flush=True)
    train, test, vblock_features, vblock_stats = add_vblock_user_features(
        train, test, train_components, test_components)

    log("6/7 календарь, суммы, устройство", flush=True)
    train, test, calendar_features, calendar_stats = add_calendar_amount_features(train, test)
    train, test, identity_features, identity_categorical, identity_stats = add_identity_features(train, test)
    convert_categories(train, test, identity_categorical)
    categorical = unique_columns(categorical, identity_categorical)

    log("7/7 распределения поведения", flush=True)
    train, test, behavior_features, behavior_stats = add_behavior_distribution_features(
        train, test, train_components, test_components)

    views = {"vblock": unique_columns(dynamics_features, vblock_features, amount_features)}
    views["structured"] = unique_columns(views["vblock"], calendar_features,
                                         identity_features, behavior_features)
    if len(views["vblock"]) != VBLOCK_FEATURE_COUNT:
        raise ValueError(f"vblock: ожидалось {VBLOCK_FEATURE_COUNT}, получено {len(views['vblock'])}")
    if len(views["structured"]) != STRUCTURED_FEATURE_COUNT:
        raise ValueError(f"structured: ожидалось {STRUCTURED_FEATURE_COUNT}, "
                         f"получено {len(views['structured'])}")
    gc.collect()
    log(f"\nготово за {(time.time() - started) / 60:.1f} мин | "
        f"vblock {len(views['vblock'])}, structured {len(views['structured'])}", flush=True)

    return {
        "train": train, "test": test, "y": y, "train_ids": train_ids, "test_ids": test_ids,
        "views": views, "categorical": categorical,
        "groups": {"base": base_features, "profile": profile_features,
                   "advanced": advanced_features, "dynamics": dynamics_features,
                   "vblock": vblock_features, "calendar": calendar_features,
                   "identity": identity_features, "behavior": behavior_features},
        "stats": {"graph": graph_stats, "advanced_graph": advanced_stats, "vblock": vblock_stats,
                  "calendar_amount": calendar_stats, "identity": identity_stats,
                  "behavior": behavior_stats},
        "elapsed_minutes": (time.time() - started) / 60.0,
    }


# -------------------------------------------------------------------- сегменты
def segment_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Три уровня прокси-идентификатора для строки."""
    result = frame.loc[:, list(SEGMENT_COLUMNS)].copy()
    result["origin_day"] = (result.TransactionDT / DAY_SECONDS - result.D1).round()
    specs = {"strict": ("card1", "addr1", "origin_day", "P_emaildomain"),
             "card_origin": ("card1", "addr1", "origin_day"),
             "origin_email": ("origin_day", "P_emaildomain")}
    for name, columns in specs.items():
        ok = ("card1", "addr1", "origin_day") if name == "strict" else columns
        result[f"{name}_valid"] = result[list(ok)].notna().all(axis=1)
        result[f"{name}_key"] = pd.util.hash_pandas_object(
            result[list(columns)], index=False, categorize=True).to_numpy(dtype="uint64", copy=False)
    return result


def seen_before(reference: pd.DataFrame, query: pd.DataFrame, name: str) -> np.ndarray:
    known = pd.Index(reference.loc[reference[f"{name}_valid"], f"{name}_key"].unique())
    return query[f"{name}_valid"].to_numpy(dtype=bool) & query[f"{name}_key"].isin(known).to_numpy()


def assign_segments(reference_frame: pd.DataFrame, query_frame: pd.DataFrame) -> np.ndarray:
    """strict — точный UID найден в истории, partial — более широкий ключ, cold — ничего."""
    reference, query = segment_keys(reference_frame), segment_keys(query_frame)
    strict = seen_before(reference, query, "strict")
    partial = seen_before(reference, query, "card_origin") | seen_before(reference, query, "origin_email")
    segments = np.full(len(query), "cold", dtype=object)
    segments[~strict & partial] = "partial"
    segments[strict] = "strict"
    return segments


# ----------------------------------------------------------------- смешивание
def uid_max_consistency(frame: pd.DataFrame, prediction, weight: float = 0.50) -> np.ndarray:
    """Смешать прогноз строки с максимумом по её строгому UID — так устроена база."""
    keys = segment_keys(frame)
    mask = keys["strict_valid"].to_numpy(dtype=bool)
    result = np.asarray(prediction, dtype="float64").copy()
    helper = pd.DataFrame({"k": keys.loc[mask, "strict_key"].to_numpy(), "p": result[mask]})
    grouped = helper.groupby("k", sort=False)["p"].max()
    result[mask] = (1.0 - weight) * result[mask] + weight * helper["k"].map(grouped).to_numpy()
    return result


def apply_residuals(base_values, additions) -> np.ndarray:
    """Слоями подмешать добавки к базе, каждую — только в её маске.

    ``additions`` — последовательность ``(prediction, weight, mask)``. Перед
    каждым следующим слоем результат переранжируется, потому что после
    смешивания значения перестают быть рангами. После последнего слоя
    переранжирования нет: именно эти значения лежат в зафиксированном сабмите,
    и лишний ранг сохранил бы порядок, но изменил бы числа.
    """
    stage = rank_prediction(base_values)
    for layer, (prediction, weight, mask) in enumerate(additions):
        if layer:
            stage = rank_prediction(stage)
        updated = stage.copy()
        updated[mask] = (1 - weight) * stage[mask] + weight * rank_prediction(prediction)[mask]
        stage = updated
    if not np.isfinite(stage).all() or np.any((stage < 0) | (stage > 1)):
        raise ValueError("прогноз вне [0, 1] или содержит nan")
    return stage

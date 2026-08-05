"""Общие константы и мелкие утилиты, на которые опирается весь конвейер."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "isFraud"
DAY_SECONDS = 86_400.0

# 11 блоков V-колонок: внутри блока значения пусты или заполнены вместе (§1.3 ноутбука)
V_BLOCKS = ((1, 11), (12, 34), (35, 52), (53, 74), (75, 94), (95, 137),
            (138, 166), (167, 216), (217, 278), (279, 321), (322, 339))

# 30 V-колонок, по которым берутся агрегаты внутри пользователя (§2.4 ноутбука)
TOP_V_COLUMNS = ("V242", "V243", "V258", "V265", "V199", "V201", "V257", "V274", "V86", "V156",
                 "V44", "V87", "V62", "V217", "V91", "V67", "V275", "V189", "V212", "V70",
                 "V61", "V69", "V45", "V149", "V256", "V90", "V266", "V94", "V308", "V200")

# временные окна валидации: train < a | пропуск | validation [b, c)
TEMPORAL_FOLDS = ((15, 45, 60), (30, 60, 75), (45, 75, 90), (60, 90, 106))
HOLDOUT_FOLD = 3

SEGMENT_COLUMNS = ("TransactionID", "TransactionDT", "card1", "addr1", "D1", "P_emaildomain")


def rank_prediction(values):
    """Перцентильный ранг: AUC зависит только от порядка, а не от шкалы вероятностей."""
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy()


def unique_columns(*groups):
    return list(dict.fromkeys(column for group in groups for column in group))

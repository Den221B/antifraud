from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    number: int
    train_end_day: float
    validation_start_day: float
    validation_end_day: float
    train_index: np.ndarray
    validation_index: np.ndarray


FOLD_WINDOWS = (
    (15.0, 45.0, 60.0),
    (30.0, 60.0, 75.0),
    (45.0, 75.0, 90.0),
    (60.0, 90.0, 106.0),
)


def make_four_long_gap_folds(
    transaction_dt: pd.Series,
) -> list[PurgedFold]:
    day = transaction_dt.to_numpy(dtype="float64") / 86400.0
    folds = []
    for number, (train_end, validation_start, validation_end) in enumerate(
        FOLD_WINDOWS
    ):
        train_index = np.flatnonzero(day < train_end)
        validation_index = np.flatnonzero(
            (day >= validation_start) & (day < validation_end)
        )
        if len(train_index) == 0 or len(validation_index) == 0:
            raise ValueError(f"Temporal fold {number} is empty")
        folds.append(
            PurgedFold(
                number=number,
                train_end_day=train_end,
                validation_start_day=validation_start,
                validation_end_day=validation_end,
                train_index=train_index,
                validation_index=validation_index,
            )
        )
    return folds


def fold_metadata(
    folds: list[PurgedFold],
    y: pd.Series,
) -> list[dict]:
    target = y.to_numpy(dtype="int8")
    rows = []
    for fold in folds:
        rows.append(
            {
                "fold": fold.number,
                "train_end_day": fold.train_end_day,
                "validation_start_day": fold.validation_start_day,
                "validation_end_day": fold.validation_end_day,
                "embargo_days": (
                    fold.validation_start_day - fold.train_end_day
                ),
                "train_rows": int(len(fold.train_index)),
                "validation_rows": int(len(fold.validation_index)),
                "train_fraud_rate": float(target[fold.train_index].mean()),
                "validation_fraud_rate": float(
                    target[fold.validation_index].mean()
                ),
            }
        )
    return rows

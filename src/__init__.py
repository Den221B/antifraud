"""Антифрод: конвейер признаков и сборка прогноза.

    from src.pipeline import build_all_features, assign_segments, apply_residuals

Разбиение по смыслу:

* ``common``          — константы (V-блоки, окна валидации) и ранги;
* ``features_row``    — построчные признаки, UID-цепочки, частоты по train;
* ``features_graph``  — два графа пользователей и агрегаты по компонентам;
* ``features_blocks`` — V-блоки, суммы, устройство, распределения значений;
* ``pipeline``        — сборка двух наборов признаков, сегменты, смешивание.
"""
from .common import (
    DAY_SECONDS,
    HOLDOUT_FOLD,
    SEGMENT_COLUMNS,
    TARGET,
    TEMPORAL_FOLDS,
    TOP_V_COLUMNS,
    V_BLOCKS,
    rank_prediction,
    unique_columns,
)

__all__ = [
    "DAY_SECONDS", "HOLDOUT_FOLD", "SEGMENT_COLUMNS", "TARGET", "TEMPORAL_FOLDS",
    "TOP_V_COLUMNS", "V_BLOCKS", "rank_prediction", "unique_columns",
]

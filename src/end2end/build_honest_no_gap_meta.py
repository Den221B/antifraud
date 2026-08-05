"""Build a train-only selected no-gap meta ensemble.

The script combines the clean CatBoost/LightGBM/XGBoost temporal OOF matrix
with the independently trained temporal7 stack. The last official-train fold
is the only model-selection holdout. No external rows or test labels are read.
"""

from __future__ import annotations

import gc
import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
TARGET = "isFraud"
BASE_SOURCES = ("catboost", "lightgbm", "xgboost")
TEMPORAL_SOURCE = "temporal7"
META_SEEDS = (42, 2026, 3407)
LOGISTIC_C = 0.20
MAX_EPOCHS = 120
PATIENCE = 12
OUTPUT_PATH = ROOT / "submission_honest_no_gap_temporal_meta.csv"
METRICS_PATH = ROOT / "honest_no_gap_temporal_meta_metrics.json"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def rank_prediction(values: np.ndarray | pd.Series) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy()


def make_uid(frame: pd.DataFrame) -> pd.Series:
    origin_day = (frame["TransactionDT"] / 86_400 - frame["D1"]).round()
    return (
        frame["card1"].astype("string").fillna("<MISSING>")
        + "|"
        + frame["addr1"].astype("string").fillna("<MISSING>")
        + "|"
        + origin_day.astype("Int64").astype("string").fillna("<MISSING>")
        + "|"
        + frame["P_emaildomain"].astype("string").fillna("<MISSING>")
    )


def apply_uid_max(
    prediction: np.ndarray,
    uid: pd.Series,
    weight: float,
) -> np.ndarray:
    work = pd.DataFrame({"uid": uid.to_numpy(), "prediction": prediction})
    grouped = work.groupby("uid", sort=False)["prediction"].transform("max")
    return (1.0 - weight) * prediction + weight * grouped.to_numpy()


def build_meta_features(
    predictions: pd.DataFrame,
    sources: tuple[str, ...],
    fold: pd.Series | None = None,
) -> pd.DataFrame:
    probability = predictions.loc[:, list(sources)].clip(1e-6, 1 - 1e-6)
    if fold is None:
        ranks = probability.rank(method="average", pct=True)
    else:
        ranks = probability.groupby(fold).rank(method="average", pct=True)

    features: dict[str, pd.Series] = {}
    for column in sources:
        values = probability[column]
        features[f"{column}_prob"] = values
        features[f"{column}_logit"] = np.log(values / (1.0 - values))
        features[f"{column}_rank"] = ranks[column]
        features[f"{column}_rank_sq"] = ranks[column] ** 2

    features["rank_mean"] = ranks.mean(axis=1)
    features["rank_std"] = ranks.std(axis=1)
    features["rank_min"] = ranks.min(axis=1)
    features["rank_max"] = ranks.max(axis=1)
    for first, second in combinations(sources, 2):
        features[f"rank_{first}_{second}_diff"] = (
            ranks[first] - ranks[second]
        ).abs()
    return pd.DataFrame(features, index=predictions.index).astype("float32")


class MetaMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.hidden = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values) + 0.20 * self.hidden(values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_mlp(model: MetaMLP, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(values, dtype=torch.float32)
        return torch.sigmoid(model(tensor)).squeeze(1).cpu().numpy()


def train_mlp_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
) -> tuple[int, np.ndarray, float]:
    seed_everything(seed)
    model = MetaMLP(X_train.shape[1])
    positive_weight = float(np.sqrt((len(y_train) - y_train.sum()) / y_train.sum()))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=8192,
        shuffle=True,
    )

    best_auc = -np.inf
    best_epoch = 1
    best_state = None
    stale_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(values), labels)
            loss.backward()
            optimizer.step()
        prediction = predict_mlp(model, X_valid)
        score = roc_auc_score(y_valid, prediction)
        if score > best_auc + 1e-6:
            best_auc = float(score)
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("MLP did not produce a validation state")
    model.load_state_dict(best_state)
    return best_epoch, predict_mlp(model, X_valid), best_auc


def train_mlp_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    epochs: int,
) -> np.ndarray:
    seed_everything(seed)
    model = MetaMLP(X_train.shape[1])
    positive_weight = float(np.sqrt((len(y_train) - y_train.sum()) / y_train.sum()))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=8192,
        shuffle=True,
    )
    for _ in range(max(1, epochs)):
        model.train()
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(values), labels)
            loss.backward()
            optimizer.step()
    prediction = predict_mlp(model, X_test)
    del model
    gc.collect()
    return prediction


def best_rank_blend(
    y_true: np.ndarray,
    neural: np.ndarray,
    linear: np.ndarray,
) -> dict[str, float]:
    neural_rank = rank_prediction(neural)
    linear_rank = rank_prediction(linear)
    rows = []
    for neural_weight in np.linspace(0.0, 1.0, 101):
        prediction = neural_weight * neural_rank + (1.0 - neural_weight) * linear_rank
        rows.append(
            {
                "neural_weight": float(neural_weight),
                "linear_weight": float(1.0 - neural_weight),
                "auc": float(roc_auc_score(y_true, prediction)),
            }
        )
    return max(rows, key=lambda row: row["auc"])


def evaluate_source_set(
    oof: pd.DataFrame,
    uid_train: pd.Series,
    sources: tuple[str, ...],
) -> dict:
    features = build_meta_features(oof, sources, fold=oof["fold"])
    train_mask = oof["fold"] < 2
    valid_mask = oof["fold"] == 2
    y_train = oof.loc[train_mask, TARGET].to_numpy(dtype="float32")
    y_valid = oof.loc[valid_mask, TARGET].to_numpy(dtype="float32")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features.loc[train_mask]).astype("float32")
    X_valid = scaler.transform(features.loc[valid_mask]).astype("float32")
    linear = LogisticRegression(
        C=LOGISTIC_C,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    linear.fit(X_train, y_train)
    linear_valid = linear.predict_proba(X_valid)[:, 1]

    epochs: dict[str, int] = {}
    neural_parts = []
    seed_auc = {}
    for seed in META_SEEDS:
        epoch, prediction, score = train_mlp_with_validation(
            X_train, y_train, X_valid, y_valid, seed
        )
        epochs[str(seed)] = int(epoch)
        neural_parts.append(prediction)
        seed_auc[str(seed)] = float(score)
    neural_valid = np.mean(neural_parts, axis=0)
    blend = best_rank_blend(y_valid, neural_valid, linear_valid)
    prediction = (
        blend["neural_weight"] * rank_prediction(neural_valid)
        + blend["linear_weight"] * rank_prediction(linear_valid)
    )

    valid_rows = oof.loc[valid_mask, "row_index"].to_numpy(dtype="int64")
    uid = uid_train.iloc[valid_rows].reset_index(drop=True)
    uid_candidates = []
    for weight in (0.0, 0.10, 0.25, 0.50):
        processed = apply_uid_max(prediction, uid, weight)
        uid_candidates.append(
            {"weight": weight, "auc": float(roc_auc_score(y_valid, processed))}
        )
    uid_recipe = max(uid_candidates, key=lambda row: row["auc"])
    return {
        "sources": list(sources),
        "features": list(features.columns),
        "holdout_rows": int(valid_mask.sum()),
        "linear_auc": float(roc_auc_score(y_valid, linear_valid)),
        "neural_auc": float(roc_auc_score(y_valid, neural_valid)),
        "blend": blend,
        "uid_candidates": uid_candidates,
        "uid_recipe": uid_recipe,
        "epochs": epochs,
        "seed_auc": seed_auc,
    }


def fit_final(
    oof: pd.DataFrame,
    test_predictions: pd.DataFrame,
    uid_test: pd.Series,
    recipe: dict,
) -> np.ndarray:
    sources = tuple(recipe["sources"])
    train_features = build_meta_features(oof, sources, fold=oof["fold"])
    test_features = build_meta_features(test_predictions, sources)
    y = oof[TARGET].to_numpy(dtype="float32")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features).astype("float32")
    X_test = scaler.transform(test_features).astype("float32")
    linear = LogisticRegression(
        C=LOGISTIC_C,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    linear.fit(X_train, y)
    linear_test = linear.predict_proba(X_test)[:, 1]

    neural_parts = []
    for seed in META_SEEDS:
        neural_parts.append(
            train_mlp_fixed_epochs(
                X_train,
                y,
                X_test,
                seed,
                int(recipe["epochs"][str(seed)]),
            )
        )
    neural_test = np.mean(neural_parts, axis=0)
    blend = recipe["blend"]
    prediction = (
        float(blend["neural_weight"]) * rank_prediction(neural_test)
        + float(blend["linear_weight"]) * rank_prediction(linear_test)
    )
    return apply_uid_max(
        prediction,
        uid_test.reset_index(drop=True),
        float(recipe["uid_recipe"]["weight"]),
    )


def main() -> None:
    boost = pd.read_csv(ROOT / "boost3_oof_predictions.csv")
    temporal = pd.read_csv(ROOT / "temporal7_oof_predictions.csv")
    temporal = temporal[["row_index", "fold", "stack_prediction"]].rename(
        columns={"fold": "temporal_fold", "stack_prediction": TEMPORAL_SOURCE}
    )
    oof = boost.merge(temporal, on="row_index", how="left", validate="one_to_one")
    if oof[TEMPORAL_SOURCE].isna().any():
        raise ValueError("Temporal7 OOF predictions do not cover boost3 rows")
    if not np.array_equal(
        oof["temporal_fold"].to_numpy(), oof["fold"].to_numpy() + 1
    ):
        raise ValueError("Boost3 and temporal7 folds are not aligned")

    raw_columns = [
        "TransactionID",
        "TransactionDT",
        "card1",
        "addr1",
        "D1",
        "P_emaildomain",
    ]
    raw_train = pd.read_csv(ROOT / "train_transaction.csv", usecols=raw_columns)
    raw_test = pd.read_csv(ROOT / "test_transaction.csv", usecols=raw_columns)
    uid_train = make_uid(raw_train)
    uid_test = make_uid(raw_test)

    baseline = evaluate_source_set(oof, uid_train, BASE_SOURCES)
    temporal_recipe = evaluate_source_set(
        oof, uid_train, (*BASE_SOURCES, TEMPORAL_SOURCE)
    )
    holdout_gain = float(
        temporal_recipe["uid_recipe"]["auc"] - baseline["uid_recipe"]["auc"]
    )
    accepted = holdout_gain > 0.0
    selected = temporal_recipe if accepted else baseline

    test_predictions = pd.DataFrame(
        {
            "catboost": np.load(ROOT / "stack_test_catboost.npy"),
            "lightgbm": np.load(ROOT / "stack_test_lightgbm.npy"),
            "xgboost": np.load(ROOT / "stack_test_xgboost.npy"),
            TEMPORAL_SOURCE: pd.read_csv(
                ROOT / "submission_temporal7_no_gap.csv"
            )[TARGET].to_numpy(),
        }
    )
    prediction = fit_final(oof, test_predictions, uid_test, selected)
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    if not np.array_equal(sample["TransactionID"].to_numpy(), raw_test["TransactionID"]):
        raise ValueError("Sample and test TransactionID order differ")
    output = sample[["TransactionID"]].copy()
    output[TARGET] = prediction
    output.to_csv(OUTPUT_PATH, index=False)

    metrics = {
        "selection": "official-train temporal OOF only",
        "external_gap_used": False,
        "competition_test_labels_used": False,
        "baseline": baseline,
        "temporal_candidate": temporal_recipe,
        "holdout_gain": holdout_gain,
        "temporal_source_accepted": accepted,
        "selected_sources": selected["sources"],
        "output": OUTPUT_PATH.name,
        "rows": len(output),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

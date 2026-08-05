from pathlib import Path
import argparse
import gc
import json
import random
import time

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from fraud_features import TARGET, build_features, read_and_merge


DATA_DIR = Path(__file__).resolve().parent
SEED = 42
META_SEEDS = (42, 2026, 3407)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

CAT_PARAMS = {
    "iterations": 1400,
    "depth": 8,
    "learning_rate": 0.06,
    "l2_leaf_reg": 8,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.80,
    "rsm": 0.90,
    "border_count": 128,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": SEED,
    "one_hot_max_size": 10,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": 100,
}

LGB_PARAMS = {
    "n_estimators": 1800,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "min_child_samples": 40,
    "subsample": 0.70,
    "subsample_freq": 1,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "max_bin": 255,
    "max_depth": -1,
    "extra_trees": True,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}

XGB_PARAMS = {
    "n_estimators": 1800,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 20,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "gamma": 0.05,
    "max_bin": 256,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "early_stopping_rounds": 120,
    "random_state": SEED,
    "n_jobs": -1,
}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def save_metrics(metrics):
    (DATA_DIR / "boost3_neural_stack_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=json_default),
        encoding="utf-8",
    )


def is_giba_feature(column):
    return (
        column == "D1_origin_day"
        or column.startswith("uid_d1_email")
        or column.startswith("uid_card_addr_d1")
    )


def rank_prediction(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def make_folds(transaction_dt):
    day = (transaction_dt // 86400).astype("int16")
    specs = [
        (0, 30, 60, 75),
        (1, 45, 75, 90),
        (2, 60, 90, int(day.max()) + 1),
    ]
    folds = []
    for fold, train_end, valid_start, valid_end in specs:
        train_index = day.index[day < train_end]
        valid_index = day.index[(day >= valid_start) & (day < valid_end)]
        folds.append(
            {
                "fold": fold,
                "train_end_day": train_end,
                "valid_start_day": valid_start,
                "valid_end_day": valid_end,
                "train_index": train_index,
                "valid_index": valid_index,
            }
        )
    return folds


def prediction_frame(fold, valid_index, train_ids, y, prediction, name):
    return pd.DataFrame(
        {
            "row_index": valid_index.to_numpy(),
            "TransactionID": train_ids.loc[valid_index].to_numpy(),
            "fold": fold,
            TARGET: y.loc[valid_index].to_numpy(),
            name: prediction,
        }
    )


def train_catboost_oof(
    train,
    y,
    train_ids,
    feature_columns,
    categorical_columns,
    folds,
    metrics,
    force,
):
    rows = []
    for fold_info in folds:
        fold = fold_info["fold"]
        cache_path = DATA_DIR / f"stack_catboost_fold_{fold}.csv"
        model_path = DATA_DIR / f"stack_catboost_fold_{fold}.cbm"
        if cache_path.exists() and not force:
            print(f"Loading cached CatBoost fold {fold}", flush=True)
            rows.append(pd.read_csv(cache_path))
            continue

        print(f"\nCatBoost OOF fold {fold}", flush=True)
        started = time.time()
        model = CatBoostClassifier(
            **{
                **CAT_PARAMS,
                "random_seed": SEED + fold,
            }
        )
        model.fit(
            train.loc[fold_info["train_index"], feature_columns],
            y.loc[fold_info["train_index"]],
            cat_features=categorical_columns,
            eval_set=(
                train.loc[fold_info["valid_index"], feature_columns],
                y.loc[fold_info["valid_index"]],
            ),
            early_stopping_rounds=120,
            use_best_model=True,
        )
        prediction = model.predict_proba(
            train.loc[fold_info["valid_index"], feature_columns]
        )[:, 1]
        frame = prediction_frame(
            fold,
            fold_info["valid_index"],
            train_ids,
            y,
            prediction,
            "catboost",
        )
        frame.to_csv(cache_path, index=False)
        model.save_model(model_path)
        rows.append(frame)
        metrics["catboost_folds"][str(fold)] = {
            "auc": roc_auc_score(frame[TARGET], frame["catboost"]),
            "best_iteration": model.get_best_iteration() + 1,
            "minutes": (time.time() - started) / 60,
        }
        save_metrics(metrics)
        del model, prediction, frame
        gc.collect()
    return pd.concat(rows, ignore_index=True)


def convert_categories_for_lgb(train, test, categorical_columns):
    for column in categorical_columns:
        categories = pd.Index(train[column].dropna().unique())
        dtype = pd.CategoricalDtype(categories=categories)
        train[column] = train[column].astype(dtype)
        test[column] = test[column].astype(dtype)


def train_lightgbm_oof(
    train,
    y,
    train_ids,
    feature_columns,
    categorical_columns,
    folds,
    metrics,
    force,
):
    rows = []
    for fold_info in folds:
        fold = fold_info["fold"]
        cache_path = DATA_DIR / f"stack_lightgbm_fold_{fold}.csv"
        model_path = DATA_DIR / f"stack_lightgbm_fold_{fold}.txt"
        if cache_path.exists() and not force:
            print(f"Loading cached LightGBM fold {fold}", flush=True)
            rows.append(pd.read_csv(cache_path))
            continue

        print(f"\nLightGBM OOF fold {fold}", flush=True)
        started = time.time()
        model = lgb.LGBMClassifier(
            **{
                **LGB_PARAMS,
                "random_state": SEED + fold,
            }
        )
        model.fit(
            train.loc[fold_info["train_index"], feature_columns],
            y.loc[fold_info["train_index"]],
            categorical_feature=categorical_columns,
            eval_set=[
                (
                    train.loc[fold_info["valid_index"], feature_columns],
                    y.loc[fold_info["valid_index"]],
                )
            ],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(120, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        prediction = model.predict_proba(
            train.loc[fold_info["valid_index"], feature_columns],
            num_iteration=model.best_iteration_,
        )[:, 1]
        frame = prediction_frame(
            fold,
            fold_info["valid_index"],
            train_ids,
            y,
            prediction,
            "lightgbm",
        )
        frame.to_csv(cache_path, index=False)
        model.booster_.save_model(
            model_path,
            num_iteration=model.best_iteration_,
        )
        rows.append(frame)
        metrics["lightgbm_folds"][str(fold)] = {
            "auc": roc_auc_score(frame[TARGET], frame["lightgbm"]),
            "best_iteration": model.best_iteration_,
            "minutes": (time.time() - started) / 60,
        }
        save_metrics(metrics)
        del model, prediction, frame
        gc.collect()
    return pd.concat(rows, ignore_index=True)


def select_xgb_features(feature_columns, limit=240):
    scores = pd.Series(0.0, index=feature_columns)

    cat_model = CatBoostClassifier()
    cat_model.load_model(DATA_DIR / "catboost_giba_validation.cbm")
    cat_importance = pd.Series(
        cat_model.get_feature_importance(),
        index=cat_model.feature_names_,
    )
    if cat_importance.sum() > 0:
        scores = scores.add(cat_importance / cat_importance.sum(), fill_value=0)

    lgb_model = lgb.Booster(
        model_file=str(DATA_DIR / "lightgbm_giba_validation.txt")
    )
    lgb_importance = pd.Series(
        lgb_model.feature_importance(importance_type="gain"),
        index=lgb_model.feature_name(),
    )
    if lgb_importance.sum() > 0:
        scores = scores.add(lgb_importance / lgb_importance.sum(), fill_value=0)

    top_features = scores.sort_values(ascending=False).head(limit).index.tolist()
    giba_features = [
        column for column in feature_columns if is_giba_feature(column)
    ]
    selected = list(dict.fromkeys([*top_features, *giba_features]))
    return selected, scores.sort_values(ascending=False)


def convert_categories_for_xgb(train, test, categorical_columns):
    for column in categorical_columns:
        train[column] = train[column].cat.codes.astype("int32")
        test[column] = test[column].cat.codes.astype("int32")


def train_xgboost_oof(
    train,
    y,
    train_ids,
    xgb_features,
    folds,
    metrics,
    force,
):
    rows = []
    for fold_info in folds:
        fold = fold_info["fold"]
        cache_path = DATA_DIR / f"stack_xgboost_fold_{fold}.csv"
        model_path = DATA_DIR / f"stack_xgboost_fold_{fold}.json"
        if cache_path.exists() and not force:
            print(f"Loading cached XGBoost fold {fold}", flush=True)
            rows.append(pd.read_csv(cache_path))
            continue

        print(f"\nXGBoost OOF fold {fold}", flush=True)
        started = time.time()
        model = xgb.XGBClassifier(
            **{
                **XGB_PARAMS,
                "random_state": SEED + fold,
            }
        )
        model.fit(
            train.loc[fold_info["train_index"], xgb_features],
            y.loc[fold_info["train_index"]],
            eval_set=[
                (
                    train.loc[fold_info["valid_index"], xgb_features],
                    y.loc[fold_info["valid_index"]],
                )
            ],
            verbose=100,
        )
        prediction = model.predict_proba(
            train.loc[fold_info["valid_index"], xgb_features]
        )[:, 1]
        frame = prediction_frame(
            fold,
            fold_info["valid_index"],
            train_ids,
            y,
            prediction,
            "xgboost",
        )
        frame.to_csv(cache_path, index=False)
        model.save_model(model_path)
        rows.append(frame)
        metrics["xgboost_folds"][str(fold)] = {
            "auc": roc_auc_score(frame[TARGET], frame["xgboost"]),
            "best_iteration": int(model.best_iteration + 1),
            "minutes": (time.time() - started) / 60,
        }
        save_metrics(metrics)
        del model, prediction, frame
        gc.collect()
    return pd.concat(rows, ignore_index=True)


def merge_oof(cat_oof, lgb_oof, xgb_oof):
    keys = ["row_index", "TransactionID", "fold", TARGET]
    result = cat_oof.merge(lgb_oof, on=keys, validate="one_to_one")
    result = result.merge(xgb_oof, on=keys, validate="one_to_one")
    return result.sort_values(["fold", "row_index"]).reset_index(drop=True)


def build_meta_features(predictions, fold=None):
    base_columns = ["catboost", "lightgbm", "xgboost"]
    probability = predictions[base_columns].clip(1e-6, 1 - 1e-6)
    if fold is None:
        ranks = probability.rank(method="average", pct=True)
    else:
        ranks = probability.groupby(fold).rank(method="average", pct=True)

    features = {}
    for column in base_columns:
        features[f"{column}_prob"] = probability[column]
        features[f"{column}_logit"] = np.log(
            probability[column] / (1 - probability[column])
        )
        features[f"{column}_rank"] = ranks[column]
        features[f"{column}_rank_sq"] = ranks[column] ** 2

    features["rank_mean"] = ranks.mean(axis=1)
    features["rank_std"] = ranks.std(axis=1)
    features["rank_min"] = ranks.min(axis=1)
    features["rank_max"] = ranks.max(axis=1)
    features["rank_cat_lgb_diff"] = (
        ranks["catboost"] - ranks["lightgbm"]
    ).abs()
    features["rank_cat_xgb_diff"] = (
        ranks["catboost"] - ranks["xgboost"]
    ).abs()
    features["rank_lgb_xgb_diff"] = (
        ranks["lightgbm"] - ranks["xgboost"]
    ).abs()
    return pd.DataFrame(features, index=predictions.index).astype("float32")


class MetaMLP(nn.Module):
    def __init__(self, input_size):
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

    def forward(self, values):
        return self.linear(values) + 0.20 * self.hidden(values)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_mlp(model, values):
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(values, dtype=torch.float32)
        return torch.sigmoid(model(tensor)).squeeze(1).cpu().numpy()


def train_mlp_with_validation(
    X_train,
    y_train,
    X_valid,
    y_valid,
    seed,
):
    seed_everything(seed)
    model = MetaMLP(X_train.shape[1])
    positive_weight = float(
        np.sqrt((len(y_train) - y_train.sum()) / y_train.sum())
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=2e-4,
    )
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=8192,
        shuffle=True,
    )

    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    patience = 12
    stale_epochs = 0
    for epoch in range(1, 121):
        model.train()
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(values), labels)
            loss.backward()
            optimizer.step()

        valid_prediction = predict_mlp(model, X_valid)
        valid_auc = roc_auc_score(y_valid, valid_prediction)
        if valid_auc > best_auc + 1e-6:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % 10 == 0:
            print(
                f"MLP seed={seed} epoch={epoch} "
                f"valid_auc={valid_auc:.9f} best={best_auc:.9f}",
                flush=True,
            )
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, predict_mlp(model, X_valid), best_auc


def train_mlp_fixed_epochs(X, y, X_test, seed, epochs):
    seed_everything(seed)
    print(f"Final MLP seed={seed}, epochs={epochs}", flush=True)
    model = MetaMLP(X.shape[1])
    positive_weight = float(np.sqrt((len(y) - y.sum()) / y.sum()))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=2e-4,
    )
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=8192,
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(values), labels)
            loss.backward()
            optimizer.step()
    return model, predict_mlp(model, X_test)


def best_rank_blend(y_true, first, second):
    first_rank = rank_prediction(first)
    second_rank = rank_prediction(second)
    rows = []
    for first_weight in np.linspace(0, 1, 101):
        prediction = (
            first_weight * first_rank
            + (1 - first_weight) * second_rank
        )
        rows.append(
            {
                "neural_weight": first_weight,
                "linear_weight": 1 - first_weight,
                "auc": roc_auc_score(y_true, prediction),
            }
        )
    return max(rows, key=lambda row: row["auc"])


def apply_uid_postprocess(prediction, uid, method, weight):
    frame = pd.DataFrame({"prediction": prediction, "uid": uid.to_numpy()})
    if method == "none":
        return prediction
    aggregate = frame.groupby("uid", sort=False)["prediction"].transform(method)
    return (1 - weight) * prediction + weight * aggregate.to_numpy()


def choose_uid_postprocess(y_true, prediction, uid):
    candidates = [{"method": "none", "weight": 0.0}]
    for method in ("mean", "max"):
        for weight in (0.25, 0.50, 0.75, 1.0):
            candidates.append({"method": method, "weight": weight})
    for candidate in candidates:
        processed = apply_uid_postprocess(
            prediction,
            uid,
            candidate["method"],
            candidate["weight"],
        )
        candidate["auc"] = roc_auc_score(y_true, processed)
    return max(candidates, key=lambda row: row["auc"]), candidates


parser = argparse.ArgumentParser()
parser.add_argument(
    "--force",
    action="store_true",
    help="Ignore cached OOF/test predictions and retrain every model.",
)
args = parser.parse_args()

started_at = time.time()
metrics_path = DATA_DIR / "boost3_neural_stack_metrics.json"
if metrics_path.exists() and not args.force:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.setdefault("catboost_folds", {})
    metrics.setdefault("lightgbm_folds", {})
    metrics.setdefault("xgboost_folds", {})
else:
    metrics = {
        "scheme": "3 boosting models -> PyTorch MLP",
        "base_models": ["CatBoost", "LightGBM", "XGBoost"],
        "validation": (
            "three non-overlapping temporal OOF windows with 30-day gap"
        ),
        "catboost_folds": {},
        "lightgbm_folds": {},
        "xgboost_folds": {},
    }

print("Reading data and building Giba UID features...", flush=True)
train = read_and_merge(DATA_DIR, "train")
test = read_and_merge(DATA_DIR, "test")
train, test, feature_columns, categorical_columns = build_features(
    train,
    test,
    giba_features=True,
)
y = train.pop(TARGET)
train_ids = train.pop("TransactionID")
test_ids = test.pop("TransactionID")
transaction_dt = train["TransactionDT"].copy()
uid_train = train["uid_card_addr_d1_email"].astype("string").copy()
uid_test = test["uid_card_addr_d1_email"].astype("string").copy()
folds = make_folds(transaction_dt)

metrics["features"] = {
    "boosting_features": len(feature_columns),
    "categorical": len(categorical_columns),
}
metrics["folds"] = [
    {
        "fold": info["fold"],
        "train_rows": len(info["train_index"]),
        "valid_rows": len(info["valid_index"]),
        "train_end_day": info["train_end_day"],
        "valid_start_day": info["valid_start_day"],
        "valid_end_day": info["valid_end_day"],
    }
    for info in folds
]
save_metrics(metrics)

print("\nStage 1/4: CatBoost OOF...", flush=True)
cat_oof = train_catboost_oof(
    train,
    y,
    train_ids,
    feature_columns,
    categorical_columns,
    folds,
    metrics,
    args.force,
)

cat_test_path = DATA_DIR / "stack_test_catboost.npy"
if cat_test_path.exists() and not args.force:
    cat_test = np.load(cat_test_path)
else:
    final_cat = CatBoostClassifier()
    final_cat.load_model(DATA_DIR / "catboost_giba_final.cbm")
    cat_test = final_cat.predict_proba(
        test[final_cat.feature_names_]
    )[:, 1]
    np.save(cat_test_path, cat_test)
    del final_cat
    gc.collect()

print("\nStage 2/4: LightGBM OOF...", flush=True)
convert_categories_for_lgb(train, test, categorical_columns)
lgb_oof = train_lightgbm_oof(
    train,
    y,
    train_ids,
    feature_columns,
    categorical_columns,
    folds,
    metrics,
    args.force,
)

lgb_test_path = DATA_DIR / "stack_test_lightgbm.npy"
if lgb_test_path.exists() and not args.force:
    lgb_test = np.load(lgb_test_path)
else:
    final_lgb = lgb.Booster(
        model_file=str(DATA_DIR / "lightgbm_giba_final.txt")
    )
    lgb_test = final_lgb.predict(test[final_lgb.feature_name()])
    np.save(lgb_test_path, lgb_test)
    del final_lgb
    gc.collect()

print("\nStage 3/4: XGBoost OOF and final model...", flush=True)
xgb_features, xgb_importance = select_xgb_features(feature_columns)
xgb_importance.rename("combined_importance").to_csv(
    DATA_DIR / "stack_xgboost_feature_ranking.csv",
    header=True,
)
metrics["features"]["xgboost_selected"] = len(xgb_features)
convert_categories_for_xgb(train, test, categorical_columns)
xgb_oof = train_xgboost_oof(
    train,
    y,
    train_ids,
    xgb_features,
    folds,
    metrics,
    args.force,
)

xgb_test_path = DATA_DIR / "stack_test_xgboost.npy"
xgb_final_path = DATA_DIR / "stack_xgboost_final.json"
if xgb_test_path.exists() and xgb_final_path.exists() and not args.force:
    xgb_test = np.load(xgb_test_path)
else:
    best_iterations = [
        row["best_iteration"]
        for row in metrics["xgboost_folds"].values()
    ]
    final_iterations = min(
        XGB_PARAMS["n_estimators"],
        int(np.ceil(np.median(best_iterations) * 1.20)),
    )
    print(f"Training final XGBoost: {final_iterations} trees", flush=True)
    final_xgb = xgb.XGBClassifier(
        **{
            **XGB_PARAMS,
            "n_estimators": final_iterations,
            "early_stopping_rounds": None,
            "random_state": 2026,
        }
    )
    final_xgb.fit(train[xgb_features], y, verbose=100)
    xgb_test = final_xgb.predict_proba(test[xgb_features])[:, 1]
    final_xgb.save_model(xgb_final_path)
    np.save(xgb_test_path, xgb_test)
    metrics["xgboost_final_iterations"] = final_iterations
    save_metrics(metrics)
    del final_xgb
    gc.collect()

print("\nStage 4/4: neural meta-learner...", flush=True)
oof = merge_oof(cat_oof, lgb_oof, xgb_oof)
oof.to_csv(DATA_DIR / "boost3_oof_predictions.csv", index=False)
base_test = pd.DataFrame(
    {
        "catboost": cat_test,
        "lightgbm": lgb_test,
        "xgboost": xgb_test,
    }
)
meta_features = build_meta_features(oof, fold=oof["fold"])
test_meta_features = build_meta_features(base_test)
del (
    train,
    test,
    transaction_dt,
    cat_oof,
    lgb_oof,
    xgb_oof,
    feature_columns,
    categorical_columns,
)
gc.collect()

meta_train_mask = oof["fold"] < 2
meta_valid_mask = oof["fold"] == 2
y_meta_train = oof.loc[meta_train_mask, TARGET].to_numpy(dtype="float32")
y_meta_valid = oof.loc[meta_valid_mask, TARGET].to_numpy(dtype="float32")

scaler = StandardScaler()
X_meta_train = scaler.fit_transform(
    meta_features.loc[meta_train_mask]
).astype("float32")
X_meta_valid = scaler.transform(
    meta_features.loc[meta_valid_mask]
).astype("float32")

linear = LogisticRegression(
    C=0.20,
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED,
)
linear.fit(X_meta_train, y_meta_train)
linear_valid = linear.predict_proba(X_meta_valid)[:, 1]

mlp_valid_parts = []
best_epochs = []
for seed in META_SEEDS:
    model, epoch, prediction, score = train_mlp_with_validation(
        X_meta_train,
        y_meta_train,
        X_meta_valid,
        y_meta_valid,
        seed,
    )
    mlp_valid_parts.append(prediction)
    best_epochs.append(epoch)
    metrics.setdefault("meta_seed_results", {})[str(seed)] = {
        "best_epoch": epoch,
        "valid_auc": score,
    }
    del model
    gc.collect()

mlp_valid = np.mean(mlp_valid_parts, axis=0)
rank_average_valid = oof.loc[
    meta_valid_mask,
    ["catboost", "lightgbm", "xgboost"],
].rank(pct=True).mean(axis=1).to_numpy()
giba_reference_valid = (
    0.775
    * rank_prediction(oof.loc[meta_valid_mask, "catboost"].to_numpy())
    + 0.225
    * rank_prediction(oof.loc[meta_valid_mask, "lightgbm"].to_numpy())
)
meta_blend = best_rank_blend(
    y_meta_valid,
    mlp_valid,
    linear_valid,
)
meta_blend_valid = (
    meta_blend["neural_weight"] * rank_prediction(mlp_valid)
    + meta_blend["linear_weight"] * rank_prediction(linear_valid)
)

valid_indices = oof.loc[meta_valid_mask, "row_index"].astype("int64")
uid_best, uid_candidates = choose_uid_postprocess(
    y_meta_valid,
    meta_blend_valid,
    uid_train.loc[valid_indices],
)
neural_uid_best, neural_uid_candidates = choose_uid_postprocess(
    y_meta_valid,
    rank_prediction(mlp_valid),
    uid_train.loc[valid_indices],
)

metrics["meta_validation"] = {
    "rows": int(meta_valid_mask.sum()),
    "catboost_auc": roc_auc_score(
        y_meta_valid,
        oof.loc[meta_valid_mask, "catboost"],
    ),
    "lightgbm_auc": roc_auc_score(
        y_meta_valid,
        oof.loc[meta_valid_mask, "lightgbm"],
    ),
    "xgboost_auc": roc_auc_score(
        y_meta_valid,
        oof.loc[meta_valid_mask, "xgboost"],
    ),
    "rank_average_auc": roc_auc_score(
        y_meta_valid,
        rank_average_valid,
    ),
    "giba_cat_lgb_reference_auc": roc_auc_score(
        y_meta_valid,
        giba_reference_valid,
    ),
    "linear_meta_auc": roc_auc_score(y_meta_valid, linear_valid),
    "neural_meta_auc": roc_auc_score(y_meta_valid, mlp_valid),
    "neural_linear_blend": meta_blend,
    "uid_postprocess": uid_best,
    "neural_only_uid_postprocess": neural_uid_best,
    "gain_over_giba_reference": (
        uid_best["auc"]
        - roc_auc_score(y_meta_valid, giba_reference_valid)
    ),
}
metrics["uid_postprocess_candidates"] = uid_candidates
metrics["neural_only_uid_postprocess_candidates"] = neural_uid_candidates
save_metrics(metrics)
print(json.dumps(metrics["meta_validation"], indent=2), flush=True)

all_scaler = StandardScaler()
X_meta_all = all_scaler.fit_transform(meta_features).astype("float32")
X_meta_test = all_scaler.transform(test_meta_features).astype("float32")
y_meta_all = oof[TARGET].to_numpy(dtype="float32")
np.savez(
    DATA_DIR / "boost3_meta_scaler.npz",
    mean=all_scaler.mean_,
    scale=all_scaler.scale_,
    features=np.array(meta_features.columns),
)

final_linear = LogisticRegression(
    C=0.20,
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED,
)
final_linear.fit(X_meta_all, y_meta_all)
linear_test = final_linear.predict_proba(X_meta_test)[:, 1]

final_epochs = {
    str(seed): max(1, int(epoch))
    for seed, epoch in zip(META_SEEDS, best_epochs)
}
mlp_test_parts = []
for seed in META_SEEDS:
    model, prediction = train_mlp_fixed_epochs(
        X_meta_all,
        y_meta_all,
        X_meta_test,
        seed,
        final_epochs[str(seed)],
    )
    torch.save(
        model.state_dict(),
        DATA_DIR / f"boost3_meta_mlp_seed_{seed}.pt",
    )
    mlp_test_parts.append(prediction)
    del model
    gc.collect()

mlp_test = np.mean(mlp_test_parts, axis=0)
neural_only_test = rank_prediction(mlp_test)
neural_only_uid_test = apply_uid_postprocess(
    neural_only_test,
    uid_test,
    neural_uid_best["method"],
    neural_uid_best["weight"],
)
stack_test = (
    meta_blend["neural_weight"] * rank_prediction(mlp_test)
    + meta_blend["linear_weight"] * rank_prediction(linear_test)
)
stack_uid_test = apply_uid_postprocess(
    stack_test,
    uid_test,
    uid_best["method"],
    uid_best["weight"],
)

raw_submission = pd.DataFrame(
    {
        "TransactionID": test_ids,
        TARGET: stack_test,
    }
)
raw_submission.to_csv(
    DATA_DIR / "submission_boost3_neural_raw.csv",
    index=False,
)
neural_only_submission = raw_submission.copy()
neural_only_submission[TARGET] = neural_only_uid_test
neural_only_submission.to_csv(
    DATA_DIR / "submission_boost3_neural_only.csv",
    index=False,
)
submission = raw_submission.copy()
submission[TARGET] = stack_uid_test
submission.to_csv(
    DATA_DIR / "submission_boost3_neural_stack.csv",
    index=False,
)

metrics["final_meta_epochs"] = final_epochs
metrics["elapsed_minutes"] = (time.time() - started_at) / 60
metrics["submissions"] = [
    "submission_boost3_neural_stack.csv",
    "submission_boost3_neural_only.csv",
    "submission_boost3_neural_raw.csv",
]
save_metrics(metrics)

print("\nBoost3 neural stack complete", flush=True)
print(json.dumps(metrics, indent=2, default=json_default), flush=True)

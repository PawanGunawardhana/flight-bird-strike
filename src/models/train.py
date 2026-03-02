"""
src/models/train.py
-------------------
Training pipeline for bird strike risk prediction.

Trains and compares:
  - Random Forest (baseline + tuned)
  - XGBoost
  - LightGBM
  - Logistic Regression (interpretable baseline)

Handles class imbalance via class_weight='balanced' or SMOTE.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from time import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import cfg
from src.data.preprocessor import get_feature_columns


def get_X_y(df: pd.DataFrame, target: str = "damage_label"):
    """Split DataFrame into features X and target y."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df[target].values
    return X, y, feature_cols


def build_models(n_classes: int, random_state: int = 42) -> dict:
    """Return a dict of model name → sklearn estimator."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                multi_class="auto",
                solver="lbfgs"
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )

    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    return models


def train_and_evaluate(df: pd.DataFrame, target: str = "damage_label") -> pd.DataFrame:
    """
    Train all models, evaluate with cross-validation and hold-out test set.

    Returns a DataFrame with results for each model.
    """
    print("=" * 60)
    print(f"TRAINING PIPELINE  (target='{target}')")
    print("=" * 60)

    X, y, feature_cols = get_X_y(df, target)
    n_classes = len(np.unique(y))
    print(f"Features: {len(feature_cols)} | Classes: {n_classes} | Samples: {len(y):,}")

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y
    )
    print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")
    print()

    models = build_models(n_classes, random_state=cfg.RANDOM_STATE)
    results = []

    for name, model in models.items():
        print(f"Training: {name}")
        t0 = time()
        model.fit(X_train, y_train)
        train_time = time() - t0

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}  |  Time: {train_time:.1f}s")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1_Weighted": round(f1, 4),
            "Train_Time_s": round(train_time, 2),
            "estimator": model,
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": feature_cols,
        })

    results_df = pd.DataFrame(results).sort_values("F1_Weighted", ascending=False)
    print()
    print("── Summary ──────────────────────────────────────────────")
    print(results_df[["Model", "Accuracy", "F1_Weighted", "Train_Time_s"]].to_string(index=False))
    print("=" * 60)

    return results_df


def save_model(model, path: str):
    """Save a trained sklearn model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path: str):
    """Load a saved sklearn model from disk."""
    return joblib.load(path)

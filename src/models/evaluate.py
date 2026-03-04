"""
src/models/evaluate.py
----------------------
Evaluation utilities: confusion matrix, classification report,
feature importance, ROC curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import cfg
from src.data.preprocessor import DAMAGE_LEVEL_LABELS


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", save: bool = True):
    """Plot and optionally save a confusion matrix."""
    labels = sorted(set(y_true) | set(y_pred))
    label_names = [DAMAGE_LEVEL_LABELS.get(l, str(l)) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
        path = os.path.join(cfg.FIGURES_DIR, f"{title.replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def plot_feature_importance(model, feature_cols: list, top_n: int = 20,
                             title: str = "Feature Importance", save: bool = True):
    """Plot top N feature importances for tree-based models."""
    # Handle Pipeline (e.g. LogReg wrapped in Pipeline)
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)

    if not hasattr(clf, "feature_importances_"):
        print(f"Model {type(clf).__name__} does not support feature_importances_. Skipping.")
        return

    importances = clf.feature_importances_
    # argsort ascending, take top_n — already in ascending order for barh
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_cols[i] for i in indices],
        importances[indices],
        color="steelblue"
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
        path = os.path.join(cfg.FIGURES_DIR, f"{title.replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def print_classification_report(y_true, y_pred):
    """Print a nicely formatted classification report."""
    labels = sorted(set(y_true))
    target_names = [DAMAGE_LEVEL_LABELS.get(l, str(l)) for l in labels]
    print(classification_report(y_true, y_pred, labels=labels,
                                 target_names=target_names, zero_division=0))


def save_results_csv(results_df: pd.DataFrame, filename: str = "model_comparison.csv"):
    """Save model comparison results to reports/results/."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, filename)
    # Drop non-serialisable columns before saving
    cols = [c for c in results_df.columns if c not in ("estimator", "X_test", "y_test", "feature_cols")]
    results_df[cols].to_csv(path, index=False)
    print(f"Results saved to: {path}")

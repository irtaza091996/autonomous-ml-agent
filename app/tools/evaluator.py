"""
Tool: evaluate_model
Evaluates a trained model — metrics, confusion matrix, feature importance.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from app.config import JOBS_DIR


@tool
def evaluate_model(job_id: str, model_name: str) -> str:
    """
    Evaluate a trained model and generate performance metrics and plots.
    Saves confusion matrix and feature importance charts.

    Args:
        job_id:     Unique job identifier.
        model_name: Model to evaluate (must have been trained first).

    Returns:
        Performance metrics summary.
    """
    model_path = JOBS_DIR / job_id / "models" / f"{model_name}.pkl"
    if not model_path.exists():
        return f"ERROR: Model '{model_name}' not found. Train it first."

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model         = artifact["model"]
    X_test        = artifact["X_test"]
    y_test        = artifact["y_test"]
    feature_names = artifact["feature_names"]

    with open(JOBS_DIR / job_id / "metadata.json") as f:
        meta = json.load(f)
    problem_type = meta["problem_type"]

    eval_dir = JOBS_DIR / job_id / "eval"
    eval_dir.mkdir(exist_ok=True)

    metrics: dict = {"model": model_name, "problem_type": problem_type}

    # ── Classification metrics ─────────────────────────────────────────────────
    if problem_type != "regression":
        y_pred = model.predict(X_test)
        y_prob = None
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass

        acc    = float(accuracy_score(y_test, y_pred))
        f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_min = float(f1_score(y_test, y_pred, average="binary", zero_division=0)) \
                 if len(np.unique(y_test)) == 2 else f1_mac

        roc_auc = None
        if y_prob is not None and len(np.unique(y_test)) == 2:
            try:
                roc_auc = float(roc_auc_score(y_test, y_prob))
            except Exception:
                pass

        metrics.update({
            "accuracy":       round(acc, 4),
            "f1_macro":       round(f1_mac, 4),
            "f1_minority":    round(f1_min, 4),
            "roc_auc":        round(roc_auc, 4) if roc_auc else None,
        })

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        classes = [str(c) for c in sorted(np.unique(y_test))]
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        fig.savefig(eval_dir / f"confusion_matrix_{model_name}.png", dpi=120)
        plt.close(fig)

        metrics["classification_report"] = classification_report(
            y_test, y_pred, zero_division=0
        )

    # ── Regression metrics ─────────────────────────────────────────────────────
    else:
        y_pred = model.predict(X_test)
        metrics.update({
            "r2":   round(float(r2_score(y_test, y_pred)), 4),
            "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        })

    # ── Feature importance ─────────────────────────────────────────────────────
    importance_data = None
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        else:
            imp = None

        if imp is not None and len(feature_names) == len(imp):
            top_n = 15
            idx   = np.argsort(imp)[::-1][:top_n]
            top_features = [(feature_names[i], round(float(imp[i]), 4)) for i in idx]
            importance_data = top_features

            fig, ax = plt.subplots(figsize=(8, 5))
            names  = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            ax.barh(names[::-1], values[::-1], color="#0f3460")
            ax.set_title(f"Feature Importance — {model_name}")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            fig.savefig(eval_dir / f"feature_importance_{model_name}.png", dpi=120)
            plt.close(fig)

            metrics["top_features"] = top_features[:5]
    except Exception:
        pass

    # Save metrics
    all_metrics_path = JOBS_DIR / job_id / "metrics.json"
    all_metrics = {}
    if all_metrics_path.exists():
        with open(all_metrics_path) as f:
            all_metrics = json.load(f)
    all_metrics[model_name] = {k: v for k, v in metrics.items()
                                if k != "classification_report"}
    with open(all_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Build result string
    result = f"Evaluation results for '{model_name}':\n"
    if problem_type != "regression":
        result += (
            f"  Accuracy:     {metrics['accuracy']:.1%}\n"
            f"  F1 (macro):   {metrics['f1_macro']:.3f}\n"
            f"  F1 (minority):{metrics['f1_minority']:.3f}\n"
        )
        if metrics.get("roc_auc"):
            result += f"  ROC-AUC:      {metrics['roc_auc']:.3f}\n"
    else:
        result += (
            f"  R2:   {metrics['r2']:.4f}\n"
            f"  MAE:  {metrics['mae']:.4f}\n"
            f"  RMSE: {metrics['rmse']:.4f}\n"
        )
    if importance_data:
        result += f"  Top 5 features: {importance_data[:5]}\n"

    return result

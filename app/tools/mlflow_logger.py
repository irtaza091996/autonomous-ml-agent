"""
Tool: log_to_mlflow
Logs the best model metrics and artifacts to MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from langchain_core.tools import tool

from app.config import JOBS_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


@tool
def log_to_mlflow(job_id: str, best_model: str) -> str:
    """
    Log the best model's metrics and artifacts to MLflow for experiment tracking.
    Call this after evaluating all models and identifying the best one.

    Args:
        job_id:     Unique job identifier.
        best_model: Name of the best performing model to log.

    Returns:
        Confirmation with MLflow run ID.
    """
    metrics_path = JOBS_DIR / job_id / "metrics.json"
    metadata_path = JOBS_DIR / job_id / "metadata.json"

    if not metrics_path.exists():
        return "ERROR: No metrics found. Evaluate models first."

    with open(metrics_path) as f:
        all_metrics = json.load(f)
    with open(metadata_path) as f:
        meta = json.load(f)

    if best_model not in all_metrics:
        return (
            f"ERROR: No metrics for '{best_model}'. "
            f"Available: {list(all_metrics.keys())}"
        )

    metrics = all_metrics[best_model]

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"{job_id}-{best_model}") as run:
            # Params
            mlflow.log_params({
                "job_id":        job_id,
                "model":         best_model,
                "problem_type":  meta["problem_type"],
                "target_column": meta["target_column"],
                "n_rows":        meta["shape"][0],
                "n_features":    meta["shape"][1] - 1,
            })

            # Metrics (skip non-numeric)
            log_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None
            }
            mlflow.log_metrics(log_metrics)

            # Artifacts — plots
            eval_dir = JOBS_DIR / job_id / "eval"
            if eval_dir.exists():
                for png in eval_dir.glob(f"*{best_model}*.png"):
                    mlflow.log_artifact(str(png), artifact_path="plots")

            eda_dir = JOBS_DIR / job_id / "eda"
            if eda_dir.exists():
                for png in eda_dir.glob("*.png"):
                    mlflow.log_artifact(str(png), artifact_path="eda")

            if metrics_path.exists():
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            run_id = run.info.run_id

        return (
            f"Logged to MLflow successfully.\n"
            f"Experiment: {MLFLOW_EXPERIMENT}\n"
            f"Run ID: {run_id}\n"
            f"Model: {best_model}\n"
            f"Metrics logged: {list(log_metrics.keys())}"
        )
    except Exception as e:
        return f"MLflow logging failed (non-fatal): {e}"

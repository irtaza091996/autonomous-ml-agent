"""
Tool: load_dataset
Loads a CSV, infers problem type, detects imbalance, saves metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from app.config import JOBS_DIR, UPLOADS_DIR


def _infer_problem_type(series: pd.Series) -> str:
    n_unique = series.nunique()
    if n_unique == 2:
        return "binary_classification"
    if n_unique <= 20 and series.dtype in ["object", "int64", "int32"]:
        return "multiclass_classification"
    return "regression"


@tool
def load_dataset(job_id: str, target_column: str = "auto") -> str:
    """
    Load the uploaded CSV dataset for a job and return a summary.

    Args:
        job_id:        Unique job identifier.
        target_column: Name of the target/label column. Use 'auto' to detect automatically.

    Returns:
        A text summary of the dataset suitable for planning the ML pipeline.
    """
    dataset_path = UPLOADS_DIR / job_id / "dataset.csv"
    if not dataset_path.exists():
        return f"ERROR: Dataset not found at {dataset_path}. Upload a CSV first."

    df = pd.read_csv(dataset_path)

    # Auto-detect target column
    if target_column == "auto":
        for candidate in ["target", "label", "y", "class", "outcome", "both_correct"]:
            if candidate in df.columns:
                target_column = candidate
                break
        else:
            target_column = df.columns[-1]

    if target_column not in df.columns:
        return (
            f"ERROR: Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    problem_type   = _infer_problem_type(df[target_column])
    missing_total  = int(df.isnull().sum().sum())
    missing_per_col = df.isnull().sum()[df.isnull().sum() > 0].to_dict()

    class_dist: dict = {}
    imbalance_warning = ""
    if problem_type != "regression":
        class_dist = (
            df[target_column].value_counts(normalize=True).round(4).to_dict()
        )
        class_dist = {str(k): float(v) for k, v in class_dist.items()}
        min_ratio = min(class_dist.values())
        if min_ratio < 0.15:
            imbalance_warning = (
                f"\nWARNING: Class imbalance detected — "
                f"minority class ratio = {min_ratio:.1%}. "
                "Consider SMOTE oversampling."
            )

    numeric_cols     = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(include="object").columns)

    # Save metadata for other tools
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    metadata = {
        "job_id":            job_id,
        "target_column":     target_column,
        "problem_type":      problem_type,
        "shape":             list(df.shape),
        "columns":           list(df.columns),
        "dtypes":            df.dtypes.astype(str).to_dict(),
        "missing_values":    {str(k): int(v) for k, v in missing_per_col.items()},
        "missing_total":     missing_total,
        "class_distribution": class_dist,
        "numeric_columns":   numeric_cols,
        "categorical_columns": categorical_cols,
    }
    with open(job_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return (
        f"Dataset loaded successfully.\n"
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Target column: '{target_column}'\n"
        f"Problem type: {problem_type}\n"
        f"Missing values: {missing_total}"
        + (f" (in columns: {list(missing_per_col.keys())})" if missing_per_col else "")
        + f"\nNumeric features: {len(numeric_cols)}\n"
        f"Categorical features: {len(categorical_cols)}\n"
        f"Class distribution: {class_dist}"
        + imbalance_warning
    )

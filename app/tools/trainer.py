"""
Tool: train_model
Trains a specified ML model on the dataset, handles preprocessing and SMOTE.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.config import JOBS_DIR, RANDOM_SEED, TEST_SIZE, UPLOADS_DIR

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=200, random_state=RANDOM_SEED,
        eval_metric="logloss", verbosity=0
    ) if XGBOOST_AVAILABLE else RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED
    ),
}

REGRESSORS = {
    "ridge":          lambda: Ridge(random_state=RANDOM_SEED),
    "random_forest":  lambda: RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "xgboost":        lambda: XGBRegressor(
        n_estimators=200, random_state=RANDOM_SEED, verbosity=0
    ) if XGBOOST_AVAILABLE else RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_SEED
    ),
}


def _preprocess(df: pd.DataFrame, target_col: str, meta: dict):
    """Encode categoricals, impute missing, return X, y, feature names."""
    df = df.copy()

    # Drop columns with too many missing values (>50%)
    drop_cols = [c for c in df.columns
                 if df[c].isnull().mean() > 0.5 and c != target_col]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Encode target for classification
    le = None
    if y.dtype == "object" or str(y.dtype) == "bool":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # Impute missing
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == "object":
                X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown",
                              inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)

    # Encode categoricals
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        lenc = LabelEncoder()
        X[col] = lenc.fit_transform(X[col].astype(str))

    feature_names = list(X.columns)
    return X.values, np.array(y), feature_names, le


@tool
def train_model(job_id: str, model_name: str, use_smote: bool = False) -> str:
    """
    Train a machine learning model on the dataset.

    Args:
        job_id:     Unique job identifier.
        model_name: One of 'logistic_regression', 'random_forest', 'xgboost',
                    'ridge' (regression only).
        use_smote:  If True, apply SMOTE oversampling to handle class imbalance
                    (classification only).

    Returns:
        Training summary with train/val split sizes and model info.
    """
    metadata_path = JOBS_DIR / job_id / "metadata.json"
    if not metadata_path.exists():
        return "ERROR: Run load_dataset first."

    with open(metadata_path) as f:
        meta = json.load(f)

    problem_type = meta["problem_type"]
    target_col   = meta["target_column"]

    df = pd.read_csv(UPLOADS_DIR / job_id / "dataset.csv")
    X, y, feature_names, le = _preprocess(df, target_col, meta)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=y if problem_type != "regression" else None,
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE
    smote_applied = False
    if use_smote and problem_type != "regression" and SMOTE_AVAILABLE:
        try:
            sm = SMOTE(random_state=RANDOM_SEED)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            smote_applied = True
        except Exception as e:
            pass

    # Select and train model
    if problem_type == "regression":
        model_map = REGRESSORS
    else:
        model_map = CLASSIFIERS

    if model_name not in model_map:
        available = list(model_map.keys())
        return f"ERROR: Unknown model '{model_name}'. Available: {available}"

    model = model_map[model_name]()

    # XGBoost class weight for imbalance
    if model_name == "xgboost" and problem_type != "regression" and not smote_applied:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos > 0 and len(np.unique(y_train)) == 2:
            model.set_params(scale_pos_weight=n_neg / n_pos)

    model.fit(X_train, y_train)

    # Save model artifacts
    models_dir = JOBS_DIR / job_id / "models"
    models_dir.mkdir(exist_ok=True)

    artifact = {
        "model":         model,
        "scaler":        scaler,
        "feature_names": feature_names,
        "label_encoder": le,
        "X_test":        X_test,
        "y_test":        y_test,
    }
    with open(models_dir / f"{model_name}.pkl", "wb") as f:
        pickle.dump(artifact, f)

    return (
        f"Model '{model_name}' trained successfully.\n"
        f"Problem type: {problem_type}\n"
        f"Training samples: {len(X_train)}"
        + (" (after SMOTE)" if smote_applied else "")
        + f"\nValidation samples: {len(X_test)}\n"
        f"Features used: {len(feature_names)}\n"
        f"SMOTE applied: {smote_applied}\n"
        "Call evaluate_model to get metrics."
    )

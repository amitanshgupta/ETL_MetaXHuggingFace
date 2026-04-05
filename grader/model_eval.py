"""
grader/model_eval.py
Trains a simple model on cleaned data, scores against ground truth.
Used as the ML performance component of the final grader score.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def model_performance_score(
    cleaned_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",  # "classification" or "regression"
) -> float:
    """
    Trains a RandomForest on cleaned_df, evaluates on clean_df.
    Returns normalized score in [0, 1].
    """
    try:
        X_clean, y_clean = _prepare(clean_df, target_col)
        X_pred, y_pred = _prepare(cleaned_df, target_col)

        if len(X_pred) < 10:
            return 0.0

        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(model, X_pred, y_pred, cv=3, scoring="accuracy")
            baseline = cross_val_score(model, X_clean, y_clean, cv=3, scoring="accuracy")
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            scores = cross_val_score(model, X_pred, y_pred, cv=3, scoring="r2")
            baseline = cross_val_score(model, X_clean, y_clean, cv=3, scoring="r2")

        # Normalize: how close is cleaned performance to ground truth performance
        pred_score = max(scores.mean(), 0.0)
        base_score = max(baseline.mean(), 1e-6)
        return round(min(pred_score / base_score, 1.0), 4)

    except Exception:
        return 0.0


def _prepare(df: pd.DataFrame, target_col: str):
    df = df.copy().dropna()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categoricals
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    return X, y
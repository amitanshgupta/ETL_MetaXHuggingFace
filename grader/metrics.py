"""
grader/metrics.py
Individual scoring metrics used by the Grader.
All return float in [0, 1].
"""

import pandas as pd
import numpy as np


def schema_match_score(cleaned_df: pd.DataFrame, expected_schema: dict[str, str]) -> float:
    """
    Fraction of expected columns that exist AND have correct dtype.
    """
    if not expected_schema:
        return 1.0
    matches = sum(
        1 for col, dtype in expected_schema.items()
        if col in cleaned_df.columns and dtype in str(cleaned_df[col].dtype)
    )
    return round(matches / len(expected_schema), 4)


def value_similarity_score(cleaned_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Column-wise similarity between cleaned and ground truth.
    Numeric columns: 1 - normalized MAE
    Categorical columns: exact match rate
    Averaged across all common columns.
    """
    common_cols = [c for c in clean_df.columns if c in cleaned_df.columns]
    if not common_cols:
        return 0.0

    scores = []
    for col in common_cols:
        try:
            gt = clean_df[col]
            pred = cleaned_df[col]

            if pd.api.types.is_numeric_dtype(gt):
                gt_num = pd.to_numeric(gt, errors="coerce")
                pred_num = pd.to_numeric(pred, errors="coerce")
                mae = (gt_num - pred_num).abs().mean()
                norm = gt_num.abs().mean()
                score = 1.0 - min(mae / norm if norm > 0 else 0.0, 1.0)
            else:
                gt_str = gt.astype(str).str.strip().str.lower()
                pred_str = pred.astype(str).str.strip().str.lower()
                score = (gt_str == pred_str).mean()

            scores.append(float(score))
        except Exception:
            scores.append(0.0)

    return round(np.mean(scores), 4)


def completeness_score(cleaned_df: pd.DataFrame) -> float:
    """Fraction of non-null values across entire dataframe."""
    return round(1.0 - cleaned_df.isna().mean().mean(), 4)


def duplicate_score(cleaned_df: pd.DataFrame) -> float:
    """Fraction of non-duplicate rows."""
    if len(cleaned_df) == 0:
        return 1.0
    return round(1.0 - cleaned_df.duplicated().sum() / len(cleaned_df), 4)
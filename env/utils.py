"""
env/utils.py
Shared helpers used across observation, reward, and grader.
"""

import pandas as pd


def compute_validity(df: pd.DataFrame, expected_schema: dict[str, str]) -> float:
    """Fraction of expected columns whose dtype matches expected."""
    if not expected_schema:
        return 1.0
    matches = sum(
        1 for col, dtype in expected_schema.items()
        if col in df.columns and dtype in str(df[col].dtype)
    )
    return matches / len(expected_schema)


def compute_consistency(df: pd.DataFrame, expected_schema: dict[str, str]) -> float:
    """Fraction of expected columns present in df."""
    if not expected_schema:
        return 1.0
    present = sum(1 for col in expected_schema if col in df.columns)
    return present / len(expected_schema)


def missing_rate(df: pd.DataFrame) -> float:
    return df.isna().mean().mean()


def duplicate_rate(df: pd.DataFrame) -> float:
    return df.duplicated().sum() / len(df) if len(df) > 0 else 0.0
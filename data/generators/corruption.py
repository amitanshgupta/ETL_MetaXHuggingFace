"""
data/generators/corruption.py
Applies deterministic corruption to clean datasets.
Seed is fixed — results are always reproducible.
Run after dataset_loader.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

CLEAN = Path("data/clean")
DIRTY = Path("data/dirty")
DIRTY.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(seed=42)  # fixed seed = deterministic


def inject_missing(df: pd.DataFrame, cols: list[str], frac: float = 0.15) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        idx = RNG.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


def inject_type_mismatch(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype(object)  # allow mixed types
        idx = RNG.choice(df.index, size=int(len(df) * 0.10), replace=False)
        df.loc[idx, col] = "N/A"
    return df


def inject_duplicates(df: pd.DataFrame, frac: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    dupes = df.sample(frac=frac, random_state=42)
    return pd.concat([df, dupes], ignore_index=True)


def inject_noise(df: pd.DataFrame, cols: list[str], std_frac: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        noise = RNG.normal(0, numeric.std() * std_frac, size=len(df))
        df[col] = (numeric + noise).round(2)
    return df


def corrupt_titanic():
    df = pd.read_csv(CLEAN / "titanic_clean.csv")
    df = inject_missing(df, ["Age", "Fare", "Embarked"], frac=0.20)
    df = inject_duplicates(df, frac=0.05)
    df.to_csv(DIRTY / "titanic_dirty.csv", index=False)
    print(f"Titanic dirty: {df.shape}")


def corrupt_house_prices():
    df = pd.read_csv(CLEAN / "house_prices_clean.csv")
    df = inject_missing(df, ["TotalBsmtSF", "GarageCars"], frac=0.15)
    df = inject_type_mismatch(df, ["OverallQual", "YearBuilt"])
    df = inject_noise(df, ["SalePrice", "LotArea"], std_frac=0.03)
    df.to_csv(DIRTY / "house_prices_dirty.csv", index=False)
    print(f"House prices dirty: {df.shape}")


def corrupt_olist():
    df = pd.read_csv(CLEAN / "olist_clean.csv")
    df = inject_missing(df, ["customer_city", "customer_state", "price"], frac=0.15)
    df = inject_duplicates(df, frac=0.08)
    df = inject_type_mismatch(df, ["price", "freight_value"])
    df.to_csv(DIRTY / "olist_dirty.csv", index=False)
    print(f"Olist dirty: {df.shape}")


if __name__ == "__main__":
    corrupt_titanic()
    corrupt_house_prices()
    corrupt_olist()
    print("All dirty datasets saved to data/dirty/")
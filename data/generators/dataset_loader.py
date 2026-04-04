"""
data/generators/dataset_loader.py
Loads raw CSVs, does minimal cleaning, saves as ground truth to data/clean/
Run this ONCE before anything else.
"""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)


def prepare_titanic():
    df = pd.read_csv(RAW / "titanic.csv")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Cabin"] = df["Cabin"].fillna("Unknown")
    df.to_csv(CLEAN / "titanic_clean.csv", index=False)
    print(f"Titanic clean: {df.shape}")


def prepare_house_prices():
    df = pd.read_csv(RAW / "house_prices_train.csv")
    # Keep only schema columns defined in YAML
    cols = ["Id","MSSubClass","LotArea","OverallQual","OverallCond",
            "YearBuilt","TotalBsmtSF","GrLivArea","FullBath",
            "BedroomAbvGr","TotRmsAbvGrd","GarageCars","SalePrice"]
    df = df[cols]
    df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(0.0)
    df["GarageCars"] = df["GarageCars"].fillna(0.0)
    df.to_csv(CLEAN / "house_prices_clean.csv", index=False)
    print(f"House prices clean: {df.shape}")


def prepare_olist():
    orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")

    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(items[["order_id","order_item_id","price","freight_value"]], on="order_id", how="left")

    cols = ["order_id","customer_id","order_status","order_purchase_timestamp",
            "customer_city","customer_state","order_item_id","price","freight_value"]
    df = df[cols].dropna(subset=["order_id","customer_id"])
    df.to_csv(CLEAN / "olist_clean.csv", index=False)
    print(f"Olist clean: {df.shape}")


if __name__ == "__main__":
    prepare_titanic()
    prepare_house_prices()
    prepare_olist()
    print("All clean datasets saved to data/clean/")
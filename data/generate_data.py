"""
data/generate_data.py
---------------------
Generates synthetic retail sales data that simulates a real-world
store transaction database export.

WHY THIS EXISTS:
  In a real company, this step would be a SQL query or API call pulling
  live data from a data warehouse. Since we are building a learning project,
  we simulate a realistic data source instead.

HOW TO RUN:
  python data/generate_data.py

OUTPUT:
  data/raw/sales_data.csv  (~36,000 rows)

WHAT MAKES THE DATA REALISTIC:
  - Weekend sales are 30–50% higher than weekdays
  - Promotions boost sales by 20–60%
  - Snacks & Drinks spike in summer months (May–Aug)
  - Dairy products dip slightly in summer
  - Each row has a 5% chance of a missing value (to practice data cleaning)
  - Some stores are busier than others (store multiplier)
"""

import sys
from pathlib import Path

# Make sure imports work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import (
    DATA_START_DATE,
    DATA_END_DATE,
    STORES,
    PRODUCTS,
    RAW_DATA_PATH,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ── Store multipliers (some stores are busier) ────────────────────────────────
STORE_MULTIPLIER = {
    "store_1": 1.0,
    "store_2": 1.3,
    "store_3": 0.8,
    "store_4": 1.1,
    "store_5": 0.95,
}


def compute_seasonal_factor(month: int, category: str) -> float:
    """
    Returns a multiplier based on the product category and month.
    - Snacks & Drinks sell more in summer
    - Dairy sells slightly less in summer
    - Other categories stay roughly flat
    """
    summer_months = {5, 6, 7, 8}
    winter_months = {11, 12, 1, 2}

    if category in ("Snacks", "Drinks"):
        return 1.35 if month in summer_months else 1.0
    elif category == "Dairy":
        return 0.85 if month in summer_months else 1.05
    elif category == "Care":
        return 1.1 if month in winter_months else 1.0
    else:
        return 1.0


def generate_row(
    date: pd.Timestamp,
    store: str,
    product: str,
    product_info: dict,
) -> dict:
    """
    Generates a single sales row for one (date, store, product) combination.
    Applies weekend, promotion, seasonal, and store-level effects.
    """
    base_sales = product_info["base_sales"]
    category   = product_info["category"]
    price      = product_info["base_price"]

    # ── Weekend boost ─────────────────────────────────────────────────────────
    weekend_factor = 1.4 if date.dayofweek >= 5 else 1.0

    # ── Promotion: 20% of days have a promotion ───────────────────────────────
    on_promotion = int(np.random.rand() < 0.20)
    promo_factor = np.random.uniform(1.2, 1.6) if on_promotion else 1.0

    # ── Seasonal factor ───────────────────────────────────────────────────────
    seasonal_factor = compute_seasonal_factor(date.month, category)

    # ── Store traffic factor ──────────────────────────────────────────────────
    store_factor = STORE_MULTIPLIER[store]

    # ── Final sales with Gaussian noise (realistic variance) ──────────────────
    expected_sales = (
        base_sales
        * weekend_factor
        * promo_factor
        * seasonal_factor
        * store_factor
    )
    units_sold = max(0, int(np.random.normal(expected_sales, expected_sales * 0.15)))

    # ── Price with small variance day-to-day ──────────────────────────────────
    actual_price = round(price * np.random.uniform(0.95, 1.05), 2)

    return {
        "date":        date.strftime("%Y-%m-%d"),
        "store_id":    store,
        "product":     product,
        "category":    category,
        "price":       actual_price,
        "promotion":   on_promotion,
        "units_sold":  units_sold,
    }


def introduce_missing_values(df: pd.DataFrame, missing_rate: float = 0.05) -> pd.DataFrame:
    """
    Randomly nullify ~5% of values in selected columns.
    This simulates real-world data quality issues that our
    preprocessing pipeline will need to handle.
    """
    cols_to_corrupt = ["price", "units_sold", "promotion"]
    df = df.copy()

    for col in cols_to_corrupt:
        mask = np.random.rand(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    return df


def generate_sales_data() -> pd.DataFrame:
    """
    Main function: iterates over every (date, store, product) combination
    and generates a sales row for each. Returns the full DataFrame.
    """
    date_range = pd.date_range(start=DATA_START_DATE, end=DATA_END_DATE, freq="D")

    rows = []
    total = len(date_range) * len(STORES) * len(PRODUCTS)

    print(f"Generating {total:,} rows of synthetic sales data…")

    for date in date_range:
        for store in STORES:
            for product, product_info in PRODUCTS.items():
                row = generate_row(date, store, product, product_info)
                rows.append(row)

    df = pd.DataFrame(rows)
    df = introduce_missing_values(df, missing_rate=0.05)
    return df


def main():
    # Ensure output directory exists
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_sales_data()

    # Save to CSV
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"\n✅ Data saved to: {RAW_DATA_PATH}")
    print(f"   Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Date range   : {df['date'].min()} → {df['date'].max()}")
    print(f"   Stores       : {df['store_id'].nunique()}")
    print(f"   Products     : {df['product'].nunique()}")
    print(f"\n   Sample (first 5 rows):")
    print(df.head().to_string(index=False))

    # Quick missing-value report
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n   Missing values introduced (for ETL practice):")
    for col, count in missing.items():
        print(f"     {col}: {count} nulls ({count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()

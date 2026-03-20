"""
src/pipeline/preprocess.py
--------------------------
Step 2 of the ETL pipeline: Data Cleaning & Preprocessing.

WHAT THIS DOES:
  Receives the raw DataFrame from ingest.py and cleans it so the model
  can actually work with it. Real-world data is always messy — this module
  handles all the mess in one place.

  Cleaning steps performed:
    1. Fix data types (ensure date is datetime, numerics are float/int)
    2. Handle missing values (fill or drop depending on column)
    3. Remove duplicates
    4. Remove impossible values (e.g., negative prices)
    5. Standardise string columns (lowercase, strip whitespace)

FUNCTION:
  clean_data(df) → pd.DataFrame

HOW TO RUN STANDALONE:
  python -m src.pipeline.preprocess
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np

from config.settings import CLEAN_DATA_PATH
from src.pipeline.ingest import load_raw_data


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every column has the correct data type.
    parse_dates in read_csv handles 'date', but we still sanitise numerics.
    """
    # Numeric columns should be float (they may have NaN from generation)
    df["price"]      = pd.to_numeric(df["price"],      errors="coerce")
    df["promotion"]  = pd.to_numeric(df["promotion"],  errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy per column:
      - price      → fill with median price PER product (logical default)
      - promotion  → fill with 0 (assume no promotion if unknown)
      - units_sold → drop rows where target is missing
                     (we cannot train on rows with no label)
    """
    before = len(df)

    # Fill price with per-product median
    df["price"] = df.groupby("product")["price"].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill promotion with 0 (conservative default)
    df["promotion"] = df["promotion"].fillna(0).astype(int)

    # Drop rows where target (units_sold) is missing — unusable for training
    df = df.dropna(subset=["units_sold"])

    after = len(df)
    print(f"[preprocess] Dropped {before - after:,} rows with missing target")

    return df


def _remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with physically impossible values:
      - Negative price (data entry error)
      - Negative units_sold (impossible)
    """
    before = len(df)
    df = df[(df["price"] > 0) & (df["units_sold"] >= 0)]
    after = len(df)

    if before > after:
        print(f"[preprocess] Removed {before - after:,} rows with invalid values")

    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    A genuine duplicate would mean the same store sold the same product
    on the same date twice — which shouldn't happen in clean data.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["date", "store_id", "product"])
    after = len(df)

    if before > after:
        print(f"[preprocess] Dropped {before - after:,} duplicate rows")

    return df


def _standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase and strip whitespace from string columns.
    Prevents issues like 'Milk' vs 'milk' vs ' Milk ' being treated differently.
    """
    str_cols = ["store_id", "product", "category"]
    for col in str_cols:
        df[col] = df[col].str.strip().str.lower()
    return df


def _cast_final_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final type cast after cleaning.
    units_sold is our target — cast to int now that NaN rows are gone.
    """
    df["units_sold"] = df["units_sold"].astype(int)
    df["promotion"]  = df["promotion"].astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all cleaning steps in order and returns a clean DataFrame.

    Pipeline:
      fix dtypes → handle missing → remove invalids → remove duplicates
      → standardise strings → final cast

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from ingest.load_raw_data()

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame, ready for feature engineering.
    """
    print(f"[preprocess] Starting cleaning — input shape: {df.shape}")

    df = _fix_dtypes(df)
    df = _handle_missing_values(df)
    df = _remove_invalid_rows(df)
    df = _remove_duplicates(df)
    df = _standardise_strings(df)
    df = _cast_final_types(df)

    # Reset index after all drops
    df = df.reset_index(drop=True)

    print(f"[preprocess] Cleaning complete — output shape: {df.shape}")
    print(f"[preprocess] Remaining missing values: {df.isnull().sum().sum()}")

    return df


def save_clean_data(df: pd.DataFrame) -> None:
    """Saves the cleaned DataFrame to data/processed/sales_clean.csv"""
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"[preprocess] Saved clean data → {CLEAN_DATA_PATH}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_df   = load_raw_data()
    clean_df = clean_data(raw_df)
    save_clean_data(clean_df)

    print("\nCleaned sample (5 rows):")
    print(clean_df.head().to_string(index=False))
    print("\nData types:")
    print(clean_df.dtypes)

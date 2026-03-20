"""
src/pipeline/features.py
------------------------
Step 3 of the ETL pipeline: Feature Engineering.

WHAT THIS DOES:
  Takes the clean DataFrame and creates NEW columns that help the ML model
  learn patterns better. Raw columns like 'date' carry almost no signal —
  but breaking it into 'day_of_week', 'month', 'is_weekend' gives the model
  real patterns to learn from.

  Features created:
    TIME-BASED:
      - day_of_week      (0=Mon … 6=Sun)
      - month            (1–12)
      - quarter          (1–4)
      - is_weekend       (1 if Sat/Sun, else 0)
      - day_of_year      (1–365, captures seasonality)
      - week_of_year     (1–52)

    LAG FEATURES (using historical sales):
      - sales_lag_7      (same store+product, 7 days ago)
      - sales_lag_14     (14 days ago)
      - sales_lag_28     (28 days ago)

    ROLLING AVERAGES (smooth out daily noise):
      - sales_rolling_7  (7-day rolling mean per store+product)
      - sales_rolling_28 (28-day rolling mean per store+product)

    CATEGORICAL ENCODING:
      - store_id  → integer codes
      - product   → integer codes
      - category  → integer codes

FUNCTION:
  engineer_features(df) → pd.DataFrame

HOW TO RUN STANDALONE:
  python -m src.pipeline.features
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np

from config.settings import CLEAN_DATA_PATH, FEAT_DATA_PATH
from src.pipeline.ingest import load_raw_data
from src.pipeline.preprocess import clean_data, save_clean_data


# ── Time-based features ───────────────────────────────────────────────────────

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar-based features from the date column.
    These help the model learn weekly and seasonal patterns.
    """
    df["day_of_week"]  = df["date"].dt.dayofweek          # 0=Monday, 6=Sunday
    df["month"]        = df["date"].dt.month               # 1–12
    df["quarter"]      = df["date"].dt.quarter             # 1–4
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)
    df["day_of_year"]  = df["date"].dt.dayofyear           # 1–365
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    print("[features] Added time-based features: day_of_week, month, quarter, "
          "is_weekend, day_of_year, week_of_year")
    return df


# ── Lag features ─────────────────────────────────────────────────────────────

def _add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lag and rolling features using a single pivot+stack+merge:
      1. Sort by (date, store_id, product)
      2. Pivot to wide: rows=date, columns=group_key (store__product)
      3. Compute all 5 features at once on the wide DataFrame
      4. Stack all features together → one long DataFrame
      5. Single merge back onto df

    This avoids 5 separate Python-level merge loops and is fast.
    """
    df = df.sort_values(["date", "store_id", "product"]).reset_index(drop=True)
    df["_group"] = df["store_id"].astype(str) + "__" + df["product"].astype(str)

    # ── Pivot ──────────────────────────────────────────────────────────────────
    pivot = (
        df.pivot_table(index="date", columns="_group", values="units_sold", aggfunc="sum")
          .sort_index()
    )

    # ── Compute all features on wide pivot ─────────────────────────────────────
    all_feats = {
        "sales_lag_7":      pivot.shift(7),
        "sales_lag_14":     pivot.shift(14),
        "sales_lag_28":     pivot.shift(28),
        "sales_rolling_7":  pivot.shift(1).rolling(7,  min_periods=1).mean(),
        "sales_rolling_28": pivot.shift(1).rolling(28, min_periods=1).mean(),
    }

    # ── Concatenate into a single MultiLevel DataFrame and stack once ──────────
    combined = pd.concat(all_feats, axis=1)          # cols: (feat_name, group_key)
    combined.columns.names = ["feature", "_group"]
    long = (
        combined
        .stack(level="_group")                        # one row per (date, group)
        .reset_index()                                # cols: date, _group, feat1, feat2 …
    )

    # ── Single merge ───────────────────────────────────────────────────────────
    df = df.merge(long, on=["date", "_group"], how="left")
    df = df.drop(columns=["_group"])

    print("[features] Added: sales_lag_7/14/28, sales_rolling_7/28")
    return df


# ── Categorical encoding ──────────────────────────────────────────────────────

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string columns to integer codes.

    WHY: ML models work with numbers, not strings. 'store_1' → 0, 'store_2' → 1 etc.
    We use pandas Categorical codes — simple, deterministic, and reproducible.
    """
    for col in ["store_id", "product", "category"]:
        df[col] = pd.Categorical(df[col]).codes

    print("[features] Encoded categoricals: store_id, product, category → int codes")
    return df


# ── Fill NaN from lag/rolling ─────────────────────────────────────────────────

def _fill_lag_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag and rolling features produce NaN for the first N rows of each group
    (because there's no history yet). Fill with 0 — the model will learn
    to discount these rows.
    """
    lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
    df[lag_cols] = df[lag_cols].fillna(0)
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all feature engineering steps and returns the final feature DataFrame.

    Pipeline:
      time features → lag features → rolling averages
      → categorical encoding → fill NaN → drop raw date column

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from preprocess.clean_data()

    Returns
    -------
    pd.DataFrame
        Feature-rich DataFrame ready for model training.
        Original 'date' column is dropped (replaced by numeric time features).
    """
    print(f"[features] Starting feature engineering — input shape: {df.shape}")

    df = _add_time_features(df)
    df = _add_lag_and_rolling_features(df)
    df = _encode_categoricals(df)
    df = _fill_lag_nans(df)

    # Drop the raw date column — the model uses numeric time features instead
    df = df.drop(columns=["date"])

    print(f"[features] Feature engineering complete — output shape: {df.shape}")
    print(f"[features] Final columns ({len(df.columns)}): {list(df.columns)}")

    return df


def save_feature_data(df: pd.DataFrame) -> None:
    """Saves the feature DataFrame to data/processed/sales_features.csv"""
    FEAT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEAT_DATA_PATH, index=False)
    print(f"[features] Saved feature data → {FEAT_DATA_PATH}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_df   = load_raw_data()
    clean_df = clean_data(raw_df)
    save_clean_data(clean_df)

    feat_df  = engineer_features(clean_df)
    save_feature_data(feat_df)

    print("\nFeature sample (5 rows):")
    print(feat_df.head().to_string(index=False))
    print("\nData types:")
    print(feat_df.dtypes)
    print(f"\nTarget stats:\n{feat_df['units_sold'].describe()}")

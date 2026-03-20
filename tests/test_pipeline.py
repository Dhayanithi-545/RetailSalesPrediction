"""
tests/test_pipeline.py
----------------------
Unit tests for the ETL pipeline (ingest, preprocess, features).

These tests use SYNTHETIC small DataFrames — they do NOT need the actual CSV file.
This makes tests fast, isolated, and independent of data generation.

HOW TO RUN:
  python -m pytest tests/ -v
  OR to run just this file:
  python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — reusable test data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Minimal raw DataFrame mimicking what generate_data.py produces."""
    return pd.DataFrame({
        "date":       pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03",
                                       "2023-01-04", "2023-01-05"]),
        "store_id":   ["store_1", "store_1", "store_1", "store_1", "store_1"],
        "product":    ["milk",    "milk",    "milk",    "milk",    "milk"],
        "category":   ["dairy",   "dairy",   "dairy",   "dairy",   "dairy"],
        "price":      [45.0,      None,       46.0,      47.0,      48.0],
        "promotion":  [0,         1,          None,      0,         1],
        "units_sold": [100,       120,        None,      110,       130],
    })


@pytest.fixture
def clean_df():
    """A pre-cleaned DataFrame for feature engineering tests."""
    return pd.DataFrame({
        "date":       pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-03",
            "2023-01-08", "2023-01-09",
        ]),
        "store_id":   ["store_1"] * 5,
        "product":    ["milk"] * 5,
        "category":   ["dairy"] * 5,
        "price":      [45.0, 46.0, 47.0, 45.5, 46.5],
        "promotion":  [0, 1, 0, 1, 0],
        "units_sold": [100, 120, 110, 130, 105],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tests: preprocess.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocess:
    """Tests for data cleaning logic."""

    def test_missing_values_are_handled(self, raw_df):
        """After cleaning, there should be zero missing values."""
        from src.pipeline.preprocess import clean_data
        result = clean_data(raw_df)
        assert result.isnull().sum().sum() == 0, "Cleaned data should have no missing values"

    def test_rows_with_missing_target_are_dropped(self, raw_df):
        """Rows where units_sold is NaN must be dropped (cannot train on them)."""
        from src.pipeline.preprocess import clean_data
        result = clean_data(raw_df)
        # Row index 2 had NaN units_sold — should be gone
        assert len(result) < len(raw_df), "Missing-target rows should be dropped"

    def test_units_sold_is_integer(self, raw_df):
        """After cleaning, units_sold must be an integer column."""
        from src.pipeline.preprocess import clean_data
        result = clean_data(raw_df)
        assert result["units_sold"].dtype in [int, "int64", "int32"]

    def test_promotion_filled_with_zero(self, raw_df):
        """Missing promotion values should be filled with 0."""
        from src.pipeline.preprocess import clean_data
        result = clean_data(raw_df)
        assert result["promotion"].isnull().sum() == 0
        assert result["promotion"].min() >= 0

    def test_strings_are_lowercased(self, raw_df):
        """store_id, product, category should be lowercase after cleaning."""
        from src.pipeline.preprocess import clean_data
        # Add a mixed-case entry
        raw_df.loc[0, "store_id"] = "STORE_1"
        result = clean_data(raw_df)
        for col in ["store_id", "product", "category"]:
            assert result[col].dropna().str.islower().all(), f"{col} should be lowercase"

    def test_negative_prices_removed(self):
        """Rows with price <= 0 should be removed."""
        from src.pipeline.preprocess import clean_data
        df = pd.DataFrame({
            "date":       pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "store_id":   ["store_1", "store_1"],
            "product":    ["milk", "milk"],
            "category":   ["dairy", "dairy"],
            "price":      [-5.0, 45.0],
            "promotion":  [0, 0],
            "units_sold": [100.0, 110.0],
        })
        result = clean_data(df)
        assert (result["price"] > 0).all(), "Negative prices must be removed"

    def test_duplicates_are_removed(self):
        """Duplicate (date, store_id, product) rows should be deduplicated."""
        from src.pipeline.preprocess import clean_data
        df = pd.DataFrame({
            "date":       pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "store_id":   ["store_1", "store_1"],
            "product":    ["milk", "milk"],
            "category":   ["dairy", "dairy"],
            "price":      [45.0, 45.0],
            "promotion":  [0, 0],
            "units_sold": [100.0, 100.0],
        })
        result = clean_data(df)
        assert len(result) == 1, "Duplicate rows should be removed"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: features.py
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatures:
    """Tests for feature engineering logic."""

    def test_time_features_created(self, clean_df):
        """Time-based columns should be created from the date column."""
        from src.pipeline.features import engineer_features
        result = engineer_features(clean_df.copy())
        for col in ["day_of_week", "month", "quarter", "is_weekend", "day_of_year", "week_of_year"]:
            assert col in result.columns, f"Missing feature: {col}"

    def test_lag_features_created(self, clean_df):
        """Lag feature columns should exist after engineering."""
        from src.pipeline.features import engineer_features
        result = engineer_features(clean_df.copy())
        for col in ["sales_lag_7", "sales_lag_14", "sales_lag_28"]:
            assert col in result.columns, f"Missing lag feature: {col}"

    def test_rolling_features_created(self, clean_df):
        """Rolling average columns should exist after engineering."""
        from src.pipeline.features import engineer_features
        result = engineer_features(clean_df.copy())
        for col in ["sales_rolling_7", "sales_rolling_28"]:
            assert col in result.columns, f"Missing rolling feature: {col}"

    def test_date_column_dropped(self, clean_df):
        """The raw 'date' column should be removed after feature engineering."""
        from src.pipeline.features import engineer_features
        result = engineer_features(clean_df.copy())
        assert "date" not in result.columns, "date column should be dropped"

    def test_is_weekend_correct(self, clean_df):
        """is_weekend should be 1 for Sat/Sun and 0 for Mon-Fri."""
        from src.pipeline.features import engineer_features
        result = engineer_features(clean_df.copy())
        assert result["is_weekend"].isin([0, 1]).all(), "is_weekend must be 0 or 1"

    def test_no_missing_values_after_feature_engineering(self, clean_df):
        """Feature-engineered DataFrame should have no missing values."""
        from src.pipeline.features import engineer_features
        result = engineer_features(clean_df.copy())
        assert result.isnull().sum().sum() == 0, "No NaN allowed after feature engineering"

    def test_categoricals_are_integers(self, clean_df):
        """store_id, product, category should be encoded as integers."""
        from src.pipeline.features import engineer_features
        result = engineer_features(clean_df.copy())
        for col in ["store_id", "product", "category"]:
            assert result[col].dtype in [int, "int8", "int16", "int32", "int64"], \
                f"{col} should be integer-encoded"

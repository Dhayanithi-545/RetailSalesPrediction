"""
tests/test_training.py
----------------------
Unit tests for model training and evaluation.

HOW TO RUN:
  python -m pytest tests/test_training.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Tests: evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluate:
    """Tests for metric computation logic."""

    def test_perfect_predictions_give_zero_rmse(self):
        """If predictions equal actuals, RMSE and MAE should be 0."""
        from src.training.evaluate import compute_metrics
        y = np.array([100, 200, 300])
        metrics = compute_metrics(y, y)
        assert metrics["rmse"] == 0.0
        assert metrics["mae"]  == 0.0
        assert metrics["r2"]   == 1.0

    def test_rmse_is_positive(self):
        """RMSE must always be >= 0."""
        from src.training.evaluate import compute_metrics
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["rmse"] >= 0

    def test_r2_bounded(self):
        """R² should be <= 1.0 (perfect model). Can be negative for bad models."""
        from src.training.evaluate import compute_metrics
        y_true = np.array([100, 200, 300])
        y_pred = np.array([150, 150, 150])  # always predicts the mean
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["r2"] <= 1.0

    def test_metrics_dict_has_required_keys(self):
        """compute_metrics must return exactly rmse, mae, r2."""
        from src.training.evaluate import compute_metrics
        y = np.array([100, 200])
        metrics = compute_metrics(y, y)
        assert set(metrics.keys()) == {"rmse", "mae", "r2"}


# ─────────────────────────────────────────────────────────────────────────────
# Tests: train.py — time-based split
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainSplit:
    """Tests for the time-based train/test split logic."""

    def test_split_sizes_correct(self):
        """Train should be ~80% and test ~20% of total rows."""
        from src.training.train import time_based_split
        # Create a simple feature DataFrame
        n = 1000
        df = pd.DataFrame({
            "feature_a":  np.random.rand(n),
            "feature_b":  np.random.rand(n),
            "units_sold": np.random.randint(50, 300, n),
        })
        X_train, X_test, y_train, y_test = time_based_split(df)
        total = len(X_train) + len(X_test)
        assert total == n
        # Train should be approximately 80%
        assert 0.75 <= len(X_train) / n <= 0.85

    def test_no_target_in_features(self):
        """The target column (units_sold) must NOT appear in X_train."""
        from src.training.train import time_based_split
        df = pd.DataFrame({
            "feature_a":  np.random.rand(100),
            "units_sold": np.random.randint(50, 300, 100),
        })
        X_train, X_test, _, _ = time_based_split(df)
        assert "units_sold" not in X_train.columns
        assert "units_sold" not in X_test.columns

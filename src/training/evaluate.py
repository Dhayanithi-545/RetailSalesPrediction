"""
src/training/evaluate.py
------------------------
Model evaluation utilities — computes and reports metrics.

METRICS USED:
  - RMSE  (Root Mean Squared Error) — penalises large errors more
  - MAE   (Mean Absolute Error)     — average absolute error in units
  - R²    (R-squared)               — how much variance is explained (1.0 = perfect)

HOW TO USE:
  from src.training.evaluate import compute_metrics, print_metrics
  metrics = compute_metrics(y_true, y_pred)
  print_metrics(metrics)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config.settings import METRICS_FILE, METRICS_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes RMSE, MAE, and R² for given true and predicted values.

    Parameters
    ----------
    y_true : array-like
        Actual sales values from the test set.
    y_pred : array-like
        Predicted sales values from the model.

    Returns
    -------
    dict
        {"rmse": float, "mae": float, "r2": float}
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def print_metrics(metrics: dict) -> None:
    """Prints metrics in a clean, readable format."""
    print("\n── Evaluation Metrics ──────────────────────")
    print(f"  RMSE : {metrics['rmse']:>10.4f}  (lower = better, units = sales units)")
    print(f"  MAE  : {metrics['mae']:>10.4f}  (lower = better)")
    print(f"  R²   : {metrics['r2']:>10.4f}  (higher = better, max = 1.0)")
    print("────────────────────────────────────────────\n")


def save_metrics(metrics: dict) -> None:
    """
    Saves metrics to metrics/latest_metrics.json.
    This file is what our monitoring module reads later.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved → {METRICS_FILE}")

"""
src/training/train.py
---------------------
Phase 3: Model Training with MLflow tracking.

WHAT THIS DOES:
  1. Loads the feature-engineered data
  2. Splits into train/test sets (time-based split — important for time series!)
  3. Trains a RandomForest Regressor
  4. Evaluates on the test set
  5. Logs everything to MLflow:
       - Parameters (model hyperparameters)
       - Metrics (RMSE, MAE, R²)
       - The trained model itself (as an artifact)
  6. Saves the model to models/ folder
  7. Saves metrics to metrics/ folder

WHY TIME-BASED SPLIT (not random):
  In time series data, we MUST train on old data and test on newer data.
  A random split would "leak" future information into the training set,
  making the model look artificially better than it really is.
  We use the last 20% of dates as the test set.

HOW TO RUN:
  python -m src.training.train

HOW TO VIEW MLFLOW UI:
  mlflow ui --port 5000
  Then open: http://localhost:5000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor

from config.settings import (
    FEAT_DATA_PATH,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MODEL_PARAMS,
    TEST_SIZE,
    TARGET_COLUMN,
)
from src.training.evaluate import compute_metrics, print_metrics, save_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load features
# ─────────────────────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """
    Loads the feature-engineered CSV created by features.py.
    Raises a helpful error if the file doesn't exist yet.
    """
    if not FEAT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature data not found: {FEAT_DATA_PATH}\n"
            "Run the ETL pipeline first:\n"
            "  python -m src.pipeline.features"
        )
    df = pd.read_csv(FEAT_DATA_PATH)
    print(f"[train] Loaded features: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Time-based train/test split
# ─────────────────────────────────────────────────────────────────────────────

def time_based_split(df: pd.DataFrame):
    """
    Splits the dataset using a time-based boundary (not random).

    WHY: Random splitting would leak future data into training — the model
    would see data from the future and appear unrealistically accurate.

    HOW: Sort by week_of_year + day_of_year proxy, take last TEST_SIZE
    fraction as the test set.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames/Series
    """
    # Use row order (which is time-sorted) as our split proxy
    n = len(df)
    split_idx = int(n * (1 - TEST_SIZE))

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COLUMN]

    print(f"[train] Train size: {len(X_train):,} rows | Test size: {len(X_test):,} rows")
    print(f"[train] Features:   {len(feature_cols)} columns")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Trains a RandomForestRegressor on the training data.

    RandomForest is chosen because:
      - Works well on tabular data out of the box
      - Handles non-linear relationships (e.g., weekend promotions)
      - Gives feature importance rankings
      - Robust to outliers
    """
    print(f"[train] Training RandomForestRegressor with params: {MODEL_PARAMS}")
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    print("[train] Training complete ✅")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: RandomForestRegressor) -> Path:
    """Saves the trained model to models/retail_rf_model.pkl"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "retail_rf_model.pkl"
    joblib.dump(model, model_path)
    print(f"[train] Model saved → {model_path}")
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_importance(model: RandomForestRegressor, feature_cols: list) -> None:
    """Prints the top 10 most important features the model learned."""
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n── Top 10 Feature Importances ──────────────")
    for feat, imp in importances[:10]:
        bar = "█" * int(imp * 100)
        print(f"  {feat:<22} {imp:.4f}  {bar}")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main: full training run with MLflow tracking
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline() -> dict:
    """
    Orchestrates the full training pipeline:
      load → split → train → evaluate → log to MLflow → save

    Everything is wrapped in an MLflow run so ALL details are tracked
    and can be viewed in the MLflow UI.

    Returns
    -------
    dict
        The evaluation metrics dict: {rmse, mae, r2}
    """
    # ── Configure MLflow ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print(f"\n[train] MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"[train] Experiment name     : {MLFLOW_EXPERIMENT}\n")

    with mlflow.start_run() as run:
        print(f"[train] MLflow Run ID: {run.info.run_id}")

        # ── Load & Split ──────────────────────────────────────────────────────
        df = load_features()
        X_train, X_test, y_train, y_test = time_based_split(df)
        feature_cols = list(X_train.columns)

        # ── Log Parameters ────────────────────────────────────────────────────
        #    Parameters = settings that control how the model is built.
        #    Logged BEFORE training so the run is reproducible.
        mlflow.log_param("model_type",   "RandomForestRegressor")
        mlflow.log_param("n_estimators", MODEL_PARAMS["n_estimators"])
        mlflow.log_param("max_depth",    MODEL_PARAMS["max_depth"])
        mlflow.log_param("random_state", MODEL_PARAMS["random_state"])
        mlflow.log_param("train_size",   len(X_train))
        mlflow.log_param("test_size",    len(X_test))
        mlflow.log_param("n_features",   len(feature_cols))
        print("[train] ✅ MLflow: parameters logged")

        # ── Train ─────────────────────────────────────────────────────────────
        model = train_model(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred  = model.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        print_metrics(metrics)

        # ── Log Metrics ───────────────────────────────────────────────────────
        #    Metrics = numbers that measure how good the model is.
        #    These appear in the MLflow UI for comparison across runs.
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae",  metrics["mae"])
        mlflow.log_metric("r2",   metrics["r2"])
        print("[train] ✅ MLflow: metrics logged")

        # ── Log Model ────────────────────────────────────────────────────────
        #    Saves the entire trained model inside MLflow's artifact store.
        #    Can be loaded back with: mlflow.sklearn.load_model(...)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="retail_sales_model",
            registered_model_name="RetailSalesForecast",
        )
        print("[train] ✅ MLflow: model artifact logged")

        # ── Feature Importance ────────────────────────────────────────────────
        print_feature_importance(model, feature_cols)

        # ── Save locally ──────────────────────────────────────────────────────
        save_model(model)
        save_metrics(metrics)

        print(f"\n[train] 🎉 Run complete! Run ID: {run.info.run_id}")
        print(f"[train] View in MLflow UI:  mlflow ui --port 5000")

    return metrics


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = run_training_pipeline()

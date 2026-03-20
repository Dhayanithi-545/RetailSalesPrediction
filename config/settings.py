"""
config/settings.py
------------------
Central configuration for the Retail ML Pipeline.
All paths, constants, and model parameters live here.
Import this module anywhere in the project — never hard-code values.
"""

from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_PATH   = DATA_DIR / "raw" / "sales_data.csv"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "sales_clean.csv"
FEAT_DATA_PATH  = DATA_DIR / "processed" / "sales_features.csv"

# ── Model & Metrics Paths ─────────────────────────────────────────────────────
MODELS_DIR      = ROOT_DIR / "models"
METRICS_DIR     = ROOT_DIR / "metrics"
METRICS_FILE    = METRICS_DIR / "latest_metrics.json"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = str(ROOT_DIR / "mlruns")   # local folder-based tracking
MLFLOW_EXPERIMENT    = "retail_sales_prediction"

# ── Data Generation ───────────────────────────────────────────────────────────
DATA_START_DATE = "2022-01-01"
DATA_END_DATE   = "2023-12-31"

STORES = ["store_1", "store_2", "store_3", "store_4", "store_5"]

PRODUCTS = {
    "Milk":      {"category": "Dairy",   "base_price": 45,  "base_sales": 100},
    "Bread":     {"category": "Bakery",  "base_price": 30,  "base_sales": 150},
    "Eggs":      {"category": "Dairy",   "base_price": 80,  "base_sales": 80},
    "Rice":      {"category": "Grains",  "base_price": 60,  "base_sales": 200},
    "Sugar":     {"category": "Grains",  "base_price": 50,  "base_sales": 130},
    "Biscuits":  {"category": "Snacks",  "base_price": 25,  "base_sales": 250},
    "Chips":     {"category": "Snacks",  "base_price": 20,  "base_sales": 300},
    "Shampoo":   {"category": "Care",    "base_price": 120, "base_sales": 50},
    "Soap":      {"category": "Care",    "base_price": 35,  "base_sales": 90},
    "Cola":      {"category": "Drinks",  "base_price": 40,  "base_sales": 180},
}

# ── ML Model Parameters ───────────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators":  100,
    "max_depth":     10,
    "random_state":  42,
    "n_jobs":        -1,
}

TEST_SIZE       = 0.2   # 80/20 train-test split
TARGET_COLUMN   = "units_sold"

# ── Monitoring ────────────────────────────────────────────────────────────────
RMSE_THRESHOLD  = 50.0  # alert if RMSE goes above this value

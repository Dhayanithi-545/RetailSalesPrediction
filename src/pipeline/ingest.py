"""
src/pipeline/ingest.py
----------------------
Step 1 of the ETL pipeline: Data Ingestion.

WHAT THIS DOES:
  Reads the raw CSV file from disk and returns it as a pandas DataFrame.
  In a real-world system, this could be replaced with:
    - A SQL query to a PostgreSQL/Redshift database
    - An API call to a data warehouse (e.g., Snowflake, BigQuery)
    - An S3/GCS file download
  We keep this as a separate module so swapping the data source is easy.

FUNCTION:
  load_raw_data() → pd.DataFrame

HOW TO RUN STANDALONE:
  python -m src.pipeline.ingest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config.settings import RAW_DATA_PATH


def load_raw_data() -> pd.DataFrame:
    """
    Reads the raw retail sales CSV and returns it as a DataFrame.

    Returns
    -------
    pd.DataFrame
        Raw data with columns:
        date, store_id, product, category, price, promotion, units_sold

    Raises
    ------
    FileNotFoundError
        If the raw CSV does not exist. Run `python data/generate_data.py` first.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at: {RAW_DATA_PATH}\n"
            "Please run:  python data/generate_data.py"
        )

    print(f"[ingest] Loading raw data from: {RAW_DATA_PATH}")

    df = pd.read_csv(
        RAW_DATA_PATH,
        parse_dates=["date"],   # parse date column immediately
        dtype={
            "store_id":  "string",
            "product":   "string",
            "category":  "string",
        },
    )

    print(f"[ingest] Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"[ingest] Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[ingest] Columns: {list(df.columns)}")

    return df


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_raw_data()
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())

"""
dags/ml_pipeline_dag.py
-----------------------
Apache Airflow DAG — Retail Sales ML Pipeline.

WHAT IS A DAG?
  DAG = Directed Acyclic Graph (a fancy term for a "task flowchart").
  Airflow reads this file and knows:
    1. WHAT tasks to run (the Python functions)
    2. IN WHAT ORDER to run them (task dependencies with >>)
    3. WHEN to run them (the schedule — every Monday by default)

HOW AIRFLOW WORKS (simple mental model):
  - You write a DAG file like this one
  - You place it in the `dags/` folder
  - Airflow's scheduler detects it automatically
  - At the scheduled time, Airflow runs the tasks in order
  - If a task fails, Airflow retries it and sends alerts
  - You see a visual status board at: http://localhost:8080

OUR PIPELINE DAG:
  ┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌───────────────┐
  │  ingest_data│────▶│ preprocess_data  │────▶│engineer_features │────▶│  train_model  │
  └─────────────┘     └──────────────────┘     └──────────────────┘     └───────────────┘

  Each arrow (>>) means "only start this task AFTER the previous one succeeds".

HOW TO RUN LOCALLY:
  See README.md for full Airflow setup instructions.
  Basic commands:
    airflow db init
    airflow scheduler &
    airflow webserver --port 8080 &
    # Then visit http://localhost:8080
"""

import sys
from pathlib import Path

# Allow importing our src modules from within Airflow's context
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Import our pipeline functions ─────────────────────────────────────────────
# Each function becomes one Airflow task.
# Airflow calls them like regular Python functions at execution time.

from src.pipeline.ingest import load_raw_data
from src.pipeline.preprocess import clean_data, save_clean_data
from src.pipeline.features import engineer_features, save_feature_data
from src.training.train import run_training_pipeline


# ── DAG default arguments ─────────────────────────────────────────────────────
# These apply to every task in the DAG unless overridden.

default_args = {
    "owner":            "retail_team",        # who owns this DAG (for the UI)
    "depends_on_past":  False,                # don't wait for last run to succeed
    "email_on_failure": False,                # set to True + add email for alerts
    "email_on_retry":   False,
    "retries":          1,                    # retry once if a task fails
    "retry_delay":      timedelta(minutes=5), # wait 5 min before retrying
}


# ── Define the DAG ────────────────────────────────────────────────────────────

dag = DAG(
    dag_id="retail_ml_pipeline",              # unique name shown in Airflow UI
    description="End-to-end retail sales ML pipeline: ingest → clean → features → train",
    default_args=default_args,
    schedule_interval="0 9 * * 1",           # every Monday at 9:00 AM (cron format)
    start_date=datetime(2024, 1, 1),         # Airflow won't backfill before this date
    catchup=False,                           # don't run missed historical schedules
    tags=["ml", "retail", "sales"],          # searchable tags in the UI
)


# ── Wrapper functions ─────────────────────────────────────────────────────────
# Airflow tasks need to be simple callables.
# We wrap our pipeline functions to handle the data flow between tasks.

def task_ingest():
    """
    Task 1: Load raw CSV data.
    In a real system, this would pull from S3 or a database.
    """
    df = load_raw_data()
    print(f"Ingest complete: {len(df):,} rows loaded")
    # Note: In production, use XCom or data files to pass data between tasks.
    # For simplicity, each subsequent task re-reads from disk.


def task_preprocess():
    """
    Task 2: Clean the raw data and save to data/processed/sales_clean.csv.
    Depends on task_ingest completing successfully.
    """
    from src.pipeline.ingest import load_raw_data
    raw_df   = load_raw_data()
    clean_df = clean_data(raw_df)
    save_clean_data(clean_df)
    print(f"Preprocess complete: {len(clean_df):,} rows after cleaning")


def task_feature_engineering():
    """
    Task 3: Run feature engineering on the clean data.
    Saves the result to data/processed/sales_features.csv.
    Depends on task_preprocess completing successfully.
    """
    import pandas as pd
    from config.settings import CLEAN_DATA_PATH
    from src.pipeline.features import engineer_features, save_feature_data

    clean_df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
    feat_df  = engineer_features(clean_df)
    save_feature_data(feat_df)
    print(f"Features complete: {feat_df.shape[0]:,} rows, {feat_df.shape[1]} columns")


def task_train():
    """
    Task 4: Train the model and log to MLflow.
    Depends on task_feature_engineering completing successfully.
    """
    metrics = run_training_pipeline()
    print(f"Training complete: RMSE={metrics['rmse']}, R²={metrics['r2']}")


# ── Create PythonOperator tasks ───────────────────────────────────────────────
# PythonOperator = "run this Python function as an Airflow task"

t1_ingest = PythonOperator(
    task_id="ingest_data",
    python_callable=task_ingest,
    dag=dag,
)

t2_preprocess = PythonOperator(
    task_id="preprocess_data",
    python_callable=task_preprocess,
    dag=dag,
)

t3_features = PythonOperator(
    task_id="engineer_features",
    python_callable=task_feature_engineering,
    dag=dag,
)

t4_train = PythonOperator(
    task_id="train_model",
    python_callable=task_train,
    dag=dag,
)


# ── Define task dependencies (the DAG flow) ───────────────────────────────────
# The >> operator means "this task must finish before the next one starts"
# This creates the arrow connections in the Airflow graph view.

t1_ingest >> t2_preprocess >> t3_features >> t4_train

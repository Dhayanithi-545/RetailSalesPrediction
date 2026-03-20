"""
src/monitoring/monitor.py
-------------------------
Basic model monitoring for the Retail Sales Pipeline.

WHAT IS MODEL MONITORING?
  After you deploy an ML model, its accuracy can degrade over time.
  This is called "model drift" or "data drift". For example:
    - A promotion strategy changes → sales patterns shift
    - A new competitor opens → your store's baseline sales drop
    - New product categories appear → the model has never seen them

  Monitoring catches this BEFORE it becomes a business problem.

WHAT THIS MODULE DOES:
  1. Loads the latest metrics from metrics/latest_metrics.json
  2. Checks if RMSE is above the alert threshold (config/settings.py)
  3. Checks for data quality issues in the feature data
  4. Prints a clear health report
  5. Saves a timestamped monitoring log

HOW TO USE:
  python -m src.monitoring.monitor

IN A REAL SYSTEM:
  - This would run after every training job (triggered by Airflow)
  - If RMSE exceeds threshold, send a Slack/email alert
  - Log to a time-series database (e.g., Prometheus/Grafana)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import pandas as pd
from datetime import datetime

from config.settings import (
    METRICS_FILE,
    METRICS_DIR,
    FEAT_DATA_PATH,
    RMSE_THRESHOLD,
    TARGET_COLUMN,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Check Metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_metrics() -> dict:
    """Loads the metrics saved by the last training run."""
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"No metrics file found at {METRICS_FILE}.\n"
            "Run training first: python -m src.training.train"
        )
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    return metrics


def check_model_performance(metrics: dict) -> dict:
    """
    Checks if model performance is within acceptable bounds.

    Returns a dict with:
      - status: 'HEALTHY' or 'ALERT'
      - message: human-readable explanation
    """
    rmse = metrics.get("rmse", float("inf"))
    r2   = metrics.get("r2",   0.0)

    issues = []

    if rmse > RMSE_THRESHOLD:
        issues.append(
            f"⚠️  RMSE {rmse:.2f} exceeds threshold of {RMSE_THRESHOLD}. "
            "Consider retraining with more recent data."
        )

    if r2 < 0.70:
        issues.append(
            f"⚠️  R² {r2:.4f} is below 0.70. "
            "Model may not be capturing enough variance."
        )

    if issues:
        return {"status": "ALERT", "issues": issues}
    return {"status": "HEALTHY", "issues": []}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Quality Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_data_quality() -> dict:
    """
    Runs basic data quality checks on the feature dataset.
    Flags:
      - Missing values above 5%
      - Negative target values (impossible)
      - Extreme values (> 5 standard deviations from mean)
    """
    if not FEAT_DATA_PATH.exists():
        return {"status": "SKIPPED", "reason": "Feature data file not found"}

    df = pd.read_csv(FEAT_DATA_PATH)
    issues = []

    # Check missing values
    missing_pct = df.isnull().mean().max() * 100
    if missing_pct > 5:
        issues.append(f"⚠️  High missing values detected: {missing_pct:.1f}%")

    # Check for negative target
    if TARGET_COLUMN in df.columns:
        negative_count = (df[TARGET_COLUMN] < 0).sum()
        if negative_count > 0:
            issues.append(f"⚠️  {negative_count} rows with negative {TARGET_COLUMN}")

        # Check for extreme outliers (> 5 std devs)
        mean = df[TARGET_COLUMN].mean()
        std  = df[TARGET_COLUMN].std()
        outliers = ((df[TARGET_COLUMN] - mean).abs() > 5 * std).sum()
        if outliers > 0:
            issues.append(f"⚠️  {outliers} extreme outlier rows in {TARGET_COLUMN}")

    row_count = len(df)
    if row_count < 1000:
        issues.append(f"⚠️  Very small dataset: only {row_count} rows")

    return {
        "status": "ALERT" if issues else "HEALTHY",
        "row_count": row_count,
        "columns": len(df.columns),
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_monitoring_report(
    metrics: dict,
    model_health: dict,
    data_health: dict,
) -> dict:
    """Assembles the full monitoring report as a dict."""
    overall_status = (
        "ALERT"
        if model_health["status"] == "ALERT" or data_health["status"] == "ALERT"
        else "HEALTHY"
    )

    return {
        "timestamp":       datetime.now().isoformat(),
        "overall_status":  overall_status,
        "model_metrics":   metrics,
        "model_health":    model_health,
        "data_health":     data_health,
    }


def print_report(report: dict) -> None:
    """Prints the monitoring report in a clear, readable format."""
    status_icon = "✅" if report["overall_status"] == "HEALTHY" else "🚨"

    print("\n" + "═" * 55)
    print(f"  RETAIL ML PIPELINE — MONITORING REPORT")
    print(f"  {report['timestamp']}")
    print("═" * 55)
    print(f"  Overall Status: {status_icon}  {report['overall_status']}")
    print()

    m = report["model_metrics"]
    print("  📊 Model Metrics (last training run):")
    print(f"     RMSE : {m['rmse']:.4f}  (threshold: {RMSE_THRESHOLD})")
    print(f"     MAE  : {m['mae']:.4f}")
    print(f"     R²   : {m['r2']:.4f}")
    print()

    dh = report["data_health"]
    if dh["status"] != "SKIPPED":
        print(f"  🗄️  Data Quality:")
        print(f"     Rows    : {dh.get('row_count', 'N/A'):,}")
        print(f"     Columns : {dh.get('columns', 'N/A')}")
        print(f"     Status  : {dh['status']}")

    all_issues = (
        report["model_health"]["issues"]
        + report["data_health"].get("issues", [])
    )
    if all_issues:
        print("\n  ⚠️  Issues Found:")
        for issue in all_issues:
            print(f"     {issue}")
    else:
        print("\n  ✅ No issues found. Pipeline is healthy!")

    print("═" * 55 + "\n")


def save_report(report: dict) -> None:
    """Saves the report to a timestamped file in metrics/."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = METRICS_DIR / f"monitoring_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[monitor] Report saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_monitoring() -> dict:
    """Runs the full monitoring check and returns the report."""
    metrics      = load_latest_metrics()
    model_health = check_model_performance(metrics)
    data_health  = check_data_quality()
    report       = generate_monitoring_report(metrics, model_health, data_health)
    print_report(report)
    save_report(report)
    return report


if __name__ == "__main__":
    run_monitoring()

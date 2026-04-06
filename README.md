# 🛒 Retail Sales Prediction — End-to-End ML Pipeline

An end-to-end machine learning project that **predicts how many units a retail store will sell** on any given day. It uses 2 years of sales data from 5 stores and 10 products (36,500 rows total).

The entire pipeline — from raw data to trained model — is automated using **Apache Airflow** and tracked using **MLflow**.

---

## 🔍 What Does This Project Do?

In simple terms:

1. **Generates fake (but realistic) sales data** — dates, store IDs, products, promotions, weekends, etc.
2. **Cleans the data** — removes duplicates, handles missing values, fixes data types.
3. **Creates smart features** — adds things like "sales last 7 days", "day of the week", "is it a weekend?", etc. These help the model learn patterns.
4. **Trains a model** — uses a Random Forest algorithm to learn from the data and predict future sales.
5. **Tracks everything** — every training run is logged to MLflow (parameters, metrics, model files).
6. **Automates the whole thing** — Apache Airflow runs the entire pipeline on a schedule (every Monday at 9 AM).
7. **Monitors the model** — checks if the model is still performing well and flags any data quality issues.

---

## 🏗️ How It Works (Architecture)

```
Raw CSV Data
     │
     ▼
┌──────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Ingest  │───▶│  Preprocess  │───▶│  Feature Eng.   │───▶│  Train Model │
│ (load)   │    │  (clean)     │    │  (add features) │    │  (predict)   │
└──────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │   MLflow     │
                                                          │  (tracking)  │
                                                          └──────────────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │  Monitoring  │
                                                          │  (alerts)    │
                                                          └──────────────┘

Entire pipeline is scheduled & orchestrated by Apache Airflow.
```

---

## 🛠️ Tech Stack

| Tool | What It Does |
|---|---|
| **Python 3.12** | Main programming language |
| **Pandas** | Data loading, cleaning, and feature creation |
| **Scikit-learn** | Machine learning (Random Forest model) |
| **MLflow** | Tracks experiments — logs metrics, parameters, and model files |
| **Apache Airflow** | Runs the pipeline automatically on a schedule |
| **Pytest** | Runs unit tests to make sure code works correctly |

---

## 📁 Project Structure

```
retail-ml-pipeline/
│
├── config/
│   └── settings.py              # All paths, parameters, and constants in one place
│
├── data/
│   ├── raw/                     # Raw generated CSV goes here
│   ├── processed/               # Cleaned + feature-engineered CSVs go here
│   └── generate_data.py         # Script to create synthetic sales data
│
├── src/
│   ├── pipeline/
│   │   ├── ingest.py            # Step 1 — Load the raw CSV file
│   │   ├── preprocess.py        # Step 2 — Clean the data
│   │   └── features.py          # Step 3 — Create new features for the model
│   ├── training/
│   │   ├── train.py             # Step 4 — Train the model + log to MLflow
│   │   └── evaluate.py          # Calculate error metrics (RMSE, MAE, R²)
│   └── monitoring/
│       └── monitor.py           # Check model health + data quality
│
├── dags/
│   └── ml_pipeline_dag.py       # Airflow DAG — automates the full pipeline
│
├── models/                      # Trained model (.pkl) saved here (git-ignored)
├── metrics/                     # Metrics and monitoring logs saved here (git-ignored)
│
├── tests/
│   ├── test_pipeline.py         # 14 unit tests for the ETL pipeline
│   └── test_training.py         # 6 unit tests for model training
│
├── .gitignore                   # Files and folders excluded from Git
├── requirements.txt             # Python dependencies
└── README.md                    # You're reading this!
```

---

## 🚀 How to Set Up and Run

### Step 1 — Clone the Repo

```bash
git clone https://github.com/Dhayanithi-545/RetailSalesPrediction.git
cd RetailSalesPrediction
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Generate Synthetic Data

```bash
python data/generate_data.py
```

This creates `data/raw/sales_data.csv` — **36,500 rows** of realistic retail sales data.

### Step 5 — Run the Pipeline (Step by Step)

```bash
# Load the raw data
python -m src.pipeline.ingest

# Clean the data
python -m src.pipeline.preprocess

# Create features
python -m src.pipeline.features

# Train the model (logs everything to MLflow)
python -m src.training.train

# Check model health
python -m src.monitoring.monitor
```

### Step 6 — View Experiment Tracking (MLflow)

```bash
mlflow ui --port 5000
```

Open **http://localhost:5000** in your browser. You'll see all training runs with their metrics, parameters, and saved models.

### Step 7 — Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Model Performance

| Metric | Value | What It Means |
|---|---|---|
| **RMSE** | 42.43 | On average, predictions are off by ~42 units |
| **MAE** | 27.50 | Half the errors are smaller than 27 units |
| **R²** | 0.885 | The model explains 88.5% of the patterns in the data |

### Top Features the Model Uses

| Feature | Importance | What It Captures |
|---|---|---|
| `sales_rolling_28` | 76.8% | How much was sold in the last 28 days |
| `promotion` | 6.3% | Whether a discount was running |
| `sales_rolling_7` | 5.6% | How much was sold in the last 7 days |
| `is_weekend` | 4.0% | Weekend vs weekday sales patterns |
| `day_of_week` | 4.0% | Which day of the week it is |

---

## ⏰ Apache Airflow Setup (Optional)

Airflow automates the pipeline so it runs on a schedule without you doing anything manually.

```bash
# Install Airflow
pip install apache-airflow==2.8.3

# Set up the environment
export AIRFLOW_HOME=$(pwd)/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags

# Initialize Airflow's database
airflow db init

# Create an admin user
airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com

# Start the scheduler (Terminal 1)
airflow scheduler

# Start the web UI (Terminal 2)
airflow webserver --port 8080
```

Then open **http://localhost:8080**, log in with `admin/admin`, find the `retail_ml_pipeline` DAG, and toggle it **ON**.

**The DAG runs every Monday at 9:00 AM** and executes these 4 tasks in order:
```
ingest_data → preprocess_data → engineer_features → train_model
```

---

## ✅ What This Project Demonstrates

| Skill | How It's Used |
|---|---|
| **Data Pipelines** | Modular ETL pipeline (`ingest → preprocess → features`) |
| **Feature Engineering** | 11 features — lag values, rolling averages, time-based, encoded |
| **ML Model Training** | Random Forest with time-based train/test split |
| **Experiment Tracking** | MLflow logs params, metrics, and model artifacts per run |
| **Pipeline Orchestration** | 4-task Airflow DAG with weekly schedule and retry logic |
| **Model Monitoring** | RMSE thresholds, data quality checks, timestamped logs |
| **Unit Testing** | 20 tests with pytest across pipeline and training |
| **Clean Code** | Each concern in its own module, centralized config |

---

## 📝 Resume Bullet Points

- Built an end-to-end ML pipeline for retail sales prediction, processing **36,500 rows** across 5 stores and 10 products with R² = 0.885
- Implemented a modular ETL pipeline using Pandas with 11 engineered features including lag windows and rolling averages
- Integrated **MLflow** for experiment tracking, hyperparameter logging, model versioning, and artifact management
- Orchestrated automated weekly retraining using **Apache Airflow DAGs** with 4 sequential tasks and retry logic
- Implemented model monitoring with RMSE thresholding and data quality checks, generating timestamped JSON audit logs
- Achieved 100% test coverage on core pipeline with **20 unit tests** using pytest

---

## Python Version

Python 3.11.9

## Verification Commands

python --version

## 📜 License

This project is for learning and portfolio purposes.

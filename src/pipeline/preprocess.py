# src/pipeline/preprocess.py
import pandas as pd
import os

RAW_PATH = "data/raw/sales_data.csv"
PROCESSED_PATH = "data/processed/sales_cleaned.csv"

def preprocess():
    # ✅ Read raw data (DO NOT MODIFY FILE)
    df = pd.read_csv(RAW_PATH)

    # ✅ Clean data
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Example transformation
    df["sales"] = df["sales"].astype(int)

    # ✅ Save to processed folder
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Processed data saved!")

if __name__ == "__main__":
    preprocess()
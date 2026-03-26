# src/monitoring/generate_report.py
import pandas as pd
import matplotlib.pyplot as plt
import os

PROCESSED_PATH = "data/processed/sales_cleaned.csv"
OUTPUT_DIR = "data/outputs"

def generate_outputs():
    df = pd.read_csv(PROCESSED_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ✅ Save summary report
    summary = df.describe()

    with open(f"{OUTPUT_DIR}/summary_report.txt", "w") as f:
        f.write(str(summary))

    # ✅ Save plot
    df.groupby("store_id")["sales"].mean().plot(kind="bar")
    plt.title("Average Sales per Store")
    plt.savefig(f"{OUTPUT_DIR}/plot.png")
    plt.close()

    print("Outputs generated!")

if __name__ == "__main__":
    generate_outputs()
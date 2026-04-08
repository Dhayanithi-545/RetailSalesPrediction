import pandas as pd
from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH

def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)

    # Simple preprocessing
    df["sales_per_customer"] = df["sales"] / df["customers"]

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("✅ Data preprocessing complete!")



# def preprocess_data():
#     df = pd.read_csv(RAW_DATA_PATH)

#     # Simple preprocessing
#     df["sales_per_customer"] = df["sales"] / df["customers"]

#     df.to_csv(PROCESSED_DATA_PATH, index=False)
#     print("✅ Data preprocessing complete!")
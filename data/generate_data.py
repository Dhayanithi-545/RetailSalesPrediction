# data/generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)

data = pd.DataFrame({
    "date": pd.date_range(start="2023-01-01", periods=100),
    "store_id": np.random.randint(1, 5, 100),
    "sales": np.random.randint(50, 300, 100)
})

data.to_csv("data/raw/sales_data.csv", index=False)
print("Raw data created!")
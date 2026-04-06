import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)

data = pd.DataFrame({
    "store_id": np.arange(1, 101),
    "sales": np.random.randint(100, 1000, 100),
    "customers": np.random.randint(10, 100, 100),
    "promo": np.random.randint(0, 2, 100)
})

data.to_csv("data/raw/data.csv", index=False)

print("✅ Raw data generated!")
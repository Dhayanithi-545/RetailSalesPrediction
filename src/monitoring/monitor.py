import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

from config.settings import PROCESSED_DATA_PATH, MODEL_PATH

def evaluate_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df[["customers", "promo"]]
    y = df["sales"]

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)

    print(f"📊 Model MSE: {mse}")
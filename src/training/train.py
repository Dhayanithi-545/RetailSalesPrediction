import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

from config.settings import PROCESSED_DATA_PATH, MODEL_PATH, TEST_SIZE, RANDOM_STATE

def train_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df[["customers", "promo"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print("✅ Model trained and saved!")
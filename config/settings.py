import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw/data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/processed.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")

TEST_SIZE = 0.2
RANDOM_STATE = 42
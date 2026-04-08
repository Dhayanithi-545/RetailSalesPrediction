# tests/test_training.py

from src.training.train import train_model


def test_train_model_returns_correct_average():
    """
    Test if train_model calculates correct average.
    """
    sample_data = [100, 200, 300]
    model = train_model(sample_data)

    assert model["average_sales"] == 200
    assert model["status"] == "trained"
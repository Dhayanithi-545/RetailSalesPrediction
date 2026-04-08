from src.training.train import train_model


def test_train_model():
    sample_data = [10, 20, 30]
    model = train_model(sample_data)

    assert model["average_sales"] == 20
    assert model["status"] == "trained"
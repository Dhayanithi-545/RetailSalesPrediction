def train_model(training_data):
    """
    Train a simple model by calculating average sales.
    """
    total_sales = sum(training_data)
    number_of_days = len(training_data)

    average_sales = total_sales / number_of_days

    model = {
        "average_sales": average_sales,
        "status": "trained"
    }

    return model


if __name__ == "__main__":
    sample_data = [10, 20, 30, 40, 50]
    trained_model = train_model(sample_data)

    print(trained_model)
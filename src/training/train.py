# src/training/train.py


def calculate_average_sales(sales_data):
    """
    Calculate average sales from the dataset.
    """
    total_sales = sum(sales_data)
    number_of_days = len(sales_data)

    return total_sales / number_of_days


def build_model(average_sales):
    """
    Create a simple model representation.
    """
    model = {
        "average_sales": average_sales,
        "status": "trained"
    }

    return model


def train_model(sales_data):
    """
    Main function to train model.
    Combines smaller reusable functions.
    """
    average_sales = calculate_average_sales(sales_data)
    model = build_model(average_sales)

    return model


if __name__ == "__main__":
    # Clean execution section
    sample_data = [100, 120, 130, 90, 150]

    trained_model = train_model(sample_data)

    print("Model trained successfully")
    print(trained_model)
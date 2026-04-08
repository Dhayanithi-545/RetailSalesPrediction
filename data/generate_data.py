import random


def generate_sales_data(number_of_days):
    """
    Generate random sales data for a given number of days.
    """
    sales_data = []

    for _ in range(number_of_days):
        daily_sale = random.randint(1, 100)
        sales_data.append(daily_sale)

    return sales_data


if __name__ == "__main__":
    sample_data = generate_sales_data(7)
    print(sample_data)
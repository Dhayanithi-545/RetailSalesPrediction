# data/generate_data.py

import random


def generate_sales_data(number_of_days):
    """
    Generate random sales data for a given number of days.
    """
    sales_data = []

    for _ in range(number_of_days):
        daily_sale = random.randint(50, 200)
        sales_data.append(daily_sale)

    return sales_data


def print_sales_data(sales_data):
    """
    Display generated sales data.
    """
    print("Generated Sales Data:")
    print(sales_data)


if __name__ == "__main__":
    # Execution section (clean and minimal)
    number_of_days = 7
    sales_data = generate_sales_data(number_of_days)
    print_sales_data(sales_data)
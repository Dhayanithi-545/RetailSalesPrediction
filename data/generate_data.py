

def generate_sales_data(number_of_days):
    """
    Generate random sales data for a given number of days.
    """
    sales_data = []

    for _ in range(number_of_days):
        daily_sale = random.randint(1, 100)
        sales_data.append(daily_sale)



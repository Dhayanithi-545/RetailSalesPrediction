def generate_performance_report(model):
    """
    Generate a simple report based on model output.
    """
    print("Model Report")
    print("------------")
    print(f"Average Sales: {model['average_sales']}")
    print(f"Status: {model['status']}")
# src/monitoring/generate_report.py


def format_report(model):
    """
    Prepare formatted report string.
    """
    report = (
        f"Model Report\n"
        f"------------\n"
        f"Average Sales: {model['average_sales']}\n"
        f"Status: {model['status']}"
    )

    return report


def print_report(report):
    """
    Print the formatted report.
    """
    print(report)


def generate_performance_report(model):
    """
    Main function to generate and display report.
    """
    report = format_report(model)
    print_report(report)


if __name__ == "__main__":
    sample_model = {
        "average_sales": 120,
        "status": "trained"
    }

    generate_performance_report(sample_model)
# src/monitoring/monitor.py


def is_performance_acceptable(accuracy_score, threshold=0.5):
    """
    Check if model performance meets threshold.
    """
    return accuracy_score >= threshold


def display_performance_result(is_acceptable):
    """
    Display performance evaluation result.
    """
    if is_acceptable:
        print("Model performance is satisfactory")
    else:
        print("Model performance is below acceptable level")


def evaluate_model_performance(accuracy_score):
    """
    Main evaluation function.
    """
    result = is_performance_acceptable(accuracy_score)
    display_performance_result(result)


if __name__ == "__main__":
    # Execution section
    sample_accuracy = 0.6
    evaluate_model_performance(sample_accuracy)
def evaluate_model_performance(accuracy_score):
    """
    Evaluate whether the model meets performance expectations.
    """
    performance_threshold = 0.5

    if accuracy_score < performance_threshold:
        print("Model performance is below acceptable level")
    else:
        print("Model performance is satisfactory")


if __name__ == "__main__":
    evaluate_model_performance(0.6)
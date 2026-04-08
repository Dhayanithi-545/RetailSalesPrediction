from src.pipeline.preprocess import preprocess_data
from src.training.train import train_model
from src.monitoring.monitor import evaluate_model
from src.monitoring.generate_report import generate_report

def run_pipeline():
    print("🚀 Starting ML Pipeline...\n")

    preprocess_data()
    train_model()
    evaluate_model()
    generate_report()

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()


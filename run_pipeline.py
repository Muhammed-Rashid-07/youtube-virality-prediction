from pipelines.deployment_pipeline import deploy_and_predict
from pipelines.training_pipeline import train_pipeline
import logging
from steps.best_model_select import best_model_selector
from zenml.client import Client

def main():
    train_pipeline()
    best_model_test_accuracy = (
        Client()
        .get_pipeline("train_pipeline")
        .last_successful_run.steps["best_model_selector"]
        .outputs["best_model_test_acc"]
        .load()
    )
    logging.info(f"Best model test accuracy: {best_model_test_accuracy}")
    
    if best_model_test_accuracy > 0.7:
        deploy_and_predict()

if __name__ == "__main__":
    main()
from pipelines.deployment_pipeline import deploy_and_predict
from pipelines.training_pipeline import train_pipeline
import logging
from steps.best_model_select import best_model_selector
from zenml.client import Client

def main():
    # Execute the training pipeline
    train_pipeline()

    # Retrieve and log the test accuracy of the best model from the training pipeline
    best_model_test_accuracy = (
        Client()
        .get_pipeline("train_pipeline")
        .last_successful_run.steps["best_model_selector"]
        .outputs["best_model_test_acc"]
        .load()
    )
    logging.info(f"Best model test accuracy: {best_model_test_accuracy}")

    # Deploy and predict using the best model if the test accuracy meets a threshold
    if best_model_test_accuracy > 0.7:
        deploy_and_predict()

if __name__ == "__main__":
    # Entry point of the script
    main()

from pipelines.training_pipeline import train_pipeline
import click
from zenml.client import Client
import mlflow

@click.command()
@click.option(
    "--type",
    default="tracking",
    help="The type of MLflow example to run.",
    type=click.Choice(["tracking", "report", "deployment"]),
)



def main(type: str) -> None:
    if type == "tracking":
        train_pipeline()
    else:
        raise NotImplementedError(
            f"MLflow example type {type} not implemented."
        )

if __name__ == "__main__":
    tracking_uri = mlflow.get_tracking_uri()
    print(tracking_uri)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    main()
    

    
    
# zenml model-registry models register-version project2-model \
#     --description="A new version of the project2 model with accuracy 98.88%" \
#     -v 1 \
#     --model-uri="file:/home/rashid/.config/zenml/local_stores/373e9b99-a08f-45bb-9da2-6866c726029b/mlruns/667102566783201219/3973eabc151c41e6ab98baeb20c5323b/artifacts/model" \
#     -m key1 value1 -m key2 value2 \
#     --zenml-pipeline-name="train_pipeline" 

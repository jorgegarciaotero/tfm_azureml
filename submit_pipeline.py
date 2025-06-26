from azure.ai.ml import MLClient, load_component, Input
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target_date", type=str, required=True)
args = parser.parse_args()

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="0d509562-ae9c-4798-b0c6-42a19eeebe22",
    resource_group_name="intelligent_wallet_jorge",
    workspace_name="es-smartwallet-pro-ml",
)

batch_component = load_component(source="component.yaml")

@pipeline()
def my_pipeline(target_date: str):
    batch_consumer_job = batch_component(
        target_date=target_date,
    )
    batch_consumer_job.compute = "jorgegarciaotero1"
    batch_consumer_job.environment = "azureml:batch-env-final:13"
    return batch_consumer_job

job = my_pipeline(
    target_date=args.target_date,
)

ml_client.jobs.create_or_update(job)

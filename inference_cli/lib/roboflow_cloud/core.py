import typer

from inference_cli.lib.roboflow_cloud.batch_processing.core import batch_processing_app
from inference_cli.lib.roboflow_cloud.data_staging.core import data_staging_app

rf_cloud_app = typer.Typer(help="Commands for interacting with Roboflow Cloud")
rf_cloud_app.add_typer(data_staging_app, name="data-staging")
rf_cloud_app.add_typer(batch_processing_app, name="batch-processing")

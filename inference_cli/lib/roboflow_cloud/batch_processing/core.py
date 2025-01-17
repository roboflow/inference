from typing import Annotated, Optional

import typer

from inference_cli.lib.env import ROBOFLOW_API_KEY
from inference_cli.lib.roboflow_cloud.batch_processing.api_operations import (
    display_batch_job_details,
    display_batch_jobs,
    trigger_job_with_workflows_images_processing,
)

batch_processing_app = typer.Typer(
    help="Commands for interacting with Roboflow Batch Processing"
)


@batch_processing_app.command(help="List batch jobs in your workspace.")
def list_jobs(
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        display_batch_jobs(api_key=api_key)
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@batch_processing_app.command(help="Get job details.")
def describe_job(
    job_id: Annotated[
        str,
        typer.Option(
            "--job-id",
            "-j",
            help="Identifier of job",
        ),
    ],
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        display_batch_job_details(job_id=job_id, api_key=api_key)
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@batch_processing_app.command(help="Trigger batch job to process images with Workflow")
def process_images_with_workflow(
    batch_id: Annotated[
        str,
        typer.Option("--batch-id", "-b", help="Identifier of batch to be processed"),
    ],
    job_id: Annotated[
        Optional[str],
        typer.Option(
            "--job-id",
            "-j",
            help="Identifier of job (if not given - will be generated)",
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        job_id = trigger_job_with_workflows_images_processing(
            batch_id=batch_id,
            job_id=job_id,
            api_key=api_key,
        )
        print(f"Triggered job with ID: {job_id}")
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)

from typing import List, Optional

import typer
from typing_extensions import Annotated

from inference_cli.lib.env import ROBOFLOW_API_KEY
from inference_cli.lib.roboflow_cloud.batch_processing.api_operations import (
    display_batch_job_details,
    display_batch_jobs,
    trigger_job_with_workflows_images_processing,
    trigger_job_with_workflows_videos_processing,
)
from inference_cli.lib.roboflow_cloud.batch_processing.entities import (
    AggregationFormat,
    MachineSize,
    MachineType,
)

batch_processing_app = typer.Typer(
    help="Commands for interacting with Roboflow Batch Processing. THIS IS ALPHA PREVIEW OF THE FEATURE."
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
    max_pages: Annotated[
        Optional[int],
        typer.Option(
            "--max-pages",
            "-m",
            help="Number of pagination pages with batch jobs to display",
        ),
    ] = 1,
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
        display_batch_jobs(api_key=api_key, max_pages=max_pages)
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@batch_processing_app.command(help="Get job details.")
def show_job_details(
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
    workflow_id: Annotated[
        str,
        typer.Option("--workflow-id", "-w", help="Identifier of the workflow"),
    ],
    workflow_parameters_path: Annotated[
        Optional[str],
        typer.Option(
            "--workflow-params",
            help="Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and "
            "passing the parameters in CLI is not handy / impossible due to typing conversion issues.",
        ),
    ] = None,
    image_input_name: Annotated[
        Optional[str],
        typer.Option(
            "--image-input-name",
            help="Name of the Workflow input that defines placeholder for image to be processed",
        ),
    ] = None,
    save_image_outputs: Annotated[
        bool,
        typer.Option(
            "--save-image-outputs/--no-save-image-outputs",
            help="Flag controlling persistence of Workflow outputs that are images",
        ),
    ] = False,
    image_outputs_to_save: Annotated[
        Optional[List[str]],
        typer.Option(
            "--image-outputs-to-save",
            help="Use this option to filter out workflow image outputs you want to save",
        ),
    ] = None,
    part_name: Annotated[
        Optional[str],
        typer.Option(
            "--part-name",
            "-p",
            help="Name of the batch part " "(relevant for multipart batches",
        ),
    ] = None,
    machine_type: Annotated[
        Optional[MachineType],
        typer.Option("--machine-type", "-mt", help="Type of machine"),
    ] = None,
    machine_size: Annotated[
        Optional[MachineSize],
        typer.Option("--machine-size", "-ms", help="Size of machine"),
    ] = None,
    max_runtime_seconds: Annotated[
        Optional[int],
        typer.Option("--max-runtime-seconds", help="Max processing duration"),
    ] = None,
    max_parallel_tasks: Annotated[
        Optional[int],
        typer.Option(
            "--max-parallel-tasks", help="Max number of concurrent processing tasks"
        ),
    ] = None,
    aggregation_format: Annotated[
        Optional[AggregationFormat],
        typer.Option("--aggregation-format", help="Format of results aggregation"),
    ] = None,
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
            workflow_id=workflow_id,
            workflow_parameters_path=workflow_parameters_path,
            image_input_name=image_input_name,
            save_image_outputs=save_image_outputs,
            image_outputs_to_save=image_outputs_to_save,
            part_name=part_name,
            machine_type=machine_type,
            machine_size=machine_size,
            max_runtime_seconds=max_runtime_seconds,
            max_parallel_tasks=max_parallel_tasks,
            aggregation_format=aggregation_format,
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


@batch_processing_app.command(help="Trigger batch job to process videos with Workflow")
def process_videos_with_workflow(
    batch_id: Annotated[
        str,
        typer.Option("--batch-id", "-b", help="Identifier of batch to be processed"),
    ],
    workflow_id: Annotated[
        str,
        typer.Option("--workflow-id", "-w", help="Identifier of the workflow"),
    ],
    workflow_parameters_path: Annotated[
        Optional[str],
        typer.Option(
            "--workflow-params",
            help="Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and "
            "passing the parameters in CLI is not handy / impossible due to typing conversion issues.",
        ),
    ] = None,
    image_input_name: Annotated[
        Optional[str],
        typer.Option(
            "--image-input-name",
            help="Name of the Workflow input that defines placeholder for image to be processed",
        ),
    ] = None,
    save_image_outputs: Annotated[
        bool,
        typer.Option(
            "--save-image-outputs/--no-save-image-outputs",
            help="Flag controlling persistence of Workflow outputs that are images",
        ),
    ] = False,
    image_outputs_to_save: Annotated[
        Optional[List[str]],
        typer.Option(
            "--image-outputs-to-save",
            help="Use this option to filter out workflow image outputs you want to save",
        ),
    ] = None,
    part_name: Annotated[
        Optional[str],
        typer.Option(
            "--part-name",
            "-p",
            help="Name of the batch part " "(relevant for multipart batches",
        ),
    ] = None,
    machine_type: Annotated[
        Optional[MachineType],
        typer.Option("--machine-type", "-mt", help="Type of machine"),
    ] = None,
    machine_size: Annotated[
        Optional[MachineSize],
        typer.Option("--machine-size", "-ms", help="Size of machine"),
    ] = None,
    max_runtime_seconds: Annotated[
        Optional[int],
        typer.Option("--max-runtime-seconds", help="Max processing duration"),
    ] = None,
    max_parallel_tasks: Annotated[
        Optional[int],
        typer.Option(
            "--max-parallel-tasks", help="Max number of concurrent processing tasks"
        ),
    ] = None,
    aggregation_format: Annotated[
        Optional[AggregationFormat],
        typer.Option("--aggregation-format", help="Format of results aggregation"),
    ] = None,
    max_video_fps: Annotated[
        Optional[int],
        typer.Option(
            "--max-video-fps",
            help="Limit for FPS to process for video (subsampling predictions rate) - smaller FPS means faster "
            "processing and less accurate video analysis.",
        ),
    ] = None,
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
        job_id = trigger_job_with_workflows_videos_processing(
            batch_id=batch_id,
            workflow_id=workflow_id,
            workflow_parameters_path=workflow_parameters_path,
            image_input_name=image_input_name,
            save_image_outputs=save_image_outputs,
            image_outputs_to_save=image_outputs_to_save,
            part_name=part_name,
            machine_type=machine_type,
            machine_size=machine_size,
            max_runtime_seconds=max_runtime_seconds,
            max_parallel_tasks=max_parallel_tasks,
            aggregation_format=aggregation_format,
            max_video_fps=max_video_fps,
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

import json
from typing import List, Optional

import typer
from typing_extensions import Annotated

from inference_cli.lib.benchmark.dataset import PREDEFINED_DATASETS
from inference_cli.lib.benchmark_adapter import (
    run_infer_api_speed_benchmark,
    run_python_package_speed_benchmark,
    run_workflow_api_speed_benchmark,
)

benchmark_app = typer.Typer(help="Commands for running inference benchmarks.")


@benchmark_app.command()
def api_speed(
    model_id: Annotated[
        Optional[str],
        typer.Option(
            "--model_id",
            "-m",
            help="Model ID in format project/version.",
        ),
    ] = None,
    workflow_id: Annotated[
        Optional[str],
        typer.Option(
            "--workflow-id",
            "-wid",
            help="Workflow ID.",
        ),
    ] = None,
    workspace_name: Annotated[
        Optional[str],
        typer.Option(
            "--workspace-name",
            "-wn",
            help="Workspace Name.",
        ),
    ] = None,
    workflow_specification: Annotated[
        Optional[str],
        typer.Option(
            "--workflow-specification",
            "-ws",
            help="Workflow specification.",
        ),
    ] = None,
    workflow_parameters: Annotated[
        Optional[str],
        typer.Option(
            "--workflow-parameters",
            "-wp",
            help="Model ID in format project/version.",
        ),
    ] = None,
    dataset_reference: Annotated[
        str,
        typer.Option(
            "--dataset_reference",
            "-d",
            help=f"Name of predefined dataset (one of {list(PREDEFINED_DATASETS.keys())}) or path to directory with images",
        ),
    ] = "coco",
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to run inference on."),
    ] = "http://localhost:9001",
    warm_up_requests: Annotated[
        int,
        typer.Option("--warm_up_requests", "-wr", help="Number of warm-up requests"),
    ] = 10,
    benchmark_requests: Annotated[
        int,
        typer.Option(
            "--benchmark_requests", "-br", help="Number of benchmark requests"
        ),
    ] = 1000,
    request_batch_size: Annotated[
        int,
        typer.Option("--batch_size", "-bs", help="Batch size of single request"),
    ] = 1,
    number_of_clients: Annotated[
        int,
        typer.Option(
            "--clients",
            "-c",
            help="Meaningful if `rps` not specified - number of concurrent threads that will send requests one by one",
        ),
    ] = 1,
    requests_per_second: Annotated[
        Optional[int],
        typer.Option(
            "--rps",
            "-rps",
            help="Number of requests per second to emit. If not specified - requests will be sent one-by-one by requested number of client threads",
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
    model_configuration: Annotated[
        Optional[str],
        typer.Option(
            "--model_config", "-mc", help="Location of yaml file with model config"
        ),
    ] = None,
    output_location: Annotated[
        Optional[str],
        typer.Option(
            "--output_location",
            "-o",
            help="Location where to save the result (path to file or directory)",
        ),
    ] = None,
    enforce_legacy_endpoints: Annotated[
        bool,
        typer.Option(
            "--legacy-endpoints/--no-legacy-endpoints",
            "-L/-l",
            help="Boolean flag to decide if legacy endpoints should be used (applicable for self-hosted API benchmark)",
        ),
    ] = False,
    proceed_automatically: Annotated[
        bool,
        typer.Option(
            "--yes/--no",
            "-y/-n",
            help="Boolean flag to decide on auto `yes` answer given on user input required.",
        ),
    ] = False,
    max_error_rate: Annotated[
        Optional[float],
        typer.Option(
            "--max_error_rate",
            help="Max error rate for API speed benchmark - if given and the error rate is higher - command will "
            "return non-success error code. Expected percentage values in range 0.0-100.0",
        ),
    ] = None,
    endpoint: Annotated[
        Optional[str],
        typer.Option(
            "--endpoint",
            "-e",
            help="SAM3 endpoint to infer against, use /seg-preview/segment_image",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Format of the output. One of 'polygon', 'rle', or 'binary'.",
        ),
    ] = "polygon",
    text: Annotated[
        Optional[str],
        typer.Option(
            "--text", "-t", help="Text query for open-vocabulary segmentation."
        ),
    ] = None,
    points: Annotated[
        Optional[str],
        typer.Option(
            "--points",
            "-p",
            help="List of [x, y] points normalized to 0-1. Used only with instance_prompt=True. Format: '[[x1,y1],[x2,y2],...]'",
        ),
    ] = None,
    point_labels: Annotated[
        Optional[List[int]],
        typer.Option("--point_labels", "-pl", help="List of 0/1 labels for points."),
    ] = None,
    boxes: Annotated[
        Optional[str],
        typer.Option(
            "--boxes",
            "-b",
            help="List of [x, y, w, h] boxes normalized to 0-1. Format: '[[x1,y1,w1,h1],[x2,y2,w2,h2],...]'",
        ),
    ] = None,
    box_labels: Annotated[
        Optional[List[int]],
        typer.Option("--box_labels", "-bl", help="List of 0/1 labels for boxes."),
    ] = None,
    instance_prompt: Annotated[
        Optional[bool],
        typer.Option(
            "--instance_prompt",
            "-ip",
            help="Enable instance tracking style point prompts.",
        ),
    ] = False,
    output_prob_thresh: Annotated[
        Optional[float],
        typer.Option(
            "--output_prob_thresh", "-otp", help="Score threshold for outputs."
        ),
    ] = None,
):
    if "roboflow.com" in host and not proceed_automatically:
        proceed = input(
            "This action may easily exceed your Roboflow inference credits. Are you sure? [y/N] "
        )
        if proceed.lower() != "y":
            return None

    parsed_points = None
    if points:
        try:
            parsed_points = json.loads(points)
        except json.JSONDecodeError:
            typer.echo(
                f"Error parsing points: {points}. Expected format: '[[x1,y1],[x2,y2],...]'"
            )
            raise typer.Exit(code=1)

    parsed_boxes = None
    if boxes:
        try:
            parsed_boxes = json.loads(boxes)
        except json.JSONDecodeError:
            typer.echo(
                f"Error parsing boxes: {boxes}. Expected format: '[[x1,y1,w1,h1],[x2,y2,w2,h2],...]'"
            )
            raise typer.Exit(code=1)

    sam3_params = {
        "endpoint": endpoint,
        "format": format,
        "text": text,
        "points": parsed_points,
        "point_labels": point_labels,
        "boxes": parsed_boxes,
        "box_labels": box_labels,
        "instance_prompt": instance_prompt,
        "output_prob_thresh": output_prob_thresh,
    }

    try:
        if model_id:
            run_infer_api_speed_benchmark(
                model_id=model_id,
                dataset_reference=dataset_reference,
                host=host,
                warm_up_requests=warm_up_requests,
                benchmark_requests=benchmark_requests,
                request_batch_size=request_batch_size,
                number_of_clients=number_of_clients,
                requests_per_second=requests_per_second,
                api_key=api_key,
                model_configuration=model_configuration,
                output_location=output_location,
                enforce_legacy_endpoints=enforce_legacy_endpoints,
                max_error_rate=max_error_rate,
                sam3_params=sam3_params,
            )
        else:
            if workflow_specification:
                workflow_specification = json.loads(workflow_specification)
            if workflow_parameters:
                workflow_parameters = json.loads(workflow_parameters)
            run_workflow_api_speed_benchmark(
                workflow_id=workflow_id,
                workspace_name=workspace_name,
                workflow_specification=workflow_specification,
                workflow_parameters=workflow_parameters,
                dataset_reference=dataset_reference,
                host=host,
                warm_up_requests=warm_up_requests,
                benchmark_requests=benchmark_requests,
                request_batch_size=request_batch_size,
                number_of_clients=number_of_clients,
                requests_per_second=requests_per_second,
                api_key=api_key,
                model_configuration=model_configuration,
                output_location=output_location,
                max_error_rate=max_error_rate,
            )
    except Exception as error:
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@benchmark_app.command()
def python_package_speed(
    model_id: Annotated[
        str,
        typer.Option(
            "--model_id",
            "-m",
            help="Model ID in format project/version.",
        ),
    ],
    dataset_reference: Annotated[
        str,
        typer.Option(
            "--dataset_reference",
            "-d",
            help=f"Name of predefined dataset (one of {list(PREDEFINED_DATASETS.keys())}) or path to directory with images",
        ),
    ] = "coco",
    warm_up_inferences: Annotated[
        int,
        typer.Option("--warm_up_inferences", "-wi", help="Number of warm-up requests"),
    ] = 10,
    benchmark_inferences: Annotated[
        int,
        typer.Option(
            "--benchmark_requests", "-bi", help="Number of benchmark requests"
        ),
    ] = 1000,
    batch_size: Annotated[
        int,
        typer.Option("--batch_size", "-bs", help="Batch size of single request"),
    ] = 1,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    model_configuration: Annotated[
        Optional[str],
        typer.Option(
            "--model_config", "-mc", help="Location of yaml file with model config"
        ),
    ] = None,
    output_location: Annotated[
        Optional[str],
        typer.Option(
            "--output_location",
            "-o",
            help="Location where to save the result (path to file or directory)",
        ),
    ] = None,
):
    try:
        run_python_package_speed_benchmark(
            model_id=model_id,
            dataset_reference=dataset_reference,
            warm_up_inferences=warm_up_inferences,
            benchmark_inferences=benchmark_inferences,
            batch_size=batch_size,
            api_key=api_key,
            model_configuration=model_configuration,
            output_location=output_location,
        )
    except KeyboardInterrupt:
        print("Benchmark interrupted.")
        return
    except Exception as error:
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    benchmark_app()

from typing import List, Literal, Optional

import typer
from typing_extensions import Annotated

import inference_cli.lib
from inference_cli.benchmark import benchmark_app
from inference_cli.cloud import cloud_app
from inference_cli.lib.roboflow_cloud.core import rf_cloud_app
from inference_cli.server import server_app
from inference_cli.workflows import workflows_app

app = typer.Typer()
app.add_typer(server_app, name="server")
app.add_typer(cloud_app, name="cloud")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(workflows_app, name="workflows")
app.add_typer(rf_cloud_app, name="rf-cloud")


def version_callback(value: bool):
    if value:
        from importlib.metadata import PackageNotFoundError, version

        version_msg = ""
        for package in ["inference", "inference-sdk", "inference-cli"]:
            try:
                package_version = version(package)
                version_msg += f"{package} version: v{package_version}\n"
            except PackageNotFoundError:
                if package == "inference":
                    version_msg = f"{package} version: using local install"
                    break

        typer.echo(version_msg)

        raise typer.Exit()


@app.callback()
def version(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
):
    return


@app.command()
def infer(
    input_reference: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="URL or local path of image / directory with images or video to run inference on.",
        ),
    ],
    model_id: Annotated[
        str,
        typer.Option(
            "--model_id",
            "-m",
            help="Model ID in format project/version.",
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
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to run inference on."),
    ] = "http://localhost:9001",
    output_location: Annotated[
        Optional[str],
        typer.Option(
            "--output_location",
            "-o",
            help="Location where to save the result (path to directory)",
        ),
    ] = None,
    display: Annotated[
        bool,
        typer.Option(
            "--display/--no-display",
            "-D/-d",
            help="Boolean flag to decide if visualisations should be displayed on the screen",
        ),
    ] = False,
    visualise: Annotated[
        bool,
        typer.Option(
            "--visualise/--no-visualise",
            "-V/-v",
            help="Boolean flag to decide if visualisations should be preserved",
        ),
    ] = True,
    visualisation_config: Annotated[
        Optional[str],
        typer.Option(
            "--visualisation_config",
            "-c",
            help="Location of yaml file with visualisation config",
        ),
    ] = None,
    model_config: Annotated[
        Optional[str],
        typer.Option(
            "--model_config", "-mc", help="Location of yaml file with model config"
        ),
    ] = None,
    endpoint: Annotated[
        Optional[str],
        typer.Option("--endpoint", "-e", help="Endpoint to infer against"),
    ] = None,
    format: Annotated[
        Literal["polygon", "rle", "binary"],
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
        Optional[List[List[float]]],
        typer.Option(
            "--points",
            "-p",
            help="List of [x, y] points normalized to 0-1. Used only with instance_prompt=True.",
        ),
    ] = None,
    point_labels: Annotated[
        Optional[List[int]],
        typer.Option("--point_labels", "-pl", help="List of 0/1 labels for points."),
    ] = None,
    boxes: Annotated[
        Optional[List[List[float]]],
        typer.Option(
            "--boxes", "-b", help="List of [x, y, w, h] boxes normalized to 0-1."
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
    typer.echo(
        f"Running inference on {input_reference}, using model: {model_id}, and host: {host}"
    )
    try:
        sam3_params = {
            "format": format,
            "text": text,
            "points": points,
            "point_labels": point_labels,
            "boxes": boxes,
            "box_labels": box_labels,
            "instance_prompt": instance_prompt,
            "output_prob_thresh": output_prob_thresh,
        }
        inference_cli.lib.infer(
            input_reference=input_reference,
            model_id=model_id,
            api_key=api_key,
            host=host,
            output_location=output_location,
            display=display,
            visualise=visualise,
            visualisation_config=visualisation_config,
            model_configuration=model_config,
            endpoint=endpoint,
            sam3_params=sam3_params,
        )
    except Exception as error:
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

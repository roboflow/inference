import typer
from typing_extensions import Annotated
import inference_cli.actions
from inference_cli.lib import start_inference_container

app = typer.Typer()


@app.command()
def serve(
    port: int = typer.Option(
        9001,
        "-p",
        "--port",
        help="Port to run the inference server on (default is 9001).",
    ),
    project: str = typer.Option(
        "roboflow-platform",
        "-pr",
        "--project",
        help="Project to run the inference server with (default is roboflow-platform).",
    ),
):
    """Start the inference server."""
    print(f"starting inference server on port {port}")
    start_inference_container("", port=port, project=project)


@app.command()
def infer(
    image: str = typer.Option(
        None, "-i", "--image", help="URL of image to run inference on."
    ),
    project_id: str = typer.Option(
        None, "-p", "--project_id", help="Project to run inference with."
    ),
    model_version: str = typer.Option(
        None, "-v", "--model_version", help="Version of model to run inference with."
    ),
    api_key: str = typer.Option(
        None, "-a", "--api_key", help="Path to save output image to."
    ),
    host: str = typer.Option(
        "http://localhost:9001", "-h", "--host", help="Host to run inference on."
    ),
):
    typer.echo(
        f"Running inference with image {image}, project {project_id}, version {model_version}, API key {api_key}, and host {host}"
    )

    inference_cli.actions.infer(image, project_id, model_version, api_key, host)


if __name__ == "__main__":
    app()

import typer
from inference_cli.lib import check_inference_server_status, start_inference_container
from typing_extensions import Annotated

server_app = typer.Typer(
    help="""Commands for running the inference server locally. \n 
    Supported devices targets are x86 CPU, ARM64 CPU, and NVIDIA GPU."""
)


@server_app.command()
def start(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to run the inference server on (default is 9001).",
        ),
    ] = 9001,
    rf_env: Annotated[
        str,
        typer.Option(
            "--rf-env",
            "-rfe",
            help="Roboflow environment to run the inference server with (default is roboflow-platform).",
        ),
    ] = "roboflow-platform",
):
    start_inference_container("", port=port, project=rf_env)


@server_app.command()
def status():
    print("Checking status of inference server.")
    check_inference_server_status()


if __name__ == "__main__":
    server_app()

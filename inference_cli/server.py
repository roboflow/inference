from typing import Optional

import typer
from typing_extensions import Annotated

from inference_cli.lib import check_inference_server_status, start_inference_container
from inference_cli.lib.container_adapter import stop_inference_containers

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
    env_file_path: Annotated[
        Optional[str],
        typer.Option(
            "--env-file",
            "-e",
            help="Path to file with env variables (in each line KEY=VALUE). Optional. If given - values will be "
            "overriden by any explicit parameter of this command.",
        ),
    ] = None,
    development: Annotated[
        bool,
        typer.Option(
            "--dev",
            "-d",
            help="Run inference server in development mode (default is False).",
        ),
    ] = False,
    api_key: Annotated[
        str,
        typer.Option(
            "--roboflow-api-key",
            "-k",
            help="Roboflow API key (default is None).",
        ),
    ] = None,
):
    start_inference_container(
        port=port,
        project=rf_env,
        env_file_path=env_file_path,
        development=development,
        api_key=api_key,
    )


@server_app.command()
def status():
    print("Checking status of inference server.")
    check_inference_server_status()


@server_app.command()
def stop() -> None:
    print("Terminating running inference containers")
    stop_inference_containers()


if __name__ == "__main__":
    server_app()

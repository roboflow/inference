from typing import Optional

import typer
from typing_extensions import Annotated

from inference_cli.lib import check_inference_server_status, start_inference_container
from inference_cli.lib.container_adapter import (
    ensure_docker_is_running,
    stop_inference_containers,
)
from inference_cli.lib.tunnel_adapter import start_tunnel, stop_tunnel_container

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
    tunnel: Annotated[
        bool,
        typer.Option(
            "--tunnel",
            help="Start a tunnel to expose inference to external requests on a TLS-enabled "
            "https://<subdomain>.roboflow.run endpoint",
        ),
    ] = False,
    image: Annotated[
        Optional[str],
        typer.Option(
            "--image",
            help="Point specific docker image you would like to run with command (useful for development of custom "
            "builds of inference server)",
        ),
    ] = None,
    use_local_images: Annotated[
        bool,
        typer.Option(
            "--use-local-images/--not-use-local-images",
            help="Flag to allow using local images (if set False image is always attempted to be pulled)",
        ),
    ] = False,
    metrics_enabled: Annotated[
        bool,
        typer.Option(
            "--metrics-enabled/--metrics-disabled",
            help="Flag controlling if metrics are enabled (default is True)",
        ),
    ] = True,
) -> None:

    try:
        ensure_docker_is_running()
    except Exception as docker_error:
        typer.echo(docker_error)
        raise typer.Exit(code=1) from docker_error

    try:
        start_inference_container(
            image=image,
            port=port,
            project=rf_env,
            env_file_path=env_file_path,
            development=development,
            api_key=api_key,
            use_local_images=use_local_images,
            metrics_enabled=metrics_enabled,
        )
    except Exception as container_error:
        typer.echo(container_error)
        raise typer.Exit(code=1) from container_error

    if tunnel:
        if api_key is None:
            typer.echo("Roboflow API Key is required to start the tunnel")
            raise typer.Exit(code=1)

        tunnel_url = start_tunnel(
            api_key=api_key,
            inference_port=port,
        )
        typer.echo(f"Tunnel to local inference running on {tunnel_url}")


@server_app.command()
def status() -> None:
    typer.echo("Checking status of the inference server.")
    try:
        ensure_docker_is_running()
    except Exception as docker_error:
        typer.echo(docker_error)
        raise typer.Exit(code=1) from docker_error

    try:
        check_inference_server_status()
    except Exception as status_error:
        typer.echo(status_error)
        raise typer.Exit(code=1) from status_error


@server_app.command()
def stop() -> None:
    typer.echo("Terminating running inference containers.")
    try:
        ensure_docker_is_running()
    except Exception as docker_error:
        typer.echo(docker_error)
        raise typer.Exit(code=1) from docker_error

    try:
        stop_inference_containers()
    except Exception as stop_error:
        typer.echo(stop_error)
        raise typer.Exit(code=1) from stop_error

    try:
        stop_tunnel_container()
    except Exception as stop_error:
        typer.echo(stop_error)
        raise typer.Exit(code=1) from stop_error


if __name__ == "__main__":
    server_app()

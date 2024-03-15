from typing import Optional

import typer
from typing_extensions import Annotated

from inference_cli.lib import (
    cloud_deploy,
    cloud_start,
    cloud_status,
    cloud_stop,
    cloud_undeploy,
)

cloud_app = typer.Typer(
    help="""Commands for running the inference in cloud with skypilot. \n 
    Supported devices targets are x86 CPU and NVIDIA GPU VMs."""
)


@cloud_app.command()
def status():
    try:
        cloud_status()
    except Exception as error:
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@cloud_app.command()
def deploy(
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Cloud provider to deploy to. Currently aws or gcp.",
        ),
    ],
    compute_type: Annotated[
        str,
        typer.Option(
            "--compute-type",
            "-t",
            help="Execution environment to deploy to: cpu or gpu.",
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-d",
            help="Print out deployment plan without executing.",
        ),
    ] = False,
    custom: Annotated[
        str,
        typer.Option(
            "--custom",
            "-c",
            help="Path to config file to override default config.",
        ),
    ] = None,
    roboflow_api_key: Annotated[
        str,
        typer.Option(
            "--roboflow-api-key",
            "-r",
            help="Roboflow API key for your workspace.",
        ),
    ] = None,
    help: Annotated[
        bool,
        typer.Option(
            "--help",
            "-h",
            help="Print out help text.",
        ),
    ] = False,
    # For later, when notebook becomes secure
    # notebook: Annotated[
    #     bool,
    #     typer.Option(
    #         "--notebook",
    #         "-n",
    #         help="Expose the notebook instance at port 9002 (caution - can be insecure).",
    #     ),
    # ] = False,
):
    try:
        cloud_deploy(provider, compute_type, dry_run, custom, help, roboflow_api_key)
    except Exception as error:
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@cloud_app.command()
def undeploy(
    cluster_name: Annotated[str, typer.Argument(help="Name of cluster to undeploy.")]
):
    try:
        cloud_undeploy(cluster_name)
    except Exception as error:
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@cloud_app.command()
def stop(cluster_name: Annotated[str, typer.Argument(help="Name of cluster to stop.")]):
    try:
        cloud_stop(cluster_name)
    except Exception as error:
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@cloud_app.command()
def start(
    cluster_name: Annotated[str, typer.Argument(help="Name of cluster to start.")]
):
    try:
        cloud_start(cluster_name)
    except Exception as error:
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)

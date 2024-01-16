import typer
from typing_extensions import Annotated

import inference_cli.lib
from inference_cli.server import server_app

app = typer.Typer()

app.add_typer(server_app, name="server")

@app.command()
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
    help: Annotated[
        bool,
        typer.Option(
            "--help",
            "-h",
            help="Print out help text.",
        ),
    ] = False
):
    inference_cli.lib.deploy(provider, compute_type, dry_run, custom, help)

@app.command()
def undeploy(cluster_name: Annotated[str, typer.Argument(help="Name of cluster to undeploy.")]):
    inference_cli.lib.undeploy(cluster_name)

@app.command()
def infer(
    image: Annotated[
        str, typer.Argument(help="URL or local path of image to run inference on.")
    ],
    project_id: Annotated[
        str,
        typer.Option(
            "--project-id", "-p", help="Roboflow project to run inference with."
        ),
    ],
    model_version: Annotated[
        str,
        typer.Option(
            "--model-version",
            "-v",
            help="Version of model to run inference with.",
        ),
    ],
    api_key: Annotated[
        str,
        typer.Option("--api-key", "-a", help="Roboflow API key for your workspace."),
    ],
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to run inference on."),
    ] = "http://localhost:9001",
):
    typer.echo(
        f"Running inference on image {image}, using model ({project_id}/{model_version}), and host ({host})"
    )
    inference_cli.lib.infer(image, project_id, model_version, api_key, host)


if __name__ == "__main__":
    app()

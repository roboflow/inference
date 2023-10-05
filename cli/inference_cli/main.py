import typer
from inference_cli.server import server_app
import inference_cli.lib

app = typer.Typer()

app.add_typer(server_app, name="server")


@app.command()
def infer(
    image: str = typer.Option(
        None, "-i", "--image", help="URL or local path of image to run inference on."
    ),
    project_id: str = typer.Option(
        None, "-p", "--project-id", help="Roboflow project to run inference with."
    ),
    model_version: str = typer.Option(
        None, "-v", "--model-version", help="Version of model to run inference with."
    ),
    api_key: str = typer.Option(
        None, "-a", "--api-key", help="Roboflow API key for your workspace."
    ),
    host: str = typer.Option(
        "http://localhost:9001", "-h", "--host", help="Host to run inference on."
    ),
):
    typer.echo(
        f"Running inference on image {image}, using model ({project_id}/{model_version}), and host ({host})"
    )
    inference_cli.lib.infer(image, project_id, model_version, api_key, host)


if __name__ == "__main__":
    app()

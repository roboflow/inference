import typer
from inference_cli.lib import start_inference_container, check_inference_server_status

server_app = typer.Typer()


@server_app.command()
def start(
    port: int = typer.Option(
        9001,
        "-p",
        "--port",
        help="Port to run the inference server on (default is 9001).",
    ),
    rf_env: str = typer.Option(
        "roboflow-platform",
        "-rfe",
        "--rf-env",
        help="Roboflow environment to run the inference server with (default is roboflow-platform).",
    ),
):
    start_inference_container("", port=port, project=rf_env)


@server_app.command()
def status():
    print("Checking status of inference server.")
    check_inference_server_status()


if __name__ == "__main__":
    server_app()

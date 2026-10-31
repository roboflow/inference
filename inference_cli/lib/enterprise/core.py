import typer

from inference_cli.lib.enterprise.inference_compiler.cli.core import (
    inference_compiler_app,
)

enterprise_app = typer.Typer(help="Roboflow Enterprise commands")
enterprise_app.add_typer(inference_compiler_app, name="inference-compiler")

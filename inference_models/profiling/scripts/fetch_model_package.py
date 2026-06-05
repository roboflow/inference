from pathlib import Path
from typing import Optional

import click

from inference_models import BackendType, Quantization
from profiling.memory.package_resolve import resolve_package_directory


@click.command(
    help="Download an inference_models package so it can be used by profiling runs.",
)
@click.option(
    "--model-id",
    type=str,
    required=True,
)
@click.option(
    "--model-architecture",
    type=str,
    required=True,
)
@click.option(
    "--task-type",
    type=str,
    required=True,
)
@click.option(
    "--backend",
    type=click.Choice(
        [backend.value for backend in BackendType],
    ),
    required=True,
)
@click.option(
    "--package-id",
    type=str,
    default=None,
    help="Optional explicit package id. Defaults to the first matching backend.",
)
@click.option(
    "--quantization",
    type=click.Choice(
        [quantization.value for quantization in Quantization],
    ),
    default=None,
    help="Optional package quantization.",
)
@click.option(
    "--provider",
    type=str,
    default="roboflow",
    show_default=True,
)
@click.option(
    "--api-key",
    envvar="ROBOFLOW_API_KEY",
)
@click.option(
    "--target-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    default=Path("tmp"),
    show_default=True,
)
def main(
    model_id: str,
    model_architecture: str,
    task_type: str,
    backend: str,
    package_id: Optional[str],
    quantization: Optional[str],
    provider: str,
    api_key: Optional[str],
    target_dir: Path,
) -> None:
    """Download a model package for local memory profiling runs.

    Resolves metadata from the weights provider, selects a matching package for the
    requested backend and quantization, and writes artifacts under ``target_dir``.
    """
    backend_type = BackendType(backend)
    quantization_type = Quantization(quantization) if quantization else None

    try:
        package_dir, package, _model_variant = resolve_package_directory(
            model_id=model_id,
            model_architecture=model_architecture,
            task_type=task_type,
            backend=backend_type,
            quantization=quantization_type,
            package_id=package_id,
            target_dir=target_dir,
            provider=provider,
            api_key=api_key,
            fetch_if_missing=True,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    click.echo(package.get_summary())
    click.echo(f"Downloaded package to: {package_dir}")


if __name__ == "__main__":
    main()

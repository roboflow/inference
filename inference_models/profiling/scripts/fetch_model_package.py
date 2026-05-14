from pathlib import Path
from typing import Optional

import click

from inference_models import BackendType
from inference_models.developer_tools import (
    ModelMetadata,
    ModelPackageMetadata,
    download_files_to_directory,
    get_model_from_provider,
)


def _validate_metadata(
    metadata: ModelMetadata,
    *,
    model_architecture: str,
    task_type: str,
) -> None:
    if metadata.model_architecture != model_architecture:
        raise click.ClickException(
            f"Expected architecture {model_architecture!r}, "
            f"got {metadata.model_architecture!r}."
        )

    if metadata.task_type != task_type:
        raise click.ClickException(
            f"Expected task type {task_type!r}, got {metadata.task_type!r}."
        )


def _select_package(
    metadata: ModelMetadata,
    *,
    backend: BackendType,
    package_id: Optional[str],
) -> ModelPackageMetadata:
    candidates = [
        package
        for package in metadata.model_packages
        if package.backend == backend
        and (package_id is None or package.package_id == package_id)
    ]

    if not candidates:
        available = ", ".join(
            f"{package.package_id}:{package.backend.value}"
            for package in metadata.model_packages
        )
        raise click.ClickException(
            f"No package found for backend={backend.value!r}"
            + (f" and package_id={package_id!r}" if package_id else "")
            + f". Available packages: {available}"
        )

    selected_package = candidates[0]

    return selected_package


def _make_package_dir(
    target_dir: Path,
    *,
    model_id: str,
    model_architecture: str,
    task_type: str,
    backend: BackendType,
    package_id: str,
) -> Path:
    safe_model_id = model_id.replace("/", "-")
    package_dir = (
        target_dir
        / model_architecture
        / task_type
        / backend.value
        / safe_model_id
        / package_id
    )

    return package_dir


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
    provider: str,
    api_key: Optional[str],
    target_dir: Path,
) -> None:
    metadata = get_model_from_provider(
        model_id=model_id,
        provider=provider,
        api_key=api_key,
    )
    backend_type = BackendType(backend)

    _validate_metadata(
        metadata,
        model_architecture=model_architecture,
        task_type=task_type,
    )
    package = _select_package(
        metadata,
        backend=backend_type,
        package_id=package_id,
    )
    package_dir = _make_package_dir(
        target_dir,
        model_id=model_id,
        model_architecture=model_architecture,
        task_type=task_type,
        backend=backend_type,
        package_id=package.package_id,
    )

    click.echo(package.get_summary())
    click.echo(f"Downloading package to: {package_dir}")

    download_files_to_directory(
        target_dir=str(package_dir),
        files_specs=[
            (artifact.file_handle, artifact.download_url, artifact.md5_hash)
            for artifact in package.package_artefacts
        ],
        verify_hash_while_download=True,
        download_files_without_hash=False,
    )


if __name__ == "__main__":
    main()

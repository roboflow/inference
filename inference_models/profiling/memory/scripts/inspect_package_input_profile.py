"""Print registry input profile and package shape constraints for memory profiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from inference_models import BackendType, Quantization
from profiling.memory.backend_registry import resolve_registry_row
from profiling.memory.package_input_profile import describe_package_input_profile
from profiling.memory.package_resolve import resolve_package_directory


@click.command(
    help=(
        "Show input profile metadata and required profiling shapes for a model package."
    ),
)
@click.option(
    "--package-path",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    default=None,
    help="Local package directory (mutually exclusive with --model-id).",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Registered model id; resolves package under --packages-target-dir.",
)
@click.option(
    "--architecture",
    type=str,
    default=None,
    help="Registry architecture (required with --model-id).",
)
@click.option(
    "--task-type",
    type=str,
    default=None,
    help="Registry task type (required with --model-id).",
)
@click.option(
    "--backend",
    type=click.Choice(
        [
            BackendType.TORCH.value,
            BackendType.ONNX.value,
            BackendType.TRT.value,
        ],
    ),
    default=None,
    help="Profiling harness backend (required with --model-id for registry lookup).",
)
@click.option(
    "--quantization",
    type=click.Choice([q.value for q in Quantization]),
    default="fp32",
    show_default=True,
)
@click.option(
    "--package-id",
    type=str,
    default=None,
)
@click.option(
    "--packages-target-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    default=Path("/tmp/inference_model_packages"),
    show_default=True,
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
    "--batch-size",
    type=int,
    default=None,
    help="Optional: check whether these CLI values would pass profiling validation.",
)
@click.option(
    "--height",
    type=int,
    default=None,
)
@click.option(
    "--width",
    type=int,
    default=None,
)
def main(
    package_path: Optional[Path],
    model_id: Optional[str],
    architecture: Optional[str],
    task_type: Optional[str],
    backend: Optional[str],
    quantization: str,
    package_id: Optional[str],
    packages_target_dir: Path,
    provider: str,
    api_key: Optional[str],
    batch_size: Optional[int],
    height: Optional[int],
    width: Optional[int],
) -> None:
    """Emit JSON describing package input constraints and registry task profile."""
    if bool(package_path) == bool(model_id):
        raise click.UsageError("Provide exactly one of --package-path or --model-id.")

    module_name: Optional[str] = None
    class_name: Optional[str] = None
    harness_backend = backend

    if model_id:
        missing = []
        if not architecture:
            missing.append("--architecture")
        if not task_type:
            missing.append("--task-type")
        if not backend:
            missing.append("--backend")
        if missing:
            raise click.UsageError(
                "With --model-id, also pass: " + ", ".join(missing)
            )

        try:
            registry_row = resolve_registry_row(
                architecture=architecture,
                task_type=task_type,
                harness_backend=backend,
            )
        except ValueError as error:
            raise click.ClickException(str(error)) from error

        module_name = registry_row.module_name
        class_name = registry_row.class_name

        try:
            resolved_dir, _, _ = resolve_package_directory(
                model_id=model_id,
                model_architecture=architecture,
                task_type=task_type,
                backend=BackendType(backend),
                quantization=Quantization(quantization),
                package_id=package_id,
                target_dir=packages_target_dir,
                provider=provider,
                api_key=api_key,
                fetch_if_missing=False,
            )
        except ValueError as error:
            raise click.ClickException(str(error)) from error

        package_path = resolved_dir

    assert package_path is not None

    shape_args = {}
    if batch_size is not None or height is not None or width is not None:
        if batch_size is None or height is None or width is None:
            raise click.UsageError(
                "Pass all of --batch-size, --height, and --width to validate a request."
            )
        shape_args = {
            "batch_size": batch_size,
            "height": height,
            "width": width,
        }

    report = describe_package_input_profile(
        package_path,
        architecture=architecture,
        task_type=task_type,
        harness_backend=harness_backend,
        module_name=module_name,
        class_name=class_name,
        **shape_args,
    )

    click.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

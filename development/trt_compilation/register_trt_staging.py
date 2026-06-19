#!/usr/bin/env python3
"""Register a locally compiled TensorRT package to Roboflow staging or production."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import click

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPO_ROOT, REPO_ROOT / "inference_models"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _common import (  # noqa: E402
    CLASS_NAMES_FILE,
    DEFAULT_PROD_API_HOST,
    DEFAULT_STAGING_API_HOST,
    ENGINE_PLAN_FILE,
    INFERENCE_CONFIG_FILE,
    TRT_CONFIG_FILE,
    load_registration_manifest,
    validate_trt_package_dir,
)

from inference_cli.lib.enterprise.inference_compiler.adapters.models_service import (  # noqa: E402
    ModelsServiceClient,
)
from inference_cli.lib.enterprise.inference_compiler.core.compilation_handlers.utils import (  # noqa: E402
    register_model_package_artefacts,
)
from inference_cli.lib.enterprise.inference_compiler.core.entities import TRTConfig  # noqa: E402
from inference_cli.lib.enterprise.inference_compiler.errors import RequestError  # noqa: E402
from inference_cli.lib.enterprise.inference_compiler.utils.file_system import (  # noqa: E402
    calculate_local_file_md5,
)

ENVIRONMENT_API_HOSTS = {
    "staging": DEFAULT_STAGING_API_HOST,
    "production": DEFAULT_PROD_API_HOST,
}


class InternalModelsServiceClient(ModelsServiceClient):
    """Models service client authenticated with MODELS_SERVICE_INTERNAL_SECRET."""

    @classmethod
    def init_from_env(
        cls,
        *,
        api_host: str,
        service_secret: Optional[str] = None,
    ) -> "InternalModelsServiceClient":
        secret = service_secret or os.getenv("MODELS_SERVICE_INTERNAL_SECRET")
        if not secret:
            raise click.ClickException(
                "MODELS_SERVICE_INTERNAL_SECRET is required for model registration."
            )
        return cls(api_host=api_host, api_key=secret)


def _build_local_files_mapping(
    *,
    trt_package_dir: Path,
) -> Dict[str, Tuple[str, str]]:
    return {
        INFERENCE_CONFIG_FILE: (
            str(trt_package_dir / INFERENCE_CONFIG_FILE),
            calculate_local_file_md5(
                file_path=str(trt_package_dir / INFERENCE_CONFIG_FILE)
            ),
        ),
        CLASS_NAMES_FILE: (
            str(trt_package_dir / CLASS_NAMES_FILE),
            calculate_local_file_md5(file_path=str(trt_package_dir / CLASS_NAMES_FILE)),
        ),
        TRT_CONFIG_FILE: (
            str(trt_package_dir / TRT_CONFIG_FILE),
            calculate_local_file_md5(file_path=str(trt_package_dir / TRT_CONFIG_FILE)),
        ),
        ENGINE_PLAN_FILE: (
            str(trt_package_dir / ENGINE_PLAN_FILE),
            calculate_local_file_md5(file_path=str(trt_package_dir / ENGINE_PLAN_FILE)),
        ),
    }


@click.command()
@click.option(
    "--trt-package-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory produced by fetch_and_compile_trt.py (contains engine.plan, etc.).",
)
@click.option(
    "--environment",
    type=click.Choice(
        ["staging", "production"],
        case_sensitive=False,
    ),
    default="staging",
    show_default=True,
    help="Roboflow environment to register the package on.",
)
@click.option(
    "--api-host",
    "--staging-api-host",
    default=None,
    help="Override Roboflow API host. Defaults to the host for --environment.",
)
@click.option(
    "--service-secret",
    default=None,
    help="Override MODELS_SERVICE_INTERNAL_SECRET.",
)
@click.option(
    "--model-id",
    default=None,
    help="Override model id from registration_manifest.json.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate inputs and print the registration payload without uploading.",
)
def main(
    trt_package_dir: Path,
    environment: str,
    api_host: Optional[str],
    service_secret: Optional[str],
    model_id: Optional[str],
    dry_run: bool,
) -> None:
    """Upload and seal a compiled TRT package on Roboflow staging or production."""
    trt_package_dir = trt_package_dir.resolve()
    validate_trt_package_dir(trt_package_dir=trt_package_dir)
    registration_manifest = load_registration_manifest(trt_package_dir=trt_package_dir)

    resolved_environment = environment.lower()
    resolved_api_host = api_host or ENVIRONMENT_API_HOSTS[resolved_environment]

    resolved_model_id = model_id or registration_manifest["modelId"]
    package_manifest = registration_manifest["packageManifest"]
    file_handles = registration_manifest["fileHandles"]
    TRTConfig.model_validate(registration_manifest["trtConfig"])

    click.echo(f"Model id       : {resolved_model_id}")
    click.echo(f"Source model id: {registration_manifest.get('sourceModelId')}")
    click.echo(f"Precision      : {registration_manifest.get('precision')}")
    click.echo(f"Machine type   : {package_manifest.get('machineType')}")
    click.echo(f"Environment    : {resolved_environment}")
    click.echo(f"API host       : {resolved_api_host}")
    click.echo(f"Package files  : {', '.join(file_handles)}")

    if dry_run:
        click.echo("")
        click.echo("Dry run only. Registration payload:")
        click.echo(f"  modelId={resolved_model_id}")
        click.echo(f"  packageManifest.type={package_manifest.get('type')}")
        click.echo(f"  machineType={package_manifest.get('machineType')}")
        return

    models_service_client = InternalModelsServiceClient.init_from_env(
        api_host=resolved_api_host,
        service_secret=service_secret,
    )

    model_features = None
    architecture = registration_manifest.get("modelArchitecture")
    task_type = registration_manifest.get("taskType")
    model_variant = registration_manifest.get("modelVariant")
    if architecture or task_type or model_variant:
        model_features = {
            "modelArchitecture": architecture,
            "taskType": task_type,
            "modelVariant": model_variant,
        }

    click.echo("")
    click.echo(f"Registering model package on {resolved_environment} ...")
    try:
        registration_response = models_service_client.register_model_package(
            model_id=resolved_model_id,
            package_manifest=package_manifest,
            file_handles=file_handles,
            model_features=model_features,
        )
    except RequestError as error:
        if error.status_code == 409:
            raise click.ClickException(
                f"Model package already exists for this manifest on {resolved_environment}."
            ) from error
        raise click.ClickException(str(error)) from error

    local_files_mapping = _build_local_files_mapping(trt_package_dir=trt_package_dir)
    click.echo(
        "Uploading artefacts for package "
        f"{registration_response.model_package_id} ..."
    )
    register_model_package_artefacts(
        registration_response=registration_response,
        local_files_mapping=local_files_mapping,
        models_service_client=models_service_client,
    )

    click.echo("")
    click.echo("Registration complete.")
    click.echo(f"  model_id          : {registration_response.model_id}")
    click.echo(f"  model_package_id  : {registration_response.model_package_id}")
    click.echo("")
    click.echo(f"Verify on {resolved_environment}:")
    click.echo(f"  export ROBOFLOW_ENVIRONMENT={resolved_environment}")
    click.echo(f"  export ROBOFLOW_API_HOST={resolved_api_host}")
    click.echo(
        f'  python -c "from inference_models import AutoModel; '
        f"AutoModel.describe_model('{resolved_model_id}')\""
    )


if __name__ == "__main__":
    main()

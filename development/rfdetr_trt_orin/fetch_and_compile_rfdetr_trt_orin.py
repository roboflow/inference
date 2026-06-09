#!/usr/bin/env python3
"""Fetch RF-DETR ONNX from production and compile a Jetson Orin TRT package."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import click

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INFERENCE_MODELS_ROOT = REPO_ROOT / "inference_models"
for path in (SCRIPT_DIR, REPO_ROOT, INFERENCE_MODELS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _common import (  # noqa: E402
    CLASS_NAMES_FILE,
    DEFAULT_MODEL_ID,
    DEFAULT_PROD_API_HOST,
    ENGINE_PLAN_FILE,
    INFERENCE_CONFIG_FILE,
    REGISTRATION_MANIFEST_FILE,
    TRT_CONFIG_FILE,
    WEIGHTS_ONNX_FILE,
    build_registration_manifest,
    get_training_input_size,
    prepare_adjusted_inference_config,
    write_json,
)


def _configure_prod_api(*, prod_api_host: str) -> None:
    os.environ["ROBOFLOW_ENVIRONMENT"] = "prod"
    os.environ["ROBOFLOW_API_HOST"] = prod_api_host


def _fetch_onnx_package(
    *,
    model_id: str,
    output_dir: Path,
    prod_api_host: str,
    roboflow_api_key: Optional[str],
    dynamic_batch: bool,
) -> tuple[Path, dict]:
    _configure_prod_api(prod_api_host=prod_api_host)

    from inference_models.development.compilation.core import (
        download_model_packages,
        select_matching_model_packages,
    )
    from inference_models.weights_providers.core import get_model_from_provider

    model_metadata = get_model_from_provider(
        provider="roboflow",
        model_id=model_id,
        api_key=roboflow_api_key,
    )
    matching_packages = select_matching_model_packages(
        model_id=model_id,
        roboflow_api_key=roboflow_api_key,
        dynamic_dimensions_in_use=dynamic_batch,
    )
    if not matching_packages:
        raise click.ClickException(
            f"No matching ONNX packages found for {model_id} "
            f"(dynamic_batch={dynamic_batch})."
        )

    source_onnx_dir = output_dir / "source_onnx"
    source_onnx_dir.mkdir(parents=True, exist_ok=True)
    package_dirs = download_model_packages(
        matching_model_packages=matching_packages,
        target_path=str(source_onnx_dir),
    )
    if len(package_dirs) != 1:
        package_ids = [Path(path).name for path in package_dirs]
        raise click.ClickException(
            "Expected exactly one ONNX package to download, got "
            f"{len(package_dirs)}: {package_ids}"
        )

    package_dir = Path(package_dirs[0])
    for required_file in (WEIGHTS_ONNX_FILE, INFERENCE_CONFIG_FILE, CLASS_NAMES_FILE):
        if not (package_dir / required_file).is_file():
            raise click.ClickException(
                f"Downloaded ONNX package is missing required file: {required_file}"
            )

    metadata = {
        "model_id": model_metadata.model_id,
        "model_architecture": model_metadata.model_architecture,
        "task_type": model_metadata.task_type,
        "model_variant": model_metadata.model_variant,
        "source_package_id": package_dir.name,
    }
    return package_dir, metadata


def _compile_trt_package(
    *,
    source_onnx_dir: Path,
    trt_package_dir: Path,
    precision: Literal["fp32", "fp16"],
    workspace_size_gb: int,
    min_batch_size: Optional[int],
    opt_batch_size: Optional[int],
    max_batch_size: Optional[int],
    trt_forward_compatible: bool,
    trt_same_cc_compatible: bool,
) -> tuple[dict, dict]:
    import onnxruntime

    from inference_models.development.compilation.engine_builder import EngineBuilder
    from inference_models.runtime_introspection.core import x_ray_runtime_environment

    onnx_path = source_onnx_dir / WEIGHTS_ONNX_FILE
    inference_config_path = source_onnx_dir / INFERENCE_CONFIG_FILE
    class_names_path = source_onnx_dir / CLASS_NAMES_FILE

    runtime_xray = x_ray_runtime_environment()
    if runtime_xray.trt_version is None:
        raise click.ClickException(
            "TensorRT was not detected. Run this script on a Jetson Orin with JetPack."
        )

    session = onnxruntime.InferenceSession(str(onnx_path))
    training_size = get_training_input_size(inference_config_path=inference_config_path)

    dynamic_batch_sizes = None
    static_batch_size = None
    dynamic_dimensions_in_use = all(
        value is not None
        for value in (min_batch_size, opt_batch_size, max_batch_size)
    )
    if dynamic_dimensions_in_use:
        dynamic_batch_sizes = (min_batch_size, opt_batch_size, max_batch_size)
    else:
        static_batch_size = session.get_inputs()[0].shape[0]
        if isinstance(static_batch_size, str):
            raise click.ClickException(
                "ONNX input batch size is dynamic but dynamic batch args were not set."
            )

    trt_package_dir.mkdir(parents=True, exist_ok=True)
    engine_path = trt_package_dir / ENGINE_PLAN_FILE
    trt_config = {
        "static_batch_size": static_batch_size,
        "dynamic_batch_size_min": min_batch_size,
        "dynamic_batch_size_opt": opt_batch_size,
        "dynamic_batch_size_max": max_batch_size,
        "trt_version_compatible": trt_forward_compatible,
        "same_compute_compatibility": trt_same_cc_compatible,
        "precision": precision,
    }
    write_json(trt_package_dir / TRT_CONFIG_FILE, trt_config)
    prepare_adjusted_inference_config(
        inference_config_path=inference_config_path,
        target_path=trt_package_dir / INFERENCE_CONFIG_FILE,
    )
    shutil.copy2(class_names_path, trt_package_dir / CLASS_NAMES_FILE)
    write_json(
        trt_package_dir / "env-x-ray.json",
        {
            "gpu_available": runtime_xray.gpu_available,
            "gpu_devices": runtime_xray.gpu_devices,
            "gpu_devices_cc": [str(value) for value in runtime_xray.gpu_devices_cc],
            "driver_version": (
                str(runtime_xray.driver_version)
                if runtime_xray.driver_version
                else None
            ),
            "cuda_version": (
                str(runtime_xray.cuda_version) if runtime_xray.cuda_version else None
            ),
            "trt_version": (
                str(runtime_xray.trt_version) if runtime_xray.trt_version else None
            ),
            "jetson_type": runtime_xray.jetson_type,
            "l4t_version": (
                str(runtime_xray.l4t_version) if runtime_xray.l4t_version else None
            ),
            "os_version": runtime_xray.os_version,
        },
    )

    engine_builder = EngineBuilder(workspace=workspace_size_gb)
    engine_builder.create_network(onnx_path=str(onnx_path))
    engine_builder.create_engine(
        engine_path=str(engine_path),
        input_name=session.get_inputs()[0].name,
        precision=precision,
        input_size=training_size,
        dynamic_batch_sizes=dynamic_batch_sizes,
        trt_version_compatible=trt_forward_compatible,
        same_compute_compatibility=trt_same_cc_compatible,
    )

    if runtime_xray.l4t_version is not None:
        machine_type = "jetson"
        machine_specs = {
            "type": "jetson-machine-specs-v1",
            "l4tVersion": str(runtime_xray.l4t_version),
            "deviceName": runtime_xray.jetson_type or "unknown",
            "driverVersion": str(runtime_xray.driver_version),
        }
    else:
        machine_type = "gpu-server"
        machine_specs = {
            "type": "gpu-server-specs-v1",
            "driverVersion": str(runtime_xray.driver_version),
            "osVersion": runtime_xray.os_version,
        }

    package_manifest = {
        "type": "trt-model-package-v1",
        "backendType": "trt",
        "dynamicBatchSize": dynamic_batch_sizes is not None,
        "staticBatchSize": static_batch_size,
        "minBatchSize": min_batch_size,
        "optBatchSize": opt_batch_size,
        "maxBatchSize": max_batch_size,
        "quantization": precision,
        "cudaDeviceType": runtime_xray.gpu_devices[0],
        "cudaDeviceCC": str(runtime_xray.gpu_devices_cc[0]),
        "cudaVersion": str(runtime_xray.cuda_version),
        "trtVersion": str(runtime_xray.trt_version),
        "sameCCCompatible": trt_same_cc_compatible,
        "trtForwardCompatible": trt_forward_compatible,
        "trtLeanRuntimeExcluded": False,
        "machineType": machine_type,
        "machineSpecs": machine_specs,
    }
    return trt_config, package_manifest


def _verify_trt_package(*, trt_package_dir: Path) -> None:
    from inference_models import AutoModel

    model = AutoModel.from_local_package(
        package_path=str(trt_package_dir),
        backend="trt",
    )
    import numpy as np

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(image)


@click.command()
@click.option(
    "--model-id",
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Roboflow model id or alias to fetch from production.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./rfdetr-seg-nano-orin-trt-build"),
    show_default=True,
    help="Directory for downloaded ONNX and compiled TRT artefacts.",
)
@click.option(
    "--prod-api-host",
    default=DEFAULT_PROD_API_HOST,
    show_default=True,
    help="Production Roboflow API host used when fetching ONNX.",
)
@click.option(
    "--roboflow-api-key",
    default=None,
    help="Optional API key for private models. Defaults to ROBOFLOW_API_KEY.",
)
@click.option(
    "--precision",
    type=click.Choice(["fp16", "fp32"], case_sensitive=False),
    default="fp16",
    show_default=True,
)
@click.option(
    "--workspace-size-gb",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
)
@click.option(
    "--min-batch-size",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
)
@click.option(
    "--opt-batch-size",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
)
@click.option(
    "--max-batch-size",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
)
@click.option(
    "--static-batch/--dynamic-batch",
    default=False,
    show_default=True,
    help="Use static batch compilation instead of dynamic batch profiles.",
)
@click.option(
    "--trt-forward-compatible/--no-trt-forward-compatible",
    default=False,
    show_default=True,
)
@click.option(
    "--trt-same-cc-compatible/--no-trt-same-cc-compatible",
    default=False,
    show_default=True,
)
@click.option(
    "--skip-fetch",
    is_flag=True,
    help="Skip production ONNX download and reuse output-dir/source_onnx/*.",
)
@click.option(
    "--skip-compile",
    is_flag=True,
    help="Skip TRT compilation (fetch/metadata only).",
)
@click.option(
    "--verify/--no-verify",
    default=False,
    show_default=True,
    help="Run a local AutoModel smoke test after compilation.",
)
@click.option(
    "--staging-model-id",
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Model id written into registration_manifest.json for staging upload.",
)
def main(
    model_id: str,
    output_dir: Path,
    prod_api_host: str,
    roboflow_api_key: Optional[str],
    precision: Literal["fp16", "fp32"],
    workspace_size_gb: int,
    min_batch_size: int,
    opt_batch_size: int,
    max_batch_size: int,
    static_batch: bool,
    trt_forward_compatible: bool,
    trt_same_cc_compatible: bool,
    skip_fetch: bool,
    skip_compile: bool,
    verify: bool,
    staging_model_id: str,
) -> None:
    """Fetch prod ONNX and compile a registry-ready Jetson TRT package."""
    if skip_fetch and skip_compile:
        raise click.ClickException("At least one of fetch or compile must run.")

    if roboflow_api_key is None:
        roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

    output_dir = output_dir.resolve()
    source_onnx_root = output_dir / "source_onnx"
    trt_package_dir = output_dir / "trt_package"

    metadata: dict
    source_onnx_dir: Path

    if skip_fetch:
        package_dirs = sorted(path for path in source_onnx_root.iterdir() if path.is_dir())
        if len(package_dirs) != 1:
            raise click.ClickException(
                f"Expected one directory under {source_onnx_root}, found "
                f"{len(package_dirs)}."
            )
        source_onnx_dir = package_dirs[0]
        metadata = {
            "model_id": model_id,
            "model_architecture": "rfdetr",
            "task_type": "instance-segmentation",
            "model_variant": None,
            "source_package_id": source_onnx_dir.name,
        }
        click.echo(f"Reusing ONNX package at {source_onnx_dir}")
    else:
        click.echo(f"Fetching ONNX for {model_id} from {prod_api_host} ...")
        source_onnx_dir, metadata = _fetch_onnx_package(
            model_id=model_id,
            output_dir=output_dir,
            prod_api_host=prod_api_host,
            roboflow_api_key=roboflow_api_key,
            dynamic_batch=not static_batch,
        )
        click.echo(f"Downloaded ONNX package to {source_onnx_dir}")

    if skip_compile:
        click.echo("Skipping TRT compilation.")
        return

    click.echo(f"Compiling TRT package into {trt_package_dir} ...")
    batch_args: Tuple[Optional[int], Optional[int], Optional[int]]
    if static_batch:
        batch_args = (None, None, None)
    else:
        batch_args = (min_batch_size, opt_batch_size, max_batch_size)

    trt_config, package_manifest = _compile_trt_package(
        source_onnx_dir=source_onnx_dir,
        trt_package_dir=trt_package_dir,
        precision=precision,
        workspace_size_gb=workspace_size_gb,
        min_batch_size=batch_args[0],
        opt_batch_size=batch_args[1],
        max_batch_size=batch_args[2],
        trt_forward_compatible=trt_forward_compatible,
        trt_same_cc_compatible=trt_same_cc_compatible,
    )

    registration_manifest = build_registration_manifest(
        model_id=staging_model_id,
        source_model_id=metadata["model_id"],
        model_architecture=metadata["model_architecture"],
        task_type=metadata["task_type"],
        model_variant=metadata["model_variant"],
        package_manifest=package_manifest,
        trt_config=trt_config,
        precision=precision,
    )
    write_json(trt_package_dir / REGISTRATION_MANIFEST_FILE, registration_manifest)

    if verify:
        click.echo("Running local TRT smoke test ...")
        _verify_trt_package(trt_package_dir=trt_package_dir)
        click.echo("Local TRT smoke test passed.")

    click.echo("")
    click.echo("Build complete.")
    click.echo(f"  ONNX source : {source_onnx_dir}")
    click.echo(f"  TRT package : {trt_package_dir}")
    click.echo(
        "Next: run register_rfdetr_trt_orin_staging.py against trt_package/ on staging."
    )


if __name__ == "__main__":
    main()

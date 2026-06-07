"""Resolve Roboflow model package paths for profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from inference_models import BackendType, Quantization
from inference_models.developer_tools import (
    ModelMetadata,
    ModelPackageMetadata,
    download_files_to_directory,
    get_model_from_provider,
)
from inference_models.utils.file_system import read_json

def make_package_dir(
    target_dir: Path,
    *,
    model_id: str,
    package_id: str,
) -> Path:
    safe_model_id = model_id.replace("/", "-")
    package_dir = (
        target_dir
        / safe_model_id
        / package_id
    )

    return package_dir


def select_package(
    metadata: ModelMetadata,
    *,
    backend: BackendType,
    package_id: Optional[str],
    quantization: Optional[Quantization],
) -> ModelPackageMetadata:
    candidates = [
        package
        for package in metadata.model_packages
        if package.backend == backend
        and (package_id is None or package.package_id == package_id)
        and (quantization is None or package.quantization == quantization)
    ]

    if not candidates:
        available = ", ".join(
            f"{package.package_id}:{package.backend.value}:{package.quantization.value}"
            for package in metadata.model_packages
        )
        raise ValueError(
            f"No package found for backend={backend.value!r}"
            + (f" and package_id={package_id!r}" if package_id else "")
            + (f" and quantization={quantization.value!r}" if quantization else "")
            + f". Available packages: {available}"
        )

    selected_package = candidates[0]

    return selected_package


def onnx_opset_from_package(package: ModelPackageMetadata) -> Optional[int]:
    """Return ONNX opset from registry ``ModelPackageMetadata`` when present."""
    if package.onnx_package_details is None:
        return None

    return package.onnx_package_details.opset


def extract_onnx_opset_from_package_dir(package_dir: Path) -> Optional[int]:
    """Read ONNX opset from ``model_config.json`` in a local package directory."""
    config_path = package_dir / "model_config.json"
    if not config_path.is_file():
        return None

    raw = read_json(str(config_path))
    if not isinstance(raw, dict):
        return None

    opset = raw.get("opset")
    if opset is None:
        return None

    return int(opset)


def resolve_package_directory(
    *,
    model_id: str,
    model_architecture: str,
    task_type: str,
    backend: BackendType,
    quantization: Optional[Quantization],
    package_id: Optional[str],
    target_dir: Path,
    provider: str = "roboflow",
    api_key: Optional[str] = None,
    force_download: bool = False,
) -> tuple[Path, ModelPackageMetadata, Optional[str]]:
    """Return local package directory, package metadata, and model variant.

    Downloads package artifacts when the local directory is missing, or when
    ``force_download`` is True (re-download and overwrite an existing directory).

    ``package.package_id`` is the unique Roboflow registry identifier (API field ``packageId``).
    """
    metadata = get_model_from_provider(
        model_id=model_id,
        provider=provider,
        api_key=api_key,
    )

    if metadata.model_architecture != model_architecture:
        raise ValueError(
            f"Expected architecture {model_architecture!r}, "
            f"got {metadata.model_architecture!r}."
        )

    if metadata.task_type != task_type:
        raise ValueError(
            f"Expected task type {task_type!r}, got {metadata.task_type!r}."
        )

    package = select_package(
        metadata,
        backend=backend,
        package_id=package_id,
        quantization=quantization,
    )
    package_dir = make_package_dir(
        target_dir,
        model_id=model_id,
        package_id=package.package_id,
    )

    if force_download or not package_dir.is_dir():
        package_dir.mkdir(parents=True, exist_ok=True)
        download_files_to_directory(
            target_dir=str(package_dir),
            files_specs=[
                (artifact.file_handle, artifact.download_url, artifact.md5_hash)
                for artifact in package.package_artefacts
            ],
            verify_hash_while_download=True,
            download_files_without_hash=False,
        )

    return package_dir, package, metadata.model_variant


def extract_package_features(package_dir: Path) -> Dict[str, Any]:
    """Read package-local features from inference_config.json when present."""
    config_path = package_dir / "inference_config.json"
    if not config_path.is_file():
        return {}

    raw = read_json(str(config_path))
    if not isinstance(raw, dict):
        return {}

    features: Dict[str, Any] = {}
    forward_pass = raw.get("forward_pass") or {}
    if isinstance(forward_pass, dict):
        static_batch = forward_pass.get("static_batch_size")
        if static_batch is not None:
            features["static_batch_size"] = static_batch

        max_dynamic_batch = forward_pass.get("max_dynamic_batch_size")
        if max_dynamic_batch is not None:
            features["max_dynamic_batch_size"] = max_dynamic_batch

    network_input = raw.get("network_input") or {}
    if isinstance(network_input, dict):
        training_size = network_input.get("training_input_size")
        if isinstance(training_size, dict):
            features["training_input_size"] = training_size

        dynamic_spatial = network_input.get("dynamic_spatial_size_supported")
        if dynamic_spatial is not None:
            features["dynamic_spatial_size_supported"] = dynamic_spatial

    post_processing = raw.get("post_processing") or {}
    if isinstance(post_processing, dict):
        if post_processing.get("fused") is not None:
            features["nms_fused"] = bool(post_processing.get("fused"))

        nms_parameters = post_processing.get("nms_parameters")
        if isinstance(nms_parameters, dict):
            max_detections = nms_parameters.get("max_detections")
            if max_detections is not None:
                features["max_detections"] = max_detections

    return features


def extract_trt_package_features(package_dir: Path) -> Dict[str, Any]:
    """Read TensorRT batch profile from trt_config.json when present."""
    config_path = package_dir / "trt_config.json"
    if not config_path.is_file():
        return {}

    raw = read_json(str(config_path))
    if not isinstance(raw, dict):
        return {}

    features: Dict[str, Any] = {}
    for key in (
        "static_batch_size",
        "dynamic_batch_size_min",
        "dynamic_batch_size_opt",
        "dynamic_batch_size_max",
    ):
        value = raw.get(key)
        if value is not None:
            features[key] = value

    return features


def extract_trt_package_runtime_metadata(
    package_dir: Path,
) -> tuple[Optional[int], Optional[Dict[str, Any]], Optional[int]]:
    """Read TRT runtime artifacts from a local package directory.

    Returns:
        ``(engine_size_bytes, optimization_profile, max_workspace_setting_bytes)``
        where ``optimization_profile`` is the parsed ``trt_config.json`` when present.
    """
    if not package_dir.is_dir():
        return None, None, None

    engine_path = package_dir / "engine.plan"
    engine_size_bytes = (
        int(engine_path.stat().st_size) if engine_path.is_file() else None
    )

    optimization_profile: Optional[Dict[str, Any]] = None
    trt_config_path = package_dir / "trt_config.json"
    if trt_config_path.is_file():
        loaded_config = read_json(str(trt_config_path))
        if isinstance(loaded_config, dict):
            optimization_profile = loaded_config

    max_workspace_setting: Optional[int] = None
    build_config_path = package_dir / "build_config.json"
    if build_config_path.is_file():
        loaded_build_config = read_json(str(build_config_path))
        if isinstance(loaded_build_config, dict):
            workspace_gb = loaded_build_config.get("workspace_size_gb")
            if workspace_gb is not None:
                max_workspace_setting = int(workspace_gb) * (1024**3)

    return engine_size_bytes, optimization_profile, max_workspace_setting

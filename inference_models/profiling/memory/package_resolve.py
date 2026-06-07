"""Resolve Roboflow model package paths for profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from inference_models import BackendType, Quantization
from inference_models.developer_tools import (
    ModelMetadata,
    ModelPackageMetadata,
    download_files_to_directory,
    get_model_from_provider,
)
from inference_models.utils.file_system import read_json

TORCH_HARNESS_PACKAGE_BACKENDS = (
    BackendType.TORCH,
    BackendType.TORCH_SCRIPT,
    BackendType.HF,
)

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


def package_backends_for_harness(harness_backend: str) -> Sequence[BackendType]:
    """Return registry package backends that map to a profiling harness backend."""
    if harness_backend == BackendType.TORCH.value:
        return TORCH_HARNESS_PACKAGE_BACKENDS
    if harness_backend == BackendType.ONNX.value:
        return (BackendType.ONNX,)
    if harness_backend == BackendType.TRT.value:
        return (BackendType.TRT,)

    raise ValueError(
        f"Unsupported harness backend {harness_backend!r}; "
        f"expected one of: {BackendType.TORCH.value}, {BackendType.ONNX.value}, "
        f"{BackendType.TRT.value}"
    )


def harness_backend_for_package_backend(package_backend: BackendType) -> str:
    """Map a registry package backend to the profiling harness backend."""
    if package_backend in TORCH_HARNESS_PACKAGE_BACKENDS:
        return BackendType.TORCH.value
    if package_backend == BackendType.ONNX:
        return BackendType.ONNX.value
    if package_backend == BackendType.TRT:
        return BackendType.TRT.value

    raise ValueError(
        f"Package backend {package_backend.value!r} is not supported by the "
        "profiling harness."
    )


def format_package_choices(packages: Sequence[ModelPackageMetadata]) -> str:
    return ", ".join(
        f"{package.package_id}:{package.backend.value}:"
        f"{package.quantization.value if package.quantization else 'unknown'}"
        for package in packages
    )


def registry_identity_provider_model_id(
    *,
    architecture: str,
    model_variant: str,
) -> str:
    """Canonical Roboflow provider id for a registry template model."""
    return f"{architecture}/{model_variant}"


def fetch_model_metadata(
    *,
    model_id: str,
    provider: str = "roboflow",
    api_key: Optional[str] = None,
) -> ModelMetadata:
    return get_model_from_provider(
        model_id=model_id,
        provider=provider,
        api_key=api_key,
    )


def filter_packages(
    metadata: ModelMetadata,
    *,
    package_id: Optional[str] = None,
    package_backends: Optional[Sequence[BackendType]] = None,
    quantization: Optional[Quantization] = None,
) -> List[ModelPackageMetadata]:
    candidates: List[ModelPackageMetadata] = []

    for package in metadata.model_packages:
        if package_id is not None and package.package_id != package_id:
            continue
        if (
            package_backends is not None
            and package.backend not in package_backends
        ):
            continue
        if quantization is not None and package.quantization != quantization:
            continue

        candidates.append(package)

    return candidates


def require_single_package(
    candidates: Sequence[ModelPackageMetadata],
    *,
    context: str,
    available: Sequence[ModelPackageMetadata],
) -> ModelPackageMetadata:
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            f"No package found for {context}. "
            f"Available packages: {format_package_choices(available)}"
        )

    raise ValueError(
        f"Multiple packages match {context}: "
        f"{format_package_choices(candidates)}. "
        "Provide a more specific selection (for example --package-id or path 3 "
        "with --architecture, --task-type, and --model-variant)."
    )


def download_package_directory(
    *,
    model_id: str,
    package: ModelPackageMetadata,
    target_dir: Path,
    force_download: bool = False,
) -> Path:
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

    return package_dir


def select_package(
    metadata: ModelMetadata,
    *,
    backend: BackendType,
    package_id: Optional[str],
    quantization: Optional[Quantization],
) -> ModelPackageMetadata:
    harness_backend = harness_backend_for_package_backend(backend)
    package_backends = package_backends_for_harness(harness_backend)
    if backend not in package_backends:
        package_backends = (backend,)

    candidates = filter_packages(
        metadata,
        package_id=package_id,
        package_backends=package_backends,
        quantization=quantization,
    )

    return require_single_package(
        candidates,
        context=(
            f"backend={backend.value!r}"
            + (f" and package_id={package_id!r}" if package_id else "")
            + (
                f" and quantization={quantization.value!r}"
                if quantization
                else ""
            )
        ),
        available=metadata.model_packages,
    )


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
    metadata = fetch_model_metadata(
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
    package_dir = download_package_directory(
        model_id=model_id,
        package=package,
        target_dir=target_dir,
        force_download=force_download,
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

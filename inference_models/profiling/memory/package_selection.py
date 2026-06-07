"""Resolve profiling package identity through three explicit CLI input paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from inference_models import Quantization
from inference_models.developer_tools import ModelMetadata, ModelPackageMetadata
from profiling.memory.package_resolve import (
    download_package_directory,
    fetch_model_metadata,
    filter_packages,
    harness_backend_for_package_backend,
    package_backends_for_harness,
    registry_identity_provider_model_id,
    require_single_package,
)


class PackageSelectionMode(str, Enum):
    """How the profiling CLI identifies a model package."""

    BY_PACKAGE_ID = "by_package_id"
    BY_BACKEND_QUANTIZATION = "by_backend_quantization"
    BY_REGISTRY_IDENTITY = "by_registry_identity"


@dataclass(frozen=True)
class ProfilingPackageSelection:
    """Fully resolved model and package identity for a profiling run."""

    model_id: str
    package: ModelPackageMetadata
    harness_backend: str
    architecture: str
    task_type: str
    model_variant: Optional[str]
    quantization: str


def classify_package_selection_mode(
    *,
    model_id: Optional[str],
    package_id: Optional[str],
    backend: Optional[str],
    architecture: Optional[str],
    task_type: Optional[str],
    model_variant: Optional[str],
    quantization: Optional[str],
) -> PackageSelectionMode:
    """Classify which package-resolution path the CLI arguments request."""
    has_registry_identity = (
        architecture is not None
        and task_type is not None
        and model_variant is not None
    )

    if package_id is not None:
        if not model_id:
            raise ValueError("Package selection path 1 requires --model-id.")

        conflicting = [
            name
            for name, value in (
                ("--backend", backend),
                ("--quantization", quantization),
                ("--architecture", architecture),
                ("--task-type", task_type),
                ("--model-variant", model_variant),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError(
                "Package selection path 1 accepts only --model-id and --package-id; "
                f"remove: {', '.join(conflicting)}."
            )

        return PackageSelectionMode.BY_PACKAGE_ID

    if backend and quantization and has_registry_identity:
        if model_id is not None:
            raise ValueError(
                "Package selection path 3 accepts --backend, --quantization, "
                "--architecture, --task-type, and --model-variant only; "
                "do not pass --model-id."
            )

        return PackageSelectionMode.BY_REGISTRY_IDENTITY

    if backend and quantization:
        if not model_id:
            raise ValueError("Package selection path 2 requires --model-id.")

        if architecture is not None or task_type is not None or model_variant is not None:
            extra = [
                name
                for name, value in (
                    ("--architecture", architecture),
                    ("--task-type", task_type),
                    ("--model-variant", model_variant),
                )
                if value is not None
            ]
            raise ValueError(
                "Package selection path 2 accepts --model-id, --backend, and "
                f"--quantization only; remove: {', '.join(extra)}."
            )

        return PackageSelectionMode.BY_BACKEND_QUANTIZATION

    raise ValueError(
        "Specify exactly one package selection path:\n"
        "  1) --model-id and --package-id\n"
        "  2) --model-id, --backend, and --quantization\n"
        "  3) --backend, --quantization, --architecture, --task-type, and --model-variant"
    )


def _selection_from_package(
    *,
    metadata: ModelMetadata,
    package: ModelPackageMetadata,
) -> ProfilingPackageSelection:
    if not metadata.task_type:
        raise ValueError(
            f"Model {metadata.model_id!r} has no task_type in provider metadata."
        )

    quantization = package.quantization
    if quantization is None:
        raise ValueError(
            f"Package {package.package_id!r} has no quantization in provider metadata."
        )

    return ProfilingPackageSelection(
        model_id=metadata.model_id,
        package=package,
        harness_backend=harness_backend_for_package_backend(package.backend),
        architecture=metadata.model_architecture,
        task_type=metadata.task_type,
        model_variant=metadata.model_variant,
        quantization=quantization.value,
    )


def resolve_profiling_package_by_id(
    *,
    model_id: str,
    package_id: str,
    provider: str = "roboflow",
    api_key: Optional[str] = None,
) -> ProfilingPackageSelection:
    """Path 1: resolve from ``model_id`` and an explicit ``package_id``."""
    metadata = fetch_model_metadata(
        model_id=model_id,
        provider=provider,
        api_key=api_key,
    )
    candidates = filter_packages(
        metadata,
        package_id=package_id,
    )
    package = require_single_package(
        candidates,
        context=f"package_id={package_id!r} on model {model_id!r}",
        available=metadata.model_packages,
    )

    return _selection_from_package(metadata=metadata, package=package)


def resolve_profiling_package_by_backend_quantization(
    *,
    model_id: str,
    harness_backend: str,
    quantization: str,
    provider: str = "roboflow",
    api_key: Optional[str] = None,
) -> ProfilingPackageSelection:
    """Path 2: resolve from ``model_id``, harness ``backend``, and ``quantization``."""
    metadata = fetch_model_metadata(
        model_id=model_id,
        provider=provider,
        api_key=api_key,
    )
    quantization_type = Quantization(quantization)
    package_backends = package_backends_for_harness(harness_backend)
    candidates = filter_packages(
        metadata,
        package_backends=package_backends,
        quantization=quantization_type,
    )
    package = require_single_package(
        candidates,
        context=(
            f"backend={harness_backend!r} and quantization={quantization!r} "
            f"on model {model_id!r}"
        ),
        available=metadata.model_packages,
    )

    return _selection_from_package(metadata=metadata, package=package)


def _validate_registry_metadata(
    *,
    metadata: ModelMetadata,
    architecture: str,
    task_type: str,
    model_variant: str,
) -> None:
    if metadata.model_architecture != architecture:
        raise ValueError(
            f"Expected architecture {architecture!r}, "
            f"got {metadata.model_architecture!r} from model metadata."
        )

    if metadata.task_type != task_type:
        raise ValueError(
            f"Expected task type {task_type!r}, "
            f"got {metadata.task_type!r} from model metadata."
        )

    if metadata.model_variant != model_variant:
        raise ValueError(
            f"Expected model variant {model_variant!r}, "
            f"got {metadata.model_variant!r} from model metadata."
        )


def resolve_profiling_package_by_registry_identity(
    *,
    harness_backend: str,
    quantization: str,
    architecture: str,
    task_type: str,
    model_variant: str,
    provider: str = "roboflow",
    api_key: Optional[str] = None,
) -> ProfilingPackageSelection:
    """Path 3: resolve from registry identity; ``model_id`` comes from provider metadata."""
    provider_model_id = registry_identity_provider_model_id(
        architecture=architecture,
        model_variant=model_variant,
    )
    metadata = fetch_model_metadata(
        model_id=provider_model_id,
        provider=provider,
        api_key=api_key,
    )
    _validate_registry_metadata(
        metadata=metadata,
        architecture=architecture,
        task_type=task_type,
        model_variant=model_variant,
    )

    quantization_type = Quantization(quantization)
    package_backends = package_backends_for_harness(harness_backend)
    candidates = filter_packages(
        metadata,
        package_backends=package_backends,
        quantization=quantization_type,
    )
    package = require_single_package(
        candidates,
        context=(
            f"backend={harness_backend!r}, quantization={quantization!r}, "
            f"architecture={architecture!r}, task_type={task_type!r}, "
            f"model_variant={model_variant!r}"
        ),
        available=metadata.model_packages,
    )

    return ProfilingPackageSelection(
        model_id=metadata.model_id,
        package=package,
        harness_backend=harness_backend,
        architecture=architecture,
        task_type=task_type,
        model_variant=model_variant,
        quantization=quantization,
    )


def resolve_profiling_package_selection(
    *,
    model_id: Optional[str],
    package_id: Optional[str],
    backend: Optional[str],
    architecture: Optional[str],
    task_type: Optional[str],
    model_variant: Optional[str],
    quantization: Optional[str],
    provider: str = "roboflow",
    api_key: Optional[str] = None,
) -> ProfilingPackageSelection:
    """Classify CLI arguments and resolve the profiling package identity."""
    mode = classify_package_selection_mode(
        model_id=model_id,
        package_id=package_id,
        backend=backend,
        architecture=architecture,
        task_type=task_type,
        model_variant=model_variant,
        quantization=quantization,
    )

    if mode == PackageSelectionMode.BY_PACKAGE_ID:
        assert package_id is not None and model_id is not None

        return resolve_profiling_package_by_id(
            model_id=model_id,
            package_id=package_id,
            provider=provider,
            api_key=api_key,
        )

    assert backend is not None and quantization is not None

    if mode == PackageSelectionMode.BY_BACKEND_QUANTIZATION:
        assert model_id is not None

        return resolve_profiling_package_by_backend_quantization(
            model_id=model_id,
            harness_backend=backend,
            quantization=quantization,
            provider=provider,
            api_key=api_key,
        )

    assert architecture is not None and task_type is not None and model_variant is not None

    return resolve_profiling_package_by_registry_identity(
        harness_backend=backend,
        quantization=quantization,
        architecture=architecture,
        task_type=task_type,
        model_variant=model_variant,
        provider=provider,
        api_key=api_key,
    )


def resolve_profiling_package_directory(
    selection: ProfilingPackageSelection,
    *,
    packages_target_dir: Path,
    force_download: bool,
    provider: str = "roboflow",
    api_key: Optional[str] = None,
) -> Path:
    """Download or reuse the local package directory for a resolved selection."""
    return download_package_directory(
        model_id=selection.model_id,
        package=selection.package,
        target_dir=packages_target_dir,
        force_download=force_download,
    )

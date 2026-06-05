"""Resolve and validate profiling input shapes from packages and registry profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)
from profiling.memory.package_resolve import (
    extract_package_features,
    extract_trt_package_features,
)
from profiling.memory.registry_profiles import resolve_registry_input_context

AxisResolution = Literal["static", "dynamic", "unconstrained"]


class InputProfileMismatchError(ValueError):
    """Raised when CLI-requested profiling shapes disagree with the package contract."""

    def __init__(self, mismatches: List[str]) -> None:
        self.mismatches = list(mismatches)
        detail = "; ".join(mismatches)
        super().__init__(
            "Profiling input shape mismatch: "
            f"{detail}. "
            "Use `uv run python profiling/memory/scripts/inspect_package_input_profile.py` "
            "to see required values for this package, or pass matching "
            "--batch-size, --height, and --width."
        )


@dataclass(frozen=True)
class ProfilingAxisSpec:
    """One axis (batch, height, or width) for memory profiling."""

    resolution: AxisResolution
    required_value: Optional[int] = None
    source: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class PackageProfilingShapeSpec:
    """Package constraints for synthetic ``images`` profiling shapes."""

    batch: ProfilingAxisSpec
    height: ProfilingAxisSpec
    width: ProfilingAxisSpec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch": asdict(self.batch),
            "height": asdict(self.height),
            "width": asdict(self.width),
        }


def _axis_from_static(
    *,
    required_value: Optional[int],
    source: str,
) -> ProfilingAxisSpec:
    if required_value is None:
        return ProfilingAxisSpec(
            resolution="unconstrained",
            notes="No fixed value in package artifacts.",
        )

    return ProfilingAxisSpec(
        resolution="static",
        required_value=int(required_value),
        source=source,
    )


def shape_spec_from_inference_config(config: InferenceConfig) -> PackageProfilingShapeSpec:
    """Build shape spec from a parsed ``InferenceConfig``."""
    forward = config.forward_pass
    network = config.network_input

    if forward.static_batch_size is not None:
        batch_axis = ProfilingAxisSpec(
            resolution="static",
            required_value=int(forward.static_batch_size),
            source="inference_config.json#forward_pass.static_batch_size",
        )
    elif forward.max_dynamic_batch_size is not None:
        batch_axis = ProfilingAxisSpec(
            resolution="dynamic",
            required_value=int(forward.max_dynamic_batch_size),
            source="inference_config.json#forward_pass.max_dynamic_batch_size",
            notes=(
                "Dynamic batch; profile at max_dynamic_batch_size for worst-case admission."
            ),
        )
    else:
        batch_axis = ProfilingAxisSpec(resolution="unconstrained")

    training = network.training_input_size
    if network.dynamic_spatial_size_supported:
        height_axis = ProfilingAxisSpec(
            resolution="dynamic",
            required_value=int(training.height),
            source="inference_config.json#network_input.training_input_size.height",
            notes=(
                "Dynamic spatial size supported; training_input_size is a reference, "
                "not a hard profiling constraint unless you choose it explicitly."
            ),
        )
        width_axis = ProfilingAxisSpec(
            resolution="dynamic",
            required_value=int(training.width),
            source="inference_config.json#network_input.training_input_size.width",
            notes=(
                "Dynamic spatial size supported; training_input_size is a reference, "
                "not a hard profiling constraint unless you choose it explicitly."
            ),
        )
    else:
        height_axis = _axis_from_static(
            required_value=int(training.height),
            source="inference_config.json#network_input.training_input_size.height",
        )
        width_axis = _axis_from_static(
            required_value=int(training.width),
            source="inference_config.json#network_input.training_input_size.width",
        )

    return PackageProfilingShapeSpec(
        batch=batch_axis,
        height=height_axis,
        width=width_axis,
    )


def _apply_trt_batch_override(
    spec: PackageProfilingShapeSpec,
    *,
    trt_features: Dict[str, Any],
) -> PackageProfilingShapeSpec:
    static_batch = trt_features.get("static_batch_size")
    if static_batch is None:
        return spec

    batch_axis = ProfilingAxisSpec(
        resolution="static",
        required_value=int(static_batch),
        source="trt_config.json#static_batch_size",
    )

    return PackageProfilingShapeSpec(
        batch=batch_axis,
        height=spec.height,
        width=spec.width,
    )


def shape_spec_from_package_dir(package_dir: Path) -> Optional[PackageProfilingShapeSpec]:
    """Read profiling shape constraints from package artifacts on disk."""
    config_path = package_dir / "inference_config.json"
    if not config_path.is_file():
        trt_features = extract_trt_package_features(package_dir)
        if not trt_features:
            return None

        batch_axis = ProfilingAxisSpec(resolution="unconstrained")
        if trt_features.get("static_batch_size") is not None:
            batch_axis = ProfilingAxisSpec(
                resolution="static",
                required_value=int(trt_features["static_batch_size"]),
                source="trt_config.json#static_batch_size",
            )
        elif trt_features.get("dynamic_batch_size_max") is not None:
            batch_axis = ProfilingAxisSpec(
                resolution="dynamic",
                required_value=int(trt_features["dynamic_batch_size_max"]),
                source="trt_config.json#dynamic_batch_size_max",
                notes="Profile at dynamic_batch_size_max for worst-case admission.",
            )

        return PackageProfilingShapeSpec(
            batch=batch_axis,
            height=ProfilingAxisSpec(resolution="unconstrained"),
            width=ProfilingAxisSpec(resolution="unconstrained"),
        )

    config = parse_inference_config(
        str(config_path),
        allowed_resize_modes=set(ResizeMode),
    )
    spec = shape_spec_from_inference_config(config)
    trt_features = extract_trt_package_features(package_dir)

    return _apply_trt_batch_override(spec, trt_features=trt_features)


def shape_spec_from_model(model: Any) -> Optional[PackageProfilingShapeSpec]:
    """Build shape spec from ``model._inference_config`` when attached after load."""
    inference_config = getattr(model, "_inference_config", None)
    if inference_config is None:
        return None

    if not isinstance(inference_config, InferenceConfig):
        return None

    return shape_spec_from_inference_config(inference_config)


def _collect_shape_mismatches(
    *,
    batch_size: int,
    height: int,
    width: int,
    spec: PackageProfilingShapeSpec,
) -> List[str]:
    mismatches: List[str] = []

    if (
        spec.batch.resolution == "static"
        and spec.batch.required_value is not None
        and batch_size != spec.batch.required_value
    ):
        mismatches.append(
            f"batch: requested {batch_size}, package requires "
            f"{spec.batch.required_value} ({spec.batch.source})"
        )

    if (
        spec.height.resolution == "static"
        and spec.height.required_value is not None
        and height != spec.height.required_value
    ):
        mismatches.append(
            f"height: requested {height}, package requires "
            f"{spec.height.required_value} ({spec.height.source})"
        )

    if (
        spec.width.resolution == "static"
        and spec.width.required_value is not None
        and width != spec.width.required_value
    ):
        mismatches.append(
            f"width: requested {width}, package requires "
            f"{spec.width.required_value} ({spec.width.source})"
        )

    return mismatches


def validate_profiling_image_shapes(
    *,
    batch_size: int,
    height: int,
    width: int,
    spec: Optional[PackageProfilingShapeSpec],
) -> Tuple[int, int, int]:
    """Ensure profiling shapes match static package constraints.

    Returns:
        The same ``(batch_size, height, width)`` when valid.

    Raises:
        InputProfileMismatchError: When a static axis disagrees with the request.
    """
    if spec is None:
        return batch_size, height, width

    mismatches = _collect_shape_mismatches(
        batch_size=batch_size,
        height=height,
        width=width,
        spec=spec,
    )
    if mismatches:
        raise InputProfileMismatchError(mismatches)

    return batch_size, height, width


def validate_profiling_image_shapes_for_package_dir(
    package_dir: Path,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> Tuple[int, int, int]:
    """Validate shapes against on-disk package artifacts."""
    spec = shape_spec_from_package_dir(package_dir)

    return validate_profiling_image_shapes(
        batch_size=batch_size,
        height=height,
        width=width,
        spec=spec,
    )


def validate_profiling_image_shapes_for_model(
    model: Any,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> Tuple[int, int, int]:
    """Validate shapes against the loaded model package config."""
    spec = shape_spec_from_model(model)

    return validate_profiling_image_shapes(
        batch_size=batch_size,
        height=height,
        width=width,
        spec=spec,
    )


def recommended_profiling_cli_flags(spec: Optional[PackageProfilingShapeSpec]) -> Dict[str, Any]:
    """Suggest ``--batch-size`` / ``--height`` / ``--width`` for static packages."""
    if spec is None:
        return {
            "batch_size": None,
            "height": None,
            "width": None,
            "notes": ["No package shape constraints found in artifacts."],
        }

    notes: List[str] = []
    batch_size = (
        spec.batch.required_value if spec.batch.resolution == "static" else None
    )
    height = (
        spec.height.required_value if spec.height.resolution == "static" else None
    )
    width = spec.width.required_value if spec.width.resolution == "static" else None

    if spec.batch.resolution == "dynamic":
        notes.append(
            f"Use --batch-size {spec.batch.required_value} for worst-case dynamic batch."
        )
    if spec.height.resolution == "dynamic" or spec.width.resolution == "dynamic":
        notes.append(
            "Spatial axes are dynamic; choose worst-case height/width explicitly."
        )

    return {
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "notes": notes,
    }


def describe_package_input_profile(
    package_dir: Path,
    *,
    architecture: Optional[str] = None,
    task_type: Optional[str] = None,
    harness_backend: Optional[str] = None,
    module_name: Optional[str] = None,
    class_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Dict[str, Any]:
    """Return registry input profile context and package shape constraints."""
    package_path = Path(package_dir)
    shape_spec = shape_spec_from_package_dir(package_path)
    package_features = extract_package_features(package_path)
    trt_features = extract_trt_package_features(package_path)

    registry_context = resolve_registry_input_context(
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        backend=harness_backend,
    )

    report: Dict[str, Any] = {
        "package_path": str(package_path.resolve()),
        "package_features": package_features,
        "trt_package_features": trt_features,
        "profiling_shape_spec": shape_spec.to_dict() if shape_spec else None,
        "recommended_cli": recommended_profiling_cli_flags(shape_spec),
        "registry_input_context": {
            key: value
            for key, value in registry_context.items()
            if key != "task_profile_spec"
        },
    }

    task_profile_spec = registry_context.get("task_profile_spec")
    if isinstance(task_profile_spec, dict):
        report["task_inference_profile_spec"] = task_profile_spec

    if batch_size is not None and height is not None and width is not None:
        try:
            validate_profiling_image_shapes(
                batch_size=batch_size,
                height=height,
                width=width,
                spec=shape_spec,
            )
            report["profiling_shapes_valid"] = True
        except InputProfileMismatchError as error:
            report["profiling_shapes_valid"] = False
            report["profiling_shape_mismatches"] = error.mismatches

    return report

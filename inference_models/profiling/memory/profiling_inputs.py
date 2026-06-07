"""Build profiling harness inputs from registry task inference profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from inference_models.utils.file_system import read_json
from profiling.memory.registry_profiles import resolve_registry_input_context

DEFAULT_VLM_PROMPT = (
    "Describe this image in detail, listing every object, person, and activity "
    "visible in the scene including colors, positions, and relationships."
)
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_OPEN_VOCAB_CLASSES = ["object", "person", "vehicle"]
DEFAULT_MOONDREAM_CLASSES = ["object", "person", "vehicle"]
DEFAULT_FLORENCE2_PROMPT = "<CAPTION>"
DEFAULT_SAM3_PROMPTS = [{"text": "object"}]
DEFAULT_SAM_POINT_COORDINATES = [[320.0, 320.0]]
DEFAULT_SAM_POINT_LABELS = [1]

PROFILE_INPUT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "vlm_prompt": {
        "prompt": DEFAULT_VLM_PROMPT,
    },
    "vlm_florence2": {
        "prompt": DEFAULT_FLORENCE2_PROMPT,
        "task": DEFAULT_FLORENCE2_PROMPT,
    },
    "vlm_moondream2": {
        "classes": DEFAULT_MOONDREAM_CLASSES,
    },
    "open_vocabulary_detection": {
        "classes": DEFAULT_OPEN_VOCAB_CLASSES,
    },
    "sam3_text_segmentation": {
        "prompts": DEFAULT_SAM3_PROMPTS,
    },
    "interactive_sam": {
        "point_coordinates": DEFAULT_SAM_POINT_COORDINATES,
        "point_labels": DEFAULT_SAM_POINT_LABELS,
    },
}

INPUT_KIND_DEFAULTS: Dict[str, Any] = {
    "text": DEFAULT_VLM_PROMPT,
    "text_list": DEFAULT_OPEN_VOCAB_CLASSES,
    "structured_prompt_list": DEFAULT_SAM3_PROMPTS,
    "prompt_geometry": DEFAULT_SAM_POINT_COORDINATES,
}


def _get_json_path(data: Any, json_path: str) -> Any:
    current = data
    for part in json_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def read_package_json_value(
    package_dir: Path,
    *,
    filename: str,
    json_path: str,
) -> Any:
    config_path = package_dir / filename
    if not config_path.is_file():
        return None

    raw = read_json(str(config_path))
    if not isinstance(raw, dict):
        return None

    return _get_json_path(raw, json_path)


def resolve_profiling_method(
    *,
    module_name: Optional[str] = None,
    class_name: Optional[str] = None,
    architecture: Optional[str] = None,
    task_type: Optional[str] = None,
    backend: Optional[str] = None,
    user_method: Optional[str] = None,
) -> str:
    """Return the harness method name from the registry task profile unless overridden."""
    if user_method:
        return user_method

    context = resolve_registry_input_context(
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        backend=backend,
    )
    task_profile_spec = context.get("task_profile_spec")
    if isinstance(task_profile_spec, dict):
        profiling_method = task_profile_spec.get("profiling_method")
        if isinstance(profiling_method, str) and profiling_method:
            return profiling_method

    return "infer"


def _default_for_input(
    input_spec: Dict[str, Any],
    *,
    profile_name: Optional[str],
) -> Any:
    name = str(input_spec.get("name") or "")
    if profile_name and name in (PROFILE_INPUT_DEFAULTS.get(profile_name) or {}):
        return PROFILE_INPUT_DEFAULTS[profile_name][name]

    kind = str(input_spec.get("kind") or "")
    if kind in INPUT_KIND_DEFAULTS:
        return INPUT_KIND_DEFAULTS[kind]

    return None


def _default_for_method_kwarg(
    kwarg_spec: Dict[str, Any],
    *,
    package_dir: Path,
) -> Any:
    name = str(kwarg_spec.get("name") or "")
    if name == "max_new_tokens":
        resolution = kwarg_spec.get("resolution")
        if isinstance(resolution, dict):
            static_spec = resolution.get("static")
            if isinstance(static_spec, dict):
                file_name = static_spec.get("file")
                json_path = static_spec.get("json_path")
                if isinstance(file_name, str) and isinstance(json_path, str):
                    value = read_package_json_value(
                        package_dir,
                        filename=file_name,
                        json_path=json_path,
                    )
                    if value is not None:
                        return int(value)

        return DEFAULT_MAX_NEW_TOKENS

    return None


def infer_kwargs_defaults_from_task_profile(
    task_profile_spec: Optional[Dict[str, Any]],
    *,
    package_dir: Path,
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return harness defaults for non-image inputs declared by a task profile."""
    if not isinstance(task_profile_spec, dict):
        return {}

    defaults: Dict[str, Any] = {}
    profile_defaults = PROFILE_INPUT_DEFAULTS.get(profile_name or "", {})
    if profile_defaults:
        defaults.update(profile_defaults)

    for input_spec in task_profile_spec.get("inputs") or []:
        if not isinstance(input_spec, dict):
            continue

        name = str(input_spec.get("name") or "")
        if not name or name == "images":
            continue
        if name in defaults:
            continue
        if not input_spec.get("required", False):
            continue

        value = _default_for_input(input_spec, profile_name=profile_name)
        if value is not None:
            defaults[name] = value

    for kwarg_spec in task_profile_spec.get("method_kwargs_memory_relevant") or []:
        if not isinstance(kwarg_spec, dict):
            continue

        name = str(kwarg_spec.get("name") or "")
        if not name or name in defaults:
            continue

        value = _default_for_method_kwarg(kwarg_spec, package_dir=package_dir)
        if value is not None:
            defaults[name] = value

    return defaults


def build_profiling_infer_kwargs(
    *,
    package_dir: Path,
    module_name: Optional[str] = None,
    class_name: Optional[str] = None,
    architecture: Optional[str] = None,
    task_type: Optional[str] = None,
    backend: Optional[str] = None,
    user: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge registry task-profile defaults with user ``infer_kwargs`` overrides."""
    context = resolve_registry_input_context(
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        backend=backend,
    )
    profile_name = context.get("task_inference_profile")
    task_profile_spec = context.get("task_profile_spec")

    merged = infer_kwargs_defaults_from_task_profile(
        task_profile_spec if isinstance(task_profile_spec, dict) else None,
        package_dir=package_dir,
        profile_name=str(profile_name) if profile_name else None,
    )

    if user:
        merged.update(user)

    return merged


def _prompt_token_length_from_model(model: Any, prompt: str) -> Optional[int]:
    processor = getattr(model, "_processor", None)
    if processor is None:
        return None

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None

    try:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
    except Exception:
        return None

    return len(encoded)


def compute_runtime_axis_values(
    model: Any,
    infer_kwargs: Dict[str, Any],
    *,
    task_profile_spec: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute axis values that require a loaded model (e.g. prompt token length)."""
    if not isinstance(task_profile_spec, dict):
        return {}

    runtime_axes: Dict[str, Dict[str, Any]] = {}
    for input_spec in task_profile_spec.get("inputs") or []:
        if not isinstance(input_spec, dict):
            continue

        input_name = str(input_spec.get("name") or "")
        if not input_name:
            continue

        value = infer_kwargs.get(input_name)
        if value is None:
            continue

        axis_values: Dict[str, Any] = {}
        for axis_spec in input_spec.get("axes") or []:
            if not isinstance(axis_spec, dict):
                continue

            axis_name = str(axis_spec.get("name") or "")
            if axis_name == "prompt_token_length" and isinstance(value, str):
                token_length = _prompt_token_length_from_model(model, value)
                if token_length is not None:
                    axis_values[axis_name] = token_length
            elif axis_name == "num_classes" and isinstance(value, list):
                axis_values[axis_name] = len(value)
            elif axis_name in {"num_points", "num_boxes"} and hasattr(value, "__len__"):
                try:
                    axis_values[axis_name] = len(value)
                except TypeError:
                    pass

        if axis_values:
            runtime_axes[input_name] = axis_values

    return runtime_axes

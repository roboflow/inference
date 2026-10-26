"""Centralized discovery of optional optimization runtime components."""

import importlib
from functools import lru_cache
from typing import Dict, Mapping, Tuple

from inference_models.models.optimization.contracts import immutable_mapping

_RUNTIME_COMPONENT_MODULES: Dict[str, Tuple[str, ...]] = {
    "Pillow": ("PIL",),
    "TensorRT": ("tensorrt", "tensorrt_lean"),
    "torch": ("torch",),
    "torchvision": ("torchvision",),
    "triton": ("triton",),
}


def _runtime_component_is_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False

    return True


@lru_cache(maxsize=1)
def get_runtime_components() -> Mapping[str, bool]:
    """Report whether known optional optimization packages can be imported.

    A component is available when any of its candidate modules imports, probed
    in order until the first success.

    This deliberately limits discovery to stable package-level facts. Toolchain
    health and implementation-specific compilation remain runtime concerns.

    Returns:
        Runtime component names mapped to package import availability.
    """
    availability: Dict[str, bool] = {}
    for component, module_names in _RUNTIME_COMPONENT_MODULES.items():
        availability[component] = any(
            _runtime_component_is_available(module_name) for module_name in module_names
        )

    components = immutable_mapping(availability)

    return components

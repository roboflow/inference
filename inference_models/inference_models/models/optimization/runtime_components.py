"""Centralized discovery of optional optimization runtime components."""

import importlib
from functools import lru_cache
from typing import Dict, Mapping

from inference_models.models.optimization.contracts import immutable_mapping

_RUNTIME_COMPONENT_MODULES = {
    "Pillow": "PIL",
    "TensorRT": "tensorrt",
    "torch": "torch",
    "torchvision": "torchvision",
    "triton": "triton",
    "VPI": "vpi",
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

    This deliberately limits discovery to stable package-level facts. Toolchain
    health and implementation-specific compilation remain runtime concerns.

    Returns:
        Runtime component names mapped to package import availability.
    """
    availability: Dict[str, bool] = {}
    for component, module_name in _RUNTIME_COMPONENT_MODULES.items():
        availability[component] = _runtime_component_is_available(module_name)

    components = immutable_mapping(availability)

    return components

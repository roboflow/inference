from typing import Dict, TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
    from inference.core.interfaces.stream.stream import Stream
    from inference.core.models.base import Model
    from inference.models.utils import get_model, get_roboflow_model

_LAZY_ATTRIBUTES: Dict[str, Callable[[], Any]] = {
    "Stream": lambda: _import_from("inference.core.interfaces.stream.stream", "Stream"),
    "InferencePipeline": lambda: _import_from(
        "inference.core.interfaces.stream.inference_pipeline", "InferencePipeline"
    ),
    "get_model": lambda: _import_from("inference.models.utils", "get_model"),
    "get_roboflow_model": lambda: _import_from(
        "inference.models.utils", "get_roboflow_model"
    ),
}


def _import_from(module_path: str, attribute_name: str) -> Any:
    """Import and return an attribute from the specified module."""
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attribute_name)


def __getattr__(name: str) -> Any:
    """Implement lazy loading for module attributes."""
    if name in _LAZY_ATTRIBUTES:
        return _LAZY_ATTRIBUTES[name]()
    raise AttributeError(f"module 'inference' has no attribute '{name}'")


__all__ = [
    "InferencePipeline",
    "Stream",
    "get_model",
    "get_roboflow_model",
    "Model",
]

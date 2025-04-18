from typing import Any, Callable

_LAZY_ATTRIBUTES: dict[str, Callable[[], Any]] = {
    "Stream": lambda: _import_from("inference.core.interfaces.stream.stream", "Stream"),
    "InferencePipeline": lambda: _import_from(
        "inference.core.interfaces.stream.inference_pipeline", "InferencePipeline"
    ),
    "get_model": lambda: _import_model_util("get_model"),
    "get_roboflow_model": lambda: _import_model_util("get_roboflow_model"),
}


def _import_from(module_path: str, attribute_name: str) -> Any:
    """Import and return an attribute from the specified module."""
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attribute_name)


def _import_model_util(name: str) -> Any:
    from inference.models.utils import get_model, get_roboflow_model

    return locals()[name]


def __getattr__(name: str) -> Any:
    """Implement lazy loading for module attributes."""
    if name in _LAZY_ATTRIBUTES:
        return _LAZY_ATTRIBUTES[name]()
    raise AttributeError(f"module 'inference' has no attribute '{name}'")

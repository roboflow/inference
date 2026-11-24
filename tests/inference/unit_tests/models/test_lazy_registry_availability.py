"""Regressions for optional availability and registry serialization."""

import warnings
from types import SimpleNamespace

import pytest

from inference.core.registries import lazy


@pytest.mark.parametrize(
    "operation",
    ["contains", "len", "iter", "dict", "items", "values", "equal", "get", "lookup"],
)
@pytest.mark.parametrize("adapter", [False, True])
def test_unavailable_entries_obey_mapping_contract(monkeypatch, operation, adapter):
    """Keep every mapping operation consistent for failed optional imports.

    Args:
        monkeypatch: Pytest patch fixture.
        operation: Mapping operation to exercise before any lookup.
        adapter: Whether failed adapters also have unavailable fallbacks.
    """

    def _import(module_path):
        if module_path == "missing":
            raise ModuleNotFoundError("optional dependency")
        if module_path == "required":
            raise AssertionError("Required model must stay deferred")
        return SimpleNamespace(Model=int)

    monkeypatch.setattr(lazy.importlib, "import_module", _import)
    present, missing, alias, required = (
        [("task", name) for name in ("present", "missing", "alias", "required")]
        if adapter
        else ("present", "missing", "alias", "required")
    )
    reference = lazy._LazyModelClass("missing:Model", optional=True)
    registry = lazy._LazyModelRegistry(
        {present: int, missing: reference, alias: reference}
    )
    if adapter:
        for key in (missing, alias):
            registry.set_adapter(key, lazy._LazyModelClass("missing:Adapter"))
    expected = {present: int}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if operation == "contains":
            registry[required] = lazy._LazyModelClass("required:Model")
            assert required in registry
            assert missing not in registry
            assert alias not in registry
        elif operation == "len":
            assert len(registry) == 1
        elif operation == "iter":
            registry[required] = lazy._LazyModelClass("required:Model")
            assert set(registry) == {present, required}
        elif operation == "dict":
            assert dict(registry) == expected
        elif operation == "items":
            assert list(registry.items()) == list(expected.items())
        elif operation == "values":
            assert list(registry.values()) == [int]
        elif operation == "equal":
            assert registry == expected
        elif operation == "get":
            assert registry.get(missing, "default") == "default"
            assert missing not in registry
        else:
            with pytest.raises(KeyError):
                registry[missing]
            assert missing not in registry
            assert len(registry) == 1


@pytest.mark.parametrize(
    "operation",
    ["contains", "len", "iter", "dict", "items", "values", "equal", "get", "lookup"],
)
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("adapter_depth", [2, 3])
def test_nested_adapter_availability(monkeypatch, operation, available, adapter_depth):
    """Keep nested adapter availability consistent across mapping operations.

    Args:
        monkeypatch: Pytest patch fixture.
        operation: Mapping operation to exercise before any lookup.
        available: Whether the innermost optional model can be imported.
        adapter_depth: Number of adapters wrapping the optional model.
    """

    def _import(module_path):
        if module_path == "model" and available:
            return SimpleNamespace(Model=str)

        raise ModuleNotFoundError("optional dependency")

    monkeypatch.setattr(lazy.importlib, "import_module", _import)
    key = ("vlm", "qwen3vl")
    present = ("task", "present")
    registry = lazy._LazyModelRegistry(
        {present: int, key: lazy._LazyModelClass("model:Model", optional=True)}
    )
    for index in range(adapter_depth):
        registry.set_adapter(key, lazy._LazyModelClass(f"adapter{index}:Model"))

    expected = {present: int, **({key: str} if available else {})}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if operation == "contains":
            assert (key in registry) is available
        elif operation == "len":
            assert len(registry) == len(expected)
        elif operation == "iter":
            assert list(registry) == list(expected)
        elif operation == "dict":
            assert dict(registry) == expected
        elif operation == "items":
            assert list(registry.items()) == list(expected.items())
        elif operation == "values":
            assert list(registry.values()) == list(expected.values())
        elif operation == "equal":
            assert registry == expected
        elif operation == "get":
            assert registry.get(key) is expected.get(key)
        elif available:
            assert registry[key] is str
        else:
            with pytest.raises(KeyError):
                registry[key]

        assert (key in registry) is available
        assert len(registry) == len(expected)
        assert list(registry) == list(expected)
        assert dict(registry) == expected
        assert list(registry.items()) == list(expected.items())
        assert list(registry.values()) == list(expected.values())
        assert registry.get(key) is expected.get(key)

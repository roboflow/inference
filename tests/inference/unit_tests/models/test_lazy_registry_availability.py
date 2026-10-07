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

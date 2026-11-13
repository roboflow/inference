"""Regressions for optional availability and registry serialization."""

import copy
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inference.core.registries import lazy


@pytest.mark.parametrize("method", ["pickle", "deepcopy", "copy"])
@pytest.mark.parametrize("resolved", [False, True])
def test_registry_round_trip_recreates_locks(monkeypatch, method, resolved):
    """Preserve deferred entries and independent mutation after copying.

    Args:
        monkeypatch: Pytest patch fixture.
        method: Copy or serialization operation to exercise.
        resolved: Whether one alias has been resolved before copying.
    """
    importer = Mock(return_value=SimpleNamespace(Model=int))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    reference = lazy._LazyModelClass("example:Model")
    registry = lazy._LazyModelRegistry({"one": reference, "two": reference})
    registry.set_adapter(("task", "adapter"), lazy._LazyModelClass("example:Model"))
    if resolved:
        assert registry["one"] is int

    if method == "pickle":
        restored = pickle.loads(pickle.dumps(registry))
    elif method == "deepcopy":
        restored = copy.deepcopy(registry)
    else:
        restored = copy.copy(registry)
    assert restored._lock is not registry._lock
    restored["new"] = str
    assert "new" not in registry
    assert restored["one"] is restored["two"] is int
    assert restored[("task", "adapter")] is int
    if not resolved and method != "copy":
        assert restored._entries.get("two") is not reference

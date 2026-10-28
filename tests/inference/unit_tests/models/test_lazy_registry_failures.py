"""Regressions for optional availability and registry serialization."""

import gc
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from inference.core.registries import lazy
from inference.core.warnings import ModelDependencyMissing


@pytest.mark.parametrize("optional", [False, True])
def test_cached_failure_does_not_retain_completed_request(monkeypatch, optional):
    def import_model(path):
        raise ImportError("missing dependency")

    class RequestPayload:
        pass

    monkeypatch.setattr(lazy.importlib, "import_module", import_model)
    reference = lazy._LazyModelClass("missing:Model", optional=optional)
    registry = lazy._LazyModelRegistry({"model": reference})

    def request():
        payload = RequestPayload()
        payload_ref = weakref.ref(payload)
        try:
            registry["model"]
        except (ImportError, KeyError):
            pass

        return payload_ref

    for _ in range(3):
        payload_ref = request()
        gc.collect()
        assert payload_ref() is None


def test_uncopyable_import_error_preserves_original_contract(monkeypatch):
    class ImportFailure(ImportError):
        def __init__(self, message, *, code):
            super().__init__(message)
            self.code = code

    calls = []

    def import_model(path):
        calls.append(path)
        raise ImportFailure("dependency unavailable", code=42)

    monkeypatch.setattr(lazy.importlib, "import_module", import_model)
    reference = lazy._LazyModelClass("missing:Model")
    registry = lazy._LazyModelRegistry({"model": reference})

    for _ in range(2):
        with pytest.raises(ImportFailure, match="dependency unavailable") as caught:
            registry["model"]
        assert caught.value.code == 42

    assert calls == ["missing", "missing"]
    assert reference._failure is None


@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("adapter", [False, True])
def test_failed_import_is_cached_across_aliases_and_copies(
    monkeypatch, concurrent, adapter
):
    """Cache one failure and warning for shared references.

    Args:
        monkeypatch: Pytest patch fixture.
        concurrent: Whether requests overlap across threads.
        adapter: Whether the reference is an adapter without a fallback.
    """
    importer = Mock(side_effect=ModuleNotFoundError("missing dependency"))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    reference = lazy._LazyModelClass(
        "missing:Model",
        optional=True,
        warning_message="unavailable",
        warning_category=ModelDependencyMissing,
    )
    keys = [("task", "one"), ("task", "two")]
    registry = lazy._LazyModelRegistry(dict.fromkeys(keys, reference))
    if adapter:
        registry = lazy._LazyModelRegistry()
        for key in keys:
            registry.set_adapter(key, reference)

    registries = [registry.copy() for _ in range(16)]

    def _lookup(index):
        with pytest.raises(KeyError):
            registries[index][keys[index % 2]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if concurrent:
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(_lookup, range(16)))
        else:
            for index in range(16):
                _lookup(index)

    importer.assert_called_once_with("missing")
    assert len(caught) == 1


@pytest.mark.parametrize("membership", [False, True])
def test_cached_import_failure_keeps_a_bounded_traceback(monkeypatch, membership):
    """Avoid retaining earlier request frames when a model stays unavailable.

    Args:
        monkeypatch: Pytest patch fixture.
        membership: Whether to probe availability rather than request the model.
    """
    importer = Mock(side_effect=ModuleNotFoundError("missing dependency"))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    reference = lazy._LazyModelClass("missing:Model", optional=True)
    key = ("task", "missing")
    registry = lazy._LazyModelRegistry({key: reference})
    lengths = []
    for _ in range(100):
        if membership:
            assert key not in registry
        else:
            with pytest.raises(KeyError):
                registry[key]

        traceback = reference._failure.__traceback__
        length = 0
        while traceback is not None:
            length += 1
            traceback = traceback.tb_next

        lengths.append(length)

    assert max(lengths[1:]) <= lengths[0]
    importer.assert_called_once_with("missing")

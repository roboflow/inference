"""Regressions for optional availability and registry serialization."""

import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from inference.core.registries import lazy
from inference.core.warnings import ModelDependencyMissing


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

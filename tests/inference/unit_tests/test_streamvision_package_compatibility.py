"""Legacy import compatibility for the stream runtime moved into `streamvision`.

Every historical `inference.core.interfaces.{camera,stream,stream_manager}`
name listed in `streamvision_move_inventory.json` must resolve to the SAME module
object as its canonical `streamvision` name, except the four retained names that
resolve to host-owned modules under `inference.core.interfaces.legacy_stream`.
The subprocess tests use a fresh interpreter per load order.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from inference import _workflows_compat
from inference._workflows_compat_inventory import (
    _INVENTORY_LEGACY,
    _INVENTORY_PACKAGES,
    _STREAM_INVENTORY_LEGACY,
    _STREAM_INVENTORY_PACKAGES,
    _WORKFLOWS_INVENTORY_LEGACY,
    _WORKFLOWS_INVENTORY_PACKAGES,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INVENTORY_PATH = Path(__file__).with_name("streamvision_move_inventory.json")
_INVENTORY = json.loads(_INVENTORY_PATH.read_text())
_LEGACY_ROOTS = (
    "inference.core.interfaces.camera",
    "inference.core.interfaces.stream",
    "inference.core.interfaces.stream_manager",
)
_MOVED_PAIRS = [
    (entry["legacy"], entry["canonical"])
    for entry in _INVENTORY["packages"] + _INVENTORY["modules"]
]
_RETAINED_PAIRS = [
    (entry["legacy"], entry["retained"]) for entry in _INVENTORY["retained_legacy"]
]
# Canonical retained-name spellings: finder aliases only, not in the inventory JSON.
_CANONICAL_RETAINED_ALIASES = {
    "streamvision." + legacy[len("inference.core.interfaces.") :]
    for legacy, _ in _RETAINED_PAIRS
}


def test_inventory_matches_runtime_module() -> None:
    from_json = {legacy for legacy, _ in _MOVED_PAIRS + _RETAINED_PAIRS}
    assert set(_LEGACY_ROOTS) <= from_json
    assert _STREAM_INVENTORY_LEGACY == from_json | _CANONICAL_RETAINED_ALIASES
    assert _STREAM_INVENTORY_PACKAGES == {
        entry["legacy"] for entry in _INVENTORY["packages"]
    }
    assert _INVENTORY_LEGACY == _WORKFLOWS_INVENTORY_LEGACY | _STREAM_INVENTORY_LEGACY
    assert _INVENTORY_PACKAGES == (
        _WORKFLOWS_INVENTORY_PACKAGES | _STREAM_INVENTORY_PACKAGES
    )


def test_prefix_map_is_sorted_by_descending_legacy_length() -> None:
    lengths = [len(legacy) for legacy, _ in _workflows_compat._PREFIX_MAP]
    assert lengths == sorted(lengths, reverse=True)


def test_prefix_map_derives_every_inventory_target() -> None:
    for legacy, expected in _MOVED_PAIRS + _RETAINED_PAIRS:
        assert _workflows_compat._canonical_name(legacy) == expected


def test_host_exports_are_keyed_by_canonical_name() -> None:
    assert set(_workflows_compat._HOST_EXPORTS) == {"streamvision.stream.sinks"}


@pytest.mark.parametrize("legacy,canonical", _MOVED_PAIRS, ids=lambda v: v)
def test_legacy_import_returns_canonical_module(legacy: str, canonical: str) -> None:
    canonical_module = importlib.import_module(canonical)
    legacy_module = importlib.import_module(legacy)
    assert legacy_module is canonical_module
    assert sys.modules[legacy] is sys.modules[canonical]
    assert legacy_module.__name__ == canonical
    assert legacy_module.__spec__ is not None
    assert legacy_module.__spec__.name == canonical
    assert Path(legacy_module.__file__).is_relative_to(
        Path(importlib.import_module("streamvision").__file__).parent
    )


@pytest.mark.parametrize("legacy,retained", _RETAINED_PAIRS, ids=lambda v: v)
def test_retained_names_resolve_to_host_modules(
    legacy: str, retained: str, stub_ultralytics_if_missing
) -> None:
    legacy_module = importlib.import_module(legacy)
    assert legacy_module is importlib.import_module(retained)
    assert legacy_module.__name__ == retained
    parent_name, _, child_name = legacy.rpartition(".")
    canonical_parent = _workflows_compat._canonical_name(parent_name)
    canonical_parent_module = importlib.import_module(canonical_parent)
    # Uses PathFinder directly: sys.modules cache or compat finder would short-circuit.
    real_spec = importlib.machinery.PathFinder.find_spec(
        f"{canonical_parent}.{child_name}", canonical_parent_module.__path__
    )
    assert real_spec is None


def test_streamvision_stream_facade_submodules_only_resolve_via_compat_finder() -> None:
    import streamvision.stream

    for name in ("inference_pipeline", "stream"):
        assert importlib.util.find_spec(f"streamvision.stream.{name}") is not None
        real_spec = importlib.machinery.PathFinder.find_spec(
            f"streamvision.stream.{name}", streamvision.stream.__path__
        )
        assert real_spec is None


def test_unknown_legacy_child_raises_module_not_found() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("inference.core.interfaces.stream.does_not_exist")


def test_legacy_pickle_reference_loads_the_canonical_class() -> None:
    from streamvision.stream.entities import ModelConfig

    import inference.core.interfaces.stream.entities as legacy_entities

    assert legacy_entities.ModelConfig is ModelConfig
    assert ModelConfig.__module__ == "streamvision.stream.entities"
    assert pickle.loads(pickle.dumps(ModelConfig)) is ModelConfig
    legacy_reference = b"cinference.core.interfaces.stream.entities\nModelConfig\n."
    assert pickle.loads(legacy_reference) is ModelConfig


@pytest.mark.parametrize("class_name", ["WebRTCWorkerRequest", "WebRTCWorkerResult"])
def test_legacy_webrtc_worker_pickle_references_load_the_moved_classes(
    class_name: str,
) -> None:
    import streamvision.webrtc_worker.entities as canonical_entities

    moved_class = getattr(canonical_entities, class_name)
    assert moved_class.__module__ == "streamvision.webrtc_worker.entities"
    assert pickle.loads(pickle.dumps(moved_class)) is moved_class
    legacy_reference = (
        f"cinference.core.interfaces.webrtc_worker.entities\n{class_name}\n."
    ).encode()
    assert pickle.loads(legacy_reference) is moved_class


def test_reload_through_legacy_name_keeps_identity() -> None:
    import streamvision.stream.watchdog as canonical

    import inference.core.interfaces.stream.watchdog as legacy

    assert importlib.reload(legacy) is canonical
    assert canonical.__spec__.name == "streamvision.stream.watchdog"
    assert sys.modules["inference.core.interfaces.stream.watchdog"] is canonical


def test_legacy_parent_attribute_is_bound_to_canonical_child() -> None:
    import streamvision.stream.entities as canonical

    import inference.core.interfaces.stream.entities  # noqa: F401

    assert sys.modules["inference.core.interfaces.stream"].entities is canonical


def test_string_patch_through_legacy_name_patches_canonical_module() -> None:
    import streamvision.stream.sinks as canonical

    sentinel = object()
    with mock.patch("inference.core.interfaces.stream.sinks.render_boxes", sentinel):
        assert canonical.render_boxes is sentinel
    assert canonical.render_boxes is not sentinel


def test_legacy_alias_binds_host_exports_missing_from_the_canonical_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import streamvision.stream.sinks as canonical

    from inference.core.active_learning.middlewares import ActiveLearningMiddleware

    # Simulates a canonical module imported before the compat finder existed.
    monkeypatch.delattr(canonical, "ActiveLearningMiddleware")
    monkeypatch.delitem(sys.modules, "inference.core.interfaces.stream.sinks")

    legacy = importlib.import_module("inference.core.interfaces.stream.sinks")

    assert legacy is canonical
    assert canonical.ActiveLearningMiddleware is ActiveLearningMiddleware


def test_top_level_inference_pipeline_is_the_host_subclass() -> None:
    from streamvision.stream.pipeline import InferencePipeline as NeutralPipeline

    from inference import InferencePipeline
    from inference.core.interfaces.legacy_stream import inference_pipeline

    assert InferencePipeline is inference_pipeline.InferencePipeline
    assert InferencePipeline.__mro__[1] is NeutralPipeline


def test_stream_fromlist_import_resolves_the_four_retained_modules(
    stub_ultralytics_if_missing,
) -> None:
    # regression: fromlist used canonical __name__, not the legacy prefix.
    import streamvision.stream

    from inference.core.interfaces import legacy_stream
    from inference.core.interfaces.stream import inference_pipeline, stream
    from inference.core.interfaces.stream.model_handlers import (
        roboflow_models,
        yolo_world,
    )

    assert inference_pipeline is streamvision.stream.inference_pipeline
    assert inference_pipeline is legacy_stream.inference_pipeline
    assert stream is streamvision.stream.stream
    assert stream is legacy_stream.stream
    assert roboflow_models is legacy_stream.model_handlers.roboflow_models
    assert yolo_world is legacy_stream.model_handlers.yolo_world


# Fresh-interpreter tests: the parent process already imported `inference`.


def _run_subprocess(source: str, env: dict) -> subprocess.CompletedProcess:
    child_env = os.environ.copy()
    child_env.pop("MODEL_CACHE_DIR", None)
    child_env.update(env)
    child_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_REPO_ROOT / "inference_models"),
            str(_REPO_ROOT / "workflows"),
            str(_REPO_ROOT / "stream_vision"),
            *filter(None, [os.environ.get("PYTHONPATH")]),
        ]
    )
    child_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=_REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    return completed


def test_subprocess_streamvision_stream_stands_alone_without_inference() -> None:
    # blocks inference entirely; canonical names must not resolve without the finder.
    result = _run_subprocess(
        """
        import sys

        class _BlockInference:
            def find_spec(self, name, path, target=None):
                if name == "inference" or name.startswith("inference."):
                    raise ModuleNotFoundError(f"blocked: {name}", name=name)
                return None

        sys.meta_path.insert(0, _BlockInference())

        import streamvision.stream
        assert not hasattr(streamvision.stream, "inference_pipeline")
        try:
            import streamvision.stream.inference_pipeline
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("expected ModuleNotFoundError")
        print("ok")
        """,
        env={"API_LOGGING_ENABLED": "False"},
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("ok")


def test_runpy_and_inspect_resolve_through_the_compat_loader() -> None:
    # regression: runpy needs get_code; inspect/linecache need get_source/get_filename.
    result = _run_subprocess(
        """
        import importlib
        import inspect
        import runpy

        runpy.run_module(
            "inference.core.interfaces.camera.stream_error_codes",
            run_name="__main__",
        )

        legacy = importlib.import_module(
            "inference.core.interfaces.camera.stream_error_codes"
        )
        canonical = importlib.import_module("streamvision.camera.stream_error_codes")
        assert inspect.getsource(legacy) == inspect.getsource(canonical)
        print("ok")
        """,
        env={"API_LOGGING_ENABLED": "False"},
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("ok")


# Canonical-first imports `inference.core` first: see the `before-inference` case.
@pytest.mark.parametrize("order", ["legacy", "canonical"])
def test_subprocess_every_inventoried_name_shares_identity(order: str) -> None:
    pairs = _MOVED_PAIRS + _RETAINED_PAIRS
    result = _run_subprocess(
        """
        import importlib, json, os, sys, types
        from unittest.mock import MagicMock
        try:
            import ultralytics  # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("ultralytics")
            stub.YOLO = MagicMock(name="ultralytics.YOLO")
            stub.settings = MagicMock(name="ultralytics.settings")
            sys.modules["ultralytics"] = stub
        pairs = json.loads(os.environ["FRAME_FLOW_COMPAT_PAIRS"])
        if os.environ["FRAME_FLOW_COMPAT_ORDER"] == "canonical":
            import inference.core
        for legacy_name, canonical_name in pairs:
            if os.environ["FRAME_FLOW_COMPAT_ORDER"] == "canonical":
                canonical = importlib.import_module(canonical_name)
                legacy = importlib.import_module(legacy_name)
            else:
                legacy = importlib.import_module(legacy_name)
                canonical = importlib.import_module(canonical_name)
            assert legacy is canonical, legacy_name
            assert sys.modules[legacy_name] is canonical, legacy_name
            assert canonical.__spec__.name == canonical_name, legacy_name
        print("ok")
        """,
        env={
            "API_LOGGING_ENABLED": "False",
            "FRAME_FLOW_COMPAT_ORDER": order,
            "FRAME_FLOW_COMPAT_PAIRS": json.dumps(pairs),
        },
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("ok")


@pytest.mark.parametrize("first", ["legacy", "canonical"])
def test_subprocess_reload_parent_binding_and_string_patch(first: str) -> None:
    result = _run_subprocess(
        """
        import importlib, os, sys
        from unittest.mock import patch
        if os.environ["FRAME_FLOW_COMPAT_FIRST"] == "canonical":
            import inference.core
            importlib.import_module("streamvision.stream.sinks")
        legacy = importlib.import_module("inference.core.interfaces.stream.sinks")
        canonical = importlib.import_module("streamvision.stream.sinks")
        assert legacy is canonical
        assert importlib.reload(legacy) is canonical
        assert canonical.__spec__.name == "streamvision.stream.sinks"
        assert sys.modules["inference.core.interfaces.stream"].sinks is canonical
        sentinel = object()
        with patch("inference.core.interfaces.stream.sinks.render_boxes", sentinel):
            assert canonical.render_boxes is sentinel
        print("ok")
        """,
        env={"API_LOGGING_ENABLED": "False", "FRAME_FLOW_COMPAT_FIRST": first},
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("ok")


_ACTIVE_LEARNING_ORDERS = {
    "legacy-first": "",
    "canonical-after-inference": "import inference.core\nimport streamvision.stream.sinks",
    "canonical-before-inference": "import streamvision.stream.sinks",
}


@pytest.mark.parametrize("order", sorted(_ACTIVE_LEARNING_ORDERS))
def test_subprocess_sinks_export_the_concrete_active_learning_middleware(
    order: str,
) -> None:
    result = _run_subprocess(
        _ACTIVE_LEARNING_ORDERS[order] + """
from inference.core.active_learning.middlewares import ActiveLearningMiddleware
from inference.core.interfaces.stream.sinks import (
    ActiveLearningMiddleware as HistoricalExport,
)
import streamvision.stream.sinks as canonical
assert HistoricalExport is ActiveLearningMiddleware
assert canonical.ActiveLearningMiddleware is ActiveLearningMiddleware
print("ok")
""",
        env={"API_LOGGING_ENABLED": "False"},
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("ok")


# `import inference` alone is lazy; `inference.core` runs the configuration bootstrap.
_LIBRARY_ORDER_SCRIPT = """
import streamvision.stream.pipeline
import inference.core
print("ok")
"""


def test_subprocess_canonical_pipeline_then_inference_imports_by_default() -> None:
    result = _run_subprocess(
        _LIBRARY_ORDER_SCRIPT, env={"API_LOGGING_ENABLED": "False"}
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("ok")


def test_subprocess_canonical_pipeline_then_inference_rejects_a_differing_env() -> None:
    # Like roboflow_workflows: configuration is consumed once per process.
    result = _run_subprocess(
        _LIBRARY_ORDER_SCRIPT,
        env={"API_LOGGING_ENABLED": "False", "MODEL_CACHE_DIR": "/some/other/path"},
    )

    assert result.returncode != 0
    assert "WorkflowEnvironmentConfigurationError" in result.stderr
    assert "inference/core/__init__.py" in result.stderr

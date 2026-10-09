"""Check real imports and overlapping first requests for the model registry."""

import importlib
import json
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_registry_import_defers_model_implementations():
    """Keep implementation imports out of a fresh registry process."""
    probe = """
import json
import sys
from inference.models.utils import ROBOFLOW_MODEL_TYPES
assert ('object-detection', 'yolov8n') in ROBOFLOW_MODEL_TYPES
prefixes = ('transformers', 'peft', 'inference.models.yolov8',
            'inference.core.models.inference_models_adapters')
print(json.dumps([name for name in sys.modules
                  if any(name == p or name.startswith(p + '.') for p in prefixes)]))
"""
    process = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "DISABLE_VERSION_CHECK": "True"},
    )
    assert json.loads(process.stdout.strip().splitlines()[-1]) == []


@pytest.mark.parametrize("use_adapters", [False, True])
@pytest.mark.parametrize("use_proxy", [False, True])
@pytest.mark.parametrize(
    "sam3_enabled", [False, pytest.param(True, marks=pytest.mark.slow)]
)
def test_every_registry_key_imports_its_real_class(
    monkeypatch, use_adapters, use_proxy, sam3_enabled
):
    """Import every advertised class without substituting implementation modules.

    Args:
        monkeypatch: Pytest patch fixture.
        use_adapters: Whether to select inference-models adapters.
        use_proxy: Whether to select vLLM proxies.
        sam3_enabled: Enable SAM3 when its full dependency stack is installed.
    """
    from inference.core import env
    from inference.core.registries.lazy import _LazyModelRegistry
    from inference.models import utils, vllm_proxy

    inventory = json.loads(
        Path(__file__).with_name("registry_inventory.json").read_text()
    )
    for flag, enabled in inventory["flags"].items():
        monkeypatch.setattr(env, flag, enabled)
    monkeypatch.setattr(env, "CORE_MODEL_OWLV2_ENABLED", True)
    monkeypatch.setattr(env, "CORE_MODEL_SAM3_ENABLED", sam3_enabled)
    monkeypatch.setattr(env, "SAM3_3D_OBJECTS_ENABLED", sam3_enabled)
    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", use_adapters)
    monkeypatch.setattr(vllm_proxy, "VLLM_PROXY_ENABLED", use_proxy)
    registry = runpy.run_path(utils.__file__)["ROBOFLOW_MODEL_TYPES"]
    assert isinstance(registry, _LazyModelRegistry)
    expected = {
        tuple(key): path
        for path, keys in inventory["cases"][
            f"adapters={use_adapters},proxy={use_proxy}"
        ].items()
        for key in keys
        if sam3_enabled or key[1] not in {"sam3", "sam3-large"}
    }
    if use_adapters:
        owl_adapter = (
            "inference.models.owlv2.owlv2_inference_models:InferenceModelsOwlV2Adapter"
        )
        instant_adapter = (
            "inference.models.owlv2.rf_instant_inference_models:"
            "InferenceModelsRFInstantModelAdapter"
        )
        expected[("object-detection", "owlv2")] = owl_adapter
        expected[("open-vocabulary-object-detection", "owlv2")] = owl_adapter
        expected[("object-detection", "owlv2-finetuned")] = instant_adapter
        expected[("object-detection", "roboflow-instant")] = instant_adapter
    else:
        expected[("object-detection", "owlv2")] = "inference.models.owlv2.owlv2:OwlV2"
        expected[("object-detection", "owlv2-finetuned")] = (
            "inference.models.owlv2.owlv2:SerializedOwlV2"
        )
    if sam3_enabled:
        expected[("3d-reconstruction", "sam3-3d-objects")] = (
            "inference.models.sam3_3d.segment_anything_3d:SegmentAnything3_3D_Objects"
        )
    fallback_paths = {
        tuple(key): path
        for path, keys in inventory["cases"][
            f"adapters=False,proxy={use_proxy}"
        ].items()
        for key in keys
    }
    available = {}
    for key, path in expected.items():
        module_path, class_name = path.split(":", 1)
        try:
            getattr(importlib.import_module(module_path), class_name)
        except (ImportError, AttributeError):
            fallback = fallback_paths.get(key) if use_adapters else None
            if fallback is None or fallback == path:
                continue

            module_path, class_name = fallback.split(":", 1)
            try:
                getattr(importlib.import_module(module_path), class_name)
            except (ImportError, AttributeError):
                continue

            path = fallback
        available[key] = path

    expected = available
    assert set(registry) == set(expected)
    failures = []
    for key in registry:
        try:
            model_class = registry[key]
            module_path, class_name = expected[key].split(":", 1)
            expected_class = getattr(importlib.import_module(module_path), class_name)
            assert isinstance(model_class, type)
            assert model_class is expected_class
            assert registry[key] is model_class
        except Exception as error:
            cause = error.__cause__ or error
            failures.append(f"{key}: {type(cause).__name__}: {cause}")

    assert not failures, "Unresolvable registry keys:\n" + "\n".join(failures)


@pytest.mark.parametrize("copy_registry", [False, True])
def test_overlapping_first_requests_share_resolution(monkeypatch, copy_registry):
    """Resolve a shared reference once even when requests use registry copies.

    Args:
        monkeypatch: Pytest patch fixture.
        copy_registry: Whether concurrent requests use independent registry copies.
    """
    from inference.core.registries import lazy

    model_class = type("ConcurrentModel", (), {})
    ready = Barrier(8)
    entered = Event()
    release = Event()

    def _import_model(module_path):
        entered.set()
        assert release.wait(timeout=10)
        return SimpleNamespace(Model=model_class)

    importer = Mock(side_effect=_import_model)
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    reference = lazy._LazyModelClass("example:Model")
    registry = lazy._LazyModelRegistry({"one": reference, "two": reference})
    registries = [registry.copy() if copy_registry else registry for _ in range(8)]

    def _lookup(index):
        ready.wait(timeout=10)
        return registries[index]["one" if index % 2 else "two"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_lookup, index) for index in range(8)]
        try:
            assert entered.wait(timeout=10)
            done, pending = wait(futures, timeout=0.1)
            assert not done
            assert len(pending) == 8
        finally:
            release.set()
        results = [future.result(timeout=10) for future in futures]

    assert all(result is model_class for result in results)
    importer.assert_called_once_with("example")


@pytest.mark.parametrize("cached", [False, True])
def test_unrelated_lookup_finishes_during_blocked_import(monkeypatch, cached):
    """Allow unrelated lookups while another implementation import is blocked.

    Args:
        monkeypatch: Pytest patch fixture.
        cached: Whether the unrelated class has already been resolved.
    """
    from inference.core.registries import lazy

    model_class = type("ConcurrentModel", (), {})
    entered = Event()
    release = Event()
    lookup_started = Event()
    lookup_finished = Event()

    def _import_model(module_path):
        if module_path == "blocked":
            entered.set()
            assert release.wait(timeout=10)

        return SimpleNamespace(Model=model_class)

    monkeypatch.setattr(lazy.importlib, "import_module", _import_model)
    registry = lazy._LazyModelRegistry(
        {
            "blocked": lazy._LazyModelClass("blocked:Model"),
            "unrelated": lazy._LazyModelClass("unrelated:Model"),
        }
    )
    if cached:
        assert registry["unrelated"] is model_class

    def _lookup_unrelated():
        lookup_started.set()
        result = registry["unrelated"]
        lookup_finished.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        blocked = executor.submit(registry.__getitem__, "blocked")
        try:
            assert entered.wait(timeout=5)
            unrelated = executor.submit(_lookup_unrelated)
            assert lookup_started.wait(timeout=5)
            assert lookup_finished.wait(timeout=5), "Lookup waited for unrelated import"
            assert unrelated.result(timeout=5) is model_class
            assert not blocked.done()
        finally:
            release.set()
        assert blocked.result(timeout=5) is model_class


@pytest.mark.parametrize("roboflow_registry", [False, True])
@pytest.mark.parametrize("lazy_registry", [False, True])
def test_unknown_model_error_message_is_unchanged(
    monkeypatch, roboflow_registry, lazy_registry
):
    """Retain the exception type and exact message for unknown model requests.

    Args:
        monkeypatch: Pytest patch fixture.
        roboflow_registry: Whether to exercise the Roboflow metadata wrapper.
        lazy_registry: Whether to use a deferred registry mapping.
    """
    from inference.core.exceptions import ModelNotRecognisedError
    from inference.core.registries import roboflow
    from inference.core.registries.base import ModelRegistry

    key = ("unknown-task", "unknown-model")
    entries = {}
    if lazy_registry:
        from inference.core.registries.lazy import _LazyModelRegistry

        entries = _LazyModelRegistry()

    if roboflow_registry:
        monkeypatch.setattr(roboflow, "get_model_type", Mock(return_value=key))
        registry = roboflow.RoboflowModelRegistry(entries)
        expected = (
            "Model type not supported, you may want to try a different inference "
            f"server configuration or endpoint: {key}"
        )
    else:
        registry = ModelRegistry(entries)
        expected = f"Could not find model of type: {key} in configured registry."

    with pytest.raises(ModelNotRecognisedError) as caught:
        if roboflow_registry:
            registry.get_model("example/1", api_key="test-key")
        else:
            registry.get_model(key, "example/1")
    assert str(caught.value) == expected


def test_sam3_visual_segmentation_fixture_in_fresh_process():
    """Exercise fixture cleanup without preceding imports masking native state."""
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).with_name("test_sam3_visual_segmentation.py")),
            "-q",
            "--tb=short",
            "--disable-warnings",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "DISABLE_VERSION_CHECK": "True"},
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_real_registry_probe_with_missing_optional_models(monkeypatch):
    """Exercise real registry coverage without the SAM and YOLO World extras.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    from inference import models

    real_import = importlib.import_module

    class _PartialModels:
        def __getattr__(self, name):
            if name in {"SegmentAnything", "YOLOWorld"}:
                raise ImportError("Optional model dependency is unavailable")
            return getattr(models, name)

    def _import(module_path, package=None):
        if module_path == "inference.models":
            return _PartialModels()
        return real_import(module_path, package=package)

    monkeypatch.setattr(importlib, "import_module", _import)
    test_every_registry_key_imports_its_real_class(
        monkeypatch, use_adapters=False, use_proxy=False, sam3_enabled=False
    )

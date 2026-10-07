"""Exercise lazy lookup, backend selection, and the public model-loading API."""

import json
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inference.core import env
from inference.core.exceptions import ModelNotRecognisedError
from inference.core.registries import lazy
from inference.core.registries.base import ModelRegistry
from inference.core.warnings import InferenceModelsStackMissing, ModelDependencyMissing
from inference.models import utils


def test_keys_and_membership_do_not_import_models(monkeypatch):
    """Inspect enabled keys without resolving values.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    importer = Mock(side_effect=AssertionError("Unexpected model import"))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    key = ("object-detection", "example")
    registry = lazy._LazyModelRegistry({key: lazy._LazyModelClass("example:Model")})

    assert len(registry) == 1
    assert list(registry) == list(registry.keys()) == [key]
    assert key in registry
    assert ("missing", "missing") not in registry
    assert registry.get(("missing", "missing")) is None
    assert list(registry.copy()) == [key]
    registry.clear()
    assert not registry
    importer.assert_not_called()


def test_lookup_aliases_views_and_mutation_return_classes(monkeypatch):
    """Resolve aliases once and retain ordinary mutable mapping behavior.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    model_class = type("ExampleModel", (), {})
    importer = Mock(return_value=SimpleNamespace(Model=model_class))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    reference = lazy._LazyModelClass("example:Model")
    registry = lazy._LazyModelRegistry({"one": reference, "two": reference})

    assert registry["one"] is registry.get("two") is model_class
    assert list(registry.values()) == [model_class, model_class]
    assert dict(registry.items()) == {"one": model_class, "two": model_class}
    importer.assert_called_once_with("example")

    copied = registry.copy()
    registry.update({"three": model_class})
    assert registry.pop("three") is model_class
    assert registry.pop("missing", None) is None
    del registry["one"]
    assert "one" in copied and "one" not in registry
    with pytest.raises(KeyError):
        registry["missing"]


def test_concurrent_requests_resolve_one_class(monkeypatch):
    """Share a successful class resolution across concurrent requests.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    model_class = type("ConcurrentModel", (), {})
    importer = Mock(return_value=SimpleNamespace(Model=model_class))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    registry = lazy._LazyModelRegistry({"model": lazy._LazyModelClass("example:Model")})

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: registry["model"], range(32)))

    assert all(result is model_class for result in results)
    importer.assert_called_once_with("example")


def test_optional_dependency_warning_is_deferred(monkeypatch):
    """Warn with the original installation guidance only upon lookup.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    importer = Mock(side_effect=ImportError("missing dependency"))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    message = utils._PALIGEMMA_DEPENDENCY_WARNING
    reference = lazy._LazyModelClass(
        "example:Model",
        optional=True,
        warning_message=message,
        warning_category=ModelDependencyMissing,
    )
    registry = lazy._LazyModelRegistry({"optional": reference})
    assert "optional" in registry
    importer.assert_not_called()

    with pytest.warns(ModelDependencyMissing) as caught:
        with pytest.raises(KeyError):
            registry["optional"]

    assert str(caught[0].message) == (
        "Your `inference` configuration does not support PaliGemma model. "
        "Use pip install 'inference[transformers]' to install missing requirements."
        "To suppress this warning, set PALIGEMMA_ENABLED to False."
    )


def test_adapter_failure_falls_back_and_caches_legacy_class(monkeypatch):
    """Keep adapter fallback behavior at resolution time.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    model_class = type("LegacyModel", (), {})
    importer = Mock(side_effect=ImportError("missing adapter"))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    key = ("object-detection", "example")
    registry = lazy._LazyModelRegistry({key: model_class})
    registry.set_adapter(key, lazy._LazyModelClass("example:Adapter"))
    importer.assert_not_called()

    with pytest.warns(InferenceModelsStackMissing) as caught:
        assert registry[key] is model_class

    assert str(caught[0].message) == (
        "`inference-models` stack is unavailable for model: example and task: "
        "object-detection, falling back to regular `inference` stack - error: "
        "missing adapter"
    )
    assert registry[key] is model_class
    importer.assert_called_once_with("example")


def test_required_import_failures_and_interrupts_are_not_hidden(monkeypatch):
    """Preserve required import failures and process interruption.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    importer = Mock(side_effect=RuntimeError("initialization failed"))
    monkeypatch.setattr(lazy.importlib, "import_module", importer)
    registry = lazy._LazyModelRegistry({"model": lazy._LazyModelClass("example:Model")})
    with pytest.raises(RuntimeError, match="initialization failed"):
        registry["model"]

    importer.side_effect = KeyboardInterrupt()
    registry["optional"] = lazy._LazyModelClass("example:Model", optional=True)
    with pytest.raises(KeyboardInterrupt):
        registry["optional"]


@pytest.mark.parametrize("use_adapters", [False, True])
def test_registry_keys_and_representative_classes(monkeypatch, use_adapters):
    """Resolve real detection, classification, stub, and local-package entries.

    Args:
        monkeypatch: Pytest patch fixture.
        use_adapters: Whether to select the inference-models bridge.
    """
    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", use_adapters)
    namespace = runpy.run_path(utils.__file__)
    registry = namespace["ROBOFLOW_MODEL_TYPES"]
    keys = set(registry)
    assert ("object-detection", "yolov8n") in keys
    assert ("classification", "resnet50") in keys
    assert ("classification", "stub") in keys
    if use_adapters:
        from inference.core.models.inference_models_adapters import (
            InferenceModelsClassificationAdapter,
            InferenceModelsObjectDetectionAdapter,
        )

        assert registry[("object-detection", "yolov8n")] is (
            InferenceModelsObjectDetectionAdapter
        )
        assert registry[("classification", "resnet50")] is (
            InferenceModelsClassificationAdapter
        )
        assert registry[("object-detection", "inference-models-local")] is (
            InferenceModelsObjectDetectionAdapter
        )
    else:
        from inference.models.resnet import ResNetClassification
        from inference.models.yolov8 import YOLOv8ObjectDetection

        assert registry[("object-detection", "yolov8n")] is YOLOv8ObjectDetection
        assert registry[("classification", "resnet50")] is ResNetClassification
        assert ("object-detection", "inference-models-local") not in registry

    from inference.core.models.stubs import ClassificationModelStub

    assert registry[("classification", "stub")] is ClassificationModelStub


@pytest.mark.parametrize("use_adapters", [False, True])
@pytest.mark.parametrize("use_proxy", [False, True])
def test_complete_registry_matches_original_inventory(
    monkeypatch, use_adapters, use_proxy
):
    """Preserve every original key and class choice in all backend modes.

    Args:
        monkeypatch: Pytest patch fixture.
        use_adapters: Whether to select inference-models adapters.
        use_proxy: Whether to select vLLM proxies.
    """
    from inference.models import vllm_proxy

    inventory = json.loads(
        Path(__file__).with_name("registry_inventory.json").read_text()
    )
    for flag, enabled in inventory["flags"].items():
        monkeypatch.setattr(env, flag, enabled)
    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", use_adapters)
    monkeypatch.setattr(vllm_proxy, "VLLM_PROXY_ENABLED", use_proxy)
    registry = runpy.run_path(utils.__file__)["ROBOFLOW_MODEL_TYPES"]

    class _Module:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, name):
            return type(name, (), {"registry_path": f"{self.name}:{name}"})

    monkeypatch.setattr(lazy.importlib, "import_module", _Module)
    expected = {
        tuple(key): path
        for path, keys in inventory["cases"][
            f"adapters={use_adapters},proxy={use_proxy}"
        ].items()
        for key in keys
    }
    assert set(registry) == set(expected)
    for key in registry:
        assert registry[key].registry_path == expected[key]


def test_disabled_optional_flags_do_not_register_keys(monkeypatch):
    """Keep disabled optional families out of the registry.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    for flag in ("PALIGEMMA_ENABLED", "FLORENCE2_ENABLED", "QWEN_3_5_ENABLED"):
        monkeypatch.setattr(env, flag, False)
    registry = runpy.run_path(utils.__file__)["ROBOFLOW_MODEL_TYPES"]
    assert not any("paligemma" in variant for _, variant in registry)
    assert not any("florence" in variant for _, variant in registry)
    assert not any("qwen3_5" in variant for _, variant in registry)


def test_global_core_flag_still_disables_core_models(monkeypatch):
    """Preserve the package-level foundation-model switch.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    monkeypatch.setattr(env, "CORE_MODELS_ENABLED", False)
    registry = runpy.run_path(utils.__file__)["ROBOFLOW_MODEL_TYPES"]
    assert ("embed", "clip") not in registry
    assert ("embed", "sam") not in registry
    assert ("embed", "sam2") not in registry
    assert ("ocr", "doctr") not in registry
    assert ("gaze", "l2cs") not in registry


def test_missing_optional_model_keeps_registry_error_type(monkeypatch):
    """Translate unavailable lazy implementations to the existing registry error.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    monkeypatch.setattr(
        lazy.importlib, "import_module", Mock(side_effect=ImportError("missing"))
    )
    registry = ModelRegistry(
        lazy._LazyModelRegistry(
            {"optional": lazy._LazyModelClass("example:Model", optional=True)}
        )
    )
    with pytest.raises(ModelNotRecognisedError):
        registry.get_model("optional", "example/1")


def test_public_helpers_forward_model_configuration(monkeypatch):
    """Retain get_model and get_roboflow_model construction and usage binding.

    Args:
        monkeypatch: Pytest patch fixture.
    """
    constructor = Mock(return_value=object())
    registry = lazy._LazyModelRegistry({("task", "variant"): constructor})
    monkeypatch.setattr(utils, "ROBOFLOW_MODEL_TYPES", registry)
    model_type = Mock(return_value=("task", "variant"))
    bind_usage = Mock()
    monkeypatch.setattr(utils, "get_model_type", model_type)
    monkeypatch.setattr(utils, "bind_usage_model_descriptor", bind_usage)

    model = utils.get_model("example/1", api_key="test-key", option=True)
    assert (
        utils.get_roboflow_model("example/1", api_key="test-key", option=True) is model
    )
    constructor.assert_called_with("example/1", api_key="test-key", option=True)
    model_type.assert_called_with("example/1", api_key="test-key")
    bind_usage.assert_called_with(model, "example/1")


def test_fresh_import_does_not_load_optional_model_stacks():
    """Check a fresh process so preceding tests cannot hide eager imports."""
    probe = """
import json
import sys
from inference.models.utils import ROBOFLOW_MODEL_TYPES
keys = list(ROBOFLOW_MODEL_TYPES.keys())
for key in keys:
    assert key in ROBOFLOW_MODEL_TYPES
prefixes = ('transformers', 'peft', 'flash_attn',
            'inference.models.paligemma', 'inference.models.florence2',
            'inference.models.yolov8', 'inference.models.resnet',
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
    assert json.loads(process.stdout.strip()) == []

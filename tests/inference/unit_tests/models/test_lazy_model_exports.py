"""Preserve model-class exports independently of registry backend selection."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inference.core import env
from inference.core.models.base import Model
from inference.core.registries import lazy
from inference.models import utils, vllm_proxy


@pytest.mark.parametrize("use_adapters", [False, True])
@pytest.mark.parametrize("use_proxy", [False, True])
@pytest.mark.parametrize("real_imports", [False, True])
def test_original_public_model_names_resolve_to_classes(
    monkeypatch, use_adapters, use_proxy, real_imports
):
    """Check every original public class export in all backend configurations.

    Args:
        monkeypatch: Pytest patch fixture.
        use_adapters: Whether to select inference-models adapters.
        use_proxy: Whether to select vLLM proxies.
        real_imports: Import installed implementations instead of controlled classes.
    """
    inventory = json.loads(
        Path(__file__).with_name("model_export_inventory.json").read_text()
    )
    for flag, enabled in inventory["flags"].items():
        monkeypatch.setattr(env, flag, enabled)
    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", use_adapters)
    monkeypatch.setattr(vllm_proxy, "VLLM_PROXY_ENABLED", use_proxy)
    expected = inventory["cases"][f"adapters={use_adapters},proxy={use_proxy}"]

    classes = {
        path: type(path.split(":")[1], (Model,), {}) for path in expected.values()
    }
    classes["inference.core.models.base:Model"] = Model

    def _import_model(module_path):
        module = SimpleNamespace(
            **{
                path.split(":")[1]: model_class
                for path, model_class in classes.items()
                if path.split(":")[0] == module_path
            }
        )
        return module

    importer = Mock(side_effect=_import_model)
    if not real_imports:
        monkeypatch.setattr(lazy.importlib, "import_module", importer)

    spec = importlib.util.spec_from_file_location("_model_exports", utils.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    importer.assert_not_called()

    failures = []
    for name, path in expected.items():
        # SAM3 implementations need Triton and tdfy; controlled imports cover these
        # exports even in environments without those optional dependencies.
        if real_imports and (
            "sam3" in path or name.startswith(("Sam3", "SegmentAnything3"))
        ):
            continue

        try:
            namespace = {}
            exec(f"from _model_exports import {name}", namespace)
            model_class = namespace[name]
            assert isinstance(model_class, type), f"{name} is not a class"
            assert callable(model_class)
            if name not in {"Gaze", "InferenceModelsGazeAdapter"}:
                assert issubclass(model_class, Model)
            if real_imports:
                module_path, class_name = path.split(":", 1)
                expected_class = getattr(
                    importlib.import_module(module_path), class_name
                )
            else:
                expected_class = classes[path]
            assert model_class is expected_class
            assert getattr(module, name) is model_class
            assert name in dir(module)
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")

    assert not failures, "Incompatible model-class exports:\n" + "\n".join(failures)
    if not real_imports:
        namespace = {}
        exec("from _model_exports import *", namespace)
        for name, path in expected.items():
            assert namespace[name] is classes[path]
        assert importer.call_count == len(set(expected.values())) - 1

    with pytest.raises(AttributeError):
        getattr(module, "UnknownModelClass")

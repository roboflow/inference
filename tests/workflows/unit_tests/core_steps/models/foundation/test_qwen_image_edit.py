"""Unit tests for the Qwen-Image-Edit workflow block (v1)."""

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.qwen_image_edit.v1 import (
    DEFAULT_MODEL_ID,
    BlockManifest,
    QwenImageEditBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def _make_block() -> QwenImageEditBlockV1:
    # Clear the class-level model cache so tests don't share state.
    QwenImageEditBlockV1._model_cache.clear()
    return QwenImageEditBlockV1(api_key="test-api-key")


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_minimal_valid():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_image_edit@v1",
            "name": "step",
            "images": "$inputs.image",
            "prompt": "make the sky purple",
        }
    )
    assert manifest.model_id == DEFAULT_MODEL_ID
    # Steps/guidance default to None ("auto") and are resolved by the backend.
    assert manifest.num_inference_steps is None
    assert manifest.guidance_scale is None
    # Lightning is the default so the block works out of the box (pulls base
    # model + LoRA from HuggingFace, no Roboflow registry entry required).
    assert manifest.use_lightning_lora is True
    assert manifest.scale_megapixels is None
    assert manifest.seed is None


def test_manifest_accepts_selector_prompt():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_image_edit@v1",
            "name": "step",
            "images": "$inputs.image",
            "prompt": "$inputs.edit_prompt",
        }
    )
    assert manifest.prompt == "$inputs.edit_prompt"


def test_manifest_requires_prompt():
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/qwen_image_edit@v1",
                "name": "step",
                "images": "$inputs.image",
            }
        )


def test_manifest_accepts_custom_params():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_image_edit@v1",
            "name": "step",
            "images": "$inputs.image",
            "prompt": "remove background",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "seed": 42,
        }
    )
    assert manifest.num_inference_steps == 50
    assert manifest.guidance_scale == 7.5
    assert manifest.seed == 42


# ---------------------------------------------------------------------------
# Block.run()
# ---------------------------------------------------------------------------


def _fake_pil_image():
    from PIL import Image

    return Image.new("RGB", (64, 64), color=(100, 150, 200))


def _run(block, fake_model, **overrides):
    """Call block.run with defaults; patch _get_model to return fake_model."""
    defaults = dict(
        images=[_stub_image()],
        prompt="make it brighter",
        model_id=DEFAULT_MODEL_ID,
        local_weights_path=None,
        use_lightning_lora=False,
        num_inference_steps=28,
        guidance_scale=5.0,
        seed=None,
        scale_megapixels=None,
    )
    defaults.update(overrides)
    with patch.object(block, "_get_model", return_value=fake_model):
        return block.run(**defaults)


def test_run_returns_image_output():
    block = _make_block()
    fake_model = MagicMock()
    fake_model.edit.return_value = _fake_pil_image()

    results = _run(block, fake_model)

    assert len(results) == 1
    assert "image" in results[0]
    assert isinstance(results[0]["image"], WorkflowImageData)


def test_run_passes_params_to_model():
    block = _make_block()
    fake_model = MagicMock()
    fake_model.edit.return_value = _fake_pil_image()

    _run(
        block,
        fake_model,
        prompt="add a hat",
        num_inference_steps=10,
        guidance_scale=3.5,
        seed=7,
    )

    call_kwargs = fake_model.edit.call_args.kwargs
    assert call_kwargs["prompt"] == "add a hat"
    assert call_kwargs["num_inference_steps"] == 10
    assert call_kwargs["guidance_scale"] == 3.5
    assert call_kwargs["seed"] == 7


def test_run_processes_batch():
    block = _make_block()
    fake_model = MagicMock()
    fake_model.edit.return_value = _fake_pil_image()

    results = _run(
        block, fake_model, images=[_stub_image(), _stub_image(), _stub_image()]
    )

    assert len(results) == 3
    assert fake_model.edit.call_count == 3


def test_get_model_uses_local_path_directly(tmp_path):
    # The concrete backend lives in the separately-published `inference_models`
    # package; skip when it is not installed (e.g. CI before a matching release).
    qwen_hf = pytest.importorskip(
        "inference_models.models.qwen_image_edit.qwen_image_edit_hf"
    )
    QwenImageEditHF = qwen_hf.QwenImageEditHF

    block = _make_block()
    # Create a fake weights dir so the path-existence check passes.
    weights_dir = str(tmp_path)

    fake_model = MagicMock(spec=QwenImageEditHF)

    with patch.object(
        QwenImageEditHF, "from_pretrained", return_value=fake_model
    ) as mock_fp:
        result = block._get_model(
            model_id=DEFAULT_MODEL_ID,
            local_weights_path=weights_dir,
            use_lightning_lora=False,
        )

    mock_fp.assert_called_once_with(
        model_name_or_path=weights_dir,
        local_files_only=True,
        use_lightning_lora=False,
    )
    assert result is fake_model


def test_get_model_lightning_lora_loads_from_huggingface():
    qwen_hf = pytest.importorskip(
        "inference_models.models.qwen_image_edit.qwen_image_edit_hf"
    )
    MODEL_ID = qwen_hf.MODEL_ID
    QwenImageEditHF = qwen_hf.QwenImageEditHF

    block = _make_block()

    fake_model = MagicMock(spec=QwenImageEditHF)
    with patch.object(
        QwenImageEditHF, "from_pretrained", return_value=fake_model
    ) as mock_fp:
        result = block._get_model(
            model_id=DEFAULT_MODEL_ID,
            local_weights_path=None,
            use_lightning_lora=True,
        )

    mock_fp.assert_called_once_with(
        model_name_or_path=MODEL_ID,
        local_files_only=False,
        use_lightning_lora=True,
    )
    assert result is fake_model


def test_get_model_raises_for_missing_local_path():
    block = _make_block()
    with pytest.raises(ValueError, match="does not exist"):
        block._get_model(
            model_id=DEFAULT_MODEL_ID,
            local_weights_path="/nonexistent/path",
            use_lightning_lora=False,
        )


def test_get_restrictions_returns_gpu_only_hard_restrictions():
    from inference.core.workflows.prototypes.block import Runtime, Severity

    restrictions = BlockManifest.get_restrictions()
    assert len(restrictions) == 2
    runtimes = {
        r for restriction in restrictions for r in restriction.applies_to_runtimes
    }
    assert Runtime.SELF_HOSTED_CPU in runtimes
    assert Runtime.HOSTED_SERVERLESS in runtimes
    for restriction in restrictions:
        assert restriction.severity == Severity.HARD


def test_model_cache_reuses_instance():
    block = _make_block()
    fake_model = MagicMock()

    # Seed the cache directly — simulates the state after first load. The cache
    # key is (load-path-or-model-id, use_lightning_lora).
    QwenImageEditBlockV1._model_cache[(DEFAULT_MODEL_ID, False)] = fake_model

    result = block._get_model(
        DEFAULT_MODEL_ID, local_weights_path=None, use_lightning_lora=False
    )
    result2 = block._get_model(
        DEFAULT_MODEL_ID, local_weights_path=None, use_lightning_lora=False
    )

    assert result is fake_model
    assert result2 is fake_model


def _install_fake_qwen_backend(monkeypatch, from_pretrained):
    """Stub the qwen backend module so cache tests run without the real
    `inference_models` release that ships it (fixture-scoped, auto-undone)."""
    leaf_name = "inference_models.models.qwen_image_edit.qwen_image_edit_hf"
    leaf = types.ModuleType(leaf_name)
    leaf.MODEL_ID = "Qwen/Qwen-Image-Edit"
    leaf.QwenImageEditHF = types.SimpleNamespace(from_pretrained=from_pretrained)
    for name in (
        "inference_models",
        "inference_models.models",
        "inference_models.models.qwen_image_edit",
    ):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, leaf_name, leaf)


def test_get_model_concurrent_cold_load_loads_only_once(monkeypatch):
    block = _make_block()
    load_entered = threading.Event()
    release_load = threading.Event()
    load_calls = []

    def slow_from_pretrained(**kwargs):
        load_calls.append(kwargs)
        load_entered.set()
        assert release_load.wait(timeout=5)
        return MagicMock()

    _install_fake_qwen_backend(monkeypatch, slow_from_pretrained)

    results = [None, None]

    def worker(idx):
        results[idx] = block._get_model(
            model_id=DEFAULT_MODEL_ID,
            local_weights_path=None,
            use_lightning_lora=True,
        )

    first = threading.Thread(target=worker, args=(0,))
    second = threading.Thread(target=worker, args=(1,))
    first.start()
    assert load_entered.wait(timeout=5)
    second.start()
    time.sleep(0.1)  # let the second thread reach the load lock
    release_load.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert len(load_calls) == 1, "concurrent cold requests must share one load"
    assert results[0] is results[1]


def test_model_cache_evicts_previous_configuration(monkeypatch):
    block = _make_block()
    QwenImageEditBlockV1._model_cache[("/old/weights/path", False)] = MagicMock()
    new_model = MagicMock()
    _install_fake_qwen_backend(monkeypatch, lambda **kwargs: new_model)

    result = block._get_model(
        model_id=DEFAULT_MODEL_ID,
        local_weights_path=None,
        use_lightning_lora=True,
    )

    assert result is new_model
    # Single-entry policy: loading a new configuration evicts the previous one
    # so multiple multi-GB pipelines never accumulate in memory.
    assert list(QwenImageEditBlockV1._model_cache.keys()) == [(DEFAULT_MODEL_ID, True)]

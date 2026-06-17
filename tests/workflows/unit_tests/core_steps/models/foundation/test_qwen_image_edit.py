"""Unit tests for the Qwen-Image-Edit workflow block (v1)."""

from types import SimpleNamespace
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
    assert manifest.num_inference_steps == 28
    assert manifest.guidance_scale == 5.0
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
        num_inference_steps=28,
        guidance_scale=5.0,

        seed=None,
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

    _run(block, fake_model, prompt="add a hat", num_inference_steps=10,
         guidance_scale=3.5, seed=7)

    call_kwargs = fake_model.edit.call_args.kwargs
    assert call_kwargs["prompt"] == "add a hat"
    assert call_kwargs["num_inference_steps"] == 10
    assert call_kwargs["guidance_scale"] == 3.5
    assert call_kwargs["seed"] == 7


def test_run_processes_batch():
    block = _make_block()
    fake_model = MagicMock()
    fake_model.edit.return_value = _fake_pil_image()

    results = _run(block, fake_model, images=[_stub_image(), _stub_image(), _stub_image()])

    assert len(results) == 3
    assert fake_model.edit.call_count == 3


def test_get_model_uses_local_path_directly(tmp_path):
    block = _make_block()
    # Create a fake weights dir so the path-existence check passes.
    weights_dir = str(tmp_path)

    from inference_models.models.qwen_image_edit.qwen_image_edit_hf import QwenImageEditHF
    fake_model = MagicMock(spec=QwenImageEditHF)

    with patch.object(QwenImageEditHF, "from_pretrained", return_value=fake_model) as mock_fp:
        result = block._get_model(model_id=DEFAULT_MODEL_ID, local_weights_path=weights_dir)

    mock_fp.assert_called_once_with(model_name_or_path=weights_dir, local_files_only=True)
    assert result is fake_model


def test_get_model_raises_for_missing_local_path():
    block = _make_block()
    with pytest.raises(ValueError, match="does not exist"):
        block._get_model(model_id=DEFAULT_MODEL_ID, local_weights_path="/nonexistent/path")


def test_get_restrictions_returns_gpu_only_hard_restrictions():
    from inference.core.workflows.prototypes.block import Runtime, Severity
    restrictions = BlockManifest.get_restrictions()
    assert len(restrictions) == 2
    runtimes = {r for restriction in restrictions for r in restriction.applies_to_runtimes}
    assert Runtime.SELF_HOSTED_CPU in runtimes
    assert Runtime.HOSTED_SERVERLESS in runtimes
    for restriction in restrictions:
        assert restriction.severity == Severity.HARD


def test_model_cache_reuses_instance():
    block = _make_block()
    fake_model = MagicMock()

    # Seed the cache directly — simulates the state after first load.
    QwenImageEditBlockV1._model_cache[DEFAULT_MODEL_ID] = fake_model

    result = block._get_model(DEFAULT_MODEL_ID, local_weights_path=None)
    result2 = block._get_model(DEFAULT_MODEL_ID, local_weights_path=None)

    assert result is fake_model
    assert result2 is fake_model

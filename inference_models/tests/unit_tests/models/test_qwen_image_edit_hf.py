"""Unit tests for QwenImageEditHF.

All HF/diffusers imports are mocked so the test runs without weights or GPU.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from inference_models.models.qwen_image_edit.qwen_image_edit_hf import (
    QwenImageEditHF,
    _to_pil,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(pipeline=None) -> QwenImageEditHF:
    if pipeline is None:
        pipeline = MagicMock()
        pipeline.return_value = SimpleNamespace(images=[Image.new("RGB", (64, 64))])
    return QwenImageEditHF(pipeline=pipeline, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# _to_pil
# ---------------------------------------------------------------------------


def test_to_pil_from_pil_returns_rgb():
    src = Image.new("RGBA", (10, 10))
    result = _to_pil(src)
    assert result.mode == "RGB"


def test_to_pil_from_numpy_bgr_converts_to_rgb():
    # A numpy array where B=10, G=20, R=30
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 0] = 10  # B channel
    arr[:, :, 2] = 30  # R channel
    result = _to_pil(arr)
    # After BGR→RGB flip, pixel[0] should now be 30 (was R)
    pixel = result.getpixel((0, 0))
    assert pixel[0] == 30, "Red channel should be first after BGR→RGB conversion"
    assert pixel[2] == 10, "Blue channel should be last after BGR→RGB conversion"


# ---------------------------------------------------------------------------
# edit()
# ---------------------------------------------------------------------------


def test_edit_calls_pipeline_with_correct_args():
    expected_image = Image.new("RGB", (64, 64), color=(1, 2, 3))
    pipeline = MagicMock(return_value=SimpleNamespace(images=[expected_image]))
    model = _make_model(pipeline=pipeline)

    source = Image.new("RGB", (64, 64))
    result = model.edit(image=source, prompt="make it red")

    call_kwargs = pipeline.call_args.kwargs
    assert call_kwargs["prompt"] == "make it red"
    assert "image" in call_kwargs
    assert result is expected_image


def test_edit_accepts_numpy_image():
    expected_image = Image.new("RGB", (8, 8))
    pipeline = MagicMock(return_value=SimpleNamespace(images=[expected_image]))
    model = _make_model(pipeline=pipeline)

    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    result = model.edit(image=arr, prompt="brighten")

    assert result is expected_image


def test_edit_passes_generation_params():
    pipeline = MagicMock(
        return_value=SimpleNamespace(images=[Image.new("RGB", (4, 4))])
    )
    model = _make_model(pipeline=pipeline)

    model.edit(
        image=Image.new("RGB", (4, 4)),
        prompt="test",
        num_inference_steps=10,
        guidance_scale=3.0,
        strength=0.5,
    )

    kwargs = pipeline.call_args.kwargs
    assert kwargs["num_inference_steps"] == 10
    assert kwargs["guidance_scale"] == 3.0
    assert kwargs["strength"] == 0.5


def test_edit_with_seed_creates_generator():
    pipeline = MagicMock(
        return_value=SimpleNamespace(images=[Image.new("RGB", (4, 4))])
    )
    model = _make_model(pipeline=pipeline)

    model.edit(image=Image.new("RGB", (4, 4)), prompt="test", seed=42)

    kwargs = pipeline.call_args.kwargs
    assert kwargs.get("generator") is not None


def test_edit_without_seed_passes_none_generator():
    pipeline = MagicMock(
        return_value=SimpleNamespace(images=[Image.new("RGB", (4, 4))])
    )
    model = _make_model(pipeline=pipeline)

    model.edit(image=Image.new("RGB", (4, 4)), prompt="test", seed=None)

    kwargs = pipeline.call_args.kwargs
    assert kwargs.get("generator") is None


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------


def test_registry_entry_resolves():
    from inference_models.models.auto_loaders.models_registry import (
        IMAGE_EDITING_TASK,
        REGISTERED_MODELS,
    )
    from inference_models.models.auto_loaders.entities import BackendType

    key = ("qwen-image-edit", IMAGE_EDITING_TASK, BackendType.HF)
    assert key in REGISTERED_MODELS, "qwen-image-edit not found in REGISTERED_MODELS"
    cls = REGISTERED_MODELS[key].resolve()
    assert cls is QwenImageEditHF

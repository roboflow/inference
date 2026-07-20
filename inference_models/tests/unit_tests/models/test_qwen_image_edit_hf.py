"""Unit tests for QwenImageEditHF.

All HF/diffusers imports are mocked so the test runs without weights or GPU.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

import inference_models.models.qwen_image_edit.qwen_image_edit_hf as qhf
from inference_models.models.qwen_image_edit.qwen_image_edit_hf import (
    QwenImageEditHF,
    _scale_to_megapixels,
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
    # guidance is forwarded to the pipeline as `true_cfg_scale`.
    assert kwargs["true_cfg_scale"] == 3.0
    # extra kwargs are forwarded verbatim to the pipeline.
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
# Generation defaults (auto-select based on Lightning LoRA)
# ---------------------------------------------------------------------------


def test_edit_auto_defaults_without_lora():
    pipeline = MagicMock(
        return_value=SimpleNamespace(images=[Image.new("RGB", (4, 4))])
    )
    model = QwenImageEditHF(
        pipeline=pipeline,
        device=torch.device("cpu"),
        lightning_lora_applied=False,
    )

    model.edit(image=Image.new("RGB", (4, 4)), prompt="test")

    kwargs = pipeline.call_args.kwargs
    assert kwargs["num_inference_steps"] == 28
    assert kwargs["true_cfg_scale"] == 5.0


def test_edit_auto_defaults_with_lightning_lora():
    pipeline = MagicMock(
        return_value=SimpleNamespace(images=[Image.new("RGB", (4, 4))])
    )
    model = QwenImageEditHF(
        pipeline=pipeline,
        device=torch.device("cpu"),
        lightning_lora_applied=True,
    )

    model.edit(image=Image.new("RGB", (4, 4)), prompt="test")

    kwargs = pipeline.call_args.kwargs
    # Lightning recipe: ~4 steps, guidance disabled.
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["true_cfg_scale"] == 1.0


def test_edit_explicit_values_override_lightning_defaults():
    pipeline = MagicMock(
        return_value=SimpleNamespace(images=[Image.new("RGB", (4, 4))])
    )
    model = QwenImageEditHF(
        pipeline=pipeline,
        device=torch.device("cpu"),
        lightning_lora_applied=True,
    )

    model.edit(
        image=Image.new("RGB", (4, 4)),
        prompt="test",
        num_inference_steps=12,
        guidance_scale=2.5,
    )

    kwargs = pipeline.call_args.kwargs
    assert kwargs["num_inference_steps"] == 12
    assert kwargs["true_cfg_scale"] == 2.5


# ---------------------------------------------------------------------------
# from_pretrained + Lightning LoRA fusing
# ---------------------------------------------------------------------------


def _patched_pipeline_module(pipe: MagicMock):
    """Return a fake `diffusers` module exposing QwenImageEditPipeline."""
    fake_diffusers = SimpleNamespace(
        QwenImageEditPipeline=SimpleNamespace(
            from_pretrained=MagicMock(return_value=pipe)
        )
    )
    return patch.dict("sys.modules", {"diffusers": fake_diffusers})


def test_from_pretrained_loads_lightning_lora_when_enabled():
    pipe = MagicMock()
    with _patched_pipeline_module(pipe), patch.object(
        qhf, "_build_lightning_scheduler", return_value=MagicMock()
    ) as m_sched, patch.object(
        qhf, "_load_transformer", return_value=MagicMock()
    ) as m_trans, patch.object(
        qhf, "_load_lightning_lora", return_value=True
    ) as m_lora:
        model = QwenImageEditHF.from_pretrained(
            model_name_or_path="Qwen/Qwen-Image-Edit",
            device=torch.device("cpu"),
            local_files_only=False,
            use_lightning_lora=True,
        )

    # Lightning path wires the dedicated scheduler + transformer and loads LoRA.
    m_sched.assert_called_once()
    m_trans.assert_called_once()
    m_lora.assert_called_once()
    assert model.lightning_lora_applied is True


def test_from_pretrained_skips_lora_when_disabled():
    pipe = MagicMock()
    with _patched_pipeline_module(pipe), patch.object(
        qhf, "_build_lightning_scheduler"
    ) as m_sched, patch.object(qhf, "_load_transformer") as m_trans, patch.object(
        qhf, "_load_lightning_lora"
    ) as m_lora:
        model = QwenImageEditHF.from_pretrained(
            model_name_or_path="Qwen/Qwen-Image-Edit",
            device=torch.device("cpu"),
            local_files_only=True,
            use_lightning_lora=False,
        )

    m_sched.assert_not_called()
    m_trans.assert_not_called()
    m_lora.assert_not_called()
    assert model.lightning_lora_applied is False


def test_from_pretrained_lora_failure_is_non_fatal():
    pipe = MagicMock()
    with _patched_pipeline_module(pipe), patch.object(
        qhf, "_build_lightning_scheduler", return_value=MagicMock()
    ), patch.object(qhf, "_load_transformer", return_value=MagicMock()), patch.object(
        qhf, "_load_lightning_lora", return_value=False
    ):
        model = QwenImageEditHF.from_pretrained(
            model_name_or_path="Qwen/Qwen-Image-Edit",
            device=torch.device("cpu"),
            local_files_only=False,
            use_lightning_lora=True,
        )

    # LoRA load failed → falls back to base weights rather than raising.
    assert model.lightning_lora_applied is False


# ---------------------------------------------------------------------------
# Image scaling
# ---------------------------------------------------------------------------


def test_scale_to_megapixels_downscales_large_image():
    img = Image.new("RGB", (2000, 2000))  # 4 MP
    out = _scale_to_megapixels(img, 0.35)
    assert (out.width * out.height) / 1_000_000 <= 0.35 + 1e-6
    assert out.width % 8 == 0 and out.height % 8 == 0


def test_scale_to_megapixels_leaves_small_image():
    img = Image.new("RGB", (256, 256))  # ~0.065 MP
    out = _scale_to_megapixels(img, 0.35)
    assert out.size == (256, 256)


def test_edit_auto_scales_input_with_lightning_lora():
    pipeline = MagicMock(
        return_value=SimpleNamespace(images=[Image.new("RGB", (4, 4))])
    )
    model = QwenImageEditHF(
        pipeline=pipeline,
        device=torch.device("cpu"),
        lightning_lora_applied=True,
    )

    model.edit(image=Image.new("RGB", (2000, 2000)), prompt="x")

    sent = pipeline.call_args.kwargs["image"]
    assert (sent.width * sent.height) / 1_000_000 <= 0.35 + 1e-6


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------


def test_registry_entry_resolves():
    from inference_models.models.auto_loaders.entities import BackendType
    from inference_models.models.auto_loaders.models_registry import (
        IMAGE_EDITING_TASK,
        REGISTERED_MODELS,
    )

    key = ("qwen-image-edit", IMAGE_EDITING_TASK, BackendType.HF)
    assert key in REGISTERED_MODELS, "qwen-image-edit not found in REGISTERED_MODELS"
    cls = REGISTERED_MODELS[key].resolve()
    assert cls is QwenImageEditHF

"""Integration tests for the Qwen-Image-Edit HF backend.

These tests exercise the REAL diffusers pipeline (no mocks): weight loading,
Lightning LoRA fusing, CPU-offload placement and end-to-end generation. They are
GPU-only and slow — the Lightning path downloads the base model + LoRA from
HuggingFace on first run.

Set ``QWEN_IMAGE_EDIT_WEIGHTS_DIR`` to a locally downloaded weights directory to
skip the HuggingFace download and load with ``local_files_only=True``.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from inference_models.models.qwen_image_edit.qwen_image_edit_hf import (
    MODEL_ID,
    QwenImageEditHF,
)

LOCAL_WEIGHTS_DIR = os.getenv("QWEN_IMAGE_EDIT_WEIGHTS_DIR")


@pytest.fixture(scope="module")
def qwen_image_edit_lightning_model() -> QwenImageEditHF:
    if LOCAL_WEIGHTS_DIR:
        return QwenImageEditHF.from_pretrained(
            model_name_or_path=LOCAL_WEIGHTS_DIR,
            local_files_only=True,
            use_lightning_lora=True,
        )
    return QwenImageEditHF.from_pretrained(
        model_name_or_path=MODEL_ID,
        local_files_only=False,
        use_lightning_lora=True,
    )


@pytest.mark.slow
@pytest.mark.gpu_only
def test_lightning_lora_is_applied(
    qwen_image_edit_lightning_model: QwenImageEditHF,
) -> None:
    # LoRA loading is non-fatal by design — this asserts it actually succeeded,
    # otherwise the generation tests below would silently exercise the base model
    # with the Lightning step/guidance recipe.
    assert qwen_image_edit_lightning_model.lightning_lora_applied is True


@pytest.mark.slow
@pytest.mark.gpu_only
def test_edit_produces_image(
    qwen_image_edit_lightning_model: QwenImageEditHF,
    dog_image_numpy: np.ndarray,
) -> None:
    # when
    result = qwen_image_edit_lightning_model.edit(
        image=dog_image_numpy,
        prompt="make the image look like a watercolor painting",
        seed=42,
    )

    # then
    assert isinstance(result, Image.Image)
    assert result.width > 0 and result.height > 0
    # the edit must actually change pixels vs the (downscaled) input
    input_pil = Image.fromarray(dog_image_numpy[:, :, ::-1]).resize(result.size)
    assert not np.array_equal(np.asarray(result), np.asarray(input_pil))


@pytest.mark.slow
@pytest.mark.gpu_only
def test_edit_is_reproducible_with_fixed_seed(
    qwen_image_edit_lightning_model: QwenImageEditHF,
    dog_image_numpy: np.ndarray,
) -> None:
    # when
    first = qwen_image_edit_lightning_model.edit(
        image=dog_image_numpy,
        prompt="add a red hat on the dog",
        seed=1337,
    )
    second = qwen_image_edit_lightning_model.edit(
        image=dog_image_numpy,
        prompt="add a red hat on the dog",
        seed=1337,
    )

    # then
    assert np.array_equal(np.asarray(first), np.asarray(second))


@pytest.mark.slow
@pytest.mark.gpu_only
def test_edit_respects_scale_megapixels_cap(
    qwen_image_edit_lightning_model: QwenImageEditHF,
    dog_image_numpy: np.ndarray,
) -> None:
    # when
    result = qwen_image_edit_lightning_model.edit(
        image=dog_image_numpy,
        prompt="make the sky blue",
        seed=42,
        scale_megapixels=0.1,
        num_inference_steps=1,
    )

    # then
    assert (result.width * result.height) / 1_000_000 <= 0.11


@pytest.mark.slow
@pytest.mark.gpu_only
def test_edit_speed_benchmark(
    qwen_image_edit_lightning_model: QwenImageEditHF,
    dog_image_numpy: np.ndarray,
) -> None:
    """Not an assertion-heavy test — reports Lightning-path latency for the
    speed evidence required by the new-model contract (run on NVIDIA L4)."""
    import time

    # warmup
    qwen_image_edit_lightning_model.edit(
        image=dog_image_numpy, prompt="make the sky blue", seed=42
    )
    torch.cuda.synchronize()

    start = time.monotonic()
    runs = 3
    for _ in range(runs):
        qwen_image_edit_lightning_model.edit(
            image=dog_image_numpy, prompt="make the sky blue", seed=42
        )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start

    print(
        f"\nQwen-Image-Edit (Lightning) avg latency over {runs} runs: "
        f"{elapsed / runs:.2f}s on {torch.cuda.get_device_name(0)}"
    )
    assert elapsed > 0

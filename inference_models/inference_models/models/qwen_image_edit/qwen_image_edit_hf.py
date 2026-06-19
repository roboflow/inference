"""Qwen-Image-Edit HF backend.

Wraps the Qwen/Qwen-Image-Edit pipeline so it satisfies the
`from_pretrained(model_name_or_path)` contract expected by `inference_models.AutoModel`.

The model takes a source image and a text editing instruction and returns an
edited PIL Image. It is loaded via `diffusers.QwenImageEditPipeline`.
"""

from threading import Lock
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_GUIDANCE_SCALE,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_NUM_INFERENCE_STEPS,
)


class QwenImageEditHF:
    """Thin wrapper around the Qwen-Image-Edit pipeline."""

    default_dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        **kwargs,
    ) -> "QwenImageEditHF":
        from diffusers import QwenImageEditPipeline

        pipe = QwenImageEditPipeline.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        pipe.to(cls.default_dtype)

        if device.type == "cuda":
            # CPU offload moves each sub-model to GPU only when needed,
            # keeping peak VRAM usage low on smaller cards.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        return cls(pipeline=pipe, device=device)

    def __init__(self, pipeline, device: torch.device):
        self._pipeline = pipeline
        self._device = device
        self._lock = Lock()

    def edit(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: str,
        num_inference_steps: int = INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_GUIDANCE_SCALE,
        negative_prompt: str = " ",
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        pil_image = _to_pil(image)

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        with self._lock, torch.inference_mode():
            result = self._pipeline(
                image=pil_image,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
            )

        return result.images[0]


def _to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    # numpy: assume BGR (OpenCV convention used throughout inference)
    rgb = image[:, :, ::-1].copy() if image.ndim == 3 else image
    return Image.fromarray(rgb.astype(np.uint8))

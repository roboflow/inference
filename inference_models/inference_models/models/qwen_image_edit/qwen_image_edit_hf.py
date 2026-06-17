"""Qwen-Image-Edit HF backend.

Wraps the Qwen/Qwen-Image-Edit-2511 diffusion pipeline so it satisfies the
`from_pretrained(model_name_or_path)` contract expected by `inference_models.AutoModel`.

The model takes a source image and a text editing instruction and returns an
edited PIL Image.  It is a *diffusion* model (not an autoregressive LM), so
the pipeline is loaded via `diffusers.DiffusionPipeline` and returns
`pipe.images[0]` rather than text tokens.

NOTE: Qwen/Qwen-Image-Edit-2511 was released after the August 2025 knowledge
cutoff.  Before deploying, verify the exact class name and call signature on
the HuggingFace model page and adjust the TODO sections below if needed.
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
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_STRENGTH,
)


class QwenImageEditHF:
    """Thin wrapper around the Qwen-Image-Edit diffusion pipeline."""

    default_dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        **kwargs,
    ) -> "QwenImageEditHF":
        # TODO: verify the exact pipeline class on the HF model card.
        # DiffusionPipeline is the universal diffusers loader; if the model
        # ships a custom pipeline class, replace the import and constructor.
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=cls.default_dtype,
            local_files_only=local_files_only,
        )
        pipe = pipe.to(device)

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
        strength: float = INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_STRENGTH,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        pil_image = _to_pil(image)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        with self._lock, torch.inference_mode():
            result = self._pipeline(
                prompt=prompt,
                image=pil_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        return result.images[0]


def _to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    # numpy: assume BGR (OpenCV convention used throughout inference)
    rgb = image[:, :, ::-1].copy() if image.ndim == 3 else image
    return Image.fromarray(rgb.astype(np.uint8))

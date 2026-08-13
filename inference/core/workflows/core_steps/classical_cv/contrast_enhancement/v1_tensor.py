"""Tensor-native sibling of ``contrast_enhancement/v1``.

Tensor-materialised images run the per-channel chain (percentile/min-max
stretch, linear contrast around 128, optional gamma, clip, uint8) as torch ops
on the image's device; numpy/base64-born images delegate to the v1 numpy
implementation. Channel independence makes the math identical for RGB (tensor
layout) and BGR (numpy layout).

Numpy parity:
- ``astype(np.uint8)`` truncates, so the tensor path truncates too;
- flat-histogram channels (``v_max <= v_min``) pass through untouched; a flat
  single-channel image returns the input object, like the numpy grayscale
  branch;
- percentiles use numpy's linear interpolation, implemented sort-based
  (``torch.quantile`` has an input-size limit and patchy MPS support).
"""

import math
from typing import Type

import torch

from inference.core.workflows.core_steps.classical_cv.contrast_enhancement.v1 import (
    ContrastEnhancementManifest,
    enhance_contrast,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock


class ContrastEnhancementBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[ContrastEnhancementManifest]:
        return ContrastEnhancementManifest

    def run(
        self,
        image: WorkflowImageData,
        clip_limit: int,
        contrast_multiplier: float,
        normalize_brightness: bool,
        *args,
        **kwargs,
    ) -> BlockResult:
        if not image.is_tensor_materialised():
            return {
                "image": enhance_contrast(
                    image=image,
                    clip_limit=clip_limit,
                    contrast_multiplier=contrast_multiplier,
                    normalize_brightness=normalize_brightness,
                )
            }
        return {
            "image": _enhance_contrast_tensor(
                image=image,
                clip_limit=clip_limit,
                contrast_multiplier=contrast_multiplier,
                normalize_brightness=normalize_brightness,
            )
        }


def _enhance_contrast_tensor(
    image: WorkflowImageData,
    clip_limit: int,
    contrast_multiplier: float,
    normalize_brightness: bool,
) -> WorkflowImageData:
    """Device-resident mirror of ``v1.enhance_contrast`` for CHW tensors."""
    chw = image.tensor_image  # (C, H, W) uint8, C in {1, 3}
    clip_pct = float(clip_limit) / 100.0
    contrast_mult = float(contrast_multiplier)
    gamma_val = 1.0 / 1.3 if normalize_brightness else 1.0

    channels = int(chw.shape[0])
    flat = chw.detach().reshape(channels, -1).to(torch.float32)
    if clip_pct > 0:
        sorted_values, _ = torch.sort(flat, dim=1)
        v_min = _linear_percentile(sorted_values, clip_pct)
        v_max = _linear_percentile(sorted_values, 1.0 - clip_pct)
    else:
        v_min = flat.amin(dim=1)
        v_max = flat.amax(dim=1)

    valid = v_max > v_min  # (C,) - flat-histogram channels pass through raw
    if channels == 1 and not bool(valid.any()):
        # Numpy grayscale parity: nothing to stretch -> the input is returned.
        return image

    spread = torch.where(valid, v_max - v_min, torch.ones_like(v_max))
    normalized = (flat - v_min.unsqueeze(1)) / spread.unsqueeze(1) * 255.0
    if contrast_mult != 1.0:
        normalized = 128.0 + (normalized - 128.0) * contrast_mult
    if gamma_val != 1.0:
        normalized = normalized.clamp(0.0, 255.0)
        normalized = torch.pow(normalized / 255.0, gamma_val) * 255.0
    normalized = normalized.clamp(0.0, 255.0)
    enhanced = torch.where(valid.unsqueeze(1), normalized, flat)
    # Numpy parity: astype(np.uint8) truncates, and so does .to(torch.uint8).
    enhanced_chw = enhanced.to(torch.uint8).reshape(chw.shape)
    return WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        tensor_image=enhanced_chw,
    )


def _linear_percentile(sorted_values: torch.Tensor, quantile: float) -> torch.Tensor:
    """Per-row percentile of pre-sorted ``(C, N)`` values with numpy's default
    linear interpolation - equivalent to ``np.percentile(..., method="linear")``."""
    n = sorted_values.shape[1]
    position = quantile * (n - 1)
    lower = int(math.floor(position))
    upper = min(int(math.ceil(position)), n - 1)
    weight = position - lower
    return sorted_values[:, lower] * (1.0 - weight) + sorted_values[:, upper] * weight

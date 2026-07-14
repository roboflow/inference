"""Tensor-native sibling of ``dominant_color/v1``.

The k-means stays on the CPU in v1's ``find_dominant_color``: the clustered
array is tiny after downsampling, the convergence check would sync every
iteration, and the trajectory is coupled to unseeded global ``np.random``
draws. The tensor path only moves the strided downsample to the device, so
the small downsampled block crosses device->host instead of the full frame.

The host-side array is made byte-identical to v1's
``numpy_image[::scale_factor, ::scale_factor]`` (HWC, BGR) before the shared
k-means: float32 summation order in the distance computation depends on the
channel order, so RGB-ordered input could diverge the trajectory. Identical
bytes + identical RNG state => identical output; the RNG is unseeded, so
run-to-run output is nondeterministic on both paths.

Numpy/base64-born images delegate to the v1 numpy path (no materialised
tensor).
"""

from typing import Optional, Type

import numpy as np
import torch

from inference.core.workflows.core_steps.classical_cv.dominant_color.v1 import (
    DominantColorManifest,
    find_dominant_color,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock


class DominantColorBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[DominantColorManifest]:
        return DominantColorManifest

    def run(
        self,
        image: WorkflowImageData,
        color_clusters: Optional[int],
        max_iterations: Optional[int],
        target_size: Optional[int],
        *args,
        **kwargs
    ) -> BlockResult:
        if not image.is_tensor_materialised():
            np_image = image.numpy_image
            height, width = np_image.shape[:2]
            scale_factor = max(1, min(width, height) // target_size)
            downsampled = np_image[::scale_factor, ::scale_factor]
        else:
            chw = image.tensor_image
            height, width = int(chw.shape[-2]), int(chw.shape[-1])
            scale_factor = max(1, min(width, height) // target_size)
            downsampled = _downsample_to_bgr_numpy(chw=chw, scale_factor=scale_factor)
        rgb_color = find_dominant_color(
            pixels_image=downsampled,
            color_clusters=color_clusters,
            max_iterations=max_iterations,
        )
        return {"rgb_color": rgb_color}


def _downsample_to_bgr_numpy(chw: torch.Tensor, scale_factor: int) -> np.ndarray:
    """Strided downsample on the device, then one small D2H copy yielding an
    HWC BGR uint8 array byte-identical to v1's
    ``numpy_image[::scale_factor, ::scale_factor]``."""
    downsampled = chw.detach()[:, ::scale_factor, ::scale_factor]
    # CHW RGB -> HWC BGR; contiguous so only the downsampled block transfers.
    return downsampled.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()

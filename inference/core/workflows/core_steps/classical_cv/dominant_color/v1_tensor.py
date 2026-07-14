"""Tensor-native sibling of ``dominant_color/v1``.

The block's output is a single ``(r, g, b)`` tuple, not an image, and the
algorithm is an iterative k-means over a downsampled frame: unseeded global
``np.random`` initialisation, an empty-cluster reinit that also draws from the
global RNG, and a host-side convergence check every iteration. Porting the
loop to the device buys nothing - the clustered array is tiny (~100px min-dim
after downsampling, k <= 10), every iteration would sync for the convergence
check, and torch has no drop-in replica of the exact numpy arithmetic (and no
replica of the global-numpy-RNG draws the trajectory is coupled to). So the
clustering stays on the CPU in the SAME numpy code both paths share:
``find_dominant_color``, imported from v1.

What the tensor path DOES win is transfer volume: v1 materialises the full
frame on the host (megabytes for HD frames) only to immediately discard all
but every ``scale_factor``-th pixel. For a tensor-materialised image the
strided downsample runs on the device (a zero-copy strided view) and only the
small downsampled block (tens of KB) crosses to the host.

Trajectory-preservation subtlety: v1's k-means consumes pixels in HWC
row-major order with BGR channel order. Feeding RGB-ordered pixel vectors
instead would permute each coordinate triple - mathematically
distance-preserving, but the floating-point summation order in the distance
computation changes, which can flip argmin ties and diverge the trajectory.
The tensor path therefore flips the channel axis back to BGR and permutes to
HWC before the single D2H copy, making the host-side array BYTE-IDENTICAL to
v1's ``numpy_image[::scale_factor, ::scale_factor]``: identical bytes +
identical RNG state => identical trajectory => identical output.

Numpy/base64-born images delegate to the exact v1 numpy path instead of
forcing an eager host->device conversion - the same materialization-aware rule
the other tensor siblings follow. NOTE: v1 seeds nothing, so the block is
nondeterministic run-to-run on BOTH paths; parity is exact only under a pinned
global RNG (see the tests).
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
    """Strided downsample on the device, then one small D2H copy of an HWC BGR
    uint8 array byte-identical to v1's
    ``numpy_image[::scale_factor, ::scale_factor]`` (see the module docstring
    for why byte-identity - not mere colour equivalence - is required)."""
    downsampled = chw.detach()[:, ::scale_factor, ::scale_factor]
    # CHW RGB -> CHW BGR -> HWC BGR; contiguous on-device so the transfer
    # copies exactly the downsampled block, nothing more.
    return downsampled.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()

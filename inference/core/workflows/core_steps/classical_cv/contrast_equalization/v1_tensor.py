"""Tensor-native sibling of ``contrast_equalization/v1``.

'Contrast Stretching' and 'Histogram Equalization' are global value maps driven
by the image-wide 256-bin histogram (layout-independent: the same multiset of
values whether CHW/RGB or HWC/BGR). The tensor path runs ``torch.bincount`` on
the device, syncs the 256 counts to host to build a 256-entry LUT with the
identical numpy/skimage arithmetic of v1 (bit-exact result versus the numpy
block), and applies one device gather (``lut[image]``).

'Adaptive Equalization' (CLAHE) is position-dependent (tile-local mappings), so
it cannot be expressed as a value LUT and delegates to v1; numpy/base64-born
images delegate as well.
"""

import math
from typing import Type

import numpy as np
import torch

from inference.core.workflows.core_steps.classical_cv.contrast_equalization.v1 import (
    ContrastEqualizationManifest,
    update_image,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock


class ContrastEqualizationBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[ContrastEqualizationManifest]:
        return ContrastEqualizationManifest

    def run(
        self,
        image: WorkflowImageData,
        equalization_type: str,
    ) -> BlockResult:
        if (
            not image.is_tensor_materialised()
            or equalization_type == "Adaptive Equalization"
        ):
            updated_image = update_image(image.numpy_image, equalization_type)
            output = WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=updated_image,
            )
            return {OUTPUT_IMAGE_KEY: output}
        return {
            OUTPUT_IMAGE_KEY: _equalize_tensor(
                image=image, equalization_type=equalization_type
            )
        }


def _equalize_tensor(
    image: WorkflowImageData, equalization_type: str
) -> WorkflowImageData:
    if equalization_type == "Contrast Stretching":
        build_lut = _contrast_stretching_lut
    elif equalization_type == "Histogram Equalization":
        build_lut = _histogram_equalization_lut
    else:
        raise ValueError(
            f"contrast equalization type `{equalization_type}` not implemented!"
        )
    chw = image.tensor_image
    flat_indices = chw.detach().reshape(-1).long()
    counts = torch.bincount(flat_indices, minlength=256).cpu().numpy()
    lut = torch.from_numpy(build_lut(counts)).to(chw.device)
    equalized = lut[flat_indices].reshape(chw.shape)
    return WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        tensor_image=equalized,
    )


def _contrast_stretching_lut(counts: np.ndarray) -> np.ndarray:
    """256-entry LUT of ``exposure.rescale_intensity(img, in_range=(p2, p98))``
    for uint8 images, replicating skimage's float64 arithmetic exactly."""
    p2 = _percentile_of_uint8_counts(counts, 2.0)
    p98 = _percentile_of_uint8_counts(counts, 98.0)
    values = np.arange(256, dtype=np.uint8)
    clipped = np.clip(values, p2, p98)  # float64, matching skimage
    if p2 != p98:
        scaled = (clipped - p2) / (p98 - p2)
        return (scaled * 255.0).astype(np.uint8)
    return np.clip(clipped, 0.0, 255.0).astype(np.uint8)


def _histogram_equalization_lut(counts: np.ndarray) -> np.ndarray:
    """256-entry LUT of v1's ``equalize_hist(img.astype(float32) / 255) * 255``
    -> uint8 chain, replicating skimage's dtype handling exactly."""
    values = np.arange(256, dtype=np.float32) / 255  # v1's float32 division
    present = counts > 0
    # np.histogram derives bin edges from data min/max; min/max of the present
    # distinct values equal those of the full image, so the float32 bin edges
    # (and each value's bin) match skimage's histogram of all pixels.
    hist, bin_edges = np.histogram(values[present], bins=256, weights=counts[present])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    img_cdf = img_cdf.astype(np.float32)  # cumulative_distribution's cast
    equalized = np.interp(values, bin_centers, img_cdf)  # float64 output
    equalized = equalized.astype(np.float32)  # equalize_hist's cast
    return (equalized * 255).astype(np.uint8)


def _percentile_of_uint8_counts(counts: np.ndarray, q: float) -> float:
    """``np.percentile(values, q)`` (default linear method) computed from exact
    256-bin counts of the uint8 values, including numpy's ``_lerp`` fixup."""
    cumulative = np.cumsum(counts)
    n = int(cumulative[-1])
    position = (q / 100.0) * (n - 1)
    lower_rank = int(math.floor(position))
    fraction = position - lower_rank
    upper_rank = min(lower_rank + 1, n - 1)
    # k-th order statistic = smallest value whose cumulative count reaches k+1
    lower_value = float(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper_value = float(np.searchsorted(cumulative, upper_rank + 1, side="left"))
    difference = upper_value - lower_value
    if fraction >= 0.5:
        return upper_value - difference * (1.0 - fraction)
    return lower_value + difference * fraction

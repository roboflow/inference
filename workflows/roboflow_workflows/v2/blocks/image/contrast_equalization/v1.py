"""Equalize image contrast: ``roboflow_core/contrast_equalization@v1``.

Port of V1 ``roboflow_core/contrast_equalization@v1``, which used
scikit-image. All channels share one histogram of every pixel value:

======================  ==============================================  ======
equalization_type       effect (scikit-image function, as V1)           runs on
======================  ==============================================  ======
Contrast Stretching     values between the 2nd and 98th percentile      device
                        spread linearly to ``0..255``, the rest
                        clipped (``rescale_intensity``)
Histogram Equalization  values remapped so their histogram is about     device
                        flat (``equalize_hist``); the default
Adaptive Equalization   CLAHE: equalization in local tiles with         CPU
                        limited amplification, ``clip_limit=0.03``
                        (``equalize_adapthist``)
======================  ==============================================  ======

The two global types depend only on a pixel's value. The block counts the 256
values on the image's device, reads the counts back to the host (one wait on
a GPU), builds a 256-entry table with scikit-image's arithmetic and looks every
pixel up on the device. Results equal V1 exactly.

``Adaptive Equalization`` depends on each pixel's position, so it crosses to
the CPU and calls ``equalize_adapthist`` there in V1's NumPy layout
(``host.call_on_host``: BGR for three channels; scikit-image equalizes the
HSV brightness of a color image).
A one-channel image is equalized as a single plane.

The result is the same image with new pixels: identity, mappings, source and
time are kept, and the input tensor is not modified.
"""

import math
from typing import Any, Dict, Literal
import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.image.host import call_on_host
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from skimage import exposure


EqualizationType = Literal[
    "Contrast Stretching", "Histogram Equalization", "Adaptive Equalization"
]

EQUALIZATION_TYPES = (
    "Contrast Stretching",
    "Histogram Equalization",
    "Adaptive Equalization",
)

STRETCH_PERCENTILES = (2.0, 98.0)

ADAPTIVE_CLIP_LIMIT = 0.03

class ContrastEqualizationBlock(Block):
    """Equalize the contrast of one image; see the module docstring.

    Example: an image with one pixel of each value ``100..150`` spreads to
    ``5..255`` after ``"Histogram Equalization"``, as in V1. Each value moves
    to 255 times the share of pixels at or below it, so the darkest of the 51
    values becomes 255 / 51 = 5, not 0. ``"Contrast Stretching"`` reaches 0.
    """

    type = "roboflow_core/contrast_equalization@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="Equalized image with the input's identity and geometry.",
        )
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to equalize.")
        equalization_type: EqualizationType | Ref(STRING_KIND) = Field(
            default="Histogram Equalization",
            description=(
                "'Contrast Stretching' (2nd-98th percentile to full range), "
                "'Histogram Equalization' (global) or 'Adaptive Equalization' "
                "(local CLAHE, slowest)."
            ),
            examples=["Histogram Equalization", "$inputs.equalization_type"],
        )

    def run(self, *, image: ImageData, equalization_type: str) -> Dict[str, Any]:
        """Equalize ``image``.

        Args:
            image: Image to equalize.
            equalization_type: One of the types in the module docstring.

        Returns:
            ``image``: the equalized image.

        Raises:
            ValueError: On an unknown ``equalization_type``.
        """
        if equalization_type not in EQUALIZATION_TYPES:
            raise ValueError(
                f"equalization_type must be one of {EQUALIZATION_TYPES}, got "
                f"{equalization_type!r}"
            )

        pixels = image.tensor_image
        if equalization_type == "Adaptive Equalization":
            equalized = call_on_host(pixels, function=_equalize_adaptively)
        else:
            counts = torch.bincount(pixels.reshape(-1).long(), minlength=256)
            build_table = (
                stretching_table
                if equalization_type == "Contrast Stretching"
                else histogram_equalization_table
            )
            table = torch.from_numpy(build_table(counts.cpu().numpy()))
            equalized = table.to(pixels.device)[pixels.long()]
        result = {"image": image.with_pixels(equalized)}

        return result

def stretching_table(counts: np.ndarray) -> np.ndarray:
    """Map values like V1's ``rescale_intensity(img, in_range=(p2, p98))``.

    Args:
        counts: 256 pixel counts, one per value.

    Returns:
        256 ``uint8`` replacement values, in scikit-image's float64 arithmetic.
    """
    low, high = (
        _percentile_from_counts(counts, percent=p) for p in STRETCH_PERCENTILES
    )
    clipped = np.clip(np.arange(256, dtype=np.uint8), low, high)
    if low == high:
        table = np.clip(clipped, 0.0, 255.0).astype(np.uint8)
        return table

    table = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)

    return table

def histogram_equalization_table(counts: np.ndarray) -> np.ndarray:
    """Map values like V1's ``equalize_hist(img / 255) * 255`` chain.

    Args:
        counts: 256 pixel counts, one per value.

    Returns:
        256 ``uint8`` replacement values, with scikit-image's dtypes.
    """
    values = np.arange(256, dtype=np.float32) / 255
    present = counts > 0
    # np.histogram takes its edges from the data's minimum and maximum; the
    # distinct present values have the same ones as all pixels.
    histogram, edges = np.histogram(values[present], bins=256, weights=counts[present])
    centers = (edges[:-1] + edges[1:]) / 2.0
    cumulative = histogram.cumsum()
    cumulative = (cumulative / float(cumulative[-1])).astype(np.float32)
    equalized = np.interp(values, centers, cumulative).astype(np.float32)
    table = (equalized * 255).astype(np.uint8)

    return table

def _percentile_from_counts(counts: np.ndarray, *, percent: float) -> float:
    # np.percentile(values, percent) of the uint8 values with these counts,
    # including NumPy's interpolation from the nearer neighbour.
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])
    position = percent / 100.0 * (total - 1)
    lower_rank = int(math.floor(position))
    fraction = position - lower_rank
    upper_rank = min(lower_rank + 1, total - 1)
    lower = float(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper = float(np.searchsorted(cumulative, upper_rank + 1, side="left"))
    if fraction >= 0.5:
        percentile = upper - (upper - lower) * (1.0 - fraction)
        return percentile

    percentile = lower + (upper - lower) * fraction

    return percentile

def _equalize_adaptively(array: np.ndarray) -> np.ndarray:
    # V1's chain: scikit-image's CLAHE on a float32 copy scaled to 0..1.
    scaled = array.astype(np.float32) / 255
    equalized = exposure.equalize_adapthist(scaled, clip_limit=ADAPTIVE_CLIP_LIMIT)
    planes = (equalized * 255).astype(np.uint8)

    return planes

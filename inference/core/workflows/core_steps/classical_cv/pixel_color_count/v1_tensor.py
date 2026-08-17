"""Tensor-native sibling of ``pixel_color_count/v1``.

On a tensor-materialised ``(3, H, W)`` image the count is one broadcasted
``(px >= lower) & (px <= upper)`` on the device; only the final count crosses
device->host.

Parity with v1, which feeds ``cv2.inRange`` unclipped int64
``target +- tolerance`` bound arrays:

* ``cv2.inRange`` treats 3-element bounds as a per-channel Scalar: each bound
  is ``cvRound``-ed (half-to-even) and tested as the inclusive range
  ``lower <= px <= upper``, ANDed across channels, without uint8 saturation -
  a bound of -10 means "no lower limit", 310 "no upper limit", 256 matches
  nothing, inverted ranges (negative tolerance) match nothing. Reproduced
  with int16 device comparisons; bounds are pre-clamped to ``[-1, 256]``,
  which cannot change any comparison outcome against uint8 pixels but keeps
  them int16-safe at any magnitude. (Bounds beyond int32 wrap inside cv2 but
  not here; unreachable with the manifest's 0-255 tolerance.)
* Colour parsing uses v1's ``convert_color_to_bgr_tuple``, so malformed
  colours raise the identical ``ValueError`` before any image work. The
  per-channel AND is layout-independent, so the BGR bounds tuple is reversed
  to pair with the CHW RGB channel axis.
* Single-channel ``(1, H, W)`` tensors delegate: v1 raises ``cv2.error`` for
  3-element bounds against a 1-channel image, and delegation fails
  identically.
* Numpy/base64-born images delegate to v1's ``count_specific_color_pixels``
  (no materialised tensor).
"""

from typing import Tuple, Type, Union

import torch

from inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1 import (
    ColorPixelCountManifest,
    convert_color_to_bgr_tuple,
    count_specific_color_pixels,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock


class PixelationCountBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[ColorPixelCountManifest]:
        return ColorPixelCountManifest

    def run(
        self,
        image: WorkflowImageData,
        target_color: Union[str, tuple],
        tolerance: int,
    ) -> BlockResult:
        if image.is_tensor_materialised() and image.tensor_image.shape[0] == 3:
            color_pixel_count = _count_specific_color_pixels_tensor(
                image.tensor_image, target_color, tolerance
            )
        else:
            color_pixel_count = count_specific_color_pixels(
                image.numpy_image, target_color, tolerance
            )
        return {"matching_pixels_count": color_pixel_count}


def _count_specific_color_pixels_tensor(
    image: torch.Tensor,
    target_color: Union[str, Tuple[int, int, int]],
    tolerance: int,
) -> int:
    """Counts pixels of a CHW RGB uint8 tensor that match the target colour
    within tolerance, replicating ``cv2.inRange`` + ``cv2.countNonZero``
    exactly; only the final count leaves the device."""
    # Colour conversion first, before any image work - mirrors v1's error
    # ordering.
    target_color_bgr = convert_color_to_bgr_tuple(color=target_color)
    target_color_rgb = target_color_bgr[::-1]
    lower_bound = [
        _mirror_cv2_scalar_bound(channel - tolerance) for channel in target_color_rgb
    ]
    upper_bound = [
        _mirror_cv2_scalar_bound(channel + tolerance) for channel in target_color_rgb
    ]
    device = image.device
    lower = torch.tensor(lower_bound, dtype=torch.int16, device=device).view(3, 1, 1)
    upper = torch.tensor(upper_bound, dtype=torch.int16, device=device).view(3, 1, 1)
    pixels = image.detach().to(torch.int16)
    matching = ((pixels >= lower) & (pixels <= upper)).all(dim=0)
    return int(matching.sum().item())


def _mirror_cv2_scalar_bound(value: Union[int, float]) -> int:
    """cv2 converts each Scalar bound with ``cvRound`` (half-to-even, matching
    Python's ``round``) and compares in integer space without clamping.
    Clamping to ``[-1, 256]`` cannot change any comparison outcome against
    uint8 pixels while keeping the bound int16-safe at any magnitude."""
    return min(256, max(-1, round(value)))

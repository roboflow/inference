"""Tensor-native sibling of ``pixel_color_count/v1``.

On a tensor-materialised color or uint8 grayscale image the count is one broadcasted
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
* Single-channel ``(1, H, W)`` uint8 tensors with an RGB target use the
  intersection of the three channel ranges. Other grayscale dtypes and
  non-triplet target strings keep the OpenCV path.
* NumPy/base64-born images delegate to v1's ``count_specific_color_pixels``
  (no materialised tensor).
"""

from typing import Tuple, Type, Union

import torch
from roboflow_workflows.core_steps.classical_cv.pixel_color_count.v1 import (
    ColorPixelCountManifest,
    convert_color_to_bgr_tuple,
    count_specific_color_pixels,
)
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
from roboflow_workflows.prototypes.block import BlockResult, WorkflowBlock


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
        """Count matching pixels in a color or grayscale image.

        Args:
            image (WorkflowImageData): Image to analyze.
            target_color (Union[str, tuple]): Target RGB color.
            tolerance (int): Allowed difference in each color channel.

        Returns:
            BlockResult: The number of matching pixels.
        """
        tensor = image.tensor_image if image.is_tensor_materialised() else None
        if tensor is not None and (
            tensor.shape[0] == 3
            or (tensor.shape[0] == 1 and tensor.dtype == torch.uint8)
        ):
            target_color_bgr = convert_color_to_bgr_tuple(color=target_color)
            if tensor.shape[0] == 3 or len(target_color_bgr) == 3:
                color_pixel_count = _count_specific_color_pixels_tensor(
                    tensor, target_color_bgr, tolerance
                )
                return {"matching_pixels_count": color_pixel_count}

        color_pixel_count = count_specific_color_pixels(
            image.numpy_image, target_color, tolerance
        )
        return {"matching_pixels_count": color_pixel_count}


def _count_specific_color_pixels_tensor(
    image: torch.Tensor,
    target_color_bgr: Tuple[int, int, int],
    tolerance: int,
) -> int:
    """Counts pixels of a CHW RGB or grayscale uint8 tensor that match the target colour
    within tolerance, replicating ``cv2.inRange`` + ``cv2.countNonZero``
    exactly; only the final count leaves the device."""
    target_color_rgb = target_color_bgr[::-1]
    lower_bound = [
        _mirror_cv2_scalar_bound(channel - tolerance) for channel in target_color_rgb
    ]
    upper_bound = [
        _mirror_cv2_scalar_bound(channel + tolerance) for channel in target_color_rgb
    ]

    if image.shape[0] == 1:
        lower, upper = max(lower_bound), min(upper_bound)
    else:
        device = image.device
        lower = torch.tensor(lower_bound, dtype=torch.int16, device=device).view(
            3, 1, 1
        )
        upper = torch.tensor(upper_bound, dtype=torch.int16, device=device).view(
            3, 1, 1
        )

    pixels = image.detach().to(torch.int16)
    matching = ((pixels >= lower) & (pixels <= upper)).all(dim=0)
    count = int(matching.sum().item())

    return count


def _mirror_cv2_scalar_bound(value: Union[int, float]) -> int:
    """cv2 converts each Scalar bound with ``cvRound`` (half-to-even, matching
    Python's ``round``) and compares in integer space without clamping.
    Clamping to ``[-1, 256]`` cannot change any comparison outcome against
    uint8 pixels while keeping the bound int16-safe at any magnitude."""
    return min(256, max(-1, round(value)))

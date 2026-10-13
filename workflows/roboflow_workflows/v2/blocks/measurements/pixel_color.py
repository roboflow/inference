"""Count pixels near one colour: ``v2/pixel_color_count@v1``.

Port of V1 ``roboflow_core/pixel_color_count@v1`` (latest tensor version). A
pixel matches when every channel is within ``tolerance`` of the target::

    lower = target - tolerance;  upper = target + tolerance     per channel
    count = number of pixels with lower <= pixel <= upper in R, G and B

This equals V1's ``cv2.inRange`` + ``countNonZero``: bounds are not clipped
to 0..255, so a bound below 0 or above 255 is simply no limit, and a negative
tolerance matches nothing. The whole image is counted; crop first to count a
region. A crop or resized image is counted in its own pixels.

Colours: ``#RRGGBB``, ``#RGB``, ``(R, G, B)`` text, or three integers
``(R, G, B)`` / ``[R, G, B]``. All are RGB, as in V1.

The comparison runs on the image's device as int16; only the final count is
copied to the host.

Differences from V1:

* A colour may also be a list of three integers, as JSON workflow inputs
  deliver them; V1 accepted only a tuple.
* A ``(R, G, B)`` text with other than three values raises a clear
  ``ValueError``; V1 failed later inside OpenCV.
* A one-channel image raises ``ValueError``; V1 failed inside OpenCV.
"""

from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from pydantic import Field
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

__all__ = ["PixelColorCountBlock", "parse_rgb_color"]

ColorInput = Union[str, Tuple[int, int, int], List[int]]


class PixelColorCountBlock(Block):
    """Count the pixels whose colour is within a tolerance of a target colour.

    Example: an image with a 10 x 20 patch of ``#FF0000`` and noise elsewhere
    counts 200 pixels for target ``#FF0000`` with tolerance 0.
    """

    type = "v2/pixel_color_count@v1"
    prunable = True
    metadata = {
        "name": "Pixel Color Count",
        "port_of": "roboflow_core/pixel_color_count@v1",
        "section": "classical_cv",
    }
    outputs = {
        "matching_pixels_count": Output(
            INTEGER_KIND,
            source="image",
            description="Number of pixels within tolerance of the target colour.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image whose pixels are counted, all of them.",
            examples=["$inputs.image", "$steps.crop.crops"],
        )
        target_color: (
            str
            | Tuple[int, int, int]
            | List[int]
            | Ref(STRING_KIND, LIST_OF_VALUES_KIND)
        ) = Field(
            description="RGB target: '#RRGGBB', '#RGB', '(R, G, B)' or three "
            "integers.",
            examples=["#431112", "(18, 17, 67)", "$inputs.target_color"],
        )
        tolerance: int | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Allowed difference per channel, inclusive; 10 accepts "
            "21 values per channel.",
            examples=[10, "$inputs.tolerance"],
        )

    def run(
        self, *, image: ImageData, target_color: ColorInput, tolerance: int
    ) -> Dict[str, Any]:
        """Count the matching pixels.

        Args:
            image: RGB image.
            target_color: Target colour.
            tolerance: Allowed difference per channel.

        Returns:
            ``matching_pixels_count``.

        Raises:
            ValueError: On an invalid colour or a one-channel image.
        """
        rgb = parse_rgb_color(target_color)
        if image.channels != 3:
            raise ValueError(
                f"v2/pixel_color_count@v1 counts RGB pixels; the image has "
                f"{image.channels} channel(s)"
            )

        count = _count_in_range(image.tensor_image, rgb=rgb, tolerance=tolerance)
        result = {"matching_pixels_count": count}

        return result


def parse_rgb_color(color: Any) -> Tuple[int, int, int]:
    """Read a V1 pixel-count colour as an RGB triple.

    Args:
        color: ``#RRGGBB``, ``#RGB``, ``(R, G, B)`` text, or a tuple or list
            of three integers.

    Returns:
        ``(red, green, blue)``. Values are not clipped to 0..255.

    Raises:
        ValueError: When ``color`` is in none of these forms.
    """
    if isinstance(color, str):
        rgb = _parse_color_text(color)
    elif isinstance(color, (tuple, list)) and len(color) == 3:
        rgb = _three_integers(color, text=color)
    else:
        raise ValueError(f"Invalid color format: {color}")

    return rgb


def _parse_color_text(color: str) -> Tuple[int, int, int]:
    if color.startswith("#") and len(color) in (4, 7):
        digits = color[1:] if len(color) == 7 else "".join(2 * c for c in color[1:])
        try:
            rgb = tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError as error:
            raise ValueError(f"Invalid hex color format: {color}") from error
        return rgb

    if color.startswith("(") and color.endswith(")"):
        parts = color[1:-1].split(",")
        try:
            values = [int(part) for part in parts]
        except ValueError as error:
            raise ValueError(f"Invalid tuple color format: {color}") from error
        rgb = _three_integers(values, text=color)
        return rgb

    raise ValueError(f"Invalid color format: {color}")


def _three_integers(values: Sequence[Any], *, text: Any) -> Tuple[int, int, int]:
    if len(values) != 3 or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        raise ValueError(
            f"Invalid color format: {text}; expected three integers R, G, B"
        )

    rgb = (values[0], values[1], values[2])

    return rgb


def _count_in_range(
    pixels: torch.Tensor, *, rgb: Tuple[int, int, int], tolerance: int
) -> int:
    lower = [_clamp_bound(channel - tolerance) for channel in rgb]
    upper = [_clamp_bound(channel + tolerance) for channel in rgb]
    device = pixels.device
    lower_bounds = torch.tensor(lower, dtype=torch.int16, device=device).view(3, 1, 1)
    upper_bounds = torch.tensor(upper, dtype=torch.int16, device=device).view(3, 1, 1)
    values = pixels.to(dtype=torch.int16)
    matching = ((values >= lower_bounds) & (values <= upper_bounds)).all(dim=0)
    count = int(matching.sum().item())

    return count


def _clamp_bound(value: int) -> int:
    # -1 and 256 already exclude nothing from a uint8 pixel; clamping keeps
    # any bound int16-safe without changing a comparison.
    clamped = min(256, max(-1, value))

    return clamped

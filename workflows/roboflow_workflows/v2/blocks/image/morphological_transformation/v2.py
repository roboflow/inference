"""Morphological transformations: ``roboflow_core/morphological_transformation@v2``.

Port of V1 ``roboflow_core/morphological_transformation@v2``, the latest V1
version. A square ``kernel_size`` x ``kernel_size`` window slides over every
channel; erosion takes the window minimum and dilation its maximum::

    Erosion               min
    Dilation              max
    Opening               dilate(erode(x))          removes small bright spots
    Closing               erode(dilate(x))          fills small dark holes
    Opening then Closing  close(open(x))            the default
    Gradient              dilate(x) - erode(x)      outlines
    Top Hat               x - open(x)               small bright details
    Black Hat             close(x) - x              small dark details

Differences stop at 0, as in OpenCV: with an even ``kernel_size`` an opening
can be brighter than the original pixel.

Pixels equal OpenCV's ``erode``, ``dilate`` and ``morphologyEx`` with a
rectangular kernel exactly, as used by V1. Outside the image nothing
counts: the window covers only the pixels inside. For an even
``kernel_size`` the window reaches ``kernel_size // 2`` pixels up and left of
its pixel and one fewer down and right (OpenCV's default anchor).

A CPU tensor gets V1's OpenCV calls themselves (``host.call_on_host``). A
tensor on another device stays there: PyTorch takes the window minima and
maxima in ``uint8``, equal to OpenCV.

Like V1 the result has three channels: a one-channel image is first copied
into three equal channels, so later color blocks accept it. Otherwise the
result is the same image with new pixels: identity, mappings, source and time
are kept, and the input tensor is not modified.

Difference from V1: ``kernel_size`` must be at least 1. V1 passed 0 to
OpenCV, which silently used a 3 x 3 kernel, and failed inside NumPy on
negative sizes.
"""

from typing import Any, Callable, Dict, Literal
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field, StrictInt
from roboflow_workflows.v2.blocks.image.host import call_on_host
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


Operation = Literal[
    "Erosion",
    "Dilation",
    "Opening",
    "Closing",
    "Opening then Closing",
    "Gradient",
    "Top Hat",
    "Black Hat",
]

class MorphologicalTransformationBlock(Block):
    """Erode, dilate, open, close or derive details of one image.

    See the module docstring for each operation. Example: a single white
    pixel on black disappears with ``"Opening"`` and ``kernel_size=3``, and
    grows into a 3 x 3 white square with ``"Dilation"``.
    """

    type = "roboflow_core/morphological_transformation@v2"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description=(
                "Three-channel transformed image with the input's identity "
                "and geometry."
            ),
        )
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image to transform; one channel is copied to three."
        )
        kernel_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=5,
            ge=1,
            description=(
                "Side of the square window in pixels; larger windows remove or "
                "fill larger features."
            ),
            examples=[5, "$inputs.kernel_size"],
        )
        operation: Operation | Ref(STRING_KIND) = Field(
            default="Opening then Closing",
            description=(
                "Erosion, Dilation, Opening, Closing, Opening then Closing, "
                "Gradient, Top Hat or Black Hat."
            ),
            examples=["Closing", "$inputs.operation"],
        )

    def run(
        self, *, image: ImageData, kernel_size: int, operation: str
    ) -> Dict[str, Any]:
        """Transform ``image``.

        Args:
            image: Image to transform.
            kernel_size: Side of the square window, at least 1.
            operation: One of the operations in the module docstring.

        Returns:
            ``image``: the transformed three-channel image.

        Raises:
            ValueError: On an unknown operation or a ``kernel_size`` below 1.
        """
        pixels = image.tensor_image
        if image.channels == 1:
            pixels = pixels.expand(3, -1, -1)

        transformed = morphology(pixels, operation=operation, kernel_size=kernel_size)
        result = {"image": image.with_pixels(transformed)}

        return result

def morphology(
    pixels: torch.Tensor, *, operation: str, kernel_size: int
) -> torch.Tensor:
    """Apply a morphological operation with a square window to every channel.

    Args:
        pixels: ``(channels, height, width)`` ``uint8`` tensor; not modified.
        operation: One of the operations in the module docstring.
        kernel_size: Side of the square window, at least 1.

    Returns:
        A new tensor of the same shape on the same device.

    Raises:
        ValueError: On an unknown operation or a ``kernel_size`` below 1.
    """
    if operation not in OPERATIONS:
        raise ValueError(
            f"operation must be one of {list(OPERATIONS)}, got {operation!r}"
        )
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be at least 1, got {kernel_size}")

    if pixels.device.type == "cpu":
        transformed = call_on_host(
            pixels,
            function=lambda array: _opencv_morphology(
                array, operation=operation, kernel_size=kernel_size
            ),
        )
    else:
        transformed = OPERATIONS[operation](pixels, kernel_size)

    return transformed

def erode(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Take the minimum of each ``kernel_size`` square window.

    Args:
        pixels: ``(channels, height, width)`` ``uint8`` tensor.
        kernel_size: Side of the window.

    Returns:
        A new tensor of the same shape.
    """
    # 255 outside the image never lowers a minimum.
    eroded = _window_extreme(
        pixels, kernel_size=kernel_size, outside=255, take=torch.amin
    )

    return eroded

def dilate(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Take the maximum of each ``kernel_size`` square window.

    Args:
        pixels: ``(channels, height, width)`` ``uint8`` tensor.
        kernel_size: Side of the window.

    Returns:
        A new tensor of the same shape.
    """
    # 0 outside the image never raises a maximum.
    dilated = _window_extreme(
        pixels, kernel_size=kernel_size, outside=0, take=torch.amax
    )

    return dilated

def _open(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    opened = dilate(erode(pixels, kernel_size), kernel_size)

    return opened

def _close(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    closed = erode(dilate(pixels, kernel_size), kernel_size)

    return closed

def _open_then_close(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    smoothed = _close(_open(pixels, kernel_size), kernel_size)

    return smoothed

def _gradient(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    outline = _subtract_to_zero(dilate(pixels, kernel_size), erode(pixels, kernel_size))

    return outline

def _top_hat(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    bright_details = _subtract_to_zero(pixels, _open(pixels, kernel_size))

    return bright_details

def _black_hat(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    dark_details = _subtract_to_zero(_close(pixels, kernel_size), pixels)

    return dark_details

def _subtract_to_zero(minuend: torch.Tensor, subtrahend: torch.Tensor) -> torch.Tensor:
    # max(a - b, 0) for uint8 without wrapping around.
    difference = minuend - torch.minimum(minuend, subtrahend)

    return difference

OPERATIONS: Dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "Erosion": erode,
    "Dilation": dilate,
    "Opening": _open,
    "Closing": _close,
    "Opening then Closing": _open_then_close,
    "Gradient": _gradient,
    "Top Hat": _top_hat,
    "Black Hat": _black_hat,
}

def _opencv_morphology(
    array: np.ndarray, *, operation: str, kernel_size: int
) -> np.ndarray:
    # V1's calls: a square kernel of ones and OpenCV's default border.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "Erosion":
        transformed = cv2.erode(array, kernel, iterations=1)
    elif operation == "Dilation":
        transformed = cv2.dilate(array, kernel, iterations=1)
    elif operation == "Opening then Closing":
        opened = cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel)
        transformed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    else:
        transformed = cv2.morphologyEx(array, OPENCV_OPERATIONS[operation], kernel)

    return transformed

OPENCV_OPERATIONS = {
    "Opening": cv2.MORPH_OPEN,
    "Closing": cv2.MORPH_CLOSE,
    "Gradient": cv2.MORPH_GRADIENT,
    "Top Hat": cv2.MORPH_TOPHAT,
    "Black Hat": cv2.MORPH_BLACKHAT,
}

def _window_extreme(
    pixels: torch.Tensor,
    *,
    kernel_size: int,
    outside: int,
    take: Callable[..., torch.Tensor],
) -> torch.Tensor:
    # A square window is a row window followed by a column window. OpenCV's
    # anchor sits at kernel_size // 2, so even windows reach one pixel
    # further up and left. unfold makes views; the reduction reads them.
    before = kernel_size // 2
    after = kernel_size - 1 - before
    padded = F.pad(pixels, (before, after, before, after), value=outside)
    along_rows = take(padded.unfold(2, kernel_size, 1), dim=-1)
    extreme = take(along_rows.unfold(1, kernel_size, 1), dim=-1)

    return extreme

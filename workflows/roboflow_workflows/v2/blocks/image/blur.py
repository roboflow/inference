"""Blur an image: ``v2/image_blur@v1``.

Port of V1 ``roboflow_core/image_blur@v1``. Every blur equals the OpenCV call
V1 made, pixel for pixel. Common settings run in PyTorch on the image's
device; the rest cross to the CPU and call that same OpenCV function there
(``opencv_on_host``):

=========  ==========================  ======================  ==================
blur_type  OpenCV call (as V1)         on the image's device   on the CPU
=========  ==========================  ======================  ==================
average    ``blur``, k x k             odd k <= 15             even or larger k
gaussian   ``GaussianBlur``, sigma 0   k in 1, 3, 5, 7         k >= 9
median     ``medianBlur``              k <= 15                 k > 15
bilateral  ``bilateralFilter``, 75/75  never                   always
=========  ==========================  ======================  ==================

``gaussian`` and ``median`` first round ``kernel_size`` up to a positive odd
number (``0 -> 1``, ``4 -> 5``), as V1 did. Borders follow OpenCV: mirrored
without repeating the edge pixel for ``average`` and ``gaussian``, the edge
pixel repeated for ``median``. A kernel whose half width reaches the image's
height or width also runs on the CPU.

Why these limits: within them the PyTorch arithmetic is exact. Window sums
stay below ``2**24``, odd windows have no ``.5`` ties, and OpenCV's small
Gaussian kernels are fixed dyadic fractions. Beyond them OpenCV's own rounding
or algorithm choice cannot be reproduced exactly.

The result is the same image with new pixels: identity, mappings, source and
time are kept, and the input tensor is not modified. Gray and RGB images both
work; every blur treats channels independently or symmetrically.
"""

from typing import Any, Callable, Dict, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

BlurType = Literal["average", "gaussian", "median", "bilateral"]

BLUR_TYPES = ("average", "gaussian", "median", "bilateral")
BILATERAL_SIGMA = 75
MAX_DEVICE_AVERAGE_KERNEL = 15
MAX_DEVICE_MEDIAN_KERNEL = 15
# OpenCV's fixed Gaussian kernels for sigma 0 and size <= 7, in 1/256 units.
SMALL_GAUSSIAN_KERNELS = {
    1: (256,),
    3: (64, 128, 64),
    5: (16, 64, 96, 64, 16),
    7: (8, 28, 56, 72, 56, 28, 8),
}
# Upper bound of float32 elements of one chunk of median windows (~64 MB).
MEDIAN_CHUNK_ELEMENTS = 2**24


class ImageBlurBlock(Block):
    """Blur one image with an average, Gaussian, median or bilateral filter.

    See the module docstring for where each setting runs. Example: a white
    pixel (255) alone on black becomes 10 at its centre with
    ``blur_type="average"`` and ``kernel_size=5`` (255 / 25, rounded).
    """

    type = "v2/image_blur@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="Blurred image with the input's identity and geometry.",
        )
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to blur.")
        blur_type: BlurType | Ref(STRING_KIND) = Field(
            default="gaussian",
            description=(
                "Filter: 'average' (box mean), 'gaussian', 'median' (removes "
                "salt-and-pepper noise) or 'bilateral' (keeps edges, slowest)."
            ),
            examples=["gaussian", "$inputs.blur_type"],
        )
        kernel_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=5,
            description=(
                "Window size in pixels. 'gaussian' and 'median' round it up to "
                "an odd number; 'average' needs at least 1; 'bilateral' uses it "
                "as the pixel neighbourhood diameter."
            ),
            examples=[5, "$inputs.kernel_size"],
        )

    def run(
        self, *, image: ImageData, blur_type: str, kernel_size: int
    ) -> Dict[str, Any]:
        """Blur ``image``.

        Args:
            image: Image to blur.
            blur_type: ``"average"``, ``"gaussian"``, ``"median"`` or
                ``"bilateral"``.
            kernel_size: Window size in pixels.

        Returns:
            ``image``: the blurred image.

        Raises:
            ValueError: On an unknown ``blur_type``, or ``average`` with a
                ``kernel_size`` below 1.
        """
        blurred = blur(image.tensor_image, blur_type=blur_type, kernel_size=kernel_size)
        result = {"image": image.with_pixels(blurred)}

        return result


def blur(pixels: torch.Tensor, *, blur_type: str, kernel_size: int) -> torch.Tensor:
    """Blur ``(channels, height, width)`` ``uint8`` pixels like V1's OpenCV call.

    Args:
        pixels: Image tensor on any device; not modified.
        blur_type: ``"average"``, ``"gaussian"``, ``"median"`` or
            ``"bilateral"``.
        kernel_size: Window size in pixels.

    Returns:
        A new tensor of the same shape on the same device.

    Raises:
        ValueError: On an unknown ``blur_type``, or ``average`` with a
            ``kernel_size`` below 1.
    """
    if blur_type not in BLUR_TYPES:
        raise ValueError(f"blur_type must be one of {BLUR_TYPES}, got {blur_type!r}")
    if blur_type == "average" and kernel_size < 1:
        raise ValueError(f"average blur needs kernel_size >= 1, got {kernel_size}")

    if blur_type in ("gaussian", "median"):
        kernel_size = positive_odd(kernel_size)
    if not _runs_on_device(pixels, blur_type=blur_type, kernel_size=kernel_size):
        blurred = opencv_on_host(
            pixels,
            function=lambda array: _opencv_blur(
                array, blur_type=blur_type, kernel_size=kernel_size
            ),
        )
    elif blur_type == "average":
        blurred = _average(pixels, kernel_size=kernel_size)
    elif blur_type == "gaussian":
        blurred = _gaussian(pixels, kernel_size=kernel_size)
    else:
        blurred = _median(pixels, kernel_size=kernel_size)

    return blurred


def positive_odd(kernel_size: int) -> int:
    """Round a kernel size up to a positive odd number, as V1 did.

    Args:
        kernel_size: Requested size; values below 1 count as 1.

    Returns:
        ``kernel_size`` if it is positive and odd, otherwise the next odd
        number (at least 1).
    """
    size = max(1, kernel_size)
    if size % 2 == 0:
        size += 1

    return size


def opencv_on_host(
    pixels: torch.Tensor, *, function: Callable[[np.ndarray], np.ndarray]
) -> torch.Tensor:
    """Run an OpenCV reference function on the CPU, in V1's NumPy layout.

    The explicit CPU boundary of this module. The tensor is copied to host
    memory as ``(height, width)`` for one channel or ``(height, width, 3)`` in
    BGR order for three, so the call sees exactly what V1 passed to OpenCV.
    The result returns to the input's device in ``(channels, height, width)``
    RGB layout. On CUDA, the copy to the host waits for queued work.

    Args:
        pixels: ``(channels, height, width)`` ``uint8`` tensor; not modified.
        function: Maps the host array to a new array of the same layout.

    Returns:
        A new tensor of the same shape on the same device.
    """
    host = pixels.detach().cpu()
    if host.shape[0] == 3:
        host = host.flip(0)
    array = np.ascontiguousarray(host.permute(1, 2, 0).numpy())
    if array.shape[2] == 1:
        array = array[:, :, 0]

    processed = function(array).reshape(pixels.shape[1], pixels.shape[2], -1)
    restored = torch.from_numpy(np.ascontiguousarray(processed)).permute(2, 0, 1)
    if restored.shape[0] == 3:
        restored = restored.flip(0)
    on_device = restored.contiguous().to(pixels.device)

    return on_device


def _runs_on_device(pixels: torch.Tensor, *, blur_type: str, kernel_size: int) -> bool:
    if blur_type == "average":
        supported = kernel_size % 2 == 1 and kernel_size <= MAX_DEVICE_AVERAGE_KERNEL
    elif blur_type == "gaussian":
        supported = kernel_size in SMALL_GAUSSIAN_KERNELS
    elif blur_type == "median":
        supported = kernel_size <= MAX_DEVICE_MEDIAN_KERNEL
    else:
        supported = False
    # Beyond this OpenCV mirrors the border repeatedly; F.pad refuses.
    fits = kernel_size // 2 < min(int(pixels.shape[1]), int(pixels.shape[2]))

    return supported and fits


def _opencv_blur(array: np.ndarray, *, blur_type: str, kernel_size: int) -> np.ndarray:
    if blur_type == "average":
        blurred = cv2.blur(array, (kernel_size, kernel_size))
    elif blur_type == "gaussian":
        blurred = cv2.GaussianBlur(array, (kernel_size, kernel_size), 0)
    elif blur_type == "median":
        blurred = cv2.medianBlur(array, kernel_size)
    else:
        blurred = cv2.bilateralFilter(
            array, kernel_size, BILATERAL_SIGMA, BILATERAL_SIGMA
        )

    return blurred


def _average(pixels: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    # Exact integer window sums in float32; sum / k^2 never ties for odd k.
    pad = kernel_size // 2
    planes = F.pad(
        pixels.to(torch.float32).unsqueeze(1), (pad, pad, pad, pad), mode="reflect"
    )
    window = torch.ones(
        (1, 1, kernel_size, kernel_size), dtype=torch.float32, device=pixels.device
    )
    sums = F.conv2d(planes, window)
    averaged = torch.round(sums / (kernel_size * kernel_size))
    blurred = averaged.to(torch.uint8).squeeze(1)

    return blurred


def _gaussian(pixels: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    # OpenCV's fixed-point pipeline: a horizontal pass in 1/256 units, a
    # vertical pass in 1/65536 units, then half-up rounding; all exact.
    kernel = torch.tensor(
        SMALL_GAUSSIAN_KERNELS[kernel_size], dtype=torch.float32, device=pixels.device
    )
    pad = kernel_size // 2
    planes = F.pad(
        pixels.to(torch.float32).unsqueeze(1), (pad, pad, pad, pad), mode="reflect"
    )
    planes = F.conv2d(planes, kernel.view(1, 1, 1, kernel_size))
    planes = F.conv2d(planes, kernel.view(1, 1, kernel_size, 1))
    blurred = ((planes + 32768.0) / 65536.0).floor().to(torch.uint8).squeeze(1)

    return blurred


def _median(pixels: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    # The median of an odd window is one of its values: replicate the edges,
    # sort each window and take the middle, a chunk of rows at a time.
    pad = kernel_size // 2
    channels, height, width = (int(size) for size in pixels.shape)
    padded = F.pad(
        pixels.to(torch.float32).unsqueeze(0), (pad, pad, pad, pad), mode="replicate"
    ).squeeze(0)
    window_size = kernel_size * kernel_size
    rows_per_chunk = max(1, MEDIAN_CHUNK_ELEMENTS // (channels * width * window_size))

    chunks = []
    for first_row in range(0, height, rows_per_chunk):
        last_row = min(height, first_row + rows_per_chunk)
        windows = (
            padded[:, first_row : last_row + 2 * pad, :]
            .unfold(1, kernel_size, 1)
            .unfold(2, kernel_size, 1)
            .reshape(channels, last_row - first_row, width, window_size)
        )
        chunks.append(windows.sort(dim=-1).values[..., window_size // 2])
    blurred = torch.cat(chunks, dim=1).to(torch.uint8)

    return blurred

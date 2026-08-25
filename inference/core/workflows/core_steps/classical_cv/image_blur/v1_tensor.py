"""Tensor-native sibling of ``image_blur/v1``.

All tensor paths are channel-independent, so RGB (tensor layout) vs BGR
(numpy layout) is irrelevant and grayscale ``(1, H, W)`` tensors run as a
one-channel batch. Bit parity with cv2 per blur type:

- ``average`` (odd kernel <= 15): ``cv2.blur`` window sums over REFLECT_101
  borders are exact in a float32 convolution (255 * 15^2 < 2^24); k odd ->
  k^2 odd -> ``sum / k^2`` has no .5 ties, so nearest-integer rounding is
  unambiguous. Even kernels delegate: their exact .5 ties are rounded
  differently across cv2 SIMD backends/builds.

- ``gaussian`` (coerced-odd kernel in {1, 3, 5, 7}): cv2's uint8 GaussianBlur
  with sigma=0 and kernel <= 7 uses the hardcoded ``small_gaussian_tab``
  Q8.8 kernels - exact dyadic rationals - so the separable fixed-point
  pipeline (horizontal pass exact in uint16, vertical sums < 2^24, final
  rounding ``(acc + 2^15) >> 16``) is replicated with float32 convolutions.
  Kernels >= 9 delegate: cv2 derives their Q8.8 coefficients with softdouble
  arithmetic that a float64 re-derivation diverges from.

- ``median`` (coerced-odd kernel <= 15): ``cv2.medianBlur`` uses replicate
  borders, and the median of an odd-count uint8 window is a pure integer
  order statistic - replicate-pad + unfold + sort + middle element matches
  bit-for-bit. Windows are materialised in row chunks to bound the k^2-fold
  unfold blow-up to ~2^24 float32 elements (~64 MB) per chunk. Kernels > 15
  delegate (cv2 switches algorithms; unfold cost grows quadratically).

- ``bilateral``: always delegates - data-dependent float weights built from
  exp lookup tables cannot be replicated bit-exactly.

Numpy/base64-born images delegate to v1 (no materialised tensor). Unknown
blur types delegate so the exact v1 ``ValueError`` is raised. Tensor paths
also require the border pad (k // 2) to be smaller than both spatial dims:
beyond that cv2 multi-reflects while ``F.pad(mode="reflect")`` refuses, so
that regime delegates.
"""

from typing import Type

import torch
import torch.nn.functional as F

from inference.core.workflows.core_steps.classical_cv.image_blur.v1 import (
    ImageBlurManifest,
    _to_positive_odd,
    apply_blur,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock

# OpenCV small_gaussian_tab (sigma=0, ksize<=7) in Q8.8 fixed point - all
# coefficients are exact dyadic rationals, each row sums to 256.
_SMALL_GAUSSIAN_KERNELS_Q8_8 = {
    1: (256,),
    3: (64, 128, 64),
    5: (16, 64, 96, 64, 16),
    7: (8, 28, 56, 72, 56, 28, 8),
}
_MAX_TENSOR_AVERAGE_KSIZE = 15
_MAX_TENSOR_MEDIAN_KSIZE = 15
# Max unfolded float32 window elements per median chunk (floor: one output row).
_MEDIAN_UNFOLD_ELEMENT_BUDGET = 2**24


class ImageBlurBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[ImageBlurManifest]:
        return ImageBlurManifest

    def run(
        self,
        image: WorkflowImageData,
        blur_type: str,
        kernel_size: int,
        *args,
        **kwargs,
    ) -> BlockResult:
        if not image.is_tensor_materialised() or _requires_numpy_delegation(
            blur_type=blur_type,
            kernel_size=kernel_size,
            tensor_image=image.tensor_image,
        ):
            blurred_image = apply_blur(image.numpy_image, blur_type, kernel_size)
            output = WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=blurred_image,
            )
            return {OUTPUT_IMAGE_KEY: output}
        blurred_tensor = _blur_tensor(
            chw=image.tensor_image, blur_type=blur_type, kernel_size=kernel_size
        )
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            tensor_image=blurred_tensor,
        )
        return {OUTPUT_IMAGE_KEY: output}


def _requires_numpy_delegation(
    blur_type: str, kernel_size: int, tensor_image: torch.Tensor
) -> bool:
    """True for every (type, ksize, shape) regime whose bit parity with cv2
    cannot be guaranteed on the tensor path - see the module docstring."""
    if blur_type == "average":
        ksize = int(kernel_size)
        if ksize < 1 or ksize % 2 == 0 or ksize > _MAX_TENSOR_AVERAGE_KSIZE:
            return True
    elif blur_type == "gaussian":
        ksize = _to_positive_odd(kernel_size)
        if ksize not in _SMALL_GAUSSIAN_KERNELS_Q8_8:
            return True
    elif blur_type == "median":
        ksize = _to_positive_odd(kernel_size)
        if ksize > _MAX_TENSOR_MEDIAN_KSIZE:
            return True
    else:
        # bilateral and unknown blur types; the latter raise v1's exact
        # ValueError inside apply_blur().
        return True
    pad = ksize // 2
    return pad >= min(tensor_image.shape[-2], tensor_image.shape[-1])


def _blur_tensor(chw: torch.Tensor, blur_type: str, kernel_size: int) -> torch.Tensor:
    if blur_type == "average":
        return _average_blur_tensor(chw=chw, ksize=int(kernel_size))
    if blur_type == "gaussian":
        return _gaussian_blur_tensor(chw=chw, ksize=_to_positive_odd(kernel_size))
    return _median_blur_tensor(chw=chw, ksize=_to_positive_odd(kernel_size))


def _average_blur_tensor(chw: torch.Tensor, ksize: int) -> torch.Tensor:
    """``cv2.blur`` for ODD ksize: exact integer window sums over REFLECT_101
    borders, then unambiguous nearest-integer rounding of ``sum / k^2``."""
    pad = ksize // 2
    x = chw.detach().to(torch.float32).unsqueeze(1)  # (C, 1, H, W)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    weight = torch.ones((1, 1, ksize, ksize), dtype=torch.float32, device=x.device)
    sums = F.conv2d(x, weight)  # integer-valued float32, exact (< 2^24)
    return torch.round(sums * (1.0 / (ksize * ksize))).to(torch.uint8).squeeze(1)


def _gaussian_blur_tensor(chw: torch.Tensor, ksize: int) -> torch.Tensor:
    """OpenCV's bit-exact separable Q8.8 fixed-point uint8 GaussianBlur for
    the dyadic small_gaussian_tab kernels: horizontal pass exact in uint16,
    vertical pass < 2^24, final half-up rounding ``(acc + 2^15) >> 16``."""
    kernel_q8_8 = torch.tensor(
        _SMALL_GAUSSIAN_KERNELS_Q8_8[ksize], dtype=torch.float32, device=chw.device
    )
    pad = ksize // 2
    x = chw.detach().to(torch.float32).unsqueeze(1)  # (C, 1, H, W)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, kernel_q8_8.view(1, 1, 1, ksize))  # Q8.8, <= 255 * 256
    x = F.conv2d(x, kernel_q8_8.view(1, 1, ksize, 1))  # Q16.16, < 2^24
    return ((x + 32768.0) / 65536.0).floor().to(torch.uint8).squeeze(1)


def _median_blur_tensor(chw: torch.Tensor, ksize: int) -> torch.Tensor:
    """``cv2.medianBlur``: exact integer median (values 0..255 are exact in
    float32 and the median of an odd count is an element of the window) over
    replicate borders, unfolded in row chunks to bound the k^2 blow-up."""
    pad = ksize // 2
    channels, height, width = chw.shape
    x = chw.detach().to(torch.float32).unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate").squeeze(0)
    middle = (ksize * ksize) // 2
    rows_per_chunk = max(
        1, _MEDIAN_UNFOLD_ELEMENT_BUDGET // (channels * width * ksize * ksize)
    )
    chunks = []
    for row_start in range(0, height, rows_per_chunk):
        row_end = min(height, row_start + rows_per_chunk)
        windows = (
            x[:, row_start : row_end + 2 * pad, :]
            .unfold(1, ksize, 1)
            .unfold(2, ksize, 1)
            .reshape(channels, row_end - row_start, width, ksize * ksize)
        )
        chunks.append(windows.sort(dim=-1).values[..., middle])
    return torch.cat(chunks, dim=1).to(torch.uint8)

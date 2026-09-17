"""Tensor-native sibling of ``threshold/v1``.

Every path replicates the exact uint8 semantics of the cv2 calls in v1's
``apply_thresholding`` (OpenCV modules/imgproc/src/thresh.cpp) or delegates
to them - parity versus the numpy block is bit-exact:

* Fixed types (``binary`` / ``binary_inv`` / ``trunc`` / ``tozero`` /
  ``tozero_inv``): per-pixel integer comparisons with cv2's parameter
  handling - ``ithresh = cvFloor(thresh)`` with a strict ``>`` compare,
  ``imaxval = saturate_cast<uchar>(cvRound(maxval))`` (round half to even,
  then clamp to [0, 255]), ``trunc`` ignoring ``maxval`` and filling with the
  saturated ``ithresh``, and the degenerate ``ithresh < 0 / >= 255`` branches
  that fill or copy without touching pixels. cv2 applies these per-channel,
  so any channel count runs tensor-native.

* ``otsu`` (cv2 requires single-channel 8-bit input): ``torch.bincount`` runs
  on the device, one D2H sync moves the 256 counts to the host, and
  ``getThreshVal_Otsu_8u``'s double-precision scan is replicated in python
  floats operation-for-operation (FLT_EPSILON guards, strict ``>``
  tie-breaking, the ``mu1 *= q1`` pre-multiplication ordering); the found
  threshold is applied on the device with the binary semantics above.
  Three-channel tensor input delegates so cv2 raises exactly v1's error.

* ``adaptive_mean``: cv2 computes ``boxFilter(src, CV_8U, 11x11,
  normalize=True, BORDER_REPLICATE|BORDER_ISOLATED)`` - the window mean
  rounded to uint8 - then ``dst = src > mean - 2 ? imaxval : 0`` (delta 2 is
  ``cvCeil``-ed to the exact integer 2 for THRESH_BINARY). Sums of 121 uint8
  pixels stay below 2**24, so a float32 convolution over a replicate-padded
  image yields exact integer sums on any device in any summation order, and
  ``sum/121`` can never land on a .5 tie (2*sum is even, 121*(2k+1) is odd),
  so rounding matches ``cvRound`` bit-exactly.

* ``adaptive_gaussian``: delegates. cv2 computes the local mean as
  ``convertTo(GaussianBlur(float32(src), 11x11, sigma 2.0), CV_8U)``, and the
  float32 filter's bit pattern differs across SIMD backends, so no torch
  replication is bit-stable across deployment hardware.

Numpy/base64-born images delegate (no materialised tensor).
"""

import math
from typing import Type, Union

import torch
import torch.nn.functional as F

from inference.core.workflows.core_steps.classical_cv.threshold.v1 import (
    ImageThresholdManifest,
    apply_thresholding,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock

_FIXED_THRESHOLD_TYPES = {"binary", "binary_inv", "trunc", "tozero", "tozero_inv"}
_SINGLE_CHANNEL_TENSOR_TYPES = {"otsu", "adaptive_mean"}
# C's FLT_EPSILON (2**-23) - getThreshVal_Otsu_8u guards with the FLOAT
# epsilon even though the scan itself runs in double precision.
_FLT_EPSILON = 1.1920928955078125e-07
# v1 hardcodes cv2.adaptiveThreshold(..., blockSize=11, C=2) with THRESH_BINARY,
# for which cv2 turns delta into `idelta = cvCeil(2.0)` - the exact integer 2.
_ADAPTIVE_BLOCK_SIZE = 11
_ADAPTIVE_DELTA = 2


class ImageThresholdBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[ImageThresholdManifest]:
        return ImageThresholdManifest

    def run(
        self,
        image: WorkflowImageData,
        threshold_type: str,
        thresh_value: int,
        max_value: int,
        *args,
        **kwargs,
    ) -> BlockResult:
        if _requires_numpy_delegation(image=image, threshold_type=threshold_type):
            thresholded_image = apply_thresholding(
                image.numpy_image, threshold_type, thresh_value, max_value
            )
            output = WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=thresholded_image,
            )
            return {OUTPUT_IMAGE_KEY: output}
        thresholded_tensor = _apply_thresholding_tensor(
            chw=image.tensor_image,
            threshold_type=threshold_type,
            thresh_value=thresh_value,
            max_value=max_value,
        )
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            tensor_image=thresholded_tensor,
        )
        return {OUTPUT_IMAGE_KEY: output}


def _requires_numpy_delegation(image: WorkflowImageData, threshold_type: str) -> bool:
    if not image.is_tensor_materialised():
        return True
    if threshold_type == "adaptive_gaussian":
        # SIMD-dispatch-dependent float32 blur - see the module docstring.
        return True
    if threshold_type in _SINGLE_CHANNEL_TENSOR_TYPES:
        # cv2 asserts CV_8UC1 for otsu/adaptive - delegating multi-channel
        # input reproduces exactly the error v1 raises.
        return int(image.tensor_image.shape[0]) != 1
    return False


def _apply_thresholding_tensor(
    chw: torch.Tensor,
    threshold_type: str,
    thresh_value: Union[int, float],
    max_value: Union[int, float],
) -> torch.Tensor:
    if threshold_type in _FIXED_THRESHOLD_TYPES:
        return _fixed_threshold_tensor(
            chw=chw,
            threshold_type=threshold_type,
            thresh_value=thresh_value,
            max_value=max_value,
        )
    if threshold_type == "otsu":
        return _otsu_threshold_tensor(chw=chw, max_value=max_value)
    if threshold_type == "adaptive_mean":
        return _adaptive_mean_threshold_tensor(chw=chw, max_value=max_value)
    raise ValueError(f"Unknown threshold type: {threshold_type}")


def _fixed_threshold_tensor(
    chw: torch.Tensor,
    threshold_type: str,
    thresh_value: Union[int, float],
    max_value: Union[int, float],
) -> torch.Tensor:
    ithresh = _cv_floor(thresh_value)
    imaxval = _cv_round(max_value)
    if threshold_type == "trunc":
        imaxval = ithresh  # trunc ignores maxval and fills with the threshold
    imaxval = _saturate_cast_uint8(imaxval)
    if ithresh < 0 or ithresh >= 255:
        # cv2's degenerate branches fill or copy without reading pixels.
        if threshold_type == "binary":
            return torch.full_like(chw, 0 if ithresh >= 255 else imaxval)
        if threshold_type == "binary_inv":
            return torch.full_like(chw, imaxval if ithresh >= 255 else 0)
        if threshold_type in ("trunc", "tozero_inv") and ithresh < 0:
            return torch.full_like(chw, 0)
        if threshold_type == "tozero" and ithresh >= 255:
            return torch.full_like(chw, 0)
        return chw.clone()
    above = chw > ithresh  # cv2 compares strictly against the floored threshold
    zero = chw.new_zeros(())
    if threshold_type == "binary":
        return torch.where(above, chw.new_full((), imaxval), zero)
    if threshold_type == "binary_inv":
        return torch.where(above, zero, chw.new_full((), imaxval))
    if threshold_type == "trunc":
        return torch.clamp(chw, max=imaxval)  # imaxval == ithresh in [0, 254]
    if threshold_type == "tozero":
        return torch.where(above, chw, zero)
    return torch.where(above, zero, chw)  # tozero_inv


def _otsu_threshold_tensor(
    chw: torch.Tensor, max_value: Union[int, float]
) -> torch.Tensor:
    counts = torch.bincount(chw.detach().reshape(-1).long(), minlength=256)
    ithresh = _otsu_threshold_from_counts(counts=counts.cpu().tolist())
    # cv2 feeds the found threshold through the standard binary path; the scan
    # only returns values in [0, 254], so the strict-`>` in-range branch
    # always applies.
    return _fixed_threshold_tensor(
        chw=chw,
        threshold_type="binary",
        thresh_value=ithresh,
        max_value=max_value,
    )


def _otsu_threshold_from_counts(counts: list) -> int:
    """OpenCV's ``getThreshVal_Otsu_8u`` (modules/imgproc/src/thresh.cpp)
    replicated operation-for-operation in python floats (IEEE doubles): the
    same accumulation order, the FLT_EPSILON validity guards, and the strict
    ``sigma > max_sigma`` comparison that keeps the FIRST maximizer."""
    total = sum(counts)
    scale = 1.0 / total
    mu = 0.0
    for i in range(256):
        mu += i * float(counts[i])
    mu *= scale
    mu1 = 0.0
    q1 = 0.0
    max_sigma = 0.0
    max_val = 0.0
    for i in range(256):
        p_i = counts[i] * scale
        mu1 *= q1  # cv2 un-normalizes BEFORE the validity guard
        q1 += p_i
        q2 = 1.0 - q1
        if min(q1, q2) < _FLT_EPSILON or max(q1, q2) > 1.0 - _FLT_EPSILON:
            continue
        mu1 = (mu1 + i * p_i) / q1
        mu2 = (mu - q1 * mu1) / q2
        sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_val = i
    return int(max_val)


def _adaptive_mean_threshold_tensor(
    chw: torch.Tensor, max_value: Union[int, float]
) -> torch.Tensor:
    imaxval = _saturate_cast_uint8(_cv_round(max_value))
    pad = _ADAPTIVE_BLOCK_SIZE // 2
    x = chw.detach().to(torch.float32).unsqueeze(0)  # (1, 1, H, W)
    padded = _replicate_pad(x=x, pad=pad)
    kernel = torch.ones(
        (1, 1, _ADAPTIVE_BLOCK_SIZE, _ADAPTIVE_BLOCK_SIZE),
        dtype=torch.float32,
        device=chw.device,
    )
    # 121 uint8 addends keep every partial sum an exact float32 integer
    # (< 2**24), so the sums equal cv2's integer boxFilter sums bit-for-bit
    # on any device and in any accumulation order.
    sums = F.conv2d(padded, kernel)
    scale = 1.0 / (_ADAPTIVE_BLOCK_SIZE * _ADAPTIVE_BLOCK_SIZE)
    # torch.round is half-to-even like cvRound; sum/121 can never tie on .5,
    # so the rounded means match cv2's uint8 boxFilter output exactly.
    mean = torch.round(sums * scale).clamp_(0, 255).to(torch.int16)
    above = chw.to(torch.int16).unsqueeze(0) > mean - _ADAPTIVE_DELTA
    thresholded = torch.where(above, chw.new_full((), imaxval), chw.new_zeros(()))
    return thresholded.squeeze(0)


def _replicate_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """BORDER_REPLICATE as a clamped-index gather: unlike ``F.pad`` with
    ``mode='replicate'`` it also handles images smaller than the padding,
    which cv2 supports."""
    height, width = x.shape[-2], x.shape[-1]
    rows = torch.arange(-pad, height + pad, device=x.device).clamp_(0, height - 1)
    cols = torch.arange(-pad, width + pad, device=x.device).clamp_(0, width - 1)
    return x.index_select(-2, rows).index_select(-1, cols)


def _cv_floor(value: Union[int, float]) -> int:
    return math.floor(value)


def _cv_round(value: Union[int, float]) -> int:
    # cvRound rounds half to even (IEEE round-to-nearest), like python round().
    return int(round(float(value)))


def _saturate_cast_uint8(value: int) -> int:
    return min(max(int(value), 0), 255)

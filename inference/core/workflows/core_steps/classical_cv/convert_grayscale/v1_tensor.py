"""Tensor-native sibling of ``convert_grayscale/v1``.

OpenCV's uint8 BGR2GRAY is a per-pixel fixed-point map. The u8 path of the
installed OpenCV 4.x uses the 15-bit coefficients (``RY15``/``GY15``/``BY15``
in ``color_rgb2gray``):

    gray = (9798*R + 19235*G + 3735*B + (1 << 14)) >> 15

Verified exhaustively against cv2 4.10: zero mismatches over ALL 2^24 RGB
triples (a total proof over the input space, not a sample), including
odd-width/SIMD-tail layouts and non-contiguous inputs, so the map is
position-independent. Note the widely quoted legacy constants
(``4899/9617/1868 >> 14``) are NOT what modern cv2 computes - they disagree on
~0.26% of triples.

Being a pure function of (R, G, B), the conversion runs fully device-resident
on the CHW RGB tensor: cast to int32 (max accumulator ``255*32768 + 2^14 =
8_372_224``, far inside int32), weighted channel sum plus the rounding
constant, arithmetic shift, cast back to uint8, emitted as the ``(1, H, W)``
grayscale contract shape - zero host syncs, BIT-EXACT versus the numpy block.

Delegation paths:
- numpy/base64-born images (no materialised tensor) keep v1's numpy math
  instead of forcing an eager host->device conversion - the standard
  materialization-aware rule (this also covers 4-channel BGRA input, which can
  only arrive numpy-born);
- tensor-born single-channel images delegate too: materialising ``numpy_image``
  yields the 2-D view that v1 would feed to ``cv2.cvtColor``, which rejects it
  with ``cv2.error`` - so the error behaviour matches v1 exactly.
"""

from typing import Type

import cv2
import torch

from inference.core.workflows.core_steps.classical_cv.convert_grayscale.v1 import (
    ConvertGrayscaleManifest,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock

# OpenCV 4.x u8 BGR2GRAY fixed-point constants (color_rgb2gray RY15/GY15/BY15),
# exhaustively verified against the installed cv2 - see the module docstring.
_RY15 = 9798
_GY15 = 19235
_BY15 = 3735
_GRAY_SHIFT = 15
_GRAY_ROUND = 1 << (_GRAY_SHIFT - 1)


class ConvertGrayscaleBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[ConvertGrayscaleManifest]:
        return ConvertGrayscaleManifest

    def run(
        self,
        image: WorkflowImageData,
        *args,
        **kwargs,
    ) -> BlockResult:
        if not image.is_tensor_materialised() or image.tensor_image.shape[0] != 3:
            # Numpy/base64-born image (or a tensor-born single-channel one,
            # whose 2-D numpy view cv2 rejects exactly as in v1): keep v1's
            # numpy one-liner instead of forcing a host->device conversion.
            gray = cv2.cvtColor(image.numpy_image, cv2.COLOR_BGR2GRAY)
            output = WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=gray
            )
            return {OUTPUT_IMAGE_KEY: output}
        return {OUTPUT_IMAGE_KEY: _convert_grayscale_tensor(image=image)}


def _convert_grayscale_tensor(image: WorkflowImageData) -> WorkflowImageData:
    """Device-resident mirror of ``cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`` for
    ``(3, H, W)`` RGB uint8 tensors, emitting the ``(1, H, W)`` contract shape."""
    chw = image.tensor_image.detach().to(torch.int32)
    weighted = (
        chw[0] * _RY15 + chw[1] * _GY15 + chw[2] * _BY15 + _GRAY_ROUND
    ) >> _GRAY_SHIFT
    gray_chw = weighted.to(torch.uint8).unsqueeze(0)
    return WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        tensor_image=gray_chw,
    )

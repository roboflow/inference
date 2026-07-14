"""Tensor-native sibling of ``convert_grayscale/v1``.

OpenCV 4.x's uint8 BGR2GRAY is the fixed-point map (``color_rgb2gray``
``RY15``/``GY15``/``BY15`` - not the widely quoted legacy
``4899/9617/1868 >> 14`` constants, which disagree with modern cv2):

    gray = (9798*R + 19235*G + 3735*B + (1 << 14)) >> 15

The tensor path computes it in int32 on the device (max accumulator
``255*32768 + 2^14 = 8_372_224``, inside int32) and emits the ``(1, H, W)``
contract shape, bit-exact versus the numpy block.

Delegation to v1:
- numpy/base64-born images (no materialised tensor); also covers 4-channel
  BGRA input, which can only arrive numpy-born;
- tensor-born single-channel images: the materialised 2-D numpy view makes
  ``cv2.cvtColor`` raise the same ``cv2.error`` as v1.
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

# OpenCV 4.x u8 BGR2GRAY fixed-point constants (color_rgb2gray RY15/GY15/BY15).
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

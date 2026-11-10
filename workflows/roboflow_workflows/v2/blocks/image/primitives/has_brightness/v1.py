"""Decide whether the mean pixel value of an image reaches a minimum.

Canonical type: ``roboflow_core/has_brightness@v1``.
"""

from typing import Any, Dict
import torch
from pydantic import Field, StrictFloat
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class HasBrightnessBlock(Block):
    """Decide whether the mean pixel value of an image reaches a minimum.

    ``keep`` is a plain ``bool``; bind it to the ``condition`` of
    ``roboflow_core/continue_if_boolean@v1`` to gate later steps. The mean is computed on the image's
    device; only that one number is read back.
    """

    type = "roboflow_core/has_brightness@v1"
    outputs = {"keep": Output(BOOLEAN_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to measure.")
        minimum: StrictFloat | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=255,
            description="Inclusive threshold of the mean pixel value, 0..255.",
        )

    def run(self, *, image: ImageData, minimum: float) -> Dict[str, Any]:
        """Compare the mean pixel value of ``image`` with ``minimum``.

        Args:
            image: Image to measure.
            minimum: Inclusive threshold in ``0..255``.

        Returns:
            ``keep``: ``True`` when the mean over all pixels and channels is at
            least ``minimum``.
        """
        mean = image.tensor_image.mean(dtype=torch.float32).item()
        keep = bool(mean >= minimum)
        result = {"keep": keep}

        return result

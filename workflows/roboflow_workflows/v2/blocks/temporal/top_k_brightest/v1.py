"""Choose the ``k`` brightest images of a group, brightest first.

Canonical type: ``roboflow_core/top_k_brightest@v1``.
"""

from typing import Any, Dict, List
import torch
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Group, Output, Ref, Selection


def _mean_brightness(image: ImageData) -> float:
    brightness = image.tensor_image.mean(dtype=torch.float32).item()

    return brightness


class TopKBrightestBlock(Block):
    """Choose the ``k`` brightest images of a group, brightest first.

    Brightness is the mean pixel value over all channels, computed on each
    image's device. Ties keep group order. A group with fewer than ``k``
    images yields all of them. The result is a new collection along the
    ``ranked`` axis: it keeps each chosen image object and its own context,
    and stays a collection even when it holds one image. It is not a
    reduction, so it cannot be collected over time again directly.
    """

    type = "roboflow_core/top_k_brightest@v1"
    outputs = {
        "images": Output(
            IMAGE_KIND,
            expand="ranked",
            source="images",
            context_policy="selected",
            description="Chosen images, brightest first.",
        ),
        "brightness": Output(
            FLOAT_KIND,
            expand="ranked",
            source="images",
            context_policy="selected",
            description="Mean pixel value of each chosen image, 0..255.",
        ),
    }

    class Params(BlockParams):
        images: Group(IMAGE_KIND) = Field(
            description="Images to rank: the children of one parent, possibly none."
        )
        k: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1, ge=1, description="Number of images to keep."
        )

    def run(self, *, images: Batch[ImageData], k: int) -> Dict[str, Any]:
        """Rank ``images`` by brightness and keep the first ``k``.

        Args:
            images: Images of one parent, possibly none.
            k: Number of images to keep.

        Returns:
            ``images``: ``Selection`` of the chosen images. ``brightness``: the
            same selection with each image's mean pixel value.
        """
        brightness = [_mean_brightness(image) for image in images]
        ranked = sorted(range(len(images)), key=lambda position: -brightness[position])
        kept: List[int] = ranked[:k]
        chosen = [images.indices[position] for position in kept]
        result = {
            "images": Selection(chosen),
            "brightness": Selection(
                chosen, values=[brightness[position] for position in kept]
            ),
        }

        return result

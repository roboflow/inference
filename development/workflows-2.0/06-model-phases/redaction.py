"""A block that changes its input image in place, inside a phase.

``mutates = ("image",)`` covers every phase and implementation of the block.
A step that reads the same image without being ordered after this one gets a
compile warning, or an error under ``mutation_conflicts="error"``.
"""

from typing import Any, Dict, List, Tuple

from pydantic import Field, StrictInt
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.phases import phase

Rectangle = Tuple[int, int, int, int]


class RedactRegion(Block):
    """Fill a rectangle of the image with grey, modifying its tensor in place."""

    type = "model_demo/redact_region@v1"
    mutates = ("image",)
    outputs = {
        "image": Output(IMAGE_KIND, source="image", description="The same image."),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image changed in place.")
        region: List[StrictInt] = Field(
            min_length=4,
            max_length=4,
            description="Rectangle [x0, y0, x1, y1] in pixels, upper bounds exclusive.",
        )

    @phase
    def bounds(self, image: ImageData, region: List[int]) -> Rectangle:
        """Clip the rectangle to the image; reject one that misses it."""
        x0, y0, x1, y1 = region
        clipped = (
            max(0, x0),
            max(0, y0),
            min(image.width, x1),
            min(image.height, y1),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise ValueError(
                f"region {region} lies outside the {image.width}x{image.height} image"
            )

        return clipped

    @phase
    def result(self, image: ImageData, bounds: Rectangle) -> Dict[str, Any]:
        """Write grey into the bound tensor; no copy is made."""
        x0, y0, x1, y1 = bounds
        image.tensor_image[:, y0:y1, x0:x1] = 128
        result = {"image": image}

        return result

    def run(self, *, image: ImageData, region: List[int]) -> Dict[str, Any]:
        """Grey out ``region`` of ``image`` in place.

        Args:
            image: Image whose tensor is modified.
            region: Rectangle ``[x0, y0, x1, y1]``.

        Returns:
            ``image``: the same, now modified, image object.

        Raises:
            PhaseFailure: Phase ``bounds``, when ``region`` misses the image.
        """
        result = self.result(image, self.bounds(image, region))

        return result

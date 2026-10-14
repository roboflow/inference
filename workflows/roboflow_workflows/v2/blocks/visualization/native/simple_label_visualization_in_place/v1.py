"""Paste labels in place on images an earlier step of this workflow created.

Canonical type: ``roboflow_core/simple_label_visualization_in_place@v1``.
"""

from typing import Any, Dict, List
from pydantic import Field, model_validator
from roboflow_workflows.v2.blocks.visualization._shared.blocks import (
    checked_batch,
    checked_drawings,
    reject_removed_backend,
)
from roboflow_workflows.v2.blocks.visualization.drawing import DETECTION_DRAWING_KIND
from roboflow_workflows.v2.blocks.visualization.kernels import TensorKernels
from roboflow_workflows.v2.blocks.visualization.streams import ReadyScope
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)


class InPlaceLabelVisualizationBlock(Block):
    """Paste labels in place on images an earlier step of this workflow created.

    Declares ``mutates=("image",)``. Meant to follow the box painter, so boxes
    plus labels cost one copy without the fused block. Bound to a passive
    workflow input it overwrites the caller's pixels; the compiler cannot see
    that ownership, so do not do it.

    On failure the batch's scope is still waited for; images before a
    failing one stay labelled. Fast prototype, not a V1 port.
    """

    type = "roboflow_core/simple_label_visualization_in_place@v1"
    mutates = ("image",)
    outputs = {
        "image": Output(
            IMAGE_KIND, source="image", description="The same image, now labelled."
        ),
    }

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description=(
                "Contiguous RGB images this workflow created (e.g. the box "
                "painter's output), changed in place; delivered as one batch."
            )
        )
        predictions: Ref(
            OBJECT_DETECTION_PREDICTION_KIND,
            DETECTION_DRAWING_KIND,
            batch="always",  # noqa: F821
        ) = Field(
            description=(
                "Per image, in batch order: native Detections (with "
                "image_metadata['class_names']) or a prepared DetectionDrawing."
            )
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_removed_backend(cls, values: Any) -> Any:
            checked = reject_removed_backend(values, params=cls.__name__)

            return checked

    def __init__(self) -> None:
        self._scope = ReadyScope()
        # Holds bounded caches (see ``kernels``).
        self._kernels = TensorKernels()

    def run(self, *, image: Batch, predictions: Batch) -> List[Dict[str, Any]]:
        """Paste every image's labels in place, then wait once.

        Args:
            image: Contiguous RGB images on one device; changed in place.
            predictions: Their detections or prepared drawings, same order.

        Returns:
            Per image, in order: ``image``, the same object, ready to read.

        Raises:
            ValueError: Before any painting, as ``roboflow_core/simple_box_visualization@v1``,
                or on a non-contiguous image.
        """
        images, members, device = checked_batch(image, predictions, block=self.type)
        for member in images:
            if not member.tensor_image.is_contiguous():
                raise ValueError(
                    f"{self.type} paints in place and needs contiguous image "
                    "tensors; bind the box painter's output"
                )
        if not images:
            return []

        with self._scope.work(device):
            drawings = checked_drawings(members)
            scenes = [member.tensor_image for member in images]
            self._kernels.paint(scenes, drawings, boxes=False, labels=True)
        results = [{"image": member} for member in images]

        return results

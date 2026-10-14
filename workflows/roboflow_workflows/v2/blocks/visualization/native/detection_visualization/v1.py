"""Draw boxes, then labels, onto one copy per image: the fast route.

Canonical type: ``roboflow_core/detection_visualization@v1``.
"""

from typing import Any, Dict, List
import torch
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


class DetectionVisualizationBlock(Block):
    """Draw boxes, then labels, onto one copy per image: the fast route.

    Same pixels as the box painter followed by the label painter, with one
    copy, at most one host transfer and one wait per batch. Fast prototype,
    not a V1 port.
    """

    type = "roboflow_core/detection_visualization@v1"
    outputs = {
        "image": Output(IMAGE_KIND, description="New image with boxes and labels."),
    }

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to annotate, delivered as one batch; not modified."
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
        """Copy every image, draw its boxes and labels, then wait once.

        Args:
            image: The batch's RGB images, all on one device; not modified.
            predictions: Their detections or prepared drawings, same order.

        Returns:
            Per image, in order: ``image``, a new annotated image with the
            source's identity and provenance, ready to read.

        Raises:
            ValueError: Before any painting, as ``roboflow_core/simple_box_visualization@v1``.
        """
        images, members, device = checked_batch(image, predictions, block=self.type)
        if not images:
            return []

        with self._scope.work(device):
            drawings = checked_drawings(members)
            scenes = [
                member.tensor_image.clone(memory_format=torch.contiguous_format)
                for member in images
            ]
            self._kernels.paint(scenes, drawings, boxes=True, labels=True)
        results = [
            {"image": member.with_pixels(scene)}
            for member, scene in zip(images, scenes)
        ]

        return results

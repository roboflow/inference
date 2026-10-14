"""Prepare a batch's detections for drawing with one host transfer.

Canonical type: ``roboflow_core/detection_drawing@v1``.
"""

from typing import Any, Dict, List
from pydantic import Field
from roboflow_workflows.v2.blocks.visualization.drawing import (
    DETECTION_DRAWING_KIND,
    prepare_drawings,
)
from roboflow_workflows.v2.blocks.visualization.streams import ReadyScope
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)


class DetectionDrawingBlock(Block):
    """Prepare a batch's detections for drawing with one host transfer.

    Optional. Bind its ``drawing`` output to several painters so they share
    one device-to-host transfer instead of making one each.
    """

    type = "roboflow_core/detection_drawing@v1"
    outputs = {
        "drawing": Output(
            DETECTION_DRAWING_KIND,
            source="predictions",
            description="New read-only host copy of boxes, classes and label texts.",
        ),
    }

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        predictions: Ref(
            OBJECT_DETECTION_PREDICTION_KIND, batch="always"  # noqa: F821
        ) = Field(
            description=(
                "Detections of every image, delivered as one batch; each needs "
                "image_metadata['class_names'] unless empty."
            )
        )

    def __init__(self) -> None:
        self._scope = ReadyScope()

    def run(self, *, predictions: Batch) -> List[Dict[str, Any]]:
        """Copy every image's detections to the host in one transfer.

        Args:
            predictions: Detections of the batch's images; not modified.

        Returns:
            One ``drawing`` per image, in batch order.

        Raises:
            ValueError: On tensors on several devices, inconsistent shapes or
                a class id without a name.
        """
        members = list(predictions)
        if not members:
            return []

        # Keep the device copies off the caller's stream after joining its
        # queued work. prepare_drawings itself waits for the host transfer.
        with self._scope.work(members[0].xyxy.device):
            drawings = prepare_drawings(members)
        results = [{"drawing": drawing} for drawing in drawings]

        return results

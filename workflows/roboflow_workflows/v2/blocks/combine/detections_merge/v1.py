"""Merge all rows into one union box: ``roboflow_core/detections_merge@v1``.

Port of V1 ``roboflow_core/detections_merge@v1``::

    box        = (min x_min, min y_min, max x_max, max y_max) over all rows
    confidence = lowest row confidence
    class      = class_id 0, named class_name

The merged row is a new detection, so it gets a fresh row identity
(decision 015); its exported ``detection_id`` derives from it. The output is
box-only object detection, as in V1: masks and keypoints are not unioned.
It keeps the input's frame, device and float dtypes. Everything runs on the
input's device with no host read of tensor values.
"""

from typing import Any, Dict
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.query.detections import (
    CLASS_NAMES_KEY,
    boxes_of,
)
from roboflow_workflows.v2.blocks.query.errors import OperationFailure
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.masks import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    with_row_identity,
    without_row_record,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from inference_models.models.base.object_detection import Detections


MERGED_CLASS_ID = 0

PREDICTION_TYPE_KEY = "prediction_type"

MERGE_INPUT_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)

class DetectionsMergeBlock(Block):
    """Replace all detections with one box that encloses them.

    Example: boxes ``(10, 10, 20, 20)`` at 0.9 and ``(15, 5, 30, 18)`` at
    0.6 merge into ``(10, 5, 30, 20)`` at 0.6, class ``merged_detection``.
    """

    type = "roboflow_core/detections_merge@v1"
    metadata = {
        "name": "Detections Merge",
        "port_of": "roboflow_core/detections_merge@v1",
        "section": "transformation",
    }
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="predictions",
            description=(
                "One merged box with a fresh row identity, or no rows for empty "
                "input. Masks and keypoints are not carried."
            ),
        ),
    }

    class Params(BlockParams):
        predictions: Ref(*MERGE_INPUT_KINDS) = Field(
            description=(
                "Detections, instance segmentation (dense, RLE or view) or "
                "keypoint predictions with boxes."
            ),
            examples=["$steps.object_detection_model.predictions"],
        )
        class_name: str | Ref(STRING_KIND) = Field(
            default="merged_detection",
            description="Class name of the merged detection; its class id is 0.",
            examples=["merged_detection", "$inputs.merged_class_name"],
        )

    def run(self, *, predictions: Any, class_name: str) -> Dict[str, Any]:
        """Merge every row of one prediction.

        Args:
            predictions: A carrier accepted by ``merge_rows``.
            class_name: Class name of the merged detection.

        Returns:
            ``predictions``: the merged ``Detections``.

        Raises:
            ValueError: For keypoints without boxes or a non-string class name.
        """
        merged = merge_rows(predictions, class_name=class_name)

        return {"predictions": merged}

def merge_rows(predictions: Any, *, class_name: str) -> Detections:
    """Return one detection whose box encloses every input row.

    Args:
        predictions: Detections, InstanceDetections, a SegmentationView or a
            ``(KeyPoints, Detections)`` tuple.
        class_name: Name of the merged class, stored under class id 0.

    Returns:
        ``Detections`` with one row, or none for empty input, on the input's
        device. ``image_metadata`` is the input's (frame included) with
        ``class_names`` set to ``{0: class_name}``, ``prediction_type`` set to
        ``object-detection`` and a fresh row record.

    Raises:
        ValueError: For keypoints without boxes or a non-string class name.
    """
    if not isinstance(class_name, str):
        raise ValueError(
            f"roboflow_core/detections_merge@v1: class_name must be a string, got "
            f"{type(class_name).__name__}"
        )

    try:
        boxes = boxes_of(predictions)
    except OperationFailure as error:
        raise ValueError(f"roboflow_core/detections_merge@v1: {error}") from None

    xyxy, confidence = boxes.xyxy, boxes.confidence
    if len(boxes) == 0:
        merged_xyxy = xyxy.new_zeros((0, 4))
        merged_confidence = confidence.new_zeros((0,))
    else:
        corners = torch.cat([xyxy[:, :2].amin(dim=0), xyxy[:, 2:].amax(dim=0)])
        merged_xyxy = corners.unsqueeze(0)
        merged_confidence = confidence.min().reshape(1)

    image_metadata = {
        **(without_row_record(boxes.image_metadata) or {}),
        CLASS_NAMES_KEY: {MERGED_CLASS_ID: class_name},
        PREDICTION_TYPE_KEY: "object-detection",
    }
    merged = Detections(
        xyxy=merged_xyxy,
        class_id=torch.full(
            (len(merged_xyxy),),
            MERGED_CLASS_ID,
            dtype=torch.int64,
            device=xyxy.device,
        ),
        confidence=merged_confidence,
        image_metadata=image_metadata,
    )
    identified = with_row_identity(merged)

    return identified

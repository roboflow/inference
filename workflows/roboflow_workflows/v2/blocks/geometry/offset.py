"""Grow detection boxes by a margin: ``v2/detection_offset@v1``.

Port of V1 ``roboflow_core/detection_offset@v1``, with V1's exact arithmetic::

    Pixels       pad = offset // 2
    Percent (%)  pad = trunc(box_side * offset / 200)
    x_min' = max(0, x_min - pad_x)      x_max' = min(W, x_max + pad_x)
    y_min' = max(0, y_min - pad_y)      y_max' = min(H, y_max + pad_y)

``W`` and ``H`` are the size of the image the prediction describes, read from
its frame. Only boxes change: masks, keypoints, classes, confidences and row
metadata stay aligned with their rows, unchanged. Like V1, a row that has a
``detection_id`` gets a new one and keeps the old one as ``parent_id``.

Keypoint predictions need their boxes: a ``(KeyPoints, None)`` tuple is a
``ContractError`` (decision 018), unlike detections with zero rows, which pass
through unchanged.
"""

import uuid
from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from pydantic import Field, StrictInt
from roboflow_workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    PARENT_ID_KEY,
)
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)


class DetectionOffsetBlock(Block):
    """Pad every box of a prediction, clipped to the image.

    Example: box ``[10, 10, 31, 20]`` with 25 % on both sides pads
    ``trunc(21 * 25 / 200) = 2`` horizontally and ``trunc(1.25) = 1``
    vertically: ``[8, 9, 33, 21]``. With ``Pixels`` and an offset of 5 the
    pad is ``5 // 2 = 2`` on each side.
    """

    type = "v2/detection_offset@v1"
    outputs = {
        "predictions": Output(
            *PREDICTION_KINDS,
            source="predictions",
            description="The prediction with padded boxes; all rows kept.",
        ),
    }

    class Params(BlockParams):
        predictions: Ref(*PREDICTION_KINDS) = Field(
            description="Object detection, instance segmentation or keypoint prediction."
        )
        offset_width: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1,
            description=(
                "Horizontal padding, split over both sides: pixels, or percent "
                "of each box's width."
            ),
        )
        offset_height: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1,
            description=(
                "Vertical padding, split over both sides: pixels, or percent "
                "of each box's height."
            ),
        )
        units: Literal["Pixels", "Percent (%)"] = Field(
            default="Pixels",
            description="Unit of both offsets (V1 spellings).",
        )

    def run(
        self, *, predictions: Any, offset_width: int, offset_height: int, units: str
    ) -> Dict[str, Any]:
        """Pad the boxes of ``predictions``.

        Args:
            predictions: ``Detections``, ``InstanceDetections`` or a
                ``(KeyPoints, Detections)`` tuple.
            offset_width: Horizontal offset.
            offset_height: Vertical offset.
            units: ``"Pixels"`` or ``"Percent (%)"``.

        Returns:
            ``predictions``: same type, boxes padded.

        Raises:
            ValueError: When the prediction has no frame to clip against.
            ContractError: When keypoints come without boxes.
        """
        if isinstance(predictions, tuple):
            key_points, detections = predictions
            if detections is None:
                raise ContractError(
                    f"{self.type}: the keypoint prediction has no bounding boxes "
                    f"({len(key_points)} skeletons, Detections None), so there "
                    "are no boxes to offset; use a keypoint model that also "
                    "returns boxes"
                )
        else:
            key_points, detections = None, predictions
        if detections is None or len(detections.xyxy) == 0:
            return {"predictions": predictions}

        image_height, image_width = frame_of(detections).size_hw
        padded = replace(
            detections,
            xyxy=padded_boxes(
                detections.xyxy,
                offset_wh=(offset_width, offset_height),
                percent=units == "Percent (%)",
                image_wh=(image_width, image_height),
            ),
            bboxes_metadata=_renewed_row_ids(detections.bboxes_metadata),
        )
        result = {"predictions": padded if key_points is None else (key_points, padded)}

        return result


def padded_boxes(
    xyxy: torch.Tensor,
    *,
    offset_wh: Tuple[int, int],
    percent: bool,
    image_wh: Tuple[int, int],
) -> torch.Tensor:
    """Return V1-padded boxes on the boxes' device and dtype.

    Args:
        xyxy: ``(N, 4)`` boxes.
        offset_wh: ``(offset_width, offset_height)``.
        percent: Percent of each box side instead of pixels.
        image_wh: ``(W, H)`` clip limits.

    Returns:
        New ``(N, 4)`` tensor.
    """
    x_min, y_min, x_max, y_max = xyxy.unbind(dim=1)
    offset_width, offset_height = offset_wh
    if percent:
        # trunc toward zero, as V1's int(); sides are never negative.
        pad_x = ((x_max - x_min) * offset_width / 200).to(torch.int64).to(xyxy.dtype)
        pad_y = ((y_max - y_min) * offset_height / 200).to(torch.int64).to(xyxy.dtype)
    else:
        pad_x = torch.full_like(x_min, offset_width // 2)
        pad_y = torch.full_like(y_min, offset_height // 2)

    image_width, image_height = image_wh
    padded = torch.stack(
        (
            (x_min - pad_x).clamp(min=0),
            (y_min - pad_y).clamp(min=0),
            (x_max + pad_x).clamp(max=image_width),
            (y_max + pad_y).clamp(max=image_height),
        ),
        dim=1,
    )

    return padded


def _renewed_row_ids(rows: Optional[List[dict]]) -> Optional[List[dict]]:
    if rows is None:
        return None

    renewed = []
    for row in rows:
        entry = without_host_mirrors(row)
        if DETECTION_ID_KEY in entry:
            entry[PARENT_ID_KEY] = entry[DETECTION_ID_KEY]
            entry[DETECTION_ID_KEY] = str(uuid.uuid4())
        renewed.append(entry)

    return renewed

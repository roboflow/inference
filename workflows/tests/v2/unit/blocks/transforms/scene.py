"""A four-row street scene in every carrier the transform blocks accept.

Rows as a modern model emits them: a row record (namespace ``street``), no
per-row ``detection_id``::

    row  class   box                 conf
    0    car     (10, 20, 50, 60)    0.875
    1    person  (110, 20, 130, 60)  0.5
    2    car     (200, 40, 260, 80)  0.875
    3    bus     (100, 10, 180, 70)  0.375

Dense masks are 100x300 with row ``r`` set at pixel ``(r, r)``. The view puts
each row on a compact 5x15 logit grid (stride 20).
"""

import dataclasses

import torch
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.data.rows import PredictionRows

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

BOXES = [[10, 20, 50, 60], [110, 20, 130, 60], [200, 40, 260, 80], [100, 10, 180, 70]]
CLASS_IDS = [0, 1, 0, 2]
CONFIDENCES = [0.875, 0.5, 0.875, 0.375]
CLASS_NAMES = {0: "car", 1: "person", 2: "bus"}
TRACKER_IDS = [7, 8, 9, 10]


def plain_boxes(rows=(0, 1, 2, 3), *, bboxes_metadata=None) -> Detections:
    rows = list(rows)
    return Detections(
        xyxy=torch.tensor([BOXES[r] for r in rows], dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor([CLASS_IDS[r] for r in rows], dtype=torch.int64),
        confidence=torch.tensor([CONFIDENCES[r] for r in rows], dtype=torch.float32),
        image_metadata={"class_names": dict(CLASS_NAMES)},
        bboxes_metadata=bboxes_metadata,
    )


def boxes(rows=(0, 1, 2, 3), *, bboxes_metadata=None) -> Detections:
    detections = plain_boxes(rows, bboxes_metadata=bboxes_metadata)
    return with_row_identity(detections, namespace="street")


def dense() -> InstanceDetections:
    masks = torch.zeros((4, 100, 300), dtype=torch.bool)
    for row in range(4):
        masks[row, row, row] = True
    detections = plain_boxes()
    instances = InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=masks,
        image_metadata=detections.image_metadata,
    )
    return with_row_identity(instances, namespace="street")


def rle() -> InstanceDetections:
    detections = plain_boxes()
    instances = InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=InstancesRLEMasks(
            image_size=(100, 300), masks=[b"r0", b"r1", b"r2", b"r3"]
        ),
        image_metadata=detections.image_metadata,
    )
    return with_row_identity(instances, namespace="street")


def view() -> SegmentationView:
    scores = torch.full((4, 5, 15), -8.0)
    for row, (x_min, y_min, _, _) in enumerate(BOXES):
        scores[row, y_min // 20 : y_min // 20 + 2, x_min // 20 : x_min // 20 + 2] = 8.0
    geometry = MaskGridGeometry(
        grid_size_hw=(5, 15),
        inference_size_hw=(20, 60),
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(0.2, 0.2),
        crop_size_hw=(100, 300),
        original_size_hw=(100, 300),
        static_crop_xywh=(0, 0, 300, 100),
    )
    return SegmentationView(
        detections=boxes(),
        scores=scores,
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )


def key_points(*, with_boxes: bool = True):
    points = KeyPoints(
        xy=torch.arange(16, dtype=torch.float32).reshape(4, 2, 2),
        class_id=torch.tensor(CLASS_IDS),
        confidence=torch.full((4, 2), 0.5),
    )
    detections = plain_boxes() if with_boxes else None
    return with_row_identity((points, detections), namespace="street")


def rows() -> PredictionRows:
    return PredictionRows(
        native=boxes(), columns={"tracker_id": torch.tensor(TRACKER_IDS)}
    )


def empty() -> Detections:
    return boxes(rows=())


def with_row_classes(names) -> Detections:
    """Boxes whose rows carry their own ``class`` metadata (``None``: none)."""
    metadata = [{} if name is None else {"class": name} for name in names]
    return dataclasses.replace(boxes(), bboxes_metadata=metadata)


def row_ids(prediction) -> list:
    """Semantic row ids of any carrier, in row order."""
    if isinstance(prediction, SegmentationView):
        prediction = prediction.detections
    if isinstance(prediction, PredictionRows):
        prediction = prediction.native
    return row_record(prediction).host_ids()

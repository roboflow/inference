"""Two models' predictions of one 40x60 image, in every carrier combine accepts.

People model (namespace ``people``, class map ``{0: person}``)::

    row  box               conf
    0    (2, 4, 10, 20)    0.9
    1    (30, 6, 38, 30)   0.6

Vehicle model (namespace ``vehicles``, class map ``{0: car, 1: bus}``)::

    row  class  box                conf
    0    car    (12, 18, 28, 30)   0.8
    1    bus    (40, 2, 58, 22)    0.7
    2    car    (20, 30, 36, 38)   0.4

Class id 0 means ``person`` in one map and ``car`` in the other. Dense masks
fill each box; the view stores them on a stride-2 logit grid.
"""

import dataclasses

import numpy as np
import pycocotools.mask as mask_utils
import supervision as sv
import torch
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import FrameMapping
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

SIZE_HW = (40, 60)
STRIDE = 2
PEOPLE = {
    "boxes": [[2, 4, 10, 20], [30, 6, 38, 30]],
    "class_ids": [0, 0],
    "confidence": [0.9, 0.6],
    "class_names": {0: "person"},
}
VEHICLES = {
    "boxes": [[12, 18, 28, 30], [40, 2, 58, 22], [20, 30, 36, 38]],
    "class_ids": [0, 1, 0],
    "confidence": [0.8, 0.7, 0.4],
    "class_names": {0: "car", 1: "bus"},
}


def frame_metadata(image_id: str = "street", size_hw=SIZE_HW) -> dict:
    mapping = FrameMapping.identity(image_id, size_hw)
    frame = FrameRef(image_id=image_id, size_hw=size_hw, parent=mapping, root=mapping)
    return frame.prediction_metadata()


def boxes(model=PEOPLE, *, namespace="people", rows=None, image_id="street"):
    """Identified ``Detections`` of the selected rows (all by default)."""
    rows = range(len(model["boxes"])) if rows is None else rows
    detections = Detections(
        xyxy=torch.tensor(
            [model["boxes"][r] for r in rows], dtype=torch.float32
        ).reshape(-1, 4),
        class_id=torch.tensor([model["class_ids"][r] for r in rows], dtype=torch.int64),
        confidence=torch.tensor(
            [model["confidence"][r] for r in rows], dtype=torch.float32
        ),
        image_metadata={
            **frame_metadata(image_id),
            "class_names": dict(model["class_names"]),
        },
    )
    if namespace is None:
        return detections
    return with_row_identity(detections, namespace=namespace)


def box_masks(detections) -> torch.Tensor:
    masks = torch.zeros((len(detections), *SIZE_HW), dtype=torch.bool)
    for row, (x0, y0, x1, y1) in enumerate(detections.xyxy.int().tolist()):
        masks[row, y0:y1, x0:x1] = True
    return masks


def dense(model=PEOPLE, *, namespace="people") -> InstanceDetections:
    detections = boxes(model, namespace=namespace)
    return InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=box_masks(detections),
        image_metadata=detections.image_metadata,
    )


def rle(model=PEOPLE, *, namespace="people") -> InstanceDetections:
    instances = dense(model, namespace=namespace)
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))["counts"]
        for mask in instances.mask
    ]
    return dataclasses.replace(
        instances, mask=InstancesRLEMasks(image_size=SIZE_HW, masks=encoded)
    )


def grid_geometry() -> MaskGridGeometry:
    height, width = SIZE_HW
    return MaskGridGeometry(
        grid_size_hw=(height // STRIDE, width // STRIDE),
        inference_size_hw=SIZE_HW,
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(1.0, 1.0),
        crop_size_hw=SIZE_HW,
        original_size_hw=SIZE_HW,
        static_crop_xywh=(0, 0, width, height),
    )


def view(model=PEOPLE, *, namespace="people", threshold=0.0) -> SegmentationView:
    instances = dense(model, namespace=namespace)
    cells = torch.nn.functional.max_pool2d(
        instances.mask.float().unsqueeze(1), STRIDE
    ).squeeze(1)
    return SegmentationView(
        detections=boxes(model, namespace=namespace),
        scores=cells * 16.0 - 8.0,
        score_type="logits",
        threshold=threshold,
        geometry=grid_geometry(),
    )


def key_points(*, with_boxes: bool = True):
    detections = boxes(namespace=None)
    points = KeyPoints(
        xy=torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
        class_id=detections.class_id,
        confidence=torch.full((2, 2), 0.5),
        image_metadata=detections.image_metadata,
    )
    if not with_boxes:
        return with_row_identity((points, None), namespace="people")
    return with_row_identity((points, detections), namespace="people")


def as_sv(detections, *, with_masks: bool = False) -> sv.Detections:
    """The V1 ``sv.Detections`` of a native prediction, as V1 models emit it."""
    names = detections.image_metadata["class_names"]
    class_ids = detections.class_id.tolist()
    return sv.Detections(
        xyxy=detections.xyxy.numpy(),
        mask=detections.mask.numpy() if with_masks else None,
        confidence=detections.confidence.numpy(),
        class_id=np.array(class_ids),
        data={
            "class_name": np.array([names[c] for c in class_ids]),
            "detection_id": np.array([f"id-{i}" for i in range(len(class_ids))]),
        },
    )

"""Deterministic model stand-ins for transform workflows; no model download.

Test images paint objects as rectangles: channel 0 is 255 inside an object,
channel 1 holds the object's number (1..9) and channel 2 its class id. An
object's confidence is ``number / 10``.

``test/synthetic_segmenter@v1`` follows the G1 model recipe (decision 015):
row identity is attached last, and rows carry no legacy ``detection_id``.
It returns the same rows twice: dense ``predictions`` and a compact
``segmentation`` view on a stride-4 logit grid.
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from roboflow_workflows.v2.blocks import create_catalogue, transforms
from roboflow_workflows.v2.blocks.geometry import BLOCKS as GEOMETRY_BLOCKS
from roboflow_workflows.v2.blocks.image import DynamicCropBlock
from roboflow_workflows.v2.blocks.query import compile_operations
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import DICTIONARY_KIND
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    MaskGridGeometry,
    SegmentationView,
)
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    with_classification_parent,
    with_row_identity,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "car", 1: "person", 2: "bus"}
STRIDE = 4

# (x0, y0, x1, y1, object number, class id); boxes are exclusive at x1/y1.
Marker = Tuple[int, int, int, int, int, int]

# Street: car 0.9, person 0.8, car 0.4, bus 0.7, person 0.3.
STREET = [
    (4, 4, 20, 16, 9, 0),
    (24, 4, 32, 24, 8, 1),
    (36, 8, 56, 20, 4, 0),
    (4, 24, 28, 36, 7, 2),
    (40, 24, 48, 40, 3, 1),
]


def painted_image(size_hw: Tuple[int, int], markers: Sequence[Marker]) -> np.ndarray:
    """Return an RGB ``uint8`` image with the given markers painted."""
    pixels = np.zeros((*size_hw, 3), dtype=np.uint8)
    for x0, y0, x1, y1, number, class_id in markers:
        pixels[y0:y1, x0:x1] = (255, number, class_id)

    return pixels


class SyntheticSegmenterBlock(Block):
    """Segment painted markers; returns dense predictions and a compact view."""

    type = "test/synthetic_segmenter@v1"
    outputs = {
        "predictions": Output(INSTANCE_SEGMENTATION_PREDICTION_KIND, source="image"),
        "segmentation": Output(INSTANCE_SEGMENTATION_VIEW_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to segment.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        pixels = image.tensor_image
        height, width = pixels.shape[1:]
        inside = pixels[0] == 255
        numbers = torch.unique(pixels[1][inside]).tolist()
        boxes, classes, scores, masks = [], [], [], []
        for number in numbers:
            mask = inside & (pixels[1] == number)
            ys, xs = torch.nonzero(mask, as_tuple=True)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            classes.append(int(pixels[2][ys[0], xs[0]]))
            scores.append(number / 10)
            masks.append(mask)
        predictions = InstanceDetections(
            xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            class_id=torch.tensor(classes, dtype=torch.int64),
            confidence=torch.tensor(scores, dtype=torch.float32),
            mask=(
                torch.stack(masks) if masks else torch.zeros((0, height, width)).bool()
            ),
            image_metadata={
                **FrameRef.from_image(image).prediction_metadata(),
                "class_names": dict(CLASS_NAMES),
            },
        )
        predictions = with_row_identity(predictions)
        view = SegmentationView(
            detections=predictions_as_boxes(predictions),
            scores=logit_grid(predictions.mask),
            score_type="logits",
            threshold=0.0,
            geometry=MaskGridGeometry(
                grid_size_hw=(height // STRIDE, width // STRIDE),
                inference_size_hw=(height, width),
                padding_ltrb=(0, 0, 0, 0),
                scale_xy=(1.0, 1.0),
                crop_size_hw=(height, width),
                original_size_hw=(height, width),
                static_crop_xywh=(0, 0, width, height),
            ),
            take_ownership=True,
        )

        return {"predictions": predictions, "segmentation": view}


def predictions_as_boxes(predictions: InstanceDetections) -> Detections:
    """The same rows and metadata (row record included) without masks."""
    boxes = Detections(
        xyxy=predictions.xyxy,
        class_id=predictions.class_id,
        confidence=predictions.confidence,
        image_metadata=predictions.image_metadata,
        bboxes_metadata=predictions.bboxes_metadata,
    )
    return boxes


def logit_grid(masks: torch.Tensor) -> torch.Tensor:
    """``+8`` where a stride cell touches the mask, ``-8`` elsewhere."""
    if masks.shape[0] == 0:
        height, width = masks.shape[1:]
        return torch.zeros((0, height // STRIDE, width // STRIDE))
    touched = F.max_pool2d(masks.float().unsqueeze(1), STRIDE).squeeze(1)
    return touched * 16.0 - 8.0


class ClassIdClassifierBlock(Block):
    """Classify a crop by the class id painted at its centre; names it ``seen-<id>``."""

    type = "test/class_id_classifier@v1"
    outputs = {"predictions": Output(CLASSIFICATION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Crop to classify.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        pixels = image.tensor_image
        height, width = pixels.shape[1:]
        seen = int(pixels[1][height // 2, width // 2])
        confidence = torch.zeros((1, 10))
        confidence[0, seen] = 1.0
        prediction = ClassificationPrediction(
            class_id=torch.tensor([seen]),
            confidence=confidence,
            images_metadata=[{"class_names": {i: f"seen-{i}" for i in range(10)}}],
        )
        prediction = with_classification_parent(prediction, image.parent_row)
        return {"predictions": prediction}


class ToDictionaryBlock(Block):
    """Export predictions with the UQL ``DetectionsToDictionary`` operation."""

    type = "test/detections_to_dictionary@v1"
    outputs = {"serialized": Output(DICTIONARY_KIND, source="predictions")}

    class Params(BlockParams):
        predictions: Ref(*transforms.carriers.PREDICTION_KINDS) = Field(
            description="Predictions to export."
        )

    def __init__(self) -> None:
        self._export = compile_operations([{"type": "DetectionsToDictionary"}])

    def run(self, *, predictions: Any) -> Dict[str, Any]:
        return {"serialized": self._export.apply(predictions)}


def transforms_catalogue() -> Catalogue:
    """The V2 catalogue plus transforms, crop/replacement blocks and stand-ins."""
    catalogue = create_catalogue().with_blocks(
        [
            *transforms.BLOCKS,
            *GEOMETRY_BLOCKS,
            DynamicCropBlock,
            SyntheticSegmenterBlock,
            ClassIdClassifierBlock,
            ToDictionaryBlock,
        ]
    )
    return catalogue

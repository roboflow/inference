"""Deterministic detector stand-ins for merge/combine workflows; no model download.

Test images paint objects as rectangles: channel 0 is 255 inside an object,
channel 1 holds the object's number (1..9) and channel 2 its painted class
(0 person, 1 car, 2 bus). An object's confidence is ``number / 10``.

``test/painted_detector@v1`` plays two models with their own class maps::

    model     paints it detects   its class map
    people    person              {0: person}
    vehicles  car, bus            {0: car, 1: bus}

Rows are ordered by object number. Each call attaches a fresh row identity
(decision 015). ``segmentation`` returns the same rows as a compact view on
a stride-4 logit grid.
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from roboflow_workflows.v2.blocks import combine, create_catalogue, transforms
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
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.object_detection import Detections

STRIDE = 4
MODELS = {
    "people": {0: (0, "person")},
    "vehicles": {1: (0, "car"), 2: (1, "bus")},
}

# (x0, y0, x1, y1, object number, painted class); boxes exclusive at x1/y1.
Marker = Tuple[int, int, int, int, int, int]

# person 0.9, car 0.8, person 0.6, bus 0.7, car 0.4 on a 40x60 image.
STREET = [
    (4, 4, 12, 20, 9, 0),
    (12, 16, 28, 28, 8, 1),
    (32, 8, 40, 32, 6, 0),
    (40, 4, 56, 24, 7, 2),
    (20, 28, 36, 36, 4, 1),
]


def painted_image(size_hw: Tuple[int, int], markers: Sequence[Marker]) -> np.ndarray:
    """Return an RGB ``uint8`` image with the given markers painted."""
    pixels = np.zeros((*size_hw, 3), dtype=np.uint8)
    for x0, y0, x1, y1, number, painted_class in markers:
        pixels[y0:y1, x0:x1] = (255, number, painted_class)

    return pixels


class PaintedDetectorBlock(Block):
    """Detect painted objects of one model's classes; boxes and a compact view."""

    type = "test/painted_detector@v1"
    outputs = {
        "predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image"),
        "segmentation": Output(INSTANCE_SEGMENTATION_VIEW_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to detect on.")
        model: str = Field(description="people or vehicles.")

    def run(self, *, image: ImageData, model: str) -> Dict[str, Any]:
        classes = MODELS[model]
        pixels = image.tensor_image
        height, width = pixels.shape[1:]
        inside = pixels[0] == 255
        boxes, class_ids, scores, masks = [], [], [], []
        for number in sorted(torch.unique(pixels[1][inside]).tolist(), reverse=True):
            mask = inside & (pixels[1] == number)
            ys, xs = torch.nonzero(mask, as_tuple=True)
            painted = int(pixels[2][ys[0], xs[0]])
            if painted not in classes:
                continue
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            class_ids.append(classes[painted][0])
            scores.append(number / 10)
            masks.append(mask)
        detections = Detections(
            xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            class_id=torch.tensor(class_ids, dtype=torch.int64),
            confidence=torch.tensor(scores, dtype=torch.float32),
            image_metadata={
                **FrameRef.from_image(image).prediction_metadata(),
                "class_names": dict(classes.values()),
            },
        )
        detections = with_row_identity(detections)
        stacked = torch.stack(masks) if masks else torch.zeros((0, height, width))
        view = SegmentationView(
            detections=detections,
            scores=logit_grid(stacked.float(), size_hw=(height, width)),
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

        return {"predictions": detections, "segmentation": view}


def logit_grid(masks: torch.Tensor, *, size_hw: Tuple[int, int]) -> torch.Tensor:
    """``+8`` where a stride cell touches the mask, ``-8`` elsewhere."""
    if masks.shape[0] == 0:
        height, width = size_hw
        return torch.zeros((0, height // STRIDE, width // STRIDE))
    touched = F.max_pool2d(masks.unsqueeze(1), STRIDE).squeeze(1)
    return touched * 16.0 - 8.0


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


def combine_catalogue() -> Catalogue:
    """The V2 catalogue plus combine, transforms and the stand-ins."""
    catalogue = create_catalogue().with_blocks(
        [*combine.BLOCKS, *transforms.BLOCKS, PaintedDetectorBlock, ToDictionaryBlock]
    )
    return catalogue

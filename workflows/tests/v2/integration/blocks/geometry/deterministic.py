"""Deterministic stand-ins for a detector and a classifier, for static tests.

Test images paint objects as rectangles: channel 0 is 255 inside an object,
channel 1 holds the object's number (1, 2, ...) and channel 2 its class id.
``MarkerDetectorBlock`` returns the box of every object number it sees, so a
slice or crop finds exactly the part of each object it contains.
``ColorClassifierBlock`` classifies an image by its brightest mean channel.

Both follow the G1 model recipe (decision 015): the detector mints row identity
for each result as its last step; the classifier records ``image.parent_row``.
``RowSelectionBlock`` stands in for a filter or sort: it keeps given rows in a
given order through the shared row selection. ``ColorNameBlock`` returns a
plain string, like OCR. ``MaskedMarkerDetectorBlock`` returns instance
segmentation as predictions and as a ``SegmentationView`` over the same rows.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.geometry import BLOCKS as GEOMETRY_BLOCKS
from roboflow_workflows.v2.blocks.image import (
    AbsoluteStaticCropBlock,
    DynamicCropBlock,
    ImagePreprocessingBlock,
    ImageSlicerBlock,
    RelativeStaticCropBlock,
)
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
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

CLASS_NAMES = {0: "box", 1: "disc"}
COLOR_NAMES = {0: "red", 1: "green", 2: "blue"}

# (x0, y0, x1, y1, object number, class id); boxes are exclusive at x1/y1.
Marker = Tuple[int, int, int, int, int, int]


def painted_image(size_hw: Tuple[int, int], markers: Sequence[Marker]) -> np.ndarray:
    """Return an RGB ``uint8`` image with the given markers painted."""
    pixels = np.zeros((*size_hw, 3), dtype=np.uint8)
    for x0, y0, x1, y1, number, class_id in markers:
        pixels[y0:y1, x0:x1] = (255, number, class_id)

    return pixels


class MarkerDetectorBlock(Block):
    """Detect painted markers; confidence is ``0.5 + visible area / 1000``.

    A marker cut by the image edge is smaller, so it is less confident.
    """

    type = "test/marker_detector@v1"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to search.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        pixels = image.tensor_image
        inside = pixels[0] == 255
        numbers = torch.unique(pixels[1][inside]).tolist()
        boxes, classes, scores = [], [], []
        for number in numbers:
            ys, xs = torch.nonzero(inside & (pixels[1] == number), as_tuple=True)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            classes.append(int(pixels[2][ys[0], xs[0]]))
            scores.append(0.5 + len(xs) / 1000)
        detections = Detections(
            xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            class_id=torch.tensor(classes, dtype=torch.int64),
            confidence=torch.tensor(scores, dtype=torch.float32),
            image_metadata={
                **FrameRef.from_image(image).prediction_metadata(),
                "class_names": dict(CLASS_NAMES),
            },
            bboxes_metadata=[{"marker": int(n)} for n in numbers],
        )
        detections = with_row_identity(detections)

        return {"predictions": detections}


class ColorClassifierBlock(Block):
    """Classify an image as red, green or blue by its brightest mean channel."""

    type = "test/color_classifier@v1"
    outputs = {"predictions": Output(CLASSIFICATION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to classify.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        means = image.tensor_image.to(torch.float32).mean(dim=(1, 2))
        confidence = (means / means.sum().clamp(min=1)).reshape(1, -1)
        prediction = ClassificationPrediction(
            class_id=torch.argmax(means).reshape(1),
            confidence=confidence,
            images_metadata=[{"class_names": dict(COLOR_NAMES)}],
        )
        prediction = with_classification_parent(prediction, image.parent_row)

        return {"predictions": prediction}


class RowSelectionBlock(Block):
    """Keep the rows at ``rows``, in that order: a stand-in filter and sort."""

    type = "test/row_selection@v1"
    outputs = {
        "predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="predictions")
    }

    class Params(BlockParams):
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Detections to select from."
        )
        rows: List[int] = Field(description="Row positions to keep, in order.")

    def run(self, *, predictions: Any, rows: List[int]) -> Dict[str, Any]:
        rows = [row for row in rows if row < len(predictions.xyxy)]
        selected = select_predictions(predictions, indices=rows)

        return {"predictions": selected}


class ColorNameBlock(Block):
    """Name an image's brightest mean channel as plain text, like OCR."""

    type = "test/color_name@v1"
    outputs = {"text": Output(STRING_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to name.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        means = image.tensor_image.to(torch.float32).mean(dim=(1, 2))

        return {"text": COLOR_NAMES[int(torch.argmax(means))]}


class MaskedMarkerDetectorBlock(Block):
    """``MarkerDetectorBlock`` with each marker's pixels as its mask."""

    type = "test/masked_marker_detector@v1"
    outputs = {
        "predictions": Output(INSTANCE_SEGMENTATION_PREDICTION_KIND, source="image"),
        "segmentation": Output(INSTANCE_SEGMENTATION_VIEW_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to search.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        boxes = MarkerDetectorBlock().run(image=image)["predictions"]
        pixels = image.tensor_image
        numbers = torch.tensor([row["marker"] for row in boxes.bboxes_metadata])
        mask = (pixels[0] == 255).unsqueeze(0) & (
            pixels[1].unsqueeze(0) == numbers.reshape(-1, 1, 1).to(pixels.dtype)
        )
        predictions = InstanceDetections(
            xyxy=boxes.xyxy,
            class_id=boxes.class_id,
            confidence=boxes.confidence,
            mask=mask,
            image_metadata=boxes.image_metadata,
            bboxes_metadata=boxes.bboxes_metadata,
        )

        return {
            "predictions": predictions,
            "segmentation": SegmentationView.from_dense(predictions),
        }


REGION_IMAGE_BLOCKS = (
    AbsoluteStaticCropBlock,
    RelativeStaticCropBlock,
    DynamicCropBlock,
    ImagePreprocessingBlock,
    ImageSlicerBlock,
)


def regions_catalogue() -> Catalogue:
    """The V2 catalogue plus the ten region blocks and both test stand-ins."""
    catalogue = create_catalogue().with_blocks(
        [
            *REGION_IMAGE_BLOCKS,
            *GEOMETRY_BLOCKS,
            MarkerDetectorBlock,
            ColorClassifierBlock,
            RowSelectionBlock,
            ColorNameBlock,
            MaskedMarkerDetectorBlock,
        ]
    )

    return catalogue

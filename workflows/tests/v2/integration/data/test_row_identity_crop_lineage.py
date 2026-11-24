"""Crop lineage through a compiled workflow: select, crop, resize, classify, replace.

The decision 015 case: detections ``[A, B, C]`` are filtered, sorted or
repeated; a crop of the first selected row passes through the real resize
block and a classifier; the replacement updates the original row the crop
came from. The crop carries a ``RowRef``, never a position in the selected
list. A compact SegmentationView of the same detections follows the same
selection and keeps its row ids without materializing masks.
"""

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.image.primitives.resize.v1 import ResizeBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    MaskGridGeometry,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    NATIVE_KINDS,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    RowRef,
    classification_parent,
    require_row_record,
    with_classification_parent,
    with_row_identity,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.object_detection import Detections

LABELS = ("A", "B", "C")
# Selection before the crop; the crop takes the first selected row.
SCENARIOS = {
    "filter": ([1], ["A", "classified", "C"]),
    "sort": ([2, 0, 1], ["A", "B", "classified"]),
    "repeat": ([1, 1], ["A", "classified", "C"]),
}


class _Detect(Block):
    type = "row_lineage_test/detect"
    outputs = {
        "predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image"),
        "view": Output(INSTANCE_SEGMENTATION_VIEW_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to detect on.")

    def run(self, *, image):
        detections = Detections(
            xyxy=torch.tensor(
                [
                    [2.0, 2.0, 10.0, 10.0],
                    [12.0, 2.0, 20.0, 10.0],
                    [22.0, 2.0, 30.0, 10.0],
                ]
            ),
            class_id=torch.arange(3),
            confidence=torch.full((3,), 0.5),
            image_metadata=image.prediction_metadata(),
            bboxes_metadata=[{"label": label} for label in LABELS],
        )
        identified = with_row_identity(detections)
        view = SegmentationView(
            detections=identified,
            scores=torch.randn(3, 4, 4),
            score_type="logits",
            threshold=0.0,
            geometry=MaskGridGeometry.identity((4, 4)),
        )
        return {"predictions": identified, "view": view}


def _select_block(indices):
    class _Select(Block):
        type = f"row_lineage_test/select_{'_'.join(map(str, indices))}"
        outputs = {
            "predictions": Output(
                OBJECT_DETECTION_PREDICTION_KIND, source="predictions"
            ),
            "view": Output(INSTANCE_SEGMENTATION_VIEW_KIND, source="predictions"),
        }

        class Params(BlockParams):
            predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
                description="Predictions to select from."
            )
            view: Ref(INSTANCE_SEGMENTATION_VIEW_KIND) = Field(
                description="Compact view of the same predictions."
            )

        def run(self, *, predictions, view):
            return {
                "predictions": select_predictions(predictions, indices=indices),
                "view": view.select(torch.tensor(indices)),
            }

    return _Select


class _CropFirstRow(Block):
    """Stand-in for dynamic crop: one crop that refers to its source row."""

    type = "row_lineage_test/crop_first_row"
    outputs = {"crop": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to crop.")
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Rows to crop."
        )

    def run(self, *, image, predictions):
        record = require_row_record(predictions, consumer=self.type)
        ids, rectangles = record.host_ids_with(predictions.xyxy.to(torch.int64))
        crop = image.crop(rectangles[0]).with_parent_row(
            RowRef(record.namespace, ids[0])
        )
        return {"crop": crop}


class _Classify(Block):
    type = "row_lineage_test/classify"
    outputs = {"prediction": Output(CLASSIFICATION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Crop to classify.")

    def run(self, *, image):
        prediction = ClassificationPrediction(
            class_id=torch.tensor([0]),
            confidence=torch.tensor([[0.9]]),
            images_metadata=[{}],
        )
        return {"prediction": with_classification_parent(prediction, image.parent_row)}


class _Replace(Block):
    """Stand-in for class replacement: joins by row key only."""

    type = "row_lineage_test/replace"
    outputs = {
        "labels": Output(LIST_OF_VALUES_KIND, source="original"),
        "view_ids": Output(LIST_OF_VALUES_KIND, source="original"),
    }

    class Params(BlockParams):
        original: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Target rows."
        )
        classification: Ref(CLASSIFICATION_PREDICTION_KIND) = Field(
            description="Classification of one crop."
        )
        view: Ref(INSTANCE_SEGMENTATION_VIEW_KIND) = Field(
            description="Selected compact view."
        )

    def run(self, *, original, classification, view):
        targets = require_row_record(original, consumer=self.type)
        parent = classification_parent(classification)
        assert parent.namespace == targets.namespace
        labels = [
            "classified" if row_id == parent.id else row["label"]
            for row_id, row in zip(targets.host_ids(), original.bboxes_metadata)
        ]
        view_record = require_row_record(view.detections, consumer=self.type)
        assert view_record.namespace == targets.namespace
        assert view.materialization_count == 0
        return {"labels": labels, "view_ids": view_record.host_ids()}


def _plan(indices):
    select = _select_block(indices)
    catalogue = Catalogue(
        [ResizeBlock, _Detect, select, _CropFirstRow, _Classify, _Replace],
        kinds=[
            IMAGE_KIND,
            *NATIVE_KINDS,
            INSTANCE_SEGMENTATION_VIEW_KIND,
            LIST_OF_VALUES_KIND,
        ],
    )
    steps = [
        {"type": _Detect.type, "name": "detect", "image": "$inputs.images"},
        {
            "type": select.type,
            "name": "select",
            "predictions": "$steps.detect.predictions",
            "view": "$steps.detect.view",
        },
        {
            "type": _CropFirstRow.type,
            "name": "crop",
            "image": "$inputs.images",
            "predictions": "$steps.select.predictions",
        },
        {
            "type": "roboflow_core/resize@v1",
            "name": "resize",
            "image": "$steps.crop.crop",
            "width": 4,
            "height": 4,
        },
        {"type": _Classify.type, "name": "classify", "image": "$steps.resize.image"},
        {
            "type": _Replace.type,
            "name": "replace",
            "original": "$steps.detect.predictions",
            "classification": "$steps.classify.prediction",
            "view": "$steps.select.view",
        },
    ]
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "images"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.replace.{name}"}
            for name in ("labels", "view_ids")
        ],
    }
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_classification_of_a_crop_replaces_its_original_row(scenario):
    indices, expected = SCENARIOS[scenario]
    images = [
        ImageData.from_tensor(
            torch.zeros((3, 30, 40), dtype=torch.uint8), image_id=name
        )
        for name in ("first", "second")
    ]

    rows = _plan(indices).create_session().run({"images": images}).rows()

    assert len(rows) == 2
    for row in rows:
        labels, view_ids = row["labels"], row["view_ids"]
        assert labels == expected
        assert view_ids == indices

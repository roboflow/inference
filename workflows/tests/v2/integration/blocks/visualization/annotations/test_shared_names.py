"""A list ``class_names`` map names rows alike in every consumer (HK-1).

Before decision 034, the painters read a list map but the query and the
per-class filter called ``.get`` on it and failed. One compiled workflow now
sends rows named by a list map to every class-name consumer::

    named ─┬─▶ per_class_confidence_filter   box, disc >= 0.6; others >= 0
           ├─▶ detections_filter              class_name in ["box"]
           ├─▶ detection_drawing ─▶ label_visualization (prepared labels)
           └─▶ label_visualization                      (native rows)

Every row has confidence 0.5, so a row survives the per-class filter exactly
when it is not named box or disc. Row ``i``'s box starts at ``x = 10 * i``.
"""

from typing import Any, Dict

import numpy as np
import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks import create_catalogue, transforms
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.object_detection import Detections

MISSING = object()
CLASS_IDS = [0, 1, 0, 1, 0]
# Row values: 7 -> box, None -> disc, "car", missing -> disc, "" (own name)
ROW_CLASSES = [7, None, "car", MISSING, ""]
NAMES = {"list": ["box", "disc"], "dict": {0: "box", 1: "disc"}}


class ListNamedDetectorBlock(Block):
    """Return five rows at confidence 0.5, named by a list or dict map."""

    type = "test/list_named_detector@v1"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image the rows belong to.")
        names: str = Field(description="Key of NAMES.")

    def run(self, *, image: ImageData, names: str) -> Dict[str, Any]:
        count = len(CLASS_IDS)
        image_metadata = FrameRef.from_image(image).prediction_metadata()
        image_metadata["class_names"] = NAMES[names]
        detections = Detections(
            xyxy=torch.tensor(
                [[10 * row, 5, 10 * row + 8, 15] for row in range(count)],
                dtype=torch.float32,
            ),
            class_id=torch.tensor(CLASS_IDS, dtype=torch.int64),
            confidence=torch.full((count,), 0.5),
            image_metadata=image_metadata,
            bboxes_metadata=[
                {} if value is MISSING else {"class": value} for value in ROW_CLASSES
            ],
        )

        return {"predictions": detections}


BOXES_ONLY = {
    "type": "DetectionsFilter",
    "filter_operation": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {
                    "type": "DynamicOperand",
                    "operations": [
                        {"type": "ExtractDetectionProperty", "property_name": "class_name"}
                    ],
                },
                "comparator": {"type": "in (Sequence)"},
                "right_operand": {"type": "StaticOperand", "value": ["box"]},
            }
        ],
    },
}  # fmt: skip


def run(names: str) -> dict:
    steps = [
        {"type": "test/list_named_detector@v1", "name": "named", "image": "$inputs.image", "names": names},
        {
            "type": "v2/per_class_confidence_filter@v1",
            "name": "confident",
            "predictions": "$steps.named.predictions",
            "class_thresholds": {"box": 0.6, "disc": 0.6},
            "default_threshold": 0.0,
        },
        {
            "type": "v2/detections_filter@v1",
            "name": "boxes",
            "predictions": "$steps.named.predictions",
            "operations": [BOXES_ONLY],
        },
        {"type": "v2/detection_drawing@v1", "name": "drawing", "predictions": "$steps.named.predictions"},
        {
            "type": "v2/label_visualization@v1",
            "name": "native_labels",
            "image": "$inputs.image",
            "predictions": "$steps.named.predictions",
        },
        {
            "type": "v2/label_visualization@v1",
            "name": "prepared_labels",
            "image": "$inputs.image",
            "predictions": "$steps.drawing.drawing",
        },
    ]  # fmt: skip
    outputs = {
        "confident": "$steps.confident.predictions",
        "boxes": "$steps.boxes.predictions",
        "drawing": "$steps.drawing.drawing",
        "native_labels": "$steps.native_labels.image",
        "prepared_labels": "$steps.prepared_labels.image",
    }
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = create_catalogue().with_blocks(
        [*transforms.BLOCKS, ListNamedDetectorBlock]
    )
    plan = compile_workflow(workflow, catalogue=catalogue)
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    row = plan.create_session().run({"image": [image]}).rows()[0]

    return row


def rows_at(predictions: Detections) -> list:
    return [int(x_min) // 10 for x_min in predictions.xyxy[:, 0].tolist()]


@pytest.mark.parametrize("names", sorted(NAMES))
def test_list_and_dict_maps_name_rows_alike_in_every_consumer(names: str) -> None:
    # when
    row = run(names)

    # then
    assert row["drawing"].labels == (
        "box 0.50",
        "disc 0.50",
        "car 0.50",
        "disc 0.50",
        " 0.50",
    )
    assert rows_at(row["boxes"]) == [0]
    assert rows_at(row["confident"]) == [2, 4]
    native, prepared = row["native_labels"], row["prepared_labels"]
    assert native.tensor_image.equal(prepared.tensor_image)

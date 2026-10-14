"""Canonical accurate/fast labels preserve relabelled row names through compilation.

Class replacement, model combination and query consumers share the canonical
label name resolver. Both label implementations must paint the same supported
style while the workflow retains native detections and per-row names.
"""

from typing import Any, Dict

import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks import create_catalogue, transforms
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization.v2 import (
    label_texts,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import read_rows
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.object_detection import Detections

MISSING = object()
# case -> (class ids, each row's "class" value, class_names map)
CASES = {
    "malformed": ([0, 1, 0, 1, 0], [7, None, "car", MISSING, ""], {0: "box", 1: "disc"}),
    "strings_without_map": ([3, 4, 3], ["car", "", "box"], None),
}  # fmt: skip


class RowNamedDetectorBlock(Block):
    """Return the rows of one ``CASES`` entry, each at confidence 0.5."""

    type = "test/row_named_detector@v1"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image the rows belong to.")
        case: str = Field(description="Key of CASES.")

    def run(self, *, image: ImageData, case: str) -> Dict[str, Any]:
        class_ids, row_classes, class_names = CASES[case]
        count = len(class_ids)
        image_metadata = FrameRef.from_image(image).prediction_metadata()
        if class_names is not None:
            image_metadata["class_names"] = dict(class_names)
        detections = Detections(
            xyxy=torch.tensor(
                [[10 * row, 5, 10 * row + 8, 15] for row in range(count)],
                dtype=torch.float32,
            ),
            class_id=torch.tensor(class_ids, dtype=torch.int64),
            confidence=torch.full((count,), 0.5),
            image_metadata=image_metadata,
            bboxes_metadata=[
                {} if value is MISSING else {"class": value} for value in row_classes
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


def run(case: str) -> dict:
    steps = [
        {"type": "test/row_named_detector@v1", "name": "named", "image": "$inputs.image", "case": case},
        {
            "type": "roboflow_core/per_class_confidence_filter@v1",
            "name": "confident",
            "predictions": "$steps.named.predictions",
            "class_thresholds": {"box": 0.6, "disc": 0.6},
            "default_threshold": 0.0,
        },
        {
            "type": "roboflow_core/detections_filter@v1",
            "name": "boxes",
            "predictions": "$steps.named.predictions",
            "operations": [BOXES_ONLY],
        },
        {
            "type": "roboflow_core/label_visualization@v2",
            "text": "Class and Confidence",
            "name": "accurate_labels",
            "image": "$inputs.image",
            "predictions": "$steps.named.predictions",
        },
        {
            "type": "roboflow_core/label_visualization@v2",
            "text": "Class and Confidence",
            "name": "fast_labels",
            "image": "$inputs.image",
            "predictions": "$steps.named.predictions",
        },
    ]  # fmt: skip
    outputs = {
        "confident": "$steps.confident.predictions",
        "boxes": "$steps.boxes.predictions",
        "named_predictions": "$steps.named.predictions",
        "accurate_labels": "$steps.accurate_labels.image",
        "fast_labels": "$steps.fast_labels.image",
    }
    workflow = {
        "version": "2.0",
        "execution": {"step_quality": {"$steps.fast_labels": "fast"}},
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = create_catalogue().with_blocks(
        [*transforms.BLOCKS, RowNamedDetectorBlock]
    )
    plan = compile_workflow(workflow, catalogue=catalogue)
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    row = plan.create_session().run({"image": [image]}).rows()[0]

    return row


def rows_at(predictions: Detections) -> list:
    return [int(x_min) // 10 for x_min in predictions.xyxy[:, 0].tolist()]


def assert_same_label_pixels(row: dict) -> None:
    native, prepared = row["accurate_labels"], row["fast_labels"]
    assert native.tensor_image.shape == prepared.tensor_image.shape
    assert native.tensor_image.equal(prepared.tensor_image)


def test_non_string_row_values_are_named_by_the_map_in_every_consumer() -> None:
    # when: rows 7 -> box, None -> disc, "car", missing -> disc, "" (own name)
    row = run("malformed")

    # then
    assert tuple(
        label_texts(
            read_rows(row["named_predictions"]),
            "Class and Confidence",
            masks=None,
            block="test",
        )
    ) == (
        "box 0.50",
        "disc 0.50",
        "car 0.50",
        "disc 0.50",
        " 0.50",
    )
    assert rows_at(row["boxes"]) == [0]
    assert rows_at(row["confident"]) == [2, 4]
    assert_same_label_pixels(row)


def test_string_row_names_need_no_map_in_any_consumer() -> None:
    # when
    row = run("strings_without_map")

    # then
    assert tuple(
        label_texts(
            read_rows(row["named_predictions"]),
            "Class and Confidence",
            masks=None,
            block="test",
        )
    ) == ("car 0.50", " 0.50", "box 0.50")
    assert rows_at(row["boxes"]) == [2]
    assert rows_at(row["confident"]) == [0, 1]
    assert_same_label_pixels(row)

"""Drawings of relabelled rows through the real V2 compiler and session.

Two workflows give rows a shared class id but different names (decision 025)::

    detect ─▶ dynamic_crop ─▶ color_name (text, like OCR) ─▶ classes_replacement
    people ──┐                                                (crops bound)
             ├─▶ detections_combine     both models use class id 0
    vehicles ┘

Each result goes to ``roboflow_core/detection_drawing@v1`` and to
``roboflow_core/simple_label_visualization@v1`` twice: once with the native predictions and once
with the prepared drawing. Both painter routes must give the same pixels.
"""

import numpy as np
from roboflow_workflows.v2.blocks import transforms
from roboflow_workflows.v2.blocks.combine import DetectionsCombineBlock
from roboflow_workflows.v2.engine.compilation import compile_workflow

from tests.v2.integration.blocks.combine.synthetic import STREET, PaintedDetectorBlock
from tests.v2.integration.blocks.combine.synthetic import painted_image as street_image
from tests.v2.integration.blocks.geometry.deterministic import (
    painted_image,
    regions_catalogue,
)


def drawing_steps(predictions: str) -> list:
    steps = [
        {"type": "roboflow_core/detection_drawing@v1", "name": "drawing", "predictions": predictions},
        {
            "type": "roboflow_core/simple_label_visualization@v1",
            "name": "native_labels",
            "image": "$inputs.image",
            "predictions": predictions,
        },
        {
            "type": "roboflow_core/simple_label_visualization@v1",
            "name": "prepared_labels",
            "image": "$inputs.image",
            "predictions": "$steps.drawing.drawing",
        },
    ]  # fmt: skip

    return steps


DRAWING_OUTPUTS = {
    "drawing": "$steps.drawing.drawing",
    "native_labels": "$steps.native_labels.image",
    "prepared_labels": "$steps.prepared_labels.image",
}


def run(steps: list, outputs: dict, image: np.ndarray) -> dict:
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in {**outputs, **DRAWING_OUTPUTS}.items()
        ],
    }
    catalogue = regions_catalogue().with_blocks(
        [DetectionsCombineBlock, *transforms.BLOCKS, PaintedDetectorBlock]
    )
    plan = compile_workflow(workflow, catalogue=catalogue)
    row = plan.create_session().run({"image": [image]}).rows()[0]

    return row


def assert_same_pixels(row: dict) -> None:
    native, prepared = row["native_labels"], row["prepared_labels"]
    assert native.tensor_image.shape == prepared.tensor_image.shape
    assert native.tensor_image.equal(prepared.tensor_image)


def plates_image() -> np.ndarray:
    # Marker 1 (10x10) is pure marker colour (255, 1, 0): its crop reads "red".
    # Marker 2 (20x20) has a (100, 250, 200) centre: its crop reads "green".
    pixels = painted_image((40, 60), [(5, 5, 15, 15, 1, 0), (30, 10, 50, 30, 2, 0)])
    pixels[12:28, 32:48] = (100, 250, 200)

    return pixels


OCR_STEPS = [
    {"type": "test/marker_detector@v1", "name": "detect", "image": "$inputs.image"},
    {
        "type": "roboflow_core/dynamic_crop@v1",
        "name": "crop",
        "image": "$inputs.image",
        "predictions": "$steps.detect.predictions",
    },
    {"type": "test/color_name@v1", "name": "read", "image": "$steps.crop.crops"},
    {
        "type": "roboflow_core/detections_classes_replacement@v1",
        "name": "relabel",
        "object_detection_predictions": "$steps.detect.predictions",
        "classification_predictions": "$steps.read.text",
        "crops": "$steps.crop.crops",
    },
]


def test_each_text_read_from_a_crop_labels_its_own_row() -> None:
    # when
    row = run(
        [*OCR_STEPS, *drawing_steps("$steps.relabel.predictions")],
        {"relabelled": "$steps.relabel.predictions"},
        plates_image(),
    )

    # then: both texts share class id 0 and confidence 1.0, as in V1; the
    # map names only the first text
    relabelled = row["relabelled"]
    assert relabelled.class_id.tolist() == [0, 0]
    assert relabelled.image_metadata["class_names"] == {0: "red"}
    assert [r["class"] for r in relabelled.bboxes_metadata] == ["red", "green"]
    assert row["drawing"].labels == ("red 1.00", "green 1.00")
    assert row["drawing"].class_id.tolist() == [0, 0]
    assert_same_pixels(row)


def test_combined_models_with_clashing_class_ids_keep_each_models_names() -> None:
    # given: people {0: person}, vehicles {0: car, 1: bus}
    steps = [
        {"type": "test/painted_detector@v1", "name": "people", "image": "$inputs.image", "model": "people"},
        {"type": "test/painted_detector@v1", "name": "vehicles", "image": "$inputs.image", "model": "vehicles"},
        {
            "type": "roboflow_core/detections_combine@v1",
            "name": "combined",
            "prediction_one": "$steps.people.predictions",
            "prediction_two": "$steps.vehicles.predictions",
        },
        {
            "type": "roboflow_core/detections_filter@v1",
            "name": "cars",
            "predictions": "$steps.combined.predictions",
            "operations": [
                {
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
                                "right_operand": {"type": "StaticOperand", "value": ["car"]},
                            }
                        ],
                    },
                }
            ],
        },
        *drawing_steps("$steps.combined.predictions"),
    ]  # fmt: skip

    # when
    row = run(
        steps,
        {"combined": "$steps.combined.predictions", "cars": "$steps.cars.predictions"},
        street_image((40, 60), STREET),
    )

    # then: drawing and the class-name filter agree on which rows are cars
    assert row["combined"].class_id.tolist() == [0, 0, 0, 1, 0]
    assert row["combined"].image_metadata["class_names"] == {0: "person", 1: "bus"}
    assert row["drawing"].labels == (
        "person 0.90",
        "person 0.60",
        "car 0.80",
        "bus 0.70",
        "car 0.40",
    )
    assert row["cars"].confidence.tolist() == [
        float(np.float32(0.8)),
        float(np.float32(0.4)),
    ]
    assert_same_pixels(row)


def test_rows_without_own_names_draw_as_before() -> None:
    # when
    row = run(
        [
            {"type": "test/painted_detector@v1", "name": "people", "image": "$inputs.image", "model": "people"},
            *drawing_steps("$steps.people.predictions"),
        ],
        {"people": "$steps.people.predictions"},
        street_image((40, 60), STREET),
    )  # fmt: skip

    # then
    assert row["people"].bboxes_metadata is None
    assert row["drawing"].labels == ("person 0.90", "person 0.60")
    assert_same_pixels(row)

"""Active providers decode predictions before crop, classification and drawing.

Requests use in-process fake OpenRouter transports. Florence-2's retained pure
helper also supplies predictions to downstream filters without a parser block.
"""

import json
from typing import Any, Dict

import numpy as np
import pytest
from roboflow_workflows.v2.blocks.image.primitives.static_crop.v1 import StaticCropBlock
from roboflow_workflows.v2.blocks.image.dynamic_crop.v1 import DynamicCropBlock
from roboflow_workflows.v2.blocks.models.vision_language_models.kimi_openrouter.v3 import (
    KimiOpenRouterBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.meta_vlm.v3 import (
    MetaVlmBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models._shared.parsers import (
    decode_florence_2_detections,
)
from roboflow_workflows.v2.blocks.transforms.detections_filter.v1 import DetectionsFilterBlock
from roboflow_workflows.v2.blocks.visualization.native.detection_drawing.v1 import DetectionDrawingBlock
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import (
    RowRef,
    classification_parent,
    row_record,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from tests.v2.support.language.fake_chat import FakeChatServer, chat_completion_body

CATALOGUE = Catalogue(
    [MetaVlmBlock, KimiOpenRouterBlock, DynamicCropBlock, StaticCropBlock]
)
IMAGE = np.zeros((200, 400, 3), dtype=np.uint8)
TWO_ANIMALS = json.dumps(
    [
        {"x_min": 0, "y_min": 0, "x_max": 250, "y_max": 500, "label": "dog"},
        {"x_min": 500, "y_min": 500, "x_max": 1000, "y_max": 1000, "label": "cat"},
    ]
)


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _detect_then_classify_crops() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/meta_vlm@v3",
                "name": "detector",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["dog", "cat"],
                "api_key": "sk-or-fake-parser-demo",
            },
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "crop",
                "image": "$inputs.image",
                "predictions": "$steps.detector.predictions",
            },
            {
                "type": "roboflow_core/kimi_openrouter@v3",
                "name": "classifier",
                "images": "$steps.crop.crops",
                "task_type": "classification",
                "classes": ["sleeping", "awake"],
                "api_key": "sk-or-fake-parser-demo",
            },
        ],
        "outputs": [
            _output("boxes", "$steps.detector.predictions"),
            _output("boxes_error", "$steps.detector.error_status"),
            _output("boxes_id", "$steps.detector.inference_id"),
            _output("labels", "$steps.classifier.predictions"),
        ],
    }
    return definition


def _run(
    definition,
    *,
    boxes_answer=TWO_ANIMALS,
    label_answer='{"class_name": "awake", "confidence": 0.75}',
    inputs=None
):
    def answer(path, request):
        content = boxes_answer if request["model"].startswith("meta/") else label_answer
        return 200, chat_completion_body(content)

    server = FakeChatServer(answer)
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session(
        resources={"openrouter_http_client": server.http_client()}
    )
    try:
        row = session.run(inputs or {"image": IMAGE}).rows()[0]
    finally:
        session.close()
    return row


def test_native_boxes_are_cropped_and_each_crop_label_joins_its_box() -> None:
    row = _run(_detect_then_classify_crops())
    boxes = row["boxes"]
    assert row["boxes_error"] is False
    assert boxes.xyxy.tolist() == [[0, 0, 100, 100], [200, 100, 400, 200]]
    assert [box["class"] for box in boxes.bboxes_metadata] == ["dog", "cat"]
    assert row_record(boxes).namespace == row["boxes_id"]
    labels = row["labels"]
    assert len(labels) == 2
    for position, label in enumerate(labels):
        assert label.class_id.tolist() == [1]
        assert float(label.confidence[0, 1]) == pytest.approx(0.75)
        assert classification_parent(label) == RowRef(row["boxes_id"], position)
    assert frame_of(labels[1]).size_hw == (100, 200)


def test_unreadable_answer_reports_error_and_skips_dependent_steps() -> None:
    row = _run(_detect_then_classify_crops(), boxes_answer="I cannot tell.")
    assert row["boxes_error"] is True
    assert row["boxes"] is None
    assert row["labels"] == []


def test_unreadable_classification_preserves_boxes_and_returns_no_labels() -> None:
    row = _run(_detect_then_classify_crops(), label_answer="not json")
    assert len(row["boxes"]) == 2
    assert row["labels"] == [None, None]


def test_boxes_read_on_static_crops_map_back_to_the_input_image() -> None:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/static_crop@v1",
                "name": "tiles",
                "image": "$inputs.image",
                "regions": [[0, 0, 200, 200], [200, 0, 400, 200]],
            },
            {
                "type": "roboflow_core/meta_vlm@v3",
                "name": "detector",
                "images": "$steps.tiles.crops",
                "task_type": "object-detection",
                "classes": ["cat"],
                "api_key": "sk-or-fake-parser-demo",
            },
        ],
        "outputs": [
            _output("in_input", "$steps.detector.predictions"),
            {
                **_output("in_tile", "$steps.detector.predictions"),
                "coordinates_system": "own",
            },
        ],
    }
    answer = (
        '[{"x_min": 250, "y_min": 100, "x_max": 750, "y_max": 500, "label": "cat"}]'
    )
    row = _run(definition, boxes_answer=answer)
    assert [p.xyxy.tolist() for p in row["in_tile"]] == [[[50, 20, 150, 100]]] * 2
    assert [p.xyxy.tolist() for p in row["in_input"]] == [
        [[50, 20, 150, 100]],
        [[250, 20, 350, 100]],
    ]


def test_florence_helper_rows_feed_filters_and_drawing_with_their_own_text() -> None:
    answer = {
        "quad_boxes": [
            [10, 20, 50, 15, 55, 40, 12, 45],
            [100, 100, 140, 100, 140, 120, 100, 120],
        ],
        "labels": ["STOP", "EXIT"],
    }
    error, boxes = decode_florence_2_detections(
        answer,
        image=ImageData.from_numpy_rgb(IMAGE),
        task_type="ocr-with-text-detection",
        classes=None,
        inference_id="florence-ocr",
    )
    assert error is False
    exit_only = {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": "class_name",
                            }
                        ],
                    },
                    "comparator": {"type": "in (Sequence)"},
                    "right_operand": {"type": "StaticOperand", "value": ["EXIT"]},
                }
            ],
        },
    }
    definition = {
        "version": "2.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "boxes",
                "kind": ["object_detection_prediction"],
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/detections_filter@v1",
                "name": "exits",
                "predictions": "$inputs.boxes",
                "operations": [exit_only],
            },
            {
                "type": "roboflow_core/detection_drawing@v1",
                "name": "drawing",
                "predictions": "$inputs.boxes",
            },
        ],
        "outputs": [
            _output("exits", "$steps.exits.predictions"),
            _output("drawing", "$steps.drawing.drawing"),
        ],
    }
    catalogue = Catalogue([DetectionsFilterBlock, DetectionDrawingBlock])
    session = compile_workflow(definition, catalogue=catalogue).create_session()
    try:
        row = session.run({"boxes": boxes}).rows()[0]
    finally:
        session.close()
    assert boxes.class_id.tolist() == [0, 0]
    assert boxes.image_metadata["class_names"] == {}
    assert row["exits"].xyxy.tolist() == [[100, 100, 140, 120]]
    assert row["exits"].bboxes_metadata == [{"class": "EXIT"}]
    assert row["drawing"].labels == ("STOP 1.00", "EXIT 1.00")


def test_detector_configuration_is_checked_at_compile_time() -> None:
    definition = _detect_then_classify_crops()
    del definition["steps"][0]["classes"]
    with pytest.raises(ParamsValidationError, match="classes"):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_selected_classes_are_checked_at_run_time() -> None:
    definition = _detect_then_classify_crops()
    definition["inputs"].append({"type": "WorkflowParameter", "name": "classes"})
    definition["steps"][0]["classes"] = "$inputs.classes"
    with pytest.raises(StepExecutionError, match="repeat"):
        _run(definition, inputs={"image": IMAGE, "classes": ["cat", "cat"]})

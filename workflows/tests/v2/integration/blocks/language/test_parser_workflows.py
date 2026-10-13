"""Compiled workflows that turn VLM answers into predictions.

The real compiler, session and executor run every workflow. Answers are
workflow inputs standing in for a VLM step, so expected boxes are exact.
"""

from typing import Any, Dict

import numpy as np
import pytest
from roboflow_workflows.v2.blocks.image.blocks import StaticCropBlock
from roboflow_workflows.v2.blocks.image.dynamic_crop import DynamicCropBlock
from roboflow_workflows.v2.blocks.language.parsers import (
    VLMAsClassifierBlock,
    VLMAsDetectorBlock,
)
from roboflow_workflows.v2.blocks.transforms.operations_chain import (
    DetectionsFilterBlock,
)
from roboflow_workflows.v2.blocks.visualization.blocks import DetectionDrawingBlock
from roboflow_workflows.v2.data.frames import frame_of
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

CATALOGUE = Catalogue(
    [VLMAsDetectorBlock, VLMAsClassifierBlock, DynamicCropBlock, StaticCropBlock]
)
IMAGE = np.zeros((200, 400, 3), dtype=np.uint8)
# Qwen answers in 0-1000 coordinates: x * 0.4, y * 0.2 on the 400 x 200 image.
TWO_ANIMALS = (
    "```json\n"
    '[{"box_2d": [0, 0, 250, 500], "label": "dog"},'
    ' {"box_2d": [500, 500, 1000, 1000], "label": "cat"}]\n'
    "```"
)


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _detect_then_classify_crops() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowParameter",
                "name": "boxes_answer",
                "kind": ["language_model_output"],
            },
            {"type": "WorkflowParameter", "name": "label_answer", "kind": ["string"]},
        ],
        "steps": [
            {
                "type": "v2/vlm_as_detector@v2",
                "name": "detector",
                "image": "$inputs.image",
                "vlm_output": "$inputs.boxes_answer",
                "model_type": "qwen",
                "classes": ["dog", "cat"],
            },
            {
                "type": "v2/dynamic_crop@v1",
                "name": "crop",
                "image": "$inputs.image",
                "predictions": "$steps.detector.predictions",
            },
            {
                "type": "v2/vlm_as_classifier@v2",
                "name": "classifier",
                "image": "$steps.crop.crops",
                "vlm_output": "$inputs.label_answer",
                "classes": ["sleeping", "awake"],
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


def _run(definition: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()
    try:
        row = session.run(inputs).rows()[0]
    finally:
        session.close()

    return row


def test_parsed_boxes_are_cropped_and_each_crop_label_joins_its_box() -> None:
    # when
    row = _run(
        _detect_then_classify_crops(),
        {
            "image": IMAGE,
            "boxes_answer": TWO_ANIMALS,
            "label_answer": '{"class_name": "awake", "confidence": 0.75}',
        },
    )

    # then
    boxes = row["boxes"]
    assert row["boxes_error"] is False
    assert boxes.xyxy.tolist() == [[0, 0, 100, 100], [200, 100, 400, 200]]
    assert [box["class"] for box in boxes.bboxes_metadata] == ["dog", "cat"]
    assert row_record(boxes).namespace == row["boxes_id"]
    labels = row["labels"]
    assert len(labels) == 2
    for position, label in enumerate(labels):
        assert label.class_id.tolist() == [1]
        assert classification_parent(label) == RowRef(row["boxes_id"], position)
    assert frame_of(labels[1]).size_hw == (100, 200)


def test_unreadable_answer_reports_error_and_skips_dependent_steps() -> None:
    # when
    row = _run(
        _detect_then_classify_crops(),
        {"image": IMAGE, "boxes_answer": "I cannot tell.", "label_answer": "{}"},
    )

    # then
    assert row["boxes_error"] is True
    assert row["boxes"] is None
    # The crop step is skipped, so the per-crop output is an empty group.
    assert row["labels"] == []


def test_decoded_dict_answer_is_accepted_as_language_model_output() -> None:
    answer = {"detections": [{"box_2d": [0, 0, 500, 500], "label": "cat"}]}

    row = _run(
        _detect_then_classify_crops(),
        {"image": IMAGE, "boxes_answer": answer, "label_answer": "not json"},
    )

    assert row["boxes"].xyxy.tolist() == [[0, 0, 200, 100]]
    assert row["labels"] == [None]


def test_boxes_read_on_static_crops_map_back_to_the_input_image() -> None:
    # given: Gemini answers yxyx in 0-1000 of each 200 x 200 crop.
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "answer"},
        ],
        "steps": [
            {
                "type": "v2/static_crop",
                "name": "tiles",
                "image": "$inputs.image",
                "regions": [[0, 0, 200, 200], [200, 0, 400, 200]],
            },
            {
                "type": "v2/vlm_as_detector@v2",
                "name": "detector",
                "image": "$steps.tiles.crops",
                "vlm_output": "$inputs.answer",
                "model_type": "google-gemini",
                "classes": ["cat"],
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
    answer = '[{"box_2d": [100, 250, 500, 750], "label": "cat"}]'

    # when
    row = _run(definition, {"image": IMAGE, "answer": answer})

    # then: each tile's box is x 50..150, y 20..100 in tile pixels.
    assert [p.xyxy.tolist() for p in row["in_tile"]] == [[[50, 20, 150, 100]]] * 2
    assert [p.xyxy.tolist() for p in row["in_input"]] == [
        [[50, 20, 150, 100]],
        [[250, 20, 350, 100]],
    ]


def test_florence_2_ocr_rows_are_filtered_and_labelled_by_their_own_text() -> None:
    # given: OCR rows share class id 0 and carry an empty class_names map.
    #   detector (OCR) ─┬─▶ detections_filter   class_name in ["EXIT"]
    #                   └─▶ detection_drawing   label texts
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
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "answer"},
        ],
        "steps": [
            {
                "type": "v2/vlm_as_detector@v2",
                "name": "detector",
                "image": "$inputs.image",
                "vlm_output": "$inputs.answer",
                "model_type": "florence-2",
                "task_type": "ocr-with-text-detection",
            },
            {
                "type": "v2/detections_filter@v1",
                "name": "exits",
                "predictions": "$steps.detector.predictions",
                "operations": [exit_only],
            },
            {
                "type": "v2/detection_drawing@v1",
                "name": "drawing",
                "predictions": "$steps.detector.predictions",
            },
        ],
        "outputs": [
            _output("boxes", "$steps.detector.predictions"),
            _output("exits", "$steps.exits.predictions"),
            _output("drawing", "$steps.drawing.drawing"),
        ],
    }
    answer = {
        "quad_boxes": [
            [10, 20, 50, 15, 55, 40, 12, 45],
            [100, 100, 140, 100, 140, 120, 100, 120],
        ],
        "labels": ["STOP", "EXIT"],
    }
    catalogue = Catalogue(
        [VLMAsDetectorBlock, DetectionsFilterBlock, DetectionDrawingBlock]
    )

    # when
    session = compile_workflow(definition, catalogue=catalogue).create_session()
    try:
        row = session.run({"image": IMAGE, "answer": answer}).rows()[0]
    finally:
        session.close()

    # then
    boxes = row["boxes"]
    assert boxes.class_id.tolist() == [0, 0]
    assert boxes.image_metadata["class_names"] == {}
    assert row["exits"].xyxy.tolist() == [[100, 100, 140, 120]]
    assert row["exits"].bboxes_metadata == [{"class": "EXIT"}]
    assert row["drawing"].labels == ("STOP 1.00", "EXIT 1.00")


def test_detector_configuration_is_checked_at_compile_time() -> None:
    definition = _detect_then_classify_crops()
    del definition["steps"][0]["classes"]

    with pytest.raises(ParamsValidationError, match="classes is required"):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_selected_classes_are_checked_at_run_time() -> None:
    definition = _detect_then_classify_crops()
    definition["inputs"].append({"type": "WorkflowParameter", "name": "classes"})
    definition["steps"][0]["classes"] = "$inputs.classes"
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()

    try:
        with pytest.raises(StepExecutionError, match="repeat"):
            session.run(
                {
                    "image": IMAGE,
                    "boxes_answer": "[]",
                    "label_answer": "{}",
                    "classes": ["cat", "cat"],
                }
            )
    finally:
        session.close()

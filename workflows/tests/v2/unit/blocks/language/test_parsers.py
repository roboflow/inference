"""Retained Florence-2 decoding behavior and retired parser exports."""

import json
from typing import get_args

import numpy as np
import pytest
from roboflow_workflows.v2.blocks.language._shared import parsers as parsers
from roboflow_workflows.v2.blocks.language._shared.parsers import (
    MODEL_BOX_FORMATS,
    decode_florence_2_detections,
)
from roboflow_workflows.v2.blocks.language.vlm_as_detector.v2 import ModelType
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import row_record

IMAGE = ImageData.from_numpy_rgb(np.zeros((3000, 4000, 3), dtype=np.uint8))


def _decode(answer, *, task_type="object-detection", classes=None):
    error, predictions = decode_florence_2_detections(
        answer,
        image=IMAGE,
        task_type=task_type,
        classes=classes,
        inference_id="florence-helper-test",
    )
    return {"error_status": error, "predictions": predictions}


def test_every_detector_model_type_has_a_box_format() -> None:
    # A model_type without an entry would fail with KeyError at run time.
    assert set(get_args(ModelType)) == set(MODEL_BOX_FORMATS) | {"florence-2"}


def test_retired_parser_classes_are_not_public_exports() -> None:
    assert {"VLMAsClassifierBlock", "VLMAsDetectorBlock"}.isdisjoint(parsers.__all__)
    assert "decode_florence_2_detections" in parsers.__all__


@pytest.mark.parametrize(
    "task_type, answer, labels, class_ids, class_names",
    [
        # V1 get_4digit_from_md5: "car" -> 9605, "a red car" -> 6448.
        (
            "object-detection",
            {"bboxes": [[1, 2, 3, 4], [5, 6, 7, 8]], "labels": ["car", "car"]},
            ["car", "car"],
            [9605, 9605],
            {9605: "car"},
        ),
        (
            "phrase-grounded-object-detection",
            {"bboxes": [[1, 2, 3, 4]], "labels": ["a red car"]},
            ["a red car"],
            [6448],
            {6448: "a red car"},
        ),
        (
            "open-vocabulary-object-detection",
            {"bboxes": [[1, 2, 3, 4], [1, 2, 3, 4]], "bboxes_labels": ["cat", "fox"]},
            ["cat", "fox"],
            [1, -1],
            {0: "dog", 1: "cat"},
        ),
        (
            "object-detection-and-caption",
            {"bboxes": [[1, 2, 3, 4]], "labels": ["a car on a road"]},
            ["a car on a road"],
            [0],
            {},
        ),
        (
            "region-proposal",
            {"bboxes": [[1, 2, 3, 4]], "labels": [""]},
            ["roi"],
            [0],
            {},
        ),
    ],
)
def test_florence_2_tasks_label_and_number_boxes_as_v1(
    task_type, answer, labels, class_ids, class_names
) -> None:
    result = _decode(json.dumps(answer), task_type=task_type, classes=["dog", "cat"])

    detections = result["predictions"]
    OBJECT_DETECTION_PREDICTION_KIND.check(detections)
    assert [row["class"] for row in detections.bboxes_metadata] == labels
    assert detections.xyxy[0].tolist() == [1, 2, 3, 4]
    assert detections.confidence.tolist() == [1.0] * len(labels)
    assert detections.class_id.tolist() == class_ids
    assert detections.image_metadata["class_names"] == class_names


@pytest.mark.parametrize(
    "task_type, answer, boxes",
    [
        (
            "ocr-with-text-detection",
            {
                "quad_boxes": [
                    [10, 20, 50, 15, 55, 40, 12, 45],
                    [100, 100, 140, 100, 140, 120, 100, 120],
                ],
                "labels": ["STOP", "EXIT"],
            },
            [[10, 15, 55, 45], [100, 100, 140, 120]],
        ),
        (
            "object-detection-and-caption",
            {"bboxes": [[1, 2, 3, 4], [5, 6, 7, 8]], "labels": ["STOP", "EXIT"]},
            [[1, 2, 3, 4], [5, 6, 7, 8]],
        ),
    ],
)
def test_florence_2_text_rows_name_themselves_and_leave_class_names_empty(
    task_type, answer, boxes
) -> None:
    # given: two rows with different texts; both get class id 0 as in V1.
    # when
    result = _decode(answer, task_type=task_type)

    # then: each row keeps its own text; no id -> name entry stands for both.
    detections = result["predictions"]
    assert result["error_status"] is False
    assert detections.bboxes_metadata == [{"class": "STOP"}, {"class": "EXIT"}]
    assert detections.xyxy.tolist() == boxes
    assert detections.class_id.tolist() == [0, 0]
    assert detections.image_metadata["class_names"] == {}


def test_florence_2_ocr_keeps_the_box_enclosing_each_quad() -> None:
    answer = {"quad_boxes": [[10, 20, 50, 15, 55, 40, 12, 45]], "labels": ["STOP"]}

    result = _decode(answer, task_type="ocr-with-text-detection")

    assert result["predictions"].xyxy.tolist() == [[10, 15, 55, 45]]
    assert result["predictions"].bboxes_metadata == [{"class": "STOP"}]


@pytest.mark.parametrize(
    "answer",
    [
        {"bboxes": [[1, 2, 3, 4]], "labels": []},
        {"bboxes": [[1, 2, 3]], "labels": ["a"]},
        {"bboxes": [[1, 2, 3, float("nan")]], "labels": ["a"]},
        {"bboxes": ["1234"], "labels": ["a"]},
        {"labels": ["a"]},
        ["not", "a", "dict"],
        "no json",
    ],
)
def test_malformed_florence_2_answers_set_error_status(answer) -> None:
    result = _decode(answer, task_type="object-detection")

    assert (result["error_status"], result["predictions"]) == (True, None)


def test_empty_florence_2_answer_is_an_empty_prediction() -> None:
    result = _decode({"bboxes": [], "labels": []})

    assert result["error_status"] is False
    assert result["predictions"].xyxy.shape == (0, 4)


def test_florence_helper_keeps_the_supplied_row_namespace() -> None:
    result = _decode({"bboxes": [[1, 2, 3, 4]], "labels": ["car"]})
    assert row_record(result["predictions"]).namespace == "florence-helper-test"

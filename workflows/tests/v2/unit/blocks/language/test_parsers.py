"""VLM parser blocks: model-to-format mapping, Florence-2 tasks, parameters.

Box decoding itself is covered by ``test_decoding``; these tests check what
each block adds on a 4000 x 3000 (width x height) image, large enough for the
OpenAI and Claude upload resizes to apply.
"""

import json

import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.language.parsers import (
    MODEL_BOX_FORMATS,
    VLMAsClassifierBlock,
    VLMAsDetectorBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import row_record

IMAGE = ImageData.from_numpy_rgb(np.zeros((3000, 4000, 3), dtype=np.uint8))


def _detect(answer, model_type: str, **kwargs):
    result = VLMAsDetectorBlock().run(
        image=IMAGE, vlm_output=answer, model_type=model_type, **kwargs
    )

    return result


@pytest.mark.parametrize(
    "model_type, entry, expected",
    [
        # Upload 2048 x 1536 (longest edge 2048); x and y scale by 4000 / 2048.
        ("openai", {"box_2d": [512, 384, 1024, 768]}, [1000, 750, 2000, 1500]),
        # Claude reads 2212 x 1659; 1106 * 4000 / 2212 = 2000, 829.5 -> 1500.
        ("anthropic-claude", {"box_2d": [0, 0, 1106, 829.5]}, [0, 0, 2000, 1500]),
        ("google-gemini", {"box_2d": [250, 250, 500, 500]}, [1000, 750, 2000, 1500]),
        ("spacexai", {"box_2d": [25, 25, 50, 50]}, [1000, 750, 2000, 1500]),
        ("qwen", {"box_2d": [250, 250, 500, 500]}, [1000, 750, 2000, 1500]),
        ("zai", {"box_2d": [250, 250, 500, 500]}, [1000, 750, 2000, 1500]),
        ("zai-flash", {"bbox_2d": [250, 250, 500, 500]}, [1000, 750, 2000, 1500]),
        (
            "muse",
            {"x_min": 250, "y_min": 250, "x_max": 500, "y_max": 500},
            [1000, 750, 2000, 1500],
        ),
    ],
)
def test_each_model_type_reads_its_coordinate_contract(
    model_type, entry, expected
) -> None:
    result = _detect([{**entry, "label": "cat"}], model_type, classes=["dog", "cat"])

    assert result["error_status"] is False
    detections = result["predictions"]
    OBJECT_DETECTION_PREDICTION_KIND.check(detections)
    assert detections.xyxy.tolist() == [expected]
    assert detections.class_id.tolist() == [1]
    assert row_record(detections).namespace == result["inference_id"]


@pytest.mark.parametrize("model_type", ["openai", "anthropic-claude", "google-gemini"])
def test_v1_normalized_named_answers_are_still_read(model_type) -> None:
    answer = {
        "detections": [
            {
                "x_min": 0.25,
                "y_min": 0.25,
                "x_max": 0.5,
                "y_max": 0.5,
                "class_name": "cat",
            }
        ]
    }

    result = _detect(answer, model_type, classes=["cat"])

    assert result["predictions"].xyxy.tolist() == [[1000, 750, 2000, 1500]]


@pytest.mark.parametrize(
    "model_type, reads_confidence",
    [("google-gemini", True), ("qwen", False), ("muse", False), ("spacexai", True)],
)
def test_confidence_follows_v1_per_model(model_type, reads_confidence) -> None:
    entry = {"box_2d": [0, 0, 10, 10], "confidence": 0.3, "label": "cat"}
    entry.update({"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10})

    result = _detect([entry], model_type, classes=["cat"])

    expected = 0.3 if reads_confidence else 1.0
    assert result["predictions"].confidence.tolist() == pytest.approx([expected])


def test_every_non_florence_model_type_has_a_box_format() -> None:
    model_types = VLMAsDetectorBlock.Params.model_fields["model_type"].annotation
    assert set(model_types.__args__) - {"florence-2"} == set(MODEL_BOX_FORMATS)


def test_unreadable_answer_sets_error_status_and_no_prediction() -> None:
    result = _detect("I see a cat.", "qwen", classes=["cat"])

    assert result["error_status"] is True
    assert result["predictions"] is None
    assert isinstance(result["inference_id"], str) and result["inference_id"]


# --- Florence-2 ----------------------------------------------------------------------


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
    result = _detect(
        json.dumps(answer), "florence-2", task_type=task_type, classes=["dog", "cat"]
    )

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
    result = _detect(answer, "florence-2", task_type=task_type)

    # then: each row keeps its own text; no id -> name entry stands for both.
    detections = result["predictions"]
    assert result["error_status"] is False
    assert detections.bboxes_metadata == [{"class": "STOP"}, {"class": "EXIT"}]
    assert detections.xyxy.tolist() == boxes
    assert detections.class_id.tolist() == [0, 0]
    assert detections.image_metadata["class_names"] == {}


def test_florence_2_ocr_keeps_the_box_enclosing_each_quad() -> None:
    answer = {"quad_boxes": [[10, 20, 50, 15, 55, 40, 12, 45]], "labels": ["STOP"]}

    result = _detect(answer, "florence-2", task_type="ocr-with-text-detection")

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
    result = _detect(answer, "florence-2", task_type="object-detection")

    assert (result["error_status"], result["predictions"]) == (True, None)


def test_empty_florence_2_answer_is_an_empty_prediction() -> None:
    result = _detect({"bboxes": [], "labels": []}, "florence-2")

    assert result["error_status"] is False
    assert result["predictions"].xyxy.shape == (0, 4)


# --- parameters -------------------------------------------------------------------------


def _detector_params(**fields):
    params = VLMAsDetectorBlock.Params(
        image="$inputs.image",
        vlm_output="$steps.vlm.output",
        **fields,
    )

    return params


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"model_type": "qwen"}, "classes is required"),
        (
            {"model_type": "qwen", "classes": ["a"], "task_type": "region-proposal"},
            "only task_type",
        ),
        (
            {
                "model_type": "florence-2",
                "task_type": "open-vocabulary-object-detection",
            },
            "classes is required",
        ),
        ({"model_type": "llava", "classes": ["a"]}, "model_type"),
    ],
)
def test_invalid_detector_configurations_are_rejected(fields, message) -> None:
    with pytest.raises(ValidationError, match=message):
        _detector_params(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"model_type": "florence-2", "task_type": "region-proposal"},
        {"model_type": "qwen", "classes": "$inputs.classes"},
    ],
)
def test_valid_detector_configurations_are_accepted(fields) -> None:
    _detector_params(**fields)


def test_duplicate_classes_fail_the_step() -> None:
    with pytest.raises(ValueError, match="repeat"):
        _detect("[]", "qwen", classes=["cat", "cat"])


# --- classifier ----------------------------------------------------------------------------


def test_classifier_returns_prediction_status_and_inference_id() -> None:
    result = VLMAsClassifierBlock().run(
        image=IMAGE,
        vlm_output='```json\n{"class_name": "cat", "confidence": 0.9}\n```',
        classes=["dog", "cat"],
    )

    assert result["error_status"] is False
    CLASSIFICATION_PREDICTION_KIND.check(result["predictions"])
    assert result["predictions"].class_id.tolist() == [1]
    assert result["inference_id"]


def test_classifier_reports_unreadable_answers() -> None:
    result = VLMAsClassifierBlock().run(
        image=IMAGE, vlm_output='{"answer": "cat"}', classes=["cat"]
    )

    assert (result["error_status"], result["predictions"]) == (True, None)


def test_block_outputs_declare_v1_names_and_kinds() -> None:
    detector = VLMAsDetectorBlock.outputs
    classifier = VLMAsClassifierBlock.outputs

    assert detector["predictions"].kind_names == ("object_detection_prediction",)
    assert classifier["predictions"].kind_names == ("classification_prediction",)
    for outputs in (detector, classifier):
        assert set(outputs) == {"predictions", "error_status", "inference_id"}

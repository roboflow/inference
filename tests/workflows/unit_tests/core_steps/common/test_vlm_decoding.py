"""Unit tests for the shared VLM decoding package."""

import json

import numpy as np
import pytest
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.vlm_decoding import (
    CLASSIFICATION_TASKS,
    DETECTION_BOX_FORMATS,
    DETECTION_TASKS,
    actual_vlm_prediction_outputs,
    build_object_detection_prompt,
    create_classes_index,
    decode_classification,
    decode_object_detections,
    decode_vlm_output,
    describe_vlm_prediction_outputs,
    extract_detection_entries,
    extract_json,
    get_detection_class_name,
    get_detection_confidence,
    scale_confidence,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    INFERENCE_ID_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from tests.workflows.unit_tests.core_steps._vlm_prediction_readers import (
    classification_top_class,
    detection_boxes,
    is_detection_prediction,
)

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 400

# Every non-absolute format below encodes the same box, so the expected
# pixel result is shared: x 80..400, y 100..300 of an 800x400 image.
EXPECTED_XYXY = [80.0, 100.0, 400.0, 300.0]


def _build_image(
    width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT
) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_parses_plain_json() -> None:
    error_status, parsed = extract_json('{"a": 1}')

    assert error_status is False
    assert parsed == {"a": 1}


def test_extract_json_prefers_first_markdown_block() -> None:
    raw = 'blah\n```json\n[{"label": "a"}]\n```\ntail\n```json\n[]\n```'

    error_status, parsed = extract_json(raw)

    assert error_status is False
    assert parsed == [{"label": "a"}]


def test_extract_json_rejects_scalar_root() -> None:
    error_status, parsed = extract_json("42")

    assert error_status is True
    assert parsed == {}


def test_extract_json_recovers_prose_wrapped_array() -> None:
    raw = 'Here you go: [{"label": "cat", "box_2d": [1, 2, 3, 4]}] hope that helps'

    error_status, parsed = extract_json(raw)

    assert error_status is False
    assert parsed == [{"label": "cat", "box_2d": [1, 2, 3, 4]}]


def test_extract_json_recovers_loose_object_sequence() -> None:
    raw = (
        '{"label": "cat", "x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}, '
        '{"label": "dog", "x_min": 5, "y_min": 6, "x_max": 7, "y_max": 8}'
    )

    error_status, parsed = extract_json(raw)

    assert error_status is False
    assert [entry["label"] for entry in parsed] == ["cat", "dog"]


def test_extract_json_reports_error_for_garbage() -> None:
    error_status, parsed = extract_json("I cannot help with that.")

    assert error_status is True
    assert parsed == {}


# ---------------------------------------------------------------------------
# Entry handling
# ---------------------------------------------------------------------------


def test_extract_detection_entries_accepts_root_list() -> None:
    assert extract_detection_entries([{"a": 1}, "junk", {"b": 2}]) == [
        {"a": 1},
        {"b": 2},
    ]


def test_extract_detection_entries_accepts_detections_wrapper() -> None:
    assert extract_detection_entries({"detections": [{"a": 1}]}) == [{"a": 1}]


def test_extract_detection_entries_accepts_single_bare_entry() -> None:
    entry = {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}

    assert extract_detection_entries(entry) == [entry]


def test_extract_detection_entries_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError):
        extract_detection_entries({"answer": "no objects"})


def test_get_detection_class_name_reads_aliases_and_defaults() -> None:
    assert get_detection_class_name({"label": "cat"}) == "cat"
    assert get_detection_class_name({"class_name": "dog"}) == "dog"
    assert get_detection_class_name({"class": "bird"}) == "bird"
    assert get_detection_class_name({"description": "fish"}) == "fish"
    assert get_detection_class_name({"label": ""}) == "unknown"
    assert get_detection_class_name({}) == "unknown"


def test_get_detection_confidence_defaults_and_clamps() -> None:
    assert get_detection_confidence({}) == 1.0
    assert get_detection_confidence({"confidence": "high"}) == 1.0
    assert get_detection_confidence({"confidence": 0.25}) == 0.25
    assert get_detection_confidence({"confidence": 4}) == 1.0
    assert get_detection_confidence({"confidence": -1}) == 0.0


def test_small_utils() -> None:
    assert create_classes_index(["a", "b"]) == {"a": 0, "b": 1}
    assert scale_confidence(2.0) == 1.0
    assert scale_confidence(-2.0) == 0.0


# ---------------------------------------------------------------------------
# Per-format happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "box_format, entry",
    [
        pytest.param(
            "xyxy_0_1000",
            {"box_2d": [100, 250, 500, 750], "label": "cat"},
            id="xyxy_0_1000",
        ),
        pytest.param(
            "xyxy_0_1000",
            {"bbox_2d": [100, 250, 500, 750], "label": "cat"},
            id="xyxy_0_1000-bbox-alias",
        ),
        pytest.param(
            "yxyx_0_1000",
            {"box_2d": [250, 100, 750, 500], "label": "cat"},
            id="yxyx_0_1000",
        ),
        pytest.param(
            "xyxy_percent",
            {"box_2d": [10, 25, 50, 75], "label": "cat"},
            id="xyxy_percent",
        ),
        pytest.param(
            "named_0_1000",
            {"x_min": 100, "y_min": 250, "x_max": 500, "y_max": 750, "label": "cat"},
            id="named_0_1000",
        ),
        pytest.param(
            "named_normalized",
            {
                "x_min": 0.1,
                "y_min": 0.25,
                "x_max": 0.5,
                "y_max": 0.75,
                "class_name": "cat",
            },
            id="named_normalized",
        ),
    ],
)
def test_decode_object_detections_produces_exact_pixel_boxes(
    box_format: str, entry: dict
) -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output=f"[{json.dumps(entry)}]",
        box_format=box_format,
        image=image,
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert detections.xyxy.tolist() == [EXPECTED_XYXY]
    assert detections.class_id.tolist() == [0]
    assert detections.data[CLASS_NAME_DATA_FIELD].tolist() == ["cat"]
    assert detections.confidence.tolist() == [1.0]


def test_decode_object_detections_xyxy_absolute_rescales_from_upload_dimensions() -> (
    None
):
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output='[{"box_2d": [10, 20, 110, 120], "label": "cat"}]',
        box_format="xyxy_absolute",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
        upload_width=400,
        upload_height=200,
    )

    assert error_status is False
    assert detections.xyxy.tolist() == [[20.0, 40.0, 220.0, 240.0]]


def test_decode_object_detections_xyxy_absolute_requires_upload_dimensions() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output='[{"box_2d": [10, 20, 110, 120], "label": "cat"}]',
        box_format="xyxy_absolute",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is True
    assert detections is None


def test_decode_object_detections_clamps_out_of_range_coordinates() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output='[{"box_2d": [-50, -10, 1200, 1100], "label": "cat"}]',
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert detections.xyxy.tolist() == [
        [0.0, 0.0, float(IMAGE_WIDTH), float(IMAGE_HEIGHT)]
    ]


# ---------------------------------------------------------------------------
# Shared decoding behavior
# ---------------------------------------------------------------------------


def test_decode_object_detections_accepts_detections_wrapper() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output='{"detections": [{"box_2d": [100, 250, 500, 750], "label": "cat"}]}',
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert detections.xyxy.tolist() == [EXPECTED_XYXY]


def test_decode_object_detections_accepts_markdown_fenced_output() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output=(
            "Sure!\n```json\n"
            '[{"box_2d": [100, 250, 500, 750], "label": "cat"}]\n'
            "```\n"
        ),
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert detections.xyxy.tolist() == [EXPECTED_XYXY]


def test_decode_object_detections_recovers_loose_named_entries() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output=(
            '{"label": "cat", "x_min": 100, "y_min": 250, '
            '"x_max": 500, "y_max": 750}, '
            '{"label": "cat", "x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}'
        ),
        box_format="named_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert len(detections) == 2
    assert detections.xyxy[0].tolist() == EXPECTED_XYXY


def test_decode_object_detections_keeps_unknown_label_with_negative_class_id() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output='[{"box_2d": [100, 250, 500, 750], "label": "unicorn"}]',
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert detections.class_id.tolist() == [-1]
    assert detections.data[CLASS_NAME_DATA_FIELD].tolist() == ["unicorn"]


def test_decode_object_detections_reads_confidence_when_present() -> None:
    # Quoted numbers are accepted too; the per-vendor parsers this module
    # replaced used a bare float(). The Gemini contract honours confidence.
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output=(
            '[{"box_2d": [250, 100, 750, 500], "label": "cat", "confidence": "0.4"}]'
        ),
        box_format="yxyx_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert detections.confidence.tolist() == [0.4]


def test_decode_object_detections_skips_malformed_entries() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output=(
            "["
            '{"box_2d": [100, 250, 500, 750], "label": "cat"},'
            '{"label": "no-box"},'
            '{"box_2d": [1, 2, 3], "label": "short-box"},'
            '{"box_2d": [1, 2, 3, NaN], "label": "nan-box"},'
            '"not-a-dict"'
            "]"
        ),
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert len(detections) == 1
    assert detections.xyxy.tolist() == [EXPECTED_XYXY]


def test_decode_object_detections_reports_error_when_no_entry_matches_format() -> None:
    # The model answered in the OpenRouter normalized-named contract while the
    # block asked for 0-1000 `box_2d` boxes (a format without the legacy
    # named fallback). Every entry is unusable, and reporting that as zero
    # detections would be indistinguishable from "nothing detected".
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output=json.dumps(
            {
                "detections": [
                    {
                        "x_min": 0.1,
                        "y_min": 0.2,
                        "x_max": 0.3,
                        "y_max": 0.4,
                        "class_name": "cat",
                        "confidence": 0.7,
                    }
                ]
            }
        ),
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
        upload_width=IMAGE_WIDTH,
        upload_height=IMAGE_HEIGHT,
    )

    assert error_status is True
    assert detections is None


def test_decode_object_detections_accepts_numeric_strings() -> None:
    # Models routinely quote coordinates; the per-vendor parsers this module
    # replaced used a bare float() and accepted them.
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output=(
            '[{"box_2d": ["100", "250", "500", "750"], "label": "cat", '
            '"confidence": "0.5"}]'
        ),
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert detections.xyxy.tolist() == [EXPECTED_XYXY]
    # the Qwen contract ignores unsolicited confidences, like its legacy parser
    assert detections.confidence.tolist() == [1.0]


def test_decode_object_detections_skips_non_numeric_string_boxes() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output=(
            "["
            '{"box_2d": ["100", "250", "500", "750"], "label": "cat"},'
            '{"box_2d": ["100", "250", "500", "seven-fifty"], "label": "cat"},'
            '{"box_2d": ["", "", "", ""], "label": "cat"}'
            "]"
        ),
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert len(detections) == 1
    assert detections.xyxy.tolist() == [EXPECTED_XYXY]


def test_decode_object_detections_attaches_full_metadata() -> None:
    image = _build_image()

    _, detections = decode_object_detections(
        raw_output='[{"box_2d": [100, 250, 500, 750], "label": "cat"}]',
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert detections.data[IMAGE_DIMENSIONS_KEY].tolist() == [
        [IMAGE_HEIGHT, IMAGE_WIDTH]
    ]
    assert detections.data[INFERENCE_ID_KEY].tolist() == ["inference-id"]
    assert detections.data[PREDICTION_TYPE_KEY].tolist() == ["object-detection"]
    assert len(set(detections.data[DETECTION_ID_KEY].tolist())) == 1
    assert detections.data[PARENT_ID_KEY].tolist() == ["parent"]
    assert detections.data[ROOT_PARENT_ID_KEY].tolist() == ["parent"]


def test_decode_object_detections_returns_empty_detections_for_empty_list() -> None:
    image = _build_image()

    error_status, detections = decode_object_detections(
        raw_output="[]",
        box_format="xyxy_0_1000",
        image=image,
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert len(detections) == 0
    assert PARENT_ID_KEY in detections.data


def test_decode_object_detections_reports_error_for_list_without_entries() -> None:
    # given - a bare coordinate list lifted out of leaked reasoning text; the
    # legacy VLM as Detector block flags this as an error rather than "no objects"
    raw_output = (
        "Let me identify the most obvious ones: the first crack is at "
        "approximately [0.404, 0.515, 0.458, 0.612] and another near"
    )

    # when
    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format="named_normalized",
        image=_build_image(width=100, height=100),
        classes=["crack"],
        inference_id="iid",
    )

    # then
    assert error_status is True
    assert detections is None


def test_decode_object_detections_reports_error_for_unparsable_output() -> None:
    error_status, detections = decode_object_detections(
        raw_output="I could not find anything",
        box_format="xyxy_0_1000",
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is True
    assert detections is None


def test_decode_object_detections_reports_error_for_unknown_format() -> None:
    error_status, detections = decode_object_detections(
        raw_output="[]",
        box_format="not-a-format",
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is True
    assert detections is None


def test_decode_object_detections_reports_error_for_unexpected_shape() -> None:
    error_status, detections = decode_object_detections(
        raw_output='{"answer": "there are no cats"}',
        box_format="xyxy_0_1000",
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is True
    assert detections is None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_build_object_detection_prompt_fills_class_list() -> None:
    prompt = build_object_detection_prompt(
        box_format="xyxy_0_1000", classes=["cat", "dog"]
    )

    assert prompt.endswith("Only use these labels: cat, dog")
    assert "{" not in prompt


def test_build_object_detection_prompt_fills_upload_dimensions() -> None:
    prompt = build_object_detection_prompt(
        box_format="xyxy_absolute",
        classes=["cat"],
        upload_width=640,
        upload_height=480,
    )

    assert "640x480 pixel image" in prompt


def test_build_object_detection_prompt_requires_upload_dimensions() -> None:
    with pytest.raises(ValueError):
        build_object_detection_prompt(box_format="xyxy_absolute", classes=["cat"])


def test_build_object_detection_prompt_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        build_object_detection_prompt(box_format="nope", classes=["cat"])


def test_build_object_detection_prompt_rejects_format_without_template() -> None:
    # `named_normalized` is prompted by `common/openrouter.py`, not from here.
    with pytest.raises(ValueError):
        build_object_detection_prompt(box_format="named_normalized", classes=["cat"])


@pytest.mark.parametrize(
    "box_format",
    sorted(
        name
        for name, box_format in DETECTION_BOX_FORMATS.items()
        if box_format.prompt_template is not None
    ),
)
def test_every_format_renders_a_prompt_without_leftover_placeholders(
    box_format: str,
) -> None:
    prompt = build_object_detection_prompt(
        box_format=box_format,
        classes=["cat"],
        upload_width=640,
        upload_height=480,
    )

    assert "cat" in prompt
    assert "{class_list}" not in prompt
    assert "{width}" not in prompt


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_decode_classification_single_class() -> None:
    image = _build_image()

    error_status, predictions = decode_classification(
        raw_output='```json\n{"class_name": "cat", "confidence": 0.7}\n```',
        image=image,
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert predictions == {
        "image": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
        "predictions": [
            {"class": "cat", "class_id": 0, "confidence": 0.7},
            {"class": "dog", "class_id": 1, "confidence": 0.0},
        ],
        "top": "cat",
        "confidence": 0.7,
        "inference_id": "inference-id",
        "parent_id": "parent",
    }


def test_decode_classification_single_class_with_unknown_top_class() -> None:
    error_status, predictions = decode_classification(
        raw_output='{"class_name": "unicorn", "confidence": 1.5}',
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert predictions["predictions"] == [
        {"class": "unicorn", "class_id": -1, "confidence": 1.0},
        {"class": "cat", "class_id": 0, "confidence": 0.0},
    ]
    assert predictions["top"] == "unicorn"


def test_decode_classification_multi_label() -> None:
    error_status, predictions = decode_classification(
        raw_output=(
            '{"predicted_classes": ['
            '{"class": "cat", "confidence": 0.9},'
            '{"class": "cat", "confidence": 0.3},'
            '{"class": "unicorn", "confidence": 0.5}]}'
        ),
        image=_build_image(),
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert predictions == {
        "image": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
        "predictions": {
            "cat": {"confidence": 0.9, "class_id": 0},
            "dog": {"confidence": 0.0, "class_id": 1},
            "unicorn": {"confidence": 0.5, "class_id": -1},
        },
        "predicted_classes": ["cat", "unicorn"],
        "inference_id": "inference-id",
        "parent_id": "parent",
    }


def test_decode_classification_reports_error_for_unrecognised_payload() -> None:
    error_status, predictions = decode_classification(
        raw_output='{"answer": "a cat"}',
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is True
    assert predictions is None


def test_decode_classification_reports_error_for_unparsable_output() -> None:
    error_status, predictions = decode_classification(
        raw_output="not json at all",
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is True
    assert predictions is None


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def test_describe_vlm_prediction_outputs() -> None:
    outputs = describe_vlm_prediction_outputs()

    assert [output.name for output in outputs] == [
        "predictions",
        "error_status",
        "inference_id",
    ]
    assert outputs[0].kind == [
        OBJECT_DETECTION_PREDICTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
    ]
    assert outputs[1].kind == [BOOLEAN_KIND]
    assert outputs[2].kind == [INFERENCE_ID_KIND]


@pytest.mark.parametrize("task_type", sorted(DETECTION_TASKS))
def test_actual_outputs_narrow_to_detections(task_type: str) -> None:
    outputs = actual_vlm_prediction_outputs(task_type)

    assert outputs[0].kind == [OBJECT_DETECTION_PREDICTION_KIND]


@pytest.mark.parametrize("task_type", sorted(CLASSIFICATION_TASKS))
def test_actual_outputs_narrow_to_classification(task_type: str) -> None:
    outputs = actual_vlm_prediction_outputs(task_type)

    assert outputs[0].kind == [CLASSIFICATION_PREDICTION_KIND]


def test_actual_outputs_keep_union_for_other_tasks() -> None:
    outputs = actual_vlm_prediction_outputs("unconstrained")

    assert [output.name for output in outputs] == [
        "predictions",
        "error_status",
        "inference_id",
    ]
    assert outputs[0].kind == [
        OBJECT_DETECTION_PREDICTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
    ]


def test_actual_outputs_names_match_describe_outputs() -> None:
    declared = {output.name for output in describe_vlm_prediction_outputs()}

    for task_type in ["object-detection", "classification", "unconstrained"]:
        assert {
            output.name for output in actual_vlm_prediction_outputs(task_type)
        } == declared


def test_decode_vlm_output_dispatches_to_detections() -> None:
    error_status, predictions = decode_vlm_output(
        task_type="object-detection",
        raw_output='[{"box_2d": [100, 250, 500, 750], "label": "cat"}]',
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
        box_format="xyxy_0_1000",
    )

    assert error_status is False
    assert is_detection_prediction(predictions)
    assert detection_boxes(predictions) == [EXPECTED_XYXY]


@pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="numpy carrier; the tensor-native carriers are covered by "
    "test_vlm_decoding_tensor.py",
)
def test_decode_vlm_output_returns_numpy_carriers() -> None:
    _, detections = decode_vlm_output(
        task_type="object-detection",
        raw_output='[{"box_2d": [100, 250, 500, 750], "label": "cat"}]',
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
        box_format="xyxy_0_1000",
    )
    _, classification = decode_vlm_output(
        task_type="classification",
        raw_output='{"class_name": "cat", "confidence": 0.7}',
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert isinstance(detections, sv.Detections)
    assert isinstance(classification, dict)


def test_decode_vlm_output_dispatches_to_classification() -> None:
    error_status, predictions = decode_vlm_output(
        task_type="classification",
        raw_output='{"class_name": "cat", "confidence": 0.7}',
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is False
    assert classification_top_class(predictions) == "cat"


def test_decode_vlm_output_requires_box_format_for_detection() -> None:
    error_status, predictions = decode_vlm_output(
        task_type="object-detection",
        raw_output="[]",
        image=_build_image(),
        classes=["cat"],
        inference_id="inference-id",
    )

    assert error_status is True
    assert predictions is None


def test_decode_vlm_output_requires_classes_for_detection() -> None:
    error_status, predictions = decode_vlm_output(
        task_type="object-detection",
        raw_output="[]",
        image=_build_image(),
        classes=None,
        inference_id="inference-id",
        box_format="xyxy_0_1000",
    )

    assert error_status is True
    assert predictions is None


def test_decode_vlm_output_passes_through_non_decoding_task() -> None:
    error_status, predictions = decode_vlm_output(
        task_type="unconstrained",
        raw_output="a cat sitting on a mat",
        image=_build_image(),
        classes=None,
        inference_id="inference-id",
    )

    assert error_status is False
    assert predictions is None


def test_yxyx_format_accepts_legacy_named_normalized_entries() -> None:
    # given - the deprecated Gemini parser accepted x_min.. entries normalized
    # to 0-1 whenever box_2d was absent; the yxyx format keeps that tolerance
    raw_output = (
        '[{"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6, "label": "cat"}]'
    )

    # when
    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format="yxyx_0_1000",
        image=_build_image(width=100, height=200),
        classes=["cat"],
        inference_id="iid",
    )

    # then
    assert error_status is False
    assert detections.xyxy.tolist() == [[10.0, 40.0, 50.0, 120.0]]


def test_absolute_format_accepts_legacy_named_normalized_entries() -> None:
    # given - the deprecated OpenAI / Claude parsers fell back to the
    # normalized x_min.. contract of older block versions
    raw_output = (
        '{"detections": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6, '
        '"class_name": "cat"}]}'
    )

    # when
    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format="xyxy_absolute",
        image=_build_image(width=100, height=200),
        classes=["cat"],
        inference_id="iid",
        upload_width=50,
        upload_height=100,
    )

    # then
    assert error_status is False
    assert detections.xyxy.tolist() == [[10.0, 40.0, 50.0, 120.0]]


def test_decode_object_detections_treats_non_finite_confidence_as_missing() -> None:
    # given - json.loads accepts NaN, which would otherwise leak into the
    # confidence vector
    raw_output = '[{"box_2d": [100, 200, 500, 600], "label": "cat", "confidence": NaN}]'

    # when
    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format="xyxy_0_1000",
        image=_build_image(width=1000, height=1000),
        classes=["cat"],
        inference_id="iid",
    )

    # then
    assert error_status is False
    assert detections.confidence.tolist() == [1.0]


def test_decode_object_detections_normalises_swapped_corners() -> None:
    # given
    raw_output = '[{"box_2d": [900, 600, 100, 200], "label": "cat"}]'

    # when
    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format="xyxy_0_1000",
        image=_build_image(width=1000, height=1000),
        classes=["cat"],
        inference_id="iid",
    )

    # then
    assert error_status is False
    assert detections.xyxy.tolist() == [[100.0, 200.0, 900.0, 600.0]]


# ---------------------------------------------------------------------------
# Malformed container shapes recovered by the shared extractor (samples from
# roboflow/inference#2930, captured from GLM 5.3 Flash and Qwen3.6 35B)
# ---------------------------------------------------------------------------


def test_decode_object_detections_accepts_single_detection_object() -> None:
    # given - one detection emitted without the enclosing list
    error_status, detections = decode_object_detections(
        raw_output='{"bbox_2d": [147, 0, 432, 690], "label": "gun"}',
        box_format="xyxy_0_1000",
        image=_build_image(width=1000, height=1000),
        classes=["gun"],
        inference_id="iid",
    )

    assert error_status is False
    assert detections.data["class_name"].tolist() == ["gun"]
    assert detections.xyxy.tolist() == [[147.0, 0.0, 432.0, 690.0]]


def test_decode_object_detections_accepts_list_missing_opening_bracket() -> None:
    # given - `{...}, {...}]` with the opening bracket dropped
    raw_output = (
        '{"bbox_2d": [419, 587, 459, 856], "label": "blue player"}, '
        '{"bbox_2d": [607, 104, 681, 326], "label": "basket"}]'
    )

    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format="xyxy_0_1000",
        image=_build_image(width=1000, height=1000),
        classes=["blue player", "basket"],
        inference_id="iid",
    )

    assert error_status is False
    assert detections.class_id.tolist() == [0, 1]


def test_decode_object_detections_accepts_repeated_empty_arrays() -> None:
    # given - "[]\n[]" for an image with no matches
    error_status, detections = decode_object_detections(
        raw_output="[]\n[]",
        box_format="xyxy_0_1000",
        image=_build_image(width=1000, height=1000),
        classes=["car"],
        inference_id="iid",
    )

    assert error_status is False
    assert len(detections) == 0


def test_decode_object_detections_accepts_json_lines() -> None:
    # given - one object per line, no array
    raw_output = (
        '{"box_2d": [10, 20, 30, 40], "label": "cat"}\n'
        '{"box_2d": [50, 60, 70, 80], "label": "dog"}'
    )

    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format="xyxy_0_1000",
        image=_build_image(width=1000, height=1000),
        classes=["cat", "dog"],
        inference_id="iid",
    )

    assert error_status is False
    assert detections.class_id.tolist() == [0, 1]


def test_decode_classification_accepts_bare_predicted_classes_array() -> None:
    # given - the multi-label `predicted_classes` array without its wrapper
    raw_output = '[{"class": "Vehicle", "confidence": 0.95}, {"class": "fire", "confidence": 0.8}]'

    error_status, prediction = decode_classification(
        raw_output=raw_output,
        image=_build_image(width=100, height=100),
        classes=["Vehicle", "fire", "smoke"],
        inference_id="iid",
    )

    assert error_status is False
    assert prediction["predicted_classes"] == ["Vehicle", "fire"]
    assert prediction["predictions"]["Vehicle"]["confidence"] == pytest.approx(0.95)
    assert prediction["predictions"]["smoke"]["confidence"] == 0.0


def test_decode_object_detections_salvages_complete_entries_of_truncated_answer() -> (
    None
):
    # given - a Muse answer cut at max_tokens mid-entry; the deprecated
    # formatter keeps the complete entries, so the shared decoder must too
    raw_output = (
        '[{"label":"pill","x_min":249,"y_min":198,"x_max":312,"y_max":252},'
        '{"label":"pill","x_min":443,"y_min":181,"x_max":504,"y_max":246},'
        '{"label":"pill","x_min":611,"y_min":'
    )

    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format="named_0_1000",
        image=_build_image(width=1000, height=1000),
        classes=["pill"],
        inference_id="iid",
    )

    assert error_status is False
    assert detections.xyxy.tolist() == [
        [249.0, 198.0, 312.0, 252.0],
        [443.0, 181.0, 504.0, 246.0],
    ]


def test_extract_json_reports_error_for_non_string_output() -> None:
    assert extract_json(None) == (True, {})


@pytest.mark.parametrize(
    "box_format, raw_output, expected_confidence",
    [
        # Qwen / Z.ai and Muse contracts: the deprecated parsers forced 1.0
        (
            "xyxy_0_1000",
            '[{"box_2d": [100, 200, 500, 600], "label": "cat", "confidence": 0.05}]',
            1.0,
        ),
        (
            "named_0_1000",
            '[{"x_min": 100, "y_min": 200, "x_max": 500, "y_max": 600, "label": "cat", "confidence": 0.05}]',
            1.0,
        ),
        # OpenAI-style contracts honoured a provided confidence
        (
            "named_normalized",
            '[{"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6, "class_name": "cat", "confidence": 0.05}]',
            0.05,
        ),
    ],
)
def test_unsolicited_confidence_is_honoured_per_format_like_legacy_parsers(
    box_format: str, raw_output: str, expected_confidence: float
) -> None:
    error_status, detections = decode_object_detections(
        raw_output=raw_output,
        box_format=box_format,
        image=_build_image(width=1000, height=1000),
        classes=["cat"],
        inference_id="iid",
    )

    assert error_status is False
    assert detections.confidence.tolist() == pytest.approx([expected_confidence])


def test_decode_classification_reports_error_for_non_finite_confidence() -> None:
    error_status, prediction = decode_classification(
        raw_output='{"class_name": "cat", "confidence": NaN}',
        image=_build_image(width=100, height=100),
        classes=["cat", "dog"],
        inference_id="iid",
    )

    assert error_status is True
    assert prediction is None


def test_decode_classification_does_not_salvage_truncated_detection_lists() -> None:
    # given - the detection-only truncation salvage must not turn a box list
    # into a multi-label classification
    raw_output = (
        '[{"label":"cat","x_min":1,"y_min":2,"x_max":3,"y_max":4},'
        '{"label":"dog","x_min":1,"y_min":2,"x_max":3,"y_max":4},{"label":"'
    )

    error_status, prediction = decode_classification(
        raw_output=raw_output,
        image=_build_image(width=100, height=100),
        classes=["cat", "dog"],
        inference_id="iid",
    )

    assert error_status is True
    assert prediction is None

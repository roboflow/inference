"""Tests for the VLM oriented-object-detection prompt, schemas and decoder.

The decoder turns the answer of ``cx, cy, length, width, angle_deg`` entries into
the native OBB carrier of ``common/oriented.py``, in both data representations.
"""

import importlib.util
import json
import math

import numpy as np
import pytest
from roboflow_workflows.core_steps.common.oriented import serialise_native_obb
from roboflow_workflows.core_steps.common.vlm_decoding import oriented_detection
from roboflow_workflows.core_steps.common.vlm_decoding.oriented_detection import (
    build_oriented_detection_prompt,
    build_oriented_json_schema,
    build_oriented_response_schema,
    decode_oriented_detections,
    resolve_class_heading,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

LABELS = ["forklift", "scissor-lift"]
CLASS_HEADING = {"forklift": True, "scissor-lift": False}
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200


def _build_image(
    width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT
) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _entry(
    label: str = "forklift",
    cx: float = 100.0,
    cy: float = 50.0,
    length: float = 80.0,
    width: float = 30.0,
    angle_deg: float = 0.0,
) -> dict:
    return {
        "label": label,
        "cx": cx,
        "cy": cy,
        "length": length,
        "width": width,
        "angle_deg": angle_deg,
    }


def _answer(*entries: dict) -> str:
    return json.dumps({"oriented_detections": list(entries)})


def _decode(
    raw_output: str,
    image: WorkflowImageData = None,
    labels: list = LABELS,
    class_heading: dict = CLASS_HEADING,
    upload_width: int = IMAGE_WIDTH,
    upload_height: int = IMAGE_HEIGHT,
):
    return decode_oriented_detections(
        raw_output,
        image=image or _build_image(),
        labels=labels,
        class_heading=class_heading,
        upload_width=upload_width,
        upload_height=upload_height,
    )


def _rows(raw_output: str, **kwargs) -> list:
    error_status, detections = _decode(raw_output, **kwargs)
    assert error_status is False
    return serialise_native_obb(detections)["predictions"]


def test_resolve_class_heading_completes_labels_and_ignores_unknown_names() -> None:
    # when
    result = resolve_class_heading(
        {"forklift": True, "unrelated-class": True}, labels=LABELS
    )

    # then
    assert result == {"forklift": True, "scissor-lift": False}


def test_resolve_class_heading_without_mapping_has_no_front() -> None:
    # when
    result = resolve_class_heading(None, labels=LABELS)

    # then
    assert result == {"forklift": False, "scissor-lift": False}


@pytest.mark.parametrize(
    "class_heading",
    [{"forklift": "yes"}, {"forklift": 1}, {1: True}, ["forklift"], "forklift"],
)
def test_resolve_class_heading_rejects_anything_but_class_to_boolean(
    class_heading,
) -> None:
    with pytest.raises(ValueError, match="true or false"):
        resolve_class_heading(class_heading, labels=LABELS)


def test_prompt_names_the_frame_the_labels_and_both_front_rules() -> None:
    # when
    prompt = build_oriented_detection_prompt(
        LABELS,
        class_heading=CLASS_HEADING,
        upload_width=1280,
        upload_height=800,
    )

    # then
    assert "the image is 1280x800 pixels" in prompt
    assert '"oriented_detections"' in prompt
    assert "Classes with a front: forklift." in prompt
    assert "Classes without a front: scissor-lift." in prompt
    assert prompt.endswith("Only use these labels: forklift, scissor-lift")


@pytest.mark.parametrize(
    "class_heading, present, absent",
    [
        ({"forklift": True, "scissor-lift": True}, "with a front", "without a front"),
        ({}, "without a front", "Classes with a front"),
    ],
)
def test_prompt_states_only_the_rule_that_applies(
    class_heading: dict, present: str, absent: str
) -> None:
    # when
    prompt = build_oriented_detection_prompt(
        LABELS,
        class_heading=class_heading,
        upload_width=100,
        upload_height=100,
    )

    # then
    assert present in prompt
    assert absent not in prompt


def test_prompt_requires_classes_and_the_upload_dimensions() -> None:
    with pytest.raises(ValueError, match="at least one class"):
        build_oriented_detection_prompt(
            [], class_heading={}, upload_width=10, upload_height=10
        )
    with pytest.raises(ValueError, match="dimensions of the uploaded image"):
        build_oriented_detection_prompt(
            LABELS, class_heading={}, upload_width=0, upload_height=10
        )


def test_json_schema_is_strict_and_constrains_the_label() -> None:
    # when
    schema = build_oriented_json_schema(["cat", "dog", "cat"])

    # then
    assert schema["required"] == ["oriented_detections"]
    assert schema["additionalProperties"] is False
    item = schema["properties"]["oriented_detections"]["items"]
    assert item["additionalProperties"] is False
    assert item["required"] == [
        "label",
        "cx",
        "cy",
        "length",
        "width",
        "angle_deg",
    ]
    assert item["properties"]["label"] == {"type": "string", "enum": ["cat", "dog"]}
    assert item["properties"]["angle_deg"] == {"type": "number"}


def test_response_schema_uses_gemini_types_and_keeps_the_field_order() -> None:
    # when
    schema = build_oriented_response_schema(LABELS)

    # then
    assert schema["type"] == "OBJECT"
    item = schema["properties"]["oriented_detections"]["items"]
    assert item["properties"]["label"] == {"type": "STRING", "enum": LABELS}
    assert item["properties"]["cx"] == {"type": "NUMBER"}
    assert item["propertyOrdering"] == item["required"]


def test_decode_maps_a_class_with_a_front_onto_the_native_tuple() -> None:
    # given
    raw_output = _answer(_entry(angle_deg=30.0))

    # when
    error_status, detections = _decode(raw_output)

    # then
    assert error_status is False
    result = serialise_native_obb(detections)
    assert result["task_type"] == "oriented-object-detection"
    assert result["image"] == {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT}
    assert result["class_heading"] == CLASS_HEADING
    (row,) = result["predictions"]
    assert row["obb"] == pytest.approx([100.0, 50.0, 80.0, 30.0, math.radians(30)])
    assert row["class"] == "forklift"
    assert row["class_id"] == 0
    assert row["heading"] is True
    assert row["confidence"] == 1.0
    assert row["parent_id"] == "parent"
    assert isinstance(row["detection_id"], str) and row["detection_id"]


def test_decode_gives_every_detection_its_own_identity() -> None:
    # when
    rows = _rows(_answer(_entry(), _entry(cx=500.0), _entry(label="scissor-lift")))

    # then
    assert [row["class_id"] for row in rows] == [0, 0, 1]
    assert len({row["detection_id"] for row in rows}) == 3


@pytest.mark.parametrize(
    "angle_deg, expected_deg",
    [(0, 0), (-90, -90), (170, 170), (180, -180), (190, -170), (-190, 170), (360, 0)],
)
def test_decode_keeps_the_full_circle_for_a_class_with_a_front(
    angle_deg: float, expected_deg: float
) -> None:
    # when
    (row,) = _rows(_answer(_entry(angle_deg=angle_deg)))

    # then: the native tuple is read back from the corners, so +-180 degrees may
    # come back as either sign; compare as points on the circle
    assert math.cos(row["obb"][4]) == pytest.approx(
        math.cos(math.radians(expected_deg))
    )
    assert math.sin(row["obb"][4]) == pytest.approx(
        math.sin(math.radians(expected_deg)), abs=1e-9
    )
    assert row["obb"][2:4] == pytest.approx([80.0, 30.0])


def test_decode_keeps_the_sides_of_a_front_class_whose_front_axis_is_the_short_one() -> (
    None
):
    # when
    (row,) = _rows(_answer(_entry(length=20.0, width=60.0, angle_deg=45.0)))

    # then
    assert row["obb"][2:4] == pytest.approx([20.0, 60.0])
    assert row["obb"][4] == pytest.approx(math.radians(45))


@pytest.mark.parametrize(
    "angle_deg, expected_deg",
    [(0, 0), (89, 89), (90, -90), (100, -80), (-100, 80), (180, 0), (-45, -45)],
)
def test_decode_folds_an_undirected_class_into_half_a_turn(
    angle_deg: float, expected_deg: float
) -> None:
    # when
    (row,) = _rows(_answer(_entry(label="scissor-lift", angle_deg=angle_deg)))

    # then
    assert row["heading"] is False
    assert row["obb"][4] == pytest.approx(math.radians(expected_deg))


def test_decode_writes_an_undirected_class_with_its_longer_side_first() -> None:
    # given: the model reported the short side as the main axis, pointing right
    raw_output = _answer(
        _entry(label="scissor-lift", length=30.0, width=80.0, angle_deg=10.0)
    )

    # when
    (row,) = _rows(raw_output)

    # then: same footprint, long side first, a quarter turn later
    assert row["obb"][2:4] == pytest.approx([80.0, 30.0])
    assert row["obb"][4] == pytest.approx(math.radians(-80))


def test_decode_scales_the_geometry_from_the_uploaded_frame_to_the_original() -> None:
    # given: the upload was a 2x downscale of the original image
    raw_output = _answer(_entry(cx=100.0, cy=50.0, length=80.0, width=30.0))

    # when
    (row,) = _rows(
        raw_output,
        image=_build_image(width=1920, height=1200),
        upload_width=960,
        upload_height=600,
    )

    # then
    assert row["obb"] == pytest.approx([200.0, 100.0, 160.0, 60.0, 0.0])


def test_decode_accepts_a_bare_list_and_numeric_strings() -> None:
    # given
    raw_output = json.dumps(
        [{**_entry(), "cx": "100", "length": "80.5", "angle_deg": "15"}]
    )

    # when
    (row,) = _rows(raw_output)

    # then
    assert row["obb"] == pytest.approx([100.0, 50.0, 80.5, 30.0, math.radians(15)])


def test_decode_drops_malformed_entries_and_keeps_the_rest() -> None:
    # given
    raw_output = _answer(
        _entry(),
        _entry(label="pallet"),
        {k: v for k, v in _entry().items() if k != "angle_deg"},
        _entry(length=0.0),
        _entry(width=-4.0),
        _entry(cx="left"),
        _entry(cy=True),
    )

    # when
    rows = _rows(raw_output)

    # then
    assert len(rows) == 1
    assert rows[0]["class"] == "forklift"


def test_decode_drops_non_finite_numbers() -> None:
    # given: json.loads accepts the bare NaN and Infinity tokens
    raw_output = (
        '{"oriented_detections": ['
        '{"label": "forklift", "cx": NaN, "cy": 1, "length": 8, "width": 3, "angle_deg": 0},'
        '{"label": "forklift", "cx": 1, "cy": 1, "length": Infinity, "width": 3, "angle_deg": 0},'
        '{"label": "forklift", "cx": 1, "cy": 1, "length": 8, "width": 3, "angle_deg": 0}'
        "]}"
    )

    # when
    rows = _rows(raw_output)

    # then
    assert len(rows) == 1


def test_decode_reports_an_error_when_no_entry_is_well_formed() -> None:
    # given
    raw_output = _answer(_entry(label="pallet"), _entry(length=0.0))

    # when
    error_status, detections = _decode(raw_output)

    # then
    assert error_status is True
    assert detections is None


@pytest.mark.parametrize(
    "raw_output",
    ["not json at all", "[1, 2, 3]", '{"detections": "none"}', "42"],
)
def test_decode_reports_an_error_for_an_unusable_answer(raw_output: str) -> None:
    # when
    error_status, detections = _decode(raw_output)

    # then
    assert error_status is True
    assert detections is None


def test_decode_reports_an_error_without_the_upload_dimensions() -> None:
    # when
    error_status, detections = _decode(_answer(_entry()), upload_width=0)

    # then
    assert error_status is True
    assert detections is None


def test_decode_of_nothing_detected_is_an_empty_native_prediction() -> None:
    # when
    error_status, detections = _decode(_answer())

    # then
    assert error_status is False
    result = serialise_native_obb(detections)
    assert result["predictions"] == []
    assert result["class_heading"] == CLASS_HEADING
    assert result["image"] == {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT}


def test_decode_freezes_the_front_of_every_label_not_only_the_detected_ones() -> None:
    # when
    error_status, detections = _decode(_answer(_entry()), class_heading={})

    # then
    assert error_status is False
    result = serialise_native_obb(detections)
    assert result["class_heading"] == {"forklift": False, "scissor-lift": False}
    assert result["predictions"][0]["heading"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs the tensor carrier"
)
def test_decode_returns_the_tensor_carrier_when_the_tensor_representation_is_on(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(oriented_detection, "ENABLE_TENSOR_DATA_REPRESENTATION", True)

    # when
    error_status, detections = _decode(
        _answer(_entry(angle_deg=30.0), _entry(label="scissor-lift"))
    )

    # then
    assert error_status is False
    assert hasattr(detections, "image_metadata")
    assert detections.xyxy.shape[0] == 2
    result = serialise_native_obb(detections)
    assert result["class_heading"] == CLASS_HEADING
    assert [row["heading"] for row in result["predictions"]] == [True, False]
    assert result["predictions"][0]["obb"] == pytest.approx(
        [100.0, 50.0, 80.0, 30.0, math.radians(30)]
    )

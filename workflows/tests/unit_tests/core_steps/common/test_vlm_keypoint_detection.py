"""Tests for VLM keypoint detection: skeleton parsing, request, schema, decoding.

The decoded values are read through the keypoint kind serializer, so the suite
asserts the same output with the numpy representation and with
``ENABLE_TENSOR_DATA_REPRESENTATION`` (the tensor-only checks skip on the flag).
"""

import json
from typing import Any, List

import numpy as np
import pytest
from roboflow_workflows.core_steps.common.vlm_decoding.keypoint_detection import (
    OCCLUDED_KEYPOINT_CONFIDENCE,
    VISIBLE_KEYPOINT_CONFIDENCE,
    KeypointSkeleton,
    build_keypoint_detection_prompt,
    build_keypoint_response_schema,
    decode_keypoint_detections,
    resolve_keypoint_skeletons,
)
from roboflow_workflows.core_steps.loader import KINDS_SERIALIZERS
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    KEYPOINT_DETECTION_PREDICTION_KIND,
)

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 400
PLATE_SKELETON = {
    "keypoints": ["TL", "TR", "BR", "BL"],
    "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
    "template": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
}
CAR_SKELETON = {"keypoints": ["wheel", "wheel"]}
SCHEMA = {"plate": PLATE_SKELETON, "car": CAR_SKELETON}

# [y_min, x_min, y_max, x_max] normalized to 0-1000 - on an 800x400 image this
# is exactly [80, 100, 400, 300] pixels.
PLATE_BOX = [250, 100, 750, 500]
PLATE_XYXY = [80.0, 100.0, 400.0, 300.0]
PLATE_INSTANCE = {
    "box_2d": PLATE_BOX,
    "keypoints": {
        "TL": {"point": [250, 100], "occluded": False},
        "TR": {"point": [250, 500], "occluded": True},
        "BR": None,
        "BL": {"point": [750, 100], "occluded": False},
    },
}
PLATE_KEYPOINTS = [
    {
        "class_id": 0,
        "class": "TL",
        "confidence": VISIBLE_KEYPOINT_CONFIDENCE,
        "x": 80.0,
        "y": 100.0,
    },
    {
        "class_id": 1,
        "class": "TR",
        "confidence": OCCLUDED_KEYPOINT_CONFIDENCE,
        "x": 400.0,
        "y": 100.0,
    },
    {
        "class_id": 3,
        "class": "BL",
        "confidence": VISIBLE_KEYPOINT_CONFIDENCE,
        "x": 80.0,
        "y": 300.0,
    },
]


def _build_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
    )


def _decode(answer: Any, classes: List[str] = None, schema: dict = None):
    raw_output = answer if isinstance(answer, str) else json.dumps(answer)
    skeletons = resolve_keypoint_skeletons(schema or SCHEMA, classes=classes)

    return decode_keypoint_detections(
        raw_output,
        image=_build_image(),
        skeletons=skeletons,
        classes=classes,
        inference_id="inference",
    )


def _serialize(predictions: Any) -> List[dict]:
    serializer = KINDS_SERIALIZERS[KEYPOINT_DETECTION_PREDICTION_KIND.name]

    return serializer(predictions)["predictions"]


def _box(prediction: dict) -> List[float]:
    half_width, half_height = prediction["width"] / 2, prediction["height"] / 2
    return [
        prediction["x"] - half_width,
        prediction["y"] - half_height,
        prediction["x"] + half_width,
        prediction["y"] + half_height,
    ]


def test_resolve_keeps_the_requested_classes_in_order() -> None:
    # when
    skeletons = resolve_keypoint_skeletons(SCHEMA, classes=["car", "plate", "car"])

    # then
    assert list(skeletons) == ["car", "plate"]
    assert skeletons["plate"].keypoints == ["TL", "TR", "BR", "BL"]
    assert skeletons["plate"].edges == [(0, 1), (1, 2), (2, 3), (3, 0)]


@pytest.mark.parametrize("classes", [None, []])
def test_resolve_without_classes_takes_every_skeleton(classes) -> None:
    # when
    skeletons = resolve_keypoint_skeletons(SCHEMA, classes=classes)

    # then
    assert list(skeletons) == ["plate", "car"]


def test_resolve_accepts_parsed_skeletons() -> None:
    # given
    skeleton = KeypointSkeleton.model_validate(PLATE_SKELETON)

    # when
    skeletons = resolve_keypoint_skeletons({"plate": skeleton}, classes=["plate"])

    # then
    assert skeletons["plate"] is skeleton


@pytest.mark.parametrize("schema", [None, {}, [], "plate"])
def test_resolve_rejects_a_missing_schema(schema: Any) -> None:
    with pytest.raises(ValueError, match="must map each class name to its skeleton"):
        resolve_keypoint_skeletons(schema, classes=["plate"])


def test_resolve_rejects_a_class_without_skeleton() -> None:
    with pytest.raises(ValueError, match=r"no skeleton for classes: \['dog'\]"):
        resolve_keypoint_skeletons(SCHEMA, classes=["plate", "dog"])


@pytest.mark.parametrize(
    "skeleton, problem",
    [
        ({"keypoints": []}, "keypoints"),
        ({"keypoints": [""]}, "keypoints.0"),
        ({"keypoints": ["a", "b"], "edges": [[0, 2]]}, "points outside the 2"),
        ({"keypoints": ["a", "b"], "template": [[0.5, 0.5]]}, "1 positions for 2"),
        ({"keypoints": ["a"], "template": [[float("nan"), 0.5]]}, "template.0"),
        ({"edges": [[0, 1]]}, "keypoints: Field required"),
    ],
)
def test_resolve_rejects_an_invalid_skeleton(skeleton: dict, problem: str) -> None:
    with pytest.raises(ValueError, match="invalid skeleton for 'plate'") as error:
        resolve_keypoint_skeletons({"plate": skeleton}, classes=["plate"])

    assert problem in str(error.value)


def test_answer_keys_number_repeated_names_without_collisions() -> None:
    # given
    skeleton = KeypointSkeleton(keypoints=["wheel (2)", "wheel", "wheel", "axle"])

    # then
    assert skeleton.answer_keys == ["wheel (2)", "wheel", "wheel (3)", "axle"]


def test_prompt_describes_every_skeleton() -> None:
    # given
    skeletons = resolve_keypoint_skeletons(SCHEMA, classes=["plate", "car"])

    # when
    prompt = build_keypoint_detection_prompt(skeletons)

    # then
    assert prompt.startswith(
        "Find every instance of these classes in this image and locate its "
        "keypoints: plate, car."
    )
    assert '"box_2d"' in prompt and '{"point": [y, x], "occluded": bool}' in prompt
    assert "Skeleton of a plate:\nKeypoints: TL, TR, BR, BL\n" in prompt
    assert "Edges (connected keypoints): TL-TR, TR-BR, BR-BL, BL-TL" in prompt
    assert (
        "(x, y) fractions of the box width and height from its top-left corner: "
        "TL (0.00, 0.00), TR (1.00, 0.00), BR (1.00, 1.00), BL (0.00, 1.00)."
    ) in prompt
    assert prompt.endswith("Skeleton of a car:\nKeypoints: wheel, wheel (2)")


def test_prompt_lists_only_the_known_template_positions() -> None:
    # given
    skeletons = resolve_keypoint_skeletons(
        {"face": {"keypoints": ["eye", "nose"], "template": [None, [0.5, 0.75]]}},
        classes=["face"],
    )

    # when
    prompt = build_keypoint_detection_prompt(skeletons)

    # then
    assert prompt.endswith("from its top-left corner: nose (0.50, 0.75).")
    assert "Edges" not in prompt


def test_response_schema_requires_every_class_and_keypoint() -> None:
    # given
    skeletons = resolve_keypoint_skeletons(SCHEMA, classes=["plate", "car"])

    # when
    schema = build_keypoint_response_schema(skeletons)

    # then
    assert schema["required"] == schema["propertyOrdering"] == ["plate", "car"]
    instance = schema["properties"]["car"]["items"]
    assert instance["required"] == ["box_2d", "keypoints"]
    assert instance["properties"]["box_2d"]["minItems"] == 4
    keypoints = instance["properties"]["keypoints"]
    assert keypoints["required"] == ["wheel", "wheel (2)"]
    assert keypoints["properties"]["wheel"]["nullable"] is True
    assert keypoints["properties"]["wheel"]["required"] == ["point", "occluded"]
    assert keypoints["properties"]["wheel"] is not keypoints["properties"]["wheel (2)"]


def test_decode_places_keypoints_on_original_image_pixels() -> None:
    # when
    error, predictions = _decode({"plate": [PLATE_INSTANCE]}, classes=["plate"])

    # then
    assert error is False
    serialized = _serialize(predictions)
    assert len(serialized) == 1
    assert _box(serialized[0]) == PLATE_XYXY
    assert serialized[0]["class"] == "plate"
    assert serialized[0]["class_id"] == 0
    assert serialized[0]["confidence"] == 1.0
    assert serialized[0]["keypoints"] == PLATE_KEYPOINTS


def test_decode_reads_every_class_and_numbers_them_by_classes() -> None:
    # given
    car = {
        "box_2d": [0, 0, 500, 1000],
        "keypoints": {
            "wheel": {"point": [500, 250], "occluded": False},
            "wheel (2)": {"point": [500, 750], "occluded": False},
        },
    }

    # when
    error, predictions = _decode(
        {"plate": [PLATE_INSTANCE], "car": [car, car], "dog": [PLATE_INSTANCE]},
        classes=["car", "plate"],
    )

    # then
    assert error is False
    serialized = _serialize(predictions)
    assert [(entry["class"], entry["class_id"]) for entry in serialized] == [
        ("car", 0),
        ("car", 0),
        ("plate", 1),
    ]
    assert [
        (keypoint["class_id"], keypoint["class"], keypoint["x"], keypoint["y"])
        for keypoint in serialized[0]["keypoints"]
    ] == [(0, "wheel", 200.0, 200.0), (1, "wheel", 600.0, 200.0)]
    assert serialized[2]["keypoints"] == PLATE_KEYPOINTS


@pytest.mark.parametrize(
    "value",
    [
        {"point": [1001, 100], "occluded": False},
        {"point": [-1, 100], "occluded": False},
        {"point": [True, 100], "occluded": False},
        {"point": ["250", 100], "occluded": False},
        {"point": [250], "occluded": False},
        {"point": None, "occluded": True},
        [250, 100],
        None,
    ],
)
def test_decode_leaves_out_an_unusable_keypoint(value: Any) -> None:
    # given
    answer = {"plate": [{**PLATE_INSTANCE, "keypoints": {"TL": value}}]}

    # when
    error, predictions = _decode(answer, classes=["plate"])

    # then
    assert error is False
    assert _serialize(predictions)[0]["keypoints"] == []


def test_decode_takes_a_keypoint_without_occlusion_flag_as_visible() -> None:
    # given
    answer = {"plate": [{**PLATE_INSTANCE, "keypoints": {"BR": {"point": [750, 500]}}}]}

    # when
    error, predictions = _decode(answer, classes=["plate"])

    # then
    assert error is False
    assert _serialize(predictions)[0]["keypoints"] == [
        {
            "class_id": 2,
            "class": "BR",
            "confidence": VISIBLE_KEYPOINT_CONFIDENCE,
            "x": 400.0,
            "y": 300.0,
        }
    ]


def test_decode_skips_an_instance_without_box() -> None:
    # given
    answer = {
        "plate": [
            {"keypoints": PLATE_INSTANCE["keypoints"]},
            {"box_2d": PLATE_BOX},
            "plate",
            PLATE_INSTANCE,
        ]
    }

    # when
    error, predictions = _decode(answer, classes=["plate"])

    # then
    assert error is False
    serialized = _serialize(predictions)
    assert len(serialized) == 1
    assert serialized[0]["keypoints"] == PLATE_KEYPOINTS


@pytest.mark.parametrize("answer", [{"plate": []}, {}, {"plate": None}])
def test_decode_an_answer_without_instances(answer: dict) -> None:
    # when
    error, predictions = _decode(answer, classes=["plate"])

    # then
    assert error is False
    assert _serialize(predictions) == []


@pytest.mark.parametrize(
    "answer",
    [
        "not json",
        [PLATE_INSTANCE],
        {"plate": PLATE_INSTANCE},
        {"plate": [{"keypoints": {}}, {"box_2d": [1, 2, 3]}]},
    ],
)
def test_decode_reports_an_invalid_answer(answer: Any) -> None:
    # when
    error, predictions = _decode(answer, classes=["plate"])

    # then
    assert error is True
    assert predictions is None


@pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native carriers; runs only with "
    "ENABLE_TENSOR_DATA_REPRESENTATION=True",
)
def test_decode_returns_the_native_keypoint_tuple() -> None:
    from inference_models.models.base.keypoints_detection import KeyPoints
    from inference_models.models.base.object_detection import Detections

    # when
    error, predictions = _decode({"plate": [PLATE_INSTANCE]}, classes=["plate"])

    # then
    assert error is False
    key_points, detections = predictions
    assert isinstance(key_points, KeyPoints)
    assert isinstance(detections, Detections)
    assert detections.xyxy.cpu().tolist() == [PLATE_XYXY]
    # Every keypoint sits at the slot of its keypoint class id; BR was not placed.
    assert key_points.xy.cpu().tolist() == [
        [[80.0, 100.0], [400.0, 100.0], [0.0, 0.0], [80.0, 300.0]]
    ]
    assert key_points.confidence.cpu().tolist() == [
        [VISIBLE_KEYPOINT_CONFIDENCE, OCCLUDED_KEYPOINT_CONFIDENCE, 0.0, 1.0]
    ]
    assert key_points.class_id.cpu().tolist() == [0]

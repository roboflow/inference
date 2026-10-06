from typing import Optional

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.common.deserializers import (
    deserialize_detections_kind,
)
from roboflow_workflows.core_steps.common.utils import (
    convert_inference_detections_batch_to_sv_detections,
    filter_out_invalid_polygons,
    post_process_ocr_result,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_convert_inference_detections_batch_to_sv_detections_with_invalid_polygons() -> (
    None
):
    # given
    # supervision skips polygons with < 3 points.
    # If we have 2 predictions, one valid and one invalid,
    # sv.Detections.from_inference returns a Detections object of length 1.
    predictions = [
        {
            "image": {"height": 200, "width": 100},
            "predictions": [
                {
                    "width": 50,
                    "height": 100,
                    "x": 50,
                    "y": 100,
                    "confidence": 0.1,
                    "class_id": 1,
                    "points": [
                        {"x": 30, "y": 80},
                        {"x": 30, "y": 120},
                        {"x": 70, "y": 120},
                    ],
                    "class": "dog",
                    "detection_id": "valid",
                    "parent_id": "image",
                },
                {
                    "width": 50,
                    "height": 100,
                    "x": 75,
                    "y": 175,
                    "confidence": 0.2,
                    "class_id": 0,
                    "points": [
                        {"x": 90, "y": 170},
                        {"x": 90, "y": 190},
                    ],  # ONLY 2 POINTS - will be skipped by supervision
                    "class": "cat",
                    "detection_id": "invalid",
                    "parent_id": "image",
                },
            ],
        }
    ]

    # when
    result = convert_inference_detections_batch_to_sv_detections(
        predictions=predictions,
    )

    # then
    assert len(result) == 1
    detections = result[0]

    # Core fields length
    assert len(detections.xyxy) == 1
    assert len(detections.confidence) == 1
    assert len(detections.class_id) == 1

    # Metadata fields length (THIS WAS THE BUG - they used to be length 2)
    assert len(detections.data["detection_id"]) == 1
    assert len(detections.data["parent_id"]) == 1
    assert len(detections.data["image_dimensions"]) == 1

    # Ensure they contain the right data (the valid one)
    assert detections.data["detection_id"][0] == "valid"


def test_deserialize_detections_kind_with_invalid_polygons() -> None:
    # given
    detections_input = {
        "image": {"height": 200, "width": 100},
        "predictions": [
            {
                "width": 10,
                "height": 10,
                "x": 5,
                "y": 5,
                "confidence": 0.5,
                "class_id": 1,
                "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 5, "y": 10}],
                "class": "valid",
            },
            {
                "width": 10,
                "height": 10,
                "x": 15,
                "y": 15,
                "confidence": 0.6,
                "class_id": 2,
                "points": [{"x": 10, "y": 10}, {"x": 20, "y": 20}],  # INVALID
                "class": "invalid",
            },
        ],
    }

    # when
    result = deserialize_detections_kind(
        parameter="test_param",
        detections=detections_input,
    )

    # then
    assert len(result) == 1
    assert len(result.data["detection_id"]) == 1
    assert len(result.data["parent_id"]) == 1
    assert len(result.data["image_dimensions"]) == 1
    assert result.data["parent_id"][0] == "test_param"


def test_post_process_ocr_result_with_invalid_polygons() -> None:
    # given
    predictions = [
        {
            "image": {"height": 200, "width": 100},
            "predictions": [
                {
                    "width": 10,
                    "height": 10,
                    "x": 5,
                    "y": 5,
                    "confidence": 0.5,
                    "class_id": 1,
                    "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 5, "y": 10}],
                    "class": "valid",
                    "detection_id": "valid_ocr",
                },
                {
                    "width": 10,
                    "height": 10,
                    "x": 15,
                    "y": 15,
                    "confidence": 0.6,
                    "class_id": 2,
                    "points": [{"x": 10, "y": 10}, {"x": 20, "y": 20}],  # INVALID
                    "class": "invalid",
                    "detection_id": "invalid_ocr",
                },
            ],
        }
    ]
    images = [
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="image_root"),
            workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="image_root"),
            numpy_image=np.zeros((200, 100, 3), dtype=np.uint8),
        )
    ]
    expected_output_keys = {
        "result",
        "parent_id",
        "root_parent_id",
        "prediction_type",
        "predictions",
    }

    # when
    result = post_process_ocr_result(
        images=images,
        predictions=predictions,
        expected_output_keys=expected_output_keys,
    )

    # then
    assert len(result) == 1
    detections = result[0]["predictions"]
    assert len(detections) == 1
    assert len(detections.data["detection_id"]) == 1
    assert detections.data["detection_id"][0] == "valid_ocr"


def _prediction_with_rle(detection_id: str, *, points: Optional[list]) -> dict:
    return {
        "width": 4,
        "height": 4,
        "x": 2,
        "y": 2,
        "confidence": 0.5,
        "class_id": 0,
        "class": "rle",
        "points": points,
        "rle": {"size": [4, 4], "counts": [0, 16]},
        "detection_id": detection_id,
        "parent_id": "image",
    }


def _mixed_polygon_and_rle_predictions() -> list:
    return [
        {
            "width": 4,
            "height": 4,
            "x": 2,
            "y": 2,
            "confidence": 0.9,
            "class_id": 1,
            "class": "valid",
            "points": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 2, "y": 4}],
            "detection_id": "valid",
            "parent_id": "image",
        },
        {
            "width": 4,
            "height": 4,
            "x": 2,
            "y": 2,
            "confidence": 0.8,
            "class_id": 2,
            "class": "invalid",
            "points": [{"x": 0, "y": 0}, {"x": 4, "y": 4}],
            "detection_id": "invalid",
            "parent_id": "image",
        },
        _prediction_with_rle("rle", points=[]),
    ]


def test_convert_inference_detections_batch_keeps_rle_predictions_with_short_points() -> (
    None
):
    # given
    predictions = [
        {
            "image": {"height": 4, "width": 4},
            "predictions": _mixed_polygon_and_rle_predictions(),
        }
    ]

    # when
    result = convert_inference_detections_batch_to_sv_detections(
        predictions=predictions,
    )

    # then
    detections = result[0]
    assert len(detections) == 2
    assert list(detections.data["detection_id"]) == ["valid", "rle"]
    assert list(detections.data["parent_id"]) == ["image", "image"]
    assert list(detections.class_id) == [1, 0]


@pytest.mark.parametrize("points", [[], None])
def test_convert_inference_detections_batch_keeps_all_rle_predictions_with_empty_or_null_points(
    points: Optional[list],
) -> None:
    # given
    predictions = [
        {
            "image": {"height": 4, "width": 4},
            "predictions": [
                _prediction_with_rle("first", points=points),
                _prediction_with_rle("second", points=points),
            ],
        }
    ]

    # when
    result = convert_inference_detections_batch_to_sv_detections(
        predictions=predictions,
    )

    # then
    detections = result[0]
    assert len(detections) == 2
    assert list(detections.data["detection_id"]) == ["first", "second"]
    assert detections.mask is not None
    assert detections.mask.shape == (2, 4, 4)
    assert detections.mask.all()


def test_deserialize_detections_kind_keeps_rle_predictions_with_short_points() -> None:
    # given
    detections_input = {
        "image": {"height": 4, "width": 4},
        "predictions": _mixed_polygon_and_rle_predictions(),
    }

    # when
    result = deserialize_detections_kind(
        parameter="test_param",
        detections=detections_input,
    )

    # then
    assert len(result) == 2
    assert list(result.data["detection_id"]) == ["valid", "rle"]
    assert list(result.data["parent_id"]) == ["image", "image"]
    assert len(result.data["image_dimensions"]) == 2
    assert list(result.class_id) == [1, 0]


@pytest.mark.parametrize("points", [[], None])
def test_deserialize_detections_kind_keeps_all_rle_predictions_with_empty_or_null_points(
    points: Optional[list],
) -> None:
    # given
    detections_input = {
        "image": {"height": 4, "width": 4},
        "predictions": [
            _prediction_with_rle("first", points=points),
            _prediction_with_rle("second", points=points),
        ],
    }

    # when
    result = deserialize_detections_kind(
        parameter="test_param",
        detections=detections_input,
    )

    # then
    assert len(result) == 2
    assert list(result.data["detection_id"]) == ["first", "second"]
    assert result.mask is not None
    assert result.mask.shape == (2, 4, 4)
    assert result.mask.all()


def test_deserialize_detections_kind_returns_empty_detections_when_predictions_is_none() -> (
    None
):
    # given
    detections_input = {"image": {"height": 4, "width": 4}, "predictions": None}

    # when
    result = deserialize_detections_kind(
        parameter="test_param",
        detections=detections_input,
    )

    # then
    assert len(result) == 0


def test_deserialize_detections_kind_keeps_rle_mask_predictions_with_short_points() -> (
    None
):
    # given
    rle_mask_prediction = _prediction_with_rle("rle_mask", points=[{"x": 0, "y": 0}])
    rle_mask_prediction["rle_mask"] = rle_mask_prediction.pop("rle")
    detections_input = {
        "image": {"height": 4, "width": 4},
        "predictions": [rle_mask_prediction],
    }

    # when
    result = deserialize_detections_kind(
        parameter="test_param",
        detections=detections_input,
    )

    # then
    assert list(result.data["detection_id"]) == ["rle_mask"]
    assert result.mask is not None
    assert result.mask.shape == (1, 4, 4)


_TWO_POINTS = [{"x": 0, "y": 0}, {"x": 4, "y": 4}]
_THREE_POINTS = [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 2, "y": 4}]
_VALID_RLE = {"size": [4, 4], "counts": [0, 16]}


@pytest.mark.parametrize(
    "item, is_kept",
    [
        # dropped: a dict without valid RLE whose `points` is a list/tuple of < 3
        ({"points": []}, False),
        ({"points": _TWO_POINTS[:1]}, False),
        ({"points": _TWO_POINTS}, False),
        ({"points": tuple(_TWO_POINTS)}, False),
        ({"points": _TWO_POINTS, "rle": {"size": [4, 4]}}, False),
        ({"points": _TWO_POINTS, "rle": "not a dict"}, False),
        # kept: 3 or more points
        ({"points": _THREE_POINTS}, True),
        # kept: valid RLE wins over `points`, under either key
        ({"points": _TWO_POINTS, "rle": _VALID_RLE}, True),
        ({"points": _TWO_POINTS, "rle_mask": _VALID_RLE}, True),
        # kept: anything else is left for supervision to handle
        ({}, True),
        ({"points": None}, True),
        ({"points": "ab"}, True),
        ("points", True),
        (None, True),
    ],
)
def test_filter_out_invalid_polygons_drops_only_short_polygons_without_rle(
    item: object,
    is_kept: bool,
) -> None:
    # when
    result = filter_out_invalid_polygons(predictions=[item])

    # then
    assert result == ([item] if is_kept else [])


def test_filter_out_invalid_polygons_passes_non_list_container_through() -> None:
    # when
    result = filter_out_invalid_polygons(predictions=None)

    # then
    assert result is None

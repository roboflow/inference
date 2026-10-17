import numpy as np
import pytest

from inference.core.nms import non_max_suppression_fast, w_np_non_max_suppression


def test_non_max_suppression_fast_when_no_boxes() -> None:
    # given
    boxes = np.empty((0, 5))

    # when
    result = non_max_suppression_fast(boxes, overlapThresh=0.5)

    # then
    assert result == []


def test_non_max_suppression_fast_removes_lower_confidence_overlapping_box() -> None:
    # given
    boxes = np.array(
        [
            [0, 0, 9, 9, 0.9],
            [1, 1, 10, 10, 0.8],
        ],
        dtype=float,
    )

    # when
    result = non_max_suppression_fast(boxes, overlapThresh=0.5)

    # then
    assert len(result) == 1
    assert result[0][4] == 0.9


def test_non_max_suppression_fast_keeps_non_overlapping_boxes() -> None:
    # given
    boxes = np.array(
        [
            [0, 0, 9, 9, 0.9],
            [50, 50, 59, 59, 0.8],
        ],
        dtype=float,
    )

    # when
    result = non_max_suppression_fast(boxes, overlapThresh=0.5)

    # then
    assert len(result) == 2
    assert [box[4] for box in result] == [0.9, 0.8]


def test_w_np_non_max_suppression_raises_on_invalid_box_format() -> None:
    # given
    prediction = np.array([[[50, 50, 20, 20, 0.9, 0.9, 0.1]]], dtype=float)

    # when / then
    with pytest.raises(ValueError):
        w_np_non_max_suppression(prediction, box_format="invalid")


def test_w_np_non_max_suppression_when_no_box_passes_conf_thresh() -> None:
    # given
    prediction = np.array([[[50, 50, 20, 20, 0.1, 0.9, 0.1]]], dtype=float)

    # when
    result = w_np_non_max_suppression(prediction, conf_thresh=0.25)

    # then
    assert result == [[]]


def test_w_np_non_max_suppression_class_agnostic_suppresses_across_classes() -> None:
    # given
    prediction = np.array(
        [
            [
                [0, 0, 9, 9, 0.9, 0.9, 0.0],
                [1, 1, 10, 10, 0.8, 0.0, 0.8],
            ]
        ],
        dtype=float,
    )

    # when
    per_class = w_np_non_max_suppression(
        prediction, box_format="xyxy", class_agnostic=False
    )
    agnostic = w_np_non_max_suppression(
        prediction, box_format="xyxy", class_agnostic=True
    )

    # then
    assert len(per_class[0]) == 2
    assert len(agnostic[0]) == 1
    assert agnostic[0][0][4] == 0.9


def test_w_np_non_max_suppression_respects_max_detections() -> None:
    # given
    prediction = np.array(
        [
            [
                [0, 0, 9, 9, 0.6, 0.6],
                [50, 50, 59, 59, 0.9, 0.9],
                [100, 100, 109, 109, 0.8, 0.8],
            ]
        ],
        dtype=float,
    )

    # when
    result = w_np_non_max_suppression(prediction, box_format="xyxy", max_detections=2)

    # then
    assert [box[4] for box in result[0]] == [0.9, 0.8]


def test_w_np_non_max_suppression_returns_one_result_list_per_image() -> None:
    # given
    prediction = np.array(
        [
            [[0, 0, 9, 9, 0.9, 0.9]],
            [[0, 0, 9, 9, 0.1, 0.1]],
        ],
        dtype=float,
    )

    # when
    result = w_np_non_max_suppression(prediction, box_format="xyxy")

    # then
    assert len(result) == 2
    assert len(result[0]) == 1
    assert result[1] == []

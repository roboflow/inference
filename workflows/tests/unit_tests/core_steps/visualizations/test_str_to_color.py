import numpy as np
import pycocotools.mask as mask_utils
import pytest
import supervision as sv
from roboflow_workflows.core_steps.visualizations.common.utils import (
    ensure_dense_masks,
    str_to_color,
)


def test_str_to_color_with_hex_color() -> None:
    # given
    color = "#FF0000"

    # when
    result = str_to_color(color)

    # then
    assert result == sv.Color.from_hex(color)


def test_str_to_color_with_rgb_color() -> None:
    # given
    color = "rgb(255, 0, 0)"
    expected_color = sv.Color.from_rgb_tuple((255, 0, 0))

    # when
    result = str_to_color(color)

    # then
    assert result == expected_color


def test_str_to_color_with_bgr_color() -> None:
    # given
    color = "bgr(0, 0, 255)"
    expected_color = sv.Color.from_bgr_tuple((0, 0, 255))

    # when
    result = str_to_color(color)

    # then
    assert result == expected_color


def test_str_to_color_with_color_name() -> None:
    # given
    color = "WHITE"

    # when
    result = str_to_color(color)

    # then
    assert result == sv.Color.WHITE


def test_str_to_color_with_invalid_color() -> None:
    # given
    color = "invalid"

    # when
    with pytest.raises(ValueError):
        _ = str_to_color(color)


def _rle_from_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def test_ensure_dense_masks_returns_input_when_mask_already_set() -> None:
    mask = np.zeros((1, 8, 8), dtype=bool)
    mask[0, 2:6, 2:6] = True
    detections = sv.Detections(
        xyxy=np.array([[2, 2, 6, 6]], dtype=np.float64),
        mask=mask,
        class_id=np.array([1]),
    )

    result = ensure_dense_masks(detections)

    assert result is detections
    assert result.mask is mask


def test_ensure_dense_masks_returns_input_when_no_rle_key() -> None:
    detections = sv.Detections(
        xyxy=np.array([[2, 2, 6, 6]], dtype=np.float64),
        mask=None,
        class_id=np.array([1]),
    )

    result = ensure_dense_masks(detections)

    assert result is detections
    assert result.mask is None


def test_ensure_dense_masks_decodes_rle_without_mutating_input() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    detections = sv.Detections(
        xyxy=np.array([[4, 4, 12, 12]], dtype=np.float64),
        mask=None,
        class_id=np.array([1]),
        data={"rle_mask": np.array([_rle_from_mask(mask)], dtype=object)},
    )

    result = ensure_dense_masks(detections)

    assert detections.mask is None
    assert result is not detections
    assert result.mask is not None
    assert result.mask.shape == (1, 16, 16)
    assert result.mask.dtype == bool
    assert np.array_equal(result.mask[0], mask)

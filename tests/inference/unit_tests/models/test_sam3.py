import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from pycocotools import mask as mask_utils


def _load_segment_anything3_functions():
    sam3_mock = MagicMock()
    sam3_mock.eval = MagicMock()
    sam3_mock.eval.postprocessors = MagicMock()
    sam3_mock.model = MagicMock()
    sam3_mock.model.utils = MagicMock()
    sam3_mock.model.utils.misc = MagicMock()
    sam3_mock.train = MagicMock()
    sam3_mock.train.data = MagicMock()
    sam3_mock.train.data.collator = MagicMock()
    sam3_mock.train.data.sam3_image_dataset = MagicMock()
    sam3_mock.train.transforms = MagicMock()
    sam3_mock.train.transforms.basic_for_api = MagicMock()

    mocks = {
        "sam3": sam3_mock,
        "sam3.eval": sam3_mock.eval,
        "sam3.eval.postprocessors": sam3_mock.eval.postprocessors,
        "sam3.model": sam3_mock.model,
        "sam3.model.utils": sam3_mock.model.utils,
        "sam3.model.utils.misc": sam3_mock.model.utils.misc,
        "sam3.model.sam3_image_processor": MagicMock(),
        "sam3.train": sam3_mock.train,
        "sam3.train.data": sam3_mock.train.data,
        "sam3.train.data.collator": sam3_mock.train.data.collator,
        "sam3.train.data.sam3_image_dataset": sam3_mock.train.data.sam3_image_dataset,
        "sam3.train.transforms": sam3_mock.train.transforms,
        "sam3.train.transforms.basic_for_api": sam3_mock.train.transforms.basic_for_api,
    }

    for name, mock in mocks.items():
        sys.modules[name] = mock

    import inference.models.sam3.segment_anything3 as seg_module

    return seg_module


_seg_module = _load_segment_anything3_functions()
_filter_by_threshold = _seg_module._filter_by_threshold
_nms_greedy_pycocotools = _seg_module._nms_greedy_pycocotools
_to_numpy_masks = _seg_module._to_numpy_masks
_masks_to_predictions = _seg_module._masks_to_predictions
_apply_nms_cross_prompt = _seg_module._apply_nms_cross_prompt
_regroup_masks_by_prompt = _seg_module._regroup_masks_by_prompt


# _filter_by_threshold


def test_filter_by_threshold_returns_empty_when_no_scores_pass():
    """When no scores pass the threshold, should return empty array with shape (0, 0, 0)."""
    masks = np.ones((3, 10, 10), dtype=np.uint8)
    scores = [0.1, 0.2, 0.3]

    result_masks, result_scores = _filter_by_threshold(masks, scores, 0.5)

    assert result_masks.shape == (0, 0, 0)
    assert result_masks.dtype == np.uint8
    assert result_scores == []


def test_filter_by_threshold_keeps_scores_above_threshold():
    masks = np.ones((3, 10, 10), dtype=np.uint8)
    masks[0] *= 1
    masks[1] *= 2
    masks[2] *= 3
    scores = [0.3, 0.6, 0.8]

    result_masks, result_scores = _filter_by_threshold(masks, scores, 0.5)

    assert result_masks.shape == (2, 10, 10)
    assert result_scores == [0.6, 0.8]
    assert np.all(result_masks[0] == 2)
    assert np.all(result_masks[1] == 3)


def test_filter_by_threshold_keeps_score_equal_to_threshold():
    masks = np.ones((2, 10, 10), dtype=np.uint8)
    scores = [0.5, 0.3]

    result_masks, result_scores = _filter_by_threshold(masks, scores, 0.5)

    assert result_masks.shape == (1, 10, 10)
    assert result_scores == [0.5]


def test_filter_by_threshold_returns_unchanged_for_invalid_ndim():
    masks_2d = np.ones((10, 10), dtype=np.uint8)
    scores = [0.8]

    result_masks, result_scores = _filter_by_threshold(masks_2d, scores, 0.5)

    assert np.array_equal(result_masks, masks_2d)
    assert result_scores == scores


def test_filter_by_threshold_returns_unchanged_for_empty_masks():
    masks = np.zeros((0, 10, 10), dtype=np.uint8)
    scores = []

    result_masks, result_scores = _filter_by_threshold(masks, scores, 0.5)

    assert np.array_equal(result_masks, masks)
    assert result_scores == []


# _nms_greedy_pycocotools


def _create_rle(x, y, w, h, img_h=100, img_w=100):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 1
    return mask_utils.encode(np.asfortranarray(mask))


def test_nms_greedy_pycocotools_empty_input():
    result = _nms_greedy_pycocotools([], np.array([]), 0.5)

    assert len(result) == 0
    assert result.dtype == bool


def test_nms_greedy_pycocotools_no_suppression_when_no_overlap():
    rle1 = _create_rle(0, 0, 10, 10)
    rle2 = _create_rle(50, 50, 10, 10)

    result = _nms_greedy_pycocotools([rle1, rle2], np.array([0.9, 0.8]), 0.5)

    assert np.all(result == [True, True])


def test_nms_greedy_pycocotools_suppresses_overlapping():
    rle1 = _create_rle(0, 0, 20, 20)
    rle2 = _create_rle(0, 0, 20, 20)

    result = _nms_greedy_pycocotools([rle1, rle2], np.array([0.9, 0.8]), 0.5)

    assert np.all(result == [True, False])


def test_nms_greedy_pycocotools_preserves_original_order():
    rle1 = _create_rle(0, 0, 20, 20)
    rle2 = _create_rle(0, 0, 20, 20)
    rle3 = _create_rle(0, 0, 20, 20)
    confidences = np.array([0.3, 0.9, 0.5])

    result = _nms_greedy_pycocotools([rle1, rle2, rle3], confidences, 0.5)

    assert result[0] == False  # idx 0 suppressed
    assert result[1] == True  # idx 1 kept (highest conf)
    assert result[2] == False  # idx 2 suppressed


def test_nms_greedy_pycocotools_partial_overlap_below_threshold():
    rle1 = _create_rle(0, 0, 20, 20)
    rle2 = _create_rle(10, 10, 20, 20)  # IoU ~0.14, below 0.5

    result = _nms_greedy_pycocotools([rle1, rle2], np.array([0.9, 0.8]), 0.5)

    assert np.all(result == [True, True])


def test_nms_greedy_pycocotools_single_detection():
    rle = _create_rle(0, 0, 20, 20)

    result = _nms_greedy_pycocotools([rle], np.array([0.9]), 0.5)

    assert np.all(result == [True])


# _masks_to_predictions


def test_masks_to_predictions_invalid_ndim():
    masks_2d = np.ones((10, 10), dtype=np.uint8)

    result = _masks_to_predictions(masks_2d, [0.9], "polygon")

    assert result == []


def test_masks_to_predictions_empty_shape():
    masks = np.zeros((0, 10, 10), dtype=np.uint8)

    result = _masks_to_predictions(masks, [], "polygon")

    assert result == []


def test_masks_to_predictions_rle_format():
    masks = np.zeros((1, 10, 10), dtype=np.uint8)
    masks[0, 2:8, 2:8] = 1

    result = _masks_to_predictions(masks, [0.9], "rle")

    assert len(result) == 1
    assert result[0].format == "rle"
    assert result[0].confidence == 0.9
    assert isinstance(result[0].masks, dict)
    assert "counts" in result[0].masks


def test_masks_to_predictions_polygon_format():
    masks = np.zeros((1, 10, 10), dtype=np.uint8)
    masks[0, 2:8, 2:8] = 1

    result = _masks_to_predictions(masks, [0.9], "polygon")

    assert len(result) == 1
    assert result[0].format == "polygon"
    assert result[0].confidence == 0.9
    assert isinstance(result[0].masks, list)


# _apply_nms_cross_prompt


def test_apply_nms_cross_prompt_empty_list():
    result = _apply_nms_cross_prompt([], 0.5)

    assert result == []


def test_apply_nms_cross_prompt_no_overlap():
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[0:20, 0:20] = 1
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[50:70, 50:70] = 1

    result = _apply_nms_cross_prompt([(0, mask1, 0.9), (1, mask2, 0.8)], 0.5)

    assert len(result) == 2


def test_apply_nms_cross_prompt_with_overlap():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:50, 0:50] = 1

    result = _apply_nms_cross_prompt(
        [(0, mask.copy(), 0.9), (1, mask.copy(), 0.7)], 0.5
    )

    assert len(result) == 1
    assert result[0][2] == 0.9


# _regroup_masks_by_prompt


def test_regroup_masks_by_prompt_empty():
    result = _regroup_masks_by_prompt([], 3)

    assert result == {0: [], 1: [], 2: []}


def test_regroup_masks_by_prompt_groups_correctly():
    mask1 = np.ones((10, 10), dtype=np.uint8)
    mask2 = np.ones((10, 10), dtype=np.uint8) * 2
    mask3 = np.ones((10, 10), dtype=np.uint8) * 3

    filtered_masks = [
        (0, mask1, 0.9),
        (1, mask2, 0.8),
        (0, mask3, 0.7),
    ]

    result = _regroup_masks_by_prompt(filtered_masks, 2)

    assert len(result[0]) == 2
    assert len(result[1]) == 1
    assert result[0][0] == (mask1, 0.9)
    assert result[0][1] == (mask3, 0.7)
    assert result[1][0] == (mask2, 0.8)

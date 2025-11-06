import numpy as np
import pytest
from pycocotools import mask as mask_utils

from inference.core.nms import nms_rle
from inference.core.workflows.core_steps.models.foundation.seg_preview.v1 import (
    apply_nms_to_polygon,
)


def test_nms_with_empty_input():
    """Test NMS with empty input returns empty array"""
    # given
    rles = []
    confidences = np.array([])
    iou_threshold = 0.5

    # when
    result = nms_rle(rles, confidences, iou_threshold)

    # then
    assert len(result) == 0
    assert result.dtype == bool


def test_nms_with_single_detection():
    """Test NMS with single detection keeps it"""
    # given
    # Create a simple mask: 10x10 box at position (5, 5)
    mask = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask[5:15, 5:15] = 1
    rle = mask_utils.encode(mask)

    rles = [rle]
    confidences = np.array([0.9])
    iou_threshold = 0.5

    # when
    result = nms_rle(rles, confidences, iou_threshold)

    # then
    assert len(result) == 1
    assert result[0] == True


def test_nms_with_non_overlapping_detections():
    """Test NMS keeps all non-overlapping detections"""
    # given
    # Create two non-overlapping masks
    mask1 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask1[0:5, 0:5] = 1
    rle1 = mask_utils.encode(mask1)

    mask2 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask2[15:20, 15:20] = 1
    rle2 = mask_utils.encode(mask2)

    rles = [rle1, rle2]
    confidences = np.array([0.9, 0.8])
    iou_threshold = 0.5

    # when
    result = nms_rle(rles, confidences, iou_threshold)

    # then
    assert len(result) == 2
    assert np.all(result == True)


def test_nms_suppresses_overlapping_lower_confidence():
    """Test NMS suppresses highly overlapping detection with lower confidence"""
    # given
    # Create two highly overlapping masks
    mask1 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask1[5:15, 5:15] = 1
    rle1 = mask_utils.encode(mask1)

    mask2 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask2[6:16, 6:16] = 1  # Slightly shifted, high overlap
    rle2 = mask_utils.encode(mask2)

    rles = [rle1, rle2]
    confidences = np.array([0.9, 0.8])  # First has higher confidence
    iou_threshold = 0.5

    # when
    result = nms_rle(rles, confidences, iou_threshold)

    # then
    assert len(result) == 2
    assert result[0] == True  # Higher confidence kept
    assert result[1] == False  # Lower confidence suppressed


def test_nms_keeps_lower_overlap_detection():
    """Test NMS keeps detection with overlap below threshold"""
    # given
    # Create two masks with small overlap (below threshold)
    mask1 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask1[5:10, 5:10] = 1
    rle1 = mask_utils.encode(mask1)

    mask2 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask2[8:13, 8:13] = 1  # Small overlap
    rle2 = mask_utils.encode(mask2)

    rles = [rle1, rle2]
    confidences = np.array([0.9, 0.8])
    iou_threshold = 0.8  # High threshold

    # when
    result = nms_rle(rles, confidences, iou_threshold)

    # then
    assert len(result) == 2
    assert np.all(result == True)  # Both kept due to low overlap


def test_nms_respects_confidence_order():
    """Test NMS processes detections in confidence order"""
    # given
    # Create three overlapping masks
    mask1 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask1[5:15, 5:15] = 1
    rle1 = mask_utils.encode(mask1)

    mask2 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask2[6:16, 6:16] = 1
    rle2 = mask_utils.encode(mask2)

    mask3 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask3[7:17, 7:17] = 1
    rle3 = mask_utils.encode(mask3)

    # Low, High, Medium confidence
    rles = [rle1, rle2, rle3]
    confidences = np.array([0.5, 0.9, 0.7])
    iou_threshold = 0.5

    # when
    result = nms_rle(rles, confidences, iou_threshold)

    # then
    assert len(result) == 3
    assert result[0] == False  # Lowest confidence suppressed
    assert result[1] == True   # Highest confidence kept
    assert result[2] == False  # Medium confidence suppressed


def test_nms_returns_indices_in_original_order():
    """Test NMS returns keep flags in original input order"""
    # given
    mask1 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask1[5:15, 5:15] = 1
    rle1 = mask_utils.encode(mask1)

    mask2 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask2[6:16, 6:16] = 1
    rle2 = mask_utils.encode(mask2)

    # Give them in reverse confidence order
    rles = [rle1, rle2]
    confidences = np.array([0.6, 0.9])  # Second has higher confidence
    iou_threshold = 0.5

    # when
    result = nms_rle(rles, confidences, iou_threshold)

    # then
    assert len(result) == 2
    assert result[0] == False  # First (lower confidence) suppressed
    assert result[1] == True   # Second (higher confidence) kept


def test_nms_with_different_iou_thresholds():
    """Test NMS behavior with different IoU thresholds"""
    # given
    mask1 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask1[5:15, 5:15] = 1
    rle1 = mask_utils.encode(mask1)

    mask2 = np.zeros((20, 20), dtype=np.uint8, order='F')
    mask2[8:18, 8:18] = 1  # Moderate overlap
    rle2 = mask_utils.encode(mask2)

    rles = [rle1, rle2]
    confidences = np.array([0.9, 0.8])

    # when - high threshold (more permissive)
    result_high = nms_rle(rles, confidences, iou_threshold=0.8)
    # when - low threshold (more strict)
    result_low = nms_rle(rles, confidences, iou_threshold=0.2)

    # then
    assert np.all(result_high == True)  # Both kept with high threshold
    assert result_low[0] == True and result_low[1] == False  # One suppressed with low threshold


def test_apply_nms_to_polygon_with_empty_input():
    """Test apply_nms_to_polygon with empty input returns empty list"""
    # given
    polygon_list = []
    image_width, image_height = 100, 100
    iou_threshold = 0.5

    # when
    result = apply_nms_to_polygon(polygon_list, image_width, image_height, iou_threshold)

    # then
    assert result == []


def test_apply_nms_to_polygon_with_single_detection():
    """Test apply_nms_to_polygon with single detection returns it"""
    # given
    polygon_list = [
        {
            "points": [
                {"x": 10, "y": 10},
                {"x": 20, "y": 10},
                {"x": 20, "y": 20},
                {"x": 10, "y": 20}
            ],
            "confidence": 0.9,
            "class": "person"
        }
    ]
    image_width, image_height = 100, 100
    iou_threshold = 0.5

    # when
    result = apply_nms_to_polygon(polygon_list, image_width, image_height, iou_threshold)

    # then
    assert len(result) == 1
    assert result[0] == polygon_list[0]


def test_apply_nms_to_polygon_keeps_non_overlapping():
    """Test apply_nms_to_polygon keeps non-overlapping detections"""
    # given
    polygon_list = [
        {
            "points": [
                {"x": 10, "y": 10},
                {"x": 20, "y": 10},
                {"x": 20, "y": 20},
                {"x": 10, "y": 20}
            ],
            "confidence": 0.9,
            "class": "person"
        },
        {
            "points": [
                {"x": 80, "y": 80},
                {"x": 90, "y": 80},
                {"x": 90, "y": 90},
                {"x": 80, "y": 90}
            ],
            "confidence": 0.8,
            "class": "person"
        }
    ]
    image_width, image_height = 100, 100
    iou_threshold = 0.5

    # when
    result = apply_nms_to_polygon(polygon_list, image_width, image_height, iou_threshold)

    # then
    assert len(result) == 2


def test_apply_nms_to_polygon_suppresses_overlapping():
    """Test apply_nms_to_polygon suppresses overlapping lower confidence detection"""
    # given
    polygon_list = [
        {
            "points": [
                {"x": 10, "y": 10},
                {"x": 30, "y": 10},
                {"x": 30, "y": 30},
                {"x": 10, "y": 30}
            ],
            "confidence": 0.9,
            "class": "person"
        },
        {
            "points": [
                {"x": 12, "y": 12},
                {"x": 32, "y": 12},
                {"x": 32, "y": 32},
                {"x": 12, "y": 32}
            ],
            "confidence": 0.8,
            "class": "person"
        }
    ]
    image_width, image_height = 100, 100
    iou_threshold = 0.5

    # when
    result = apply_nms_to_polygon(polygon_list, image_width, image_height, iou_threshold)

    # then
    assert len(result) == 1
    assert result[0] == polygon_list[0]  # Only highest confidence kept


def test_apply_nms_to_polygon_preserves_metadata():
    """Test apply_nms_to_polygon preserves all polygon metadata"""
    # given
    polygon_list = [
        {
            "points": [
                {"x": 10, "y": 10},
                {"x": 20, "y": 10},
                {"x": 20, "y": 20},
                {"x": 10, "y": 20}
            ],
            "confidence": 0.9,
            "class": "person",
            "class_id": 0,
            "extra_field": "test_value"
        }
    ]
    image_width, image_height = 100, 100
    iou_threshold = 0.5

    # when
    result = apply_nms_to_polygon(polygon_list, image_width, image_height, iou_threshold)

    # then
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9
    assert result[0]["class"] == "person"
    assert result[0]["class_id"] == 0
    assert result[0]["extra_field"] == "test_value"

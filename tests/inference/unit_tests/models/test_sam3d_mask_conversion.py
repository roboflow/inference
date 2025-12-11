"""Tests for SAM3D mask conversion utilities."""

import numpy as np
import pytest

from inference.models.sam3_3d.segment_anything_3d import (
    convert_mask_to_binary,
    _is_single_mask_input,
    _normalize_binary_mask,
    _parse_polygon_to_points,
)


class TestConvertMaskToBinary:
    """Test convert_mask_to_binary function."""

    def test_binary_mask_passthrough(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        result = convert_mask_to_binary(mask, (100, 100))
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert np.array_equal(result, mask)

    def test_binary_mask_bool(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        result = convert_mask_to_binary(mask, (100, 100))
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert result[50, 50] == 255
        assert result[0, 0] == 0

    def test_binary_mask_0_1_range(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        result = convert_mask_to_binary(mask, (100, 100))
        assert result[50, 50] == 255
        assert result[0, 0] == 0

    def test_binary_mask_resize(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        result = convert_mask_to_binary(mask, (100, 100))
        assert result.shape == (100, 100)

    def test_polygon_flat_coco(self):
        polygon = [10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]
        result = convert_mask_to_binary(polygon, (100, 100))
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert result[50, 50] == 255
        assert result[0, 0] == 0

    def test_polygon_list_of_points(self):
        polygon = [[10, 10], [90, 10], [90, 90], [10, 90]]
        result = convert_mask_to_binary(polygon, (100, 100))
        assert result.shape == (100, 100)
        assert result[50, 50] == 255
        assert result[0, 0] == 0

    def test_polygon_numpy_array(self):
        polygon = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        result = convert_mask_to_binary(polygon, (100, 100))
        assert result.shape == (100, 100)
        assert result[50, 50] == 255

    def test_polygon_tuples(self):
        polygon = [(10, 10), (90, 10), (90, 90), (10, 90)]
        result = convert_mask_to_binary(polygon, (100, 100))
        assert result.shape == (100, 100)
        assert result[50, 50] == 255

    def test_empty_polygon(self):
        result = convert_mask_to_binary([], (100, 100))
        assert result.shape == (100, 100)
        assert result.sum() == 0

    def test_invalid_polygon_too_few_points(self):
        polygon = [10.0, 10.0, 20.0, 20.0]  # Only 2 points
        result = convert_mask_to_binary(polygon, (100, 100))
        assert result.shape == (100, 100)
        assert result.sum() == 0

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            convert_mask_to_binary("invalid", (100, 100))


class TestRLEConversion:
    """Test RLE mask conversion (requires pycocotools)."""

    @pytest.fixture
    def sample_rle(self):
        pytest.importorskip("pycocotools")
        import pycocotools.mask as mask_utils
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def test_rle_conversion(self, sample_rle):
        result = convert_mask_to_binary(sample_rle, (100, 100))
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert result[50, 50] == 255
        assert result[0, 0] == 0

    def test_rle_bytes_counts(self):
        pytest.importorskip("pycocotools")
        import pycocotools.mask as mask_utils
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        # Keep counts as bytes
        result = convert_mask_to_binary(rle, (100, 100))
        assert result.shape == (100, 100)


class TestIsSingleMaskInput:
    """Test _is_single_mask_input function."""

    def test_none(self):
        assert _is_single_mask_input(None) is True

    def test_empty_list(self):
        assert _is_single_mask_input([]) is True

    def test_2d_array(self):
        assert _is_single_mask_input(np.zeros((100, 100))) is True

    def test_3d_array(self):
        assert _is_single_mask_input(np.zeros((3, 100, 100))) is False

    def test_flat_polygon(self):
        assert _is_single_mask_input([10.0, 10.0, 20.0, 20.0]) is True

    def test_list_of_points(self):
        assert _is_single_mask_input([[10, 10], [20, 20]]) is True

    def test_multiple_flat_polygons(self):
        assert _is_single_mask_input([[10, 10, 20, 20], [30, 30, 40, 40]]) is False

    def test_single_rle(self):
        assert _is_single_mask_input({"counts": "abc", "size": [100, 100]}) is True

    def test_multiple_rle(self):
        rles = [{"counts": "abc", "size": [100, 100]}, {"counts": "def", "size": [100, 100]}]
        assert _is_single_mask_input(rles) is False

    def test_list_of_binary_masks(self):
        masks = [np.zeros((100, 100), dtype=np.uint8), np.zeros((100, 100), dtype=np.uint8)]
        assert _is_single_mask_input(masks) is False


class TestNormalizeBinaryMask:
    """Test _normalize_binary_mask function."""

    def test_3d_to_2d(self):
        mask = np.zeros((100, 100, 1), dtype=np.uint8)
        mask[50, 50, 0] = 255
        result = _normalize_binary_mask(mask, (100, 100))
        assert result.ndim == 2
        assert result[50, 50] == 255

    def test_float_mask(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[50, 50] = 1.0
        result = _normalize_binary_mask(mask, (100, 100))
        assert result.dtype == np.uint8
        assert result[50, 50] == 255


class TestParsePolygonToPoints:
    """Test _parse_polygon_to_points function."""

    def test_flat_list(self):
        result = _parse_polygon_to_points([10, 20, 30, 40])
        assert result == [(10.0, 20.0), (30.0, 40.0)]

    def test_list_of_lists(self):
        result = _parse_polygon_to_points([[10, 20], [30, 40]])
        assert result == [(10.0, 20.0), (30.0, 40.0)]

    def test_numpy_2d(self):
        arr = np.array([[10, 20], [30, 40]])
        result = _parse_polygon_to_points(arr)
        assert result == [(10.0, 20.0), (30.0, 40.0)]

    def test_empty(self):
        assert _parse_polygon_to_points([]) == []
        assert _parse_polygon_to_points(None) == []


class TestMultipleMasksConversion:
    """Test converting lists of different mask types."""

    def test_list_of_binary_masks(self):
        masks = [
            np.zeros((100, 100), dtype=np.uint8),
            np.zeros((100, 100), dtype=np.uint8),
        ]
        masks[0][20:80, 20:80] = 255
        masks[1][30:70, 30:70] = 255

        assert _is_single_mask_input(masks) is False
        results = [convert_mask_to_binary(m, (100, 100)) for m in masks]
        assert len(results) == 2
        assert results[0][50, 50] == 255
        assert results[1][50, 50] == 255

    def test_list_of_flat_polygons(self):
        polygons = [
            [10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0],
            [20.0, 20.0, 80.0, 20.0, 80.0, 80.0, 20.0, 80.0],
        ]

        assert _is_single_mask_input(polygons) is False
        results = [convert_mask_to_binary(p, (100, 100)) for p in polygons]
        assert len(results) == 2
        assert results[0][50, 50] == 255
        assert results[1][50, 50] == 255

    def test_list_of_rle_dicts(self):
        pytest.importorskip("pycocotools")
        import pycocotools.mask as mask_utils

        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:80, 20:80] = 1
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[30:70, 30:70] = 1

        rle1 = mask_utils.encode(np.asfortranarray(mask1))
        rle1["counts"] = rle1["counts"].decode("utf-8")
        rle2 = mask_utils.encode(np.asfortranarray(mask2))
        rle2["counts"] = rle2["counts"].decode("utf-8")

        rles = [rle1, rle2]

        assert _is_single_mask_input(rles) is False
        results = [convert_mask_to_binary(r, (100, 100)) for r in rles]
        assert len(results) == 2
        assert results[0][50, 50] == 255
        assert results[1][50, 50] == 255


class TestWorkflowExtractMasks:
    """Test workflow block mask extraction."""

    def test_extract_from_sv_detections(self):
        sv = pytest.importorskip("supervision")
        from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
            extract_masks_from_input,
        )

        masks = np.array([
            np.zeros((100, 100), dtype=bool),
            np.zeros((100, 100), dtype=bool),
        ])
        masks[0, 20:80, 20:80] = True
        masks[1, 30:70, 30:70] = True

        detections = sv.Detections(
            xyxy=np.array([[20, 20, 80, 80], [30, 30, 70, 70]]),
            mask=masks,
        )

        result = extract_masks_from_input(detections)
        assert len(result) == 2
        assert result[0].shape == (100, 100)

    def test_passthrough_polygon(self):
        from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
            extract_masks_from_input,
        )
        polygon = [10.0, 10.0, 90.0, 90.0]
        result = extract_masks_from_input(polygon)
        assert result == polygon

    def test_passthrough_binary_mask(self):
        from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
            extract_masks_from_input,
        )
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = extract_masks_from_input(mask)
        assert np.array_equal(result, mask)

    def test_empty_detections_raises(self):
        sv = pytest.importorskip("supervision")
        from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
            extract_masks_from_input,
        )

        detections = sv.Detections.empty()
        with pytest.raises(ValueError, match="no detections"):
            extract_masks_from_input(detections)

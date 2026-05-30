import warnings
from typing import Iterable, List, Tuple

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from inference.core.utils.postprocess import mask2poly
from inference.core.utils.rle_to_polygon import rle_masks_to_polygons
from inference_models.models.base.types import InstancesRLEMasks

pytestmark = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword.*:DeprecationWarning"
)


def _legacy_rle_masks2poly(masks: InstancesRLEMasks) -> List[np.ndarray]:
    segments = []
    h, w = masks.image_size
    for counts in masks.masks:
        rle_dict = {"size": [h, w], "counts": counts}
        decoded_rle = np.ascontiguousarray(mask_utils.decode(rle_dict))
        if not np.any(decoded_rle):
            segments.append(np.zeros((0, 2), dtype=np.float32))
            continue
        segments.append(mask2poly(decoded_rle))
    return segments


def _to_instances(masks: Iterable[np.ndarray]) -> InstancesRLEMasks:
    masks = list(masks)
    assert masks
    image_size = tuple(masks[0].shape)
    rles = [_encode_mask(mask) for mask in masks]
    return InstancesRLEMasks(image_size=image_size, masks=rles)


def _encode_mask(mask: np.ndarray) -> bytes:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="__array__ implementation doesn't accept a copy keyword.*",
            category=DeprecationWarning,
        )
        return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]


def _assert_polygons_exactly_equal(
    actual: List[np.ndarray],
    expected: List[np.ndarray],
) -> None:
    assert len(actual) == len(expected)
    for actual_poly, expected_poly in zip(actual, expected):
        assert actual_poly.dtype == expected_poly.dtype
        assert actual_poly.shape == expected_poly.shape
        assert np.array_equal(actual_poly, expected_poly)


def _uncompressed_counts(mask: np.ndarray) -> List[int]:
    flat = mask.astype(np.uint8).ravel(order="F")
    if flat.size == 0:
        return []
    counts = []
    current = 0
    run_length = 0
    for value in flat:
        value = int(value)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            current = value
            run_length = 1
    counts.append(run_length)
    return counts


class _FakeLazyRLEMasks:
    def __init__(self, image_size: Tuple[int, int], masks: List[np.ndarray]) -> None:
        self.image_size = image_size
        counts = [_uncompressed_counts(mask) for mask in masks]
        max_len = max(len(c) for c in counts)
        self._rle_counts_cpu = np.zeros((len(counts), max_len), dtype=np.int32)
        self._rle_lengths_cpu = np.asarray([len(c) for c in counts], dtype=np.int32)
        for i, count in enumerate(counts):
            self._rle_counts_cpu[i, : len(count)] = count

    def _ensure_rle_cpu(self) -> None:
        pass


def _deterministic_masks() -> List[np.ndarray]:
    masks = []

    masks.append(np.zeros((12, 14), dtype=np.uint8))

    single_pixels = np.zeros((12, 14), dtype=np.uint8)
    single_pixels[0, 0] = 1
    single_pixels[5, 7] = 1
    single_pixels[11, 13] = 1
    masks.append(single_pixels)

    full = np.ones((12, 14), dtype=np.uint8)
    masks.append(full)

    touching_border = np.zeros((12, 14), dtype=np.uint8)
    touching_border[0:8, 0:5] = 1
    touching_border[4:12, 9:14] = 1
    masks.append(touching_border)

    ring = np.zeros((18, 20), dtype=np.uint8)
    ring[2:16, 2:18] = 1
    ring[5:13, 6:14] = 0
    masks.append(ring)

    side_by_side_holes = np.zeros((18, 22), dtype=np.uint8)
    side_by_side_holes[2:16, 2:20] = 1
    side_by_side_holes[5:13, 5:8] = 0
    side_by_side_holes[5:13, 13:16] = 0
    masks.append(side_by_side_holes)

    diagonal = np.zeros((16, 16), dtype=np.uint8)
    for i in range(2, 14):
        diagonal[i, i] = 1
        diagonal[i, i - 1] = 1
    masks.append(diagonal)

    equal_length_components = np.zeros((16, 20), dtype=np.uint8)
    equal_length_components[2:6, 2:6] = 1
    equal_length_components[10:14, 14:18] = 1
    masks.append(equal_length_components)

    jagged = np.zeros((24, 26), dtype=np.uint8)
    jagged[3:20, 4:22] = 1
    jagged[7:11, 8:20] = 0
    jagged[14:18, 9:17] = 0
    jagged[5:9, 21:24] = 1
    jagged[18:23, 2:9] = 1
    masks.append(jagged)

    return masks


def test_rle_masks_to_polygons_matches_legacy_dense_path_on_adversarial_masks() -> None:
    for mask in _deterministic_masks():
        instances = _to_instances([mask])

        actual = rle_masks_to_polygons(masks=instances)
        expected = _legacy_rle_masks2poly(masks=instances)

        _assert_polygons_exactly_equal(actual=actual, expected=expected)


def test_rle_masks_to_polygons_matches_legacy_dense_path_on_random_masks() -> None:
    rng = np.random.default_rng(20260530)
    for height, width in [(1, 1), (2, 3), (5, 7), (16, 17), (31, 29), (64, 64)]:
        masks = []
        for density in [0.0, 0.01, 0.05, 0.15, 0.35, 0.65, 1.0]:
            for _ in range(8):
                masks.append((rng.random((height, width)) < density).astype(np.uint8))
        instances = _to_instances(masks)

        actual = rle_masks_to_polygons(masks=instances)
        expected = _legacy_rle_masks2poly(masks=instances)

        _assert_polygons_exactly_equal(actual=actual, expected=expected)


def test_rle_masks_to_polygons_matches_legacy_dense_path_for_lazy_uncompressed_counts() -> (
    None
):
    for mask in _deterministic_masks():
        legacy_instances = _to_instances([mask])
        lazy_instances = _FakeLazyRLEMasks(
            image_size=legacy_instances.image_size,
            masks=[mask],
        )

        actual = rle_masks_to_polygons(masks=lazy_instances)
        expected = _legacy_rle_masks2poly(masks=legacy_instances)

        _assert_polygons_exactly_equal(actual=actual, expected=expected)


def test_adapter_rle_masks2poly_matches_legacy_dense_path() -> None:
    from inference.core.models.inference_models_adapters import rle_masks2poly

    for mask in _deterministic_masks():
        instances = _to_instances([mask])

        actual = rle_masks2poly(masks=instances)
        expected = _legacy_rle_masks2poly(masks=instances)

        _assert_polygons_exactly_equal(actual=actual, expected=expected)

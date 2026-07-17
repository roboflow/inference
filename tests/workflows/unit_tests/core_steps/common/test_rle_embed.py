from typing import List, Tuple

import numpy as np
import pytest
import torch

from inference.core.workflows.core_steps.common.tensor_native import (
    embed_rle_masks_in_larger_canvas,
)
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)


def _encode_dense_slices(dense_slices: List[np.ndarray]) -> InstancesRLEMasks:
    """Encode a list of dense (h, w) bool/uint8 slices through the real encoder."""
    assert len(dense_slices) > 0
    h, w = dense_slices[0].shape
    encoded = [
        torch_mask_to_coco_rle(torch.from_numpy(np.ascontiguousarray(s).astype(bool)))
        for s in dense_slices
    ]
    return InstancesRLEMasks.from_coco_rle_masks(image_size=(h, w), masks=encoded)


def _numpy_reference_canvas(
    dense_slice: np.ndarray,
    offset_xy: Tuple[int, int],
    target_size_hw: Tuple[int, int],
) -> np.ndarray:
    """All-zeros (H, W) canvas with the slice written at [y0:y0+h, x0:x0+w]."""
    h, w = dense_slice.shape
    x0, y0 = offset_xy
    target_h, target_w = target_size_hw
    canvas = np.zeros((target_h, target_w), dtype=bool)
    canvas[y0 : y0 + h, x0 : x0 + w] = dense_slice.astype(bool)
    return canvas


def _assert_embed_matches_reference(
    dense_slices: List[np.ndarray],
    offset_xy: Tuple[int, int],
    target_size_hw: Tuple[int, int],
) -> None:
    wrapped = _encode_dense_slices(dense_slices)
    embedded = embed_rle_masks_in_larger_canvas(
        masks=wrapped, offset_xy=offset_xy, target_size_hw=target_size_hw
    )
    assert embedded.image_size == target_size_hw
    assert len(embedded.masks) == len(dense_slices)
    decoded = coco_rle_masks_to_numpy_mask(embedded)
    assert decoded.shape == (len(dense_slices), target_size_hw[0], target_size_hw[1])
    for i, dense_slice in enumerate(dense_slices):
        expected = _numpy_reference_canvas(dense_slice, offset_xy, target_size_hw)
        np.testing.assert_array_equal(decoded[i], expected)


def test_embed_empty_masks_returns_empty_on_canvas() -> None:
    wrapped = InstancesRLEMasks(image_size=(10, 12), masks=[])
    embedded = embed_rle_masks_in_larger_canvas(
        masks=wrapped, offset_xy=(2, 3), target_size_hw=(20, 30)
    )
    assert embedded.image_size == (20, 30)
    assert embedded.masks == []
    decoded = coco_rle_masks_to_numpy_mask(embedded)
    assert decoded.shape == (0, 20, 30)


def test_embed_single_instance_interior_placement() -> None:
    rng = np.random.default_rng(seed=1)
    dense = rng.integers(0, 2, size=(6, 8)).astype(bool)
    _assert_embed_matches_reference([dense], offset_xy=(5, 4), target_size_hw=(20, 25))


def test_embed_single_instance_top_left_origin() -> None:
    rng = np.random.default_rng(seed=2)
    dense = rng.integers(0, 2, size=(7, 9)).astype(bool)
    _assert_embed_matches_reference([dense], offset_xy=(0, 0), target_size_hw=(15, 18))


def test_embed_multiple_instances_interior() -> None:
    rng = np.random.default_rng(seed=3)
    dense_slices = [rng.integers(0, 2, size=(5, 6)).astype(bool) for _ in range(4)]
    _assert_embed_matches_reference(
        dense_slices, offset_xy=(3, 7), target_size_hw=(20, 16)
    )


def test_embed_full_zero_slice() -> None:
    dense = np.zeros((4, 5), dtype=bool)
    _assert_embed_matches_reference([dense], offset_xy=(2, 1), target_size_hw=(12, 14))


def test_embed_full_one_slice() -> None:
    dense = np.ones((4, 5), dtype=bool)
    _assert_embed_matches_reference([dense], offset_xy=(2, 1), target_size_hw=(12, 14))


def test_embed_slice_with_full_zero_and_full_one_columns() -> None:
    # column 0 all zeros, column 1 all ones, then mixed columns.
    dense = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    )
    _assert_embed_matches_reference([dense], offset_xy=(4, 2), target_size_hw=(10, 12))


def test_embed_slice_fills_entire_canvas() -> None:
    rng = np.random.default_rng(seed=4)
    dense = rng.integers(0, 2, size=(8, 11)).astype(bool)
    _assert_embed_matches_reference([dense], offset_xy=(0, 0), target_size_hw=(8, 11))


def test_embed_slice_flush_to_bottom_right() -> None:
    rng = np.random.default_rng(seed=5)
    h, w = 6, 7
    target_h, target_w = 20, 18
    dense = rng.integers(0, 2, size=(h, w)).astype(bool)
    _assert_embed_matches_reference(
        [dense],
        offset_xy=(target_w - w, target_h - h),
        target_size_hw=(target_h, target_w),
    )


def test_embed_single_pixel_slice() -> None:
    dense = np.ones((1, 1), dtype=bool)
    _assert_embed_matches_reference([dense], offset_xy=(9, 13), target_size_hw=(30, 25))


@pytest.mark.parametrize("seed", list(range(12)))
def test_embed_random_small_masks_match_reference(seed: int) -> None:
    rng = np.random.default_rng(seed=seed)
    h = int(rng.integers(1, 9))
    w = int(rng.integers(1, 9))
    target_h = h + int(rng.integers(0, 12))
    target_w = w + int(rng.integers(0, 12))
    x0 = int(rng.integers(0, target_w - w + 1))
    y0 = int(rng.integers(0, target_h - h + 1))
    n = int(rng.integers(1, 4))
    dense_slices = [rng.integers(0, 2, size=(h, w)).astype(bool) for _ in range(n)]
    _assert_embed_matches_reference(
        dense_slices, offset_xy=(x0, y0), target_size_hw=(target_h, target_w)
    )


def test_embed_raises_when_slice_does_not_fit_in_width() -> None:
    wrapped = _encode_dense_slices([np.ones((4, 6), dtype=bool)])
    with pytest.raises(ValueError):
        embed_rle_masks_in_larger_canvas(
            masks=wrapped, offset_xy=(5, 0), target_size_hw=(10, 10)
        )


def test_embed_raises_when_slice_does_not_fit_in_height() -> None:
    wrapped = _encode_dense_slices([np.ones((6, 4), dtype=bool)])
    with pytest.raises(ValueError):
        embed_rle_masks_in_larger_canvas(
            masks=wrapped, offset_xy=(0, 5), target_size_hw=(10, 10)
        )


def test_embed_raises_on_negative_offset() -> None:
    wrapped = _encode_dense_slices([np.ones((4, 4), dtype=bool)])
    with pytest.raises(ValueError):
        embed_rle_masks_in_larger_canvas(
            masks=wrapped, offset_xy=(-1, 0), target_size_hw=(10, 10)
        )

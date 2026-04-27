from typing import List, Tuple

import numpy as np
import pytest
import torch

from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    coco_rle_masks_to_torch_mask,
    torch_mask_to_coco_rle,
)


def _make_rectangle_mask(
    h: int, w: int, box: Tuple[int, int, int, int]
) -> torch.Tensor:
    """Build a (H, W) bool tensor with a filled rectangle at `box=(y0, x0, y1, x1)`."""
    m = torch.zeros((h, w), dtype=torch.bool)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def _rle_from_tensors(masks: List[torch.Tensor]) -> InstancesRLEMasks:
    """Encode a list of (H, W) bool tensors through the real encoder and wrap."""
    assert len(masks) > 0
    h, w = masks[0].shape
    encoded = [torch_mask_to_coco_rle(m) for m in masks]
    return InstancesRLEMasks.from_coco_rle_masks(image_size=(h, w), masks=encoded)


def test_torch_mask_to_coco_rle_returns_dict_with_size_and_counts() -> None:
    mask = _make_rectangle_mask(20, 30, (5, 10, 15, 25))
    rle = torch_mask_to_coco_rle(mask)
    assert isinstance(rle, dict)
    assert "size" in rle and "counts" in rle
    assert list(rle["size"]) == [20, 30]
    assert isinstance(rle["counts"], (bytes, bytearray))


def test_torch_mask_to_coco_rle_roundtrip_via_numpy_decode() -> None:
    mask = _make_rectangle_mask(50, 40, (10, 5, 30, 35))
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(50, 40), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    assert decoded.shape == (1, 50, 40)
    np.testing.assert_array_equal(decoded[0], mask.numpy())


def test_torch_mask_to_coco_rle_all_zeros_mask() -> None:
    mask = torch.zeros((16, 16), dtype=torch.bool)
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(16, 16), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    assert decoded.shape == (1, 16, 16)
    assert not decoded.any()


def test_torch_mask_to_coco_rle_all_ones_mask() -> None:
    mask = torch.ones((16, 24), dtype=torch.bool)
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(16, 24), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    assert decoded.shape == (1, 16, 24)
    assert decoded.all()


def test_torch_mask_to_coco_rle_single_pixel_mask() -> None:
    mask = torch.zeros((10, 10), dtype=torch.bool)
    mask[3, 7] = True
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(10, 10), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    np.testing.assert_array_equal(decoded[0], mask.numpy())


def test_torch_mask_to_coco_rle_non_square_mask() -> None:
    mask = _make_rectangle_mask(5, 100, (0, 10, 5, 90))
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(5, 100), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    np.testing.assert_array_equal(decoded[0], mask.numpy())


def test_torch_mask_to_coco_rle_accepts_non_bool_dtype() -> None:
    # Integer masks should encode the same as their boolean equivalent.
    mask = torch.zeros((12, 12), dtype=torch.int32)
    mask[2:8, 4:10] = 1
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(12, 12), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    np.testing.assert_array_equal(decoded[0], mask.bool().numpy())


def test_torch_mask_to_coco_rle_does_not_mutate_input() -> None:
    mask = _make_rectangle_mask(20, 20, (5, 5, 15, 15))
    original = mask.clone()
    _ = torch_mask_to_coco_rle(mask)
    assert torch.equal(mask, original)


def test_torch_mask_to_coco_rle_handles_non_contiguous_tensor() -> None:
    # Build a larger tensor and take a non-contiguous slice via transpose.
    big = _make_rectangle_mask(32, 32, (4, 4, 20, 24))
    non_contig = big.t()  # shape (32, 32), not C-contiguous
    assert not non_contig.is_contiguous()
    rle = torch_mask_to_coco_rle(non_contig)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(32, 32), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    np.testing.assert_array_equal(decoded[0], non_contig.numpy())


def test_torch_mask_to_coco_rle_handles_tensor_with_grad() -> None:
    # Function calls .detach() internally; must not raise on requires_grad.
    mask = torch.zeros((8, 8), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        mask[2:6, 2:6] = 1.0
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(8, 8), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    np.testing.assert_array_equal(decoded[0], (mask > 0).detach().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_torch_mask_to_coco_rle_accepts_cuda_tensor() -> None:
    mask = _make_rectangle_mask(20, 20, (5, 5, 15, 15)).cuda()
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(20, 20), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    np.testing.assert_array_equal(decoded[0], mask.cpu().numpy())


def test_coco_rle_masks_to_numpy_mask_empty_returns_correct_shape() -> None:
    empty = InstancesRLEMasks(image_size=(480, 640), masks=[])
    out = coco_rle_masks_to_numpy_mask(empty)
    assert out.shape == (0, 480, 640)
    assert out.dtype == bool


def test_coco_rle_masks_to_numpy_mask_empty_for_zero_size_image() -> None:
    empty = InstancesRLEMasks(image_size=(0, 0), masks=[])
    out = coco_rle_masks_to_numpy_mask(empty)
    assert out.shape == (0, 0, 0)
    assert out.dtype == bool


def test_coco_rle_masks_to_numpy_mask_single_mask_decodes_correctly() -> None:
    mask = _make_rectangle_mask(40, 60, (10, 15, 25, 45))
    wrapped = _rle_from_tensors([mask])
    out = coco_rle_masks_to_numpy_mask(wrapped)
    assert out.shape == (1, 40, 60)
    assert out.dtype == bool
    np.testing.assert_array_equal(out[0], mask.numpy())


def test_coco_rle_masks_to_numpy_mask_multiple_masks_preserve_order() -> None:
    m0 = _make_rectangle_mask(30, 30, (0, 0, 10, 10))
    m1 = _make_rectangle_mask(30, 30, (10, 10, 20, 20))
    m2 = _make_rectangle_mask(30, 30, (20, 20, 30, 30))
    wrapped = _rle_from_tensors([m0, m1, m2])
    out = coco_rle_masks_to_numpy_mask(wrapped)
    assert out.shape == (3, 30, 30)
    np.testing.assert_array_equal(out[0], m0.numpy())
    np.testing.assert_array_equal(out[1], m1.numpy())
    np.testing.assert_array_equal(out[2], m2.numpy())


def test_coco_rle_masks_to_numpy_mask_overlapping_masks() -> None:
    m0 = _make_rectangle_mask(20, 20, (0, 0, 15, 15))
    m1 = _make_rectangle_mask(20, 20, (5, 5, 20, 20))
    wrapped = _rle_from_tensors([m0, m1])
    out = coco_rle_masks_to_numpy_mask(wrapped)
    # Overlap region should be True in both
    np.testing.assert_array_equal(out[0] & out[1], (m0 & m1).numpy())


def test_coco_rle_masks_to_numpy_mask_non_square_image_size() -> None:
    m = _make_rectangle_mask(5, 100, (1, 10, 4, 90))
    wrapped = _rle_from_tensors([m])
    out = coco_rle_masks_to_numpy_mask(wrapped)
    assert out.shape == (1, 5, 100)
    np.testing.assert_array_equal(out[0], m.numpy())


def test_coco_rle_masks_to_numpy_mask_returns_bool_dtype() -> None:
    mask = _make_rectangle_mask(16, 16, (4, 4, 12, 12))
    wrapped = _rle_from_tensors([mask])
    out = coco_rle_masks_to_numpy_mask(wrapped)
    assert out.dtype == bool


# ---------------------------------------------------------------------------
# coco_rle_masks_to_torch_mask
# ---------------------------------------------------------------------------


def test_coco_rle_masks_to_torch_mask_empty_returns_empty_tensor() -> None:
    empty = InstancesRLEMasks(image_size=(100, 200), masks=[])
    out = coco_rle_masks_to_torch_mask(empty)
    assert out.shape == (0, 100, 200)
    assert out.dtype == torch.bool
    assert out.device.type == "cpu"


def test_coco_rle_masks_to_torch_mask_empty_with_explicit_device() -> None:
    empty = InstancesRLEMasks(image_size=(50, 50), masks=[])
    out = coco_rle_masks_to_torch_mask(empty, device=torch.device("cpu"))
    assert out.shape == (0, 50, 50)
    assert out.dtype == torch.bool
    assert out.device.type == "cpu"


def test_coco_rle_masks_to_torch_mask_single_mask_matches_input() -> None:
    mask = _make_rectangle_mask(40, 30, (5, 5, 30, 25))
    wrapped = _rle_from_tensors([mask])
    out = coco_rle_masks_to_torch_mask(wrapped)
    assert out.shape == (1, 40, 30)
    assert out.dtype == torch.bool
    assert torch.equal(out[0], mask)


def test_coco_rle_masks_to_torch_mask_multiple_masks_preserve_order() -> None:
    masks = [
        _make_rectangle_mask(25, 25, (0, 0, 10, 10)),
        _make_rectangle_mask(25, 25, (15, 15, 25, 25)),
    ]
    wrapped = _rle_from_tensors(masks)
    out = coco_rle_masks_to_torch_mask(wrapped)
    for i, m in enumerate(masks):
        assert torch.equal(out[i], m)


def test_coco_rle_masks_to_torch_mask_default_device_is_cpu() -> None:
    mask = _make_rectangle_mask(10, 10, (2, 2, 8, 8))
    wrapped = _rle_from_tensors([mask])
    out = coco_rle_masks_to_torch_mask(wrapped)
    assert out.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coco_rle_masks_to_torch_mask_explicit_cuda_device() -> None:
    mask = _make_rectangle_mask(20, 20, (5, 5, 15, 15))
    wrapped = _rle_from_tensors([mask])
    out = coco_rle_masks_to_torch_mask(wrapped, device=torch.device("cuda"))
    assert out.device.type == "cuda"
    assert torch.equal(out[0].cpu(), mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coco_rle_masks_to_torch_mask_empty_on_cuda() -> None:
    empty = InstancesRLEMasks(image_size=(64, 64), masks=[])
    out = coco_rle_masks_to_torch_mask(empty, device=torch.device("cuda"))
    assert out.shape == (0, 64, 64)
    assert out.device.type == "cuda"


def test_coco_rle_masks_to_torch_mask_returns_bool_dtype() -> None:
    mask = _make_rectangle_mask(16, 16, (4, 4, 12, 12))
    wrapped = _rle_from_tensors([mask])
    out = coco_rle_masks_to_torch_mask(wrapped)
    assert out.dtype == torch.bool


def test_coco_rle_masks_to_torch_mask_numpy_and_torch_decoders_agree() -> None:
    masks = [
        _make_rectangle_mask(32, 48, (0, 0, 16, 24)),
        _make_rectangle_mask(32, 48, (16, 24, 32, 48)),
        torch.zeros((32, 48), dtype=torch.bool),  # all-zeros
        torch.ones((32, 48), dtype=torch.bool),  # all-ones
    ]
    wrapped = _rle_from_tensors(masks)
    np_out = coco_rle_masks_to_numpy_mask(wrapped)
    torch_out = coco_rle_masks_to_torch_mask(wrapped)
    np.testing.assert_array_equal(np_out, torch_out.numpy())


@pytest.mark.parametrize("h,w", [(1, 1), (16, 16), (7, 13), (200, 50)])
def test_roundtrip_torch_to_rle_to_numpy(h: int, w: int) -> None:
    rng = np.random.default_rng(seed=42)
    mask = torch.from_numpy(rng.integers(0, 2, size=(h, w)).astype(bool))
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(h, w), masks=[rle])
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    np.testing.assert_array_equal(decoded[0], mask.numpy())


@pytest.mark.parametrize("h,w", [(1, 1), (16, 16), (7, 13), (200, 50)])
def test_roundtrip_torch_to_rle_to_torch(h: int, w: int) -> None:
    rng = np.random.default_rng(seed=123)
    mask = torch.from_numpy(rng.integers(0, 2, size=(h, w)).astype(bool))
    rle = torch_mask_to_coco_rle(mask)
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(h, w), masks=[rle])
    decoded = coco_rle_masks_to_torch_mask(wrapped)
    assert torch.equal(decoded[0], mask)


def test_roundtrip_many_masks_preserves_all() -> None:
    rng = np.random.default_rng(seed=7)
    h, w, n = 40, 60, 8
    masks = [
        torch.from_numpy(rng.integers(0, 2, size=(h, w)).astype(bool)) for _ in range(n)
    ]
    rles = [torch_mask_to_coco_rle(m) for m in masks]
    wrapped = InstancesRLEMasks.from_coco_rle_masks(image_size=(h, w), masks=rles)
    decoded = coco_rle_masks_to_torch_mask(wrapped)
    assert decoded.shape == (n, h, w)
    for i, m in enumerate(masks):
        assert torch.equal(decoded[i], m)

"""Regression lock for the batched instance-segmentation RLE path.

The instance-seg models (RF-DETR + YOLOv5/7/8/26 + YOLACT) used to encode masks
one detection at a time via ``align_instance_segmentation_results_to_rle_masks``,
which calls ``torch_mask_to_coco_rle`` per mask -- and that does a ``.cpu()``
sync per detection. On Jetson those per-detection syncs serialize the GPU N times
per frame and dominate seg post-processing.

The replacement uses the batched ``align_instance_segmentation_results`` followed
by ``torch_masks_to_coco_rle_batch`` (a single device->host transfer + one
vectorized ``pycocotools.encode``). These tests lock that the new path is
*output-identical* to the old one: same boxes, byte-identical RLE.
"""

from unittest import mock

import numpy as np
import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
    torch_masks_to_coco_rle_batch,
)
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    align_instance_segmentation_results_to_rle_masks,
    align_instance_segmentation_results_to_rle_masks_batched,
)

# ---------------------------------------------------------------------------
# torch_masks_to_coco_rle_batch is a drop-in for per-mask torch_mask_to_coco_rle
# ---------------------------------------------------------------------------


def test_batch_encode_empty_returns_empty_list() -> None:
    assert (
        torch_masks_to_coco_rle_batch(torch.zeros((0, 16, 16), dtype=torch.bool)) == []
    )


@pytest.mark.parametrize("n,h,w", [(1, 32, 32), (3, 20, 40), (8, 17, 13), (32, 64, 48)])
def test_batch_encode_byte_identical_to_per_mask(n: int, h: int, w: int) -> None:
    rng = np.random.default_rng(seed=n * 100 + h)
    masks = torch.from_numpy(rng.integers(0, 2, size=(n, h, w)).astype(bool))
    batch = torch_masks_to_coco_rle_batch(masks)
    per_mask = [torch_mask_to_coco_rle(m) for m in masks]
    assert len(batch) == len(per_mask) == n
    for got, want in zip(batch, per_mask):
        assert got["counts"] == want["counts"]
        assert list(got["size"]) == list(want["size"])


@pytest.mark.parametrize(
    "builder",
    [
        lambda: torch.zeros((3, 24, 24), dtype=torch.bool),  # all background
        lambda: torch.ones((3, 24, 24), dtype=torch.bool),  # all foreground
    ],
)
def test_batch_encode_degenerate_masks_match_per_mask(builder) -> None:
    masks = builder()
    for got, want in zip(
        torch_masks_to_coco_rle_batch(masks),
        [torch_mask_to_coco_rle(m) for m in masks],
    ):
        assert got["counts"] == want["counts"]


def test_batch_encode_roundtrip_matches_input() -> None:
    rng = np.random.default_rng(seed=2024)
    n, h, w = 6, 50, 70
    masks = torch.from_numpy(rng.integers(0, 2, size=(n, h, w)).astype(bool))
    wrapped = InstancesRLEMasks.from_coco_rle_masks(
        image_size=(h, w), masks=torch_masks_to_coco_rle_batch(masks)
    )
    decoded = coco_rle_masks_to_numpy_mask(wrapped)
    assert decoded.shape == (n, h, w)
    np.testing.assert_array_equal(decoded, masks.numpy())


def test_batch_encode_does_not_mutate_input() -> None:
    rng = np.random.default_rng(seed=5)
    masks = torch.from_numpy(rng.integers(0, 2, size=(4, 20, 20)).astype(bool))
    original = masks.clone()
    _ = torch_masks_to_coco_rle_batch(masks)
    assert torch.equal(masks, original)


@pytest.mark.parametrize(
    "slicer",
    [
        lambda m: m[:, ::2, ::2],  # strided in both spatial dims
        lambda m: m[::2, 1:, :-3],  # strided batch dim + offset/truncated views
        lambda m: m.transpose(1, 2),  # permuted strides
    ],
)
def test_batch_encode_non_contiguous_input_matches_per_mask(slicer) -> None:
    rng = np.random.default_rng(seed=17)
    base = torch.from_numpy(rng.integers(0, 2, size=(6, 41, 37)).astype(bool))
    masks = slicer(base)
    assert not masks.is_contiguous()
    batch = torch_masks_to_coco_rle_batch(masks)
    per_mask = [torch_mask_to_coco_rle(m) for m in masks]
    assert len(batch) == len(per_mask) == masks.shape[0]
    for got, want in zip(batch, per_mask):
        assert got["counts"] == want["counts"]
        assert list(got["size"]) == list(want["size"])
    wrapped = InstancesRLEMasks.from_coco_rle_masks(
        image_size=(masks.shape[1], masks.shape[2]), masks=batch
    )
    np.testing.assert_array_equal(coco_rle_masks_to_numpy_mask(wrapped), masks.numpy())


def test_batch_encode_accepts_tensor_requiring_grad() -> None:
    # A float mask still attached to an autograd graph must encode instead of
    # raising in .numpy(). The uint8 cast already drops the graph today; the
    # explicit .detach() keeps that true if the conversion chain ever changes.
    rng = np.random.default_rng(seed=23)
    logits = torch.from_numpy(rng.standard_normal(size=(3, 18, 22)).astype(np.float32))
    logits.requires_grad_(True)
    masks = logits * 1.0  # attached to the autograd graph
    assert masks.requires_grad
    batch = torch_masks_to_coco_rle_batch(masks)
    expected = torch_masks_to_coco_rle_batch(masks.detach())
    assert [r["counts"] for r in batch] == [r["counts"] for r in expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batch_encode_cuda_matches_cpu() -> None:
    rng = np.random.default_rng(seed=11)
    masks = torch.from_numpy(rng.integers(0, 2, size=(5, 40, 40)).astype(bool))
    for cpu, cuda in zip(
        torch_masks_to_coco_rle_batch(masks),
        torch_masks_to_coco_rle_batch(masks.cuda()),
    ):
        assert cpu["counts"] == cuda["counts"]


# ---------------------------------------------------------------------------
# batched align + batch encode == per-detection generator (the model refactor)
# ---------------------------------------------------------------------------


def _boxes(n: int, w: int, h: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    b = torch.zeros(n, 6)
    b[:, 0] = torch.rand(n, generator=g) * w * 0.4
    b[:, 1] = torch.rand(n, generator=g) * h * 0.4
    b[:, 2] = b[:, 0] + torch.rand(n, generator=g) * w * 0.5 + 2
    b[:, 3] = b[:, 1] + torch.rand(n, generator=g) * h * 0.5 + 2
    b[:, 4] = torch.rand(n, generator=g)
    b[:, 5] = torch.randint(0, 5, (n,), generator=g).float()
    return b


@pytest.mark.parametrize(
    "n,mh,mw,ow,oh,sw,sh,iw,ih,pad,scl,cx,cy,seed",
    [
        (4, 64, 64, 100, 80, 100, 80, 64, 64, (4, 4, 4, 4), 1.0, 0, 0, 1),
        (5, 64, 64, 120, 90, 120, 90, 64, 64, (6, 6, 6, 6), 1.25, 0, 0, 2),
        (
            3,
            64,
            64,
            120,
            100,
            80,
            64,
            64,
            64,
            (4, 4, 4, 4),
            1.0,
            20,
            18,
            3,
        ),  # static crop
        (3, 48, 48, 99, 77, 99, 77, 48, 48, (5, 3, 5, 3), 1.0, 0, 0, 4),  # odd dims
        (1, 32, 32, 64, 64, 64, 64, 32, 32, (2, 2, 2, 2), 1.0, 0, 0, 5),
        (0, 64, 64, 100, 80, 100, 80, 64, 64, (4, 4, 4, 4), 1.0, 0, 0, 6),  # empty
    ],
)
def test_batched_align_encode_matches_generator(
    n, mh, mw, ow, oh, sw, sh, iw, ih, pad, scl, cx, cy, seed
) -> None:
    boxes = _boxes(n, ow, oh, seed)
    masks = (
        torch.rand(n, mh, mw, generator=torch.Generator().manual_seed(seed + 7)) - 0.5
    )
    kw = dict(
        padding=pad,
        scale_width=scl,
        scale_height=scl,
        original_size=ImageDimensions(width=ow, height=oh),
        size_after_pre_processing=ImageDimensions(width=sw, height=sh),
        inference_size=ImageDimensions(width=iw, height=ih),
        static_crop_offset=StaticCropOffset(
            offset_x=cx, offset_y=cy, crop_width=ow, crop_height=oh
        ),
    )

    gen_boxes, gen_rles = [], []
    for bb, mk in align_instance_segmentation_results_to_rle_masks(
        image_bboxes=boxes.clone(), masks=masks.clone(), **kw
    ):
        gen_boxes.append(bb)
        gen_rles.append(mk)
    gen_boxes_t = torch.stack(gen_boxes) if gen_boxes else torch.empty((0, 6))

    bat_boxes, bat_masks = align_instance_segmentation_results(
        image_bboxes=boxes.clone(), masks=masks.clone(), **kw
    )
    bat_rles = torch_masks_to_coco_rle_batch(bat_masks)

    assert gen_boxes_t.shape == bat_boxes.shape
    assert torch.equal(gen_boxes_t, bat_boxes)
    assert len(gen_rles) == len(bat_rles) == n
    for got, want in zip(bat_rles, gen_rles):
        assert got["counts"] == want["counts"]
        assert list(got["size"]) == list(want["size"])


@pytest.mark.parametrize("chunk", [1, 2, 3, 16])
@pytest.mark.parametrize(
    "n,ow,oh,sw,sh,pad,scl,cx,cy,seed",
    [
        (7, 100, 80, 100, 80, (4, 4, 4, 4), 1.0, 0, 0, 11),
        (5, 120, 100, 80, 64, (4, 4, 4, 4), 1.0, 20, 18, 12),  # static crop
        (4, 99, 77, 99, 77, (5, 3, 5, 3), 1.25, 0, 0, 13),  # odd dims + scale
        (0, 100, 80, 100, 80, (4, 4, 4, 4), 1.0, 0, 0, 14),  # empty
    ],
)
def test_chunked_align_encode_matches_generator(
    chunk, n, ow, oh, sw, sh, pad, scl, cx, cy, seed
) -> None:
    mh = mw = 64
    boxes = _boxes(n, ow, oh, seed)
    masks = (
        torch.rand(n, mh, mw, generator=torch.Generator().manual_seed(seed + 7)) - 0.5
    )
    kw = dict(
        padding=pad,
        scale_width=scl,
        scale_height=scl,
        original_size=ImageDimensions(width=ow, height=oh),
        size_after_pre_processing=ImageDimensions(width=sw, height=sh),
        inference_size=ImageDimensions(width=64, height=64),
        static_crop_offset=StaticCropOffset(
            offset_x=cx, offset_y=cy, crop_width=ow, crop_height=oh
        ),
    )
    gen = list(
        align_instance_segmentation_results_to_rle_masks(
            image_bboxes=boxes.clone(), masks=masks.clone(), **kw
        )
    )
    gen_boxes = torch.stack([b for b, _ in gen]) if gen else torch.empty((0, 6))

    got_boxes, got_rles = align_instance_segmentation_results_to_rle_masks_batched(
        image_bboxes=boxes.clone(), masks=masks.clone(), mask_chunk_size=chunk, **kw
    )

    assert got_boxes.shape == gen_boxes.shape
    assert torch.equal(got_boxes, gen_boxes)
    assert len(got_rles) == len(gen) == n
    for got, (_, want) in zip(got_rles, gen):
        assert got["counts"] == want["counts"]
        assert list(got["size"]) == list(want["size"])


def test_chunked_align_encode_bounds_full_resolution_masks_per_call() -> None:
    # The RLE path must never materialize all N full-resolution masks at once:
    # each align call (and each host transfer) sees at most `chunk` masks.
    from inference_models.models.common.roboflow import post_processing

    n, chunk = 10, 3
    boxes = _boxes(n, 100, 80, seed=21)
    masks = torch.rand(n, 64, 64, generator=torch.Generator().manual_seed(22)) - 0.5
    real_align = post_processing.align_instance_segmentation_results
    seen = []

    def spy(**kwargs):
        out_boxes, out_masks = real_align(**kwargs)
        seen.append(out_masks.shape[0])
        return out_boxes, out_masks

    with mock.patch.object(
        post_processing, "align_instance_segmentation_results", side_effect=spy
    ):
        _, rles = align_instance_segmentation_results_to_rle_masks_batched(
            image_bboxes=boxes,
            masks=masks,
            padding=(4, 4, 4, 4),
            scale_width=1.0,
            scale_height=1.0,
            original_size=ImageDimensions(width=100, height=80),
            size_after_pre_processing=ImageDimensions(width=100, height=80),
            inference_size=ImageDimensions(width=64, height=64),
            static_crop_offset=StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=100, crop_height=80
            ),
            mask_chunk_size=chunk,
        )
    assert seen == [3, 3, 3, 1]
    assert len(rles) == n

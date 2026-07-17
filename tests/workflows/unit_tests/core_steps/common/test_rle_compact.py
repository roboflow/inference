"""Tests for the zero-densification InstancesRLEMasks -> CompactMask transcode.

The parity contract: `compact_mask_from_coco_rle` must produce a CompactMask
that decodes identically to
`CompactMask.from_dense(coco_rle_masks_to_numpy_mask(rle), xyxy, image_shape)`,
without ever building the dense (N, H, W) stack.
"""

import numpy as np
import pytest
import torch

pytest.importorskip(
    "supervision.detection.compact_mask",
    reason="supervision build without CompactMask (needs the compact-masks release)",
)

from supervision.detection.compact_mask import CompactMask  # noqa: E402

from inference.core.workflows.core_steps.common.rle_compact import (  # noqa: E402
    _decode_coco_counts,
    compact_mask_from_coco_rle,
    instances_rle_to_compact_mask,
)
from inference_models.models.base.types import InstancesRLEMasks  # noqa: E402
from inference_models.models.common.rle_utils import (  # noqa: E402
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)


def _encode(masks: np.ndarray) -> InstancesRLEMasks:
    """(N, H, W) bool -> InstancesRLEMasks using the same encoder inference uses."""
    h, w = masks.shape[1], masks.shape[2]
    counts = [
        torch_mask_to_coco_rle(torch.from_numpy(masks[i]))["counts"]
        for i in range(masks.shape[0])
    ]
    return InstancesRLEMasks(image_size=(h, w), masks=counts)


def _tight_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def test_coco_counts_decoder_inverts_pycocotools():
    rng = np.random.default_rng(0)
    for _ in range(100):
        h = int(rng.integers(1, 40))
        w = int(rng.integers(1, 40))
        mask = (rng.random((h, w)) < rng.random()).astype(np.uint8)
        compressed = torch_mask_to_coco_rle(torch.from_numpy(mask))["counts"]

        decoded = _decode_coco_counts(compressed)

        flat = np.zeros(h * w, dtype=bool)
        pos = 0
        for idx, run_len in enumerate(decoded):
            if idx % 2 == 1:
                flat[pos : pos + run_len] = True
            pos += run_len
        assert pos == h * w
        # COCO is column-major (F-order): reshape (w, h) then transpose.
        np.testing.assert_array_equal(flat.reshape(w, h).T, mask.astype(bool))


@pytest.mark.parametrize("seed", range(25))
def test_parity_with_from_dense(seed: int):
    rng = np.random.default_rng(seed)
    h = int(rng.integers(8, 64))
    w = int(rng.integers(8, 64))
    n = int(rng.integers(0, 6))

    masks = np.zeros((n, h, w), dtype=bool)
    xyxy = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        kind = rng.random()
        if kind < 0.15:
            pass  # all-False
        elif kind < 0.30:
            masks[i, :, :] = True  # all-True
        else:
            y1, y2 = sorted(rng.integers(0, h, size=2))
            x1, x2 = sorted(rng.integers(0, w, size=2))
            masks[i, y1 : y2 + 1, x1 : x2 + 1] = (
                rng.random((y2 - y1 + 1, x2 - x1 + 1)) < 0.6
            )
        box = _tight_box(masks[i])
        # Exercise clipping: degenerate and out-of-bounds boxes.
        jitter = rng.random()
        if jitter < 0.2:
            box = np.array([box[0], box[1], box[0] - 1, box[1] - 1], dtype=np.float32)
        elif jitter < 0.35:
            box = box + np.array([-3, -3, 5, 5], dtype=np.float32)
        xyxy[i] = box

    rle = _encode(masks) if n else InstancesRLEMasks((h, w), [])
    dense = coco_rle_masks_to_numpy_mask(rle) if n else np.zeros((0, h, w), dtype=bool)

    reference = CompactMask.from_dense(dense, xyxy, (h, w))
    candidate = compact_mask_from_coco_rle((h, w), rle.masks, xyxy)

    np.testing.assert_array_equal(candidate.to_dense(), reference.to_dense())
    for i in range(n):
        assert candidate[i].shape == (h, w)
        np.testing.assert_array_equal(candidate[i], reference[i])


def test_adapter_matches_full_frame_decode():
    rng = np.random.default_rng(123)
    h, w, n = 80, 100, 4
    masks = np.zeros((n, h, w), dtype=bool)
    xyxy = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        x1 = int(rng.integers(0, w - 30))
        y1 = int(rng.integers(0, h - 30))
        masks[i, y1 : y1 + 20, x1 : x1 + 25] = True
        xyxy[i] = _tight_box(masks[i])

    rle = _encode(masks)
    compact = instances_rle_to_compact_mask(rle, xyxy)

    assert isinstance(compact, CompactMask)
    # The compact form decodes to exactly the full-frame dense masks ...
    np.testing.assert_array_equal(compact.to_dense(), masks)
    np.testing.assert_array_equal(compact.to_dense(), coco_rle_masks_to_numpy_mask(rle))
    # ... while storing only crop-area pixels, never the dense stack.
    crop_px = int(np.prod(compact._crop_shapes, axis=1).sum())
    assert crop_px < n * h * w


def test_empty_masks():
    compact = compact_mask_from_coco_rle(
        (50, 50), [], np.empty((0, 4), dtype=np.float32)
    )
    assert isinstance(compact, CompactMask)
    assert compact.to_dense().shape == (0, 50, 50)

"""Parity tests: ChunkedPostProcessImage vs the stock sam3 PostProcessImage.

Requires the real `sam3` package (CUDA extras); skipped in the stubbed
unit-test environment via importorskip. Runs on CPU tensors with
`always_interpolate_masks_on_gpu=False`.
"""

import numpy as np
import pytest
import torch

sam3 = pytest.importorskip("sam3")

from sam3.eval.postprocessors import PostProcessImage  # noqa: E402

from inference_models.models.sam3.chunked_postprocessing import (  # noqa: E402
    ChunkedPostProcessImage,
)

B, Q, LOW = 1, 12, 16
TARGET_H, TARGET_W = 64, 48


def _synthetic_outputs(seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    return {
        "pred_boxes": torch.rand(B, Q, 4, generator=g) * 0.5 + 0.25,
        "pred_logits": torch.randn(B, Q, 1, generator=g),
        "pred_masks": torch.randn(B, Q, LOW, LOW, generator=g) * 4,
    }


def _sizes():
    return torch.tensor([[TARGET_H, TARGET_W]])


def _build(cls, **overrides):
    kwargs = dict(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=True,
        always_interpolate_masks_on_gpu=False,
        use_presence=False,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


def _rle_counts(entry):
    counts = entry["counts"]
    return counts.decode() if isinstance(counts, bytes) else counts


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_dense_masks_match_parent_across_chunk_sizes(chunk):
    outputs = _synthetic_outputs()
    parent = _build(PostProcessImage)(outputs, _sizes(), _sizes())
    ours = _build(ChunkedPostProcessImage, mask_chunk_size=chunk)(
        outputs, _sizes(), _sizes()
    )
    assert torch.equal(parent[0]["scores"], ours[0]["scores"])
    assert torch.equal(parent[0]["boxes"], ours[0]["boxes"])
    assert torch.equal(parent[0]["masks"], ours[0]["masks"])


def test_rle_masks_match_parent():
    outputs = _synthetic_outputs()
    parent = _build(PostProcessImage, convert_mask_to_rle=True)(
        outputs, _sizes(), _sizes()
    )
    ours = _build(ChunkedPostProcessImage, convert_mask_to_rle=True, mask_chunk_size=3)(
        outputs, _sizes(), _sizes()
    )
    p_rles, o_rles = parent[0]["masks_rle"], ours[0]["masks_rle"]
    assert len(p_rles) == len(o_rles)
    for p, o in zip(p_rles, o_rles):
        assert list(p["size"]) == list(o["size"])
        assert _rle_counts(p) == _rle_counts(o)


def test_cap_returns_topk_by_score_in_parent_late_topk_order():
    outputs = _synthetic_outputs()
    parent = _build(PostProcessImage)(outputs, _sizes(), _sizes())
    ours = _build(ChunkedPostProcessImage, max_dets_per_img=3)(
        outputs, _sizes(), _sizes()
    )
    expected_scores, expected_idx = torch.topk(parent[0]["scores"], 3)
    assert torch.equal(ours[0]["scores"], expected_scores)
    assert torch.equal(ours[0]["masks"], parent[0]["masks"][expected_idx])
    assert torch.equal(ours[0]["boxes"], parent[0]["boxes"][expected_idx])


def test_cap_applies_with_thresholding_disabled():
    # regression: the cap must bound the pipeline even when
    # detection_threshold <= 0 (xreview #2670 consensus finding)
    outputs = _synthetic_outputs()
    ours = _build(
        ChunkedPostProcessImage, detection_threshold=-1.0, max_dets_per_img=4
    )(outputs, _sizes(), _sizes())
    assert ours[0]["scores"].shape[0] == 4
    assert ours[0]["masks"].shape[0] == 4
    # score-descending, and equal to the global top-4 over all queries
    all_scores = _build(PostProcessImage, detection_threshold=-1.0)(
        outputs, _sizes(), _sizes()
    )[0]["scores"]
    assert torch.equal(ours[0]["scores"], torch.topk(all_scores, 4).values)


def test_zero_detections_shapes():
    outputs = _synthetic_outputs()
    dense = _build(ChunkedPostProcessImage, detection_threshold=0.999999)(
        outputs, _sizes(), _sizes()
    )
    assert dense[0]["masks"].shape == (0, 1, TARGET_H, TARGET_W)
    assert dense[0]["masks"].dtype == torch.bool
    rle = _build(
        ChunkedPostProcessImage,
        detection_threshold=0.999999,
        convert_mask_to_rle=True,
    )(outputs, _sizes(), _sizes())
    assert rle[0]["masks_rle"] == []


def test_uncapped_default_is_bitwise_parent_parity():
    # the serving default (max_detections=-1, dense): selection, ordering,
    # scores and mask pixels must be byte-identical to the stock parent
    for seed in (0, 1, 2):
        outputs = _synthetic_outputs(seed)
        parent = _build(PostProcessImage)(outputs, _sizes(), _sizes())
        ours = _build(ChunkedPostProcessImage)(outputs, _sizes(), _sizes())
        for key in ("scores", "boxes", "labels", "masks"):
            assert torch.equal(parent[0][key], ours[0][key]), (seed, key)

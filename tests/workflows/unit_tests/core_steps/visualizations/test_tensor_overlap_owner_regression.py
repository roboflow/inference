"""Regression tests for the tensor painters' overlapping-box ownership
resolution.

Background: on CUDA the previous formulation (`torch.empty` owner buffer +
`scatter_reduce_(amax, include_self=False)`) could surface an out-of-range
owner under heavily duplicated scatter destinations, which turned into an
asynchronous device assert in the winner-color gather and killed video job
processes with SIGABRT (no Python traceback). The resolver now lives in
`resolve_overlap_winners` and must keep every gathered owner provably within
`[0, num_candidates)` for ANY duplication pattern.

Every test here is deterministic (seeded geometry, no time/randomness at run
time) and parametrized over cpu + cuda; the cuda variants are the actual
regression tests for the L40S abort and must be run on a GPU host, e.g.::

    CUDA_LAUNCH_BLOCKING=1 python -m pytest \
        tests/workflows/unit_tests/core_steps/visualizations/test_tensor_overlap_owner_regression.py -v
"""

import numpy as np
import pytest
import torch

from inference.core.workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    gpu_draw_boxes,
)
from inference.core.workflows.core_steps.visualizations.common import base_tensor
from inference.core.workflows.core_steps.visualizations.common.base_tensor import (
    resolve_overlap_winners,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

SCENE_H, SCENE_W = 240, 320


def _distinct_colors(n: int) -> np.ndarray:
    """n distinct RGB colors, none black (the test background)."""
    idx = np.arange(1, n + 1, dtype=np.int64)
    return np.stack(
        [(idx & 0xFF), ((idx >> 8) & 0xFF) + 1, np.full(n, 200)], axis=1
    ).astype(np.uint8)


def _sequential_band_reference(
    xyxy: np.ndarray, colors: np.ndarray, thickness: int, h: int, w: int
) -> np.ndarray:
    """Paint the exact band geometry of gpu_draw_boxes (roundness=0) box by
    box in index order — the ground truth for later-box-wins ownership."""
    scene = np.zeros((h, w, 3), np.uint8)
    outer = thickness // 2
    t = thickness
    for (a, b, c, d), color in zip(xyxy.astype(int), colors):
        bx1, bx2 = min(a, c), max(a, c)
        by1, by2 = min(b, d), max(b, d)
        x1, x2, y1, y2 = bx1 - outer, bx2 + outer, by1 - outer, by2 + outer
        for r1, r2, c1, c2 in (
            (y1, y1 + t - 1, x1, x2),  # top band
            (y2 - t + 1, y2, x1, x2),  # bottom band
            (by1, by2, x1, x1 + t - 1),  # left band
            (by1, by2, x2 - t + 1, x2),  # right band
        ):
            r1, c1 = max(r1, 0), max(c1, 0)
            r2, c2 = min(r2, h - 1), min(c2, w - 1)
            if r2 >= r1 and c2 >= c1:
                scene[r1 : r2 + 1, c1 : c2 + 1] = color
    return scene


def _dense_overlap_boxes() -> np.ndarray:
    """Deterministic worst-case geometry: 96 boxes crammed into one region so
    every border band contests pixels with many others, 16 of them EXACTLY
    identical (maximal duplicated scatter destinations), plus clipped,
    degenerate, inverted, and out-of-frame boxes."""
    rng = np.random.default_rng(20240817)
    clustered = np.stack(
        [
            rng.integers(40, 120, 64),
            rng.integers(30, 110, 64),
            rng.integers(130, 260, 64),
            rng.integers(120, 220, 64),
        ],
        axis=1,
    ).astype(float)
    identical = np.tile(np.array([[60.0, 50.0, 200.0, 180.0]]), (16, 1))
    edge_cases = np.array(
        [
            [-30.0, -20.0, 80.0, 70.0],  # clipped top-left
            [250.0, 180.0, 380.0, 300.0],  # clipped bottom-right
            [90.0, 90.0, 90.0, 90.0],  # fully degenerate point
            [100.0, 60.0, 100.0, 160.0],  # zero-width line
            [220.0, 140.0, 140.0, 40.0],  # inverted x and y
            [-100.0, -100.0, -40.0, -40.0],  # fully off-frame (top-left)
            [SCENE_W + 5.0, 10.0, SCENE_W + 60.0, 90.0],  # fully off-frame
            [10.6, 20.4, 199.9, 201.2],  # sub-pixel floats
        ]
    )
    jitter = clustered.copy()
    jitter[:, [0, 1]] += rng.integers(0, 3, (64, 2))  # near-duplicates
    return np.concatenate([clustered, identical, edge_cases, jitter[:8]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("thickness", [1, 2, 3, 5])
def test_dense_overlap_matches_sequential_paint_exactly(
    device: str, thickness: int
) -> None:
    xyxy = _dense_overlap_boxes()
    colors = _distinct_colors(xyxy.shape[0])
    scene = torch.zeros((3, SCENE_H, SCENE_W), dtype=torch.uint8, device=device)
    annotated = gpu_draw_boxes(scene, xyxy.astype(int), colors, thickness)
    if device == "cuda":
        torch.cuda.synchronize()
    got = annotated.permute(1, 2, 0).cpu().numpy()
    expected = _sequential_band_reference(xyxy, colors, thickness, SCENE_H, SCENE_W)
    mismatched = int((got != expected).any(axis=2).sum())
    assert mismatched == 0, (
        f"{mismatched} pixels disagree with sequential later-box-wins paint "
        f"(thickness={thickness}, device={device})"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_dense_overlap_repeated_iterations_are_stable(device: str) -> None:
    # The historical failure was intermittent (more likely under load /
    # repetition): repeat the worst-case frame many times and require every
    # iteration to be finite, correct, and identical to the first.
    xyxy = _dense_overlap_boxes()
    colors = _distinct_colors(xyxy.shape[0])
    expected = _sequential_band_reference(xyxy, colors, 2, SCENE_H, SCENE_W)
    iterations = 100 if device == "cuda" else 25
    first = None
    for i in range(iterations):
        scene = torch.zeros((3, SCENE_H, SCENE_W), dtype=torch.uint8, device=device)
        annotated = gpu_draw_boxes(scene, xyxy.astype(int), colors, 2)
        if device == "cuda":
            torch.cuda.synchronize()  # surface async device asserts HERE
        got = annotated.permute(1, 2, 0).cpu().numpy()
        assert np.array_equal(got, expected), f"iteration {i} diverged"
        if first is None:
            first = got
        else:
            assert np.array_equal(got, first), f"iteration {i} nondeterministic"


@pytest.mark.parametrize("device", DEVICES)
def test_resolver_all_pixels_to_one_cell(device: str) -> None:
    # Maximal duplication: every scattered pixel targets the same cell.
    total = 4096
    flat = torch.zeros(total, dtype=torch.int64, device=device)
    priority = torch.arange(total, dtype=torch.int32, device=device)
    winners = resolve_overlap_winners(
        flat, priority, num_cells=SCENE_H * SCENE_W, num_candidates=total
    )
    if device == "cuda":
        torch.cuda.synchronize()
    assert winners.dtype == torch.int64
    assert bool((winners == total - 1).all().item())


@pytest.mark.parametrize("device", DEVICES)
def test_resolver_owners_always_in_range_and_deterministic(device: str) -> None:
    # Adversarial duplication pattern: heavy collisions in shuffled order,
    # repeated — every gathered owner must equal the per-cell amax and stay
    # in [0, n) on every repeat.
    rng = np.random.default_rng(7)
    n = 64
    pixels = 20_000
    cells = SCENE_H * SCENE_W
    flat_np = rng.integers(0, 500, pixels)  # 500 cells, ~40 collisions each
    priority_np = rng.integers(0, n, pixels)
    expected_np = np.full(500, -1, dtype=np.int64)
    np.maximum.at(expected_np, flat_np, priority_np)
    flat = torch.tensor(flat_np, dtype=torch.int64, device=device)
    priority = torch.tensor(priority_np, dtype=torch.int32, device=device)
    reference = None
    for repeat in range(50):
        winners = resolve_overlap_winners(
            flat, priority, num_cells=cells, num_candidates=n
        )
        if device == "cuda":
            torch.cuda.synchronize()
        winners_np = winners.cpu().numpy()
        assert winners_np.min() >= 0 and winners_np.max() < n
        assert np.array_equal(
            winners_np, expected_np[flat_np]
        ), f"repeat {repeat}: winners disagree with per-cell amax"
        if reference is None:
            reference = winners_np
        else:
            assert np.array_equal(winners_np, reference)


def test_resolver_strict_mode_rejects_out_of_range_scatter_index(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        base_tensor, "WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS", True
    )
    flat = torch.tensor([0, 5, 10_000], dtype=torch.int64)
    priority = torch.tensor([0, 1, 2], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="out-of-range pixel indices"):
        resolve_overlap_winners(flat, priority, num_cells=100, num_candidates=3)


def test_resolver_strict_mode_passes_valid_input(monkeypatch) -> None:
    monkeypatch.setattr(
        base_tensor, "WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS", True
    )
    flat = torch.tensor([3, 3, 3, 7], dtype=torch.int64)
    priority = torch.tensor([0, 2, 1, 3], dtype=torch.int32)
    winners = resolve_overlap_winners(flat, priority, num_cells=10, num_candidates=4)
    assert winners.tolist() == [2, 2, 2, 3]


def test_winner_validation_names_the_failure() -> None:
    with pytest.raises(RuntimeError, match="out-of-range owners"):
        base_tensor._validate_winners(
            torch.tensor([0, 1, -2147483648], dtype=torch.int64), 3
        )


@pytest.mark.parametrize("device", DEVICES)
def test_off_frame_and_degenerate_only_boxes_do_not_scatter_out_of_range(
    device: str,
) -> None:
    # Geometry whose bands clip to nothing must never build an out-of-range
    # flat index (this is what the host-side geometry check protects).
    xyxy = np.array(
        [
            [-100, -100, -40, -40],
            [SCENE_W + 5, SCENE_H + 5, SCENE_W + 60, SCENE_H + 60],
            [50, 50, 50, 50],
            [-10, 30, SCENE_W + 10, 35],  # horizontal strip crossing frame
        ]
    )
    colors = _distinct_colors(xyxy.shape[0])
    scene = torch.zeros((3, SCENE_H, SCENE_W), dtype=torch.uint8, device=device)
    annotated = gpu_draw_boxes(scene, xyxy, colors, 3)
    if device == "cuda":
        torch.cuda.synchronize()
    expected = _sequential_band_reference(xyxy, colors, 3, SCENE_H, SCENE_W)
    assert np.array_equal(annotated.permute(1, 2, 0).cpu().numpy(), expected)

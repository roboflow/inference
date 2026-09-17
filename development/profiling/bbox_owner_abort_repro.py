"""CUDA repro/validation harness for the tensor bbox painter SIGABRT.

Context: video job processes on L40S (NVDEC + tensor-native workflows +
`roboflow_core/bounding_box_visualization@v1`) died with exit -6 (SIGABRT)
and no traceback; the suspect was the overlap branch of `gpu_draw_boxes` —
`torch.empty` owner + `scatter_reduce_(amax, include_self=False)` gathering a
possibly-invalid owner into the color table.

This harness runs three phases ON A CUDA HOST:

  A. legacy-pattern probe: the retired empty+include_self=False formulation
     in a tight loop with range checks — reproduces the invalid owner on any
     torch build where the theory holds (checked WITHOUT indexing, so it
     reports instead of aborting).
  B. painter stress: the current gpu_draw_boxes on deterministic worst-case
     geometry (dense overlap, duplicates, clipped/degenerate/inverted boxes),
     verified against an exact host-side sequential-paint reference.
  C. multi-process stress: N child processes (default 8, mirroring the c8
     failure profile) each hammering phase B on the same GPU; children that
     die with a signal are reported like the supervisor saw it.

Recommended invocation on the L40S box (from the repo root, GPU venv):

    CUDA_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1 \
        python development/profiling/bbox_owner_abort_repro.py --iterations 2000 --children 8

CUDA_LAUNCH_BLOCKING makes any device assert surface synchronously at the
launching op (identifying the failing kernel), and faulthandler prints the
Python stack on SIGABRT. TORCH_USE_CUDA_DSA additionally enables device-side
assertions on torch builds compiled with DSA support (stock wheels are not;
harmless to set).
"""

import argparse
import faulthandler
import multiprocessing
import sys

import numpy as np
import torch

faulthandler.enable()

SCENE_H, SCENE_W = 1080, 1920  # production-like frame


def _dense_overlap_boxes() -> np.ndarray:
    rng = np.random.default_rng(20240817)
    clustered = np.stack(
        [
            rng.integers(200, 700, 96),
            rng.integers(150, 500, 96),
            rng.integers(750, 1500, 96),
            rng.integers(550, 1000, 96),
        ],
        axis=1,
    ).astype(float)
    identical = np.tile(np.array([[300.0, 250.0, 1200.0, 800.0]]), (32, 1))
    edge_cases = np.array(
        [
            [-100.0, -80.0, 400.0, 350.0],
            [1700.0, 900.0, 2200.0, 1400.0],
            [500.0, 500.0, 500.0, 500.0],
            [600.0, 300.0, 600.0, 900.0],
            [1400.0, 800.0, 800.0, 200.0],
            [-500.0, -500.0, -200.0, -200.0],
        ]
    )
    return np.concatenate([clustered, identical, edge_cases])


def _distinct_colors(n: int) -> np.ndarray:
    idx = np.arange(1, n + 1, dtype=np.int64)
    return np.stack(
        [(idx & 0xFF), ((idx >> 8) & 0xFF) + 1, np.full(n, 200)], axis=1
    ).astype(np.uint8)


def _sequential_band_reference(xyxy, colors, thickness, h, w) -> np.ndarray:
    scene = np.zeros((h, w, 3), np.uint8)
    outer = thickness // 2
    t = thickness
    for (a, b, c, d), color in zip(xyxy.astype(int), colors):
        bx1, bx2 = min(a, c), max(a, c)
        by1, by2 = min(b, d), max(b, d)
        x1, x2, y1, y2 = bx1 - outer, bx2 + outer, by1 - outer, by2 + outer
        for r1, r2, c1, c2 in (
            (y1, y1 + t - 1, x1, x2),
            (y2 - t + 1, y2, x1, x2),
            (by1, by2, x1, x1 + t - 1),
            (by1, by2, x2 - t + 1, x2),
        ):
            r1, c1 = max(r1, 0), max(c1, 0)
            r2, c2 = min(r2, h - 1), min(c2, w - 1)
            if r2 >= r1 and c2 >= c1:
                scene[r1 : r2 + 1, c1 : c2 + 1] = color
    return scene


def phase_a_legacy_pattern_probe(iterations: int, device: str) -> int:
    """Tight loop over the RETIRED formulation with the owner range checked
    on host BEFORE any gather-by-owner, so an invalid value is reported
    instead of tripping a device assert."""
    print(f"[A] legacy empty+include_self=False probe, {iterations} iters")
    rng = np.random.default_rng(1)
    anomalies = 0
    cells = SCENE_H * SCENE_W
    for i in range(iterations):
        pixels = 200_000
        flat_np = rng.integers(0, 50_000, pixels)  # heavy duplication
        box_np = rng.integers(0, 128, pixels)
        flat = torch.tensor(flat_np, dtype=torch.int64, device=device)
        pixel_box = torch.tensor(box_np, dtype=torch.int32, device=device)
        owner = torch.empty(cells, dtype=torch.int32, device=device)
        owner.scatter_reduce_(0, flat, pixel_box, reduce="amax", include_self=False)
        gathered = owner[flat]
        torch.cuda.synchronize() if device == "cuda" else None
        lo = int(gathered.min().item())
        hi = int(gathered.max().item())
        if lo < 0 or hi >= 128:
            anomalies += 1
            print(
                f"[A]  ANOMALY iter {i}: gathered owner range [{lo}, {hi}] "
                f"outside [0, 128) — theory REPRODUCED on this torch build"
            )
    print(f"[A] done: {anomalies} anomalies in {iterations} iterations")
    return anomalies


def phase_b_painter_stress(iterations: int, device: str) -> int:
    from inference.core.workflows.core_steps.visualizations.bounding_box.v1_tensor import (
        gpu_draw_boxes,
    )

    print(f"[B] gpu_draw_boxes stress, {iterations} iters on {device}")
    xyxy = _dense_overlap_boxes()
    colors = _distinct_colors(xyxy.shape[0])
    expected = _sequential_band_reference(xyxy, colors, 2, SCENE_H, SCENE_W)
    failures = 0
    for i in range(iterations):
        scene = torch.zeros((3, SCENE_H, SCENE_W), dtype=torch.uint8, device=device)
        annotated = gpu_draw_boxes(scene, xyxy.astype(int), colors, 2)
        if device == "cuda":
            torch.cuda.synchronize()
        if i % 50 == 0 or i == iterations - 1:
            got = annotated.permute(1, 2, 0).cpu().numpy()
            if not np.array_equal(got, expected):
                failures += 1
                bad = int((got != expected).any(axis=2).sum())
                print(f"[B]  iter {i}: {bad} mismatched pixels")
    print(f"[B] done: {failures} verification failures")
    return failures


def _child_worker(iterations: int) -> None:
    faulthandler.enable()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = phase_a_legacy_pattern_probe(iterations // 4, device)
    failures += phase_b_painter_stress(iterations, device)
    sys.exit(1 if failures else 0)


def phase_c_multiprocess(iterations: int, children: int) -> int:
    print(f"[C] {children} child processes x {iterations} iters (c{children} profile)")
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=_child_worker, args=(iterations,), name=f"job-{i}")
        for i in range(children)
    ]
    for p in procs:
        p.start()
    bad = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            bad += 1
            kind = (
                f"signal {-p.exitcode} (SIGABRT)"
                if p.exitcode == -6
                else f"exit {p.exitcode}"
            )
            print(f"[C]  {p.name} FAILED: {kind}")
    print(f"[C] done: {bad}/{children} children failed")
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--children", type=int, default=8)
    parser.add_argument(
        "--skip-multiprocess", action="store_true", help="phases A+B only"
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}, device={device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    failures = phase_a_legacy_pattern_probe(args.iterations, device)
    failures += phase_b_painter_stress(args.iterations, device)
    if not args.skip_multiprocess:
        failures += phase_c_multiprocess(args.iterations, args.children)
    print("RESULT:", "FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

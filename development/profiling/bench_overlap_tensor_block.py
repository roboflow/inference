"""Benchmark for the tensor-pipeline overlap_analysis block.

Scenarios: bbox-only detections at several sizes, and instance-segmentation
detections with dense GPU masks (the per-detection mask D2H hot path).
Prints BENCH_JSON on the last line.
"""

import argparse
import json
import statistics
import time
from uuid import uuid4

import numpy as np
import torch

from inference.core.workflows.core_steps.fusion.overlap_analysis.v1_tensor import (
    OverlapAnalysisBlockV1,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

W, H = 1280, 720


def make_boxes(n: int, seed: int) -> np.ndarray:
    """Clustered boxes so a decent fraction of cross-set pairs overlap."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform([100, 100], [W - 100, H - 100], size=(max(n, 1), 2))
    sizes = rng.uniform(40, 160, size=(max(n, 1), 2))
    xyxy = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)
    return xyxy[:n].astype(np.float32)


def make_detections(n: int, seed: int, device: str, with_masks: bool):
    xyxy_np = make_boxes(n, seed)
    xyxy = torch.as_tensor(xyxy_np, device=device)
    class_id = torch.zeros(n, dtype=torch.int64, device=device)
    confidence = torch.full((n,), 0.9, device=device)
    image_metadata = {"class_names": {0: "object"}}
    bboxes_metadata = [{"detection_id": str(uuid4())} for _ in range(n)]
    if not with_masks:
        return Detections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    mask = torch.zeros((n, H, W), dtype=torch.bool, device=device)
    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in xyxy_np[i]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W), min(y2, H)
        # ellipse-ish blob inside the box so contours are non-trivial
        yy, xx = torch.meshgrid(
            torch.arange(y1, y2, device=device),
            torch.arange(x1, x2, device=device),
            indexing="ij",
        )
        cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
        ry, rx = max((y2 - y1) / 2, 1), max((x2 - x1) / 2, 1)
        blob = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        mask[i, y1:y2, x1:x2] = blob
    return InstanceDetections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def timed(fn, iters, warmup, sync):
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p90_ms": round(statistics.quantiles(samples, n=10)[8], 4),
        "mean_ms": round(statistics.fmean(samples), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sync = args.device == "cuda"
    block = OverlapAnalysisBlockV1()
    report = {"device": args.device, "torch": torch.__version__, "iters": args.iters, "e2e": {}}

    cases = [
        ("bbox_n0", 0, 0, False),
        ("bbox_n10", 10, 10, False),
        ("bbox_n50", 50, 50, False),
        ("bbox_n100", 100, 100, False),
        ("masks_n10", 10, 10, True),
        ("masks_n20", 20, 20, True),
    ]
    for name, n_ref, n_cand, with_masks in cases:
        ref = make_detections(n_ref, seed=1, device=args.device, with_masks=with_masks)
        cand = make_detections(n_cand, seed=2, device=args.device, with_masks=with_masks)
        # sanity: count pairs once
        out = block.run(
            reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
        )
        n_pairs = len(out["overlaps"])
        stats = timed(
            lambda: block.run(
                reference_predictions=ref,
                candidate_predictions=cand,
                min_overlap=0.1,
            ),
            args.iters if not with_masks else max(args.iters // 4, 20),
            args.warmup,
            sync,
        )
        stats["pairs_emitted"] = n_pairs
        report["e2e"][name] = stats
        print(f"[e2e {name}] {json.dumps(stats)}", flush=True)
    print("BENCH_JSON:" + json.dumps(report))


if __name__ == "__main__":
    main()

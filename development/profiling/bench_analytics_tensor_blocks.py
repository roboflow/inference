"""Stage-level benchmark for tensor-pipeline line_counter / time_in_zone blocks.

Runs on a Jetson inside a roboflow-inference-server container. Measures:
  1. end-to-end block.run() latency (line_counter@v2 tensor, time_in_zone@v3 tensor)
  2. stage breakdown of the shared hot pattern (D2H, sv ctor, trigger, take_by_mask)

Usage: python bench_analytics_tensor_blocks.py [--iters 300] [--device cuda]
Prints a JSON report to stdout (last line) for easy scraping.
"""

import argparse
import json
import statistics
import time
from datetime import datetime

import numpy as np
import supervision as sv
import torch

from inference.core.workflows.core_steps.analytics.line_counter.v2_tensor import (
    LineCounterBlockV2,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v3_tensor import (
    TimeInZoneBlockV3,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    take_prediction_by_mask,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections

W, H = 1280, 720
LINE = [[W // 2, 0], [W // 2, H]]  # vertical line at x=640
ZONE = [[100, 100], [100, 600], [600, 600], [600, 100]]  # left-side square


def make_frames(n_boxes: int, n_frames: int, device: str, crossing: bool):
    """Boxes drift horizontally; if crossing, they sweep across the line/zone,
    else they stay on the left side (no crossings, out of zone edge cases mixed)."""
    rng = np.random.default_rng(42)
    base_y = rng.uniform(50, H - 150, size=n_boxes)
    box_w = rng.uniform(30, 90, size=n_boxes)
    box_h = rng.uniform(30, 90, size=n_boxes)
    phase = rng.uniform(0, 2 * np.pi, size=n_boxes)
    frames = []
    for f in range(n_frames):
        t = f / max(n_frames - 1, 1)
        if crossing:
            cx = 200 + (W - 400) * (0.5 + 0.5 * np.sin(2 * np.pi * t + phase))
        else:
            cx = 150 + 100 * np.sin(2 * np.pi * t + phase)
        x1 = cx - box_w / 2
        y1 = base_y
        xyxy = np.stack([x1, y1, x1 + box_w, y1 + box_h], axis=1).astype(np.float32)
        frames.append(xyxy)
    tensors = [torch.as_tensor(x, device=device) for x in frames]
    if device == "cuda":
        torch.cuda.synchronize()
    return tensors


def make_detections(xyxy_t: torch.Tensor, device: str) -> Detections:
    n = xyxy_t.shape[0]
    return Detections(
        xyxy=xyxy_t,
        class_id=torch.zeros(n, dtype=torch.int64, device=device),
        confidence=torch.full((n,), 0.9, device=device),
        image_metadata={},
        bboxes_metadata=[{"tracker_id": i + 1} for i in range(n)],
    )


def make_image(frame_number: int, vid: str) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="bench"),
        numpy_image=_SCENE,
        video_metadata=VideoMetadata(
            video_identifier=vid,
            frame_number=frame_number,
            frame_timestamp=datetime.fromtimestamp(1_700_000_000 + frame_number / 30),
            fps=30,
            comes_from_video_file=True,
        ),
    )


_SCENE = np.zeros((H, W, 3), dtype=np.uint8)


def timed_loop(fn, iters, warmup, sync):
    for _ in range(warmup):
        fn(0)
    if sync:
        torch.cuda.synchronize()
    samples = []
    for i in range(iters):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(i)
        if sync:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p90_ms": round(statistics.quantiles(samples, n=10)[8], 4),
        "mean_ms": round(statistics.fmean(samples), 4),
    }


def bench_block_e2e(device, iters, warmup, n_boxes, crossing, sync):
    frames = make_frames(n_boxes, iters + warmup, device, crossing)
    dets = [make_detections(x, device) for x in frames]
    images = [make_image(i, f"lc-{n_boxes}-{crossing}") for i in range(iters + warmup)]

    lc = LineCounterBlockV2()
    idx = {"i": 0}

    def run_lc(_):
        i = idx["i"]
        idx["i"] += 1
        lc.run(
            detections=dets[i % len(dets)],
            image=images[i % len(images)],
            line_segment=LINE,
            triggering_anchor="CENTER",
        )

    lc_stats = timed_loop(run_lc, iters, warmup, sync)

    tz = TimeInZoneBlockV3()
    idx2 = {"i": 0}
    images2 = [make_image(i, f"tz-{n_boxes}-{crossing}") for i in range(iters + warmup)]

    def run_tz(_):
        i = idx2["i"]
        idx2["i"] += 1
        tz.run(
            image=images2[i % len(images2)],
            detections=dets[i % len(dets)],
            zone=ZONE,
            triggering_anchor="CENTER",
            remove_out_of_zone_detections=False,
            reset_out_of_zone_detections=True,
        )

    tz_stats = timed_loop(run_tz, iters, warmup, sync)
    return {"line_counter_v2": lc_stats, "time_in_zone_v3": tz_stats}


def bench_stages(device, iters, warmup, n_boxes, sync):
    frames = make_frames(n_boxes, iters + warmup, device, crossing=True)
    dets = [make_detections(x, device) for x in frames]
    line_zone = sv.LineZone(
        start=sv.Point(*LINE[0]),
        end=sv.Point(*LINE[1]),
        triggering_anchors=[sv.Position("CENTER")],
    )
    polygon_zone = sv.PolygonZone(
        polygon=np.array(ZONE), triggering_anchors=(sv.Position("CENTER"),)
    )
    n = n_boxes
    tracker_np = np.arange(1, n + 1, dtype=int)

    stages = {}

    def s_d2h(i):
        dets[i % len(dets)].xyxy.detach().to("cpu").numpy().astype(float)

    stages["d2h_astype"] = timed_loop(s_d2h, iters, warmup, sync)

    def s_tracker_extract(i):
        d = dets[i % len(dets)]
        bm = d.bboxes_metadata or [{} for _ in range(n)]
        tids = [m.get("tracker_id") for m in bm]
        any(t is None for t in tids)
        np.array([int(t) for t in tids], dtype=int)

    stages["tracker_extract"] = timed_loop(s_tracker_extract, iters, warmup, sync)

    host_xyxy = [d.xyxy.detach().to("cpu").numpy().astype(float) for d in dets]

    def s_sv_ctor(i):
        sv.Detections(xyxy=host_xyxy[i % len(host_xyxy)], tracker_id=tracker_np)

    stages["sv_detections_ctor"] = timed_loop(s_sv_ctor, iters, warmup, sync)

    sv_dets = [
        sv.Detections(xyxy=x, tracker_id=tracker_np.copy()) for x in host_xyxy
    ]

    def s_line_trigger(i):
        line_zone.trigger(detections=sv_dets[i % len(sv_dets)])

    stages["line_trigger"] = timed_loop(s_line_trigger, iters, warmup, sync)

    def s_poly_trigger(i):
        polygon_zone.trigger(sv_dets[i % len(sv_dets)])

    stages["polygon_trigger"] = timed_loop(s_poly_trigger, iters, warmup, sync)

    empty_mask = np.zeros(n, dtype=bool)
    half_mask = np.zeros(n, dtype=bool)
    half_mask[: n // 2] = True
    full_mask = np.ones(n, dtype=bool)
    for name, mask in [("empty", empty_mask), ("half", half_mask), ("full", full_mask)]:

        def s_take(i, m=mask):
            take_prediction_by_mask(dets[i % len(dets)], m)

        stages[f"take_by_mask_{name}"] = timed_loop(s_take, iters, warmup, sync)

    return stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sync = args.device == "cuda"

    report = {
        "device": args.device,
        "torch": torch.__version__,
        "supervision": sv.__version__,
        "iters": args.iters,
        "e2e": {},
        "stages": {},
    }
    for n_boxes in (0, 5, 20, 100):
        for crossing in ((False, True) if n_boxes else (False,)):
            key = f"n{n_boxes}_{'crossing' if crossing else 'static'}"
            report["e2e"][key] = bench_block_e2e(
                args.device, args.iters, args.warmup, n_boxes, crossing, sync
            )
            print(f"[e2e {key}] {json.dumps(report['e2e'][key])}", flush=True)
    for n_boxes in (5, 20, 100):
        report["stages"][f"n{n_boxes}"] = bench_stages(
            args.device, args.iters, args.warmup, n_boxes, sync
        )
        print(f"[stages n{n_boxes}] done", flush=True)
    print("BENCH_JSON:" + json.dumps(report))


if __name__ == "__main__":
    main()

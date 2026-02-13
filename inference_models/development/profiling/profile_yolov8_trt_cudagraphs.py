import os
import time

import numpy as np
import torch
from tqdm import tqdm

from inference_models import AutoModel

DEVICE = os.environ.get("DEVICE", "cuda:0")
CYCLES = int(os.environ.get("CYCLES", "10_000"))
WARMUP = int(os.environ.get("WARMUP", "50"))
RECAPTURE_CYCLES = int(os.environ.get("RECAPTURE_CYCLES", "100"))

BATCH_SIZES = [1, 2, 3]


def main() -> None:

    model = AutoModel.from_pretrained(
        model_id_or_path="yolov8n-640",
        device=torch.device(DEVICE),
        backend="trt",
        batch_size=(1, max(BATCH_SIZES)),
    )

    image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    pre_processed_single, _ = model.pre_process(image)

    batches = {
        bs: pre_processed_single.repeat(bs, 1, 1, 1) for bs in BATCH_SIZES
    }

    # ── Warmup ──────────────────────────────────────────────────────────
    for _ in range(WARMUP):
        for batch in batches.values():
            model.forward(batch, use_cuda_graph=False)
            model.forward(batch, use_cuda_graph=True)

    bs_label = "/".join(str(bs) for bs in BATCH_SIZES)

    # ── (1) Cycling batch sizes, no CUDA graphs ─────────────────────────
    print(f"Timing without CUDA graphs, cycling bs={bs_label}...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(CYCLES):
        batch = batches[BATCH_SIZES[i % len(BATCH_SIZES)]]
        model.forward(batch, use_cuda_graph=False)
    torch.cuda.synchronize()
    baseline_fps = CYCLES / (time.perf_counter() - start)

    # ── (2) Cycling batch sizes, CUDA graphs with forced recapture ──────
    print(
        f"Timing with CUDA graph recapture every iteration, cycling bs={bs_label} "
        f"({RECAPTURE_CYCLES} iters)..."
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(RECAPTURE_CYCLES):
        model._trt_cuda_graph_cache = None
        batch = batches[BATCH_SIZES[i % len(BATCH_SIZES)]]
        model.forward(batch, use_cuda_graph=True)
    torch.cuda.synchronize()
    recapture_fps = RECAPTURE_CYCLES / (time.perf_counter() - start)

    # ── (3) Cycling batch sizes, CUDA graphs with normal caching ────────
    model._trt_cuda_graph_cache = None
    for batch in batches.values():
        model.forward(batch, use_cuda_graph=True)

    print(f"Timing with CUDA graph cache replay, cycling bs={bs_label}...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(CYCLES):
        batch = batches[BATCH_SIZES[i % len(BATCH_SIZES)]]
        model.forward(batch, use_cuda_graph=True)
    torch.cuda.synchronize()
    replay_fps = CYCLES / (time.perf_counter() - start)

    # ── Results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  yolov8n-640 TRT — cycling batch sizes {BATCH_SIZES}")
    print(f"  {CYCLES} iterations (recapture: {RECAPTURE_CYCLES})")
    print(f"{'='*60}")
    print(f"  No CUDA graphs:          {baseline_fps:>8.1f} fwd/s")
    print(f"  CUDA graph recapture:    {recapture_fps:>8.1f} fwd/s  ({recapture_fps / baseline_fps:.2f}x)")
    print(f"  CUDA graph replay:       {replay_fps:>8.1f} fwd/s  ({replay_fps / baseline_fps:.2f}x)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

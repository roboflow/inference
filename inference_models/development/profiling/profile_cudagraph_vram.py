"""Profile VRAM usage as the number of cached CUDA graphs grows.

Loads yolov8n-640 as a TRT model with dynamic batch size, then runs forward
passes with varying batch sizes (in shuffled order) to force new graph captures.

Measures VRAM two ways after each capture:
  - "Tensor bytes": directly summed from input_buffer + output_buffers in the cache.
  - "Driver bytes": total GPU memory used, via torch.cuda.mem_get_info() which
    queries the NVIDIA driver. This captures opaque allocations (TRT execution
    contexts, CUDA graph objects, streams, internal workspace) that are invisible
    to PyTorch's allocator.

The difference (driver - tensor - baseline) isolates the opaque overhead.

Example invocation:
    python profile_cudagraph_vram.py --device cuda:0 --max-batch-size 32

    python profile_cudagraph_vram.py --device cuda:0 --max-batch-size 16 --output vram.png
"""

import argparse
import gc
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from inference_models import AutoModel
from inference_models.models.common.trt import TRTCudaGraphLRUCache, TRTCudaGraphState

MODEL_ID = "yolov8n-640"


def graph_state_tensor_bytes(state: TRTCudaGraphState) -> int:
    total = state.input_buffer.nbytes
    for buf in state.output_buffers:
        total += buf.nbytes
    return total


def cache_total_tensor_bytes(cache: TRTCudaGraphLRUCache) -> int:
    total = 0
    for state in cache.cache.values():
        total += graph_state_tensor_bytes(state)
    return total


def driver_used_bytes(device: torch.device) -> int:
    free, total = torch.cuda.mem_get_info(device)
    return total - free


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile VRAM usage vs. number of cached CUDA graphs (varying batch size).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Largest batch size to test. Each batch size from 1..max creates a new graph.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot image. Defaults to 'vram_yolov8n-640.png'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = AutoModel.from_pretrained(
        model_id_or_path=MODEL_ID,
        device=device,
        backend="trt",
        batch_size=(1, args.max_batch_size),
        cuda_graph_cache_capacity=args.max_batch_size + 10,
    )

    image = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    single_preprocessed, _ = model.pre_process(image)

    model.forward(single_preprocessed, use_cuda_graph=False)
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()

    baseline_driver_bytes = driver_used_bytes(device)

    model._trt_cuda_graph_cache = TRTCudaGraphLRUCache(
        capacity=args.max_batch_size + 10,
    )

    batch_size_order = list(range(1, args.max_batch_size + 1))
    random.Random(42).shuffle(batch_size_order)

    batch_sizes = []
    cumulative_tensor_mb = []
    cumulative_driver_mb = []
    per_graph_tensor_mb = []
    per_graph_driver_mb = []

    prev_tensor_bytes = 0
    prev_driver_bytes = baseline_driver_bytes
    for i, bs in enumerate(batch_size_order):
        batched = single_preprocessed.expand(bs, -1, -1, -1).contiguous()
        model.forward(batched, use_cuda_graph=True)
        torch.cuda.synchronize(device)

        tensor_bytes = cache_total_tensor_bytes(model._trt_cuda_graph_cache)
        drv_bytes = driver_used_bytes(device)

        tensor_delta = tensor_bytes - prev_tensor_bytes
        driver_delta = drv_bytes - prev_driver_bytes

        batch_sizes.append(bs)
        cumulative_tensor_mb.append(tensor_bytes / (1024 ** 2))
        cumulative_driver_mb.append((drv_bytes - baseline_driver_bytes) / (1024 ** 2))
        per_graph_tensor_mb.append(tensor_delta / (1024 ** 2))
        per_graph_driver_mb.append(driver_delta / (1024 ** 2))

        prev_tensor_bytes = tensor_bytes
        prev_driver_bytes = drv_bytes
        print(
            f"[{i + 1}/{args.max_batch_size}] "
            f"bs={bs:>2d} | "
            f"tensors: {tensor_bytes / (1024 ** 2):>7.1f} MB (+{tensor_delta / (1024 ** 2):>6.1f}) | "
            f"driver:  {(drv_bytes - baseline_driver_bytes) / (1024 ** 2):>7.1f} MB (+{driver_delta / (1024 ** 2):>6.1f})"
        )

    output_path = Path(args.output) if args.output else Path(f"vram_{MODEL_ID}.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"CUDA Graph Cache VRAM (varying batch size) — {MODEL_ID}",
        fontsize=14,
    )

    capture_order = list(range(1, len(batch_sizes) + 1))
    bar_width = 0.35

    ax_cum = axes[0]
    x_cum = np.arange(len(capture_order))
    ax_cum.bar(
        x_cum - bar_width / 2, cumulative_driver_mb, bar_width,
        color="steelblue", label="Driver-level (total GPU)",
    )
    ax_cum.bar(
        x_cum + bar_width / 2, cumulative_tensor_mb, bar_width,
        color="darkorange", label="Cache tensors only",
    )
    ax_cum.set_ylabel("Cumulative VRAM above baseline (MB)")
    ax_cum.set_xlabel("Number of Cached Graphs (capture order)")
    ax_cum.set_xticks(x_cum)
    ax_cum.set_xticklabels(
        [f"{n}\n(bs={bs})" for n, bs in zip(capture_order, batch_sizes)],
        fontsize=7,
    )
    ax_cum.legend()

    sorted_indices = sorted(range(len(batch_sizes)), key=lambda k: batch_sizes[k])
    sorted_bs = [batch_sizes[k] for k in sorted_indices]
    sorted_driver = [per_graph_driver_mb[k] for k in sorted_indices]
    sorted_tensor = [per_graph_tensor_mb[k] for k in sorted_indices]

    ax_pg = axes[1]
    x_pg = np.arange(len(sorted_bs))
    ax_pg.bar(
        x_pg - bar_width / 2, sorted_driver, bar_width,
        color="steelblue", label="Driver-level (total GPU)",
    )
    ax_pg.bar(
        x_pg + bar_width / 2, sorted_tensor, bar_width,
        color="darkorange", label="Cache tensors only",
    )
    ax_pg.set_ylabel("Per-Graph VRAM (MB)")
    ax_pg.set_xlabel("Batch Size")
    ax_pg.set_xticks(x_pg)
    ax_pg.set_xticklabels([str(bs) for bs in sorted_bs])
    ax_pg.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")

    total_tensor = prev_tensor_bytes / (1024 ** 2)
    total_driver = (prev_driver_bytes - baseline_driver_bytes) / (1024 ** 2)
    n = len(batch_sizes)
    print(f"\nAfter {n} graphs:")
    print(f"  Cache tensor VRAM:  {total_tensor:.1f} MB (avg {total_tensor / n:.1f} MB/graph)")
    print(f"  Driver-level VRAM:  {total_driver:.1f} MB (avg {total_driver / n:.1f} MB/graph)")
    print(f"  Opaque overhead:    {total_driver - total_tensor:.1f} MB (avg {(total_driver - total_tensor) / n:.1f} MB/graph)")


if __name__ == "__main__":
    main()

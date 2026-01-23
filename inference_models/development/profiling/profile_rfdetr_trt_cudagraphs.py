import os
import time

import cv2
import numpy as np
import torch

from inference_models import AutoModel

IMAGE_PATH = os.environ.get("IMAGE_PATH", None)
DEVICE = os.environ.get("DEVICE", "cuda:0")
CYCLES = int(os.environ.get("CYCLES", "10_000"))
WARMUP = int(os.environ.get("WARMUP", "50"))


def main() -> None:

    model = AutoModel.from_pretrained(
        model_id_or_path="rfdetr-nano", device=torch.device(DEVICE), backend="trt"
    )

    if IMAGE_PATH is not None:
        image = cv2.imread(IMAGE_PATH)
    else:
        image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)

    pre_processed, _ = model.pre_process(image)

    for _ in range(WARMUP):
        model.forward(pre_processed, use_cuda_graph=False)
        model.forward(pre_processed, use_cuda_graph=True)

    print("Timing without CUDA graphs...")
    start = time.perf_counter()
    for _ in range(CYCLES):
        model.forward(pre_processed, use_cuda_graph=False)
    baseline_fps = CYCLES / (time.perf_counter() - start)

    print("Timing with CUDA graphs...")
    start = time.perf_counter()
    for _ in range(CYCLES):
        model.forward(pre_processed, use_cuda_graph=True)
    cudagraph_fps = CYCLES / (time.perf_counter() - start)

    print(f"\n{'='*50}")
    print(f"Forward pass FPS (no CUDA graphs): {baseline_fps:.1f}")
    print(f"Forward pass FPS (CUDA graphs):    {cudagraph_fps:.1f}")
    print(f"Speedup: {cudagraph_fps / baseline_fps:.2f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

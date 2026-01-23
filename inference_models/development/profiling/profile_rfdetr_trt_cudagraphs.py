import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from inference_models import AutoModel

IMAGE_PATH_WARMUP = os.environ.get("IMAGE_PATH_WARMUP", None)
IMAGE_PATH_PROFILING = os.environ.get("IMAGE_PATH_PROFILING", None)
DEVICE = os.environ.get("DEVICE", "cuda:0")
CYCLES = int(os.environ.get("CYCLES", "10_000"))
WARMUP = int(os.environ.get("WARMUP", "50"))


def main() -> None:

    model = AutoModel.from_pretrained(
        model_id_or_path="rfdetr-nano", device=torch.device(DEVICE), backend="trt"
    )

    if IMAGE_PATH_WARMUP is not None:
        image_warmup = cv2.imread(IMAGE_PATH_WARMUP)
    else:
        image_warmup = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)

    if IMAGE_PATH_PROFILING is not None:
        image_profiling = cv2.imread(IMAGE_PATH_PROFILING)
    else:
        image_profiling = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)

    pre_processed_warmup, _ = model.pre_process(image_warmup)
    pre_processed_profiling, _ = model.pre_process(image_profiling)

    for _ in range(WARMUP):
        model.forward(pre_processed_warmup, use_cuda_graph=False)
        model.forward(pre_processed_warmup, use_cuda_graph=True)

    expected_output_warmup_image = model.forward(
        pre_processed_warmup, use_cuda_graph=False
    )
    expected_output_profiling_image = model.forward(
        pre_processed_profiling, use_cuda_graph=False
    )

    cudagraph_output_warmup_image = model.forward(
        pre_processed_warmup, use_cuda_graph=True
    )
    cudagraph_output_profiling_image = model.forward(
        pre_processed_profiling, use_cuda_graph=True
    )

    assert torch.allclose(
        expected_output_warmup_image[0], cudagraph_output_warmup_image[0], atol=1e-6
    )
    assert torch.allclose(
        expected_output_profiling_image[0],
        cudagraph_output_profiling_image[0],
        atol=1e-6,
    )
    assert torch.allclose(
        expected_output_warmup_image[1], cudagraph_output_warmup_image[1], atol=1e-6
    )
    assert torch.allclose(
        expected_output_profiling_image[1],
        cudagraph_output_profiling_image[1],
        atol=1e-6,
    )

    start = time.perf_counter()
    for _ in tqdm(range(CYCLES), desc="Without CUDA graphs"):
        model.forward(pre_processed_profiling, use_cuda_graph=False)
    baseline_fps = CYCLES / (time.perf_counter() - start)

    start = time.perf_counter()
    for _ in tqdm(range(CYCLES), desc="With CUDA graphs"):
        model.forward(pre_processed_profiling, use_cuda_graph=True)
    cudagraph_fps = CYCLES / (time.perf_counter() - start)

    print(f"\n{'='*50}")
    print(f"Forward pass FPS (no CUDA graphs): {baseline_fps:.1f}")
    print(f"Forward pass FPS (CUDA graphs):    {cudagraph_fps:.1f}")
    print(f"Speedup: {cudagraph_fps / baseline_fps:.2f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

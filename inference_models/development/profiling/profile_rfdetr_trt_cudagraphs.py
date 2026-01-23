import os
import time

import cv2
import torch
from tqdm import tqdm

from inference_models import AutoModel

IMAGE_PATH_WARMUP = "/home/mkaic/inference/tests/inference/unit_tests/core/utils/assets/1.jpg"
# IMAGE_PATH_PROFILING = IMAGE_PATH_WARMUP
IMAGE_PATH_PROFILING = "/home/mkaic/inference/tests/workflows/integration_tests/execution/assets/car.jpg"
DEVICE = os.environ.get("DEVICE", "cuda:0")
CYCLES = 500
WARMUP = 50


def main() -> None:
    image = cv2.imread(IMAGE_PATH_WARMUP)
    model = AutoModel.from_pretrained(
        model_id_or_path="rfdetr-nano", device=torch.device(DEVICE), backend="trt"
    )

    image_warmup = cv2.imread(IMAGE_PATH_WARMUP)
    pre_processed_warmup, metadata = model.pre_process(image_warmup)
    print(f"Pre-processed image shape: {pre_processed_warmup.shape}")

    print(f"Warming up ({WARMUP} iterations each)...")
    for _ in range(WARMUP):
        model.forward(pre_processed_warmup, use_cuda_graph=False)
        model.forward(pre_processed_warmup, use_cuda_graph=True)
    # torch.cuda.synchronize()

    print(f"Profiling ({CYCLES} iterations each)...")
    image_profiling = cv2.imread(IMAGE_PATH_PROFILING)
    pre_processed_profiling, metadata = model.pre_process(image_profiling)
    print(f"Pre-processed image shape: {pre_processed_profiling.shape}")

    start = time.perf_counter()
    for _ in tqdm(range(CYCLES), desc="Without CUDA graphs"):
        model.forward(pre_processed_profiling, use_cuda_graph=False)
    # torch.cuda.synchronize()
    baseline_fps = CYCLES / (time.perf_counter() - start)

    start = time.perf_counter()
    for _ in tqdm(range(CYCLES), desc="With CUDA graphs"):
        model.forward(pre_processed_profiling, use_cuda_graph=True)
    # torch.cuda.synchronize()
    cudagraph_fps = CYCLES / (time.perf_counter() - start)

    result_baseline = model.forward(pre_processed_profiling, use_cuda_graph=False)
    result_cudagraph = model.forward(pre_processed_profiling, use_cuda_graph=True)
    # torch.cuda.synchronize()

    print(f"Result baseline: {result_baseline}")
    print(f"Result cudagraph: {result_cudagraph}")

    dets_match = torch.allclose(result_baseline[0], result_cudagraph[0], atol=1e-4)
    labels_match = torch.allclose(result_baseline[1], result_cudagraph[1], atol=1e-4)

    print(f"\n{'='*50}")
    print(f"Forward pass FPS (no CUDA graphs): {baseline_fps:.1f}")
    print(f"Forward pass FPS (CUDA graphs):    {cudagraph_fps:.1f}")
    print(f"Speedup: {cudagraph_fps / baseline_fps:.2f}x")
    print(f"Outputs match: dets={dets_match}, labels={labels_match}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

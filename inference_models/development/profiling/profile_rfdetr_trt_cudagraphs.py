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
CYCLES = 10_000
WARMUP = 50


def main() -> None:
    image = cv2.imread(IMAGE_PATH_WARMUP)
    model = AutoModel.from_pretrained(
        model_id_or_path="rfdetr-seg-preview", device=torch.device(DEVICE), backend="trt"
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

    expected_warmup = model.forward(pre_processed_warmup, use_cuda_graph=False)
    expected_profiling = model.forward(pre_processed_profiling, use_cuda_graph=False)

    print("Testing for race conditions (alternating inputs 20 times)...")
    all_match = True
    for i in range(20):
        if i % 2 == 0:
            result = model.forward(pre_processed_warmup, use_cuda_graph=True)
            expected = expected_warmup
            img_name = "warmup"
        else:
            result = model.forward(pre_processed_profiling, use_cuda_graph=True)
            expected = expected_profiling
            img_name = "profiling"

        dets_match = torch.allclose(result[0], expected[0], atol=1e-6)
        labels_match = torch.allclose(result[1], expected[1], atol=1e-6)
        if not (dets_match and labels_match):
            print(f"  MISMATCH at iteration {i} ({img_name}): dets={dets_match}, labels={labels_match}")
            all_match = False

    if all_match:
        print("  All 20 iterations matched expected outputs.")

    print(f"\n{'='*50}")
    print(f"Forward pass FPS (no CUDA graphs): {baseline_fps:.1f}")
    print(f"Forward pass FPS (CUDA graphs):    {cudagraph_fps:.1f}")
    print(f"Speedup: {cudagraph_fps / baseline_fps:.2f}x")
    print(f"Race condition test: {'PASSED' if all_match else 'FAILED'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

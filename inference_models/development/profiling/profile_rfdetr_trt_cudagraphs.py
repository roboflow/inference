import os
import time

import cv2
import numpy as np
import torch

from inference_models import AutoModel

IMAGE_1 = os.environ.get("IMAGE_PATH_WARMUP", None)
IMAGE_2 = os.environ.get("IMAGE_PATH_PROFILING", None)
DEVICE = os.environ.get("DEVICE", "cuda:0")
CYCLES = int(os.environ.get("CYCLES", "10_000"))
WARMUP = int(os.environ.get("WARMUP", "50"))


def main() -> None:

    model = AutoModel.from_pretrained(
        model_id_or_path="rfdetr-nano", device=torch.device(DEVICE), backend="trt"
    )

    if IMAGE_1 is not None:
        image_1 = cv2.imread(IMAGE_1)
    else:
        image_1 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)

    if IMAGE_2 is not None:
        image_2 = cv2.imread(IMAGE_2)
    else:
        image_2 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)

    pre_processed_1, _ = model.pre_process(image_1)
    pre_processed_2, _ = model.pre_process(image_2)


    expected_output_1_no_cuda_graph = model.forward(pre_processed_1, use_cuda_graph=False)
    expected_output_2_no_cuda_graph = model.forward(pre_processed_2, use_cuda_graph=False)

    expected_output_1_capture_cuda_graph = model.forward(pre_processed_1, use_cuda_graph=True)
    expected_output_2_capture_cudagraph = model.forward(pre_processed_2, use_cuda_graph=True)

    expected_output_1_replayed_cudagraph = model.forward(pre_processed_1, use_cuda_graph=True)
    expected_output_2_replayed_cudagraph = model.forward(pre_processed_2, use_cuda_graph=True)

    for i in [0, 1]:
        assert torch.allclose(expected_output_1_no_cuda_graph[i], expected_output_1_capture_cuda_graph[i], atol=1e-6)
        assert torch.allclose(expected_output_2_no_cuda_graph[i], expected_output_2_capture_cudagraph[i], atol=1e-6)
        assert torch.allclose(expected_output_1_no_cuda_graph[i], expected_output_1_replayed_cudagraph[i], atol=1e-6)
        assert torch.allclose(expected_output_2_no_cuda_graph[i], expected_output_2_replayed_cudagraph[i], atol=1e-6)

    print("Timing without CUDA graphs...")
    start = time.perf_counter()
    for _ in range(CYCLES):
        model.forward(pre_processed_2, use_cuda_graph=False)
    baseline_fps = CYCLES / (time.perf_counter() - start)

    print("Timing with CUDA graphs...")
    start = time.perf_counter()
    for _ in range(CYCLES):
        model.forward(pre_processed_2, use_cuda_graph=True)
    cudagraph_fps = CYCLES / (time.perf_counter() - start)

    print(f"\n{'='*50}")
    print(f"Forward pass FPS (no CUDA graphs): {baseline_fps:.1f}")
    print(f"Forward pass FPS (CUDA graphs):    {cudagraph_fps:.1f}")
    print(f"Speedup: {cudagraph_fps / baseline_fps:.2f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

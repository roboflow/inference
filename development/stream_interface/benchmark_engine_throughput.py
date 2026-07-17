#!/usr/bin/env python3
"""Micro-benchmark that isolates Execution Engine throughput from the per-frame image
decode + numpy->GPU conversion.

It decodes `--decoded-frames` frames from a video ONCE, up front, into the format the
active data representation wants:
  * ENABLE_TENSOR_DATA_REPRESENTATION=True  -> CHW RGB uint8 tensors on the GPU
    (WORKFLOWS_IMAGE_TENSOR_DEVICE), i.e. exactly what WorkflowImageData.tensor_image
    builds — so the conversion is NOT paid inside the loop;
  * ENABLE_TENSOR_DATA_REPRESENTATION=False -> numpy BGR HWC frames.

Then it runs a single ExecutionEngine instance `--engine-runs` times in a loop, cycling
through the pre-decoded frames by modulo. The measured FPS therefore reflects EE + model
throughput only, with the per-frame image materialization removed from the hot path.

Run it twice on the same clip to A/B the data path:
    ENABLE_TENSOR_DATA_REPRESENTATION=True  python ...benchmark_engine_throughput.py ...
    ENABLE_TENSOR_DATA_REPRESENTATION=False python ...benchmark_engine_throughput.py ...
If the tensor run now matches (or beats) the numpy run here, while the live pipeline
showed tensor at half the FPS, the gap is the per-frame numpy->GPU conversion.
"""
import argparse
import os
import sys
import time
from typing import List

import cv2
import numpy as np
import torch

# The top-level `inference` package is not pip-installed in dev; resolve it from the repo
# root so this script runs from any working dir.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    MAX_ACTIVE_MODELS,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.models.utils import ROBOFLOW_MODEL_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Execution Engine throughput over pre-decoded frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument(
        "--decoded-frames",
        type=int,
        default=64,
        help="Number of frames to decode up front (tensor mode: onto the GPU).",
    )
    parser.add_argument(
        "--engine-runs",
        type=int,
        default=1000,
        help="Number of timed ExecutionEngine.run() calls (frames cycled by modulo).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Frames fed per ExecutionEngine.run() call (a batch of images).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warm-up runs (model load / TRT build / cache warm-up).",
    )
    parser.add_argument(
        "--workspace", required=True, help="Roboflow workspace name (the URL slug)."
    )
    parser.add_argument(
        "--workflow-id",
        required=True,
        help="Registered workflow id (fetched from the Roboflow platform).",
    )
    parser.add_argument(
        "--workflow-version-id",
        default=None,
        help="Optional specific workflow version id.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional value for the workflow's `model_id` parameter.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY"),
        help="Roboflow API key (defaults to the ROBOFLOW_API_KEY / API_KEY env var).",
    )
    parser.add_argument(
        "--image-input-name", default="image", help="Workflow image input name."
    )
    return parser.parse_args()


def build_model_manager() -> ModelManager:
    # Same construction as tests/workflows/integration_tests/conftest.py::model_manager.
    registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    return WithFixedSizeCache(ModelManager(model_registry=registry), max_size=MAX_ACTIVE_MODELS)


def decode_frames(video_path: str, count: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frames: List[np.ndarray] = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)  # numpy HWC BGR (uint8)
    cap.release()
    if not frames:
        raise SystemExit("Decoded 0 frames — empty or unreadable video.")
    return frames


def to_gpu_tensor(bgr_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    # Exactly what WorkflowImageData.tensor_image builds: CHW RGB uint8 on the device.
    rgb = bgr_hwc[:, :, ::-1].copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(device)


def _cuda_sync_if_needed() -> None:
    if ENABLE_TENSOR_DATA_REPRESENTATION and WORKFLOWS_IMAGE_TENSOR_DEVICE.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.video):
        raise SystemExit(f"Video file not found: {args.video}")
    if not args.api_key:
        raise SystemExit(
            "A Roboflow API key is required to fetch the workflow from the platform. "
            "Pass --api-key or set ROBOFLOW_API_KEY."
        )

    # Fetch the workflow definition from the Roboflow platform (by workspace + id).
    workflow = get_workflow_specification(
        api_key=args.api_key,
        workspace_id=args.workspace,
        workflow_id=args.workflow_id,
        workflow_version_id=args.workflow_version_id,
        use_cache=True,
    )

    print(
        f"torch.cuda.is_available() = {torch.cuda.is_available()} | "
        f"WORKFLOWS_IMAGE_TENSOR_DEVICE = {WORKFLOWS_IMAGE_TENSOR_DEVICE} | "
        f"ENABLE_TENSOR_DATA_REPRESENTATION = {ENABLE_TENSOR_DATA_REPRESENTATION}"
    )
    if ENABLE_TENSOR_DATA_REPRESENTATION and WORKFLOWS_IMAGE_TENSOR_DEVICE.type != "cuda":
        print(
            "WARNING: tensor mode is on but the tensor device is not CUDA — frames live on "
            f"'{WORKFLOWS_IMAGE_TENSOR_DEVICE}', so the model will still copy to GPU itself."
        )

    raw = decode_frames(args.video, args.decoded_frames)

    # Pre-build the runtime image inputs ONCE, in the active representation's format, so
    # the per-frame decode/conversion is out of the timed loop.
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        device = WORKFLOWS_IMAGE_TENSOR_DEVICE
        images = [to_gpu_tensor(f, device) for f in raw]
        _cuda_sync_if_needed()  # make sure the H2D copies are done before timing
        print(
            f"pre-decoded {len(images)} frames -> GPU tensors on {device}, "
            f"shape {tuple(images[0].shape)} dtype {images[0].dtype}"
        )
    else:
        images = raw  # numpy HWC BGR — the legacy path
        print(
            f"pre-decoded {len(images)} frames -> numpy, "
            f"shape {images[0].shape} dtype {images[0].dtype}"
        )

    model_manager = build_model_manager()
    engine = ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": args.api_key,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        workflow_id=args.workflow_id,
    )

    extra_params = {"model_id": args.model_id} if args.model_id else {}
    n = len(images)
    batch_size = max(1, args.batch_size)
    if batch_size > n:
        print(
            f"NOTE: batch size {batch_size} > pre-decoded frames {n}; frames repeat within a batch."
        )

    def run_once(i: int):
        # A batch is `batch_size` frames pulled from the pre-decoded pool (modulo).
        batch = [images[(i * batch_size + j) % n] for j in range(batch_size)]
        return engine.run(
            runtime_parameters={args.image_input_name: batch, **extra_params}
        )

    print(f"warming up ({args.warmup} runs — model load / TRT build)...", flush=True)
    for i in range(args.warmup):
        run_once(i)
    _cuda_sync_if_needed()

    print(
        f"running {args.engine_runs} EE iterations (batch size {batch_size})...",
        flush=True,
    )
    start = time.monotonic()
    for i in range(args.engine_runs):
        run_once(i)
    _cuda_sync_if_needed()  # flush async GPU work before stopping the clock
    elapsed = time.monotonic() - start

    total_frames = args.engine_runs * batch_size
    runs_per_s = args.engine_runs / elapsed if elapsed > 0 else 0.0
    frames_per_s = total_frames / elapsed if elapsed > 0 else 0.0
    print("\n=== Summary ===")
    print(f"data representation : {'TENSOR' if ENABLE_TENSOR_DATA_REPRESENTATION else 'NUMPY'}")
    print(f"pre-decoded frames  : {n}")
    print(f"batch size          : {batch_size}")
    print(f"engine runs (timed) : {args.engine_runs}")
    print(f"frames processed    : {total_frames}")
    print(f"elapsed             : {elapsed:.3f} s")
    print(f"throughput          : {frames_per_s:.1f} frames/s ({runs_per_s:.1f} runs/s)")
    print(f"per-run latency     : {elapsed / args.engine_runs * 1000:.3f} ms")
    print(f"per-frame latency   : {elapsed / total_frames * 1000:.3f} ms")


if __name__ == "__main__":
    main()

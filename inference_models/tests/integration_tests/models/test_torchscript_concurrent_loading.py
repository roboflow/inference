"""Regression test for concurrent TorchScript model loading.

TorchScript deserialization mutates a process-global type registry that is not
thread-safe. Loading TorchScript-backed models concurrently (as the HTTP server does
when preloading several PINNED_MODELS on a ThreadPoolExecutor) could corrupt it,
surfacing non-deterministically as:

    KeyError: '__torch__.torch.nn.functional.interpolate'
    RuntimeError: ... Enum<...___torch_mangle_0.InterpolationMode> ...

This test hammers `torch.jit.load` directly from several threads, releasing them through
a barrier immediately before the load on every round so the unsafe deserialization
windows overlap, repeated over many rounds to make the race fire deterministically.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE

ROUNDS = 50
WORKERS = 8


@pytest.mark.slow
@pytest.mark.torch_models
def test_concurrent_torchscript_loading_does_not_corrupt_global_registry(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
) -> None:
    # given
    weights_paths = [
        os.path.join(pkg, "weights.torchscript")
        for pkg in (
            coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
            asl_yolov8n_torchscript_seg_static_bs_stretch,
            yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        )
    ]
    for path in weights_paths:
        assert os.path.exists(path), path

    barrier = threading.Barrier(WORKERS)
    errors = []

    def _load(worker_idx: int) -> None:
        path = weights_paths[worker_idx % len(weights_paths)]
        for _ in range(ROUNDS):
            barrier.wait(timeout=120)
            try:
                torch.jit.load(path, map_location=DEFAULT_DEVICE).eval()
            except Exception as error:
                errors.append(error)
                raise

    # when
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_load, i) for i in range(WORKERS)]
        for future in futures:
            future.result(timeout=600)

    # then
    assert not errors

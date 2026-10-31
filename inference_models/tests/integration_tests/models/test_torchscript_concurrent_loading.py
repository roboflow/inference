"""Regression test: concurrent TorchScript work corrupts the global registry.

`torch.jit.script` (run by SAM3 at build time on torchvision transforms) and
`torch.jit.load` (run by the YOLO/CLIP TorchScript loaders) mutate the SAME process-global,
non-thread-safe TorchScript type registry. Running them concurrently (as the HTTP server
does when preloading several PINNED_MODELS on a ThreadPoolExecutor) corrupts it, surfacing
non-deterministically as:

    KeyError: '__torch__.torch.nn.functional.interpolate'
    RuntimeError: ... Enum<...___torch_mangle_0.InterpolationMode> ...
    RuntimeError: Can't redefine method: forward on class: ...Resize...

Threads loading a TorchScript model via the real loader run alongside threads scripting
`nn.Sequential(Resize, Normalize)` (mirroring SAM3's build), released together through a
barrier on every round. Both serialize on one shared `torchscript_state_global_lock`, so
the test passes; with `lock=None` it reproduces the corruption above.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn
from torchvision.transforms import Normalize, Resize

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.common.torch import torchscript_global_lock
from inference_models.models.yolov8.yolov8_object_detection_torch_script import (
    YOLOv8ForObjectDetectionTorchScript,
)

ROUNDS = 1
LOAD_WORKERS = 2
SCRIPT_WORKERS = 3
RESOLUTION = 1024


@pytest.mark.slow
@pytest.mark.torch_models
def test_concurrent_torchscript_work_does_not_corrupt_global_registry(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
) -> None:
    # given
    lock = threading.Lock()
    barrier = threading.Barrier(LOAD_WORKERS + SCRIPT_WORKERS)
    errors = []

    def _load_worker() -> None:
        for _ in range(ROUNDS):
            barrier.wait(timeout=120)
            try:
                YOLOv8ForObjectDetectionTorchScript.from_pretrained(
                    model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
                    device=DEFAULT_DEVICE,
                    torchscript_state_global_lock=lock,
                )
            except Exception as error:
                errors.append(("load", repr(error)))
                raise

    def _script_worker() -> None:
        for _ in range(ROUNDS):
            barrier.wait(timeout=120)
            try:
                with torchscript_global_lock(lock):
                    torch.jit.script(
                        nn.Sequential(
                            Resize((RESOLUTION, RESOLUTION)),
                            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                        )
                    )
            except Exception as error:
                errors.append(("script", repr(error)))
                raise

    # when
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS + SCRIPT_WORKERS) as executor:
        futures = [executor.submit(_load_worker) for _ in range(LOAD_WORKERS)]
        futures += [executor.submit(_script_worker) for _ in range(SCRIPT_WORKERS)]
        for future in futures:
            future.result(timeout=30)

    # then
    assert not errors

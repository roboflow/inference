"""Regression test for concurrent TorchScript work corrupting the global registry.

`torch.jit.script` (run by SAM3 at build time on torchvision transforms) and
`torch.jit.load` (run by the YOLO/CLIP TorchScript loaders) mutate the SAME process-global,
non-thread-safe TorchScript type registry. Running them concurrently (as the HTTP server
does when preloading several PINNED_MODELS on a ThreadPoolExecutor) corrupts it, surfacing
non-deterministically as:

    KeyError: '__torch__.torch.nn.functional.interpolate'
    RuntimeError: ... Enum<...___torch_mangle_0.InterpolationMode> ...
    RuntimeError: Can't redefine method: forward on class: ...Resize...

Every TorchScript-touching path now serializes on a single process-wide lock
(`inference_models.models.common.torch.torchscript_load_lock`). This test reproduces the
mixed scenario: threads loading a TorchScript model via the real loader (which takes the
lock) run alongside threads scripting `nn.Sequential(Resize, Normalize)` under the same
lock (mirroring SAM3's build), released together through a barrier on every round.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn
from torchvision.transforms import Normalize, Resize

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.common.torch import torchscript_load_lock
from inference_models.models.yolov8.yolov8_object_detection_torch_script import (
    YOLOv8ForObjectDetectionTorchScript,
)

ROUNDS = 5
LOAD_WORKERS = 2
SCRIPT_WORKERS = 3
RESOLUTION = 1024


@pytest.mark.slow
@pytest.mark.torch_models
def test_concurrent_torchscript_work_does_not_corrupt_global_registry(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
) -> None:
    # given
    barrier = threading.Barrier(LOAD_WORKERS + SCRIPT_WORKERS)
    errors = []

    def _load_worker() -> None:
        for _ in range(ROUNDS):
            barrier.wait(timeout=300)
            try:
                YOLOv8ForObjectDetectionTorchScript.from_pretrained(
                    model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
                    device=DEFAULT_DEVICE,
                )
            except Exception as error:
                errors.append(("load", repr(error)))
                raise

    def _script_worker() -> None:
        for _ in range(ROUNDS):
            barrier.wait(timeout=300)
            try:
                with torchscript_load_lock():
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
            future.result(timeout=1200)

    # then
    assert not errors

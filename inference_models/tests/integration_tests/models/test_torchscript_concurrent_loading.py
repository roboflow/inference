"""Regression test for concurrent TorchScript model loading.

TorchScript deserialization mutates a process-global type registry that is not
thread-safe. Loading TorchScript-backed models concurrently (as the HTTP server does
when preloading several PINNED_MODELS on a ThreadPoolExecutor) could corrupt it,
surfacing non-deterministically as:

    KeyError: '__torch__.torch.nn.functional.interpolate'
    RuntimeError: ... Enum<...___torch_mangle_0.InterpolationMode> ...

The loaders now serialize TorchScript loads behind a shared lock
(`inference_models.models.common.torch.torchscript_load_lock`). This test loads several
different TorchScript packages (object detection + instance segmentation + key points)
at once, releasing every thread into `torch.jit.load` simultaneously via a barrier, and
asserts every load succeeds.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.yolov8.yolov8_instance_segmentation_torch_script import (
    YOLOv8ForInstanceSegmentationTorchScript,
)
from inference_models.models.yolov8.yolov8_key_points_detection_torch_script import (
    YOLOv8ForKeyPointsDetectionTorchScript,
)
from inference_models.models.yolov8.yolov8_object_detection_torch_script import (
    YOLOv8ForObjectDetectionTorchScript,
)


@pytest.mark.slow
@pytest.mark.torch_models
def test_concurrent_torchscript_loading_does_not_corrupt_global_registry(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
) -> None:
    # given
    load_specs = [
        (
            YOLOv8ForObjectDetectionTorchScript,
            coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
        ),
        (
            YOLOv8ForInstanceSegmentationTorchScript,
            asl_yolov8n_torchscript_seg_static_bs_stretch,
        ),
        (
            YOLOv8ForKeyPointsDetectionTorchScript,
            yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        ),
    ] * 2
    barrier = threading.Barrier(len(load_specs))

    def _load(loader_cls, package_path):
        barrier.wait(timeout=60)
        return loader_cls.from_pretrained(
            model_name_or_path=package_path,
            device=DEFAULT_DEVICE,
        )

    # when
    with ThreadPoolExecutor(max_workers=len(load_specs)) as executor:
        futures = [
            executor.submit(_load, loader_cls, package_path)
            for loader_cls, package_path in load_specs
        ]
        models = [future.result(timeout=300) for future in futures]

    # then
    assert len(models) == len(load_specs)
    assert all(
        isinstance(model, expected_cls)
        for model, (expected_cls, _) in zip(models, load_specs)
    )

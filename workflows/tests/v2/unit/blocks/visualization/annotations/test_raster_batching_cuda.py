"""CUDA counterpart of ``test_raster_batching`` (optional).

Skipped without CUDA. On a CUDA image and predictions, blur, pixelate and crop
must keep the output on the device, leave the inputs unchanged, equal the CPU
output (blur exactly; resampled pixels within 1 level, as float32 weights may
round differently on the device) and make the same number of host-to-device
copies for 2 and 24 rows. A side stream chosen by the caller stays current and
gives the same pixels.
"""

from types import SimpleNamespace
from typing import Any, Callable, Dict, Sequence

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BlurVisualizationBlock,
    CropVisualizationBlock,
    PixelateVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from torch.profiler import (
    ProfilerActivity,
    kineto_available,
    profile,
    supported_activities,
)

from .support import detections, pixels_of, run_v2, scene
from .test_raster_batching import HEIGHT, WIDTH, instances_of, overlapping_rows

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CASES = [
    (PixelateVisualizationBlock, detections, {"pixel_size": 5}),
    (CropVisualizationBlock, detections, {"scale_factor": 1.5}),
    (BlurVisualizationBlock, instances_of, {"kernel_size": 5, "padding": 2}),
]
# Blur sums integers; pixelate and crop resample with float32 weights.
LEVELS = {
    PixelateVisualizationBlock: 1,
    CropVisualizationBlock: 1,
    BlurVisualizationBlock: 0,
}


def image_on(device: str) -> ImageData:
    pixels = torch.from_numpy(scene(HEIGHT, WIDTH, seed=3)).permute(2, 0, 1)
    image = ImageData.from_tensor(pixels.contiguous().to(device), image_id="frame")

    return image


def host_to_device_copies(run: Callable[[], Any]) -> int:
    """Count HtoD events, skipping a trace with no recorded CUDA activity.

    Args:
        run: Operation to warm up, then profile on the current CUDA stream.

    Returns:
        Number of recorded CUDA host-to-device copies.
    """
    run()  # warm-up: lazy CUDA initialisation copies are not the block's
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        run()
        torch.cuda.synchronize()
    copies = _htod_count(trace.events())

    return copies


def _htod_count(events: Sequence[Any]) -> int:
    cuda_events = [
        event for event in events if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not cuda_events:
        pytest.skip("CUDA copy counts unavailable: profiler recorded no CUDA activity")

    copies = sum("HtoD" in event.name for event in cuda_events)

    return copies


def _require_cuda_copy_profiler() -> None:
    if not kineto_available():
        pytest.skip("CUDA copy counts require Kineto; Kineto is unavailable")
    if ProfilerActivity.CUDA not in supported_activities():
        pytest.skip(
            "CUDA copy counts unavailable: profiler does not support CUDA activity"
        )

    host = torch.zeros(32, dtype=torch.uint8)
    copies = host_to_device_copies(lambda: host.to("cuda"))
    assert copies > 0, (
        "CUDA profiler recorded activity but no HtoD event for a known CPU-to-CUDA "
        "copy; block copy counts cannot be trusted"
    )


@pytest.mark.parametrize(
    "kineto, activities, reason",
    [
        (False, {ProfilerActivity.CPU}, "Kineto is unavailable"),
        (True, {ProfilerActivity.CPU}, "does not support CUDA activity"),
    ],
)
def test_copy_profiler_requires_the_recording_capability(
    monkeypatch, kineto, activities, reason
) -> None:
    monkeypatch.setattr(__name__ + ".kineto_available", lambda: kineto)
    monkeypatch.setattr(__name__ + ".supported_activities", lambda: activities)

    with pytest.raises(pytest.skip.Exception, match=reason):
        _require_cuda_copy_profiler()


def test_copy_count_skips_a_cpu_only_trace() -> None:
    events = [
        SimpleNamespace(name="aten::to", device_type=torch.autograd.DeviceType.CPU)
    ]

    with pytest.raises(pytest.skip.Exception, match="recorded no CUDA activity"):
        _htod_count(events)


def test_copy_count_counts_only_cuda_htod_events() -> None:
    events = [
        SimpleNamespace(name="Memcpy HtoD", device_type=torch.autograd.DeviceType.CUDA),
        SimpleNamespace(name="Memcpy DtoH", device_type=torch.autograd.DeviceType.CUDA),
        SimpleNamespace(name="Memcpy HtoD", device_type=torch.autograd.DeviceType.CPU),
    ]

    assert _htod_count(events) == 1


@pytest.mark.parametrize("copies", [0, 1])
def test_copy_profiler_positive_control_rejects_an_unobserved_copy(
    monkeypatch, copies
) -> None:
    monkeypatch.setattr(__name__ + ".kineto_available", lambda: True)
    monkeypatch.setattr(
        __name__ + ".supported_activities", lambda: {ProfilerActivity.CUDA}
    )
    monkeypatch.setattr(__name__ + ".host_to_device_copies", lambda run: copies)

    if copies == 0:
        with pytest.raises(AssertionError, match="known CPU-to-CUDA copy"):
            _require_cuda_copy_profiler()
    else:
        _require_cuda_copy_profiler()


@CUDA
@pytest.mark.parametrize("block, predictions_of, parameters", CASES)
def test_cuda_output_matches_cpu_and_keeps_inputs(
    block: type,
    predictions_of: Callable[..., Any],
    parameters: Dict[str, Any],
) -> None:
    image = image_on("cuda")
    predictions = predictions_of(image, overlapping_rows(24), device="cuda")
    pixels_before = image.tensor_image.clone()
    boxes_before = predictions.xyxy.clone()
    cpu_image = image_on("cpu")

    output = run_v2(block, image=image, predictions=predictions, **parameters)

    expected = run_v2(
        block,
        image=cpu_image,
        predictions=predictions_of(cpu_image, overlapping_rows(24)),
        **parameters,
    )
    assert output.tensor_image.is_cuda
    difference = np.abs(
        pixels_of(output).astype(np.int16) - pixels_of(expected).astype(np.int16)
    )
    assert int(difference.max()) <= LEVELS[block]
    assert torch.equal(image.tensor_image, pixels_before)
    assert torch.equal(predictions.xyxy, boxes_before)


@CUDA
@pytest.mark.parametrize("block, predictions_of, parameters", CASES)
def test_cuda_copies_do_not_grow_with_rows(
    block: type,
    predictions_of: Callable[..., Any],
    parameters: Dict[str, Any],
) -> None:
    _require_cuda_copy_profiler()
    image = image_on("cuda")
    few = predictions_of(image, overlapping_rows(2), device="cuda")
    many = predictions_of(image, overlapping_rows(24), device="cuda")

    copies_for_few = host_to_device_copies(
        lambda: run_v2(block, image=image, predictions=few, **parameters)
    )
    copies_for_many = host_to_device_copies(
        lambda: run_v2(block, image=image, predictions=many, **parameters)
    )

    assert copies_for_many == copies_for_few


@CUDA
@pytest.mark.parametrize("block, predictions_of, parameters", CASES)
def test_cuda_caller_stream_stays_current_and_gives_the_same_pixels(
    block: type,
    predictions_of: Callable[..., Any],
    parameters: Dict[str, Any],
) -> None:
    image = image_on("cuda")
    predictions = predictions_of(image, overlapping_rows(24), device="cuda")
    expected = run_v2(block, image=image, predictions=predictions, **parameters)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()

    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        output = run_v2(block, image=image, predictions=predictions, **parameters)
        assert torch.cuda.current_stream() == side
    side.synchronize()

    assert torch.equal(output.tensor_image, expected.tensor_image)

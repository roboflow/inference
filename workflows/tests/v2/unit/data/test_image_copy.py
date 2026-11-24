"""``ImageData.copy``: owned pixels, unchanged identity and metadata.

Every case runs on CPU and, where available, MPS and CUDA. The CUDA variants
are declared for a later device run; they skip on machines without CUDA.

The CUDA ordering checks queue ``torch.cuda._sleep`` (a debug kernel that
spins for a number of GPU cycles) so a stream is provably busy when ``copy``
is called; each asserts that control first and fails, not passes, when it does
not hold. Each check runs its scenario once untimed to load every kernel it
uses: on Thor the first call in a process saw a busy stream already idle. A
check's queued work is ``SLEEP_MS`` at the calibrated clock.
"""

import gc
import weakref
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
from unittest import mock

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.data.image import CompositeSource, ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "mps",
        id="mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="needs MPS"
        ),
    ),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]
METADATA = VideoMetadata(
    video_identifier="cam-a",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 13, 12, 0, 0),
    fps=30,
    comes_from_video_file=False,
)
FIELDS = (
    "image_id",
    "parent",
    "root",
    "video_metadata",
    "composite_sources",
    "parent_row",
)


def _frame(device: str) -> ImageData:
    pixels = torch.arange(3 * 8 * 10, dtype=torch.int64).remainder(251)
    pixels = pixels.to(torch.uint8).reshape(3, 8, 10).to(device)
    image = ImageData.from_tensor(pixels, image_id="frame-7", video_metadata=METADATA)

    return image


def _crop_with_row(device: str) -> ImageData:
    crop = _frame(device).crop((2, 1, 9, 7)).with_parent_row(RowRef("detections", 3))

    return crop


def _composite_crop(device: str) -> ImageData:
    source = _frame(device)
    placement = CompositeSource.place(source, index=(0,), canvas_xyxy=(0, 0, 10, 8))
    canvas = ImageData.composite(
        torch.zeros((3, 8, 20), dtype=torch.uint8, device=device),
        sources=[placement],
    )
    crop = canvas.crop((4, 0, 16, 8))

    return crop


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("build", [_frame, _crop_with_row, _composite_crop])
@pytest.mark.parametrize("target", [None, "cpu"])
def test_copy_keeps_every_metadata_field_and_pixel_value(device, build, target):
    image = build(device)

    copied = image.copy(device=target)

    for name in FIELDS:
        assert getattr(copied, name) == getattr(image, name), name
    assert copied.video_metadata is image.video_metadata
    assert copied.size_hw == image.size_hw
    expected_device = image.device if target is None else torch.device("cpu")
    assert copied.device == expected_device
    assert torch.equal(copied.tensor_image.cpu(), image.tensor_image.cpu())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("target", [None, "cpu"])
def test_copy_pixels_are_independent_including_cpu_to_cpu(device, target):
    image = _frame(device)
    before = image.tensor_image.cpu().clone()

    copied = image.copy(device=target)
    image.tensor_image.fill_(0)

    assert copied.tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert torch.equal(copied.tensor_image.cpu(), before)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_of_a_view_owns_only_its_own_contiguous_pixels(device):
    # A pool slot or crop view: the copy must not keep the larger buffer alive.
    slab = torch.zeros((3, 16, 16), dtype=torch.uint8, device=device)
    view = slab[:, 4:12, 2:14]
    image = ImageData.from_tensor(view)
    slab_storage = weakref.ref(slab.untyped_storage())

    copied = image.copy()
    del slab, view, image
    gc.collect()

    assert copied.tensor_image.is_contiguous()
    assert copied.tensor_image.untyped_storage().nbytes() == 3 * 8 * 12
    assert slab_storage() is None


@pytest.mark.parametrize("device", DEVICES)
def test_dropping_the_source_releases_it_while_the_copy_lives(device):
    image = _frame(device)
    source_storage = weakref.ref(image.tensor_image.untyped_storage())

    copied = image.copy()
    del image
    gc.collect()

    assert source_storage() is None
    assert copied.tensor_image.sum().item() > 0


@pytest.mark.parametrize("device", ["mps", "cuda:0", torch.device("meta")])
def test_copy_rejects_targets_other_than_the_cpu(device):
    with pytest.raises(ValueError, match="device='cpu'"):
        _frame("cpu").copy(device=device)


def test_cpu_copy_records_no_cuda_event():
    with mock.patch.object(torch.cuda, "Event") as event:
        _frame("cpu").copy()

    event.assert_not_called()


# ----------------------------------------------------------------- CUDA order
SLEEP_MS = 100  # >100x the host steps between queuing it and the check
CALIBRATION_CYCLES = 10_000_000
WARM_UP_CYCLES = 1_000
NOT_BUSY = (
    "control failed: {stream} finished its {ms} ms of queued GPU work before "
    "the check, so the result would say nothing. Check that CUDA_LAUNCH_BLOCKING "
    "is unset and the device runs streams concurrently; calibrated cycles: {cycles}"
)


@dataclass(frozen=True)
class _Gpu:
    side: torch.cuda.Stream  # the producer's and the caller's current stream
    other: torch.cuda.Stream  # unrelated work, or the producer's next frame
    sleep_cycles: int  # torch.cuda._sleep argument for about SLEEP_MS

    def not_busy(self, stream: str) -> str:
        message = NOT_BUSY.format(stream=stream, ms=SLEEP_MS, cycles=self.sleep_cycles)

        return message


@pytest.fixture(scope="module")
def gpu() -> _Gpu:
    # Fixed streams: the scenarios' warm-up fills the allocator cache for them.
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("needs torch.cuda._sleep to queue bounded GPU work")

    side, other = torch.cuda.Stream(), torch.cuda.Stream()
    elapsed_ms = []
    with torch.cuda.stream(side):
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda._sleep(CALIBRATION_CYCLES)
            end.record()
            end.synchronize()
            elapsed_ms.append(start.elapsed_time(end))
    # The fastest run gives the most cycles per ms: sleeps err on the long side.
    cycles = int(CALIBRATION_CYCLES * SLEEP_MS / min(elapsed_ms))

    return _Gpu(side=side, other=other, sleep_cycles=cycles)


@dataclass(frozen=True)
class _Observed:
    busy_before: bool  # the control: the stream was busy when copy() was called
    busy_after: bool
    copied: ImageData
    device_waits: int  # calls of torch.cuda.synchronize during copy()
    stream_waits: int  # calls of Stream.synchronize during copy()


def _copy_counting_waits(
    image: ImageData, target: Optional[str]
) -> Tuple[ImageData, int, int]:
    real_stream_wait = torch.cuda.Stream.synchronize
    with mock.patch.object(
        torch.cuda, "synchronize", wraps=torch.cuda.synchronize
    ) as device_wait, mock.patch.object(
        torch.cuda.Stream, "synchronize", autospec=True, side_effect=real_stream_wait
    ) as stream_wait:
        copied = image.copy(device=target)

    return copied, device_wait.call_count, stream_wait.call_count


def _copy_behind_busy_producer(
    gpu: _Gpu, target: Optional[str], cycles: int
) -> _Observed:
    # The producer's last write is queued on the caller's stream behind a sleep;
    # after copy() returns, the producer writes its next frame into the same
    # buffer from another stream, which the copy contract allows.
    with torch.cuda.stream(gpu.side):
        image = _frame("cuda")
        torch.cuda._sleep(cycles)
        image.tensor_image.add_(1)
        busy_before = not gpu.side.query()
        copied, device_waits, stream_waits = _copy_counting_waits(image, target)
        busy_after = not gpu.side.query()
    with torch.cuda.stream(gpu.other):
        image.tensor_image.fill_(0)
    gpu.other.synchronize()
    gpu.side.synchronize()

    return _Observed(busy_before, busy_after, copied, device_waits, stream_waits)


def _copy_while_another_stream_is_busy(
    gpu: _Gpu, target: Optional[str], cycles: int
) -> _Observed:
    # The caller's stream is idle and another stream is busy. Waiting on an
    # event of the caller's stream first is the control: it shows this device
    # can wait for one stream while another one stays busy.
    with torch.cuda.stream(gpu.side):
        image = _frame("cuda")
    gpu.side.synchronize()
    with torch.cuda.stream(gpu.other):
        torch.cuda._sleep(cycles)
    with torch.cuda.stream(gpu.side):
        own_stream = torch.cuda.Event()
        own_stream.record()
        own_stream.synchronize()
        busy_before = not gpu.other.query()
        copied, device_waits, stream_waits = _copy_counting_waits(image, target)
        busy_after = not gpu.other.query()
    gpu.other.synchronize()
    gpu.side.synchronize()

    return _Observed(busy_before, busy_after, copied, device_waits, stream_waits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("target", [None, "cpu"])
def test_cuda_copy_is_complete_on_return_on_the_current_stream(gpu, target):
    _copy_behind_busy_producer(gpu, target, WARM_UP_CYCLES)

    observed = _copy_behind_busy_producer(gpu, target, gpu.sleep_cycles)

    assert observed.busy_before, gpu.not_busy("the producer's stream")
    assert not observed.busy_after, "copy() returned before its stream reached it"
    assert observed.device_waits == 0, "copy() called torch.cuda.synchronize"
    assert observed.stream_waits == 0, "copy() called Stream.synchronize"
    expected = _frame("cpu").tensor_image + 1
    assert torch.equal(observed.copied.tensor_image.cpu(), expected)
    expected_device = "cuda" if target is None else "cpu"
    assert observed.copied.device.type == expected_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("target", [None, "cpu"])
def test_cuda_copy_does_not_wait_for_other_streams(gpu, target):
    _copy_while_another_stream_is_busy(gpu, target, WARM_UP_CYCLES)

    observed = _copy_while_another_stream_is_busy(gpu, target, gpu.sleep_cycles)

    assert observed.busy_before, gpu.not_busy("the unrelated stream")
    assert observed.busy_after, "copy() waited for an unrelated stream"
    assert observed.device_waits == 0, "copy() called torch.cuda.synchronize"
    assert observed.stream_waits == 0, "copy() called Stream.synchronize"
    assert torch.equal(observed.copied.tensor_image.cpu(), _frame("cpu").tensor_image)

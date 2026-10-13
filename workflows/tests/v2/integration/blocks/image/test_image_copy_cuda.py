"""``v2/image_copy@v1`` in a compiled workflow, with CUDA images.

The real compiler, session and executor run one ``v2/image_copy@v1`` step in
both block execution modes, with ``device="same"`` and ``device="cpu"``. The
metadata and ownership cases also run with CPU images, so they are checked on
machines without CUDA; the stream case needs CUDA.

The stream case queues ``torch.cuda._sleep`` (a debug kernel that spins for a
number of GPU cycles) so the producer's stream is provably busy when the
workflow runs; it asserts that control first and fails, not passes, when it
does not hold. It runs once untimed to load every kernel the run uses: on Thor
the first call in a process saw a busy stream already idle. Its queued work is
``SLEEP_MS`` at the calibrated clock.
"""

from dataclasses import dataclass
from datetime import datetime

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image.copy import ImageCopyBlock
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.row_identity import RowRef
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.plan import CompileOptions

CATALOGUE = Catalogue([ImageCopyBlock], kinds=[IMAGE_KIND])
MODES = ["run", "phases"]
TARGETS = ["same", "cpu"]
SOURCE_DEVICES = [
    pytest.param("cpu", id="cpu"),
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
SLEEP_MS = 100  # >100x the host steps between queuing it and the check
CALIBRATION_CYCLES = 10_000_000
WARM_UP_CYCLES = 1_000


def _session(mode: str, target: str):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": ImageCopyBlock.type,
                "name": "copy",
                "image": "$inputs.image",
                "device": target,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "copy", "selector": "$steps.copy.image"}
        ],
    }
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()

    return session


def _run(session, image: ImageData) -> ImageData:
    copied = session.run({"image": image}).outputs.data["copy"]
    if isinstance(copied, Batch):
        ((_, copied),) = copied.iter_with_indices()

    return copied


def _pixels() -> torch.Tensor:
    pixels = torch.arange(3 * 8 * 10, dtype=torch.int64).remainder(251)
    pixels = pixels.to(torch.uint8).reshape(3, 8, 10)

    return pixels


def _crop_with_row(device: str) -> ImageData:
    frame = ImageData.from_tensor(
        _pixels().to(device), image_id="frame-7", video_metadata=METADATA
    )
    crop = frame.crop((2, 1, 9, 7)).with_parent_row(RowRef("detections", 3))

    return crop


def _expected_device(source: str, target: str) -> str:
    device = source if target == "same" else "cpu"

    return device


@pytest.mark.parametrize("source", SOURCE_DEVICES)
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("mode", MODES)
def test_copy_keeps_every_metadata_field_and_owns_its_pixels(source, target, mode):
    image = _crop_with_row(source)
    before = image.tensor_image.cpu().clone()
    session = _session(mode, target)

    copied = _run(session, image)
    image.tensor_image.fill_(0)
    session.close()

    for name in FIELDS:
        assert getattr(copied, name) == getattr(image, name), name
    assert copied.video_metadata is image.video_metadata
    assert copied.parent_row == RowRef("detections", 3)
    assert copied.size_hw == image.size_hw
    assert copied.device.type == _expected_device(source, target)
    assert copied.tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert torch.equal(copied.tensor_image.cpu(), before)


# ----------------------------------------------------------------- CUDA order
@dataclass(frozen=True)
class _Gpu:
    side: torch.cuda.Stream  # the producer's and the caller's current stream
    other: torch.cuda.Stream  # the producer writing its next frame
    sleep_cycles: int  # torch.cuda._sleep argument for about SLEEP_MS


@pytest.fixture(scope="module")
def gpu() -> _Gpu:
    # Fixed streams: the warm-up run fills the allocator cache for them.
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
    busy_before: bool  # the control: the stream was busy when the run started
    busy_after: bool
    copied: ImageData
    source_ptr: int


def _run_behind_busy_producer(gpu: _Gpu, session, cycles: int) -> _Observed:
    # The producer's last write is queued on the caller's stream behind a
    # sleep; after the run, the producer writes its next frame into the same
    # buffer from another stream, which the block's contract allows.
    with torch.cuda.stream(gpu.side):
        image = _crop_with_row("cuda")
        torch.cuda._sleep(cycles)
        image.tensor_image.add_(1)
        busy_before = not gpu.side.query()
        copied = _run(session, image)
        busy_after = not gpu.side.query()
    with torch.cuda.stream(gpu.other):
        image.tensor_image.fill_(0)
    gpu.other.synchronize()
    gpu.side.synchronize()

    return _Observed(busy_before, busy_after, copied, image.tensor_image.data_ptr())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("mode", MODES)
def test_copy_is_complete_when_the_run_returns_on_the_current_stream(gpu, target, mode):
    session = _session(mode, target)
    _run_behind_busy_producer(gpu, session, WARM_UP_CYCLES)

    observed = _run_behind_busy_producer(gpu, session, gpu.sleep_cycles)
    session.close()

    assert observed.busy_before, (
        f"control failed: the producer's stream finished its {SLEEP_MS} ms of "
        "queued GPU work before the run, so the result would say nothing. Check "
        "that CUDA_LAUNCH_BLOCKING is unset; calibrated cycles: "
        f"{gpu.sleep_cycles}"
    )
    assert not observed.busy_after, "the run returned before its stream reached it"
    expected = _crop_with_row("cpu").tensor_image + 1
    assert torch.equal(observed.copied.tensor_image.cpu(), expected)
    assert observed.copied.device.type == _expected_device("cuda", target)
    assert observed.copied.tensor_image.data_ptr() != observed.source_ptr

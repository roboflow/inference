"""The video sources run in real sessions over fake producers.

``FakeProducer`` stands in for a decoder; VideoSource calls the factory
registered in the ``video_producer_factories`` resource instead of choosing a
decoder. No webcam, network or GPU is used. Frames are small CPU tensors (or
OpenCV-style BGR arrays) whose pixel values encode camera and frame number::

    tensor[0] = camera value      tensor[1] = frame number % 256

Every wait is bounded by ``WAIT``; every run is finished before a test ends.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError
from roboflow_workflows.execution_engine.v2.kinds import INTEGER_KIND
from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions
from roboflow_workflows.execution_engine.v2.sources import ENGINE_CLOCK_ID
from roboflow_workflows.execution_engine.v2.state import ManagedState
from streamvision.camera.entities import SourceProperties, VideoFrameProducer
from streamvision.workflows_v2 import (
    WORKFLOWS_V2_SOURCES,
    VideoSourceFailure,
    VideoStatistics,
    VideoStopError,
    VideoStream,
    VideoStreamSet,
)
from streamvision.workflows_v2.reader import (
    END_OF_STREAM,
    FrameCollector,
    MemberReader,
)

WAIT = 10.0
"""Upper bound of every wait, in seconds."""

MODES = {
    "serial": {},
    "pipelined": {"pipeline": PipelineOptions(max_in_flight=2)},
}


class FakeProducer(VideoFrameProducer):
    """A decoder producing ``frames`` frames (``None``: until interrupted).

    Args:
        value: Camera value written into channel 0.
        frames: Number of frames before ``grab`` returns False.
        period: Seconds per ``grab``.
        is_file: Reported ``SourceProperties.is_file``.
        array: Deliver ``(H, W, 3)`` BGR arrays instead of CHW RGB tensors.
        fail_after: ``grab`` raises after this many frames.
        gate: ``grab`` blocks until the gate or an interrupt is set.
        ignore_interrupt: ``grab`` keeps blocking on the gate after an interrupt.
    """

    def __init__(
        self,
        value: int,
        *,
        frames: Optional[int] = None,
        period: float = 0.002,
        is_file: bool = True,
        array: bool = False,
        fail_after: Optional[int] = None,
        gate: Optional[threading.Event] = None,
        ignore_interrupt: bool = False,
    ) -> None:
        self.value = value
        self.frames = frames
        self.period = period
        self.is_file = is_file
        self.array = array
        self.fail_after = fail_after
        self.gate = gate
        self.ignore_interrupt = ignore_interrupt
        self.count = 0
        self.interrupted = threading.Event()
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def grab(self) -> bool:
        if self.gate is not None:
            while not self.gate.wait(0.005):
                if self.interrupted.is_set() and not self.ignore_interrupt:
                    return False
        if self.interrupted.is_set():
            return False
        if self.fail_after is not None and self.count >= self.fail_after:
            raise RuntimeError("decoder broke")
        if self.frames is not None and self.count >= self.frames:
            return False

        self.interrupted.wait(self.period)
        self.count += 1

        return True

    def retrieve(self):
        if self.array:
            pixels = np.zeros((4, 6, 3), dtype=np.uint8)
            pixels[..., 2] = self.value  # BGR: red is the last channel
            pixels[..., 1] = self.count % 256

            return True, pixels

        pixels = torch.zeros((3, 4, 6), dtype=torch.uint8)
        pixels[0] = self.value
        pixels[1] = self.count % 256

        return True, pixels

    def discover_source_properties(self) -> SourceProperties:
        properties = SourceProperties(
            width=6,
            height=4,
            total_frames=self.frames or -1,
            is_file=self.is_file,
            fps=30.0,
        )

        return properties

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        pass

    def connection_error_message(self) -> str:
        return "fake connection error"

    def interrupt(self) -> None:
        self.interrupted.set()

    def release(self) -> None:
        self.released = True


class PerMember(Block):
    """One call per camera frame; counts calls in the camera's own state."""

    type = "test/per_member@v1"
    outputs = {"seen": Output(INTEGER_KIND)}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)

    def __init__(self, *, managed_state: ManagedState) -> None:
        self.state = managed_state

    def run(self, *, image) -> dict:
        seen = self.state.source.incr("calls")

        return {"seen": seen}


CATALOGUE = Catalogue([PerMember], sources=list(WORKFLOWS_V2_SOURCES))


def factory(**settings: Any) -> Callable[[], FakeProducer]:
    def build() -> FakeProducer:
        return FakeProducer(**settings)

    return build


def failing_factory() -> FakeProducer:
    raise ConnectionError("camera unreachable")


def definition(source: Dict[str, Any], *, per_member: bool = False) -> dict:
    image = f"$sources.{source['name']}.image"
    fields = [{"type": "JsonField", "name": "image", "selector": image}]
    steps = []
    if per_member:
        steps = [{"type": PerMember.type, "name": "member", "image": image}]
        fields.append(
            {"type": "JsonField", "name": "seen", "selector": "$steps.member.seen"}
        )
    workflow = {
        "version": "2.0",
        "inputs": [],
        "sources": [source],
        "steps": steps,
        "outputs": [
            {
                "type": "OutputGroup",
                "name": "frames",
                "anchor": image,
                "outputs": fields,
            }
        ],
    }

    return workflow


def run_session(
    source: Dict[str, Any],
    factories: Dict[str, Callable[[], FakeProducer]],
    *,
    statistics: Optional[VideoStatistics] = None,
    per_member: bool = False,
    stop_after: Optional[int] = None,
    **options: Any,
) -> Dict[str, Any]:
    """Run to the end (or ``run.stop()`` after ``stop_after`` results).

    Returns:
        ``results`` delivered, ``failure`` (``ActiveRunError`` or ``None``).
    """
    plan = compile_workflow(
        definition(source, per_member=per_member), catalogue=CATALOGUE
    )
    resources: Dict[str, Any] = {"video_producer_factories": factories}
    if statistics is not None:
        resources["video_statistics"] = statistics
    if per_member:
        resources["managed_state"] = ManagedState()
    session = plan.create_session(resources)
    results: List[Any] = []
    enough = threading.Event()

    def handle(result: Any) -> None:
        results.append(result)
        if stop_after is not None and len(results) >= stop_after:
            enough.set()

    run = session.start(handlers={"frames": handle}, **options)
    failure = None
    try:
        if stop_after is not None:
            enough.wait(WAIT)
            run.stop()
        run.wait(WAIT)
    except ActiveRunError as error:
        failure = error
    finally:
        if not run.done:
            run.cancel()
            try:
                run.wait(WAIT)
            except ActiveRunError:
                pass
    assert run.done

    return {"results": results, "failure": failure}


def assert_accounting(statistics: VideoStatistics) -> None:
    """Every frame read from VideoSource is accounted for exactly once."""
    for member in statistics.members().values():
        assert member.read == (
            member.discarded + member.evicted + member.consumed + member.cleared
        ), member


def video_threads() -> List[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(("video-member-", "video-terminate-"))
    ]


# video/stream@v1 -----------------------------------------------------------


@pytest.mark.parametrize("mode", list(MODES))
def test_stream_delivers_tensor_images_with_member_identity_and_timing(mode) -> None:
    statistics = VideoStatistics()

    outcome = run_session(
        {"type": VideoStream.type, "name": "cam", "reference": "file-0"},
        {"file-0": factory(value=7, frames=6, period=0.01)},
        statistics=statistics,
        **MODES[mode],
    )

    assert outcome["failure"] is None
    results = outcome["results"]
    assert results
    frame_numbers = []
    for result in results:
        image = result.outputs.data["image"]
        assert isinstance(image, ImageData)
        assert image.tensor_image.device.type == "cpu"
        assert tuple(image.tensor_image.shape) == (3, 4, 6)
        assert int(image.tensor_image[0, 0, 0]) == 7
        assert image.parent.frame_size_hw == image.root.frame_size_hw == (4, 6)
        metadata = image.video_metadata
        assert metadata.video_identifier == "cam"
        assert metadata.comes_from_video_file is True
        assert int(image.tensor_image[1, 0, 0]) == metadata.frame_number
        frame_numbers.append(metadata.frame_number)
        sample = result.outputs.metadata["image"].sample_at(())
        temporal = result.outputs.metadata["image"].temporal_at(())
        assert (sample.source_id, sample.source_type) == ("cam", VideoStream.type)
        assert temporal.observed_coverage.clock_id == ENGINE_CLOCK_ID
        assert temporal.capture_coverage is None
        assert temporal.media_coverage is None
    assert frame_numbers == sorted(set(frame_numbers))

    member = statistics.members()["cam"]
    assert (member.end_reason, member.closed, member.close_errors) == (
        "end_of_stream",
        True,
        (),
    )
    assert member.is_file is True
    assert member.consumed == len(results)
    assert member.grabbed == 6
    assert_accounting(statistics)
    assert not video_threads()


def test_stream_converts_opencv_bgr_arrays_to_rgb_cpu_tensors() -> None:
    outcome = run_session(
        {"type": VideoStream.type, "name": "cam", "reference": "file-0"},
        {"file-0": factory(value=9, frames=2, period=0.01, array=True)},
    )

    assert outcome["failure"] is None
    image = outcome["results"][0].outputs.data["image"]
    assert image.tensor_image.dtype == torch.uint8
    assert tuple(image.tensor_image.shape) == (3, 4, 6)
    assert int(image.tensor_image[0, 0, 0]) == 9  # red first
    assert int(image.tensor_image[2, 0, 0]) == 0


def test_live_stream_end_is_a_failure_not_an_end_of_stream() -> None:
    statistics = VideoStatistics()

    outcome = run_session(
        {"type": VideoStream.type, "name": "cam", "reference": "live-0"},
        {"live-0": factory(value=1, frames=3, is_file=False)},
        statistics=statistics,
    )

    failure = outcome["failure"]
    assert failure is not None and failure.stage == "read"
    assert isinstance(failure.__cause__, VideoSourceFailure)
    assert failure.__cause__.reason == "live_stream_ended"
    assert statistics.members()["cam"].end_reason == "live_stream_ended"
    assert not video_threads()


# video/stream_set@v1 -------------------------------------------------------


@pytest.mark.parametrize("mode", list(MODES))
def test_stream_set_emits_partial_batches_with_stable_sparse_indices(mode) -> None:
    statistics = VideoStatistics()
    factories = {
        "fast-0": factory(value=10, frames=12, period=0.004),
        "slow-1": factory(value=11, frames=3, period=0.05),
        "fast-2": factory(value=12, frames=12, period=0.004),
    }

    outcome = run_session(
        {
            "type": VideoStreamSet.type,
            "name": "cams",
            "references": list(factories),
            "batch_size": 2,
        },
        factories,
        statistics=statistics,
        per_member=True,
        **MODES[mode],
    )

    assert outcome["failure"] is None
    results = outcome["results"]
    seen_per_camera: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    frames_per_camera: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for result in results:
        images = result.outputs.data["image"]
        assert isinstance(images, Batch)
        indices = images.indices
        assert 1 <= len(indices) <= 2
        assert len(set(indices)) == len(indices)
        metadata = result.outputs.metadata["image"]
        pulse = metadata.temporal_at(())
        for (camera,), image in images.iter_with_indices():
            member_id = f"cams/{camera}"
            assert int(image.tensor_image[0, 0, 0]) == 10 + camera
            assert image.video_metadata.video_identifier == member_id
            sample = metadata.sample_at((camera,))
            assert sample.source_id == member_id
            assert sample.source_metadata == {"stream": camera}
            member = metadata.temporal_at((camera,))
            assert member.observed_coverage.clock_id == ENGINE_CLOCK_ID
            assert member.observed_coverage.ticks <= pulse.observed_coverage.ticks
            assert member.capture_coverage is None and member.media_coverage is None
            frames_per_camera[camera].append(image.video_metadata.frame_number)
        for (camera,), seen in result.outputs.data["seen"].iter_with_indices():
            seen_per_camera[camera].append(seen)

    # The slow camera never held back the others: pulses without it exist.
    assert any((1,) not in r.outputs.data["image"].indices for r in results)
    for camera in range(3):
        assert frames_per_camera[camera], camera
        assert frames_per_camera[camera] == sorted(set(frames_per_camera[camera]))
        # Per-member state: each camera counts only its own frames.
        assert seen_per_camera[camera] == list(
            range(1, len(frames_per_camera[camera]) + 1)
        )
    members = statistics.members()
    assert list(members) == ["cams/0", "cams/1", "cams/2"]
    for camera, member in enumerate(members.values()):
        assert member.end_reason == "end_of_stream"
        assert member.consumed == len(frames_per_camera[camera])
        assert member.closed
    assert_accounting(statistics)
    assert not video_threads()


def test_stream_set_with_batch_size_one_serves_every_camera() -> None:
    factories = {
        f"cam-{index}": factory(value=index, frames=None, period=0.002)
        for index in range(3)
    }

    outcome = run_session(
        {
            "type": VideoStreamSet.type,
            "name": "cams",
            "references": list(factories),
            "batch_size": 1,
        },
        factories,
        stop_after=30,
    )

    assert outcome["failure"] is None
    indices = [r.outputs.data["image"].indices for r in outcome["results"]]
    assert all(len(pulse) == 1 for pulse in indices)
    assert {pulse[0] for pulse in indices} == {(0,), (1,), (2,)}
    assert not video_threads()


@pytest.mark.parametrize("mode", list(MODES))
def test_member_error_fails_the_run_and_stops_every_member(mode) -> None:
    statistics = VideoStatistics()
    factories = {
        "broken-0": factory(value=0, fail_after=3),
        "live-1": factory(value=1, frames=None, is_file=False),
    }

    outcome = run_session(
        {"type": VideoStreamSet.type, "name": "cams", "references": list(factories)},
        factories,
        statistics=statistics,
        **MODES[mode],
    )

    failure = outcome["failure"]
    assert failure is not None and failure.stage == "read"
    cause = failure.__cause__
    assert isinstance(cause, VideoSourceFailure)
    assert (cause.member_id, cause.reason) == ("cams/0", "source_error")
    assert "decoder broke" in cause.detail
    members = statistics.members()
    assert members["cams/0"].end_reason == "source_error"
    assert members["cams/1"].end_reason == "stopped"
    assert all(member.closed for member in members.values())
    assert_accounting(statistics)
    assert not video_threads()


def test_end_member_policy_keeps_other_cameras_and_records_reasons() -> None:
    statistics = VideoStatistics()
    factories = {
        "broken-0": factory(value=0, fail_after=2),
        "file-1": factory(value=1, frames=8, period=0.005),
        "missing-2": failing_factory,
    }

    outcome = run_session(
        {
            "type": VideoStreamSet.type,
            "name": "cams",
            "references": list(factories),
            "on_member_failure": "end_member",
        },
        factories,
        statistics=statistics,
    )

    assert outcome["failure"] is None
    delivered = {
        index
        for result in outcome["results"]
        for index in result.outputs.data["image"].indices
    }
    assert (1,) in delivered and (2,) not in delivered
    members = statistics.members()
    assert members["cams/0"].end_reason == "source_error"
    assert members["cams/1"].end_reason == "end_of_stream"
    assert members["cams/2"].end_reason == "open_failed"
    assert "camera unreachable" in members["cams/2"].end_detail
    assert_accounting(statistics)
    assert not video_threads()


def test_end_any_ends_the_set_at_the_first_member_end() -> None:
    statistics = VideoStatistics()
    factories = {
        "short-0": factory(value=0, frames=2, period=0.005),
        "live-1": factory(value=1, frames=None, is_file=False),
    }

    outcome = run_session(
        {
            "type": VideoStreamSet.type,
            "name": "cams",
            "references": list(factories),
            "end": "any",
        },
        factories,
        statistics=statistics,
    )

    assert outcome["failure"] is None
    members = statistics.members()
    assert members["cams/0"].end_reason == "end_of_stream"
    assert members["cams/1"].end_reason == "stopped"
    assert not video_threads()


@pytest.mark.parametrize("policy", ["fail", "end_member"])
def test_stalled_camera_does_not_block_batches_and_expires(policy) -> None:
    statistics = VideoStatistics()
    never = threading.Event()
    factories = {
        "file-0": factory(value=0, frames=40, period=0.01),
        "stalled-1": factory(value=1, gate=never, is_file=False),
    }

    outcome = run_session(
        {
            "type": VideoStreamSet.type,
            "name": "cams",
            "references": list(factories),
            "stall_timeout": 0.15,
            "on_member_failure": policy,
        },
        factories,
        statistics=statistics,
    )

    results = outcome["results"]
    assert results, "camera 0 must be delivered while camera 1 is silent"
    assert all(r.outputs.data["image"].indices == ((0,),) for r in results)
    members = statistics.members()
    assert members["cams/1"].end_reason == "stalled"
    if policy == "fail":
        assert isinstance(outcome["failure"].__cause__, VideoSourceFailure)
        assert outcome["failure"].__cause__.reason == "stalled"
        assert members["cams/0"].end_reason == "stopped"
    else:
        assert outcome["failure"] is None
        assert members["cams/0"].end_reason == "end_of_stream"
    assert all(member.closed for member in members.values())
    assert not video_threads()


def test_stop_drains_and_reports_a_clean_close() -> None:
    statistics = VideoStatistics()
    factories = {
        f"live-{index}": factory(value=index, frames=None, is_file=False)
        for index in range(2)
    }

    outcome = run_session(
        {"type": VideoStreamSet.type, "name": "cams", "references": list(factories)},
        factories,
        statistics=statistics,
        stop_after=3,
    )

    assert outcome["failure"] is None
    for member in statistics.members().values():
        assert (member.end_reason, member.closed) == ("stopped", True)
    assert_accounting(statistics)
    assert not video_threads()


def test_close_timeout_fails_the_run_and_reports_the_lingering_thread() -> None:
    statistics = VideoStatistics()
    hang = threading.Event()
    factories = {
        "live-0": factory(value=0, frames=None, is_file=False),
        "hung-1": factory(value=1, gate=hang, ignore_interrupt=True, is_file=False),
    }

    try:
        outcome = run_session(
            {
                "type": VideoStreamSet.type,
                "name": "cams",
                "references": list(factories),
                "stop_timeout": 0.2,
            },
            factories,
            statistics=statistics,
            stop_after=2,
        )

        failure = outcome["failure"]
        assert failure is not None and failure.stage == "close"
        assert isinstance(failure.__cause__, VideoStopError)
        members = statistics.members()
        assert members["cams/0"].closed is True
        assert members["cams/1"].closed is False
        assert any("still alive" in error for error in members["cams/1"].close_errors)
    finally:
        hang.set()
    deadline = time.monotonic() + WAIT
    while video_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not video_threads()


def test_statistics_resource_is_optional() -> None:
    outcome = run_session(
        {"type": VideoStream.type, "name": "cam", "reference": "file-0"},
        {"file-0": factory(value=1, frames=2, period=0.01)},
    )

    assert outcome["failure"] is None and outcome["results"]


# Reader core ---------------------------------------------------------------


class Tracked:
    """A handoff payload that records whether it died under the shared lock."""

    def __init__(self, condition: threading.Condition, deaths: List[bool]) -> None:
        self.condition = condition
        self.deaths = deaths

    def __del__(self) -> None:
        self.deaths.append(self.condition._is_owned())


def test_evicted_and_cleared_frames_die_outside_the_shared_lock() -> None:
    condition = threading.Condition()
    deaths: List[bool] = []
    reader = MemberReader(
        "m",
        reference=factory(value=0, frames=None, period=0.001, is_file=False),
        condition=condition,
        convert=lambda frame, acquired: Tracked(condition, deaths),
        handoff_size=2,
    )
    collector = FrameCollector([reader], condition=condition, batch_size=1)
    collector.start()
    deadline = time.monotonic() + WAIT
    while reader.statistics().evicted < 5 and time.monotonic() < deadline:
        time.sleep(0.005)
        with condition:
            assert len(reader.handoff) <= 2

    collector.close()

    stats = reader.statistics()
    assert stats.evicted >= 5 and stats.cleared >= 1
    assert (
        stats.read == stats.discarded + stats.evicted + stats.consumed + stats.cleared
    )
    assert deaths and not any(deaths)


def test_collector_rotates_ready_members_and_takes_one_frame_each() -> None:
    condition = threading.Condition()
    readers = [
        MemberReader(
            f"m/{index}",
            reference=factory(value=index),
            condition=condition,
            convert=lambda frame, acquired: frame,
        )
        for index in range(3)
    ]
    for reader in readers:
        reader.handoff.extend([f"{reader.member_id}:a", f"{reader.member_id}:b"])
    collector = FrameCollector(
        readers, condition=condition, batch_size=2, collection_timeout=0.0
    )

    batches = [collector.collect() for _ in range(3)]

    assert batches == [
        [(0, "m/0:a"), (1, "m/1:a")],
        [(2, "m/2:a"), (0, "m/0:b")],
        [(1, "m/1:b"), (2, "m/2:b")],
    ]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_capacities_must_be_positive_ints_on_direct_construction(value) -> None:
    for name in ("buffer_size", "handoff_size"):
        with pytest.raises(ValueError):
            MemberReader(
                "m",
                reference=factory(value=0),
                condition=threading.Condition(),
                convert=lambda frame, acquired: frame,
                **{name: value},
            )


def test_batch_size_cannot_exceed_the_number_of_streams() -> None:
    condition = threading.Condition()
    reader = MemberReader(
        "m", reference=factory(value=0), condition=condition, convert=lambda f, a: f
    )

    with pytest.raises(ValueError):
        FrameCollector([reader], condition=condition, batch_size=2)


@pytest.mark.parametrize(
    "params",
    [
        {"buffer_size": True},
        {"handoff_size": 0},
        {"batch_size": 1.5},
        {"collection_timeout": -0.1},
        {"stall_timeout": float("inf")},
        {"end": "first"},
        {"decoder": "ffmpeg"},
    ],
)
def test_invalid_set_parameters_are_rejected_at_compile_time(params) -> None:
    source = {
        "type": VideoStreamSet.type,
        "name": "cams",
        "references": ["a", "b"],
        **params,
    }

    with pytest.raises(Exception) as caught:
        compile_workflow(definition(source), catalogue=CATALOGUE)

    assert not isinstance(caught.value, AssertionError)


# Review corrections: end policy, stop, release failures, late terminator ---


class SignallingCondition(threading.Condition):
    """A condition that sets ``waiting`` whenever a thread waits on it."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.waiting.set()

        return super().wait(timeout)


def readers_for(condition: threading.Condition, count: int) -> List[MemberReader]:
    readers = [
        MemberReader(
            f"m/{index}",
            reference=factory(value=index),
            condition=condition,
            convert=lambda frame, acquired: frame,
        )
        for index in range(count)
    ]

    return readers


def collect_in_thread(collector: FrameCollector) -> Dict[str, Any]:
    outcome: Dict[str, Any] = {}

    def collect() -> None:
        try:
            outcome["batch"] = collector.collect()
        except Exception as error:  # noqa: BLE001 - asserted by the test
            outcome["error"] = error
        outcome["returned_at"] = time.monotonic()

    outcome["thread"] = threading.Thread(target=collect)
    outcome["thread"].start()

    return outcome


def test_end_any_ends_while_another_member_has_a_frame_ready() -> None:
    condition = threading.Condition()
    ended, survivor = readers_for(condition, 2)
    ended._end(END_OF_STREAM, "the video file ended")
    survivor.handoff.append("ready")
    collector = FrameCollector(
        [ended, survivor], condition=condition, batch_size=1, end="any"
    )

    assert collector.collect() is None
    assert list(survivor.handoff) == ["ready"]


def test_end_any_ends_when_a_member_ends_during_collection() -> None:
    condition = SignallingCondition()
    ready, late = readers_for(condition, 2)
    ready.handoff.append("ready")
    collector = FrameCollector(
        [ready, late],
        condition=condition,
        batch_size=2,
        collection_timeout=WAIT,
        end="any",
    )

    outcome = collect_in_thread(collector)
    assert condition.waiting.wait(WAIT)
    late._end(END_OF_STREAM, "the video file ended")
    outcome["thread"].join(WAIT)

    assert not outcome["thread"].is_alive()
    assert outcome == {**outcome, "batch": None}


def test_end_any_keeps_failure_precedence() -> None:
    condition = threading.Condition()
    ended, failed = readers_for(condition, 2)
    ended._end(END_OF_STREAM, "the video file ended")
    failed.fail("source_error", "decoder broke")
    collector = FrameCollector(
        [ended, failed], condition=condition, batch_size=1, end="any"
    )

    with pytest.raises(VideoSourceFailure) as caught:
        collector.collect()

    assert caught.value.reason == "source_error"


def test_stop_ends_a_partial_batch_despite_a_long_collection_timeout() -> None:
    condition = SignallingCondition()
    ready, silent = readers_for(condition, 2)
    ready.handoff.append("ready")
    stop = threading.Event()
    collector = FrameCollector(
        [ready, silent],
        condition=condition,
        batch_size=2,
        collection_timeout=WAIT,
        stop_event=stop,
    )

    outcome = collect_in_thread(collector)
    assert condition.waiting.wait(WAIT)
    stopped_at = time.monotonic()
    stop.set()  # The engine does not notify the shared condition.
    outcome["thread"].join(WAIT)

    assert not outcome["thread"].is_alive()
    assert outcome["batch"] is None
    # Bounded by the poll interval; the generous bound absorbs scheduling.
    assert outcome["returned_at"] - stopped_at < 1.0


class ReleaseFails(FakeProducer):
    def release(self) -> None:
        super().release()
        raise RuntimeError("release broke")


class NeverOpens(ReleaseFails):
    def isOpened(self) -> bool:
        return False


class DiscoveryFails(ReleaseFails):
    def discover_source_properties(self) -> SourceProperties:
        raise RuntimeError("no properties")


@pytest.mark.parametrize(
    "producer, end_reason",
    [
        (ReleaseFails(0, frames=1), "end_of_stream"),
        (NeverOpens(0), "open_failed"),
        (DiscoveryFails(0), "open_failed"),
    ],
    ids=["end_of_file", "not_opened", "discovery_failed"],
)
def test_injected_producer_release_failure_fails_close(producer, end_reason) -> None:
    condition = threading.Condition()
    reader = MemberReader(
        "m",
        reference=lambda: producer,
        condition=condition,
        convert=lambda frame, acquired: frame,
    )
    collector = FrameCollector([reader], condition=condition, batch_size=1)
    collector.start()
    deadline = time.monotonic() + WAIT
    while reader.statistics().end_reason is None and time.monotonic() < deadline:
        time.sleep(0.005)

    with pytest.raises(VideoStopError, match="release broke"):
        collector.close()

    stats = reader.statistics()
    assert (stats.end_reason, stats.closed, stats.decoder) == (
        end_reason,
        False,
        "factory",
    )
    assert stats.close_errors == (
        "producer release failed: RuntimeError: release broke",
    )
    assert not video_threads()


def test_forced_decoder_release_failure_fails_close(monkeypatch, tmp_path) -> None:
    from streamvision.camera import video_source

    def broken_release(self) -> None:
        raise RuntimeError("cv2 release broke")

    monkeypatch.setattr(video_source.CV2VideoFrameProducer, "release", broken_release)
    condition = threading.Condition()
    reader = MemberReader(
        "m",
        reference=str(tmp_path / "missing.mp4"),
        decoder="opencv",
        condition=condition,
        convert=lambda frame, acquired: frame,
    )
    collector = FrameCollector([reader], condition=condition, batch_size=1)
    collector.start()

    with pytest.raises(VideoSourceFailure):
        while collector.collect() is not None:
            pass
    with pytest.raises(VideoStopError, match="cv2 release broke"):
        collector.close()

    stats = reader.statistics()
    assert (stats.end_reason, stats.closed, stats.decoder) == (
        "open_failed",
        False,
        "opencv",
    )


def test_close_joins_a_terminator_started_while_close_waits_for_the_member() -> None:
    joining = threading.Event()
    opening = threading.Event()
    release_terminator = threading.Event()

    class SlowOpening(FakeProducer):
        def discover_source_properties(self) -> SourceProperties:
            opening.set()
            joining.wait(WAIT)

            return super().discover_source_properties()

    class HeldTerminator(MemberReader):
        def _terminate(self) -> None:
            release_terminator.wait(WAIT)
            super()._terminate()

    condition = threading.Condition()
    reader = HeldTerminator(
        "m",
        reference=lambda: SlowOpening(0, frames=0),
        condition=condition,
        convert=lambda frame, acquired: frame,
    )
    collector = FrameCollector(
        [reader], condition=condition, batch_size=1, stop_timeout=0.3
    )
    collector.start()
    assert opening.wait(WAIT)
    # Opening finishes only once close waits for the member thread, so the
    # member starts the terminator during that join.
    join_member = reader._thread.join

    def join(timeout: Optional[float] = None) -> None:
        joining.set()
        join_member(timeout)

    reader._thread.join = join

    try:
        with pytest.raises(VideoStopError, match="terminator thread still alive"):
            collector.close()

        stats = reader.statistics()
        assert stats.closed is False
        assert stats.close_errors == (
            "terminator thread still alive after the stop timeout",
        )
    finally:
        release_terminator.set()
    deadline = time.monotonic() + WAIT
    while video_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not video_threads()


def test_injected_factory_is_reported_with_the_sanitized_declared_reference() -> None:
    statistics = VideoStatistics()
    reference = "rtsp://user:secret@cam-0/stream"

    outcome = run_session(
        {"type": VideoStream.type, "name": "cam", "reference": reference},
        {reference: factory(value=1, frames=2, period=0.01)},
        statistics=statistics,
    )

    assert outcome["failure"] is None
    member = statistics.members()["cam"]
    assert member.decoder == "factory"
    assert "cam-0" in member.reference and "secret" not in member.reference


@pytest.mark.parametrize(
    "producer, end_reason",
    [
        (ReleaseFails(0, frames=1), "end_of_stream"),
        (NeverOpens(0), "open_failed"),
        (DiscoveryFails(0), "open_failed"),
    ],
    ids=["end_of_file", "not_opened", "discovery_failed"],
)
def test_auto_decoder_release_failure_fails_close(
    monkeypatch, producer, end_reason
) -> None:
    from streamvision.camera import video_source

    monkeypatch.setattr(
        video_source, "_build_default_producer", lambda *args, **kwargs: producer
    )
    # Model the final selected CV2 producer, so startup failure has no fallback.
    monkeypatch.setattr(video_source, "CV2VideoFrameProducer", type(producer))
    condition = threading.Condition()
    reader = MemberReader(
        "m",
        reference="fake.mp4",
        decoder="auto",
        condition=condition,
        convert=lambda frame, acquired: frame,
    )
    collector = FrameCollector([reader], condition=condition, batch_size=1)
    collector.start()
    deadline = time.monotonic() + WAIT
    while reader.statistics().end_reason is None and time.monotonic() < deadline:
        time.sleep(0.005)

    with pytest.raises(VideoStopError, match="release broke"):
        collector.close()

    stats = reader.statistics()
    assert (stats.end_reason, stats.closed, stats.decoder) == (
        end_reason,
        False,
        "auto",
    )
    assert stats.close_errors == (
        "producer release failed: RuntimeError: release broke",
    )
    assert not video_threads()

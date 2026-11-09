"""Tests for the NVENC ring.

The ring logic is tested with a fake ``NvencAdapter`` (no CUDA, no PyNvVideoCodec)
on top of the real ``ring.BaseEncodedRing``. The real-encoder tests at the bottom
run only where ``nvenc_ring.available()`` is true (CUDA GPU + PyNvVideoCodec).
"""

import io
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from typing import Any, List, Optional

import av
import pytest
from roboflow_workflows.core_steps.common.video_clips import nvenc_ring
from roboflow_workflows.core_steps.common.video_clips.artifact import ClipArtifact
from roboflow_workflows.core_steps.common.video_clips.capture import (
    PARTIAL_SHUTDOWN,
    PendingCapture,
)
from roboflow_workflows.core_steps.common.video_clips.ring import (
    EncodedOutput,
    EncoderStartingError,
    PreparedFrame,
)

requires_nvenc = pytest.mark.skipif(
    not nvenc_ring.available(), reason="needs a CUDA GPU and PyNvVideoCodec"
)

IDR_ACCESS_UNIT = (
    b"\x00\x00\x00\x01\x67\x42\x00\x00\x00\x01\x68\xce\x00\x00\x01\x65\x88"
)
P_ACCESS_UNIT = b"\x00\x00\x00\x01\x41\x9a\x02"
T0 = datetime(2026, 10, 30, 12, 0, 0)


# ---- fakes ------------------------------------------------------------------------
@dataclass
class FakeTensor:
    width: int
    height: int
    device: str = "cuda:0"


class FakeImage:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.tensor_image = FakeTensor(width=width, height=height)

    def is_tensor_materialised(self) -> bool:
        return True


class FakeSession:
    def __init__(self, index: int, width: int, height: int):
        self.index = index
        self.width = width
        self.height = height
        self.pending: deque = deque()
        self.flushed = False


class FakeAdapter:
    """Records every call and the thread it ran on; encode can lag or block."""

    def __init__(
        self,
        *,
        delay_frames: int = 0,
        encode_sleep: float = 0.0,
        gate: Optional[threading.Event] = None,
        fail_encodes: int = 0,
        raw_payloads: bool = False,
        open_gate: Optional[threading.Event] = None,
    ):
        self.open_gate = open_gate
        self.delay_frames = delay_frames
        self.encode_sleep = encode_sleep
        self.gate = gate
        self.fail_encodes = fail_encodes
        self.raw_payloads = raw_payloads
        self.lock = threading.Lock()
        self.slots_allocated = 0
        self.allocations = 0
        self.staged: List[Any] = []
        self.sessions: List[FakeSession] = []
        self.destroyed: List[FakeSession] = []
        self.encoded: List[tuple] = []
        self.calls: List[str] = []
        self.memory_releases = 0
        self.threads = {
            "stage": set(),
            "open": set(),
            "encode": set(),
            "destroy": set(),
        }

    def tensor_device(self, tensor):
        return tensor.device

    def frame_size(self, tensor):
        return nvenc_ring.encoded_size(tensor.width, tensor.height)

    def staging_size(self, tensor):
        return tensor.width, tensor.height

    def slot_bytes(self, width, height):
        return width * height * 3 // 2

    def scratch_bytes(self, width, height):
        return width * height * 4

    def allocate_slots(self, count, width, height, device):
        with self.lock:
            slots = []
            for _ in range(count):
                self.slots_allocated += 1
                slots.append(("slot", self.slots_allocated, width, height))
            if count:
                self.allocations += 1
            return slots

    def stage(self, tensor, slot, width, height):
        self.threads["stage"].add(threading.get_ident())
        with self.lock:
            self.staged.append(slot)
        return ("ready", slot)

    def open_session(self, width, height, fps, gop, bitrate, device):
        self.threads["open"].add(threading.get_ident())
        if self.open_gate is not None:
            self.open_gate.wait(timeout=10)
        session = FakeSession(index=len(self.sessions), width=width, height=height)
        session.settings = {"fps": fps, "gop": gop, "bitrate": bitrate}
        self.sessions.append(session)
        self.calls.append("open")
        return session

    def encode(self, session, slot, ready, ordinal, force_idr):
        self.threads["encode"].add(threading.get_ident())
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if self.encode_sleep:
            time.sleep(self.encode_sleep)
        if self.fail_encodes > 0:
            self.fail_encodes -= 1
            raise RuntimeError("encode failed")
        assert ready == ("ready", slot)
        self.encoded.append((session.index, ordinal, force_idr))
        session.pending.append((ordinal, force_idr))
        released = []
        while len(session.pending) > self.delay_frames:
            released.append(self._access_unit(*session.pending.popleft()))
        return released

    def flush(self, session):
        self.calls.append("flush")
        session.flushed = True
        released = [self._access_unit(*item) for item in session.pending]
        session.pending.clear()
        return released

    def destroy(self, session):
        self.threads["destroy"].add(threading.get_ident())
        self.calls.append("destroy")
        with self.lock:
            self.destroyed.append(session)

    def session_vram_bytes(self, session):
        return 64 * 1024 * 1024

    def release_cached_memory(self):
        self.threads.setdefault("release", set()).add(threading.get_ident())
        with self.lock:
            self.memory_releases += 1

    def _access_unit(self, ordinal, force_idr):
        if self.raw_payloads:
            return ordinal, b"\xff\xfe" + bytes([ordinal % 256])
        return ordinal, IDR_ACCESS_UNIT if force_idr else P_ACCESS_UNIT


class FakeWarmUp:
    """A ``ProcessWarmUp`` stand-in whose status the test sets."""

    def __init__(self, status: str = nvenc_ring.WARM_UP_PENDING):
        self.state = status
        self.calls = 0

    def status(self, *, start: bool = True) -> str:
        self.calls += 1
        return self.state

    @property
    def error(self):
        return None if self.state != nvenc_ring.WARM_UP_UNAVAILABLE else "no NVENC"

    def report(self):
        return {"status": self.state}


def make_ring(adapter: FakeAdapter, **kwargs: Any):
    kwargs.setdefault("pre_roll_seconds", 5.0)
    ring = nvenc_ring.NvencRing("0", adapter=adapter, **kwargs)
    return ring


def queue_drops(ring) -> int:
    stats = ring.stats()
    if hasattr(stats, "queue_drops"):
        return stats.queue_drops
    return stats.loss["queue_drops"]


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def wait_for_slot(ring) -> bool:
    def has_slot() -> bool:
        stats = ring._backend_stats()
        return stats["pool_free_frames"] > 0 or (
            stats["pool_allocated_frames"] < stats["pool_frames"]
        )

    return wait_for(has_slot)


def timestamp(index: int, fps: float = 30.0) -> datetime:
    return T0 + timedelta(seconds=index / fps)


# ---- availability, settings, helpers ----------------------------------------------
def test_available_false_without_cuda(monkeypatch) -> None:
    import torch

    nvenc_ring.available.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        assert nvenc_ring.available() is False
    finally:
        nvenc_ring.available.cache_clear()


def test_available_false_without_pynvvideocodec(monkeypatch) -> None:
    import torch

    nvenc_ring.available.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)
    try:
        assert nvenc_ring.available() is False
    finally:
        nvenc_ring.available.cache_clear()


def test_available_never_raises_and_is_cached(monkeypatch) -> None:
    import torch

    calls = []

    def broken() -> bool:
        calls.append(1)
        raise RuntimeError("driver exploded")

    nvenc_ring.available.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", broken)
    try:
        assert nvenc_ring.available() is False
        assert nvenc_ring.available() is False
        assert len(calls) == 1
    finally:
        nvenc_ring.available.cache_clear()


def test_ring_refuses_construction_without_nvenc() -> None:
    warm_up = FakeWarmUp(nvenc_ring.WARM_UP_UNAVAILABLE)

    with pytest.raises(RuntimeError):
        nvenc_ring.NvencRing("0", pre_roll_seconds=5.0, warm_up=warm_up)


def test_default_bitrate_scales_with_pixel_rate_and_clamps() -> None:
    assert nvenc_ring.default_bitrate(1920, 1080, 30) == 4_000_000
    assert nvenc_ring.default_bitrate(1280, 720, 30) == 1_777_777
    assert nvenc_ring.default_bitrate(320, 240, 5) == 300_000
    assert nvenc_ring.default_bitrate(3840, 2160, 120) == 20_000_000


def test_resolve_bitrate_prefers_env(monkeypatch) -> None:
    monkeypatch.setenv(nvenc_ring.BITRATE_KBPS_ENV, "2500")
    assert nvenc_ring.resolve_bitrate(1920, 1080, 30) == 2_500_000

    monkeypatch.setenv(nvenc_ring.BITRATE_KBPS_ENV, "1")
    assert nvenc_ring.resolve_bitrate(1920, 1080, 30) == 100_000

    monkeypatch.setenv(nvenc_ring.BITRATE_KBPS_ENV, "fast")
    assert nvenc_ring.resolve_bitrate(1920, 1080, 30) == 4_000_000


def test_resolve_pool_frames(monkeypatch) -> None:
    monkeypatch.delenv(nvenc_ring.POOL_FRAMES_ENV, raising=False)
    assert nvenc_ring.resolve_pool_frames() == 8

    monkeypatch.setenv(nvenc_ring.POOL_FRAMES_ENV, "3")
    assert nvenc_ring.resolve_pool_frames() == 3

    monkeypatch.setenv(nvenc_ring.POOL_FRAMES_ENV, "0")
    assert nvenc_ring.resolve_pool_frames() == 1

    monkeypatch.setenv(nvenc_ring.POOL_FRAMES_ENV, "lots")
    assert nvenc_ring.resolve_pool_frames() == 8


@pytest.mark.parametrize(
    "size, max_width, expected",
    [
        ((1920, 1080), None, (1920, 1080)),
        ((1921, 1081), None, (1920, 1080)),
        ((1920, 1080), 1280, (1280, 720)),
        ((1280, 720), 1920, (1280, 720)),
        ((1000, 501), 640, (640, 320)),
    ],
)
def test_encoded_size(size, max_width, expected) -> None:
    assert nvenc_ring.encoded_size(*size, max_width=max_width) == expected


def test_is_idr_access_unit() -> None:
    assert nvenc_ring.is_idr_access_unit(IDR_ACCESS_UNIT)
    assert not nvenc_ring.is_idr_access_unit(P_ACCESS_UNIT)
    assert not nvenc_ring.is_idr_access_unit(b"")
    assert not nvenc_ring.is_idr_access_unit(b"\x00\x00\x01")


# ---- pool ---------------------------------------------------------------------------
def _open_session(ring) -> None:
    # Open the ring's session on this thread (as the encoder thread would).
    ring._open_session(size=(1920, 1080), device="cuda:0")


def test_pool_fills_then_reports_full_and_recycles_discarded_slots() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=8, startup_pool_frames=0)
    _open_session(ring)

    prepared = [ring._prepare_frame(FakeImage()) for _ in range(8)]
    overflow = ring._prepare_frame(FakeImage())
    stats = ring._backend_stats()

    assert all(isinstance(frame, PreparedFrame) for frame in prepared)
    assert overflow is None
    assert adapter.slots_allocated == 8
    assert stats["pool_drops"] == 1
    assert stats["pool_allocated_frames"] == 8
    assert stats["pool_free_frames"] == 0

    ring._discard_frame(prepared[0])
    ring._discard_frame(prepared[0])  # idempotent: one slot back, not two
    again = ring._prepare_frame(FakeImage())

    assert again is not None
    assert again.payload.slot == ("slot", 1, 1920, 1080)
    assert adapter.slots_allocated == 8
    assert ring._prepare_frame(FakeImage()) is None
    ring._close_encoder()
    ring.close()


def test_full_pool_before_the_session_opens_is_a_startup_refusal() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=2, startup_pool_frames=0)

    ring._prepare_frame(FakeImage())
    ring._prepare_frame(FakeImage())

    with pytest.raises(EncoderStartingError):
        ring._prepare_frame(FakeImage())
    assert ring._backend_stats()["pool_drops"] == 1
    ring.close()


def test_prepared_frame_holds_no_image_and_reports_no_host_bytes() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter)
    image = FakeImage(width=1281, height=721)

    prepared = ring._prepare_frame(image)

    assert (prepared.width, prepared.height) == (1280, 720)
    assert prepared.nbytes == 0
    assert image not in vars(prepared.payload).values()
    assert image.tensor_image not in vars(prepared.payload).values()
    ring.close()


def test_stage_failure_returns_the_slot() -> None:
    adapter = FakeAdapter()
    adapter.stage = lambda *args: (_ for _ in ()).throw(RuntimeError("bad tensor"))
    ring = make_ring(adapter, pool_frames=1, startup_pool_frames=0)

    with pytest.raises(RuntimeError):
        ring._prepare_frame(FakeImage())
    stats = ring._backend_stats()

    assert stats["pool_free_frames"] == 1
    assert stats["stage_errors"] == 1
    ring.close()


def test_resolution_change_resets_the_pool_and_drops_old_slots() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=2, startup_pool_frames=0)
    old = ring._prepare_frame(FakeImage(1920, 1080))

    new = ring._prepare_frame(FakeImage(1280, 720))
    ring._discard_frame(old)
    stats = ring._backend_stats()

    # the old 1080p slot is dropped, not returned to the 720p pool
    assert new.payload.size == (1280, 720)
    assert stats["pool_allocated_frames"] == 2
    assert stats["pool_free_frames"] == 1
    ring.close()


def test_full_pool_drops_frames_while_encoder_is_blocked() -> None:
    gate = threading.Event()
    adapter = FakeAdapter(gate=gate)
    ring = make_ring(adapter, pool_frames=8, startup_pool_frames=0, queue_depth=64)

    ring.submit(timestamp(0), 0, FakeImage())
    assert wait_for(lambda: ring._backend_stats()["sessions_opened"] == 1)
    for index in range(1, 20):
        ring.submit(timestamp(index), index, FakeImage())
    stats = ring._backend_stats()

    # the session is open, so the steady-state pool (8) applies, not the start-up one
    assert adapter.slots_allocated == 8
    assert len(adapter.staged) == 8
    assert stats["pool_drops"] == 12
    assert queue_drops(ring) == 12
    assert ring.stats().startup_dropped == 0
    gate.set()
    ring.close()


# ---- caller thread never encodes ----------------------------------------------------
def test_submit_returns_fast_and_never_encodes_on_the_caller_thread() -> None:
    adapter = FakeAdapter(encode_sleep=0.02)
    ring = make_ring(adapter, pool_frames=8, queue_depth=64)
    durations = []

    for index in range(30):
        started = time.perf_counter()
        ring.submit(timestamp(index), index, FakeImage())
        durations.append(time.perf_counter() - started)
    caller = threading.get_ident()
    ring.close()

    assert max(durations) < 0.01  # one encode sleeps 20 ms
    assert adapter.threads["stage"] == {caller}
    assert caller not in adapter.threads["encode"]
    assert caller not in adapter.threads["open"]
    assert adapter.threads["encode"] == adapter.threads["open"]
    assert adapter.threads["destroy"] == adapter.threads["encode"]


# ---- encoder thread bookkeeping -----------------------------------------------------
def test_encode_frame_maps_delayed_outputs_and_keyframe_flags() -> None:
    adapter = FakeAdapter(delay_frames=2)
    ring = make_ring(adapter, pool_frames=8)
    pattern = [True, False, False, True, False]
    outputs = []

    for force in pattern:
        frame = ring._prepare_frame(FakeImage())
        outputs.extend(ring._encode_frame(frame, pts=0, force_keyframe=force))
    tail = ring._flush_encoder()
    stats = ring._backend_stats()

    assert [output.is_keyframe for output in outputs + tail] == pattern
    assert all(isinstance(output, EncodedOutput) for output in outputs + tail)
    assert [force for _, _, force in adapter.encoded] == pattern
    assert stats["forced_idrs"] == 2
    assert stats["keyframes"] == 2
    # every slot comes back after its encode, so one slot at a time was in use
    assert stats["pool_peak_frames"] == 1
    assert stats["pool_free_frames"] == stats["pool_allocated_frames"]
    ring._close_encoder()
    ring.close()


def test_keyframe_falls_back_to_forced_flag_without_start_codes() -> None:
    adapter = FakeAdapter(raw_payloads=True)
    ring = make_ring(adapter)

    first = ring._encode_frame(
        ring._prepare_frame(FakeImage()), pts=0, force_keyframe=True
    )
    second = ring._encode_frame(
        ring._prepare_frame(FakeImage()), pts=3000, force_keyframe=False
    )

    assert [output.is_keyframe for output in first + second] == [True, False]
    ring._close_encoder()
    ring.close()


def test_session_opens_lazily_with_runtime_rate_settings(monkeypatch) -> None:
    monkeypatch.delenv(nvenc_ring.BITRATE_KBPS_ENV, raising=False)
    adapter = FakeAdapter()
    ring = make_ring(adapter, declared_fps=60.0, clip_fps=15.0, playback_speed=2.0)

    assert adapter.sessions == []
    ring._encode_frame(ring._prepare_frame(FakeImage()), pts=0, force_keyframe=True)
    settings = adapter.sessions[0].settings

    assert settings["fps"] == 15.0
    assert settings["gop"] == 300  # ten output seconds at 15 fps and 2x speed
    assert settings["bitrate"] == nvenc_ring.default_bitrate(1920, 1080, 15.0)
    assert ring._backend_stats()["bitrate"] == settings["bitrate"]
    ring._close_encoder()
    ring.close()


def test_bitrate_argument_overrides_default() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, bitrate=1_234_000)

    ring._encode_frame(ring._prepare_frame(FakeImage()), pts=0, force_keyframe=True)

    assert adapter.sessions[0].settings["bitrate"] == 1_234_000
    assert adapter.sessions[0].settings["fps"] == 30.0
    ring._close_encoder()
    ring.close()


def test_new_size_or_flushed_session_reopens_and_restarts_ordinals() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=4)

    ring._encode_frame(ring._prepare_frame(FakeImage()), pts=0, force_keyframe=True)
    ring._encode_frame(ring._prepare_frame(FakeImage()), pts=1, force_keyframe=False)
    ring._flush_encoder()
    ring._encode_frame(ring._prepare_frame(FakeImage()), pts=2, force_keyframe=True)
    ring._encode_frame(
        ring._prepare_frame(FakeImage(1280, 720)), pts=0, force_keyframe=True
    )
    stats = ring._backend_stats()

    assert [(index, ordinal) for index, ordinal, _ in adapter.encoded] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
    ]
    assert adapter.destroyed == adapter.sessions[:2]
    assert (adapter.sessions[2].width, adapter.sessions[2].height) == (1280, 720)
    assert stats["sessions_opened"] == 3
    assert stats["sessions_open"] == 1
    ring._close_encoder()
    ring.close()


def test_failed_encode_leaves_the_slot_to_discard_exactly_once() -> None:
    adapter = FakeAdapter(fail_encodes=1)
    ring = make_ring(adapter, pool_frames=1)
    frame = ring._prepare_frame(FakeImage())

    with pytest.raises(RuntimeError):
        ring._encode_frame(frame, pts=0, force_keyframe=True)
    ring._discard_frame(frame)
    retry = ring._prepare_frame(FakeImage())
    outputs = ring._encode_frame(retry, pts=0, force_keyframe=True)
    ring._discard_frame(retry)  # already returned by the encode: no second slot

    assert [output.is_keyframe for output in outputs] == [True]
    assert ring._backend_stats()["pool_free_frames"] == 1
    ring._close_encoder()
    ring.close()


def test_close_encoder_without_session_is_a_no_op() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter)

    ring._close_encoder()
    ring._close_encoder()

    assert adapter.destroyed == []
    assert ring._flush_encoder() == []
    ring.close()


def test_backend_stats_report_vram_estimate() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=8, startup_pool_frames=0)

    ring._encode_frame(ring._prepare_frame(FakeImage()), pts=0, force_keyframe=True)
    held = ring._prepare_frame(FakeImage())  # reuses the slot returned by encode
    ring._prepare_frame(FakeImage())
    stats = ring._backend_stats()

    assert held is not None
    # the whole steady pool is allocated with the first frame
    slots = 8 * 1920 * 1080 * 3 // 2
    scratch = 1920 * 1080 * 4
    assert stats["vram_bytes_estimate"] == 64 * 1024 * 1024 + slots + scratch
    assert stats["sessions_open"] == 1
    assert stats["session_open_ms_first"] is not None
    ring._close_encoder()
    # the session's VRAM and its conversion buffers go with it
    assert ring._backend_stats()["vram_bytes_estimate"] == slots
    ring.close()


# ---- IDR cadence through submit -----------------------------------------------------
@pytest.mark.parametrize("speed, expected_idrs", [(1.0, 3), (2.0, 2), (0.5, 6)])
def test_idr_forced_once_per_output_second(speed, expected_idrs) -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, playback_speed=speed, pool_frames=8, queue_depth=128)

    for index in range(90):
        assert wait_for_slot(ring)
        ring.submit(timestamp(index), index, FakeImage())
    assert wait_for(lambda: len(adapter.encoded) == 90)
    ring.close()

    forced = [index for index, (_, _, force) in enumerate(adapter.encoded) if force]
    assert len(forced) == expected_idrs
    assert forced[0] == 0
    assert ring._backend_stats()["forced_idrs"] == expected_idrs


def test_keyframes_reach_the_base_in_submission_order() -> None:
    adapter = FakeAdapter(delay_frames=3)
    ring = make_ring(adapter, pool_frames=8, queue_depth=128)
    try:
        for index in range(60):
            assert wait_for_slot(ring)
            ring.submit(timestamp(index), index, FakeImage())
        assert wait_for(lambda: len(adapter.encoded) == 60)
        handle = ring.capture(timestamp(0), timestamp(59))
        ring.prepare_close(5.0)  # flushes the 3 access units the fake still holds
        artifact = handle.future.result(timeout=10)
        packets = list(ring._store._packets)
    finally:
        ring.close()

    flags = [packet.is_keyframe for packet in packets]
    assert [packet.frame_number for packet in packets] == list(range(60))
    assert [index for index, flag in enumerate(flags) if flag] == [0, 30]
    assert artifact.report.frames == 60
    assert artifact.report.partial == PARTIAL_SHUTDOWN
    artifact.release()


def test_capture_resolves_to_an_mp4_artifact_through_the_real_base() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=8, queue_depth=128)
    try:
        for index in range(45):
            assert wait_for_slot(ring)
            ring.submit(timestamp(index), index, FakeImage())
        handle = ring.capture(timestamp(35), timestamp(50))
        for index in range(45, 60):
            assert wait_for_slot(ring)
            ring.submit(timestamp(index), index, FakeImage())
        # the post-roll is satisfied by frame 51, so no close is needed
        artifact = handle.future.result(timeout=10)
    finally:
        ring.close()

    assert isinstance(handle, PendingCapture)
    assert isinstance(artifact, ClipArtifact)
    assert isinstance(artifact.bytes_or_path, bytes)
    with av.open(io.BytesIO(artifact.bytes_or_path)) as container:
        demuxed = [packet for packet in container.demux(video=0) if packet.size > 0]
    report = artifact.report
    assert report.backend == "nvenc"
    assert report.partial is None
    # GOP-aligned: the window starts on the IDR of frame 30 and ends at frame 50
    assert report.frames == 21
    assert len(demuxed) == 21
    assert demuxed[0].pts == 0 and demuxed[0].is_keyframe
    artifact.release()


# ---- lifecycle ----------------------------------------------------------------------
def test_prepare_close_drops_unencoded_slots_and_flushes_before_destroy() -> None:
    gate = threading.Event()
    adapter = FakeAdapter(gate=gate, delay_frames=2)
    ring = make_ring(adapter, pool_frames=8, startup_pool_frames=0, queue_depth=64)

    for index in range(8):
        ring.submit(timestamp(index), index, FakeImage())
    assert wait_for(lambda: len(adapter.threads["encode"]) == 1)
    ring.prepare_close(5.0)
    stats_after_prepare = ring._backend_stats()
    ring.submit(timestamp(8), 8, FakeImage())
    gate.set()
    ring.close()
    stats = ring._backend_stats()

    # one frame was inside the (blocked) encoder, the other seven were dropped
    assert stats_after_prepare["pool_free_frames"] == 7
    assert len(adapter.staged) == 8
    assert len(adapter.encoded) == 1
    assert queue_drops(ring) >= 7
    assert adapter.calls == ["open", "flush", "destroy"]
    assert stats["sessions_open"] == 0
    assert stats["pool_free_frames"] == 8


def test_prepare_close_and_close_are_idempotent_without_frames() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter)

    ring.prepare_close(None)
    ring.prepare_close(1.0)
    ring.close()
    ring.close()

    assert adapter.sessions == []
    assert adapter.destroyed == []


def test_concurrent_close_destroys_the_session_exactly_once() -> None:
    adapter = FakeAdapter(encode_sleep=0.002)
    ring = make_ring(adapter, pool_frames=8, queue_depth=64)
    for index in range(8):
        ring.submit(timestamp(index), index, FakeImage())
    assert wait_for(lambda: len(adapter.sessions) == 1)
    barrier = threading.Barrier(8)
    errors = []

    def close_ring() -> None:
        try:
            barrier.wait(timeout=5)
            ring.prepare_close(2.0)
            ring.close()
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=close_ring) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert adapter.destroyed == adapter.sessions
    assert adapter.calls.count("destroy") == 1
    assert ring._backend_stats()["sessions_destroyed"] == 1


def test_atexit_hook_closes_rings_left_open() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=8, queue_depth=64)
    for index in range(3):
        ring.submit(timestamp(index), index, FakeImage())
    assert wait_for(lambda: len(adapter.encoded) == 3)

    nvenc_ring._close_live_rings()

    assert adapter.calls[-1] == "destroy"
    assert ring._backend_stats()["sessions_open"] == 0
    assert ring not in set(nvenc_ring._LIVE_RINGS)


# ---- process warm-up -----------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_warm_up_runs_once_in_the_background_and_reports_steps() -> None:
    gate = threading.Event()
    calls = []

    def steps(timings) -> None:
        calls.append(threading.get_ident())
        gate.wait(timeout=10)
        timings["probe_ms"] = 1.5

    warm_up = nvenc_ring.ProcessWarmUp(steps, precheck=lambda: True)

    assert warm_up.status(start=False) == nvenc_ring.WARM_UP_NOT_STARTED
    started = time.perf_counter()
    assert warm_up.status() == nvenc_ring.WARM_UP_PENDING
    warm_up.start()
    assert time.perf_counter() - started < 0.05  # never waits for the steps
    gate.set()
    assert warm_up.wait(5) == nvenc_ring.WARM_UP_READY
    report = warm_up.report()

    assert calls and calls[0] != threading.get_ident()
    assert len(calls) == 1
    assert report["status"] == nvenc_ring.WARM_UP_READY
    assert report["steps_ms"] == {"probe_ms": 1.5}
    assert report["warm_up_ms"] is not None
    assert report["error"] is None


def test_warm_up_without_pynvvideocodec_never_starts_a_thread() -> None:
    calls = []
    warm_up = nvenc_ring.ProcessWarmUp(calls.append, precheck=lambda: False)

    assert warm_up.status() == nvenc_ring.WARM_UP_UNAVAILABLE
    assert warm_up._thread is None
    assert calls == []
    assert "PyNvVideoCodec" in warm_up.error


def test_warm_up_failure_is_unavailable_with_the_error() -> None:
    def steps(timings) -> None:
        raise RuntimeError("no CUDA device")

    warm_up = nvenc_ring.ProcessWarmUp(steps, precheck=lambda: True)

    assert warm_up.wait(5) == nvenc_ring.WARM_UP_UNAVAILABLE
    assert "no CUDA device" in warm_up.error


def test_warm_up_that_overruns_its_bound_becomes_unavailable_for_good() -> None:
    gate = threading.Event()
    clock = FakeClock()
    warm_up = nvenc_ring.ProcessWarmUp(
        lambda timings: gate.wait(timeout=10),
        precheck=lambda: True,
        timeout_seconds=5.0,
        clock=clock,
    )

    assert warm_up.status() == nvenc_ring.WARM_UP_PENDING
    clock.now += 5.5
    assert warm_up.status() == nvenc_ring.WARM_UP_UNAVAILABLE
    gate.set()
    warm_up.wait(5)

    # a late success never flips a decision the recorder already made
    assert warm_up.status() == nvenc_ring.WARM_UP_UNAVAILABLE
    assert "did not finish" in warm_up.error


class _FakeCuda:
    def __init__(self, initialised: bool):
        self.initialised = initialised

    def is_initialized(self) -> bool:
        return self.initialised

    def is_available(self) -> bool:  # pragma: no cover - must never be called
        raise AssertionError("should_warm_up must not touch the CUDA driver")


class _FakeTorch:
    def __init__(self, initialised: bool = False):
        self.cuda = _FakeCuda(initialised)


@dataclass
class _Device:
    type: str


@pytest.mark.parametrize(
    "torch_module, device, installed, expected",
    [
        (None, _Device("cuda"), True, False),  # host never imported torch
        (_FakeTorch(), _Device("cuda"), True, True),
        (_FakeTorch(), "cuda:0", True, True),
        (_FakeTorch(), _Device("cpu"), True, False),
        (_FakeTorch(), None, True, False),
        (_FakeTorch(initialised=True), None, True, True),
        (_FakeTorch(), _Device("cuda"), False, False),
    ],
)
def test_should_warm_up_is_cheap_and_import_free(
    monkeypatch, torch_module, device, installed, expected
) -> None:
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(nvenc_ring, "nvenc_installed", lambda: installed)

    assert nvenc_ring.should_warm_up(device) is expected


@pytest.mark.parametrize(
    "width, height, expected",
    [(1920, 1080, 32), (1280, 720, 32), (3840, 2160, 8), (640, 360, 32)],
)
def test_startup_pool_is_bounded_by_frames_and_device_bytes(
    width, height, expected
) -> None:
    rgb_slot_bytes = 3 * width * height

    assert nvenc_ring.startup_pool_frames(8, rgb_slot_bytes) == expected
    assert nvenc_ring.startup_pool_frames(40, rgb_slot_bytes) == 40


# ---- start-up accounting ---------------------------------------------------------------
def test_frames_before_the_warm_up_are_startup_dropped_without_blocking() -> None:
    adapter = FakeAdapter()
    warm_up = FakeWarmUp()
    ring = make_ring(adapter, warm_up=warm_up, pool_frames=8, queue_depth=64)
    try:
        for index in range(10):
            ring.submit(timestamp(index), index, FakeImage())
        # While the warm-up is pending a submit only reads its status: it never
        # waits on it and never touches the adapter (no slot, no staging).
        pending_status_calls = warm_up.calls
        pending_staged = len(adapter.staged)
        pending_allocated = adapter.slots_allocated
        warm_up.state = nvenc_ring.WARM_UP_READY
        for index in range(10, 40):
            assert wait_for_slot(ring)
            ring.submit(timestamp(index), index, FakeImage())
        assert wait_for(lambda: len(adapter.encoded) == 30)
        handle = ring.capture(timestamp(0), timestamp(39))
        ring.prepare_close(5.0)
        artifact = handle.future.result(timeout=10)
        stats = ring.stats()
    finally:
        ring.close()

    assert pending_status_calls == 10
    assert pending_staged == 0
    assert pending_allocated == 0
    assert len(adapter.staged) == 30  # nothing was staged while pending
    assert stats.startup_dropped == 10
    assert stats.queue_drops == 0
    assert stats.to_dict()["startup_dropped"] == 10
    assert artifact.report.loss.startup_dropped == 10
    assert artifact.report.loss.queue_drops == 0
    assert artifact.report.frames == 30
    assert artifact.report.to_dict()["loss"]["startup_dropped"] == 10
    artifact.release()


def test_unavailable_warm_up_marks_the_ring_failed() -> None:
    warm_up = FakeWarmUp()
    ring = make_ring(FakeAdapter(), warm_up=warm_up)

    ring.submit(timestamp(0), 0, FakeImage())
    assert not ring.startup_failed
    warm_up.state = nvenc_ring.WARM_UP_UNAVAILABLE
    ring.submit(timestamp(1), 1, FakeImage())
    stats = ring.stats()
    ring.close()

    assert ring.startup_failed
    assert stats.startup_dropped == 2
    assert stats.extra["startup_failed"] is True


def test_real_adapter_is_built_only_once_the_warm_up_is_ready(monkeypatch) -> None:
    built = []

    def factory(*, max_width=None):
        built.append(max_width)
        return FakeAdapter()

    monkeypatch.setattr(nvenc_ring, "PyNvVideoCodecAdapter", factory)
    warm_up = FakeWarmUp()
    ring = nvenc_ring.NvencRing(
        "0", pre_roll_seconds=5.0, max_width=1280, warm_up=warm_up
    )
    ring.submit(timestamp(0), 0, FakeImage())
    built_while_pending = list(built)
    warm_up.state = nvenc_ring.WARM_UP_READY
    ring.submit(timestamp(1), 1, FakeImage())
    ring.submit(timestamp(2), 2, FakeImage())
    ring.close()

    assert built_while_pending == []
    assert built == [1280]


def test_session_open_window_uses_the_startup_pool_then_shrinks() -> None:
    open_gate = threading.Event()
    adapter = FakeAdapter(open_gate=open_gate)
    ring = make_ring(adapter, pool_frames=4, startup_pool_frames=10, queue_depth=64)
    try:
        for index in range(14):
            ring.submit(timestamp(index), index, FakeImage())
        opening = ring._backend_stats()
        startup_dropped = ring.stats().startup_dropped
        open_gate.set()
        assert wait_for(lambda: len(adapter.encoded) == 10)
        assert wait_for(lambda: ring._backend_stats()["pool_allocated_frames"] == 4)
        drained = ring._backend_stats()
        stats = ring.stats()
    finally:
        ring.close()

    # 10 slots while the session opens, in two allocations (steady + extension);
    # the 4 frames beyond are start-up loss
    assert opening["pool_allocated_frames"] == 10
    assert opening["pool_free_frames"] == 0
    assert adapter.allocations == 2
    assert opening["startup_pool_frames"] == 10
    assert startup_dropped == 4
    assert stats.queue_drops == 0
    assert drained["pool_peak_frames"] == 10
    assert drained["pool_allocated_frames"] == 4
    assert drained["pool_free_frames"] == 4
    assert drained["session_open_ms_first"] >= 0


def test_steady_state_pool_applies_after_the_startup_backlog_drains() -> None:
    gate = threading.Event()
    gate.set()
    adapter = FakeAdapter(gate=gate)
    ring = make_ring(adapter, pool_frames=4, startup_pool_frames=10, queue_depth=64)
    try:
        ring.submit(timestamp(0), 0, FakeImage())
        assert wait_for(lambda: len(adapter.encoded) == 1)
        gate.clear()  # block the encoder: the pool now fills at its steady size
        for index in range(1, 12):
            ring.submit(timestamp(index), index, FakeImage())
        stats = ring.stats()
        backend = ring._backend_stats()
        gate.set()
    finally:
        ring.close()

    assert backend["pool_allocated_frames"] == 4  # the extension was released
    assert backend["pool_free_frames"] == 0
    assert stats.startup_dropped == 0
    assert stats.queue_drops == 7  # 4 slots taken, 7 refused


def test_startup_pool_memory_goes_back_to_the_driver_once_it_shrinks() -> None:
    open_gate = threading.Event()
    adapter = FakeAdapter(open_gate=open_gate)
    ring = make_ring(adapter, pool_frames=4, startup_pool_frames=10, queue_depth=64)
    try:
        for index in range(10):
            ring.submit(timestamp(index), index, FakeImage())
        assert adapter.memory_releases == 0  # extension slots still queued
        open_gate.set()
        assert wait_for(lambda: len(adapter.encoded) == 10)
        assert wait_for(lambda: adapter.memory_releases == 1)
        for index in range(10, 40):  # steady state: never released again
            ring.submit(timestamp(index), index, FakeImage())
            assert wait_for(lambda: len(adapter.encoded) == index + 1)
        backend = ring._backend_stats()
    finally:
        ring.close()

    assert adapter.memory_releases == 1
    assert backend["startup_pool_released"] is True
    assert backend["pool_allocated_frames"] == 4
    assert adapter.threads["release"] == adapter.threads["encode"]


def test_no_memory_release_without_a_startup_extension() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=4, startup_pool_frames=0)
    try:
        for index in range(12):
            ring.submit(timestamp(index), index, FakeImage())
            assert wait_for(lambda: len(adapter.encoded) == index + 1)
        backend = ring._backend_stats()
    finally:
        ring.close()

    assert adapter.memory_releases == 0
    assert backend["startup_pool_released"] is False


def test_nvenc_queue_depth_covers_the_startup_pool() -> None:
    ring = make_ring(FakeAdapter(), pool_frames=8)
    try:
        assert ring._queue_depth >= nvenc_ring.STARTUP_POOL_MAX_FRAMES
    finally:
        ring.close()


# ---- real encoder (CUDA + PyNvVideoCodec) -------------------------------------------
def _synthetic_frames(count: int, width: int = 1920, height: int = 1080):
    import torch

    base = torch.arange(width, device="cuda", dtype=torch.int32) % 256
    for index in range(count):
        frame = torch.empty((3, height, width), dtype=torch.uint8, device="cuda")
        frame[0] = ((base + index * 4) % 256).to(torch.uint8)
        frame[1] = ((base.flip(0) + index * 2) % 256).to(torch.uint8)
        frame[2] = (index * 3) % 256
        yield frame


def _mux_annex_b(access_units: List[bytes], path, *, fps: int) -> None:
    import av

    container = av.open(str(path), "w", format="mp4")
    stream = container.add_stream("h264", rate=fps)
    stream.time_base = Fraction(1, 90000)
    ticks = 90000 // fps
    for index, data in enumerate(access_units):
        packet = av.Packet(data)
        packet.pts = packet.dts = index * ticks
        packet.duration = ticks
        packet.time_base = stream.time_base
        packet.stream = stream
        container.mux(packet)
    container.close()


def _decode(path):
    import av

    with av.open(str(path)) as container:
        frames = [frame for frame in container.decode(video=0)]
        size = (frames[0].width, frames[0].height) if frames else None
        return len(frames), [frame.pts for frame in frames], size


@requires_nvenc
def test_real_adapter_encodes_1080p_without_host_copies(tmp_path) -> None:
    adapter = nvenc_ring.PyNvVideoCodecAdapter()
    import torch

    device = torch.device("cuda", 0)
    session = adapter.open_session(1920, 1080, 30.0, 300, 4_000_000, device)
    (slot,) = adapter.allocate_slots(1, 1920, 1080, device)
    access_units = []
    try:
        for ordinal, frame in enumerate(_synthetic_frames(300)):
            ready = adapter.stage(frame, slot, 1920, 1080)
            released = adapter.encode(session, slot, ready, ordinal, ordinal % 30 == 0)
            access_units.extend(released)
        access_units.extend(adapter.flush(session))
    finally:
        adapter.destroy(session)

    ordinals = [ordinal for ordinal, _ in access_units]
    keyframes = [
        index
        for index, (_, data) in enumerate(access_units)
        if nvenc_ring.is_idr_access_unit(data)
    ]
    path = tmp_path / "real_adapter.mp4"
    _mux_annex_b([data for _, data in access_units], path, fps=30)
    count, pts, size = _decode(path)

    assert ordinals == list(range(300))
    assert keyframes[:10] == [index * 30 for index in range(10)]
    assert count == 300
    assert size == (1920, 1080)
    assert pts == sorted(pts) and len(set(pts)) == 300


@requires_nvenc
def test_two_real_sessions_encode_simultaneously(tmp_path) -> None:
    import torch

    results = {}

    def run(name: str) -> None:
        adapter = nvenc_ring.PyNvVideoCodecAdapter()
        device = torch.device("cuda", 0)
        session = adapter.open_session(1920, 1080, 30.0, 300, 4_000_000, device)
        (slot,) = adapter.allocate_slots(1, 1920, 1080, device)
        units = []
        try:
            for ordinal, frame in enumerate(_synthetic_frames(120)):
                ready = adapter.stage(frame, slot, 1920, 1080)
                units.extend(
                    adapter.encode(session, slot, ready, ordinal, ordinal % 30 == 0)
                )
            units.extend(adapter.flush(session))
        finally:
            adapter.destroy(session)
        path = tmp_path / f"{name}.mp4"
        _mux_annex_b([data for _, data in units], path, fps=30)
        results[name] = _decode(path)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert sorted(results) == ["a", "b"]
    assert all(count == 120 for count, _, _ in results.values())


def _colour_test_frame(width: int = 1920, height: int = 1080):
    import torch

    # Gradients plus saturated colour bars in the top third.
    x = torch.linspace(0, 255, width, device="cuda")
    frame = torch.zeros(3, height, width, device="cuda")
    frame[0] = x
    frame[1] = x.flip(0)
    frame[2] = torch.linspace(0, 255, height, device="cuda")[:, None]
    bars = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0)]
    bars += [(255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0)]
    for index, colour in enumerate(bars):
        left, right = index * width // 8, (index + 1) * width // 8
        frame[:, : height // 3, left:right] = torch.tensor(
            colour, device="cuda", dtype=torch.float32
        )[:, None, None]
    return frame.round().clamp(0, 255).to(torch.uint8)


@requires_nvenc
@pytest.mark.parametrize("max_width", [None, 1280])
def test_real_adapter_keeps_the_colours_of_the_rgb_frame(max_width, tmp_path) -> None:
    import numpy as np
    import torch

    frame = _colour_test_frame()
    adapter = nvenc_ring.PyNvVideoCodecAdapter(max_width=max_width)
    width, height = nvenc_ring.encoded_size(1920, 1080, max_width=max_width)
    session = adapter.open_session(
        width, height, 30.0, 300, 8_000_000, torch.device("cuda", 0)
    )
    (slot,) = adapter.allocate_slots(1, 1920, 1080, session.device)
    units = []
    try:
        for ordinal in range(30):
            ready = adapter.stage(frame, slot, width, height)
            units.extend(adapter.encode(session, slot, ready, ordinal, ordinal == 0))
        units.extend(adapter.flush(session))
    finally:
        adapter.destroy(session)
    path = tmp_path / "colours.mp4"
    _mux_annex_b([data for _, data in units], path, fps=30)
    with av.open(str(path)) as container:
        # The stream carries no colour description (as with the NV12 input it
        # replaced): decode it as BT.709, the matrix both conversions use.
        decoded = [
            f.reformat(format="rgb24", src_colorspace="ITU709").to_ndarray()
            for f in container.decode(video=0)
        ]
    expected = adapter._fit(frame, width=width, height=height)
    expected = expected.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    error = np.abs(decoded[15].astype(int) - expected.astype(int))

    assert len(decoded) == 30
    assert decoded[15].shape == (height, width, 3)
    # same fidelity as the BT.709 NV12 conversion it replaced (L4: mean 1.0, p99 3)
    assert error.mean() < 2.0, error.mean()
    assert np.percentile(error, 99) <= 8


def _cuda_image(tensor):
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="nvenc-test"),
        numpy_image=None,
        tensor_image=tensor[:, :2, :2].cpu(),
    )
    image._tensor_image = tensor  # keep it on CUDA regardless of the device setting
    return image


def _artifact_path(artifact, tmp_path, name: str):
    payload = artifact.bytes_or_path
    if isinstance(payload, (bytes, bytearray)):
        path = tmp_path / name
        path.write_bytes(payload)
        return path
    return payload


def _wait_until_drained(ring, timeout: float = 10.0) -> None:
    # Every submitted frame has left the ingest queue: encoded or counted lost.
    def drained() -> bool:
        stats = ring.stats()
        lost = stats.startup_dropped + stats.queue_drops + stats.decimated
        return stats.queue_depth == 0 and (
            stats.frames_encoded + stats.encode_errors + lost >= stats.frames_submitted
        )

    assert wait_for(drained, timeout=timeout), ring.stats().to_dict()


def _loss_details(artifact, stats) -> dict:
    # Which counter took a frame: the clip report's loss fields and the ring's.
    details = {
        "report_loss": artifact.report.loss.to_dict(),
        "report_frames": artifact.report.frames,
        "report_partial": artifact.report.partial,
        "ring": {
            key: value
            for key, value in stats.to_dict().items()
            if key != "extra" and isinstance(value, int)
        },
    }

    return details


def _warm_process() -> None:
    # What the recorder block starts in its __init__; on a real host the model
    # load hides it before the first frame.
    status = nvenc_ring.process_warm_up().wait(120)
    assert status == nvenc_ring.WARM_UP_READY, nvenc_ring.process_warm_up().report()


@requires_nvenc
def test_warm_up_readies_the_process_and_records_its_steps() -> None:
    _warm_process()
    report = nvenc_ring.process_warm_up().report()

    assert set(report["steps_ms"]) >= {"probe_ms", "cuda_init_ms", "nvenc_open_ms"}
    assert report["error"] is None


@requires_nvenc
def test_ring_records_300_1080p_frames_into_a_decodable_mp4(tmp_path) -> None:
    _warm_process()
    ring = nvenc_ring.NvencRing("0", pre_roll_seconds=20.0, declared_fps=30.0)
    startup_drop_indices = []
    submit_ms = []
    try:
        for index, frame in enumerate(_synthetic_frames(300)):
            image = _cuda_image(frame)  # its first call imports the engine
            before = ring.stats().startup_dropped
            started = time.perf_counter()
            ring.submit(timestamp(index), index, image)
            submit_ms.append((time.perf_counter() - started) * 1000)
            if ring.stats().startup_dropped > before:
                startup_drop_indices.append(index)
            time.sleep(1 / 120)  # 4x real time: stresses the session-open window
        handle = ring.capture(timestamp(0), timestamp(299))
        # prepare_close drops frames still waiting for the encoder (counted as
        # queue_drops by design), so let the queue drain first: at 4x real time
        # the last frame can still be queued when the loop ends.
        _wait_until_drained(ring)
        ring.prepare_close(30.0)
        artifact = handle.future.result(timeout=60)
    finally:
        ring.close()
    count, pts, size = _decode(_artifact_path(artifact, tmp_path, "ring.mp4"))
    stats = ring.stats()
    losses = _loss_details(artifact, stats)

    # every frame is either in the clip or reported as start-up loss; on failure
    # the message names the counter that took the frame
    assert count + stats.startup_dropped == 300, (count, losses)
    assert artifact.report.loss.startup_dropped == stats.startup_dropped, losses
    # steady state: the pool never fills
    assert stats.queue_drops == 0, losses
    # start-up loss, if any, is confined to the session-open window (< 1 s)
    assert max(startup_drop_indices, default=-1) < 120
    # the warmed process never stalls the caller (it used to take 1.4-4.2 s)
    assert max(submit_ms) < 100
    assert size == (1920, 1080)
    assert pts[0] == 0 and pts == sorted(pts) and len(set(pts)) == count
    assert stats.extra["sessions_open"] == 0


@requires_nvenc
def test_startup_pool_returns_its_device_memory_after_the_backlog_drains() -> None:
    import torch

    _warm_process()
    frames = list(_synthetic_frames(90))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reserved_before = torch.cuda.memory_reserved()
    ring = nvenc_ring.NvencRing("0", pre_roll_seconds=5.0, declared_fps=30.0)
    try:
        ring.submit(timestamp(0), 0, _cuda_image(frames[0]))
        stats = ring._backend_stats()
        extension_bytes = (stats["startup_pool_frames"] - stats["pool_frames"]) * (
            3 * 1920 * 1080
        )
        reserved_peak = torch.cuda.memory_reserved()
        for index, frame in enumerate(frames[1:], start=1):
            ring.submit(timestamp(index), index, _cuda_image(frame))
            time.sleep(1 / 30)
        deadline = time.monotonic() + 10
        while (
            not ring._backend_stats()["startup_pool_released"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        backend = ring._backend_stats()
        reserved_after = torch.cuda.memory_reserved()
    finally:
        ring.close()

    steady_bytes = backend["pool_frames"] * 3 * 1920 * 1080
    session_bytes = nvenc_ring.PyNvVideoCodecAdapter().scratch_bytes(1920, 1080)
    margin = 16 * 1024 * 1024  # allocator rounding, graph pool, events
    details = (reserved_before, reserved_peak, reserved_after, extension_bytes)
    assert stats["startup_pool_frames"] > stats["pool_frames"]
    assert backend["startup_pool_released"], backend
    assert backend["pool_allocated_frames"] == backend["pool_frames"]
    # the extension block (24 slots, 149 MB at 1080p) left the caching allocator;
    # the session's own buffers were allocated after the peak was sampled
    assert (
        reserved_peak - reserved_after >= extension_bytes - session_bytes - margin
    ), details
    # steady state: the 8-slot pool and the session buffers, nothing more
    assert (
        reserved_after - reserved_before <= steady_bytes + session_bytes + margin
    ), details


@requires_nvenc
def test_two_rings_record_simultaneously_at_30_fps_without_loss(tmp_path) -> None:
    _warm_process()
    rings = [
        nvenc_ring.NvencRing(name, pre_roll_seconds=10.0, declared_fps=30.0)
        for name in ("raw", "annotated")
    ]
    try:
        for index, frame in enumerate(_synthetic_frames(150)):
            for ring in rings:
                ring.submit(timestamp(index), index, _cuda_image(frame))
            time.sleep(1 / 30)
        handles = [ring.capture(timestamp(0), timestamp(149)) for ring in rings]
        for ring in rings:
            _wait_until_drained(ring)
        for ring in rings:
            ring.prepare_close(30.0)
        artifacts = [handle.future.result(timeout=60) for handle in handles]
    finally:
        for ring in rings:
            ring.close()

    for index, artifact in enumerate(artifacts):
        count, pts, _ = _decode(_artifact_path(artifact, tmp_path, f"ring{index}.mp4"))
        stats = rings[index].stats()
        losses = _loss_details(artifact, stats)
        # the start-up pool covers both sessions opening at once
        assert count == 150, (count, losses)
        assert stats.startup_dropped == 0, losses
        assert stats.queue_drops == 0, losses
        assert pts[0] == 0
    assert all(ring.stats().extra["sessions_open"] == 0 for ring in rings)

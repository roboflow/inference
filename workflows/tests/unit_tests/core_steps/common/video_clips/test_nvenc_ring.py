"""Tests for the NVENC ring.

The ring logic is tested with a fake ``NvencAdapter`` (no CUDA, no PyNvVideoCodec).
``ring.BaseEncodedRing`` comes from the shared ``ring.py``; while that module is not
importable (it is developed in parallel) a minimal stand-in that follows the agreed
hook contract is installed from THIS file. The real-encoder tests at the bottom run
only where ``nvenc_ring.available()`` is true (CUDA GPU + PyNvVideoCodec).
"""

import importlib
import sys
import threading
import time
import types
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from queue import Empty, Queue
from typing import Any, List, Optional

import pytest

RING_MODULE = "roboflow_workflows.core_steps.common.video_clips.ring"


# ---- stand-in for ring.BaseEncodedRing (used only when ring.py is missing) ------
@dataclass
class _StubPreparedFrame:
    payload: Any
    width: int
    height: int
    nbytes: int


@dataclass
class _StubEncodedOutput:
    payload: bytes
    is_keyframe: bool


class _StubBaseEncodedRing:
    """Follows the hook contract agreed with the ring.py owner; nothing more."""

    backend = "base"

    def __init__(
        self,
        video_identifier,
        *,
        pre_roll_seconds,
        playback_speed=1.0,
        clip_fps=None,
        max_width=None,
        declared_fps=None,
        budget=None,
        queue_depth=None,
        byte_cap_bytes=None,
        stall_grace_seconds=10.0,
    ):
        self.video_identifier = video_identifier
        self.pre_roll_seconds = pre_roll_seconds
        self.playback_speed = playback_speed
        self.clip_fps = clip_fps
        self.max_width = max_width
        self.declared_fps = declared_fps
        self._stub_queue: Queue = Queue(maxsize=queue_depth or 30)
        self._stub_lock = threading.Lock()
        self._stub_close_lock = threading.Lock()
        self._stub_thread: Optional[threading.Thread] = None
        self._stub_closing = False
        self._stub_closed = False
        self._stub_queue_drops = 0
        self._stub_encode_errors = 0
        self._stub_size = None
        self._stub_t0 = None
        self._stub_last_key = None
        self.stub_outputs: List[Any] = []

    def submit(self, frame_timestamp, frame_number, image):
        if self._stub_closing:
            return None
        if self._stub_queue.full():
            self._stub_queue_drops += 1
            return None
        prepared = self._prepare_frame(image)
        if prepared is None:
            self._stub_queue_drops += 1
            return None
        with self._stub_lock:
            if self._stub_closing:
                self._discard_frame(prepared)
                return None
            self._stub_queue.put((frame_timestamp, frame_number, prepared))
            if self._stub_thread is None:
                self._stub_thread = threading.Thread(target=self._stub_run, daemon=True)
                self._stub_thread.start()

    def _stub_run(self):
        while True:
            item = self._stub_queue.get()
            if item is None:
                break
            timestamp, _, frame = item
            size = (frame.width, frame.height)
            if self._stub_size is not None and size != self._stub_size:
                self.stub_outputs.extend(self._flush_encoder())
                self._close_encoder()
                self._stub_t0 = None
                self._stub_last_key = None
            self._stub_size = size
            if self._stub_t0 is None:
                self._stub_t0 = timestamp
            elapsed = (timestamp - self._stub_t0).total_seconds() / self.playback_speed
            pts = int(round(elapsed * 90000))
            last = self._stub_last_key
            force = last is None or (
                (timestamp - last).total_seconds() / self.playback_speed >= 1.0 - 1e-6
            )
            try:
                outputs = self._encode_frame(frame, pts=pts, force_keyframe=force)
            except Exception:
                self._stub_encode_errors += 1
                self._discard_frame(frame)
                continue
            if force:
                self._stub_last_key = timestamp
            self.stub_outputs.extend(outputs)
        self.stub_outputs.extend(self._flush_encoder())
        self._close_encoder()

    def capture(self, start, end_inclusive):
        raise NotImplementedError("stub")

    def prepare_close(self, deadline):
        with self._stub_lock:
            if self._stub_closing:
                return None
            self._stub_closing = True
            while True:
                try:
                    item = self._stub_queue.get_nowait()
                except Empty:
                    break
                self._stub_queue_drops += 1
                self._discard_frame(item[2])
            if self._stub_thread is not None:
                self._stub_queue.put(None)

    def close(self):
        with self._stub_close_lock:
            if self._stub_closed:
                return None
            self.prepare_close(None)
            if self._stub_thread is not None:
                self._stub_thread.join(timeout=10)
            self._stub_closed = True

    def stats(self):
        return types.SimpleNamespace(
            backend=self.backend,
            queue_drops=self._stub_queue_drops,
            encode_errors=self._stub_encode_errors,
            extra=self._backend_stats(),
        )

    def _discard_frame(self, frame):
        return None

    def _backend_stats(self):
        return {}


def _import_nvenc_ring():
    try:
        ring = importlib.import_module(RING_MODULE)
        uses_stub = False
    except ImportError:
        ring = types.ModuleType(RING_MODULE)
        ring.BaseEncodedRing = _StubBaseEncodedRing
        ring.PreparedFrame = _StubPreparedFrame
        ring.EncodedOutput = _StubEncodedOutput
        sys.modules[RING_MODULE] = ring
        uses_stub = True
    try:
        module = importlib.import_module(
            "roboflow_workflows.core_steps.common.video_clips.nvenc_ring"
        )
    finally:
        if uses_stub:
            # nvenc_ring keeps its references; nothing else may see the stand-in.
            sys.modules.pop(RING_MODULE, None)
    return module, ring, uses_stub


nvenc_ring, _ring_module, USES_STUB_BASE = _import_nvenc_ring()
EncodedOutput = _ring_module.EncodedOutput
PreparedFrame = _ring_module.PreparedFrame

requires_stub_base = pytest.mark.skipif(
    not USES_STUB_BASE, reason="reads the stand-in base's recorded outputs"
)
requires_nvenc = pytest.mark.skipif(
    not nvenc_ring.available(), reason="needs a CUDA GPU and PyNvVideoCodec"
)
requires_real_base = pytest.mark.skipif(
    USES_STUB_BASE, reason="needs the real ring.py (packet store and remux)"
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
    ):
        self.delay_frames = delay_frames
        self.encode_sleep = encode_sleep
        self.gate = gate
        self.fail_encodes = fail_encodes
        self.raw_payloads = raw_payloads
        self.lock = threading.Lock()
        self.slots_allocated = 0
        self.staged: List[Any] = []
        self.sessions: List[FakeSession] = []
        self.destroyed: List[FakeSession] = []
        self.encoded: List[tuple] = []
        self.calls: List[str] = []
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

    def slot_bytes(self, width, height):
        return width * height * 3 // 2

    def allocate_slot(self, width, height, device):
        with self.lock:
            self.slots_allocated += 1
            return ("slot", self.slots_allocated, width, height)

    def stage(self, tensor, slot, width, height):
        self.threads["stage"].add(threading.get_ident())
        with self.lock:
            self.staged.append(slot)
        return ("ready", slot)

    def open_session(self, width, height, fps, gop, bitrate, device):
        self.threads["open"].add(threading.get_ident())
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

    def _access_unit(self, ordinal, force_idr):
        if self.raw_payloads:
            return ordinal, b"\xff\xfe" + bytes([ordinal % 256])
        return ordinal, IDR_ACCESS_UNIT if force_idr else P_ACCESS_UNIT


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


def test_ring_refuses_construction_without_nvenc(monkeypatch) -> None:
    monkeypatch.setattr(nvenc_ring, "available", lambda: False)

    with pytest.raises(RuntimeError):
        nvenc_ring.NvencRing("0", pre_roll_seconds=5.0)


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
def test_pool_fills_then_reports_full_and_recycles_discarded_slots() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=8)

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
    ring = make_ring(adapter, pool_frames=1)

    with pytest.raises(RuntimeError):
        ring._prepare_frame(FakeImage())
    stats = ring._backend_stats()

    assert stats["pool_free_frames"] == 1
    assert stats["stage_errors"] == 1
    ring.close()


def test_resolution_change_resets_the_pool_and_drops_old_slots() -> None:
    adapter = FakeAdapter()
    ring = make_ring(adapter, pool_frames=2)
    old = ring._prepare_frame(FakeImage(1920, 1080))

    new = ring._prepare_frame(FakeImage(1280, 720))
    ring._discard_frame(old)
    stats = ring._backend_stats()

    assert new.payload.size == (1280, 720)
    assert stats["pool_allocated_frames"] == 1
    assert stats["pool_free_frames"] == 0
    ring.close()


def test_full_pool_drops_frames_while_encoder_is_blocked() -> None:
    gate = threading.Event()
    adapter = FakeAdapter(gate=gate)
    ring = make_ring(adapter, pool_frames=8, queue_depth=64)

    for index in range(20):
        ring.submit(timestamp(index), index, FakeImage())
    stats = ring._backend_stats()

    assert adapter.slots_allocated == 8
    assert len(adapter.staged) == 8
    assert stats["pool_drops"] == 12
    assert queue_drops(ring) >= 12
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
    # every slot comes back after its encode, so one slot served all five frames
    assert stats["pool_allocated_frames"] == 1
    assert stats["pool_free_frames"] == 1
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
    ring = make_ring(adapter, pool_frames=8)

    ring._encode_frame(ring._prepare_frame(FakeImage()), pts=0, force_keyframe=True)
    held = ring._prepare_frame(FakeImage())  # reuses the slot returned by encode
    ring._prepare_frame(FakeImage())
    stats = ring._backend_stats()

    assert held is not None
    expected = 64 * 1024 * 1024 + 2 * 1920 * 1080 * 3 // 2 + 1920 * 1080 * 4
    assert stats["vram_bytes_estimate"] == expected
    assert stats["sessions_open"] == 1
    ring._close_encoder()
    assert ring._backend_stats()["vram_bytes_estimate"] == expected - 64 * 1024 * 1024
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


@requires_stub_base
def test_keyframes_reach_the_base_in_submission_order() -> None:
    adapter = FakeAdapter(delay_frames=3)
    ring = make_ring(adapter, pool_frames=8, queue_depth=128)

    for index in range(60):
        assert wait_for_slot(ring)
        ring.submit(timestamp(index), index, FakeImage())
    assert wait_for(lambda: len(adapter.encoded) == 60)
    ring.close()

    flags = [output.is_keyframe for output in ring.stub_outputs]
    assert len(flags) == 60
    assert [index for index, flag in enumerate(flags) if flag] == [0, 30]


# ---- lifecycle ----------------------------------------------------------------------
def test_prepare_close_drops_unencoded_slots_and_flushes_before_destroy() -> None:
    gate = threading.Event()
    adapter = FakeAdapter(gate=gate, delay_frames=2)
    ring = make_ring(adapter, pool_frames=8, queue_depth=64)

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
    slot = adapter.allocate_slot(1920, 1080, device)
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
        slot = adapter.allocate_slot(1920, 1080, device)
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


@requires_nvenc
@requires_real_base
def test_ring_records_300_1080p_frames_into_a_decodable_mp4(tmp_path) -> None:
    ring = nvenc_ring.NvencRing("0", pre_roll_seconds=20.0, declared_fps=30.0)
    try:
        for index, frame in enumerate(_synthetic_frames(300)):
            ring.submit(timestamp(index), index, _cuda_image(frame))
            time.sleep(1 / 120)  # keep the 8-slot pool from filling
        handle = ring.capture(timestamp(0), timestamp(299))
        ring.prepare_close(30.0)
        artifact = handle.future.result(timeout=60)
    finally:
        ring.close()
    count, pts, size = _decode(_artifact_path(artifact, tmp_path, "ring.mp4"))
    extra = ring.stats().extra

    assert count + extra["pool_drops"] == 300
    assert count >= 290
    assert size == (1920, 1080)
    assert pts[0] == 0 and pts == sorted(pts) and len(set(pts)) == count
    assert extra["sessions_open"] == 0


@requires_nvenc
@requires_real_base
def test_two_rings_record_simultaneously(tmp_path) -> None:
    rings = [
        nvenc_ring.NvencRing(name, pre_roll_seconds=10.0, declared_fps=30.0)
        for name in ("raw", "annotated")
    ]
    try:
        for index, frame in enumerate(_synthetic_frames(150)):
            for ring in rings:
                ring.submit(timestamp(index), index, _cuda_image(frame))
            time.sleep(1 / 60)
        handles = [ring.capture(timestamp(0), timestamp(149)) for ring in rings]
        for ring in rings:
            ring.prepare_close(30.0)
        artifacts = [handle.future.result(timeout=60) for handle in handles]
    finally:
        for ring in rings:
            ring.close()

    for index, artifact in enumerate(artifacts):
        count, pts, _ = _decode(_artifact_path(artifact, tmp_path, f"ring{index}.mp4"))
        assert count >= 140
        assert pts[0] == 0
    assert all(ring.stats().extra["sessions_open"] == 0 for ring in rings)

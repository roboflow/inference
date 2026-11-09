"""NVENC encoder ring fed straight from CUDA tensors (PyNvVideoCodec).

``NvencRing`` is the ``Ring`` backend the Video Clip Recorder selects when the
workflow image is a CUDA tensor and NVENC can be used (Serverless Streaming GPU
runners, where every frame is a CHW RGB uint8 CUDA tensor and a numpy download
costs 17-18 ms per 1080p frame on the pipeline thread). No frame is copied to the
CPU; only the encoded access units are.

The shared ``ring.BaseEncodedRing`` owns the ingest queue, the encoder thread, grid
decimation, PTS, the IDR cadence (one per output second), the packet store with
GOP-aligned captures, lifecycle and the MP4 remux. This module adds the encoder:

* Process warm-up (``ProcessWarmUp``, once per process, on its own daemon thread):
  the availability probe, the PyNvVideoCodec import, torch's CUDA context, the
  conversion kernels on a private stream and a throwaway NVENC session that loads
  the encoder driver. The recorder starts it from its ``__init__`` on CUDA hosts,
  so it normally finishes during the host's model load. ``run()`` never waits for
  it: frames that arrive earlier are counted as ``startup_dropped``. If the warm-up
  finds no usable NVENC (or exceeds ``WARM_UP_TIMEOUT_SECONDS``) the recorder uses
  x264 instead.
* Caller thread (``_prepare_frame``): take a free slot of the GPU staging pool or
  report the pool full, then enqueue ONE device-to-device copy of the RGB frame
  into the slot on the producer's (caller's) current stream and record the slot's
  event there: two CUDA API calls per frame. Later in-place drawing on the image
  is ordered after the copy by the stream itself, so no reference to the frame
  survives the call. Every other CUDA call (stream waits, resize, NV12 conversion,
  encode, synchronize) runs on the encoder thread, which matters under gVisor
  where each driver call is an ioctl through the sentry.
* Encoder thread (``_encode_frame`` / ``_flush_encoder`` / ``_close_encoder``): one
  NVENC session per ring, opened at the first frame with the Serverless Streaming
  runtime's settings (rf-video-processor ``nvenc_encoders.NvEncoder``): P3,
  ultra-low-latency tuning, CBR with a 250 ms VBV, ``bf=0``, no lookahead, SPS/PPS
  repeated on every IDR, torch's primary CUDA context and a private CUDA stream.
  Per frame it waits on the slot's event, copies the slot into the session's
  source buffer and replays a CUDA graph of the resize (``max_width``) and the
  RGB to BT.709 NV12 conversion (captured once per session), then encodes and
  synchronizes: about five torch calls per frame instead of about forty. Each
  eager torch call releases and re-takes the GIL, and under gVisor every
  contended GIL hand-off is a futex round trip through the sentry, so the eager
  path slowed the host's own threads; it remains the fallback when a graph
  cannot be captured. IDRs are forced
  when the base asks; keyframe flags are read back from the access units (IDR
  NAL). The session is destroyed on the encoder thread before it exits; a session
  still alive at interpreter exit segfaults, so an ``atexit`` hook also closes any
  ring left open.

Pool: ``WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES`` (default 8) slots in steady state; a
full pool counts ``queue_drops``. Until the ring's session is open the pool may grow
to a start-up headroom of ``STARTUP_POOL_MAX_FRAMES`` (32) slots, capped at
``STARTUP_POOL_MAX_BYTES`` (192 MiB) of device memory, so the frames that arrive
while NVENC opens the session (0.35-0.8 s measured) are queued instead of dropped;
the extra slots are freed once the backlog drains and their block is handed back
to the driver (one ``torch.cuda.empty_cache()`` per stream start-up, on the encoder
thread), so a recorder's steady-state VRAM is the 8-slot pool plus the session.
Frames refused in that window
count as ``startup_dropped``. Device memory is not host memory, so the host byte
budget does not count the pool; ``vram_bytes_estimate`` in the ring's stats does.

Bitrate: ``WORKFLOWS_VIDEO_CLIP_NVENC_BITRATE_KBPS`` when set, else the runtime's
scaled default, 4 Mbit/s at 1080p30 scaled by pixel rate (about 0.064 bit per pixel)
and clamped to 0.3-20 Mbit/s: the quality the runtime's preview already ships, and
at 1080p30 about 30 MB for a 60 s pre-roll. The rate-control fps is the declared
source fps capped by ``clip_fps`` (30 when unknown).

Every torch / PyNvVideoCodec / CUDA driver call sits behind ``NvencAdapter``; the
ring logic is unit tested with a fake adapter and ``PyNvVideoCodecAdapter`` (the
real one) by a CUDA-gated test.
"""

import atexit
import ctypes
import functools
import importlib.util
import logging
import sys
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple

from roboflow_workflows.core_steps.common.video_clips import read_env
from roboflow_workflows.core_steps.common.video_clips.ring import (
    DEFAULT_QUEUE_DEPTH,
    BaseEncodedRing,
    EncodedOutput,
    EncoderStartingError,
    PreparedFrame,
)

LOGGER = logging.getLogger(__name__)

BACKEND_NAME = "nvenc"
POOL_FRAMES_ENV = "WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES"
BITRATE_KBPS_ENV = "WORKFLOWS_VIDEO_CLIP_NVENC_BITRATE_KBPS"
DEFAULT_POOL_FRAMES = 8
MAX_POOL_FRAMES = 256
DEFAULT_FPS = 30.0
MAX_FPS = 120.0
REFERENCE_BITRATE = 4_000_000
MIN_DEFAULT_BITRATE = 300_000
MAX_DEFAULT_BITRATE = 20_000_000
MIN_BITRATE_KBPS = 100
MAX_BITRATE_KBPS = 200_000
VBV_SECONDS = 0.25
IDR_OUTPUT_SECONDS = 1.0
ATEXIT_CLOSE_DEADLINE_SECONDS = 2.0
# Start-up headroom: until a ring's session is open (0.35-0.4 s for one session on
# an L4, about 0.8 s for two opening at once), its pool may grow to this many
# slots, capped by device bytes (32 slots of 1080p RGB = 190 MiB).
STARTUP_POOL_MAX_FRAMES = 32
STARTUP_POOL_MAX_BYTES = 192 * 1024 * 1024
# The process warm-up: status values and the bound after which a warm-up that has
# not finished counts as unavailable (the recorder then records with x264).
WARM_UP_NOT_STARTED = "not_started"
WARM_UP_PENDING = "pending"
WARM_UP_READY = "ready"
WARM_UP_UNAVAILABLE = "unavailable"
WARM_UP_TIMEOUT_SECONDS = 60.0
# Probe frame for the warm-up: small, but encodable by every NVENC generation.
WARM_UP_PROBE_WIDTH = 256
WARM_UP_PROBE_HEIGHT = 144
WARM_UP_PROBE_FRAMES = 2
WARM_UP_PROBE_BITRATE = 300_000
# NVENC's own GOP is only a backstop: the base ring forces an IDR every output
# second, so the encoder's periodic IDR is set to ten output seconds.
BACKSTOP_GOP_OUTPUT_SECONDS = 10.0
H264_NAL_IDR = 5

# BT.709 limited range: Y = 16 + 219/255 (Kr R + Kg G + Kb B); chroma uses 224/255.
# The chroma rows fold in the 2x2 box average (/4); chroma is linear in RGB.
_KR, _KB = 0.2126, 0.0722
_KG = 1 - _KR - _KB
_Y = tuple(219 / 255 * k for k in (_KR, _KG, _KB))
_U = tuple(224 / 255 / 4 * k for k in (-_KR / (2 - 2 * _KB), -_KG / (2 - 2 * _KB), 0.5))
_V = tuple(224 / 255 / 4 * k for k in (0.5, -_KG / (2 - 2 * _KR), -_KB / (2 - 2 * _KR)))


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """Report whether NVENC encoding from CUDA tensors can be used in this process.

    True only when torch sees a CUDA device and ``PyNvVideoCodec`` imports. No CUDA
    context or encoder session is created. The result is cached for the process and
    the check never raises.

    Returns:
        True when ``NvencRing`` can be constructed and fed CUDA tensors.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        import PyNvVideoCodec  # noqa: F401
    except Exception:
        return False

    return True


def nvenc_installed() -> bool:
    """Tell whether ``PyNvVideoCodec`` can be imported, without importing it.

    Returns:
        True when an import spec for ``PyNvVideoCodec`` exists.
    """
    try:
        spec = importlib.util.find_spec("PyNvVideoCodec")
    except (ImportError, ValueError):
        return False

    return spec is not None


def should_warm_up(configured_device: Any = None) -> bool:
    """Decide cheaply whether this process is a CUDA host worth warming up.

    Never imports torch: true only when the host already imported it, either
    configured CUDA image tensors or already initialised CUDA, and
    ``PyNvVideoCodec`` is installed. No CUDA driver call is made.

    Args:
        configured_device: The host's image tensor device (``torch.device``,
            string or None), e.g. ``WORKFLOWS_IMAGE_TENSOR_DEVICE``.

    Returns:
        True when the NVENC warm-up should start now.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False

    device_type = getattr(configured_device, "type", None)
    if device_type is None and configured_device is not None:
        device_type = str(configured_device).split(":")[0]
    if device_type != "cuda":
        try:
            cuda_initialised = bool(torch.cuda.is_initialized())
        except Exception:
            return False
        if not cuda_initialised:
            return False

    installed = nvenc_installed()

    return installed


class ProcessWarmUp:
    """One-shot, process-wide NVENC warm-up on a background daemon thread.

    Does the work the first ``NvencRing.submit()`` used to do on the pipeline
    thread (1.4 s on a warm L4 process, 4.2 s cold): the availability probe, the
    PyNvVideoCodec import, torch's CUDA context, the conversion kernels and a
    throwaway NVENC session that loads the encoder driver. ``status()`` never
    blocks; a warm-up that has not finished ``timeout_seconds`` after it started
    counts as unavailable from then on.

    Args:
        steps: The warm-up work; receives a dict to record step timings (ms) in
            and raises when NVENC cannot be used. Defaults to the real warm-up.
        precheck: Cheap synchronous check run by ``start()``; when false, the
            status becomes unavailable without starting a thread. Defaults to
            :func:`nvenc_installed`.
        timeout_seconds: Bound on the pending state.
        clock: Monotonic clock (tests).
    """

    def __init__(
        self,
        steps: Optional[Callable[[Dict[str, float]], None]] = None,
        *,
        precheck: Optional[Callable[[], bool]] = None,
        timeout_seconds: float = WARM_UP_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._steps = steps or _warm_up_process
        self._precheck = precheck or nvenc_installed
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = WARM_UP_NOT_STARTED
        self._started_at: Optional[float] = None
        self._elapsed_ms: Optional[float] = None
        self._timings: Dict[str, float] = {}
        self._error: Optional[str] = None

    def start(self) -> None:
        """Start the warm-up thread once; later calls return immediately."""
        with self._lock:
            if self._status != WARM_UP_NOT_STARTED:
                return None
            self._started_at = self._clock()
            try:
                installed = bool(self._precheck())
            except Exception as error:
                installed = False
                self._error = repr(error)
            if not installed:
                self._status = WARM_UP_UNAVAILABLE
                self._error = self._error or "PyNvVideoCodec is not installed"
                self._elapsed_ms = 0.0
                self._done.set()
                return None
            self._status = WARM_UP_PENDING
            self._thread = threading.Thread(
                target=self._run, name="video-clip-nvenc-warm-up", daemon=True
            )
            self._thread.start()

    def status(self, *, start: bool = True) -> str:
        """Return the warm-up status without blocking.

        Args:
            start: Start the warm-up when it has not started yet.

        Returns:
            One of ``not_started``, ``pending``, ``ready`` or ``unavailable``.
        """
        if start:
            self.start()
        with self._lock:
            if (
                self._status == WARM_UP_PENDING
                and self._clock() - self._started_at > self._timeout_seconds
            ):
                self._status = WARM_UP_UNAVAILABLE
                self._error = (
                    f"the NVENC warm-up did not finish in {self._timeout_seconds} s"
                )
            status = self._status

        return status

    def wait(self, timeout: Optional[float] = None) -> str:
        """Start the warm-up and wait for it to finish (tests and benchmarks).

        Args:
            timeout: Seconds to wait; None waits until it finishes.

        Returns:
            The status after waiting.
        """
        self.start()
        self._done.wait(timeout)
        status = self.status(start=False)

        return status

    def wait_if_running(self, timeout: float) -> None:
        """Wait up to ``timeout`` seconds for a warm-up thread that is running.

        Args:
            timeout: Seconds to wait; returns at once when no thread is running.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            return None

        thread.join(timeout)

    @property
    def error(self) -> Optional[str]:
        """Why the warm-up is unavailable, if it is."""
        with self._lock:
            return self._error

    def report(self) -> Dict[str, Any]:
        """Return the warm-up outcome and its step timings.

        Returns:
            ``status``, ``warm_up_ms`` (None while pending), ``steps_ms`` and
            ``error``.
        """
        status = self.status(start=False)
        with self._lock:
            report = {
                "status": status,
                "warm_up_ms": self._elapsed_ms,
                "steps_ms": dict(self._timings),
                "error": self._error,
            }

        return report

    def _run(self) -> None:
        started = time.perf_counter()
        timings: Dict[str, float] = {}
        try:
            self._steps(timings)
        except Exception as error:
            outcome, failure = WARM_UP_UNAVAILABLE, repr(error)
        else:
            outcome, failure = WARM_UP_READY, None
        elapsed_ms = (time.perf_counter() - started) * 1000
        with self._lock:
            self._timings = dict(timings)
            self._elapsed_ms = elapsed_ms
            if self._status == WARM_UP_PENDING:
                self._status = outcome
                self._error = failure
            final_status = self._status
        self._done.set()
        LOGGER.info(
            "NVENC warm-up finished in %.0f ms: %s %s (steps: %s)",
            elapsed_ms,
            final_status,
            failure or "",
            {name: round(value, 1) for name, value in timings.items()},
        )


def _warm_up_process(timings: Dict[str, float]) -> None:
    # The real warm-up: everything process-wide the first frame used to pay for,
    # run through the same adapter code paths a ring uses.
    started = time.perf_counter()
    if not available():
        raise RuntimeError("NVENC needs torch CUDA and PyNvVideoCodec")
    timings["probe_ms"] = (time.perf_counter() - started) * 1000

    import torch

    step = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        torch.zeros(1, device=device).add_(1)
    stream.synchronize()
    _cuda_driver()
    timings["cuda_init_ms"] = (time.perf_counter() - step) * 1000

    step = time.perf_counter()
    adapter = PyNvVideoCodecAdapter()
    width, height = WARM_UP_PROBE_WIDTH, WARM_UP_PROBE_HEIGHT
    with torch.cuda.stream(stream):
        frame = torch.randint(0, 256, (3, height, width), dtype=torch.uint8)
        frame = frame.to(device)
        # The resize path (``max_width``) and a strided (cropped) copy.
        adapter._fit(frame, width=width // 2, height=height // 2)
        gray = frame[:1].expand(3, -1, -1)
    stream.synchronize()
    timings["kernels_ms"] = (time.perf_counter() - step) * 1000

    step = time.perf_counter()
    session = adapter.open_session(
        width, height, 30.0, 30, WARM_UP_PROBE_BITRATE, device
    )
    timings["nvenc_open_ms"] = (time.perf_counter() - step) * 1000
    step = time.perf_counter()
    try:
        (slot,) = adapter.allocate_slots(1, width, height, device)
        with torch.cuda.stream(stream):
            for ordinal in range(WARM_UP_PROBE_FRAMES):
                source = frame if ordinal % 2 == 0 else gray
                ready = adapter.stage(source, slot, width, height)
                adapter.encode(session, slot, ready, ordinal, ordinal == 0)
        adapter.flush(session)
    finally:
        adapter.destroy(session)
    stream.synchronize()
    timings["nvenc_encode_ms"] = (time.perf_counter() - step) * 1000


_PROCESS_WARM_UP = ProcessWarmUp()


def process_warm_up() -> ProcessWarmUp:
    """Return the process-wide warm-up (not started until someone asks).

    Returns:
        The singleton ``ProcessWarmUp``.
    """
    return _PROCESS_WARM_UP


def start_warm_up() -> None:
    """Start the process-wide NVENC warm-up in the background (idempotent)."""
    process_warm_up().start()


def default_bitrate(width: int, height: int, fps: float) -> int:
    """Compute the runtime's scaled default CBR bitrate.

    Args:
        width: Encoded width in pixels.
        height: Encoded height in pixels.
        fps: Rate-control frame rate.

    Returns:
        Bits per second: 4 Mbit/s at 1080p30 scaled by pixel rate, clamped to
        0.3-20 Mbit/s.
    """
    rate = REFERENCE_BITRATE * width * height * fps / (1920 * 1080 * 30)
    bitrate = int(min(MAX_DEFAULT_BITRATE, max(MIN_DEFAULT_BITRATE, rate)))

    return bitrate


def resolve_bitrate(width: int, height: int, fps: float) -> int:
    """Resolve the CBR bitrate from the environment or the scaled default.

    Args:
        width: Encoded width in pixels.
        height: Encoded height in pixels.
        fps: Rate-control frame rate.

    Returns:
        Bits per second. ``WORKFLOWS_VIDEO_CLIP_NVENC_BITRATE_KBPS`` (clamped to
        100-200000 kbit/s) wins when it parses; otherwise :func:`default_bitrate`.
    """
    raw = read_env(BITRATE_KBPS_ENV)
    if raw:
        try:
            kbps = int(float(raw))
        except ValueError:
            LOGGER.warning("Ignoring non-numeric %s=%r", BITRATE_KBPS_ENV, raw)
        else:
            kbps = min(MAX_BITRATE_KBPS, max(MIN_BITRATE_KBPS, kbps))
            return kbps * 1000

    bitrate = default_bitrate(width, height, fps)

    return bitrate


def resolve_pool_frames() -> int:
    """Resolve the GPU pool size from ``WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES``.

    Returns:
        Number of NV12 frame slots per ring, default 8, clamped to 1-256.
    """
    raw = read_env(POOL_FRAMES_ENV)
    if not raw:
        return DEFAULT_POOL_FRAMES

    try:
        frames = int(float(raw))
    except ValueError:
        LOGGER.warning("Ignoring non-numeric %s=%r", POOL_FRAMES_ENV, raw)
        return DEFAULT_POOL_FRAMES

    frames = min(MAX_POOL_FRAMES, max(1, frames))

    return frames


def startup_pool_frames(pool_frames: int, slot_bytes: int) -> int:
    """Size the start-up pool for one ring.

    Args:
        pool_frames: The steady-state pool size.
        slot_bytes: Device bytes of one staging slot.

    Returns:
        ``STARTUP_POOL_MAX_FRAMES`` capped at ``STARTUP_POOL_MAX_BYTES`` of slots,
        never below ``pool_frames``.
    """
    by_bytes = STARTUP_POOL_MAX_BYTES // max(1, slot_bytes)
    frames = max(pool_frames, min(STARTUP_POOL_MAX_FRAMES, by_bytes))

    return frames


def encoded_size(
    width: int, height: int, *, max_width: Optional[int] = None
) -> Tuple[int, int]:
    """Compute the encoded frame size for a source frame.

    Scales down (never up) to ``max_width`` keeping the aspect ratio, then drops the
    odd edge row/column because NV12 needs even dimensions.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        max_width: Optional output width cap.

    Returns:
        ``(width, height)`` of the encoded frames, both even and at least 2.
    """
    if max_width is not None and width > max_width:
        height = max(2, int(round(height * max_width / width)))
        width = max_width

    size = (max(2, width - width % 2), max(2, height - height % 2))

    return size


def is_idr_access_unit(data: bytes) -> bool:
    """Tell whether an Annex B H.264 access unit contains an IDR slice.

    Args:
        data: One encoded access unit with 3- or 4-byte start codes.

    Returns:
        True when any NAL unit in ``data`` has ``nal_unit_type`` 5.
    """
    index = data.find(b"\x00\x00\x01")
    while index != -1 and index + 3 < len(data):
        if data[index + 3] & 0x1F == H264_NAL_IDR:
            return True
        index = data.find(b"\x00\x00\x01", index + 3)

    return False


class NvencAdapter(Protocol):
    """Every torch / PyNvVideoCodec call the ring makes (the testability seam).

    ``frame_size``, ``staging_size``, ``allocate_slots`` and ``stage`` run on the
    pipeline thread; ``stage`` must enqueue at most one copy and one event on the
    caller's stream. ``open_session``, ``encode``, ``flush``, ``destroy`` and
    ``release_cached_memory`` run on the ring thread only. ``encode``/``flush`` return ``(input_ordinal,
    access_unit)`` pairs in output order; the ring maps ordinals back to frames.
    """

    def frame_size(self, tensor: Any) -> Tuple[int, int]: ...

    def staging_size(self, tensor: Any) -> Tuple[int, int]: ...

    def allocate_slots(
        self, count: int, width: int, height: int, device: Any
    ) -> List[Any]: ...

    def stage(self, tensor: Any, slot: Any, width: int, height: int) -> Any: ...

    def open_session(
        self, width: int, height: int, fps: float, gop: int, bitrate: int, device: Any
    ) -> Any: ...

    def encode(
        self, session: Any, slot: Any, ready: Any, ordinal: int, force_idr: bool
    ) -> List[Tuple[int, bytes]]: ...

    def flush(self, session: Any) -> List[Tuple[int, bytes]]: ...

    def destroy(self, session: Any) -> None: ...

    def release_cached_memory(self) -> None: ...

    def tensor_device(self, tensor: Any) -> Any: ...

    def slot_bytes(self, width: int, height: int) -> int: ...

    def scratch_bytes(self, width: int, height: int) -> int: ...

    def session_vram_bytes(self, session: Any) -> int: ...


def rgb_to_nv12(rgb: Any, nv12: Any, work: Any) -> Any:
    """Convert a CHW RGB tensor to limited-range BT.709 NV12 on its device.

    Same arithmetic as the Serverless Streaming runtime: one float32 luma-sized
    scratch reused for both chroma planes, 2x2 box chroma subsampling.

    Args:
        rgb: ``(3, H, W)`` uint8 or float tensor, even ``H`` and ``W``.
        nv12: ``(H * 3 // 2, W)`` uint8 output tensor on the same device.
        work: ``(H, W)`` float32 scratch tensor on the same device.

    Returns:
        ``nv12``, filled.
    """
    import torch

    _, height, width = rgb.shape
    r, g, b = rgb.unbind(0)
    torch.mul(r, _Y[0], out=work)
    work.add_(g, alpha=_Y[1]).add_(b, alpha=_Y[2]).add_(16).round_().clamp_(16, 235)
    nv12[:height].copy_(work)
    quarter = (height // 2) * (width // 2)
    for index, coefficients in enumerate((_U, _V)):
        plane = work.view(-1)[index * quarter : (index + 1) * quarter]
        plane = plane.view(height // 2, width // 2)
        terms = [
            (channel[dy::2, dx::2], k)
            for dy in (0, 1)
            for dx in (0, 1)
            for channel, k in zip((r, g, b), coefficients)
        ]
        torch.mul(terms[0][0], terms[0][1], out=plane)
        for channel, k in terms[1:]:
            plane.add_(channel, alpha=k)
        plane.add_(128).round_().clamp_(16, 240)
        nv12[height:, index::2].copy_(plane)

    return nv12


def _rgb_chw(tensor: Any) -> Any:
    # Gray (HW, 1HW) or RGBA (4HW) -> 3HW RGB view on the same device.
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        return tensor.expand(3, -1, -1)
    if tensor.shape[0] == 4:
        return tensor[:3]
    return tensor


class _CudaPlane:
    def __init__(self, tensor: Any, shape: tuple, strides: tuple, offset: int, stream):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "strides": strides,
            "typestr": "|u1",
            "version": 3,
            "data": (tensor.data_ptr() + offset, False),
            "stream": stream,
        }


class _NV12Surface:
    def __init__(self, tensor: Any, width: int, height: int, stream: int):
        self.tensor = tensor  # ownership, not just pointers
        self.planes = [
            _CudaPlane(tensor, (height, width, 1), (width, 1, 1), 0, stream),
            _CudaPlane(
                tensor,
                (height // 2, width // 2, 2),
                (width, 2, 1),
                width * height,
                stream,
            ),
        ]

    def cuda(self) -> list:
        return self.planes


class _CudaDriver:
    """``cuCtxGetCurrent`` / ``cuCtxSetCurrent`` from libcuda (torch's context only)."""

    def __init__(self):
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        driver.cuCtxGetCurrent.restype = ctypes.c_int
        driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        driver.cuCtxSetCurrent.restype = ctypes.c_int
        self._driver = driver

    def current_context(self) -> int:
        context = ctypes.c_void_p()
        if self._driver.cuCtxGetCurrent(ctypes.byref(context)) != 0:
            raise RuntimeError("cuCtxGetCurrent failed")
        if not context.value:
            raise RuntimeError("no current CUDA context")
        return context.value

    def set_current(self, context: int) -> None:
        if self._driver.cuCtxSetCurrent(context) != 0:
            raise RuntimeError("could not make the encoder CUDA context current")


@functools.lru_cache(maxsize=1)
def _cuda_driver() -> _CudaDriver:
    return _CudaDriver()


@dataclass
class _StagingSlot:
    """One pool slot: a CHW RGB uint8 frame copy and the event that marks it."""

    rgb: Any
    event: Any


@dataclass
class _Session:
    encoder: Any
    device: Any
    stream: Any
    context: int
    width: int
    height: int
    vram_bytes: int
    nv12: Any = None
    work: Any = None
    # The captured conversion (``_GraphConverter``) or None for the eager path.
    converter: Any = None
    conversion: str = "pending"


class _GraphConverter:
    """A CUDA graph of ``_fit`` + ``rgb_to_nv12`` reading a fixed source buffer.

    Replaying it costs one torch call (one GIL release) instead of the ~40 the
    eager conversion makes. The source buffer has the staging slot's shape; each
    frame is copied into it on the session stream before the replay.
    """

    def __init__(self, torch: Any, source: Any, graph: Any):
        self._torch = torch
        self.source = source
        self.graph = graph

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.source.shape)

    def __call__(self, rgb: Any) -> None:
        self.source.copy_(rgb)
        self.graph.replay()


class PyNvVideoCodecAdapter:
    """Real ``NvencAdapter``: torch CUDA streams + PyNvVideoCodec NVENC.

    The caller thread only copies the frame into a pooled RGB slot on its own
    stream and records the slot's event. The encoder thread waits on that event
    on the session's private stream, scales (``max_width``), converts to NV12 and
    encodes.

    Args:
        max_width: Optional output width cap; frames wider than this are scaled
            down on the GPU (area filter) before the NV12 conversion.
    """

    def __init__(self, *, max_width: Optional[int] = None):
        import PyNvVideoCodec
        import torch

        self._torch = torch
        self._nvc = PyNvVideoCodec
        self._max_width = max_width

    def tensor_device(self, tensor: Any) -> Any:
        """Return the CUDA device for ``tensor`` (the current one for host tensors)."""
        if tensor.is_cuda:
            return tensor.device

        device = self._torch.device("cuda", self._torch.cuda.current_device())

        return device

    def frame_size(self, tensor: Any) -> Tuple[int, int]:
        """Return the encoded ``(width, height)`` for ``tensor``."""
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        size = encoded_size(width, height, max_width=self._max_width)

        return size

    def staging_size(self, tensor: Any) -> Tuple[int, int]:
        """Return the ``(width, height)`` of the slot ``tensor`` is copied into."""
        size = (int(tensor.shape[-1]), int(tensor.shape[-2]))

        return size

    def slot_bytes(self, width: int, height: int) -> int:
        """Return the device bytes of one RGB staging slot."""
        return 3 * width * height

    def scratch_bytes(self, width: int, height: int) -> int:
        """Return the device bytes a session's NV12, float and graph-source buffers take."""
        return width * height * 3 // 2 + width * height * 4 + width * height * 3

    def allocate_slots(
        self, count: int, width: int, height: int, device: Any
    ) -> List[_StagingSlot]:
        """Allocate ``count`` RGB staging slots on ``device`` in one block.

        One allocation (one ``cudaMalloc`` at most) instead of one per slot, and
        each slot's event is created now: the pool is filled at a stream's first
        frame, often while another stream's encoder session is opening, and
        per-slot allocations then stalled the pipeline thread for up to 20 ms.

        Returns:
            The slots; they share the block, which is freed with the last one.
        """
        if count <= 0:
            return []

        torch = self._torch
        block = torch.empty((count, 3, height, width), dtype=torch.uint8, device=device)
        stream = torch.cuda.current_stream(device)
        slots = []
        for index in range(count):
            event = torch.cuda.Event()
            event.record(stream)
            slots.append(_StagingSlot(rgb=block[index], event=event))

        return slots

    def stage(self, tensor: Any, slot: _StagingSlot, width: int, height: int) -> Any:
        """Copy ``tensor`` into ``slot`` on the caller's current stream.

        Two CUDA API calls: the device-to-device copy and the event record, both
        on the producer's stream. The copy runs before any later work the
        producer enqueues (in-place drawing, or the allocator reusing the frame's
        memory), so no reference to ``tensor`` is kept and no stream waits are
        needed on this thread.

        Returns:
            The slot's event; the encoder thread waits on it.
        """
        torch = self._torch
        if not tensor.is_cuda:
            # Not the expected input (the recorder picks NVENC for CUDA tensors);
            # upload rather than fail.
            tensor = tensor.to(slot.rgb.device)
        slot.rgb.copy_(_rgb_chw(tensor))
        slot.event.record(torch.cuda.current_stream(slot.rgb.device))

        return slot.event

    def open_session(
        self, width: int, height: int, fps: float, gop: int, bitrate: int, device: Any
    ) -> _Session:
        """Open one NVENC session on the calling (ring) thread."""
        torch = self._torch
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device)
        # Bind torch's primary context to this thread; never create a foreign one.
        torch.zeros(1, device=device)
        driver = _cuda_driver()
        context = driver.current_context()
        free_before = torch.cuda.mem_get_info(device)[0]
        encoder = self._nvc.CreateEncoder(
            width,
            height,
            "NV12",
            False,
            gpu_id=device.index or 0,
            cudacontext=context,
            cudastream=stream.cuda_stream,
            codec="h264",
            preset="P3",
            tuning_info="ultra_low_latency",
            rc="cbr",
            bitrate=bitrate,
            maxbitrate=bitrate,
            vbvbufsize=max(1, int(bitrate * VBV_SECONDS)),
            fps=max(1, int(round(fps))),
            gop=gop,
            bf=0,
            lookahead=0,
            repeatspspps=1,
        )
        if encoder.GetNumBFrames() != 0:
            del encoder
            raise RuntimeError("NVENC reports B-frames; refusing a reordered stream")

        vram_bytes = max(0, free_before - torch.cuda.mem_get_info(device)[0])
        with torch.cuda.stream(stream):
            nv12 = torch.empty(
                (height * 3 // 2, width), dtype=torch.uint8, device=device
            )
            work = torch.empty((height, width), dtype=torch.float32, device=device)
        session = _Session(
            encoder=encoder,
            device=device,
            stream=stream,
            context=context,
            width=width,
            height=height,
            vram_bytes=vram_bytes,
            nv12=nv12,
            work=work,
        )

        return session

    def encode(
        self,
        session: _Session,
        slot: _StagingSlot,
        ready: Any,
        ordinal: int,
        force_idr: bool,
    ) -> List[Tuple[int, bytes]]:
        """Convert and encode one slot; returns the access units NVENC released.

        Runs on the thread that opened the session, whose current CUDA device
        ``open_session`` set, so no per-frame device switch is needed.
        """
        torch = self._torch
        _cuda_driver().set_current(session.context)
        with torch.cuda.stream(session.stream):
            if ready is not None:
                session.stream.wait_event(ready)
            self._convert(session, slot.rgb)
            surface = _NV12Surface(
                session.nv12, session.width, session.height, session.stream.cuda_stream
            )
            params = self._nvc.NV_ENC_PIC_PARAMS()
            # NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS
            params.encodePicFlags = 6 if force_idr else 0
            # 2.0.5 returns encoder-local input ordinals as packet timestamps; we
            # pass the ordinal as the input timestamp so either semantics agree.
            params.inputTimeStamp = ordinal
            packets = session.encoder.Encode(surface, params)
            # The binding copies CUDA-array input into its own registered buffers;
            # once the stream is idle the slot and the NV12 buffer can be reused.
            session.stream.synchronize()
        access_units = _packets_to_bytes(packets)

        return access_units

    def flush(self, session: _Session) -> List[Tuple[int, bytes]]:
        """Drain the frames NVENC still holds (``EndEncode``)."""
        _cuda_driver().set_current(session.context)
        tail = session.encoder.EndEncode()
        session.stream.synchronize()
        access_units = _packets_to_bytes(tail)

        return access_units

    def destroy(self, session: _Session) -> None:
        """Release the NVENC session on the calling (ring) thread."""
        _cuda_driver().set_current(session.context)
        session.stream.synchronize()
        session.encoder = None
        session.converter = None
        session.nv12 = None
        session.work = None
        session.stream = None

    def conversion_mode(self, session: _Session) -> str:
        """Return ``graph``, ``eager`` or ``pending`` (no frame converted yet)."""
        return session.conversion

    def _convert(self, session: _Session, rgb: Any) -> None:
        # Session stream is current. Graph replay when one is captured for this
        # source shape, else capture it now; eager conversion as the fallback.
        converter = session.converter
        if converter is not None and converter.shape == tuple(rgb.shape):
            converter(rgb)
            return None

        if session.conversion != "eager":
            converter = self._capture_converter(session, rgb)
            if converter is not None:
                session.converter = converter
                session.conversion = "graph"
                converter(rgb)
                return None
            session.conversion = "eager"
        self._convert_eager(session, rgb)

    def _convert_eager(self, session: _Session, rgb: Any) -> None:
        fitted = self._fit(rgb, width=session.width, height=session.height)
        rgb_to_nv12(fitted, session.nv12, session.work)

    def _capture_converter(
        self, session: _Session, rgb: Any
    ) -> Optional[_GraphConverter]:
        # Captured once per session (and source shape) on the session's stream in
        # thread-local mode, so the host's threads keep using CUDA meanwhile. The
        # graph only writes session buffers (nv12, work) and its own pool.
        torch = self._torch
        try:
            source = torch.empty_like(rgb)
            source.copy_(rgb)
            self._convert_eager(session, source)  # loads the kernels first
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                self._convert_eager(session, source)
            finally:
                graph.capture_end()
        except Exception as error:
            LOGGER.warning(
                "NVENC conversion graph capture failed, converting eagerly: %r", error
            )
            return None

        converter = _GraphConverter(torch, source, graph)

        return converter

    def session_vram_bytes(self, session: _Session) -> int:
        """Return the VRAM the session took when it opened (free-memory delta)."""
        return session.vram_bytes

    def release_cached_memory(self) -> None:
        """Give device blocks the caching allocator holds back to the driver.

        Called once per ring, after the start-up pool's extension block was
        freed: torch keeps freed blocks cached, so without this the start-up
        headroom (up to 192 MiB per stream) would stay reserved for the life of
        the process. ``empty_cache`` is process-wide (it also returns the
        host's own unused cached blocks, which torch re-allocates on demand)
        and synchronises the device once; that is acceptable once per stream
        start-up and is never done per frame.
        """
        self._torch.cuda.empty_cache()

    def _fit(self, rgb: Any, *, width: int, height: int) -> Any:
        source_height, source_width = int(rgb.shape[1]), int(rgb.shape[2])
        if source_width - source_width % 2 == width and (
            source_height - source_height % 2 == height
        ):
            return rgb[:, :height, :width]
        scaled = self._torch.nn.functional.interpolate(
            rgb.unsqueeze(0).float(), size=(height, width), mode="area"
        )
        return scaled.squeeze(0)


def _packets_to_bytes(packets: Any) -> List[Tuple[int, bytes]]:
    access_units = []
    for packet in packets or []:
        access_units.append((int(packet["timestamp"]), bytes(packet["data"])))
    return access_units


@dataclass
class _StagedFrame:
    """A pool slot holding one copied frame (the ``PreparedFrame.payload``)."""

    slot: Any
    ready: Any
    size: Tuple[int, int]
    staging: Tuple[int, int]
    device: Any
    released: bool = False


_LIVE_RINGS: "weakref.WeakSet[NvencRing]" = weakref.WeakSet()


class NvencRing(BaseEncodedRing):
    """``Ring`` backend encoding one ``video_identifier`` with NVENC from CUDA tensors.

    The base ring owns the ingest queue, the encoder thread, grid decimation, PTS,
    the IDR cadence (one per output second), the packet store, captures, lifecycle
    and the MP4 remux. This class supplies the encoder: a GPU pool of RGB staging
    slots filled on the caller's thread with one copy each, and one NVENC session
    per ring, opened on the encoder thread at the first frame and destroyed there
    by ``_close_encoder``.

    Start-up: until the process warm-up (:class:`ProcessWarmUp`) is ready, frames
    are counted as ``startup_dropped`` and never block the caller; if the warm-up
    turns out unavailable, ``startup_failed`` becomes true and the owner should
    switch to another backend. Until the session is open the pool may grow to
    :func:`startup_pool_frames` slots; a frame refused in that window is also
    ``startup_dropped``. Afterwards a full pool counts ``queue_drops``.

    Args:
        video_identifier: The stream this ring records.
        pre_roll_seconds: Longest pre-roll the ring must retain.
        playback_speed: Output playback speed.
        clip_fps: Optional output frame-rate cap (grid).
        max_width: Optional output width cap; applied on the GPU.
        declared_fps: Declared source frame rate (rate-control hint).
        budget: Shared ``admission.Budget``.
        queue_depth: Ingest queue depth override; defaults to at least the
            start-up pool size (queued NVENC frames hold no host memory).
        byte_cap_bytes: Ring byte cap override.
        stall_grace_seconds: Seconds past the post-roll before a capture stalls.
        pool_frames: Steady-state GPU pool size; defaults to
            ``WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES`` (8).
        startup_pool_frames: Start-up pool size override; ``0`` disables the
            start-up headroom. Defaults to :func:`startup_pool_frames`.
        bitrate: CBR bits per second; defaults to :func:`resolve_bitrate` at the
            encoded size and rate-control fps.
        adapter: The torch / PyNvVideoCodec seam; defaults to
            :class:`PyNvVideoCodecAdapter`, built once the warm-up is ready.
        warm_up: The warm-up gating the first frames; defaults to the process
            warm-up when no ``adapter`` is given, else no gating.

    Raises:
        RuntimeError: If no adapter is given and the warm-up already reports
            NVENC unavailable.
    """

    backend = BACKEND_NAME

    def __init__(
        self,
        video_identifier: str,
        *,
        pre_roll_seconds: float,
        playback_speed: float = 1.0,
        clip_fps: Optional[float] = None,
        max_width: Optional[int] = None,
        declared_fps: Optional[float] = None,
        budget: Any = None,
        queue_depth: Optional[int] = None,
        byte_cap_bytes: Optional[int] = None,
        stall_grace_seconds: float = 10.0,
        pool_frames: Optional[int] = None,
        startup_pool_frames: Optional[int] = None,
        bitrate: Optional[int] = None,
        adapter: Optional[NvencAdapter] = None,
        warm_up: Optional[ProcessWarmUp] = None,
    ):
        adapter_factory: Optional[Callable[[], NvencAdapter]] = None
        if adapter is None:
            warm_up = warm_up or process_warm_up()
            if warm_up.status() == WARM_UP_UNAVAILABLE:
                raise RuntimeError(
                    "NVENC needs torch CUDA and PyNvVideoCodec: "
                    f"{warm_up.error or 'unavailable'}"
                )
            adapter_factory = functools.partial(
                PyNvVideoCodecAdapter, max_width=max_width
            )

        self._adapter = adapter
        self._adapter_factory = adapter_factory
        self._warm_up = warm_up
        self._warmed = warm_up is None
        self._startup_failed = False
        self._pool_frames = pool_frames or resolve_pool_frames()
        self._startup_pool_override = startup_pool_frames
        self._bitrate_override = bitrate
        self._pool_lock = threading.Lock()
        self._pool_size: Optional[Tuple[int, int]] = None
        self._pool_limit_startup = self._pool_frames
        self._free_slots: List[Any] = []
        self._extension_slots: set = set()
        self._extension_allocated = 0
        # The start-up extension block, once every slot of it is gone, is handed
        # back to the driver by the encoder thread (``release_cached_memory``).
        self._startup_release_pending = False
        self._startup_pool_released = False
        self._allocated_slots = 0
        self._peak_in_use = 0
        self._pool_drops = 0
        self._stage_errors = 0
        # Pool-lock state shared with the encoder thread: is a session open, and
        # is the start-up headroom still in force (until the backlog drains).
        self._encoder_open = False
        self._startup_headroom = True
        self._scratch_size: Optional[Tuple[int, int]] = None
        self._first_session_open_ms: Optional[float] = None
        self._last_session_open_ms: Optional[float] = None

        # Encoder-thread state.
        self._session: Any = None
        self._session_flushed = False
        self._session_size: Optional[Tuple[int, int]] = None
        self._session_vram_bytes = 0
        self._sessions_opened = 0
        self._sessions_destroyed = 0
        self._bitrate: Optional[int] = None
        self._gop: Optional[int] = None
        self._next_ordinal = 0
        self._forced: Deque[Tuple[int, bool]] = deque()
        self._forced_idrs = 0
        self._keyframes = 0
        self._last_conversion: Optional[str] = None

        if queue_depth is None:
            queue_depth = max(
                DEFAULT_QUEUE_DEPTH,
                self._pool_frames,
                STARTUP_POOL_MAX_FRAMES,
                startup_pool_frames or 0,
            )
        super().__init__(
            video_identifier,
            pre_roll_seconds=pre_roll_seconds,
            playback_speed=playback_speed,
            clip_fps=clip_fps,
            max_width=max_width,
            declared_fps=declared_fps,
            budget=budget,
            queue_depth=queue_depth,
            byte_cap_bytes=byte_cap_bytes,
            stall_grace_seconds=stall_grace_seconds,
        )
        _LIVE_RINGS.add(self)

    @property
    def startup_failed(self) -> bool:
        """True once the warm-up reported NVENC unavailable for this ring."""
        return self._startup_failed

    def close(self) -> None:
        """Close the ring (see ``BaseEncodedRing.close``) and drop the exit hook."""
        super().close()
        _LIVE_RINGS.discard(self)

    def _after_release(self) -> None:
        # A retired ring released itself: its session was destroyed on the
        # encoder thread, so the exit hook no longer needs it.
        _LIVE_RINGS.discard(self)

    # ---- caller thread ---------------------------------------------------------
    def _prepare_frame(self, image: Any) -> Optional[PreparedFrame]:
        # Pool slot, then ONE copy + ONE event record on the caller's stream.
        # Refusals while the encoder is still starting raise EncoderStartingError
        # (counted as startup_dropped); a full pool afterwards returns None
        # (queue_drops). No reference to the image survives this call.
        adapter = self._ready_adapter()
        tensor = getattr(image, "tensor_image", image)
        width, height = adapter.frame_size(tensor)
        staging = adapter.staging_size(tensor)
        device = adapter.tensor_device(tensor)
        slot = self._acquire_slot(staging, device=device)
        if slot is None:
            with self._pool_lock:
                encoder_open = self._encoder_open
            if not encoder_open:
                raise EncoderStartingError("the NVENC session is still opening")
            return None

        staged = _StagedFrame(
            slot=slot,
            ready=None,
            size=(width, height),
            staging=staging,
            device=device,
        )
        try:
            staged.ready = adapter.stage(tensor, slot, width, height)
        except Exception:
            self._release_slot(staged)
            with self._pool_lock:
                self._stage_errors += 1
            raise

        # Device memory is not host memory: the host byte budget does not count it.
        prepared = PreparedFrame(payload=staged, width=width, height=height, nbytes=0)

        return prepared

    def _ready_adapter(self) -> NvencAdapter:
        if self._warmed:
            return self._adapter

        status = self._warm_up.status()
        if status == WARM_UP_READY:
            if self._adapter is None:
                self._adapter = self._adapter_factory()
            self._warmed = True
            return self._adapter

        if status == WARM_UP_UNAVAILABLE:
            self._startup_failed = True
        raise EncoderStartingError(f"the NVENC warm-up is {status}")

    def _discard_frame(self, frame: PreparedFrame) -> None:
        self._release_slot(frame.payload)

    # ---- encoder thread --------------------------------------------------------
    def _encode_frame(
        self, frame: PreparedFrame, *, pts: int, force_keyframe: bool
    ) -> List[EncodedOutput]:
        staged: _StagedFrame = frame.payload
        if self._session is not None and (
            self._session_flushed or self._session_size != staged.size
        ):
            self._close_encoder()
        if self._session is None:
            self._open_session(size=staged.size, device=staged.device)

        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._forced.append((ordinal, force_keyframe))
        try:
            access_units = self._adapter.encode(
                self._session, staged.slot, staged.ready, ordinal, force_keyframe
            )
        except Exception:
            self._forced.pop()
            raise

        # The encoder is done with the slot (stream synchronised): reuse it now.
        self._release_slot(staged)
        self._release_startup_pool_memory()
        if force_keyframe:
            self._forced_idrs += 1
        outputs = self._to_outputs(access_units)

        return outputs

    def _flush_encoder(self) -> List[EncodedOutput]:
        if self._session is None or self._session_flushed:
            return []

        self._session_flushed = True
        access_units = self._adapter.flush(self._session)
        outputs = self._to_outputs(access_units)

        return outputs

    def _close_encoder(self) -> None:
        session = self._session
        if session is not None and hasattr(self._adapter, "conversion_mode"):
            try:
                self._last_conversion = self._adapter.conversion_mode(session)
            except Exception:
                pass
        self._session = None
        self._session_flushed = False
        self._session_size = None
        self._forced.clear()
        if session is None:
            return None

        try:
            self._adapter.destroy(session)
        finally:
            with self._pool_lock:
                self._sessions_destroyed += 1
                self._session_vram_bytes = 0
                self._scratch_size = None
                self._encoder_open = False

    def _backend_stats(self) -> Dict[str, Any]:
        session, adapter = self._session, self._adapter
        conversion = None
        if session is not None and hasattr(adapter, "conversion_mode"):
            try:
                conversion = adapter.conversion_mode(session)
            except Exception:
                conversion = None
        with self._pool_lock:
            size = self._pool_size
            allocated = self._allocated_slots
            details = {
                "pool_frames": self._pool_frames,
                "startup_pool_frames": self._pool_limit_startup,
                "pool_allocated_frames": allocated,
                "pool_peak_frames": self._peak_in_use,
                "pool_free_frames": len(self._free_slots),
                "pool_drops": self._pool_drops,
                "startup_pool_released": self._startup_pool_released,
                "stage_errors": self._stage_errors,
                "sessions_open": self._sessions_opened - self._sessions_destroyed,
                "sessions_opened": self._sessions_opened,
                "sessions_destroyed": self._sessions_destroyed,
                "session_open_ms_first": self._first_session_open_ms,
                "session_open_ms_last": self._last_session_open_ms,
                "bitrate": self._bitrate,
                "gop": self._gop,
                "forced_idrs": self._forced_idrs,
                "keyframes": self._keyframes,
                "startup_failed": self._startup_failed,
                "conversion": conversion or self._last_conversion,
            }
            vram_bytes = self._session_vram_bytes
            scratch_size = self._scratch_size
        adapter = self._adapter
        if adapter is not None and size is not None:
            vram_bytes += allocated * adapter.slot_bytes(size[0], size[1])
        if adapter is not None and scratch_size is not None:
            vram_bytes += adapter.scratch_bytes(scratch_size[0], scratch_size[1])
        details["vram_bytes_estimate"] = vram_bytes
        if self._warm_up is not None:
            details["warm_up"] = self._warm_up.report()

        return details

    # ---- helpers ---------------------------------------------------------------
    def _acquire_slot(self, staging: Tuple[int, int], *, device: Any) -> Optional[Any]:
        width, height = staging
        with self._pool_lock:
            if self._pool_size == staging:
                return self._take_free_slot_locked()
            # First frame at this size: forget idle slots (outstanding ones are
            # dropped on return) and fill the whole pool below, outside the lock.
            self._pool_size = staging
            self._free_slots = []
            self._extension_slots = set()
            self._allocated_slots = 0
            self._peak_in_use = 0
            limit = self._startup_limit(width, height)
            self._pool_limit_startup = limit
            extension = limit - self._pool_frames if self._startup_headroom else 0

        try:
            steady = self._adapter.allocate_slots(
                self._pool_frames, width, height, device
            )
            extra = self._adapter.allocate_slots(extension, width, height, device)
        except Exception:
            with self._pool_lock:
                if self._pool_size == staging:
                    self._pool_size = None
            raise

        with self._pool_lock:
            if self._pool_size != staging:
                return None
            self._free_slots = list(reversed(steady)) + list(reversed(extra))
            self._extension_slots = {id(slot) for slot in extra}
            self._extension_allocated += len(extra)
            self._allocated_slots = len(steady) + len(extra)
            slot = self._take_free_slot_locked()

        return slot

    def _take_free_slot_locked(self) -> Optional[Any]:
        if not self._free_slots:
            self._pool_drops += 1
            return None

        slot = self._free_slots.pop()
        in_use = self._allocated_slots - len(self._free_slots)
        self._peak_in_use = max(self._peak_in_use, in_use)

        return slot

    def _startup_limit(self, width: int, height: int) -> int:
        # Pool lock held.
        if self._startup_pool_override is not None:
            return max(self._pool_frames, self._startup_pool_override)

        limit = startup_pool_frames(
            self._pool_frames, self._adapter.slot_bytes(width, height)
        )

        return limit

    def _release_slot(self, staged: _StagedFrame) -> None:
        with self._pool_lock:
            if staged.released:
                return None
            staged.released = True
            slot, staged.slot, staged.ready = staged.slot, None, None
            if staged.staging != self._pool_size:
                return None
            if not self._startup_headroom and id(slot) in self._extension_slots:
                # Start-up is over: let the extension block go.
                self._extension_slots.discard(id(slot))
                self._allocated_slots -= 1
                self._mark_extension_gone_locked()
                return None
            self._free_slots.append(slot)
            self._end_startup_headroom_locked()

    def _end_startup_headroom_locked(self) -> None:
        # The headroom lasts until the session is open and the start-up backlog
        # has drained to the steady-state pool; then the extension slots are
        # dropped (in-use ones when they come back) and their block is freed.
        if not self._startup_headroom or not self._encoder_open:
            return None

        in_use = self._allocated_slots - len(self._free_slots)
        if in_use > self._pool_frames:
            return None

        self._startup_headroom = False
        kept = [
            slot for slot in self._free_slots if id(slot) not in self._extension_slots
        ]
        dropped = [
            slot for slot in self._free_slots if id(slot) in self._extension_slots
        ]
        for slot in dropped:
            self._extension_slots.discard(id(slot))
        self._free_slots = kept
        self._allocated_slots -= len(dropped)
        self._mark_extension_gone_locked()

    def _mark_extension_gone_locked(self) -> None:
        # Once no slot of the extension block is referenced, torch has the block
        # back in its cache; the encoder thread then returns it to the driver.
        if self._extension_slots or self._extension_allocated == 0:
            return None
        if not self._startup_pool_released:
            self._startup_release_pending = True

    def _release_startup_pool_memory(self) -> None:
        # Encoder thread, pool lock not held: at most once per ring.
        with self._pool_lock:
            if not self._startup_release_pending:
                return None
            self._startup_release_pending = False
            self._startup_pool_released = True
        release = getattr(self._adapter, "release_cached_memory", None)
        if release is None:
            return None
        try:
            release()
        except Exception as error:
            LOGGER.warning("Releasing the NVENC start-up pool failed: %r", error)

    def _open_session(self, *, size: Tuple[int, int], device: Any) -> None:
        width, height = size
        fps = self._rate_control_fps()
        bitrate = self._bitrate_override or resolve_bitrate(width, height, fps)
        speed = self.playback_speed if self.playback_speed > 0 else 1.0
        gop = max(1, int(round(fps * speed * BACKSTOP_GOP_OUTPUT_SECONDS)))
        started = time.perf_counter()
        session = self._adapter.open_session(width, height, fps, gop, bitrate, device)
        open_ms = (time.perf_counter() - started) * 1000
        vram_bytes = self._adapter.session_vram_bytes(session)
        self._session = session
        self._session_flushed = False
        self._session_size = size
        # Encoder-local input ordinals restart with every session (binding 2.0.5).
        self._next_ordinal = 0
        self._forced.clear()
        with self._pool_lock:
            self._bitrate = bitrate
            self._gop = gop
            self._session_vram_bytes = vram_bytes
            self._sessions_opened += 1
            self._scratch_size = size
            self._encoder_open = True
            if self._first_session_open_ms is None:
                self._first_session_open_ms = open_ms
            self._last_session_open_ms = open_ms
        LOGGER.info(
            "NVENC ring %s opened %dx%d at %d bit/s, %.0f fps, gop %d, ~%d bytes "
            "VRAM in %.0f ms",
            self.video_identifier,
            width,
            height,
            bitrate,
            fps,
            gop,
            vram_bytes,
            open_ms,
        )

    def _rate_control_fps(self) -> float:
        # The kept frame rate: declared source fps capped by clip_fps, else 30.
        rates = [
            float(rate)
            for rate in (self.declared_fps, self.clip_fps)
            if isinstance(rate, (int, float)) and rate > 0
        ]
        fps = min(MAX_FPS, max(1.0, min(rates))) if rates else DEFAULT_FPS

        return fps

    def _to_outputs(self, access_units: List[Tuple[int, bytes]]) -> List[EncodedOutput]:
        outputs = []
        for ordinal, data in access_units:
            forced = self._pop_forced(ordinal)
            if not data:
                continue
            keyframe = is_idr_access_unit(data)
            if not keyframe and forced and b"\x00\x00\x01" not in data:
                keyframe = True
            self._keyframes += int(keyframe)
            outputs.append(EncodedOutput(payload=data, is_keyframe=keyframe))
        return outputs

    def _pop_forced(self, ordinal: int) -> bool:
        # Outputs are FIFO (bf=0); the ordinal only recovers the forced-IDR flag.
        while self._forced and self._forced[0][0] < ordinal:
            self._forced.popleft()
        if self._forced and self._forced[0][0] == ordinal:
            return self._forced.popleft()[1]
        if self._forced:
            return self._forced.popleft()[1]
        return False


def _close_live_rings() -> None:
    # A live NVENC session at interpreter exit segfaults (G1.2, G1.8b): close
    # every ring that is still open, with a short deadline. A warm-up still
    # running holds a probe session of its own: give it the same deadline.
    _PROCESS_WARM_UP.wait_if_running(ATEXIT_CLOSE_DEADLINE_SECONDS)
    for ring in list(_LIVE_RINGS):
        try:
            ring.prepare_close(ATEXIT_CLOSE_DEADLINE_SECONDS)
            ring.close()
        except Exception as error:
            LOGGER.warning("NVENC ring close at exit failed: %r", error)


atexit.register(_close_live_rings)

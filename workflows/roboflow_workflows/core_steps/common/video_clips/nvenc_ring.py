"""NVENC encoder ring fed straight from CUDA tensors (PyNvVideoCodec).

``NvencRing`` is the ``Ring`` backend the Video Clip Recorder selects when the
workflow image is a CUDA tensor and :func:`available` is true (Serverless Streaming
GPU runners, where every frame is a CHW RGB uint8 CUDA tensor and a numpy download
costs 17-18 ms per 1080p frame on the pipeline thread). No frame is copied to the
CPU; only the encoded access units are.

The shared ``ring.BaseEncodedRing`` owns the ingest queue, the encoder thread, grid
decimation, PTS, the IDR cadence (one per output second), the packet store with
GOP-aligned captures, lifecycle and the MP4 remux. This module adds the encoder:

* Caller thread (``_prepare_frame``): take a free slot of the GPU pool
  (``WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES``, default 8) or report the pool full
  (the base counts ``queue_drops``); convert the tensor to NV12 into the slot on
  the ring's own CUDA stream, ordered after the producer and before any later
  in-place drawing on the caller's stream. No synchronize, no encode, no host copy.
* Encoder thread (``_encode_frame`` / ``_flush_encoder`` / ``_close_encoder``): one
  NVENC session per ring, opened lazily at the first frame with the Serverless
  Streaming runtime's settings (rf-video-processor ``nvenc_encoders.NvEncoder``):
  P3, ultra-low-latency tuning, CBR with a 250 ms VBV, ``bf=0``, no lookahead,
  SPS/PPS repeated on every IDR, torch's primary CUDA context and a private CUDA
  stream. IDRs are forced when the base asks; keyframe flags are read back from
  the access units (IDR NAL). The session is destroyed on the encoder thread before
  it exits; a session still alive at interpreter exit segfaults, so an ``atexit``
  hook also closes any ring left open.

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
import logging
import os
import threading
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Protocol, Tuple

from roboflow_workflows.core_steps.common.video_clips.ring import (
    BaseEncodedRing,
    EncodedOutput,
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
    raw = os.getenv(BITRATE_KBPS_ENV, "").strip()
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
    raw = os.getenv(POOL_FRAMES_ENV, "").strip()
    if not raw:
        return DEFAULT_POOL_FRAMES

    try:
        frames = int(float(raw))
    except ValueError:
        LOGGER.warning("Ignoring non-numeric %s=%r", POOL_FRAMES_ENV, raw)
        return DEFAULT_POOL_FRAMES

    frames = min(MAX_POOL_FRAMES, max(1, frames))

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

    ``frame_size``, ``allocate_slot`` and ``stage`` run on the pipeline thread;
    ``open_session``, ``encode``, ``flush`` and ``destroy`` run on the ring thread
    only. ``encode``/``flush`` return ``(input_ordinal, access_unit)`` pairs in
    output order; the ring maps ordinals back to frames.
    """

    def frame_size(self, tensor: Any) -> Tuple[int, int]: ...

    def allocate_slot(self, width: int, height: int, device: Any) -> Any: ...

    def stage(self, tensor: Any, slot: Any, width: int, height: int) -> Any: ...

    def open_session(
        self, width: int, height: int, fps: float, gop: int, bitrate: int, device: Any
    ) -> Any: ...

    def encode(
        self, session: Any, slot: Any, ready: Any, ordinal: int, force_idr: bool
    ) -> List[Tuple[int, bytes]]: ...

    def flush(self, session: Any) -> List[Tuple[int, bytes]]: ...

    def destroy(self, session: Any) -> None: ...

    def tensor_device(self, tensor: Any) -> Any: ...

    def slot_bytes(self, width: int, height: int) -> int: ...

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


@dataclass
class _Session:
    encoder: Any
    device: Any
    stream: Any
    context: int
    width: int
    height: int
    vram_bytes: int


class PyNvVideoCodecAdapter:
    """Real ``NvencAdapter``: torch on a private CUDA stream + PyNvVideoCodec NVENC.

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
        self._driver: Optional[_CudaDriver] = None
        self._conversion_streams: dict = {}
        self._work: Optional[Tuple[tuple, Any]] = None

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

    def slot_bytes(self, width: int, height: int) -> int:
        """Return the device bytes of one NV12 slot."""
        return width * height * 3 // 2

    def allocate_slot(self, width: int, height: int, device: Any) -> Any:
        """Allocate one NV12 slot on ``device``."""
        slot = self._torch.empty(
            (height * 3 // 2, width), dtype=self._torch.uint8, device=device
        )

        return slot

    def stage(self, tensor: Any, slot: Any, width: int, height: int) -> Any:
        """Convert ``tensor`` into ``slot`` on the ring's CUDA stream (no host sync).

        The conversion stream waits for the caller's current stream (the producer),
        and the caller's stream then waits for the conversion, so later in-place
        drawing on the image cannot race the read. ``record_stream`` keeps the
        caching allocator from recycling the image while the conversion reads it.

        Returns:
            A CUDA event recorded after the conversion; the encoder waits on it.
        """
        torch = self._torch
        if not tensor.is_cuda:
            # Not the expected input (the recorder picks NVENC for CUDA tensors);
            # upload rather than fail.
            tensor = tensor.to(self.tensor_device(tensor))
        device = tensor.device
        producer = torch.cuda.current_stream(device)
        stream = self._conversion_stream(device)
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            rgb = self._fit(_rgb_chw(tensor), width=width, height=height)
            rgb_to_nv12(rgb, slot, self._work_buffer(width, height, device))
        tensor.record_stream(stream)
        ready = torch.cuda.Event()
        ready.record(stream)
        producer.wait_event(ready)

        return ready

    def open_session(
        self, width: int, height: int, fps: float, gop: int, bitrate: int, device: Any
    ) -> _Session:
        """Open one NVENC session on the calling (ring) thread."""
        torch = self._torch
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device)
        # Bind torch's primary context to this thread; never create a foreign one.
        torch.zeros(1, device=device)
        if self._driver is None:
            self._driver = _CudaDriver()
        context = self._driver.current_context()
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
        session = _Session(
            encoder=encoder,
            device=device,
            stream=stream,
            context=context,
            width=width,
            height=height,
            vram_bytes=vram_bytes,
        )

        return session

    def encode(
        self, session: _Session, slot: Any, ready: Any, ordinal: int, force_idr: bool
    ) -> List[Tuple[int, bytes]]:
        """Encode one slot; returns the access units NVENC released."""
        torch = self._torch
        self._driver.set_current(session.context)
        with torch.cuda.device(session.device), torch.cuda.stream(session.stream):
            if ready is not None:
                session.stream.wait_event(ready)
            surface = _NV12Surface(
                slot, session.width, session.height, session.stream.cuda_stream
            )
            params = self._nvc.NV_ENC_PIC_PARAMS()
            # NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS
            params.encodePicFlags = 6 if force_idr else 0
            # 2.0.5 returns encoder-local input ordinals as packet timestamps; we
            # pass the ordinal as the input timestamp so either semantics agree.
            params.inputTimeStamp = ordinal
            packets = session.encoder.Encode(surface, params)
            # The binding copies CUDA-array input into its own registered buffers;
            # after this the slot can be reused.
            session.stream.synchronize()
        access_units = _packets_to_bytes(packets)

        return access_units

    def flush(self, session: _Session) -> List[Tuple[int, bytes]]:
        """Drain the frames NVENC still holds (``EndEncode``)."""
        self._driver.set_current(session.context)
        tail = session.encoder.EndEncode()
        session.stream.synchronize()
        access_units = _packets_to_bytes(tail)

        return access_units

    def destroy(self, session: _Session) -> None:
        """Release the NVENC session on the calling (ring) thread."""
        if self._driver is not None:
            self._driver.set_current(session.context)
        session.stream.synchronize()
        session.encoder = None
        session.stream = None

    def session_vram_bytes(self, session: _Session) -> int:
        """Return the VRAM the session took when it opened (free-memory delta)."""
        return session.vram_bytes

    def _conversion_stream(self, device: Any) -> Any:
        stream = self._conversion_streams.get(device)
        if stream is None:
            stream = self._torch.cuda.Stream(device=device)
            self._conversion_streams[device] = stream
        return stream

    def _work_buffer(self, width: int, height: int, device: Any) -> Any:
        key = (height, width, str(device))
        if self._work is None or self._work[0] != key:
            buffer = self._torch.empty(
                (height, width), dtype=self._torch.float32, device=device
            )
            self._work = (key, buffer)
        return self._work[1]

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
    """A pool slot holding one converted NV12 frame (the ``PreparedFrame.payload``)."""

    slot: Any
    ready: Any
    size: Tuple[int, int]
    device: Any
    released: bool = False


_LIVE_RINGS: "weakref.WeakSet[NvencRing]" = weakref.WeakSet()


class NvencRing(BaseEncodedRing):
    """``Ring`` backend encoding one ``video_identifier`` with NVENC from CUDA tensors.

    The base ring owns the ingest queue, the encoder thread, grid decimation, PTS,
    the IDR cadence (one per output second), the packet store, captures, lifecycle
    and the MP4 remux. This class supplies the encoder: a GPU pool of NV12 slots
    filled on the caller's thread and one NVENC session per ring, opened lazily on
    the encoder thread and destroyed there by ``_close_encoder``.

    Args:
        video_identifier: The stream this ring records.
        pre_roll_seconds: Longest pre-roll the ring must retain.
        playback_speed: Output playback speed.
        clip_fps: Optional output frame-rate cap (grid).
        max_width: Optional output width cap; applied on the GPU.
        declared_fps: Declared source frame rate (rate-control hint).
        budget: Shared ``admission.Budget``.
        queue_depth: Ingest queue depth override.
        byte_cap_bytes: Ring byte cap override.
        stall_grace_seconds: Seconds past the post-roll before a capture stalls.
        pool_frames: GPU pool size; defaults to
            ``WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES`` (8). A full pool counts
            ``queue_drops``.
        bitrate: CBR bits per second; defaults to :func:`resolve_bitrate` at the
            encoded size and rate-control fps.
        adapter: The torch / PyNvVideoCodec seam; defaults to
            :class:`PyNvVideoCodecAdapter`.

    Raises:
        RuntimeError: If no adapter is given and :func:`available` is false.
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
        bitrate: Optional[int] = None,
        adapter: Optional[NvencAdapter] = None,
    ):
        if adapter is None:
            if not available():
                raise RuntimeError("NVENC needs torch CUDA and PyNvVideoCodec")
            adapter = PyNvVideoCodecAdapter(max_width=max_width)

        self._adapter = adapter
        self._pool_frames = pool_frames or resolve_pool_frames()
        self._bitrate_override = bitrate
        self._pool_lock = threading.Lock()
        self._pool_size: Optional[Tuple[int, int]] = None
        self._free_slots: List[Any] = []
        self._allocated_slots = 0
        self._pool_drops = 0
        self._stage_errors = 0

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

    def close(self) -> None:
        """Close the ring (see ``BaseEncodedRing.close``) and drop the exit hook."""
        super().close()
        _LIVE_RINGS.discard(self)

    # ---- caller thread ---------------------------------------------------------
    def _prepare_frame(self, image: Any) -> Optional[PreparedFrame]:
        # Pool slot (or None = full, the base counts queue_drops), then an async
        # RGB->NV12 conversion on the ring's CUDA stream. No reference to the
        # image survives this call (the allocator holds the memory via
        # record_stream until the conversion ran).
        tensor = getattr(image, "tensor_image", image)
        width, height = self._adapter.frame_size(tensor)
        device = self._adapter.tensor_device(tensor)
        slot = self._acquire_slot(width=width, height=height, device=device)
        if slot is None:
            return None

        staged = _StagedFrame(
            slot=slot, ready=None, size=(width, height), device=device
        )
        try:
            staged.ready = self._adapter.stage(tensor, slot, width, height)
        except Exception:
            self._release_slot(staged)
            with self._pool_lock:
                self._stage_errors += 1
            raise

        # Device memory is not host memory: the host byte budget does not count it.
        prepared = PreparedFrame(payload=staged, width=width, height=height, nbytes=0)

        return prepared

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

        # The encoder copied the slot into its own input buffer: reuse it now.
        self._release_slot(staged)
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
        session, self._session = self._session, None
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

    def _backend_stats(self) -> Dict[str, Any]:
        with self._pool_lock:
            size = self._pool_size
            allocated = self._allocated_slots
            details = {
                "pool_frames": self._pool_frames,
                "pool_allocated_frames": allocated,
                "pool_free_frames": len(self._free_slots),
                "pool_drops": self._pool_drops,
                "stage_errors": self._stage_errors,
                "sessions_open": self._sessions_opened - self._sessions_destroyed,
                "sessions_opened": self._sessions_opened,
                "sessions_destroyed": self._sessions_destroyed,
                "bitrate": self._bitrate,
                "gop": self._gop,
                "forced_idrs": self._forced_idrs,
                "keyframes": self._keyframes,
            }
            vram_bytes = self._session_vram_bytes
        if size is not None:
            vram_bytes += allocated * self._adapter.slot_bytes(size[0], size[1])
            vram_bytes += size[0] * size[1] * 4
        details["vram_bytes_estimate"] = vram_bytes

        return details

    # ---- helpers ---------------------------------------------------------------
    def _acquire_slot(self, *, width: int, height: int, device: Any) -> Optional[Any]:
        with self._pool_lock:
            if self._pool_size != (width, height):
                # New size: forget idle slots; outstanding ones are dropped on return.
                self._pool_size = (width, height)
                self._free_slots = []
                self._allocated_slots = 0
            if self._free_slots:
                return self._free_slots.pop()
            if self._allocated_slots >= self._pool_frames:
                self._pool_drops += 1
                return None
            self._allocated_slots += 1

        try:
            slot = self._adapter.allocate_slot(width, height, device)
        except Exception:
            with self._pool_lock:
                if self._pool_size == (width, height):
                    self._allocated_slots -= 1
            raise

        return slot

    def _release_slot(self, staged: _StagedFrame) -> None:
        with self._pool_lock:
            if staged.released:
                return None
            staged.released = True
            if staged.size == self._pool_size:
                self._free_slots.append(staged.slot)
            staged.slot = None
            staged.ready = None

    def _open_session(self, *, size: Tuple[int, int], device: Any) -> None:
        width, height = size
        fps = self._rate_control_fps()
        bitrate = self._bitrate_override or resolve_bitrate(width, height, fps)
        speed = self.playback_speed if self.playback_speed > 0 else 1.0
        gop = max(1, int(round(fps * speed * BACKSTOP_GOP_OUTPUT_SECONDS)))
        session = self._adapter.open_session(width, height, fps, gop, bitrate, device)
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
        LOGGER.info(
            "NVENC ring %s opened %dx%d at %d bit/s, %.0f fps, gop %d, ~%d bytes VRAM",
            self.video_identifier,
            width,
            height,
            bitrate,
            fps,
            gop,
            vram_bytes,
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
    # every ring that is still open, with a short deadline.
    for ring in list(_LIVE_RINGS):
        try:
            ring.prepare_close(ATEXIT_CLOSE_DEADLINE_SECONDS)
            ring.close()
        except Exception as error:
            LOGGER.warning("NVENC ring close at exit failed: %r", error)


atexit.register(_close_live_rings)

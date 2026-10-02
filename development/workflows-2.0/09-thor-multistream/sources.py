"""Jetson NVDEC sources feeding one latest-frame slot each.

```
RTSP ─ nvv4l2decoder ─ native bridge slot (latest wins, 1 tensor)
     ─ VideoSource capture thread ─ VideoSource buffer (1 frame, DROP_OLDEST)
     ─ reader thread ─ runner slot (1 frame, newest replaces) ─ take_next()
```

Every source always runs its capture and reader threads. The runner takes
frames round-robin from the slots, so admission is the same in every mode.
Drops are counted at each of the three layers above; anything lost before the
native bridge (RTSP jitter buffer, decoder) is not observable here.

The decoder is forced: ``VideoSource`` gets a callable that builds a
``JetsonVideoFrameProducer``. With a callable reference ``VideoSource`` has no
cv2 fallback, so a decoder failure fails the run. Every frame is checked to be
a CUDA ``uint8`` CHW RGB tensor.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from streamvision.camera.buffer_strategies import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from streamvision.camera.entities import StatusUpdate, VideoFrame
from streamvision.camera.exceptions import (
    EndOfStreamError,
    StreamOperationNotAllowedError,
)
from streamvision.camera.jetson_producer import JetsonVideoFrameProducer
from streamvision.camera.jetson_tensor_bridge import jetson_tensor_bridge_available
from streamvision.camera.source_reference_sanitizer import sanitize_source_reference
from streamvision.camera.video_source import FRAME_DROPPED_EVENT, VideoSource

READ_POLL_SECONDS = 0.25
TERMINATE_RETRY_SECONDS = 0.05
NATIVE_STATS_KEYS = (
    "frames",
    "nvmm_frames",
    "frames_dropped_by_consumer",
    "host_pixel_maps",
    "host_to_device_copies",
    "device_to_host_copies",
    "array_flatten_copies",
    "conversion_kernels",
    "egl_cache_hits",
    "egl_cache_misses",
    "unique_buffer_fds",
    "sync_ns",
    "sync_max_ns",
)
HOST_FALLBACK_STATS_KEYS = (
    "host_pixel_maps",
    "host_to_device_copies",
    "device_to_host_copies",
    "array_flatten_copies",
)
"""Native counters that stay zero on the zero-copy path."""


@dataclass
class ArrivedFrame:
    """A decoded frame handed to the runner.

    Holding this object keeps the native tensor (a bridge pool buffer) alive.

    Attributes:
        source_index: Index of the source in the run.
        frame: The streamvision frame; ``frame.image`` is the CUDA tensor and
            ``frame.frame_id`` the VideoSource grab counter.
        arrival_ns: ``time.monotonic_ns()`` when the reader thread received the
            frame from VideoSource (decoder handoff, not camera capture).
    """

    source_index: int
    frame: VideoFrame
    arrival_ns: int


@dataclass
class SourceCounters:
    """Per-source frame counts; mutated under the shared wakeup lock.

    Attributes:
        arrived: Frames the reader thread put into the runner slot.
        replaced_in_slot: Frames replaced in the runner slot before admission.
        taken: Frames admitted by the runner.
        video_source_dropped: VideoSource DROP_OLDEST drops (status events).
        discarded_at_stop: Frames read after stop was requested.
    """

    arrived: int = 0
    replaced_in_slot: int = 0
    taken: int = 0
    video_source_dropped: int = 0
    discarded_at_stop: int = 0


@dataclass
class SourceEnd:
    """Why a source stopped producing frames.

    Attributes:
        reason: ``end_of_stream``, ``source_error`` or ``invalid_frame``.
        detail: Human-readable detail.
    """

    reason: str
    detail: str


@dataclass
class _StopResult:
    native_stats: Optional[Dict[str, int]] = None
    errors: List[str] = field(default_factory=list)


class JetsonSource:
    """One forced-NVDEC source with an always-running reader thread.

    Args:
        index: Source index, used as VideoSource ``source_id``.
        url: Stream reference.
        wakeup: Condition shared by all sources and the runner; its lock
            guards the slot and counters, and it is notified on every arrival.
        device: CUDA device for the native bridge, e.g. ``cuda:0``.
    """

    def __init__(
        self,
        index: int,
        *,
        url: str,
        wakeup: threading.Condition,
        device: str,
    ):
        self.index = index
        self.counters = SourceCounters()
        self.end: Optional[SourceEnd] = None
        self._url = url
        self._wakeup = wakeup
        self._device = device
        self._slot: Optional[ArrivedFrame] = None
        self._stopping = False
        self._producer: Optional[JetsonVideoFrameProducer] = None
        self._video_source: Optional[VideoSource] = None
        self._reader: Optional[threading.Thread] = None
        self._first_frame_facts: Optional[dict] = None

    @property
    def display_url(self) -> str:
        """Stream reference with credentials removed."""
        return sanitize_source_reference(self._url)

    def start(self) -> None:
        """Open the stream, wait for its first frame and start the reader.

        Raises:
            Exception: Any decoder, connection or bridge error; there is no
                fallback decoder.
        """
        self._video_source = VideoSource.init(
            video_reference=self._build_producer,
            buffer_size=1,
            status_update_handlers=[self._on_status_update],
            buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
            buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
            source_id=self.index,
            allow_tensor_frames=True,
        )
        self._video_source.start()

        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"source-{self.index}-reader",
            daemon=True,
        )
        self._reader.start()

    def take_locked(self) -> Optional[ArrivedFrame]:
        """Take the slot's frame; the caller holds the wakeup lock.

        Returns:
            The newest unadmitted frame, or None when the slot is empty.
        """
        arrived = self._slot
        if arrived is None:
            return None

        self._slot = None
        self.counters.taken += 1

        return arrived

    def native_stats(self) -> Optional[Dict[str, int]]:
        """Read the native bridge counters, or None once the producer closed."""
        if self._producer is None:
            return None
        try:
            stats = self._producer.tensor_bridge_stats
        except RuntimeError:
            return None

        selected = {key: stats[key] for key in NATIVE_STATS_KEYS if key in stats}

        return selected

    def declared_fps(self) -> Optional[float]:
        """FPS the stream announces (from the decoder caps), or None."""
        if self._video_source is None:
            return None

        properties = self._video_source.describe_source().source_properties
        if properties is None or not properties.fps:
            return None

        return properties.fps

    def facts(self) -> dict:
        """Describe the decoder and the first frame's tensor layout."""
        properties = None
        if self._video_source is not None:
            metadata = self._video_source.describe_source()
            if metadata.source_properties is not None:
                properties = {
                    "width": metadata.source_properties.width,
                    "height": metadata.source_properties.height,
                    "declared_fps": metadata.source_properties.fps,
                    "is_file": metadata.source_properties.is_file,
                }
        producer_facts = None
        if self._producer is not None:
            producer_facts = {
                "producer_class": type(self._producer).__name__,
                # RTSP pipelines name nvv4l2decoder explicitly (no autoplug);
                # the producer also rejects software decoders on first grab.
                "gstreamer_pipeline": sanitize_source_reference(
                    self._producer.pipeline
                ),
            }

        facts = {
            "index": self.index,
            "url": self.display_url,
            "source_properties": properties,
            "producer": producer_facts,
            "first_frame": self._first_frame_facts,
        }

        return facts

    def request_stop(self) -> None:
        """Stop putting frames into the slot and drop the waiting frame."""
        with self._wakeup:
            self._stopping = True
            self._slot = None

    def stop(self, *, timeout_seconds: float) -> _StopResult:
        """Interrupt the decoder, end both threads and release the stream.

        The reader keeps draining VideoSource until its end-of-stream marker,
        so the capture thread never blocks on a full buffer.

        Args:
            timeout_seconds: Wait bound for the reader thread.

        Returns:
            Native counters read just before release, and stop errors.
        """
        self.request_stop()
        result = _StopResult(native_stats=self.native_stats())
        if self._video_source is None:
            return result

        deadline = time.monotonic() + timeout_seconds
        while True:
            # Right after start() the capture thread may not have left
            # INITIALISING yet, where terminate() is not allowed.
            try:
                self._video_source.terminate(
                    wait_on_frames_consumption=False,
                    purge_frames_buffer=True,
                )
                break
            except StreamOperationNotAllowedError as error:
                if time.monotonic() >= deadline:
                    result.errors.append(f"terminate: {error}")
                    break
                time.sleep(TERMINATE_RETRY_SECONDS)

        if self._reader is not None:
            self._reader.join(timeout=timeout_seconds)
            if self._reader.is_alive():
                result.errors.append("reader thread still alive after stop")

        return result

    def threads_alive(self) -> List[str]:
        """Names of this source's threads that are still running."""
        alive = []
        if self._reader is not None and self._reader.is_alive():
            alive.append(self._reader.name)
        capture = None
        if self._video_source is not None:
            capture = self._video_source._stream_consumption_thread
        if capture is not None and capture.is_alive():
            alive.append(f"source-{self.index}-capture")

        return alive

    def _build_producer(self) -> JetsonVideoFrameProducer:
        self._producer = JetsonVideoFrameProducer(
            self._url,
            output_tensor=True,
            tensor_device=self._device,
        )

        return self._producer

    def _on_status_update(self, update: StatusUpdate) -> None:
        # Runs on the capture thread; DROP_OLDEST evictions arrive here.
        if update.event_type != FRAME_DROPPED_EVENT:
            return

        with self._wakeup:
            self.counters.video_source_dropped += 1

    def _read_loop(self) -> None:
        while True:
            try:
                frame = self._video_source.read_frame(timeout=READ_POLL_SECONDS)
            except EndOfStreamError:
                state = self._video_source.get_state().value
                self._finish(
                    reason="end_of_stream" if state == "ENDED" else "source_error",
                    detail=f"VideoSource state {state}",
                )
                return
            if frame is None:
                continue

            arrival_ns = time.monotonic_ns()
            problem = self._check_frame(frame)
            if problem is not None:
                self._finish(reason="invalid_frame", detail=problem)
                self._discard_until_end()
                return

            with self._wakeup:
                if self._stopping:
                    self.counters.discarded_at_stop += 1
                    continue
                if self._slot is not None:
                    self.counters.replaced_in_slot += 1
                self._slot = ArrivedFrame(
                    source_index=self.index,
                    frame=frame,
                    arrival_ns=arrival_ns,
                )
                self.counters.arrived += 1
                self._wakeup.notify_all()

    def _discard_until_end(self) -> None:
        # Keep draining so the capture thread can always deliver its
        # end-of-stream marker during terminate().
        while True:
            try:
                self._video_source.read_frame(timeout=READ_POLL_SECONDS)
            except EndOfStreamError:
                return

    def _finish(self, *, reason: str, detail: str) -> None:
        with self._wakeup:
            if self.end is None and not self._stopping:
                self.end = SourceEnd(reason=reason, detail=detail)
            self._slot = None
            self._wakeup.notify_all()

    def _check_frame(self, frame: VideoFrame) -> Optional[str]:
        image = frame.image
        if not isinstance(image, torch.Tensor):
            return f"frame is {type(image).__name__}, expected a CUDA tensor"

        expected_device = torch.device(self._device)
        same_device = image.device.type == "cuda" and (
            expected_device.index is None or image.device.index == expected_device.index
        )
        valid = (
            same_device
            and image.dtype == torch.uint8
            and image.ndim == 3
            and image.shape[0] == 3
        )
        if self._first_frame_facts is None:
            self._first_frame_facts = {
                "type": type(image).__name__,
                "dtype": str(image.dtype),
                "shape": list(image.shape),
                "stride": list(image.stride()),
                "contiguous": image.is_contiguous(),
                "device": str(image.device),
                "layout": "CHW",
                "channel_order": "RGB (native bridge NV12->RGB kernel)",
            }
        if not valid:
            return (
                "frame is not a CUDA uint8 CHW tensor on "
                f"{self._device}: {image.dtype} {tuple(image.shape)} {image.device}"
            )

        return None


class JetsonSourceSet:
    """All sources of a run plus round-robin admission.

    Args:
        urls: One stream reference per source.
        device: CUDA device for the native bridges.
    """

    def __init__(self, urls: List[str], *, device: str):
        self.wakeup = threading.Condition()
        self.sources = [
            JetsonSource(
                index,
                url=url,
                wakeup=self.wakeup,
                device=device,
            )
            for index, url in enumerate(urls)
        ]
        self._next_index = 0

    def start(self) -> None:
        """Start all sources concurrently; each waits for its first frame.

        Raises:
            RuntimeError: When any source fails to start. Started sources are
                left for ``stop`` to release.
        """
        errors: Dict[int, BaseException] = {}

        def start_one(source: JetsonSource) -> None:
            try:
                source.start()
            except BaseException as error:  # noqa: BLE001 - reported below
                errors[source.index] = error

        threads = [
            threading.Thread(
                target=start_one,
                args=(source,),
                name=f"source-{source.index}-start",
                daemon=True,
            )
            for source in self.sources
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if errors:
            details = "; ".join(
                f"source {index}: {error!r}" for index, error in sorted(errors.items())
            )
            raise RuntimeError(f"Sources failed to start: {details}")

    def take_next_locked(self) -> Optional[ArrivedFrame]:
        """Take one frame round-robin; the caller holds ``wakeup``.

        The scan starts after the source admitted last, so a source with a
        frame waits for at most one admission per other source.

        Returns:
            The next frame, or None when every slot is empty.
        """
        count = len(self.sources)
        for offset in range(count):
            index = (self._next_index + offset) % count
            arrived = self.sources[index].take_locked()
            if arrived is not None:
                self._next_index = (index + 1) % count
                return arrived

        return None

    def first_end_locked(self) -> Optional[SourceEnd]:
        """First source end seen, labelled with its index; caller holds ``wakeup``."""
        for source in self.sources:
            if source.end is not None:
                end = SourceEnd(
                    reason=source.end.reason,
                    detail=f"source {source.index}: {source.end.detail}",
                )
                return end

        return None

    def snapshot(self) -> List[dict]:
        """Per-source counters and native bridge counters at this instant."""
        with self.wakeup:
            counters = [vars(source.counters).copy() for source in self.sources]
        snapshot = [
            {
                "counters": source_counters,
                "native": source.native_stats(),
            }
            for source, source_counters in zip(self.sources, counters)
        ]

        return snapshot

    def stop(self, *, timeout_seconds: float) -> dict:
        """Stop all sources concurrently within ``timeout_seconds``.

        Args:
            timeout_seconds: Overall bound for interrupting and joining.

        Returns:
            Native counters read just before release, stop errors, and the
            names of any threads still alive afterwards.
        """
        for source in self.sources:
            source.request_stop()

        results: Dict[int, _StopResult] = {}

        def stop_one(source: JetsonSource) -> None:
            try:
                results[source.index] = source.stop(timeout_seconds=timeout_seconds)
            except Exception as error:  # noqa: BLE001 - reported below
                results[source.index] = _StopResult(errors=[repr(error)])

        threads = [
            threading.Thread(
                target=stop_one,
                args=(source,),
                name=f"source-{source.index}-stop",
                daemon=True,
            )
            for source in self.sources
        ]
        deadline = time.monotonic() + timeout_seconds
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

        threads_alive = [
            name for source in self.sources for name in source.threads_alive()
        ]
        stop_report = {
            "final_native": [
                results[index].native_stats if index in results else None
                for index in range(len(self.sources))
            ],
            "errors": {
                index: result.errors
                for index, result in results.items()
                if result.errors
            },
            "unfinished_stops": [
                source.index for source in self.sources if source.index not in results
            ],
            "threads_alive": threads_alive,
        }

        return stop_report


def native_bridge_facts() -> dict:
    """Whether the native Jetson bridge with the expected ABI (7) loads."""
    available, reason = jetson_tensor_bridge_available()
    facts = {
        "available": available,
        "reason": reason,
    }

    return facts

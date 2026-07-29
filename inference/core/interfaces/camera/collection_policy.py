"""Collection policies for multi-source video consumption in InferencePipeline.

Motivation (measured on Jetson AGX Orin, 8x 2K@15 RTSP): consumer cameras emit
frames in bursts separated by encoder pauses of 400-500 ms around every
I-frame. The legacy blocking batch collection waits for a fresh frame from
EVERY source per cycle, so with several staggered cameras almost every batch
stalls on whichever source is inside its pause - throughput collapses to a
fraction of the aggregate frame rate while decoded frames are silently
discarded. The policies in this module remove that coupling:

* the batch-collection window self-tunes from the pipeline's own execution
  rhythm instead of a hand-picked ``batch_collection_timeout``,
* consumption is FIFO with a bounded staleness budget - every frame is served
  while the consumer keeps up, and under overload served frames are never
  older than ``max_staleness`` (drops are counted and reported, not silent).

File sources are exempt from the staleness budget by design: file decoding is
demand-paced (a "stale" frame only means the consumer was busy) and dropping
frames from a file would silently corrupt every-frame processing guarantees.
"""

import logging
from datetime import datetime
from enum import Enum
from time import monotonic
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

from inference.core import env as core_env
from inference.core.interfaces.camera.entities import VideoFrame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from inference.core.interfaces.camera.video_source import VideoSource

logger = logging.getLogger(__name__)

DEFAULT_MAX_STALENESS_SECONDS = 0.5
MIN_COLLECTION_WINDOW_SECONDS = 0.002
MAX_COLLECTION_WINDOW_SECONDS = 0.030
INITIAL_COLLECTION_WINDOW_SECONDS = 0.005
COLLECTION_WINDOW_EXECUTION_FRACTION = 0.2
EXECUTION_GAP_EMA_ALPHA = 0.2
FRESHEST_MODE_BATCH_COLLECTION_TIMEOUT = 0.02
STALENESS_DROP_CAUSE = "STALENESS_BUDGET_EXCEEDED"


class VideoProcessingMode(str, Enum):
    """High-level intent for live multi-source consumption.

    * AUTO - FIFO consumption with a staleness budget and a self-tuning
      collection window. Serves every frame while the consumer keeps up;
      under overload degrades to freshest-at-capacity with bounded,
      reported drops.
    * EVERY_FRAME - AUTO's machinery with the staleness budget disabled:
      strict FIFO for live sources (under sustained overload latency pins
      at the decoding-buffer depth).
    * FRESHEST - legacy latest-wins semantics (EAGER consumption) with a
      small fixed collection timeout; minimal latency, silent skipping.
    """

    AUTO = "auto"
    EVERY_FRAME = "every_frame"
    FRESHEST = "freshest"


def resolve_video_processing_mode(
    explicit_mode: Optional[Union[str, VideoProcessingMode]],
) -> Optional[VideoProcessingMode]:
    """Resolve the effective processing mode.

    Order: explicit argument > tensor-representation cohort default (AUTO
    when ``ENABLE_TENSOR_DATA_REPRESENTATION`` is set - the same opt-in
    boundary that already gates decoding-buffer depth) > ``None`` meaning
    the legacy collection behavior, byte-for-byte.
    """
    if explicit_mode is not None:
        return VideoProcessingMode(explicit_mode)
    if core_env.ENABLE_TENSOR_DATA_REPRESENTATION:
        return VideoProcessingMode.AUTO
    return None


class AdaptiveWindowController:
    """Self-tunes the batch-collection window from the collection rhythm.

    The gap between a non-empty collection finishing and the next collection
    starting is, by construction, the batch execution time. The window is a
    fraction of its EMA, clamped: light models get a near-zero window
    (minimal latency, small batches are cheap), heavy models get a larger
    alignment window that fills batches at negligible relative latency cost.
    """

    def __init__(
        self,
        alpha: float = EXECUTION_GAP_EMA_ALPHA,
        execution_fraction: float = COLLECTION_WINDOW_EXECUTION_FRACTION,
        min_window: float = MIN_COLLECTION_WINDOW_SECONDS,
        max_window: float = MAX_COLLECTION_WINDOW_SECONDS,
        initial_window: float = INITIAL_COLLECTION_WINDOW_SECONDS,
        clock: Callable[[], float] = monotonic,
    ):
        self._alpha = alpha
        self._execution_fraction = execution_fraction
        self._min_window = min_window
        self._max_window = max_window
        self._window = initial_window
        self._clock = clock
        self._execution_gap_ema: Optional[float] = None
        self._last_non_empty_collection_end: Optional[float] = None

    def on_collection_start(self) -> float:
        now = self._clock()
        if self._last_non_empty_collection_end is not None:
            execution_gap = now - self._last_non_empty_collection_end
            if self._execution_gap_ema is None:
                self._execution_gap_ema = execution_gap
            else:
                self._execution_gap_ema = (
                    1 - self._alpha
                ) * self._execution_gap_ema + self._alpha * execution_gap
            self._window = min(
                max(
                    self._execution_fraction * self._execution_gap_ema,
                    self._min_window,
                ),
                self._max_window,
            )
        return self._window

    def on_collection_end(self, collected_any_frame: bool) -> None:
        self._last_non_empty_collection_end = (
            self._clock() if collected_any_frame else None
        )

    @property
    def window(self) -> float:
        return self._window

    @property
    def execution_gap_ema(self) -> Optional[float]:
        return self._execution_gap_ema


class CollectionPolicy:
    """Per-round collection behavior for AUTO / EVERY_FRAME modes.

    Live sources are read FIFO with frames older than ``max_staleness``
    dropped (counted, optionally reported via ``on_frame_dropped``). File
    sources - and sources whose properties are not yet known - are read
    as-is and NEVER dropped; liveness is resolved lazily from
    ``VideoSource.describe_source()`` and cached once known.
    """

    def __init__(
        self,
        mode: VideoProcessingMode,
        max_staleness: Optional[float] = None,
        on_frame_dropped: Optional[Callable[[VideoFrame], None]] = None,
        window_controller: Optional[AdaptiveWindowController] = None,
    ):
        if mode is VideoProcessingMode.FRESHEST:
            raise ValueError(
                "FRESHEST mode is realised through EAGER buffer consumption "
                "and does not use CollectionPolicy"
            )
        self._mode = mode
        if mode is VideoProcessingMode.EVERY_FRAME:
            self._max_staleness = None
        else:
            self._max_staleness = (
                DEFAULT_MAX_STALENESS_SECONDS
                if max_staleness is None
                else max_staleness
            )
        self._on_frame_dropped = on_frame_dropped
        self._window_controller = window_controller or AdaptiveWindowController()
        self._source_is_file: Dict[int, bool] = {}
        self._frames_dropped_on_staleness: Dict[int, int] = {}

    @property
    def mode(self) -> VideoProcessingMode:
        return self._mode

    @property
    def max_staleness(self) -> Optional[float]:
        return self._max_staleness

    @property
    def frames_dropped_on_staleness(self) -> Dict[int, int]:
        return dict(self._frames_dropped_on_staleness)

    def collection_window(self) -> float:
        return self._window_controller.on_collection_start()

    def note_collection_result(self, batch_frames: list) -> None:
        self._window_controller.on_collection_end(
            collected_any_frame=bool(batch_frames)
        )

    def read_frame(
        self,
        source_ord: int,
        source: "VideoSource",
        timeout: Optional[float],
    ) -> Optional[VideoFrame]:
        if self._max_staleness is None or self._source_treated_as_file(
            source_ord=source_ord, source=source
        ):
            return source.read_frame(timeout=timeout)
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(deadline - monotonic(), 0.0)
            frame = source.read_frame(timeout=remaining)
            if frame is None:
                return None
            frame_age = (datetime.now() - frame.frame_timestamp).total_seconds()
            if frame_age <= self._max_staleness:
                return frame
            self._register_staleness_drop(source_ord=source_ord, frame=frame)

    def _source_treated_as_file(self, source_ord: int, source: "VideoSource") -> bool:
        cached = self._source_is_file.get(source_ord)
        if cached is not None:
            return cached
        is_file: Optional[bool] = None
        try:
            source_metadata = source.describe_source()
            source_properties = source_metadata.source_properties
            if source_properties is not None:
                is_file = source_properties.is_file
        except Exception:  # noqa: BLE001 - metadata probe must never break reads
            is_file = None
        if is_file is None:
            # Liveness unknown (source still initialising) - never drop.
            return True
        self._source_is_file[source_ord] = is_file
        return is_file

    def _register_staleness_drop(self, source_ord: int, frame: VideoFrame) -> None:
        self._frames_dropped_on_staleness[source_ord] = (
            self._frames_dropped_on_staleness.get(source_ord, 0) + 1
        )
        if self._on_frame_dropped is None:
            return
        try:
            self._on_frame_dropped(frame)
        except Exception:  # noqa: BLE001 - reporting must never break collection
            logger.warning(
                "on_frame_dropped callback raised while reporting a staleness "
                "drop for source %s",
                frame.source_id,
            )

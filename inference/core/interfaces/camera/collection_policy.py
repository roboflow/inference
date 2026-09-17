"""Collection policies for multi-source video consumption in InferencePipeline.

Motivation (measured on Jetson AGX Orin, 8x 2K@15 RTSP): consumer cameras emit
frames in bursts separated by encoder pauses of 400-500 ms around every
I-frame. The legacy blocking batch collection waits for a fresh frame from
EVERY source per cycle, so with several staggered cameras almost every batch
stalls on whichever source is inside its pause - throughput collapses to a
fraction of the aggregate frame rate while decoded frames are silently
discarded. The policies in this module remove that coupling:

* the batch-collection window self-tunes from the pipeline's own rhythms
  instead of a hand-picked ``batch_collection_timeout``: once per-source
  arrival rates are measured, the round period is matched to the fastest
  live source's frame period (full batches whenever the model has headroom,
  floor-window under saturation); before estimates exist it falls back to a
  fraction of the measured execution time,
* consumption is FIFO with a bounded staleness budget - every frame is served
  while the consumer keeps up, and under overload served frames are never
  older than ``max_staleness`` (drops are counted and reported, not silent).

File sources are exempt from the staleness budget by design: file decoding is
demand-paced (a "stale" frame only means the consumer was busy) and dropping
frames from a file would silently corrupt every-frame processing guarantees.
"""

import logging
from collections import deque
from datetime import datetime
from enum import Enum
from time import monotonic
from typing import TYPE_CHECKING, Callable, Deque, Dict, Optional, Union

from inference.core import env as core_env
from inference.core.interfaces.camera.entities import VideoFrame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from inference.core.interfaces.camera.video_source import VideoSource

logger = logging.getLogger(__name__)

DEFAULT_MAX_STALENESS_SECONDS = 0.5
MIN_COLLECTION_WINDOW_SECONDS = 0.002
MAX_COLLECTION_WINDOW_SECONDS = 0.030
COLLECTION_WINDOW_EXECUTION_FRACTION = 0.2
EXECUTION_GAP_EMA_ALPHA = 0.2
FRESHEST_MODE_BATCH_COLLECTION_TIMEOUT = 0.02
STALENESS_DROP_CAUSE = "STALENESS_BUDGET_EXCEEDED"
LEGACY_MODE_ALIASES = frozenset({"legacy", "none"})
# Rate-matched window: cap and the arrival-period estimator's shape. The
# estimator is count-over-span (never an EMA of gaps - bursty encoders like
# consumer RTSP cameras emit 2-frame clusters around GOP pauses, and gap
# EMAs oscillate at the burst frequency while a span over several burst
# cycles converges on the true rate).
RATE_MATCHED_WINDOW_CAP_SECONDS = 0.1
# Give the first model invocation a bounded chance to receive the full live
# cohort. Shape-specialised runtimes such as TensorRT can otherwise spend their
# cold start preparing a tiny partial-batch plan before arrival estimates exist.
# After the first non-empty round, the execution/rate controller takes over.
INITIAL_COLLECTION_WINDOW_SECONDS = RATE_MATCHED_WINDOW_CAP_SECONDS
ARRIVAL_PERIOD_SAMPLE_WINDOW = 64
MIN_ARRIVAL_SAMPLES_TO_TRUST = 16
SOURCE_ACTIVITY_HORIZON_SECONDS = 2.0


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
    the legacy collection behavior, byte-for-byte. The explicit strings
    ``"legacy"`` / ``"none"`` force the legacy behavior even inside the
    tensor cohort - the escape hatch from the flag-driven AUTO default.
    """
    if explicit_mode is not None:
        if (
            isinstance(explicit_mode, str)
            and explicit_mode.lower() in LEGACY_MODE_ALIASES
        ):
            return None
        return VideoProcessingMode(explicit_mode)
    if core_env.ENABLE_TENSOR_DATA_REPRESENTATION:
        return VideoProcessingMode.AUTO
    return None


class _SourceArrivalEstimator:
    """Count-over-span estimate of one live source's true frame period.

    Fed with ingress capture timestamps (``VideoFrame.frame_timestamp`` is
    stamped on the decode thread), so downstream queueing cannot distort the
    span. A gap longer than the activity horizon (reconnect, stall) clears
    the window - a rejoin gap is not a frame period - and the estimate stays
    ``None`` until enough fresh samples accumulate again.
    """

    def __init__(self, sample_window: int = ARRIVAL_PERIOD_SAMPLE_WINDOW):
        self._timestamps: Deque[datetime] = deque(maxlen=sample_window)
        self._last_seen_at: Optional[float] = None

    def observe(self, frame_timestamp: datetime, now: float) -> None:
        if self._timestamps:
            gap = (frame_timestamp - self._timestamps[-1]).total_seconds()
            if gap > SOURCE_ACTIVITY_HORIZON_SECONDS or gap < 0:
                self._timestamps.clear()
        self._timestamps.append(frame_timestamp)
        self._last_seen_at = now

    def period(self, now: float) -> Optional[float]:
        if (
            self._last_seen_at is None
            or now - self._last_seen_at > SOURCE_ACTIVITY_HORIZON_SECONDS
        ):
            return None
        if len(self._timestamps) < MIN_ARRIVAL_SAMPLES_TO_TRUST:
            return None
        span = (self._timestamps[-1] - self._timestamps[0]).total_seconds()
        if span <= 0:
            return None
        return span / (len(self._timestamps) - 1)


class AdaptiveWindowController:
    """Self-tunes the batch-collection window from the collection rhythm.

    Two regimes, picked per round by whether a trustworthy arrival-period
    estimate exists:

    * RATE-MATCHED (estimate available): ``window = clamp(min live source
      period - exec EMA, floor, cap)``. The round period then equals the
      fastest source's frame period, so in the light-load regime every
      round finds each source with exactly one fresh frame - full batches
      by construction. Under saturation (exec >= period) the subtraction
      goes negative and the window floors, which is the correct move: every
      source already has frames queued when the round starts. The added
      collection wait is offset by removed queue wait (a frame that misses
      a round today sits in the LAZY queue for a full round period), so
      end-to-end latency stays roughly flat while occupancy rises.
    * FALLBACK (startup, no live estimates): a fraction of the exec-time
      EMA, clamped - light models get a near-zero window, heavy models a
      larger alignment window.

    The gap between a non-empty collection finishing and the next collection
    starting is, by construction, the batch execution time.
    """

    def __init__(
        self,
        alpha: float = EXECUTION_GAP_EMA_ALPHA,
        execution_fraction: float = COLLECTION_WINDOW_EXECUTION_FRACTION,
        min_window: float = MIN_COLLECTION_WINDOW_SECONDS,
        max_window: float = MAX_COLLECTION_WINDOW_SECONDS,
        initial_window: float = INITIAL_COLLECTION_WINDOW_SECONDS,
        rate_matched_cap: float = RATE_MATCHED_WINDOW_CAP_SECONDS,
        clock: Callable[[], float] = monotonic,
    ):
        self._alpha = alpha
        self._execution_fraction = execution_fraction
        self._min_window = min_window
        self._max_window = max_window
        self._rate_matched_cap = rate_matched_cap
        self._window = initial_window
        self._clock = clock
        self._execution_gap_ema: Optional[float] = None
        self._last_non_empty_collection_end: Optional[float] = None

    def on_collection_start(
        self, minimum_arrival_period: Optional[float] = None
    ) -> float:
        now = self._clock()
        if self._last_non_empty_collection_end is not None:
            execution_gap = now - self._last_non_empty_collection_end
            if self._execution_gap_ema is None:
                self._execution_gap_ema = execution_gap
            else:
                self._execution_gap_ema = (
                    1 - self._alpha
                ) * self._execution_gap_ema + self._alpha * execution_gap
            if minimum_arrival_period is not None:
                rate_matched = minimum_arrival_period - self._execution_gap_ema
                self._window = min(
                    max(rate_matched, self._min_window),
                    self._rate_matched_cap,
                )
            else:
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
        self._arrival_estimators: Dict[int, _SourceArrivalEstimator] = {}

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
        return self._window_controller.on_collection_start(
            minimum_arrival_period=self.minimum_live_arrival_period()
        )

    def minimum_live_arrival_period(self) -> Optional[float]:
        """Smallest trustworthy frame period across ACTIVE live sources.

        The fastest source is the binding constraint for the rate-matched
        window: a round period above ANY source's frame period makes that
        source queue unboundedly. Files never contribute (demand-paced,
        always ready) and dormant sources drop out via the activity horizon
        so a dying camera cannot pin the window while it flaps.
        """
        now = monotonic()
        periods = [
            period
            for period in (
                estimator.period(now) for estimator in self._arrival_estimators.values()
            )
            if period is not None
        ]
        if not periods:
            return None
        return min(periods)

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
        treated_as_file = self._source_treated_as_file(
            source_ord=source_ord, source=source
        )
        if self._max_staleness is None or treated_as_file:
            frame = source.read_frame(timeout=timeout)
            if frame is not None and not treated_as_file:
                self._observe_arrival(source_ord=source_ord, frame=frame)
            return frame
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(deadline - monotonic(), 0.0)
            frame = source.read_frame(timeout=remaining)
            if frame is None:
                return None
            # Staleness-drained frames feed the estimator too: they are real
            # arrivals, and skipping them would bias the period estimate
            # upward exactly when the pipeline is busiest.
            self._observe_arrival(source_ord=source_ord, frame=frame)
            frame_age = (datetime.now() - frame.frame_timestamp).total_seconds()
            if frame_age <= self._max_staleness:
                return frame
            self._register_staleness_drop(source_ord=source_ord, frame=frame)

    def _observe_arrival(self, source_ord: int, frame: VideoFrame) -> None:
        estimator = self._arrival_estimators.get(source_ord)
        if estimator is None:
            estimator = _SourceArrivalEstimator()
            self._arrival_estimators[source_ord] = estimator
        estimator.observe(frame_timestamp=frame.frame_timestamp, now=monotonic())

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

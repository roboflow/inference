"""Video Clip Recorder: per-stream H.264 ring buffers that sinks turn into clips.

The block hands every processed frame to a per-`video_identifier` encoded ring
(`X264Ring`, or `NvencRing` for CUDA tensor frames when PyNvVideoCodec is available)
inside `run()`, and emits a `ClipAnchor` per frame. A sink that decides to attach a
clip calls `anchor.capture(pre, post)`; the ring resolves the returned future on its
own threads once the post-roll is buffered, so no shared executor is involved.
"""

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
from pydantic import ConfigDict, Field, field_validator
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.video_clips import env_positive_float
from roboflow_workflows.core_steps.common.video_clips.capture import (
    PARTIAL_RESOLUTION_CHANGE,
    PARTIAL_SHUTDOWN,
    PARTIAL_STALLED,
    ClipAnchor,
    bind_capture,
)
from roboflow_workflows.core_steps.common.video_clips.x264_ring import X264Ring
from roboflow_workflows.core_steps.common.workload_presets import (
    STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    VIDEO_CLIP_ANCHOR_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
)

logger = get_logger(__name__)

ANCHOR_KEY = "anchor"
BLOCK_TYPE = "roboflow_core/video_clip_recorder@v1"
DEFAULT_RECORDER_STEP_NAME = "video_clip_recorder"
DEFAULT_VIDEO_IDENTIFIER = "default"

AVAILABILITY_OK = "ok"
UNAVAILABLE_NO_TIMESTAMP = "unavailable:no_timestamp"
UNAVAILABLE_NO_ENCODER = "unavailable:no_encoder"
UNAVAILABLE_SINKS_DISABLED = "unavailable:sinks_disabled"
UNAVAILABLE_UNSUPPORTED_HOST = "unavailable:unsupported_host"

BACKEND_X264 = "x264"
BACKEND_NVENC = "nvenc"

# Upper bound on concurrently tracked `video_identifier` values, as in Frame Delay.
# The least recently seen stream's ring is retired beyond that.
MAX_TRACKED_VIDEOS = 16

MAX_PRE_ROLL_SECONDS_ENV = "WORKFLOWS_VIDEO_CLIP_MAX_PRE_ROLL_SECONDS"
MAX_POST_ROLL_SECONDS_ENV = "WORKFLOWS_VIDEO_CLIP_MAX_POST_ROLL_SECONDS"
DEFAULT_MAX_ROLL_SECONDS = 60.0
# Hosts may raise the roll caps through the environment, but never above this: ring
# memory grows with the pre-roll and the post-roll must fit the signed-URL lifetime.
HARD_MAX_ROLL_SECONDS = 300.0

MIN_PLAYBACK_SPEED = 0.1
MAX_PLAYBACK_SPEED = 32.0
MIN_CLIP_FPS = 1.0
MAX_CLIP_FPS = 120.0
MIN_MAX_WIDTH = 64
MAX_LABEL_LENGTH = 256

# Samples of `ring.submit()` cost kept per identifier for p50 / p95 telemetry.
RING_COST_SAMPLES = 2048

SHORT_DESCRIPTION = (
    "Keep a rolling encoded buffer of processed frames so the Vision Events sink "
    "can attach a video clip of the moments around an event."
)

LONG_DESCRIPTION = """
Record the frames your workflow processes into a short, rolling H.264 buffer per
video stream, and let the **Vision Events** sink attach a clip of the moments around
an event to the event it posts.

Wire the frames you want recorded (the raw input image, or a visualization output for
an annotated clip) into `image`, then bind this block's `anchor` output to the Vision
Events sink's `video_clips` input. Every frame produces an anchor; the sink only turns
it into a clip when it actually posts an event, using this recorder's
`pre_roll_seconds` and `post_roll_seconds`.

## How This Block Works

1. Reads `video_metadata` from the image: the `video_identifier` keeps per-stream
   state isolated, `frame_number` detects restarts and duplicates, and
   `frame_timestamp` is the time axis of every clip.
2. Encodes the frame into the stream's ring (libx264 on the CPU, or NVENC straight
   from GPU frames when available). The frame is converted or copied inside the step,
   so later in-place drawing on the same image never reaches the recording.
3. Emits an `anchor` for the frame. When the sink captures it, the clip covers
   `[frame - pre_roll_seconds, frame + post_roll_seconds]` (end inclusive, so a
   post-roll of 0 still contains the triggering frame), starting at the nearest
   keyframe up to one output second earlier. The clip is uploaded as soon as the
   post-roll has been recorded.

## Common Use Cases

- Attach evidence footage to safety, quality or security events.
- Record an annotated clip (wire a visualization output) next to a raw clip (wire the
  input image) with two recorders and distinct `label`s.

## Requirements and Limitations

- Needs a video stream (an InferencePipeline, WebRTC or Serverless Streaming). On a
  single-image request the anchor is `unavailable:unsupported_host`.
- Place the recorder on the unconditional path: under flow control it only records
  frames on which it runs, so clips lose pre-roll (compile-time warning). Gate the
  sink instead.
- The anchor never leaves the process: binding it into a remote step or exposing it
  as a workflow output is a compile-time error.
- Clips contain the frames the workflow processed: their density follows the
  processing rate, not the camera rate. Frames processed more than
  `max_processing_lag_seconds` after they were captured are not recorded.
- A frame-number restart or a resolution change starts a new recording; a clip in
  progress ends there and is reported as partial.
- `pre_roll_seconds` and `post_roll_seconds` are capped at 60 s by default (the host
  may raise the caps up to 300 s). At 1080p the ring holds about 0.9 MB per second of
  window per stream.
- At most 16 streams are tracked at once; the least recently seen stream's recording
  is discarded beyond that.
- With `disable_sinks` the recorder idles and anchors are
  `unavailable:sinks_disabled`. No audio is recorded.
"""


def max_pre_roll_seconds() -> float:
    """Return the largest accepted `pre_roll_seconds`.

    Returns:
        The value of `WORKFLOWS_VIDEO_CLIP_MAX_PRE_ROLL_SECONDS` (default 60),
        clamped to 300; invalid or non-positive values fall back to the default.
    """
    limit = env_positive_float(
        MAX_PRE_ROLL_SECONDS_ENV,
        default=DEFAULT_MAX_ROLL_SECONDS,
        maximum=HARD_MAX_ROLL_SECONDS,
    )

    return limit


def max_post_roll_seconds() -> float:
    """Return the largest accepted `post_roll_seconds`.

    Returns:
        The value of `WORKFLOWS_VIDEO_CLIP_MAX_POST_ROLL_SECONDS` (default 60),
        clamped to 300; invalid or non-positive values fall back to the default.
    """
    limit = env_positive_float(
        MAX_POST_ROLL_SECONDS_ENV,
        default=DEFAULT_MAX_ROLL_SECONDS,
        maximum=HARD_MAX_ROLL_SECONDS,
    )

    return limit


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Video Clip Recorder",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-video",
            },
        }
    )
    type: Literal["roboflow_core/video_clip_recorder@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Image",
        description="The video frames to record: the input image for a raw clip, or "
        "a visualization output for an annotated clip.",
        examples=["$inputs.image", "$steps.bounding_box_visualization.image"],
    )
    label: str = Field(
        default="Evidence",
        description="Caption the Vision Events sink posts with this recorder's clip. "
        "1 to 256 characters; short labels such as `Raw` or `Annotated` read best "
        "because the Vision Events UI truncates long captions.",
        examples=["Evidence", "Raw", "Annotated"],
    )
    pre_roll_seconds: float = Field(
        default=5.0,
        description="Seconds of video before the triggering frame to include in a "
        "clip. At most 60 by default (the host may raise the cap up to 300). Longer "
        "pre-rolls hold more memory: about 0.9 MB per second at 1080p per stream.",
        examples=[5.0, 10.0],
    )
    post_roll_seconds: float = Field(
        default=5.0,
        description="Seconds of video after the triggering frame to include in a "
        "clip (the triggering frame is always included). The clip is uploaded once "
        "this much video has been recorded. At most 60 by default (the host may "
        "raise the cap up to 300).",
        examples=[5.0, 0.0],
    )
    playback_speed: float = Field(
        default=1.0,
        description="Playback speed of the clip, between 0.1 and 32. Values above 1 "
        "produce a time-lapse; values below 1 a slow-motion clip.",
        examples=[1.0, 4.0],
    )
    clip_fps: Optional[float] = Field(
        default=None,
        description="Maximum frame rate of the clip, between 1 and 120. Frames are "
        "kept on a regular grid at this rate; `null` keeps every processed frame "
        "with its source timing.",
        examples=[None, 10.0],
    )
    max_width: Optional[int] = Field(
        default=None,
        description="Maximum clip width in pixels (even, at least 64). Wider frames "
        "are downscaled keeping the aspect ratio; `null` keeps the native resolution.",
        examples=[None, 1280],
    )
    max_processing_lag_seconds: float = Field(
        default=5.0,
        description="Frames processed more than this many seconds after their "
        "capture timestamp (for example the first frames after a slow model load) "
        "are not recorded. Not applied to frames read from video files, whose "
        "timestamps follow the file rather than the wall clock.",
        examples=[5.0],
    )

    @field_validator("label")
    @classmethod
    def _label_must_be_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("`label` must not be empty or only whitespace.")

        if len(stripped) > MAX_LABEL_LENGTH:
            raise ValueError(
                f"`label` must be at most {MAX_LABEL_LENGTH} characters "
                f"(got {len(stripped)})."
            )

        return stripped

    @field_validator("pre_roll_seconds")
    @classmethod
    def _pre_roll_in_range(cls, value: float) -> float:
        _validate_range(
            value,
            name="pre_roll_seconds",
            minimum=0.0,
            maximum=max_pre_roll_seconds(),
        )
        return value

    @field_validator("post_roll_seconds")
    @classmethod
    def _post_roll_in_range(cls, value: float) -> float:
        _validate_range(
            value,
            name="post_roll_seconds",
            minimum=0.0,
            maximum=max_post_roll_seconds(),
        )
        return value

    @field_validator("playback_speed")
    @classmethod
    def _playback_speed_in_range(cls, value: float) -> float:
        _validate_range(
            value,
            name="playback_speed",
            minimum=MIN_PLAYBACK_SPEED,
            maximum=MAX_PLAYBACK_SPEED,
        )
        return value

    @field_validator("clip_fps")
    @classmethod
    def _clip_fps_in_range(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value

        _validate_range(
            value,
            name="clip_fps",
            minimum=MIN_CLIP_FPS,
            maximum=MAX_CLIP_FPS,
        )
        return value

    @field_validator("max_width")
    @classmethod
    def _max_width_is_even(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value

        if value < MIN_MAX_WIDTH:
            raise ValueError(
                f"`max_width` must be at least {MIN_MAX_WIDTH} (got {value})."
            )

        if value % 2 != 0:
            raise ValueError(
                f"`max_width` must be even, as H.264 4:2:0 video requires (got "
                f"{value})."
            )

        return value

    @field_validator("max_processing_lag_seconds")
    @classmethod
    def _max_processing_lag_positive(cls, value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"`max_processing_lag_seconds` must be greater than 0 (got {value})."
            )

        return value

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["image"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name=ANCHOR_KEY, kind=[VIDEO_CLIP_ANCHOR_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.TEMPORAL_BUFFERING]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return actual_restrictions_of(
            declared=list(STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS),
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        return []


def _validate_range(value: float, *, name: str, minimum: float, maximum: float) -> None:
    if not np.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(
            f"`{name}` must be between {minimum:g} and {maximum:g} (got {value})."
        )


def serialize_video_clip_anchor(value: Any) -> Any:
    """Serialise a `ClipAnchor` to metadata only.

    Frames, encoded packets and the reference to the recorder are never
    serialised; an anchor cannot be deserialised.

    Args:
        value: A `ClipAnchor`, or None.

    Returns:
        The anchor metadata (`kind`, `recorder`, `video_identifier`,
        `frame_timestamp` in ISO 8601, `frame_number`, `availability`), or None.
    """
    if value is None:
        return None

    frame_timestamp = getattr(value, "frame_timestamp", None)
    serialised_timestamp = (
        frame_timestamp.isoformat()
        if isinstance(frame_timestamp, datetime)
        else frame_timestamp
    )
    serialised = {
        "kind": VIDEO_CLIP_ANCHOR_KIND.name,
        "recorder": getattr(value, "recorder_step", None),
        "video_identifier": getattr(value, "video_identifier", None),
        "frame_timestamp": serialised_timestamp,
        "frame_number": getattr(value, "frame_number", None),
        "availability": getattr(value, "availability", None),
    }

    return serialised


@dataclass(frozen=True)
class _RecorderSettings:
    label: str
    pre_roll_seconds: float
    post_roll_seconds: float
    playback_speed: float
    clip_fps: Optional[float]
    max_width: Optional[int]
    max_processing_lag_seconds: float


@dataclass
class _StreamState:
    """Per-identifier recording state; the ring is fixed at the first frame."""

    ring: Any = None
    backend: Optional[str] = None
    encoder_unavailable: bool = False
    last_frame_number: Optional[int] = None
    last_frame_timestamp: Optional[datetime] = None
    resolution: Optional[Tuple[int, int]] = None
    ring_costs_ms: Deque[float] = field(
        default_factory=lambda: deque(maxlen=RING_COST_SAMPLES)
    )
    frames_submitted: int = 0
    stale_dropped: int = 0
    duplicates_skipped: int = 0
    resets: int = 0


class VideoClipRecorderBlockV1(WorkflowBlock):
    """Per-stream encoded rings that turn processed frames into clip anchors.

    State is keyed by this block instance and the frame's `video_identifier`, so two
    recorders, or two streams through one recorder, never share footage.
    `prepare_close` and `close` are idempotent and safe under concurrent calls.
    """

    def __init__(self, disable_sinks: bool = False):
        self._disable_sinks = disable_sinks
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        # Ordered least- to most-recently-seen stream, for LRU retirement.
        self._streams: "OrderedDict[str, _StreamState]" = OrderedDict()
        # Rings retired by a reset or eviction: their pending captures finalise on
        # the rings' own threads. A ring releases itself and leaves this list once
        # its last capture is finalised; the rest are closed with the block.
        self._retired_rings: List[Any] = []
        self._closing = False
        self._closed = False

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["disable_sinks"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: Batch[WorkflowImageData],
        label: str = "Evidence",
        pre_roll_seconds: float = 5.0,
        post_roll_seconds: float = 5.0,
        playback_speed: float = 1.0,
        clip_fps: Optional[float] = None,
        max_width: Optional[int] = None,
        max_processing_lag_seconds: float = 5.0,
    ) -> BlockResult:
        settings = _RecorderSettings(
            label=label,
            pre_roll_seconds=float(pre_roll_seconds),
            post_roll_seconds=float(post_roll_seconds),
            playback_speed=float(playback_speed),
            clip_fps=None if clip_fps is None else float(clip_fps),
            max_width=None if max_width is None else int(max_width),
            max_processing_lag_seconds=float(max_processing_lag_seconds),
        )
        results = [
            {ANCHOR_KEY: self._record_frame(frame, settings=settings)}
            for frame in image
        ]

        return results

    def prepare_close(self, deadline_seconds: Optional[float] = None) -> None:
        """Stop recording and start finalising every pending capture.

        Stops accepting frames and new captures; each ring drops its unencoded
        ingest queue, finalises pending captures from what it already holds as
        partial (`shutdown`) and resolves their futures on its own threads. Safe to
        call more than once and from several threads.

        Args:
            deadline_seconds: Seconds the host allows for the whole close (relative);
                None lets each ring use its default.
        """
        with self._lock:
            if self._closing:
                return

            self._closing = True
            rings = self._all_rings()
        for ring in rings:
            try:
                ring.prepare_close(deadline_seconds)
            except Exception as error:
                logger.warning(
                    f"Video Clip Recorder `{self._step_name()}` could not prepare a "
                    f"ring for closing: {error}"
                )

    def close(self) -> None:
        """Finalise pending captures and release every ring and encoder.

        Calls `prepare_close` first when the host did not, then closes every ring
        (joining its threads within the deadline given to `prepare_close`). A
        concurrent second call waits for the first to finish; later calls return
        immediately.
        """
        self.prepare_close()
        with self._close_lock:
            if self._closed:
                return

            with self._lock:
                rings = self._all_rings()
                self._streams.clear()
                self._retired_rings = []
            for ring in rings:
                try:
                    ring.close()
                except Exception as error:
                    logger.warning(
                        f"Video Clip Recorder `{self._step_name()}` could not close a "
                        f"ring: {error}"
                    )
            self._closed = True

    def telemetry(self) -> Dict[str, Dict[str, Any]]:
        """Return per-identifier recording telemetry.

        Returns:
            For each tracked `video_identifier`: `backend`, `frames_submitted`,
            `stale_dropped`, `duplicates_skipped`, `resets`, and `ring_cost_ms_p50` /
            `ring_cost_ms_p95` (the time `submit()` took inside `run()`, None
            before the first frame).
        """
        with self._lock:
            snapshot = {
                video_identifier: _stream_telemetry(state=state)
                for video_identifier, state in self._streams.items()
            }

        return snapshot

    def _record_frame(
        self, image: WorkflowImageData, settings: _RecorderSettings
    ) -> ClipAnchor:
        metadata = image.video_metadata
        video_identifier = str(metadata.video_identifier or DEFAULT_VIDEO_IDENTIFIER)
        frame_number = int(metadata.frame_number)
        frame_timestamp = getattr(metadata, "frame_timestamp", None)
        if self._disable_sinks:
            return self._unavailable_anchor(
                video_identifier=video_identifier,
                frame_timestamp=frame_timestamp,
                frame_number=frame_number,
                settings=settings,
                availability=UNAVAILABLE_SINKS_DISABLED,
            )

        if not _carries_video_metadata(image=image):
            return self._unavailable_anchor(
                video_identifier=video_identifier,
                frame_timestamp=frame_timestamp,
                frame_number=frame_number,
                settings=settings,
                availability=UNAVAILABLE_UNSUPPORTED_HOST,
            )

        if not isinstance(frame_timestamp, datetime):
            return self._unavailable_anchor(
                video_identifier=video_identifier,
                frame_timestamp=frame_timestamp,
                frame_number=frame_number,
                settings=settings,
                availability=UNAVAILABLE_NO_TIMESTAMP,
            )

        with self._lock:
            if self._closing:
                return self._unavailable_anchor(
                    video_identifier=video_identifier,
                    frame_timestamp=frame_timestamp,
                    frame_number=frame_number,
                    settings=settings,
                    availability=UNAVAILABLE_NO_ENCODER,
                )

            state = self._track_stream(video_identifier=video_identifier)
            resolution = _frame_resolution(image=image)
            self._reset_if_needed(
                state=state,
                frame_number=frame_number,
                resolution=resolution,
            )
            is_duplicate = (
                state.last_frame_number is not None
                and frame_number == state.last_frame_number
            )
            if is_duplicate:
                # WebRTC runs the workflow twice on frame 1 (G0.2): the second run
                # must not enter the ring again.
                state.duplicates_skipped += 1
                ring = state.ring
            else:
                ring = self._ensure_ring(
                    state=state,
                    video_identifier=video_identifier,
                    image=image,
                    settings=settings,
                )
                self._feed_ring(
                    ring=ring,
                    state=state,
                    image=image,
                    metadata=metadata,
                    frame_timestamp=frame_timestamp,
                    frame_number=frame_number,
                    settings=settings,
                )
                ring = state.ring
                state.last_frame_number = frame_number
                state.last_frame_timestamp = frame_timestamp
                state.resolution = resolution

        if ring is None:
            return self._unavailable_anchor(
                video_identifier=video_identifier,
                frame_timestamp=frame_timestamp,
                frame_number=frame_number,
                settings=settings,
                availability=UNAVAILABLE_NO_ENCODER,
            )

        anchor = ClipAnchor(
            recorder_step=self._step_name(),
            video_identifier=video_identifier,
            frame_timestamp=frame_timestamp,
            frame_number=frame_number,
            availability=AVAILABILITY_OK,
            label=settings.label,
            pre_roll_seconds=settings.pre_roll_seconds,
            post_roll_seconds=settings.post_roll_seconds,
            _capture=bind_capture(ring, frame_timestamp),
        )

        return anchor

    def _feed_ring(
        self,
        ring: Any,
        state: _StreamState,
        image: WorkflowImageData,
        metadata: Any,
        frame_timestamp: datetime,
        frame_number: int,
        settings: _RecorderSettings,
    ) -> None:
        if ring is None:
            return

        if _is_stale(
            frame_timestamp=frame_timestamp,
            max_processing_lag_seconds=settings.max_processing_lag_seconds,
            comes_from_video_file=getattr(metadata, "comes_from_video_file", None),
        ):
            # Frames processed long after their grab time (e.g. right after a slow
            # model load) are never buffered, so they cannot start or enter a clip.
            state.stale_dropped += 1
            ring.record_stale(frame_timestamp)
            return

        started = time.perf_counter()
        try:
            ring.submit(frame_timestamp, frame_number, image)
        except Exception as error:
            # A broken encoder must never fail the workflow: the stream's anchors
            # become `unavailable:no_encoder` until the stream resets.
            logger.warning(
                f"Video Clip Recorder `{self._step_name()}` lost its encoder: {error}"
            )
            self._retire_ring(state=state, reason=PARTIAL_SHUTDOWN)
            state.encoder_unavailable = True
            return

        state.ring_costs_ms.append((time.perf_counter() - started) * 1000.0)
        state.frames_submitted += 1

    def _track_stream(self, video_identifier: str) -> _StreamState:
        state = self._streams.get(video_identifier)
        if state is None:
            state = _StreamState()
        self._streams[video_identifier] = state
        self._streams.move_to_end(video_identifier)
        while len(self._streams) > MAX_TRACKED_VIDEOS:
            _, evicted = self._streams.popitem(last=False)
            self._retire_ring(state=evicted, reason=PARTIAL_SHUTDOWN)

        return state

    def _reset_if_needed(
        self,
        state: _StreamState,
        frame_number: int,
        resolution: Optional[Tuple[int, int]],
    ) -> None:
        if state.last_frame_number is None:
            return

        if frame_number < state.last_frame_number:
            # A restarted stream reusing the identifier (e.g. a reconnect): its
            # footage must never be spliced onto the previous recording.
            reason = PARTIAL_STALLED
        elif resolution is not None and state.resolution not in (None, resolution):
            reason = PARTIAL_RESOLUTION_CHANGE
        else:
            return

        self._retire_ring(state=state, reason=reason)
        state.backend = None
        state.encoder_unavailable = False
        state.last_frame_number = None
        state.last_frame_timestamp = None
        state.resolution = None
        state.resets += 1

    def _retire_ring(self, state: _StreamState, reason: str) -> None:
        ring = state.ring
        state.ring = None
        if ring is None:
            return

        self._retired_rings.append(ring)
        try:
            ring.retire(reason, on_released=partial(self._forget_retired_ring, ring))
        except Exception as error:
            logger.warning(
                f"Video Clip Recorder `{self._step_name()}` could not retire a ring: "
                f"{error}"
            )

    def _forget_retired_ring(self, ring: Any) -> None:
        # Called by a retired ring (on its own thread, or inside `retire`) once it
        # released its packets and budget share: nothing is left to close.
        with self._lock:
            self._retired_rings = [
                retired for retired in self._retired_rings if retired is not ring
            ]

    def _ensure_ring(
        self,
        state: _StreamState,
        video_identifier: str,
        image: WorkflowImageData,
        settings: _RecorderSettings,
    ) -> Any:
        if state.ring is not None or state.encoder_unavailable:
            return state.ring

        declared_fps = getattr(image.video_metadata, "fps", None)
        backend, ring_class = _select_backend(image=image)
        try:
            state.ring = ring_class(
                video_identifier,
                pre_roll_seconds=settings.pre_roll_seconds,
                playback_speed=settings.playback_speed,
                clip_fps=settings.clip_fps,
                max_width=settings.max_width,
                declared_fps=declared_fps or None,
            )
            state.backend = backend
        except Exception as error:
            logger.warning(
                f"Video Clip Recorder `{self._step_name()}` has no encoder for stream "
                f"`{video_identifier}` (backend `{backend}`): {error}"
            )
            state.encoder_unavailable = True
            state.ring = None

        return state.ring

    def _unavailable_anchor(
        self,
        video_identifier: str,
        frame_timestamp: Any,
        frame_number: int,
        settings: _RecorderSettings,
        availability: str,
    ) -> ClipAnchor:
        anchor = ClipAnchor(
            recorder_step=self._step_name(),
            video_identifier=video_identifier,
            frame_timestamp=frame_timestamp,
            frame_number=frame_number,
            availability=availability,
            label=settings.label,
            pre_roll_seconds=settings.pre_roll_seconds,
            post_roll_seconds=settings.post_roll_seconds,
            _capture=None,
        )

        return anchor

    def _all_rings(self) -> List[Any]:
        rings = [
            state.ring for state in self._streams.values() if state.ring is not None
        ]
        rings.extend(self._retired_rings)

        return rings

    def _step_name(self) -> str:
        step_name = getattr(self, "_workflow_step_name", DEFAULT_RECORDER_STEP_NAME)

        return step_name


def _select_backend(image: WorkflowImageData) -> Tuple[str, Type]:
    # NVENC only for frames that already live on a CUDA device, so the check never
    # triggers a numpy -> tensor conversion or a GPU -> CPU download.
    if not image.is_tensor_materialised():
        return BACKEND_X264, X264Ring

    if not getattr(image.tensor_image, "is_cuda", False):
        return BACKEND_X264, X264Ring

    nvenc_ring_class = _available_nvenc_ring_class()
    if nvenc_ring_class is None:
        return BACKEND_X264, X264Ring

    return BACKEND_NVENC, nvenc_ring_class


def _available_nvenc_ring_class() -> Optional[Type]:
    try:
        from roboflow_workflows.core_steps.common.video_clips import nvenc_ring
    except ImportError:
        return None

    try:
        is_available = nvenc_ring.available()
    except Exception as error:
        logger.debug(f"NVENC ring availability check failed: {error}")
        return None

    if not is_available:
        return None

    return nvenc_ring.NvencRing


def _carries_video_metadata(image: WorkflowImageData) -> bool:
    # `WorkflowImageData.video_metadata` synthesises metadata for still images; a
    # recorder fed only such images has no stream to record (stateless HTTP).
    # Reviewer note: there is no public accessor that tells real metadata from the
    # synthesised default without side effects, so this reads the private field on
    # purpose. A public `has_video_metadata` on `WorkflowImageData` would remove it.
    return getattr(image, "_video_metadata", None) is not None


def _frame_resolution(image: WorkflowImageData) -> Optional[Tuple[int, int]]:
    # Read the shape from whichever representation exists, never converting.
    if image.is_tensor_materialised():
        tensor = image.tensor_image
        return int(tensor.shape[-2]), int(tensor.shape[-1])

    numpy_image = image.numpy_image
    if numpy_image is None:
        return None

    return int(numpy_image.shape[0]), int(numpy_image.shape[1])


def _is_stale(
    frame_timestamp: datetime,
    max_processing_lag_seconds: float,
    comes_from_video_file: Optional[bool],
) -> bool:
    if comes_from_video_file:
        # File timestamps are derived from the declared fps, not the wall clock, so
        # their distance to "now" is not a processing lag.
        return False

    if frame_timestamp.tzinfo is None:
        # Naive timestamps are stamped with the local `datetime.now()` (naive UTC on
        # UTC hosts), so compare on that same axis.
        now = datetime.now()
    else:
        now = datetime.now(timezone.utc)
    lag_seconds = (now - frame_timestamp).total_seconds()

    return lag_seconds > max_processing_lag_seconds


def _stream_telemetry(state: _StreamState) -> Dict[str, Any]:
    p50, p95 = _percentiles(samples=state.ring_costs_ms)
    telemetry = {
        "backend": state.backend,
        "frames_submitted": state.frames_submitted,
        "stale_dropped": state.stale_dropped,
        "duplicates_skipped": state.duplicates_skipped,
        "resets": state.resets,
        "ring_cost_ms_p50": p50,
        "ring_cost_ms_p95": p95,
    }

    return telemetry


def _percentiles(samples: Deque[float]) -> Tuple[Optional[float], Optional[float]]:
    if not samples:
        return None, None

    values = np.asarray(samples, dtype=np.float64)
    p50 = float(np.percentile(values, 50))
    p95 = float(np.percentile(values, 95))

    return p50, p95

from collections import OrderedDict
from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, field_validator

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    WILDCARD_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    BlockResult,
    RuntimeRestriction,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY = "output"
IS_AVAILABLE_KEY = "is_available"
REFERENCE_FRAME_KEY = "reference_frame_number"

# Largest supported |offset|. Buffered frames are retained in process memory, so a
# large offset applied to an image stream is expensive: at 1080p a single BGR frame is
# ~6 MB, so |offset| = 256 holds ~1.6 GB per stream.
MAX_OFFSET = 256

# Upper bound on the number of concurrently tracked `video_identifier` values. Streams
# are evicted least-recently-used first so that ended streams cannot retain their
# buffers for the lifetime of the workflow.
MAX_TRACKED_VIDEOS = 16

SHORT_DESCRIPTION = (
    "Reference a value from an earlier frame of the same video stream "
    "(N-k, into the past)."
)

LONG_DESCRIPTION = """
Reference a value produced on an *earlier* frame of the same video stream. Wire any
workflow output (detections, numbers, strings, images, ...) into this block, set a
negative `offset`, and the block returns that value as it was `|offset|` frames ago.

Only past (non-positive) offsets are supported. Genuine future look-ahead would require
delaying the entire workflow output, which is not possible on synchronous runtimes
(WebRTC/webexec, single-image HTTP), so it is intentionally not offered here.

## How This Block Works

1. Reads `video_metadata` from the connected `image` to obtain the `video_identifier`
   (used to keep per-stream state isolated) and the monotonic `frame_number` (`N`).
2. Stores the incoming `data` in a per-video ring buffer keyed by frame number.
3. Resolves `target_frame = N + offset` (with `offset <= 0`) and returns
   `data[target_frame]` when buffered, otherwise `default_value`.
4. Reports `is_available` (whether the target frame was buffered) and
   `reference_frame_number` (always the current frame `N`).

## Common Use Cases

- Compare the current frame to a past frame (`-1`, `-5`) for change/trend detection.
- Align a slow, delayed signal with the frame it belongs to.
- Remember what a value was N frames ago (e.g. the dominant color 10 frames earlier).

## Requirements and Limitations

- `offset` must be `<= 0`. Positive offsets are rejected.
- `|offset|` may not exceed 256. Each buffered frame is held in memory, so delaying an
  image stream by a large offset is costly (~6 MB per frame at 1080p, ~25 MB at 4K).
  Prefer delaying a small derived value over a full image where possible.
- Past offsets work in every execution context; no output delay is introduced.
- State is kept in process memory keyed by `video_identifier`; it degrades on
  stateless/multi-replica remote HTTP runtimes.
- At most 16 streams are tracked concurrently; the least recently seen stream's buffer
  is discarded beyond that.
- A stream whose `frame_number` restarts (e.g. on reconnect) has its buffer cleared.
- State persists for the lifetime of the workflow and resets on restart.
- Values are unavailable (returning `default_value`) until enough frames have been
  processed to reach the requested `|offset|` depth.
"""


def _validate_offset(offset: int) -> None:
    if offset > 0:
        raise ValueError(
            "Frame Delay only supports past offsets: `offset` must be <= 0 "
            f"(got {offset})."
        )
    if offset < -MAX_OFFSET:
        raise ValueError(
            f"Frame Delay supports an `offset` no smaller than -{MAX_OFFSET}, since "
            f"every buffered frame is retained in memory (got {offset})."
        )


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Frame Delay",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-clock-rotate-left",
            },
        }
    )
    # The legacy `time_travel@v1` / `TimeTravel` identifiers are kept as aliases so
    # workflows saved before the rename keep resolving to this block.
    type: Literal[
        "roboflow_core/frame_delay@v1",
        "FrameDelay",
        "roboflow_core/time_travel@v1",
        "TimeTravel",
    ]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Image",
        description="The image / video frame providing the video metadata "
        "(frame number and stream identifier) used to index the buffer.",
        examples=["$inputs.image", "$steps.crop.crops"],
    )
    data: Selector(
        kind=[
            WILDCARD_KIND,
            LIST_OF_VALUES_KIND,
            IMAGE_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ],
    ) = Field(
        description="The value to delay. Can be detections, numbers, strings, "
        "images, or any other workflow output.",
        examples=[
            "$steps.object_detection_model.predictions",
            "$steps.property_definition.output",
        ],
    )
    offset: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        description="Relative frame offset into the past. Must be <= 0: e.g. -1 is the "
        "previous frame, -10 is ten frames ago, 0 is the current frame. "
        f"Limited to -{MAX_OFFSET}, since every buffered frame is held in memory.",
        examples=[-1, -5, -10, "$inputs.offset"],
    )
    default_value: Optional[Union[bool, int, float, str]] = Field(
        default=None,
        description="Value returned when the requested frame is not (yet) available "
        "in the buffer.",
        examples=[None, 0, False],
    )

    @field_validator("offset")
    @classmethod
    def _offset_must_be_in_range(cls, value: Any) -> Any:
        # Selectors (e.g. "$inputs.offset") are strings resolved at runtime and are
        # validated in `run()`; only literal integers can be checked here.
        if isinstance(value, int):
            _validate_offset(value)
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[WILDCARD_KIND]),
            OutputDefinition(name=IS_AVAILABLE_KEY, kind=[BOOLEAN_KIND]),
            OutputDefinition(name=REFERENCE_FRAME_KEY, kind=[INTEGER_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]


class FrameDelayBlockV1(WorkflowBlock):
    def __init__(self):
        # Ordered least- to most-recently-seen video, so that buffers belonging to
        # streams that have ended can be evicted.
        self._buffers: "OrderedDict[str, OrderedDict[int, Any]]" = OrderedDict()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        data: Any,
        offset: int,
        default_value: Optional[Union[bool, int, float, str]] = None,
    ) -> BlockResult:
        offset = int(offset)
        _validate_offset(offset)
        metadata = image.video_metadata
        video_id = metadata.video_identifier or "default"
        frame_number = metadata.frame_number
        buffer = self._track_video(video_id=video_id)
        if buffer and frame_number < next(reversed(buffer)):
            # Frame numbers went backwards, so this is a different stream reusing the
            # identifier (e.g. a reconnect). Nothing buffered under the previous
            # numbering can be resolved against the new one, and keeping it would
            # defeat eviction, whose cutoff is relative to the newest frame.
            buffer.clear()
        buffer[frame_number] = data
        target_frame = frame_number + offset
        is_available = target_frame in buffer
        value = buffer[target_frame] if is_available else default_value
        self._evict(buffer=buffer, newest_frame=frame_number, offset=offset)
        return {
            OUTPUT_KEY: value,
            IS_AVAILABLE_KEY: is_available,
            REFERENCE_FRAME_KEY: frame_number,
        }

    def _track_video(self, video_id: str) -> "OrderedDict[int, Any]":
        buffer = self._buffers.get(video_id)
        if buffer is None:
            buffer = OrderedDict()
        self._buffers[video_id] = buffer
        self._buffers.move_to_end(video_id)
        while len(self._buffers) > MAX_TRACKED_VIDEOS:
            self._buffers.popitem(last=False)
        return buffer

    def _evict(
        self, buffer: "OrderedDict[int, Any]", newest_frame: int, offset: int
    ) -> None:
        # Retain exactly the frames the lookup can reach: `newest_frame + offset` up to
        # `newest_frame`. Older frames are never addressable, since the lookup is an
        # exact frame-number match, so holding them only costs memory.
        cutoff = newest_frame - abs(offset)
        while buffer:
            oldest_frame = next(iter(buffer))
            if oldest_frame >= cutoff:
                break
            buffer.popitem(last=False)

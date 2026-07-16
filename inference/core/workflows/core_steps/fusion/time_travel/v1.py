import os
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.core.utils.environment import str2bool

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

# Extra frames to retain in the per-video buffer beyond the strict lookup window,
# guarding against small frame-number gaps (e.g. dropped frames) when resolving
# past offsets.
BUFFER_MARGIN = 64

# Future look-ahead (positive offset) works by delaying the whole workflow's output
# via the InferencePipeline stream-pipeline emission mechanism. That only applies to
# the InferencePipeline video path (`_execute_inference`); other consumers - e.g. the
# WebRTC/webexec worker and single-image HTTP requests - call the workflow handler
# synchronously and expect an immediate result per frame, so they cannot delay output.
# Activation is therefore opt-in: when disabled (default), the block never changes the
# runner type, so every execution context keeps working. Past offsets work regardless.
STREAM_LOOKAHEAD_ENABLED = str2bool(os.getenv("TIME_TRAVEL_STREAM_LOOKAHEAD", False))

SHORT_DESCRIPTION = (
    "Reference a value from another frame of the same video stream at a relative "
    "offset (N+k for the future, N-k for the past)."
)

LONG_DESCRIPTION = """
Reference a value produced on a *different* frame of the same video stream. Wire any
workflow output (detections, numbers, strings, images, ...) into this block, set an
`offset`, and the block returns that value as it was (or will be) `offset` frames away
from the frame being emitted.

- **Past offset (`offset < 0`)**: returns the value that was produced `|offset|` frames
  earlier. Available immediately once enough frames have been processed.
- **Future offset (`offset > 0`)**: returns the value that will be produced `offset`
  frames later. This is only possible by *delaying* the workflow's output: the block
  declares a stream-pipeline depth equal to `offset`, so on a live `InferencePipeline`
  video run the sink emission is held back by `offset` frames. From the perspective of
  the frame that is finally emitted, "`offset` frames ahead" is the frame currently being
  processed - whose value is already known. The cost is `offset` frames of added latency.

## How This Block Works

1. Reads `video_metadata` from the connected `image` to obtain the `video_identifier`
   (used to keep per-stream state isolated) and the monotonic `frame_number`.
2. Stores the incoming `data` in a per-video ring buffer keyed by frame number.
3. Resolves the requested frame:
   - `reference_frame = frame_number - max(0, offset)` (the frame that will be emitted).
   - `target_frame = reference_frame + offset`.
   - Returns `data[target_frame]` when buffered, otherwise `default_value`.
4. Reports `is_available` (whether the target frame was buffered) and
   `reference_frame_number` (the frame the returned value is aligned to).

## Common Use Cases

- Look ahead to confirm an event persists (e.g. a ball is still detected `+10` frames later).
- Compare the current frame to a past frame (`-1`, `-5`) for change/trend detection.
- Align a slow, delayed signal with the frame it belongs to.

## Requirements and Limitations

- Past offsets (`offset <= 0`) work in every execution context (they need no delay).
- Future look-ahead (`offset > 0`) requires the whole workflow output to be delayed,
  which is only possible on the `InferencePipeline` video path and is opt-in via the
  `TIME_TRAVEL_STREAM_LOOKAHEAD` environment variable (default off). When it is off, a
  positive offset still returns the current frame's value (aligned to
  `reference_frame_number = N - offset`) but without any delay - i.e. no real look-ahead.
- Synchronous consumers - the WebRTC/webexec worker and single-image HTTP requests -
  process one frame at a time and cannot defer output, so genuine future look-ahead is
  not available there even with the flag on.
- State is kept in process memory keyed by `video_identifier`; it degrades on
  stateless/multi-replica remote HTTP runtimes.
- State persists for the lifetime of the workflow and resets on restart.
- v1 supports any number of *past* offsets plus a *single* future look-ahead magnitude
  per workflow. Combining several distinct future offsets (or a future offset together
  with another stream-pipelined block such as RF-DETR) is not supported: the engine's
  emission delay is global and its drain path assumes a single pipelined step.
- The last `offset` frames of a stream cannot look ahead (there is no more future) and
  are emitted without a resolved value.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Time Travel",
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
    type: Literal["roboflow_core/time_travel@v1", "TimeTravel"]
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
        description="The value to time-shift. Can be detections, numbers, strings, "
        "images, or any other workflow output.",
        examples=[
            "$steps.object_detection_model.predictions",
            "$steps.property_definition.output",
        ],
    )
    offset: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        description="Relative frame offset. Positive values look into the future "
        "(and delay the workflow output by that many frames); negative values look "
        "into the past.",
        examples=[10, -1, -5, "$inputs.offset"],
    )
    default_value: Optional[Union[bool, int, float, str]] = Field(
        default=None,
        description="Value returned when the requested frame is not (yet) available "
        "in the buffer.",
        examples=[None, 0, False],
    )

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


class TimeTravelBlockV1(WorkflowBlock):
    def __init__(self):
        self._buffers: Dict[str, "OrderedDict[int, Any]"] = {}
        self._offset: int = 0

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
        self._offset = offset
        metadata = image.video_metadata
        video_id = metadata.video_identifier or "default"
        frame_number = metadata.frame_number
        buffer = self._buffers.setdefault(video_id, OrderedDict())
        buffer[frame_number] = data
        depth = max(0, offset)
        reference_frame = frame_number - depth
        target_frame = reference_frame + offset
        is_available = target_frame in buffer
        value = buffer[target_frame] if is_available else default_value
        self._evict(buffer=buffer, newest_frame=frame_number)
        return {
            OUTPUT_KEY: value,
            IS_AVAILABLE_KEY: is_available,
            REFERENCE_FRAME_KEY: reference_frame,
        }

    def _evict(self, buffer: "OrderedDict[int, Any]", newest_frame: int) -> None:
        # For future offsets the lookup is always the current frame, so only a tiny
        # window is needed; for past offsets we must retain |offset| frames of history.
        window = abs(self._offset) + BUFFER_MARGIN
        cutoff = newest_frame - window
        while buffer:
            oldest_frame = next(iter(buffer))
            if oldest_frame >= cutoff:
                break
            buffer.popitem(last=False)

    def is_stream_pipelined(self) -> bool:
        return STREAM_LOOKAHEAD_ENABLED and self._offset > 0

    def can_activate_stream_pipeline(self) -> bool:
        # Reported before the first `run()` (when `offset` is still unknown) so the
        # InferencePipeline can wrap this step for delayed emission. Gated behind
        # TIME_TRAVEL_STREAM_LOOKAHEAD: when disabled (default) the runner is never
        # wrapped, so synchronous consumers (WebRTC/webexec, HTTP, preview) keep
        # receiving an immediate per-frame result. The effective delay is driven by
        # `stream_pipeline_depth()`, which is 0 until a positive offset is observed.
        return STREAM_LOOKAHEAD_ENABLED

    def stream_pipeline_depth(self) -> int:
        if not STREAM_LOOKAHEAD_ENABLED:
            return 0
        return max(0, self._offset)

    def flush_stream_pipeline_outputs(
        self,
    ) -> List[Tuple[List[Tuple[int, ...]], BlockResult]]:
        # The trailing `offset` frames of a stream have no future to look ahead to, so
        # nothing is emitted for them on drain (an empty flush is a valid contract).
        return []

    def close_stream_pipeline(self) -> None:
        # Intentionally a no-op: `close_stream_pipelines()` is invoked at the end of
        # every non-deferred `run_workflow` call, so the per-video buffers (which must
        # persist across frames) must NOT be cleared here. Memory stays bounded via
        # `_evict`, and the whole block instance is discarded on workflow teardown.
        return None

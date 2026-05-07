import math
import time
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field, field_validator

from inference.core.logger import logger
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

DEFAULT_FPS_FALLBACK = 30.0

MAX_STACK_SIZE = 64
MAX_RESOLUTION_WIDTH = 1920
MAX_RESOLUTION_HEIGHT = 1080
JPEG_QUALITY = 75

LONG_DESCRIPTION = """
Accumulate compressed video frames covering a fixed real-time window. Specify
the window in seconds and the block reads the source FPS from the incoming
stream's video metadata to size the buffer correctly — no need to know the
source FPS up-front.

## How This Block Works

1. Reads source FPS from each frame's `video_metadata.fps`. Falls back to 30
   (with a warning) when the stream has no FPS info.
2. Effective sampling rate defaults to the source FPS. If you set
   `subsample_fps`, frames arriving faster than that rate are dropped before
   any encoding work, which saves CPU when you don't need every frame.
3. Buffer size is derived as ceil(window_seconds * effective_fps).
4. Each kept frame is downsampled to the configured resolution, JPEG-encoded,
   and appended to the per-stream buffer. The oldest frame is evicted when the
   buffer is full.
5. Every call returns the current buffer contents (oldest-first) so downstream
   blocks see a consistent value, even on calls where the input frame was
   dropped by the subsample filter.
"""

SHORT_DESCRIPTION = (
    "Accumulate compressed video frames covering a real-time window. "
    "Sampling rate auto-detected from the stream."
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Stack",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-layer-group",
            },
        }
    )
    type: Literal["roboflow_core/image_stack@v1"]

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Video frame to feed into the time-windowed buffer.",
        examples=["$inputs.image", "$steps.preprocessing.image"],
    )
    window_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=0.5,
        title="Window Duration",
        description=(
            "Length of the time window the buffer covers, in seconds. "
            "Combined with the source FPS (or subsample_fps) this determines "
            "how many frames are kept."
        ),
        examples=[0.5, 1.0, "$inputs.window_seconds"],
    )
    subsample_fps: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(  # type: ignore
        default=None,
        title="Subsample FPS (optional)",
        description=(
            "If set, drop incoming frames so the buffer is sampled at this "
            "rate. If unset, the buffer is sampled at the source's native FPS "
            "(read from video metadata). Useful for cutting cost when the "
            "source delivers more frames than you need."
        ),
        examples=[None, 10.0, "$inputs.subsample_fps"],
    )
    resolution_width: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=MAX_RESOLUTION_WIDTH,
        description=(
            f"Maximum frame width in pixels (64-{MAX_RESOLUTION_WIDTH}). "
            "Frames wider than this are downsampled preserving aspect ratio."
        ),
        examples=[640, 1280, "$inputs.resolution_width"],
    )
    resolution_height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=MAX_RESOLUTION_HEIGHT,
        description=(
            f"Maximum frame height in pixels (64-{MAX_RESOLUTION_HEIGHT}). "
            "Frames taller than this are downsampled preserving aspect ratio."
        ),
        examples=[480, 720, "$inputs.resolution_height"],
    )
    clear: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description=(
            "When True the entire frame buffer is flushed before the current "
            "frame is added."
        ),
        examples=[False, "$inputs.clear_buffer"],
    )

    @field_validator("window_seconds")
    @classmethod
    def validate_window_seconds(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not (0.05 <= float(value) <= 60.0):
            raise ValueError("`window_seconds` must be between 0.05 and 60.0.")
        return value

    @field_validator("subsample_fps")
    @classmethod
    def validate_subsample_fps(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, (int, float)) and not (0.1 <= float(value) <= 60.0):
            raise ValueError("`subsample_fps` must be between 0.1 and 60.0.")
        return value

    @field_validator("resolution_width")
    @classmethod
    def validate_resolution_width(cls, value: Any) -> Any:
        if isinstance(value, int) and not (64 <= value <= MAX_RESOLUTION_WIDTH):
            raise ValueError(
                f"`resolution_width` must be between 64 and {MAX_RESOLUTION_WIDTH}."
            )
        return value

    @field_validator("resolution_height")
    @classmethod
    def validate_resolution_height(cls, value: Any) -> Any:
        if isinstance(value, int) and not (64 <= value <= MAX_RESOLUTION_HEIGHT):
            raise ValueError(
                f"`resolution_height` must be between 64 and {MAX_RESOLUTION_HEIGHT}."
            )
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="frames", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="frames_count", kind=[INTEGER_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def _compress_frame(
    numpy_image: np.ndarray,
    max_width: int,
    max_height: int,
) -> bytes:
    h, w = numpy_image.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        numpy_image = cv2.resize(
            numpy_image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
    return encode_image_to_jpeg_bytes(numpy_image, jpeg_quality=JPEG_QUALITY)


def _read_source_fps(image: WorkflowImageData) -> Optional[float]:
    """Source FPS as declared by the stream's video metadata, or None if absent."""
    try:
        m = image.video_metadata
        if m is not None and m.fps and m.fps > 0:
            return float(m.fps)
    except Exception:
        pass
    return None


def _frame_timestamp(image: WorkflowImageData, source_fps: Optional[float]) -> float:
    """Return a monotonically-increasing seconds timestamp for the input frame.

    Prefers video metadata (frame_number / fps); falls back to wall clock so the
    block stays useful for non-video inputs and tests.
    """
    if source_fps and source_fps > 0:
        try:
            m = image.video_metadata
            if m is not None and m.frame_number is not None:
                return float(m.frame_number) / source_fps
        except Exception:
            pass
    return time.monotonic()


class _BufferState:
    __slots__ = ("frames", "last_sampled_t", "warned_no_fps")

    def __init__(self, maxlen: int) -> None:
        self.frames: deque = deque(maxlen=maxlen)
        self.last_sampled_t: Optional[float] = None
        self.warned_no_fps: bool = False


class ImageStackBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._states: Dict[str, _BufferState] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        window_seconds: float,
        subsample_fps: Optional[float],
        resolution_width: int,
        resolution_height: int,
        clear: bool,
    ) -> BlockResult:
        window_seconds = max(0.05, min(float(window_seconds), 60.0))
        resolution_width = max(64, min(resolution_width, MAX_RESOLUTION_WIDTH))
        resolution_height = max(64, min(resolution_height, MAX_RESOLUTION_HEIGHT))

        try:
            video_id = image.video_metadata.video_identifier
        except Exception:
            video_id = "default"

        # Source FPS comes from the stream's metadata. Fall back to a default
        # only when it's missing (and warn once per video).
        source_fps = _read_source_fps(image)
        if source_fps is None:
            source_fps = DEFAULT_FPS_FALLBACK

        # Effective sampling rate: subsample if user asked, otherwise source.
        # Capped at source — can't keep frames denser than they arrive.
        if subsample_fps is not None:
            effective_fps = min(
                max(0.1, min(float(subsample_fps), 60.0)),
                source_fps,
            )
        else:
            effective_fps = source_fps

        max_buffer = max(
            1, min(math.ceil(window_seconds * effective_fps), MAX_STACK_SIZE)
        )
        sample_interval = 1.0 / effective_fps

        state = self._states.get(video_id)
        if state is None or state.frames.maxlen != max_buffer:
            old_frames = list(state.frames) if state else []
            new_state = _BufferState(maxlen=max_buffer)
            new_state.warned_no_fps = state.warned_no_fps if state else False
            for frame in old_frames[-max_buffer:]:
                new_state.frames.append(frame)
            state = new_state
            self._states[video_id] = state

        if _read_source_fps(image) is None and not state.warned_no_fps:
            logger.warning(
                f"video_metadata.fps not available for video '{video_id}'; "
                f"defaulting to {DEFAULT_FPS_FALLBACK} fps for Image Stack "
                "buffer sizing. Set subsample_fps explicitly to override."
            )
            state.warned_no_fps = True

        if clear:
            state.frames.clear()
            state.last_sampled_t = None

        now_t = _frame_timestamp(image, source_fps)
        should_sample = (
            state.last_sampled_t is None
            or (now_t - state.last_sampled_t) >= sample_interval
        )
        if should_sample:
            compressed = _compress_frame(
                image.numpy_image,
                max_width=resolution_width,
                max_height=resolution_height,
            )
            state.frames.append(compressed)
            state.last_sampled_t = now_t

        frames = list(state.frames)
        return {"frames": frames, "frames_count": len(frames)}

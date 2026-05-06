from collections import deque
from typing import Any, Dict, List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field, field_validator

from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
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

MAX_STACK_SIZE = 64
MAX_RESOLUTION_WIDTH = 1920
MAX_RESOLUTION_HEIGHT = 1080
JPEG_QUALITY = 75

LONG_DESCRIPTION = """
Accumulate compressed video frames into a fixed-size stack, returning the most recent
N frames as JPEG-encoded binary blobs. Designed for shared-hosting safety: frames are
always JPEG-compressed and downsampled to fit within resolution limits, preventing
out-of-memory conditions.

## How This Block Works

1. Receives a video frame (WorkflowImageData) each workflow cycle.
2. Downsamples the frame if it exceeds the configured resolution limits (default
   1920x1080), preserving aspect ratio.
3. JPEG-encodes the frame at quality 75 and stores the resulting bytes.
4. Maintains a per-camera FIFO buffer (deque) of up to `stack_size` compressed frames.
   When the buffer is full the oldest frame is automatically evicted.
5. If `stack_size` changes between calls (e.g. via a dynamic selector), the buffer is
   resized and existing frames are preserved up to the new limit.
6. If the `clear` input is True the buffer is flushed before the current frame is added.
7. Outputs the list of JPEG byte blobs (newest first) and the current frame count.

## Common Use Cases

- **Action / activity recognition**: accumulate a clip of N frames and pass them to a
  vision-language model (e.g. Google Gemini, Qwen) that can reason over multiple images
  to classify actions, detect events, or describe what is happening in a scene.
- **Time-lapse snapshots**: collect the last N frames for periodic visual comparison.
- **Event buffering**: keep a rolling window of frames around an event of interest.
"""

SHORT_DESCRIPTION = (
    "Accumulate compressed video frames into a fixed-size FIFO stack."
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
        description="Video frame to add to the stack.",
        examples=["$inputs.image", "$steps.preprocessing.image"],
    )
    stack_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=10,
        description=(
            f"Maximum number of frames to keep in the stack (1-{MAX_STACK_SIZE}). "
            "When the stack is full the oldest frame is evicted."
        ),
        examples=[5, 10, "$inputs.stack_size"],
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
            "frame is added. Useful for resetting state on scene changes."
        ),
        examples=[False, "$inputs.clear_buffer"],
    )

    @field_validator("stack_size")
    @classmethod
    def validate_stack_size(cls, value: Any) -> Any:
        if isinstance(value, int) and not (1 <= value <= MAX_STACK_SIZE):
            raise ValueError(
                f"`stack_size` must be between 1 and {MAX_STACK_SIZE}."
            )
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
            OutputDefinition(
                name="frames",
                kind=[LIST_OF_VALUES_KIND],
            ),
            OutputDefinition(
                name="frames_count",
                kind=[INTEGER_KIND],
            ),
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


class ImageStackBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffers: Dict[str, deque] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        stack_size: int,
        resolution_width: int,
        resolution_height: int,
        clear: bool,
    ) -> BlockResult:
        stack_size = max(1, min(stack_size, MAX_STACK_SIZE))
        resolution_width = max(64, min(resolution_width, MAX_RESOLUTION_WIDTH))
        resolution_height = max(64, min(resolution_height, MAX_RESOLUTION_HEIGHT))

        video_id = image.video_metadata.video_identifier

        buf = self._buffers.get(video_id)
        if buf is None:
            buf = deque(maxlen=stack_size)
            self._buffers[video_id] = buf
        elif buf.maxlen != stack_size:
            new_buf = deque(buf, maxlen=stack_size)
            buf = new_buf
            self._buffers[video_id] = buf

        if clear:
            buf.clear()

        compressed = _compress_frame(
            image.numpy_image,
            max_width=resolution_width,
            max_height=resolution_height,
        )
        buf.appendleft(compressed)

        frames = list(buf)
        return {
            "frames": frames,
            "frames_count": len(frames),
        }

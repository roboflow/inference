"""Bounded per-source JPEG history with a tensor-facing image input."""

from typing import Any, Dict, Tuple
import cv2
from pydantic import Field, StrictBool, StrictInt, field_validator
from roboflow_workflows.utils.image_encoding import encode_image_to_jpeg_bytes
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


MAX_STACK_SIZE = 64

MAX_RESOLUTION_WIDTH = 1920

MAX_RESOLUTION_HEIGHT = 1080

JPEG_QUALITY = 75

MAX_TRACKED_SOURCES = 256

_LIMITS = {
    "stack_size": (1, MAX_STACK_SIZE),
    "resolution_width": (64, MAX_RESOLUTION_WIDTH),
    "resolution_height": (64, MAX_RESOLUTION_HEIGHT),
}

def _bounded(value: Any, *, name: str) -> int:
    low, high = _LIMITS[name]
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")

    return value

class ImageStackCapacityError(RuntimeError):
    """A new source would exceed the configured source capacity."""

class ImageStackBlock(Block):
    """Keep JPEG bytes of the latest images, newest first, separately per source.

    JPEG quality is 75. Encoding is an explicit CPU boundary: RGB tensor pixels
    become a host BGR array, optionally downsampled with area interpolation.
    Existing bytes keep their original dimensions when resolution limits change.
    Shrinking retains the newest frames; growing cannot restore discarded ones.
    Clear flushes before adding the current image. A lower video frame number
    restarts only that source; an equal number adds another arrival.

    Source identity is video identifier, then the image input's sample source,
    then "default". No source is evicted. Returned lists are fresh; immutable
    JPEG bytes are shared safely. Input tensors are never retained or mutated.
    Outputs carry the current image context, not individual historical geometry.
    The bytes list is one payload and introduces no temporal axis.
    """

    type = "roboflow_core/image_stack@v1"
    metadata = {
        "name": "Image Stack",
        "port_of": "roboflow_core/image_stack@v1",
        "section": "video",
    }
    outputs = {
        "frames": Output(
            LIST_OF_VALUES_KIND,
            source="image",
            description=(
                "JPEG byte list, newest first; one value, not a time axis. No "
                "V2 catalogue block consumes it yet: read it in a custom Python "
                "block or a direct session.run caller. A served JSON response "
                "cannot carry raw bytes; return frames_count instead."
            ),
        ),
        "frames_count": Output(
            INTEGER_KIND, source="image", description="Number of stored JPEG frames."
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Current tensor image to encode.")
        stack_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10, description="History size, 1..64; shrink keeps newest frames."
        )
        resolution_width: StrictInt | Ref(INTEGER_KIND) = Field(
            default=MAX_RESOLUTION_WIDTH,
            description="Maximum encoded width, 64..1920; aspect ratio preserved.",
        )
        resolution_height: StrictInt | Ref(INTEGER_KIND) = Field(
            default=MAX_RESOLUTION_HEIGHT,
            description="Maximum encoded height, 64..1080; aspect ratio preserved.",
        )
        clear: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Flush this source before adding the image."
        )
        max_sources: StrictInt = Field(
            default=MAX_TRACKED_SOURCES,
            ge=1,
            description="Literal source capacity; reject new sources when full.",
        )

        @field_validator("stack_size", "resolution_width", "resolution_height")
        @classmethod
        def _limits(cls, value, info):
            if type(value) is int:
                _bounded(value, name=info.field_name)

            return value

    def __init__(self) -> None:
        self._buffers: Dict[str, Tuple[bytes, ...]] = {}
        self._last_numbers: Dict[str, int] = {}

    def run(
        self,
        *,
        image: ImageData,
        stack_size: int = 10,
        resolution_width: int = MAX_RESOLUTION_WIDTH,
        resolution_height: int = MAX_RESOLUTION_HEIGHT,
        clear: bool = False,
        max_sources: int = MAX_TRACKED_SOURCES,
    ) -> Dict[str, Any]:
        """Encode one arrival and return its source's bounded JPEG history.

        Args:
            image: Current RGB uint8 tensor image.
            stack_size: Maximum retained arrivals, 1..64.
            resolution_width: Maximum width of the new JPEG, 64..1920.
            resolution_height: Maximum height of the new JPEG, 64..1080.
            clear: Whether to discard this source's history before adding.
            max_sources: Positive source capacity, without eviction.

        Returns:
            A fresh ``frames`` list and its ``frames_count``.

        Raises:
            ValueError: When resolved configuration is invalid.
            ImageStackCapacityError: When a new source exceeds capacity.
        """
        _bounded(stack_size, name="stack_size")
        _bounded(resolution_width, name="resolution_width")
        _bounded(resolution_height, name="resolution_height")
        if type(clear) is not bool:
            raise ValueError("clear must be boolean")

        if type(max_sources) is not int or max_sources < 1:
            raise ValueError("max_sources must be a positive integer")

        key = self._source_key(image)
        if key not in self._buffers and len(self._buffers) >= max_sources:
            raise ImageStackCapacityError(
                f"{self.type}: max_sources={max_sources} is full; source {key!r} "
                "was not added. Raise max_sources or reset processing."
            )

        # Public JPEG encoder expects BGR. This is the explicit host boundary.
        pixels = image.tensor_image.detach().cpu().permute(1, 2, 0).numpy()
        if pixels.shape[2] == 3:
            pixels = pixels[:, :, ::-1].copy()
        else:
            pixels = pixels[:, :, 0]
        height, width = pixels.shape[:2]
        scale = min(1.0, resolution_width / width, resolution_height / height)
        if scale < 1:
            pixels = cv2.resize(
                pixels,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        compressed = encode_image_to_jpeg_bytes(pixels, jpeg_quality=JPEG_QUALITY)

        metadata = image.video_metadata
        number = metadata.frame_number if metadata is not None else None
        restart = number is not None and number < self._last_numbers.get(key, number)
        kept = () if clear or restart else self._buffers.get(key, ())
        history = (compressed, *kept)[:stack_size]
        self._buffers[key] = history
        if number is not None:
            self._last_numbers[key] = number
        frames = list(history)
        result = {"frames": frames, "frames_count": len(frames)}

        return result

    def reset_state(self) -> None:
        """Forget all sources, their histories and restart markers."""
        self._buffers.clear()
        self._last_numbers.clear()

    def _source_key(self, image: ImageData) -> str:
        metadata = image.video_metadata
        if metadata is not None and metadata.video_identifier:
            return metadata.video_identifier

        context = self.execution_context
        sample = context.sample_at(context.indices[0], field="image")
        key = sample.source_id if sample is not None else "default"

        return key

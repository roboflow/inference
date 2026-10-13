"""Per-source state of the history-drawing blocks (trace, heatmap).

A step keeps one state per video source. Every call of the step advances that
state once, so the step must see each frame of a source exactly once, whole::

    camera -> tracker -> trace / heatmap -> crop      supported
    camera -> resize -> trace / heatmap               supported (whole frame)
    camera -> crop -> trace / heatmap                 refused: draw before cropping

A crop of one frame shares the frame's source, so drawing crops would age the
history once per crop and mix the crops' coordinates. ``SourceStates.of``
refuses any image that does not cover its whole input frame. An image that is
its own root (an input, a mosaic or another new canvas) always covers it.
"""

import math
from typing import Callable, Dict, Generic, TypeVar

from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import Block

DEFAULT_SOURCE = "default"
"""Source key of an image with neither a video identifier nor a source."""

State = TypeVar("State")


class SourceCapacityError(RuntimeError):
    """A new source would exceed a visualization block's ``max_sources``."""


class SourceStates(Generic[State]):
    """State of one step, one entry per video source, bounded by ``max_sources``.

    A source's state is created on its first frame. A ``frame_number`` lower
    than the last one of the source (a replayed or restarted video) replaces
    it with a new state; an equal or higher number does not.

    Args:
        create: Returns the empty state of a new or restarted source.
    """

    def __init__(self, create: Callable[[], State]) -> None:
        self._create = create
        self._states: Dict[str, State] = {}
        self._last_frame_numbers: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._states)

    def of(self, block: Block, image: ImageData, *, max_sources: int) -> State:
        """Return the state of the source of ``image``.

        Args:
            block: The calling block; its call context names the image's
                source when the image has no video identifier.
            image: The whole frame, or a resize of the whole frame.
            max_sources: Most sources kept at once.

        Returns:
            The source's state, new for a new or restarted source.

        Raises:
            ValueError: When ``image`` covers only part of its input frame
                (a crop). Nothing changes.
            SourceCapacityError: When this is the first frame of a source and
                ``max_sources`` sources already have state. Nothing is created.
        """
        if not covers_root(image):
            raise ValueError(
                f"{block.type} draws a history of whole frames, but this image "
                "is a part of its input frame (a crop). Draw before cropping: "
                f"connect the full frame to {block.type} and crop its output."
            )

        key = source_key(block, image)
        state = self._states.get(key)
        if state is None and len(self._states) >= max_sources:
            raise SourceCapacityError(
                f"{block.type}: source {key!r} would be source "
                f"{len(self._states) + 1} of max_sources={max_sources}; raise "
                "max_sources or reset processing to drop ended sources"
            )

        number = (
            None if image.video_metadata is None else image.video_metadata.frame_number
        )
        restarted = number is not None and number < self._last_frame_numbers.get(
            key, number
        )
        if state is None or restarted:
            state = self._create()
            self._states[key] = state
        if number is not None:
            self._last_frame_numbers[key] = number

        return state

    def clear(self) -> None:
        """Forget every source."""
        self._states.clear()
        self._last_frame_numbers.clear()


def covers_root(image: ImageData) -> bool:
    """Tell whether ``image`` shows its whole input frame, at any scale.

    Args:
        image: The image.

    Returns:
        ``True`` when the root mapping starts at the frame's origin and its
        mapped extents equal the frame's size; ``False`` for a crop.
    """
    root = image.root
    height, width = image.size_hw
    frame_height, frame_width = root.frame_size_hw
    covered = (
        root.offset_xy == (0.0, 0.0)
        and math.isclose(width * root.scale_xy[0], frame_width)
        and math.isclose(height * root.scale_xy[1], frame_height)
    )

    return covered


def source_key(block: Block, image: ImageData) -> str:
    """Name the video source of ``image``, as ``v2/frame_delay@v1`` does.

    Args:
        block: The calling block, read for its call context.
        image: The frame.

    Returns:
        ``video_metadata.video_identifier``, else the image's ``source_id`` in
        the engine, else ``DEFAULT_SOURCE``.
    """
    metadata = image.video_metadata
    if metadata is not None and metadata.video_identifier:
        return metadata.video_identifier

    context = block.execution_context
    sample = context.sample_at(context.indices[0], field="image")
    key = sample.source_id if sample is not None else DEFAULT_SOURCE

    return key

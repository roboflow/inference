"""``roboflow_core/frame_delay@v1``: a value from ``|offset|`` frames earlier, per source.

A port of V1 ``roboflow_core/frame_delay@v1`` (and its tensor variant). Any
selected ``data`` is kept per source, indexed by the ``frame_number`` of the
step's ``image``; the block returns the value stored ``|offset|`` frame
numbers earlier, else ``default_value``::

    pulse            0    1    2    3        (offset = -1)
    cam-a frame      1    2    -    4
    cam-a output     d    1    -    d        frame 3 never arrived: default
    cam-b frame      -    7    8    1
    cam-b output     -    d    7    d        1 < 8: a restart clears cam-b only

History is block state, so it works in passive ``session.run`` and pipelines
and in active runs whose camera batches are sparse. Calls of one step run in
pulse order, which the history relies on.

Two times exist on an output row, deliberately:

* the row (``reference_frame_number``, ``is_available`` and the row's source
  and temporal context) belongs to the current pulse;
* a delayed payload keeps its own metadata; a delayed image keeps the
  ``video_metadata`` of its earlier frame.
"""

from collections import OrderedDict
from typing import Any, Dict, Literal, Optional
from typing import OrderedDict as OrderedDictOf
from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


__all__ = [
    "DEFAULT_SOURCE",
    "FrameDelayBlock",
    "FrameHistoryCapacityError",
    "MAX_OFFSET",
    "MAX_TRACKED_SOURCES",
]

MAX_OFFSET = 256

MAX_TRACKED_SOURCES = 256

DEFAULT_SOURCE = "default"

DefaultValue = None | StrictBool | StrictInt | StrictFloat | StrictStr

ImageDevice = Literal["same", "cpu"]

class FrameHistoryCapacityError(RuntimeError):
    """A new source would exceed FrameDelay's configured history capacity."""

class _History:
    """Kept values of one source, oldest frame number first.

    ``depth`` is the largest ``abs(offset)`` requested since the source
    (re)started; frames older than ``newest - depth`` are dropped.
    """

    def __init__(self) -> None:
        self.frames: OrderedDictOf[int, Any] = OrderedDict()
        self.depth = 0

    def restart(self) -> None:
        self.frames.clear()
        self.depth = 0

    def forget_before(self, number: int) -> None:
        while self.frames and next(iter(self.frames)) < number:
            self.frames.popitem(last=False)

class FrameDelayBlock(Block):
    """Return ``data`` as it was ``|offset|`` frame numbers earlier.

    Per call, with ``N = image.video_metadata.frame_number``::

        history = histories[source key]          # bounded by max_sources
        if N < newest kept number: history.restart()   # equal N: no restart
        depth = max(depth, abs(offset))
        history[N] = owned(data)                 # equal N replaces the value
        output = history[N + offset] if kept else default_value
        drop numbers < N - depth

    Frames are matched by frame number, not by arrival: a missing number is
    unavailable. A selected ``offset`` may change per frame; history is kept
    for the deepest offset seen, so returning to a deep offset finds it.

    Source key, as for the cache blocks: ``video_metadata.video_identifier``,
    else the ``image`` input's own ``SampleContext.source_id``, else
    ``"default"``. A ``data`` without a context, such as a workflow
    parameter passed on by a step, does not change the key.

    Frames are not told apart by crop: several crops of one frame (for example
    ``image=$steps.crop.crops``) share a source and frame number, so each
    call replaces the value the previous crop stored and later frames see
    the last crop's value. Delay per-crop data only when one value per frame
    is meant.

    Ownership of kept values (V1 tensor variant):

    - ``ImageData``, directly or inside lists/tuples (nested), is kept as
      ``ImageData.copy()``, so a camera or decoder buffer is released after
      the pulse. ``image_device="cpu"`` keeps the copy in host memory.
    - Everything else is kept by reference, images inside dicts or
      predictions included. Changing such a value in place changes history.
    - ``output`` is the kept object itself (for ``offset=0`` too). Steps
      that change it in place change the history; the block declares
      ``retains_inputs`` and ``retains_outputs``, so the compiler reports
      such steps.

    A present ``None`` in ``data`` is kept and returned as ``None`` with
    ``is_available=True``. A filtered or missing ``data`` skips the call:
    nothing is kept for that frame, and a later lookup of it returns
    ``default_value``.

    ``max_sources`` is a literal bound (default 256). A new source beyond it
    raises ``FrameHistoryCapacityError`` before copying data or changing history.
    Sources are never evicted; raise the bound or reset processing to free them.
    The bound does not preallocate storage.

    ``reset_state()`` forgets every source's history and nothing else.
    """

    type = "roboflow_core/frame_delay@v1"
    metadata = {
        "name": "Frame Delay",
        "port_of": "roboflow_core/frame_delay@v1",
        "section": "video",
    }
    outputs = {
        # No source field, so the row context is the common context of image
        # and data: a constant data keeps the current image's context, which
        # source="data" would drop. Without a source the compiler's mutation
        # checks trace no input through this output; retains_inputs and
        # retains_outputs report changes to kept and returned values instead.
        # With offset 0 the output is the data object itself, an undeclared
        # alias those checks do not follow. The history key is separate: it
        # reads only the image's context (_source_key).
        "output": Output(
            description=(
                "Data from frame number current + offset of this source, or "
                "default_value when it was not kept. Shared with the history."
            ),
        ),
        "is_available": Output(
            BOOLEAN_KIND,
            source="image",
            description="Whether frame number current + offset was kept.",
        ),
        "reference_frame_number": Output(
            INTEGER_KIND,
            source="image",
            description="Frame number of the current image, not of the delayed one.",
        ),
    }
    retains_inputs = ("data",)
    retains_outputs = ("output",)

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Current frame; its video_metadata gives the frame number and "
                "the source. The block runs once per image."
            ),
            examples=["$inputs.image", "$steps.crop.crops"],
        )
        data: Ref(allow_none=True) = Field(
            description=(
                "Value to delay: detections, numbers, strings, images or any "
                "other value, None included. A filtered or missing value "
                "skips the call."
            ),
            examples=["$steps.model.predictions", "$inputs.image"],
        )
        offset: StrictInt | Ref(INTEGER_KIND) = Field(
            description=(
                f"Frames into the past, -{MAX_OFFSET} to 0: -1 is the previous "
                "frame, 0 the current one."
            ),
            examples=[-1, -5, "$inputs.offset"],
        )
        default_value: DefaultValue = Field(
            default=None,
            description="Returned when the requested frame was not kept.",
            examples=[None, 0, False],
        )
        image_device: ImageDevice = Field(
            default="same",
            description=(
                "Where kept images live: 'same' as the input or 'cpu' to free "
                "device memory for the whole delay."
            ),
        )

        max_sources: StrictInt = Field(
            default=MAX_TRACKED_SOURCES,
            ge=1,
            description=(
                "Most sources with kept frame history. A new source beyond this "
                "literal bound raises FrameHistoryCapacityError without evicting "
                "history. Raise the bound or reset processing to free sources. "
                "Storage is allocated only for sources and frames seen."
            ),
            examples=[256, 512],
        )

        @field_validator("offset")
        @classmethod
        def _offset_in_range(cls, value: Any) -> Any:
            # A selector is a string here; its value is checked in run().
            if isinstance(value, int):
                _check_offset(value)

            return value

    def __init__(self) -> None:
        self._histories: Dict[str, _History] = {}

    def run(
        self,
        *,
        image: ImageData,
        data: Any,
        offset: int,
        default_value: DefaultValue,
        image_device: ImageDevice,
        max_sources: int = MAX_TRACKED_SOURCES,
    ) -> Dict[str, Any]:
        """Keep ``data`` for this frame and look up the one ``offset`` earlier.

        Args:
            image: Current frame; gives the frame number and the source.
            data: Value to keep, ``None`` included.
            offset: Frame offset from ``-MAX_OFFSET`` to 0.
            default_value: Output when the requested frame was not kept.
            image_device: ``"same"`` or ``"cpu"`` for kept images.
            max_sources: Positive literal bound on sources with kept history.

        Returns:
            ``output``, ``is_available`` and ``reference_frame_number``.

        Raises:
            ValueError: When ``image`` has no ``video_metadata`` or a
                selected ``offset`` is out of range, or ``max_sources`` is not
                a positive integer.
            FrameHistoryCapacityError: When a new source exceeds ``max_sources``.
                The rejected source changes no history; earlier batch rows
                may already have updated theirs.
        """
        if type(max_sources) is not int or max_sources < 1:
            raise ValueError("max_sources must be a positive integer")

        offset = int(offset)
        _check_offset(offset)
        if image.video_metadata is None:
            raise ValueError(
                f"{self.type} matches frames by video_metadata.frame_number; "
                f"image {image.image_id!r} has no video_metadata"
            )

        number = image.video_metadata.frame_number
        history = self._source_history(self._source_key(image), max_sources=max_sources)
        if history.frames and number < next(reversed(history.frames)):
            history.restart()
        history.depth = max(history.depth, -offset)

        device = None if image_device == "same" else "cpu"
        history.frames[number] = _owned(data, device=device)
        is_available = number + offset in history.frames
        output = history.frames[number + offset] if is_available else default_value
        history.forget_before(number - history.depth)

        result = {
            "output": output,
            "is_available": is_available,
            "reference_frame_number": number,
        }

        return result

    def reset_state(self) -> None:
        """Forget every source's kept values."""
        self._histories.clear()

    def _source_key(self, image: ImageData) -> str:
        if image.video_metadata.video_identifier:
            return image.video_metadata.video_identifier

        # The image's own context: a ``data`` without one must not drop it
        # from the call's context.
        context = self.execution_context
        sample = context.sample_at(context.indices[0], field="image")
        key = sample.source_id if sample is not None else DEFAULT_SOURCE

        return key

    def _source_history(self, key: str, *, max_sources: int) -> _History:
        history = self._histories.get(key)
        if history is None:
            if len(self._histories) >= max_sources:
                raise FrameHistoryCapacityError(
                    f"{self.type} cannot keep history for source {key!r}: "
                    f"max_sources={max_sources} is full. The rejected source changes "
                    "no history; earlier batch rows may already have updated theirs. "
                    "Raise max_sources or reset processing to free sources."
                )

            history = _History()
            self._histories[key] = history

        return history

def _check_offset(offset: int) -> None:
    if not -MAX_OFFSET <= offset <= 0:
        raise ValueError(
            f"offset must be from -{MAX_OFFSET} to 0 (past frames only), "
            f"got {offset}"
        )

def _owned(value: Any, *, device: Optional[str]) -> Any:
    """``value`` with every image in it, or in nested lists/tuples, copied."""
    if isinstance(value, ImageData):
        copied = value.copy(device=device)
        return copied

    if isinstance(value, (list, tuple)):
        items = [_owned(item, device=device) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        owned = tuple(items) if isinstance(value, tuple) else items
        return owned

    return value

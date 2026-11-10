"""``roboflow_core/background_subtraction@v1``: a foreground mask per frame, per source.

A port of V1 ``roboflow_core/background_subtraction@v1``. Each source has
its own OpenCV MOG2 background model, which learns from every frame it sees::

    image (C, H, W) uint8, any device
      ├─ source key ── that source's MOG2 model (created on its first frame)
      ├─ pixels -> host (H, W, C)            CPU boundary: OpenCV MOG2
      ├─ model.apply -> (H, W) mask: 0 background, 127 shadow, 255 foreground
      └─ mask -> image device, 3 equal channels, image geometry kept

``SourceModels`` keeps one ``SourceState`` per source; ``roboflow_core/motion_detection@v1``
shares it. A record holds everything that restarts together: the model, the
frames it has seen and the source's previous motion.

State and its limits:

* Source key: ``video_metadata.video_identifier``, else the image's own
  ``SampleContext.source_id``, else ``"default"`` (also for a plain ``run()``
  outside the engine). Sources never share a model. V1 kept one model for
  every frame the step saw.
* A source restarts (a new record: fresh background, no frames seen, no
  previous motion) when the frame number is lower than the last one seen, or
  when the image size or channel count changes. Equal or skipped frame
  numbers are ordinary frames.
* Concurrency: nothing is locked. The engine calls one step one call at a
  time, in pulse order, and each call's whole transition relies on that. Do
  not call one block instance from several threads at once.
* ``history`` counts processed frames, not time: frames dropped before the
  step never reach the model. A changed ``history`` or ``threshold`` is
  applied to the live model; V1 kept the values of the first frame.
* The first frame of a (re)started model is all foreground (255), as in V1.
* ``max_sources`` bounds the sources. A new source beyond it raises
  ``BackgroundModelCapacityError`` before any state changes. Sources are never
  evicted; ``reset_state()`` or a processing reset frees them all. One
  3-channel 1080p model holds about 216 MB of host memory.
* MOG2 compares colour vectors, so RGB and BGR pixels give the same mask.
"""

from dataclasses import dataclass
from typing import Annotated, Any, Dict, Optional, Tuple
import cv2
import numpy as np
import torch
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


__all__ = [
    "BackgroundModelCapacityError",
    "BackgroundSubtractionBlock",
    "DEFAULT_HISTORY",
    "DEFAULT_MAX_SOURCES",
    "DEFAULT_SOURCE",
    "DEFAULT_THRESHOLD",
    "PositiveInt",
    "SourceModels",
    "SourceState",
]

DEFAULT_THRESHOLD = 16

DEFAULT_HISTORY = 30

DEFAULT_MAX_SOURCES = 16

DEFAULT_SOURCE = "default"

PositiveInt = Annotated[StrictInt, Field(ge=1)]

class BackgroundModelCapacityError(RuntimeError):
    """A new source would exceed the step's ``max_sources``."""

@dataclass
class SourceState:
    """Everything one source restarts together.

    Attributes:
        subtractor: The source's MOG2 model.
        shape: ``(C, H, W)`` of the frames the model learned from.
        last_frame_number: Frame number of the last frame, if it had one.
        frames_seen: Frames applied since the source (re)started.
        previous_motion: Motion of the source's last evaluated frame; only
            ``roboflow_core/motion_detection@v1`` reads and writes it.
    """

    subtractor: cv2.BackgroundSubtractorMOG2
    shape: Tuple[int, ...]
    last_frame_number: Optional[int] = None
    frames_seen: int = 0
    previous_motion: bool = False

class SourceModels:
    """One ``SourceState`` per source, bounded and never evicted.

    One instance belongs to one block instance and is not thread-safe: the
    engine's one-call-at-a-time step order is the only guarantee.
    """

    def __init__(self, owner: str) -> None:
        self._owner = owner
        self._states: Dict[str, SourceState] = {}

    def apply(
        self,
        image: ImageData,
        *,
        source: str,
        history: int,
        threshold: int,
        max_sources: int,
    ) -> Tuple[np.ndarray, SourceState]:
        """Update ``source``'s model with ``image`` and return its mask.

        Args:
            image: Frame to learn from and classify.
            source: Source key of the frame.
            history: Processed frames the model remembers.
            threshold: MOG2 ``varThreshold``.
            max_sources: Most sources with a model.

        Returns:
            The ``(H, W)`` ``uint8`` host mask and the source's record, with
            ``frames_seen`` counting this frame. A restart returns a new
            record.

        Raises:
            BackgroundModelCapacityError: When ``source`` is new and
                ``max_sources`` sources already have a model.
        """
        shape = tuple(image.tensor_image.shape)
        number = _frame_number(image)
        state = self._states.get(source)
        if state is None and len(self._states) >= max_sources:
            raise BackgroundModelCapacityError(
                f"{self._owner}: source {source!r} would be source "
                f"{len(self._states) + 1} of max_sources={max_sources}; no "
                "source state was changed. Raise max_sources, or restart the "
                "pipeline to free ended sources."
            )
        pixels = _host_pixels(image)
        if state is None or _restarts(state, shape=shape, number=number):
            state = SourceState(_new_subtractor(history, threshold), shape)
            self._states[source] = state
        state.subtractor.setHistory(history)
        state.subtractor.setVarThreshold(float(threshold))
        mask = state.subtractor.apply(pixels)
        state.last_frame_number = number
        state.frames_seen += 1

        return mask, state

    def clear(self) -> None:
        """Forget every source's state."""
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)

class BackgroundSubtractionBlock(Block):
    """Mask the pixels that differ from each source's learned background.

    ``image`` is a mask with the input's geometry, identity and device:
    3 equal ``uint8`` channels, 255 foreground, 127 shadow, 0 background.
    OpenCV MOG2 runs on the CPU; GPU frames are copied to the host.
    """

    type = "roboflow_core/background_subtraction@v1"
    metadata = {
        "name": "Background Subtraction",
        "port_of": "roboflow_core/background_subtraction@v1",
        "section": "video",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description=(
                "Foreground mask of the frame: 255 foreground, 127 shadow, "
                "0 background, in 3 equal channels."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame of a video source. Its video_metadata names the source "
                "and gives the frame number. Images without a video identifier "
                "use their input source, else share one 'default' model."
            ),
            examples=["$inputs.image"],
        )
        threshold: PositiveInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_THRESHOLD,
            description=(
                "MOG2 varThreshold: squared colour distance from the "
                "background that counts as foreground. Lower is more sensitive."
            ),
            examples=[DEFAULT_THRESHOLD, "$inputs.threshold"],
        )
        history: PositiveInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_HISTORY,
            description="Processed frames that shape each source's background.",
            examples=[DEFAULT_HISTORY, "$inputs.history"],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with a background model (~216 MB per 1080p "
                "source). A frame of one more source fails with "
                "BackgroundModelCapacityError; no model is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._sources = SourceModels(self.type)

    def run(
        self,
        *,
        image: ImageData,
        threshold: int = DEFAULT_THRESHOLD,
        history: int = DEFAULT_HISTORY,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> Dict[str, Any]:
        """Update this source's model with one frame and return its mask.

        Args:
            image: Frame of a video source.
            threshold: MOG2 ``varThreshold``, a positive integer.
            history: Processed frames the model remembers, a positive integer.
            max_sources: Most sources with a model.

        Returns:
            ``image``: the mask as an image with the input's geometry.

        Raises:
            ValueError: When ``threshold``, ``history`` or ``max_sources`` is
                not a positive integer.
            BackgroundModelCapacityError: When this is the first frame of a
                source and ``max_sources`` sources already have a model.
        """
        check_positive_int(
            self.type, threshold=threshold, history=history, max_sources=max_sources
        )
        mask, _ = self._sources.apply(
            image,
            source=source_key(self, image),
            history=history,
            threshold=threshold,
            max_sources=max_sources,
        )
        result = {"image": image.with_pixels(_mask_pixels(mask, like=image))}

        return result

    def reset_state(self) -> None:
        """Forget every source's background model."""
        self._sources.clear()

def source_key(block: Block, image: ImageData) -> str:
    """Source of ``image``: video identifier, else sample source, else default.

    Args:
        block: Running block; its call context names the image's source.
        image: Frame whose source is wanted.

    Returns:
        The source key.
    """
    metadata = image.video_metadata
    if metadata is not None and metadata.video_identifier:
        return metadata.video_identifier

    # The image's own context: another input without one must not drop it.
    # A plain run() has no context; errors in an engine call still propagate.
    try:
        context = block.execution_context
    except NoExecutionContextError:
        return DEFAULT_SOURCE

    sample = context.sample_at(context.indices[0], field="image")
    key = sample.source_id if sample is not None else DEFAULT_SOURCE

    return key

def check_positive_int(owner: str, **values: Any) -> None:
    """Reject a value that is not a positive ``int``; selected values too.

    Args:
        owner: Block type named in the error.
        **values: Parameter names and their values.

    Raises:
        ValueError: Naming the first parameter that is not a positive int.
    """
    for name, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(
                f"{owner}: {name} must be a positive integer, got {value!r}"
            )

def _new_subtractor(history: int, threshold: int) -> cv2.BackgroundSubtractorMOG2:
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=float(threshold), detectShadows=True
    )

    return subtractor

def _frame_number(image: ImageData) -> Optional[int]:
    metadata = image.video_metadata
    number = None if metadata is None else metadata.frame_number

    return number

def _restarts(
    state: SourceState, *, shape: Tuple[int, ...], number: Optional[int]
) -> bool:
    went_back = (
        number is not None
        and state.last_frame_number is not None
        and number < state.last_frame_number
    )

    return went_back or shape != state.shape

def _host_pixels(image: ImageData) -> np.ndarray:
    # CPU boundary: one device-to-host copy; (H, W) or (H, W, 3) for OpenCV.
    pixels = image.tensor_image.detach()
    hwc = pixels[0] if pixels.shape[0] == 1 else pixels.permute(1, 2, 0)
    host = hwc.to("cpu").contiguous().numpy()

    return host

def _mask_pixels(mask: np.ndarray, *, like: ImageData) -> torch.Tensor:
    # One host-to-device copy of a single channel, repeated on the device.
    single = torch.from_numpy(mask).to(like.device)
    pixels = single.unsqueeze(0).expand(3, -1, -1).contiguous()

    return pixels

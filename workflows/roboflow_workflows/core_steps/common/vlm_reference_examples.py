"""Labeled example images for VLM object detection ("label by example").

A reference example is an image with its annotations. Detection blocks send the
examples before the image to label, so the model sees what each label looks like
and how tightly it is boxed. There are two renderings because the providers read
them differently: whole images with the boxes drawn on them (Gemini) and crops of
the boxes grouped by label (OpenAI). The examples go first in the request, so every
image of a batch shares the same prefix and the provider can cache it. The example
images load and render one at a time, so they hold one full-size image at most. The
blocks keep the rendered parts for the whole process in
``SHARED_REFERENCE_PARTS_CACHE``.

Oriented object detection takes the same examples with an ``obb`` on every
annotation, the native ``[center x, center y, width, height, angle]``. Its crops
follow the rotated box and are turned upright: the front of a class with a front is
on the right, and the long side of the others is horizontal. Its boxed images draw
the rotated outline, with a triangle on the front edge of a class with a front.
"""

import base64
import hashlib
import json
import math
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from roboflow_workflows.core_steps.common.deserializers import deserialize_image_kind
from roboflow_workflows.utils.images import encode_image_to_jpeg_bytes, load_image

REFERENCE_EXAMPLES_TASK = "object-detection"
ORIENTED_REFERENCE_EXAMPLES_TASK = "oriented-object-detection"
REFERENCE_EXAMPLES_TASKS = (REFERENCE_EXAMPLES_TASK, ORIENTED_REFERENCE_EXAMPLES_TASK)
MAX_REFERENCE_EXAMPLES = 10
MAX_CROPS_PER_LABEL = 3
BOXED_IMAGE_MAX_EDGE = 1024
CROP_MARGIN = 0.15
CROP_MIN_MARGIN = 8
CROP_MIN_EDGE = 128
CROP_MAX_EDGE = 384
CROP_FILL_VALUE = 114
SELECTOR_PREFIXES = ("$inputs.", "$steps.")
REFERENCE_PARTS_CACHE_ENTRIES = 8
REFERENCE_PARTS_CACHE_MAX_BYTES = 32 * 1024 * 1024
REFERENCE_PARTS_CACHE_TTL_SECONDS = 15 * 60
REFERENCE_PARTS_PENDING_TIMEOUT_SECONDS = 60.0
_IMAGE_KEY_CHUNK_SIZE = 64 * 1024

REFERENCE_CROPS_INTRO = (
    "Reference examples: crops of labeled objects taken from other images of the "
    "same dataset, grouped by label. They show what each label looks like."
)
REFERENCE_BOXED_INTRO = (
    "Reference examples: other images of the same dataset where every object with "
    "one of the labels is outlined by a colored box with its label written on it. "
    "They show what each label looks like and how tightly it is boxed."
)
REFERENCE_NUMBERED_INTRO = (
    "Reference examples: other images of the same dataset where every object with "
    "one of the labels is outlined by a colored box with a number written on it. "
    "They show what each label looks like and how tightly it is boxed. The numbers "
    "stand for these labels:"
)
REFERENCE_ORIENTED_CROPS_INTRO = (
    "Reference examples: crops of labeled objects taken from other images of the "
    "same dataset, grouped by label. Each crop follows the rotated box of its object "
    "and is turned so that the box is horizontal; for a label with a front, the "
    "front is on the right. They show what each label looks like."
)
REFERENCE_ORIENTED_BOXED_INTRO = (
    "Reference examples: other images of the same dataset where every object with "
    "one of the labels is outlined by a colored rotated box with its label written "
    "on it. They show what each label looks like and how tightly it is boxed."
)
REFERENCE_ORIENTED_NUMBERED_INTRO = (
    "Reference examples: other images of the same dataset where every object with "
    "one of the labels is outlined by a colored rotated box with a number written "
    "on it. They show what each label looks like and how tightly it is boxed."
)
REFERENCE_FRONT_MARKER_RULE = (
    " A filled triangle outside one side of a box marks the front of an object "
    "that has a front."
)
REFERENCE_NUMBERS_LEGEND_INTRO = " The numbers stand for these labels:"
TARGET_IMAGE_INTRO = "Target image:"
# On its own line, because a detection prompt can end with the list of labels.
TARGET_ONLY_RULE = (
    "\nThe reference images are not part of the task: return detections for the "
    "target image only."
)

REFERENCE_EXAMPLES_DESCRIPTION = (
    "Labeled example images that show the model what each class looks like and how "
    "tightly it is boxed (object-detection only). Each example is "
    '`{"image": <URL, base64 or {"type": ..., "value": ...}>, "annotations": '
    '[{"class": ..., "x": ..., "y": ..., "width": ..., "height": ...}]}`, in the '
    "Roboflow prediction format: box centers and sizes in pixels of that example "
    "image, `class` one of the block's labels. Annotate every object of those labels "
    "in each example. Local file paths are not read. At most "
    f"{MAX_REFERENCE_EXAMPLES} examples, sent with every image, so each request costs "
    "more input tokens. A selector can provide the whole list, but selectors inside "
    "an example are not resolved."
)
REFERENCE_EXAMPLES_EXAMPLE = [
    {
        "image": {"type": "url", "value": "https://example.com/labeled.jpg"},
        "annotations": [{"class": "cat", "x": 120, "y": 80, "width": 40, "height": 60}],
    }
]

# RGB, picked to stay apart from each other on photos.
_BOX_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (128, 0, 0),
    (0, 0, 128),
]


OrientedBoxValues = Annotated[List[float], Field(min_length=5, max_length=5)]


class ReferenceAnnotation(BaseModel):
    """One annotation of a reference example, in the Roboflow prediction format.

    Object detection reads the class and the box. Oriented object detection reads the
    class and `obb`, the native `[center x, center y, width, height, angle]` of a
    prediction; a box given as `obb` alone gets the axis-aligned `x`, `y`, `width` and
    `height` that contain it. The same list can serve the other tasks, which read
    other fields of a prediction: `points` or `rle_mask` for instance segmentation,
    only `class` for classification. Fields a task does not read are ignored, so a
    whole prediction is a valid annotation.
    """

    model_config = ConfigDict(populate_by_name=True)

    x: float = Field(description="Box center x, in pixels of the example image.")
    y: float = Field(description="Box center y, in pixels of the example image.")
    width: float = Field(gt=0, description="Box width in pixels.")
    height: float = Field(gt=0, description="Box height in pixels.")
    class_name: str = Field(
        alias="class",
        min_length=1,
        description="Label of the object, one of the block's labels.",
        examples=["cat"],
    )
    obb: Optional[OrientedBoxValues] = Field(
        default=None,
        description=(
            "Oriented box `[center x, center y, width, height, angle]`, in pixels of "
            "the example image: `width` runs along the heading (toward the front of a "
            "class with a front) and `angle` is that heading in radians, clockwise "
            "from +x. Required for oriented object detection."
        ),
        examples=[[120, 80, 60, 20, 0.5]],
    )

    @model_validator(mode="before")
    @classmethod
    def _contain_the_oriented_box(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("obb") is None:
            return data

        bounds = _oriented_box_bounds(data["obb"])
        if bounds is None:
            # `obb` is malformed and fails its own validation. Placeholders keep the
            # error about it from being buried in "field required" ones.
            bounds = (0.0, 0.0, 1.0, 1.0)

        filled = {**dict(zip(("x", "y", "width", "height"), bounds)), **data}
        return filled

    @field_validator("obb")
    @classmethod
    def _check_the_oriented_box(
        cls, value: Optional[List[float]]
    ) -> Optional[List[float]]:
        if value is None:
            return value

        if not all(math.isfinite(item) for item in value):
            raise ValueError("every value must be finite")

        if value[2] <= 0 or value[3] <= 0:
            raise ValueError("width and height must be greater than 0")

        return value


class ReferenceExample(BaseModel):
    """An example image with every object of the block's labels annotated."""

    image: Union[str, Dict[str, Any]] = Field(
        description='Image as a URL, a base64 string or `{"type": ..., "value": ...}`.',
        examples=[{"type": "url", "value": "https://example.com/labeled.jpg"}],
    )
    annotations: List[ReferenceAnnotation] = Field(
        description="Every object of the block's labels in the image."
    )

    @field_validator("image", "annotations", mode="before")
    @classmethod
    def _refuse_selectors(cls, value: Any) -> Any:
        source = value.get("value") if isinstance(value, dict) else value
        if isinstance(source, str) and source.startswith(SELECTOR_PREFIXES):
            raise ValueError(
                "selectors are not resolved inside an example; pass the whole list "
                "through one, such as `$inputs.reference_examples`"
            )

        return value


ReferenceExamples = Annotated[
    List[ReferenceExample], Field(max_length=MAX_REFERENCE_EXAMPLES)
]


@dataclass(frozen=True)
class LoadedReferenceExample:
    """A decoded example image (BGR) with its boxes.

    ``boxes`` are (label, x_min, y_min, x_max, y_max) clipped to the image. For the
    oriented task ``oriented_boxes`` holds (label, center x, center y, width, height,
    angle) instead, and ``boxes`` is empty.
    """

    image: np.ndarray
    boxes: Tuple[Tuple[str, float, float, float, float], ...]
    oriented_boxes: Tuple[Tuple[str, float, float, float, float, float], ...] = ()


@dataclass(frozen=True)
class _CachedParts:
    parts: List[dict]
    size: int
    expires_at: float


class _PendingBuild:
    """A build in progress; callers asking for its key wait for its outcome."""

    def __init__(self, *, expires_at: float) -> None:
        self.expires_at = expires_at
        self._done = threading.Event()
        self._parts: List[dict] = []
        self._error: Optional[BaseException] = None

    def finish(self, parts: List[dict]) -> None:
        if self._done.is_set():
            return

        self._parts = parts
        self._done.set()

    def fail(self, error: BaseException) -> None:
        if self._done.is_set():
            return

        self._error = error
        self._done.set()

    def wait(self, *, timeout: float) -> List[dict]:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for reference examples to render")

        if self._error is not None:
            raise self._error

        return self._parts


class ReferencePartsCache:
    """Reference parts built recently, shared by the block instances of a process.

    Each execution engine has its own block instances, and a host can run several
    engines at once, such as one per thread. The blocks share
    ``SHARED_REFERENCE_PARTS_CACHE``, so a process fetches and renders an example set
    once instead of once per engine.

    A hosted server runs the requests of many callers in one process, so an entry only
    serves identical examples: its key holds every example as given (the image URL or
    data, and the annotations), the labels and the rendering. A caller gets the parts
    built for another only by sending the same examples, and example images load
    through the process's image codec, under the same URL policy for every caller.
    Examples with an in-memory image have no key and render on every call.

    An entry expires ``ttl_seconds`` after it is built, so a later run sees what the
    URLs serve by then. The least recently used entries go first once there are more
    than ``max_entries`` or their strings, mostly base64 images, exceed ``max_bytes``;
    parts larger than ``max_bytes`` are not kept. Callers that ask for a key while it
    is being built wait for that build and get its parts or its error, while other keys
    build at the same time. A failed build is not kept, so the next call builds again.
    Waiting ends with TimeoutError at most ``pending_timeout_seconds`` after the
    build started. Expired builds no longer own their key, so later callers can
    retry. A late result is returned only to its original caller and never cached.
    The cache does not cancel synchronous host image loading or start worker threads.
    """

    def __init__(
        self,
        *,
        max_entries: int = REFERENCE_PARTS_CACHE_ENTRIES,
        max_bytes: int = REFERENCE_PARTS_CACHE_MAX_BYTES,
        ttl_seconds: float = REFERENCE_PARTS_CACHE_TTL_SECONDS,
        pending_timeout_seconds: float = REFERENCE_PARTS_PENDING_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty cache.

        Args:
            max_entries: Most example sets kept at once.
            max_bytes: Most bytes occupied by retained string values, including
                their Python object headers and Unicode representation.
            ttl_seconds: How long an entry serves after it is built.
            pending_timeout_seconds: How long other callers may wait for a build,
                measured from its start. Must be finite and positive.
            clock: Monotonic time in seconds.

        Raises:
            ValueError: If pending_timeout_seconds is not finite and positive.
        """
        if not math.isfinite(pending_timeout_seconds) or pending_timeout_seconds <= 0:
            raise ValueError("pending_timeout_seconds must be finite and positive")

        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_seconds
        self._pending_timeout_seconds = pending_timeout_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, _CachedParts]" = OrderedDict()
        self._size = 0
        self._pending: Dict[str, _PendingBuild] = {}

    def get_or_build(
        self, key: Optional[str], *, build: Callable[[], List[dict]]
    ) -> List[dict]:
        """Return the parts of ``key``, building them unless an entry has them.

        Args:
            key: Identifies the examples, the labels and the rendering; ``None``
                builds without caching.
            build: Fetches and renders the examples.

        Returns:
            Content parts to place before the target image, shared with other
            callers.

        Raises:
            TimeoutError: If the shared build exceeds its pending deadline.
            Exception: Whatever ``build`` raised, also in the callers that waited
                for that build.
        """
        if key is None:
            parts = build()
            return parts

        # A digest keeps the keys small when examples carry base64 images.
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        with self._lock:
            self._drop_expired()
            entry = self._entries.get(digest)
            if entry is not None:
                self._entries.move_to_end(digest)
                return entry.parts

            pending = self._pending.get(digest)
            is_builder = pending is None
            if pending is None:
                pending = _PendingBuild(
                    expires_at=self._clock() + self._pending_timeout_seconds
                )
                self._pending[digest] = pending

        if not is_builder:
            try:
                parts = pending.wait(
                    timeout=max(0.0, pending.expires_at - self._clock())
                )
            except TimeoutError as error:
                with self._lock:
                    if self._pending.get(digest) is pending:
                        del self._pending[digest]
                        pending.fail(error)
                raise

            return parts

        try:
            parts = build()
            size = _parts_size(parts)
        except BaseException as error:
            with self._lock:
                if self._pending.get(digest) is pending:
                    del self._pending[digest]
                pending.fail(error)
            raise

        with self._lock:
            if self._pending.get(digest) is pending:
                del self._pending[digest]
                if pending.expires_at <= self._clock():
                    pending.fail(
                        TimeoutError(
                            "Timed out waiting for reference examples to render"
                        )
                    )
                else:
                    try:
                        self._keep(digest, parts=parts, size=size)
                    except BaseException as error:
                        pending.fail(error)
                        raise
                    pending.finish(parts)
        return parts

    def clear(self) -> None:
        """Drop every entry; builds in progress still finish and are kept."""
        with self._lock:
            self._entries.clear()
            self._size = 0

    def _drop_expired(self) -> None:
        now = self._clock()
        expired = [
            digest for digest, entry in self._entries.items() if entry.expires_at <= now
        ]
        for digest in expired:
            self._size -= self._entries.pop(digest).size

        expired_pending = [
            digest
            for digest, pending in self._pending.items()
            if pending.expires_at <= now
        ]
        for digest in expired_pending:
            pending = self._pending.pop(digest)
            pending.fail(
                TimeoutError("Timed out waiting for reference examples to render")
            )

    def _keep(self, digest: str, *, parts: List[dict], size: int) -> None:
        if size > self._max_bytes:
            return

        expires_at = self._clock() + self._ttl_seconds
        self._entries[digest] = _CachedParts(
            parts=parts, size=size, expires_at=expires_at
        )
        self._size += size
        while len(self._entries) > self._max_entries or self._size > self._max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._size -= evicted.size


# Every instance of the Gemini and OpenAI blocks in the process uses this cache.
SHARED_REFERENCE_PARTS_CACHE = ReferencePartsCache()


def validate_reference_examples(
    value: Optional[Sequence[Any]],
    *,
    task_type: str,
    labels: Optional[Sequence[str]],
) -> List[ReferenceExample]:
    """Validate reference examples against the task and the labels of the block.

    Args:
        value: Manifest literal or resolved selector value; ``None`` and an empty
            list mean no examples.
        task_type: Task the block runs; examples are only supported for
            object-detection and oriented-object-detection.
        labels: Labels the block returns, or ``None`` when they are only known at
            runtime (a selector).

    Returns:
        Parsed examples, empty when there are none.

    Raises:
        ValueError: If the task is not supported, an example is malformed, there are
            too many, an annotation uses a label the block does not have, or an
            annotation of an oriented-object-detection example has no ``obb``.
    """
    if not value:
        return []

    if task_type not in REFERENCE_EXAMPLES_TASKS:
        raise ValueError(
            "`reference_examples` are only supported for the "
            f"{' and '.join(REFERENCE_EXAMPLES_TASKS)} tasks"
        )

    examples = _parse_reference_examples(value)
    if task_type == ORIENTED_REFERENCE_EXAMPLES_TASK:
        for index, example in enumerate(examples):
            _require_oriented_boxes(example, index=index)

    if labels is not None:
        used_labels = {
            annotation.class_name
            for example in examples
            for annotation in example.annotations
        }
        unknown = sorted(used_labels - set(labels))
        if unknown:
            raise ValueError(
                f"`reference_examples` use labels that the block does not have: {unknown}"
            )

    return examples


def numbered_labels(
    examples: Sequence[ReferenceExample],
    *,
    labels: Sequence[str],
) -> Optional[Dict[str, str]]:
    """Pick numbers to write on the boxes in place of labels the font cannot draw.

    Pillow's default font has glyphs for ASCII only and draws any other character as
    the same box, so labels in other scripts would all look alike. When an example
    uses a label outside printable ASCII, every label is written as its 1-based
    position in ``labels`` instead, and the request text maps the numbers back to the
    labels.

    Args:
        examples: Validated examples.
        labels: Block labels.

    Returns:
        A number per label the examples use, in the order of ``labels``, or ``None``
        when all of those labels are printable ASCII and are written as they are.
    """
    used_labels = {
        annotation.class_name
        for example in examples
        for annotation in example.annotations
    }
    if all(label.isascii() and label.isprintable() for label in used_labels):
        return None

    numbers = {
        label: str(position)
        for position, label in enumerate(labels, start=1)
        if label in used_labels
    }
    return numbers


def load_reference_example(
    example: ReferenceExample,
    *,
    index: int,
    task_type: str = REFERENCE_EXAMPLES_TASK,
) -> LoadedReferenceExample:
    """Decode an example image and convert its boxes to clipped corners.

    Images load like a workflow image input of an HTTP request: URLs follow the host's
    policy and a local path is never read. The block cannot see the engine's
    ``prevent_local_images_loading``, and these values come straight from the workflow
    definition or its inputs.

    Args:
        example: Validated example.
        index: Position of the example in its list, for error messages.
        task_type: Task the example is for. The oriented task keeps each box rotated
            instead of clipping it.

    Returns:
        The decoded image with its boxes.

    Raises:
        RuntimeInputError: If the image cannot be loaded, which includes a local path.
        ValueError: If a box lies entirely outside its image, or an annotation of an
            oriented example has no ``obb``.
    """
    image = deserialize_image_kind(
        f"reference_examples[{index}].image",
        example.image,
        prevent_local_images_loading=True,
    )
    pixels, _ = load_image(image.to_inference_format(numpy_preferred=True))
    height, width = pixels.shape[:2]
    if task_type == ORIENTED_REFERENCE_EXAMPLES_TASK:
        oriented_boxes = _read_oriented_boxes(
            example, index=index, width=width, height=height
        )
        loaded = LoadedReferenceExample(
            image=pixels, boxes=(), oriented_boxes=oriented_boxes
        )
        return loaded

    boxes = []
    for box in example.annotations:
        x_min = max(0.0, box.x - box.width / 2)
        y_min = max(0.0, box.y - box.height / 2)
        x_max = min(float(width), box.x + box.width / 2)
        y_max = min(float(height), box.y + box.height / 2)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"`reference_examples[{index}]` has a box outside its "
                f"{width}x{height} image"
            )
        boxes.append((box.class_name, x_min, y_min, x_max, y_max))
    loaded = LoadedReferenceExample(image=pixels, boxes=tuple(boxes))
    return loaded


def render_boxed_examples(
    examples: Sequence[ReferenceExample],
    *,
    labels: Sequence[str],
    numbers: Optional[Dict[str, str]] = None,
    task_type: str = REFERENCE_EXAMPLES_TASK,
    class_heading: Optional[Mapping[str, bool]] = None,
) -> List[str]:
    """Load each example image in turn and draw its boxes and labels on it.

    An image is released once drawn, before the next one loads, so the examples
    never hold more than one full-size image.

    Args:
        examples: Validated examples.
        labels: Block labels; a label's position picks its box color.
        numbers: Numbers to write in place of the labels, from ``numbered_labels``.
        task_type: Task the examples are for; the oriented task draws rotated boxes.
        class_heading: Which labels have a front, for the oriented task: a triangle
            marks the front edge of the boxes of those labels.

    Returns:
        One base64 JPEG per example, downscaled to ``BOXED_IMAGE_MAX_EDGE``.

    Raises:
        RuntimeInputError: If an image cannot be loaded.
        ValueError: If a box lies entirely outside its image.
    """
    label_positions = {label: index for index, label in enumerate(labels)}
    tags = {label: (numbers or {}).get(label, label) for label in labels}
    front_labels = _front_labels(class_heading)
    # Each loaded example goes straight into the drawing, so nothing keeps its image
    # once the JPEG is made.
    rendered = [
        _draw_boxes(
            load_reference_example(example, index=index, task_type=task_type),
            label_positions=label_positions,
            tags=tags,
            front_labels=front_labels,
        )
        for index, example in enumerate(examples)
    ]
    return rendered


def render_example_crops(
    examples: Sequence[ReferenceExample],
    *,
    labels: Sequence[str],
    task_type: str = REFERENCE_EXAMPLES_TASK,
    class_heading: Optional[Mapping[str, bool]] = None,
) -> List[Tuple[str, List[str]]]:
    """Load each example image in turn and cut crops of its boxes, grouped by label.

    Boxes are considered largest first within each image, encoding at most
    ``MAX_CROPS_PER_LABEL`` crops per label per image. The image is released before
    the next one loads, so the examples never hold more than one full-size image.

    Args:
        examples: Validated examples.
        labels: Block labels; groups follow this order.
        task_type: Task the examples are for. The oriented task cuts each box along
            its rotation and turns the crop upright.
        class_heading: Which labels have a front, for the oriented task: the front
            of those labels is on the right of the crop, and the long side of the
            others is horizontal.

    Returns:
        ``(label, crops)`` pairs with up to ``MAX_CROPS_PER_LABEL`` base64 JPEG crops
        each, largest boxes first and equal ones in the order given. Labels without
        boxes are left out.

    Raises:
        RuntimeInputError: If an image cannot be loaded.
        ValueError: If a box lies entirely outside its image.
    """
    largest_crops = _LargestCrops(labels, front_labels=_front_labels(class_heading))
    for index, example in enumerate(examples):
        # Passed straight in, so nothing keeps the image once its crops are cut.
        largest_crops.add(
            load_reference_example(example, index=index, task_type=task_type)
        )

    groups = largest_crops.groups()
    return groups


def gemini_reference_parts_for(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
    task_type: str = REFERENCE_EXAMPLES_TASK,
    class_heading: Optional[Mapping[str, bool]] = None,
    cache: Optional[ReferencePartsCache] = None,
) -> List[dict]:
    """Build Gemini content parts for validated examples: boxed images.

    When ``numbered_labels`` picks numbers for the boxes, the text before the images
    maps them to the labels.

    Args:
        value: Examples accepted by ``validate_reference_examples``.
        labels: Block labels.
        task_type: Task the examples are for; the oriented task draws rotated boxes.
        class_heading: Which labels have a front, for the oriented task.
        cache: Keeps the parts for later calls, as ``SHARED_REFERENCE_PARTS_CACHE``
            does for the block; without one the examples load on every call.

    Returns:
        Content parts to place before the target image. A cache shares them with
        other callers, so they are not to be modified.
    """
    oriented = task_type == ORIENTED_REFERENCE_EXAMPLES_TASK

    def render(examples: List[ReferenceExample]) -> List[dict]:
        numbers = numbered_labels(examples, labels=labels)
        boxed_images = render_boxed_examples(
            examples,
            labels=labels,
            numbers=numbers,
            task_type=task_type,
            class_heading=class_heading,
        )
        rendered = _gemini_parts(
            boxed_images,
            numbers=numbers,
            oriented=oriented,
            front_marked=oriented and _uses_a_front(examples, class_heading),
        )
        return rendered

    parts = _reference_parts(
        value,
        labels=labels,
        rendering="gemini",
        task_type=task_type,
        class_heading=class_heading,
        cache=cache,
        render=render,
    )
    return parts


def openai_reference_parts_for(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
    task_type: str = REFERENCE_EXAMPLES_TASK,
    class_heading: Optional[Mapping[str, bool]] = None,
    cache: Optional[ReferencePartsCache] = None,
) -> List[dict]:
    """Build OpenAI Responses API content parts for validated examples: crops.

    Args:
        value: Examples accepted by ``validate_reference_examples``.
        labels: Labels the model returns.
        task_type: Task the examples are for; the oriented task turns each crop
            upright along its box.
        class_heading: Which labels have a front, for the oriented task.
        cache: Keeps the parts for later calls, as ``SHARED_REFERENCE_PARTS_CACHE``
            does for the block; without one the examples load on every call.

    Returns:
        Content parts to place before the target image. A cache shares them with
        other callers, so they are not to be modified.
    """
    parts = _reference_parts(
        value,
        labels=labels,
        rendering="openai",
        task_type=task_type,
        class_heading=class_heading,
        cache=cache,
        render=lambda examples: _openai_parts(
            render_example_crops(
                examples,
                labels=labels,
                task_type=task_type,
                class_heading=class_heading,
            ),
            oriented=task_type == ORIENTED_REFERENCE_EXAMPLES_TASK,
        ),
    )
    return parts


def with_reference_parts(
    parts: Sequence[dict],
    *,
    reference_parts: Sequence[dict],
    target_intro: dict,
    is_image_part: Callable[[dict], bool],
) -> List[dict]:
    """Put the references first and point the detection prompt at the target image.

    Args:
        parts: Content parts of the detection request without references.
        reference_parts: Parts built for the examples.
        target_intro: Text part placed right before the image to label.
        is_image_part: Tells the image part apart from text parts.

    Returns:
        New content parts. The last text part, which holds the detection
        instructions, has "in this image" pointed at the target image and ends with
        a sentence that rules out labeling the references.
    """
    text_indices = [index for index, part in enumerate(parts) if "text" in part]
    instructions_index = text_indices[-1] if text_indices else None
    combined = list(reference_parts)
    for index, part in enumerate(parts):
        if is_image_part(part):
            combined.append(target_intro)
            combined.append(part)
        elif index == instructions_index:
            text = part["text"].replace("in this image", "in the target image", 1)
            combined.append({**part, "text": text + TARGET_ONLY_RULE})
        else:
            combined.append(part)
    return combined


def _parse_reference_examples(value: Optional[Sequence[Any]]) -> List[ReferenceExample]:
    if not value:
        return []

    if not isinstance(value, (list, tuple)):
        raise ValueError("`reference_examples` must be a list of examples")
    if len(value) > MAX_REFERENCE_EXAMPLES:
        raise ValueError(
            f"`reference_examples` accepts at most {MAX_REFERENCE_EXAMPLES} examples, "
            f"got {len(value)}"
        )

    examples = []
    for index, item in enumerate(value):
        if isinstance(item, ReferenceExample):
            examples.append(item)
            continue
        try:
            examples.append(ReferenceExample.model_validate(item))
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
                for detail in error.errors()
            )
            raise ValueError(
                f"`reference_examples[{index}]` is not a valid example: {problems}"
            ) from None
    return examples


def _gemini_parts(
    boxed_images: Sequence[str],
    *,
    numbers: Optional[Dict[str, str]],
    oriented: bool = False,
    front_marked: bool = False,
) -> List[dict]:
    legend = ""
    if numbers:
        legend = "\n".join(
            f"{number}: {json.dumps(label, ensure_ascii=False)}"
            for label, number in numbers.items()
        )

    if oriented:
        marker = REFERENCE_FRONT_MARKER_RULE if front_marked else ""
        if numbers:
            intro = (
                f"{REFERENCE_ORIENTED_NUMBERED_INTRO}{marker}"
                f"{REFERENCE_NUMBERS_LEGEND_INTRO}\n{legend}"
            )
        else:
            intro = f"{REFERENCE_ORIENTED_BOXED_INTRO}{marker}"
    else:
        intro = (
            f"{REFERENCE_NUMBERED_INTRO}\n{legend}"
            if numbers
            else REFERENCE_BOXED_INTRO
        )

    parts = [{"text": intro}]
    for index, data in enumerate(boxed_images, start=1):
        parts.append({"text": f"Reference image {index}:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": data}})
    return parts


def _openai_parts(
    crop_groups: Sequence[Tuple[str, Sequence[str]]], *, oriented: bool = False
) -> List[dict]:
    intro = REFERENCE_ORIENTED_CROPS_INTRO if oriented else REFERENCE_CROPS_INTRO
    parts = [{"type": "input_text", "text": intro}]
    for label, crops in crop_groups:
        parts.append({"type": "input_text", "text": f'Examples of "{label}":'})
        parts.extend(
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{crop}"}
            for crop in crops
        )
    return parts


def _reference_parts(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
    rendering: str,
    cache: Optional[ReferencePartsCache],
    render: Callable[[List[ReferenceExample]], List[dict]],
    task_type: str = REFERENCE_EXAMPLES_TASK,
    class_heading: Optional[Mapping[str, bool]] = None,
) -> List[dict]:
    def build() -> List[dict]:
        built = render(_parse_reference_examples(value))
        return built

    if cache is None:
        parts = build()
        return parts

    key = _cache_key(
        value,
        labels=labels,
        rendering=rendering,
        task_type=task_type,
        class_heading=class_heading,
    )
    parts = cache.get_or_build(key, build=build)
    return parts


def _cache_key(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
    rendering: str,
    task_type: str = REFERENCE_EXAMPLES_TASK,
    class_heading: Optional[Mapping[str, bool]] = None,
) -> Optional[str]:
    items = []
    image_digests = []
    for item in value or []:
        if isinstance(item, ReferenceExample):
            item = item.model_dump(by_alias=True)

        digest = None
        if isinstance(item, dict) and "image" in item:
            image, digest = _image_cache_key(item["image"])
            if digest is not None:
                item = {**item, "image": image}

        items.append(item)
        image_digests.append(digest)

    key_parts = [rendering, items, list(labels)]
    if task_type == ORIENTED_REFERENCE_EXAMPLES_TASK:
        # Which labels have a front changes how the boxes are rendered.
        front_labels = _front_labels(class_heading)
        key_parts.append(
            {
                "task_type": task_type,
                "front": [label for label in labels if label in front_labels],
            }
        )
    if any(image_digests):
        # Keep digests outside caller-controlled values. A literal None or digest
        # in an image cannot impersonate a fingerprinted payload. Order matters.
        key_parts.append(image_digests)

    try:
        # Both blocks share the cache, and each renders the examples its own way.
        key = json.dumps(key_parts, sort_keys=True)
    except TypeError:
        # An in-memory image has no JSON form, and rendering it needs no fetch.
        return None

    return key


def _image_cache_key(image: Any) -> Tuple[Any, Optional[str]]:
    """Fingerprint large strings without copying them into a JSON document."""
    payload = image.get("value") if isinstance(image, dict) else image
    if not isinstance(payload, str) or len(payload) <= _IMAGE_KEY_CHUNK_SIZE:
        return image, None

    # Re-read every character on every lookup, with bounded temporary buffers.
    # No identity memoization or extra cache retains caller-owned image data.
    digest = hashlib.sha256()
    for offset in range(0, len(payload), _IMAGE_KEY_CHUNK_SIZE):
        chunk = payload[offset : offset + _IMAGE_KEY_CHUNK_SIZE]
        digest.update(chunk.encode("utf-8", errors="surrogatepass"))

    fingerprint = digest.hexdigest()
    metadata = {**image, "value": None} if isinstance(image, dict) else None
    return metadata, fingerprint


def _parts_size(value: Any) -> int:
    # Count retained storage without allocating an encoded copy of large images.
    # Summing repeated references conservatively overcounts shared strings.
    if isinstance(value, str):
        size = sys.getsizeof(value)
    elif isinstance(value, dict):
        size = sum(_parts_size(item) for item in value.values())
    elif isinstance(value, (list, tuple)):
        size = sum(_parts_size(item) for item in value)
    else:
        size = 0
    return size


class _LargestCrops:
    """Crops of the largest boxes of each label, cut while each example is loaded."""

    def __init__(
        self, labels: Sequence[str], *, front_labels: FrozenSet[str] = frozenset()
    ) -> None:
        self._labels = list(labels)
        self._front_labels = front_labels
        # (-area, arrival, crop) sorts the largest box first, then the first given.
        self._kept: Dict[str, List[Tuple[float, int, str]]] = {
            label: [] for label in labels
        }
        self._arrivals = 0

    def add(self, example: LoadedReferenceExample) -> None:
        candidates = []
        for label, x_min, y_min, x_max, y_max in example.boxes:
            box = (x_min, y_min, x_max, y_max)
            candidates.append(
                (label, _area(box), partial(_crop_base64, example.image, box=box))
            )
        for label, *values in example.oriented_boxes:
            oriented_box = tuple(values)
            cut = partial(
                _oriented_crop_base64,
                example.image,
                oriented_box=oriented_box,
                has_front=label in self._front_labels,
            )
            candidates.append((label, oriented_box[2] * oriented_box[3], cut))

        # Processing larger boxes first avoids replacing crops within this image.
        for label, area, cut in sorted(candidates, key=lambda item: -item[1]):
            rank = (-area, self._arrivals)
            self._arrivals += 1
            kept = self._kept[label]
            if len(kept) == MAX_CROPS_PER_LABEL:
                last = max(kept)
                if rank > last[:2]:
                    continue

                kept.remove(last)
            kept.append((*rank, cut()))

    def groups(self) -> List[Tuple[str, List[str]]]:
        groups = [
            (label, [crop for _, _, crop in sorted(self._kept[label])])
            for label in self._labels
            if self._kept[label]
        ]
        return groups


def _draw_boxes(
    example: LoadedReferenceExample,
    *,
    label_positions: Dict[str, int],
    tags: Dict[str, str],
    front_labels: FrozenSet[str] = frozenset(),
) -> str:
    height, width = example.image.shape[:2]
    scale = min(1.0, BOXED_IMAGE_MAX_EDGE / max(height, width))
    pixels = example.image
    if scale < 1.0:
        pixels = cv2.resize(
            pixels,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    canvas = Image.fromarray(np.ascontiguousarray(pixels[:, :, ::-1]))
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(max(canvas.size) / 400))
    font = _label_font(max(12, round(max(canvas.size) / 70)))

    for label, x_min, y_min, x_max, y_max in example.boxes:
        color = _BOX_COLORS[label_positions[label] % len(_BOX_COLORS)]
        box = [x_min * scale, y_min * scale, x_max * scale, y_max * scale]
        draw.rectangle(box, outline=color, width=line_width)
        _draw_tag(
            draw,
            tag=tags[label],
            font=font,
            color=color,
            left=box[0],
            anchor_top=box[1],
            line_width=line_width,
        )

    for label, *geometry in example.oriented_boxes:
        color = _BOX_COLORS[label_positions[label] % len(_BOX_COLORS)]
        corners = _oriented_corners(*geometry) * scale
        points = [(float(x), float(y)) for x, y in corners]
        draw.line([*points, points[0]], fill=color, width=line_width, joint="curve")
        if label in front_labels:
            _draw_front_marker(
                draw, corners, color=color, size=max(16, max(canvas.size) / 40)
            )
        left, anchor_top = min(points, key=lambda point: point[1])
        _draw_tag(
            draw,
            tag=tags[label],
            font=font,
            color=color,
            left=left,
            anchor_top=anchor_top,
            line_width=line_width,
        )

    bgr = np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1])
    encoded = _jpeg_base64(bgr)
    return encoded


def _draw_tag(
    draw: ImageDraw.ImageDraw,
    *,
    tag: str,
    font: ImageFont.ImageFont,
    color: Tuple[int, int, int],
    left: float,
    anchor_top: float,
    line_width: int,
) -> None:
    """Write ``tag`` on a filled label above ``anchor_top``, or inside when it is cut off."""
    text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), tag, font=font)
    tag_top = anchor_top - (text_bottom - text_top) - 4
    if tag_top < 0:
        tag_top = anchor_top + line_width
    draw.rectangle(
        [
            left,
            tag_top,
            left + text_right - text_left + 4,
            tag_top + text_bottom - text_top + 4,
        ],
        fill=color,
    )
    draw.text(
        (left + 2 - text_left, tag_top + 2 - text_top),
        tag,
        fill=_text_color(color),
        font=font,
    )


def _draw_front_marker(
    draw: ImageDraw.ImageDraw,
    corners: np.ndarray,
    *,
    color: Tuple[int, int, int],
    size: float,
) -> None:
    """Draw a filled triangle on the outside of the front edge of an oriented box.

    The front edge joins the second and third corners, the end of the heading axis.
    The triangle is as wide and as tall as ``size``, or as wide as the edge when the
    edge is shorter.
    """
    start, end = corners[1], corners[2]
    heading = start - corners[0]
    across = end - start
    heading_length = max(float(np.linalg.norm(heading)), 1e-9)
    across_length = max(float(np.linalg.norm(across)), 1e-9)
    base = min(size, across_length)
    middle = (start + end) / 2
    apex = middle + heading / heading_length * base
    base_start = middle + across / across_length * base / 2
    base_end = middle - across / across_length * base / 2
    draw.polygon([tuple(base_start), tuple(apex), tuple(base_end)], fill=color)


def _area(box: Tuple[float, float, float, float]) -> float:
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)


def _front_labels(class_heading: Optional[Mapping[str, bool]]) -> FrozenSet[str]:
    labels = frozenset(
        label for label, has_front in (class_heading or {}).items() if has_front
    )
    return labels


def _uses_a_front(
    examples: Sequence[ReferenceExample], class_heading: Optional[Mapping[str, bool]]
) -> bool:
    front_labels = _front_labels(class_heading)
    used = any(
        annotation.class_name in front_labels
        for example in examples
        for annotation in example.annotations
    )
    return used


def _oriented_box_bounds(value: Any) -> Optional[Tuple[float, float, float, float]]:
    """Center and size of the axis-aligned box that contains an oriented box.

    Returns ``None`` when ``value`` is not a valid ``[cx, cy, width, height, angle]``.
    """
    try:
        center_x, center_y, width, height, angle = (float(item) for item in value)
    except (TypeError, ValueError):
        return None

    values = (center_x, center_y, width, height, angle)
    if not all(math.isfinite(item) for item in values) or width <= 0 or height <= 0:
        return None

    cos_angle, sin_angle = abs(math.cos(angle)), abs(math.sin(angle))
    bounds = (
        center_x,
        center_y,
        width * cos_angle + height * sin_angle,
        width * sin_angle + height * cos_angle,
    )
    return bounds


def _oriented_corners(
    center_x: float, center_y: float, width: float, height: float, angle: float
) -> np.ndarray:
    """Ordered corners of an oriented box; the first edge runs along the heading."""
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    local = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    rotation = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])
    corners = local @ rotation + [center_x, center_y]
    return corners


def _lies_outside(corners: np.ndarray, *, width: int, height: int) -> bool:
    """Whether an oriented box shares no area with the image.

    Two convex shapes are apart when some side's normal separates them, so it is
    enough to compare their projections on the image's axes and the box's.
    """
    image_corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=float
    )
    axes = (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        corners[1] - corners[0],
        corners[3] - corners[0],
    )
    for axis in axes:
        on_box = corners @ axis
        on_image = image_corners @ axis
        if on_box.max() <= on_image.min() or on_image.max() <= on_box.min():
            return True

    return False


def _require_oriented_boxes(example: ReferenceExample, *, index: int) -> None:
    if any(annotation.obb is None for annotation in example.annotations):
        raise ValueError(
            f"`reference_examples[{index}]` needs an `obb` on every annotation for "
            f"the {ORIENTED_REFERENCE_EXAMPLES_TASK} task"
        )


def _read_oriented_boxes(
    example: ReferenceExample, *, index: int, width: int, height: int
) -> Tuple[Tuple[str, float, float, float, float, float], ...]:
    _require_oriented_boxes(example, index=index)
    boxes = []
    for annotation in example.annotations:
        center_x, center_y, box_width, box_height, angle = annotation.obb
        corners = _oriented_corners(center_x, center_y, box_width, box_height, angle)
        if _lies_outside(corners, width=width, height=height):
            raise ValueError(
                f"`reference_examples[{index}]` has a box outside its "
                f"{width}x{height} image"
            )
        boxes.append(
            (annotation.class_name, center_x, center_y, box_width, box_height, angle)
        )
    return tuple(boxes)


def _oriented_crop_base64(
    image: np.ndarray,
    *,
    oriented_box: Tuple[float, float, float, float, float],
    has_front: bool,
) -> str:
    """Cut an oriented box out of ``image`` with its margin, turned upright.

    The crop is horizontal along the heading with the front on the right. A box
    without a front is turned so that its long side is horizontal. Parts outside the
    image are filled with gray.
    """
    center_x, center_y, width, height, angle = oriented_box
    if not has_front and height > width:
        width, height, angle = height, width, angle + math.pi / 2
    patch_width = width + 2 * max(CROP_MIN_MARGIN, width * CROP_MARGIN)
    patch_height = height + 2 * max(CROP_MIN_MARGIN, height * CROP_MARGIN)
    longest = max(patch_width, patch_height)
    if longest > CROP_MAX_EDGE:
        scale = CROP_MAX_EDGE / longest
    elif longest < CROP_MIN_EDGE:
        scale = CROP_MIN_EDGE / longest
    else:
        scale = 1.0
    out_width = max(1, round(patch_width * scale))
    out_height = max(1, round(patch_height * scale))

    # The part of the image the patch covers, with the pixels the interpolation reads
    # around it, so a large box never allocates more than the image holds.
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    half_x = (abs(cos_angle) * patch_width + abs(sin_angle) * patch_height) / 2
    half_y = (abs(sin_angle) * patch_width + abs(cos_angle) * patch_height) / 2
    image_height, image_width = image.shape[:2]
    left = min(max(0, math.floor(center_x - half_x) - 2), image_width - 1)
    top = min(max(0, math.floor(center_y - half_y) - 2), image_height - 1)
    right = max(left + 1, min(image_width, math.ceil(center_x + half_x) + 2))
    bottom = max(top + 1, min(image_height, math.ceil(center_y + half_y) + 2))
    window = image[top:bottom, left:right]
    shrink = np.array([1.0, 1.0])
    if scale < 0.5:
        # Averaging before the warp keeps a big box from aliasing.
        window_height, window_width = window.shape[:2]
        size = (
            max(1, round(window_width * 2 * scale)),
            max(1, round(window_height * 2 * scale)),
        )
        shrink = np.array([size[0] / window_width, size[1] / window_height])
        window = cv2.resize(window, size, interpolation=cv2.INTER_AREA)

    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    linear = np.diag(shrink) @ rotation / scale
    # Pixel centers sit at half-integers of the continuous coordinates the box is
    # given in, and at integers in the arrays OpenCV samples.
    out_center = np.array([(out_width - 1) / 2, (out_height - 1) / 2])
    origin = np.array([center_x - left, center_y - top]) * shrink - 0.5
    matrix = np.hstack([linear, (origin - linear @ out_center)[:, None]])
    crop = cv2.warpAffine(
        window,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(CROP_FILL_VALUE,) * 3,
    )
    encoded = _jpeg_base64(np.ascontiguousarray(crop))
    return encoded


def _crop_base64(image: np.ndarray, *, box: Tuple[float, float, float, float]) -> str:
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = box
    margin_x = max(CROP_MIN_MARGIN, (x_max - x_min) * CROP_MARGIN)
    margin_y = max(CROP_MIN_MARGIN, (y_max - y_min) * CROP_MARGIN)
    left, top = max(0, int(x_min - margin_x)), max(0, int(y_min - margin_y))
    right = min(width, int(np.ceil(x_max + margin_x)))
    bottom = min(height, int(np.ceil(y_max + margin_y)))
    crop = image[top:bottom, left:right]

    longest = max(crop.shape[:2])
    if longest < CROP_MIN_EDGE:
        crop = _resize_longest_edge(
            crop, edge=CROP_MIN_EDGE, interpolation=cv2.INTER_CUBIC
        )
    elif longest > CROP_MAX_EDGE:
        crop = _resize_longest_edge(
            crop, edge=CROP_MAX_EDGE, interpolation=cv2.INTER_AREA
        )

    encoded = _jpeg_base64(np.ascontiguousarray(crop))
    return encoded


def _resize_longest_edge(
    image: np.ndarray, *, edge: int, interpolation: int
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = edge / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=interpolation)
    return resized


def _jpeg_base64(image: np.ndarray) -> str:
    encoded = base64.b64encode(encode_image_to_jpeg_bytes(image)).decode("ascii")
    return encoded


def _label_font(size: int) -> ImageFont.ImageFont:
    # The scalable default font needs FreeType; without it Pillow has only a
    # fixed-size bitmap font.
    try:
        font = ImageFont.load_default(size=size)
    except (TypeError, OSError):
        font = ImageFont.load_default()
    return font


def _text_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    red, green, blue = color
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)

"""Labeled example images for VLM object and keypoint detection ("label by example").

A reference example is an image with its annotations. Detection blocks send the
examples before the image to label, so the model sees what each label looks like
and how tightly it is boxed. There are two renderings because the providers read
them differently: whole images with the boxes drawn on them (Gemini) and crops of
the boxes grouped by label (OpenAI). For keypoint detection, both renderings also
draw each instance's skeleton, with every keypoint numbered by its position in the
class skeleton, and the text before the images maps the numbers to the keypoint
names. The examples go first in the request, so every image of a batch shares the
same prefix and the provider can cache it. The example images load and render one
at a time, so they hold one full-size image at most. The blocks keep the rendered
parts for the whole process in ``SHARED_REFERENCE_PARTS_CACHE``.
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
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from roboflow_workflows.core_steps.common.deserializers import deserialize_image_kind
from roboflow_workflows.core_steps.common.vlm_decoding.keypoint_detection import (
    KEYPOINT_DETECTION_TASK,
    KeypointSkeleton,
)
from roboflow_workflows.utils.images import encode_image_to_jpeg_bytes, load_image

REFERENCE_EXAMPLES_TASK = "object-detection"
REFERENCE_EXAMPLES_TASKS = (REFERENCE_EXAMPLES_TASK, KEYPOINT_DETECTION_TASK)
MAX_REFERENCE_EXAMPLES = 10
# Bounds the boxes drawn for one request, whatever the task.
MAX_REFERENCE_ANNOTATIONS = 5_000
# Bounds the skeleton drawing of one request: a keypoint is a dot and a number tag,
# an edge a line. The largest request allowed renders in a few seconds.
MAX_REFERENCE_DRAW_ITEMS = 20_000
MAX_CROPS_PER_LABEL = 3
BOXED_IMAGE_MAX_EDGE = 1024
CROP_MARGIN = 0.15
CROP_MIN_MARGIN = 8
CROP_MIN_EDGE = 128
CROP_MAX_EDGE = 384
# Keypoint crops are larger, so the number next to each keypoint stays legible.
KEYPOINT_CROP_MIN_EDGE = 256
KEYPOINT_CROP_MAX_EDGE = 512
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
TARGET_IMAGE_INTRO = "Target image:"
# On its own line, because a detection prompt can end with the list of labels.
TARGET_ONLY_RULE = (
    "\nThe reference images are not part of the task: return detections for the "
    "target image only."
)
REFERENCE_SKELETONS_INTRO = (
    "Reference examples: other images of the same dataset where every instance of "
    "the classes is outlined by a colored box with its class written on it."
)
REFERENCE_NUMBERED_SKELETONS_INTRO = (
    "Reference examples: other images of the same dataset where every instance of "
    "the classes is outlined by a colored box with its class number written on it."
)
REFERENCE_SKELETON_CROPS_INTRO = (
    "Reference examples: crops of annotated instances taken from other images of the "
    "same dataset, grouped by class. Each instance is outlined by a colored box."
)
REFERENCE_KEYPOINTS_LEGEND = (
    "Its keypoints are numbered dots joined by the edges of the class skeleton: a "
    "filled dot is a visible keypoint and a hollow ring an occluded one, whose "
    "position is estimated. A keypoint the instance does not show is not drawn. They "
    "show what each class looks like and where each of its keypoints goes."
)

REFERENCE_EXAMPLES_DESCRIPTION = (
    "Labeled example images that show the model what each class looks like and how "
    "tightly it is boxed (object-detection and keypoint-detection). Each example is "
    '`{"image": <URL, base64 or {"type": ..., "value": ...}>, "annotations": '
    '[{"class": ..., "x": ..., "y": ..., "width": ..., "height": ...}]}`, in the '
    "Roboflow prediction format: box centers and sizes in pixels of that example "
    "image, `class` one of the block's labels. For keypoint-detection, each "
    'annotation also lists its `keypoints`: `[{"class": <keypoint name>, "x": ..., '
    '"y": ..., "occluded": false}]`, in pixels of the example image, leaving out the '
    "keypoints the instance does not show; `class_id`, the keypoint's position in "
    "the class skeleton, picks between keypoints that share a name. Annotate every "
    "object of those labels in each example. Local file paths are not read. At most "
    f"{MAX_REFERENCE_EXAMPLES} examples, sent with every image, so each request costs "
    f"more input tokens. An example takes at most {MAX_REFERENCE_ANNOTATIONS} "
    "annotations, and the keypoints and skeleton edges of all the examples at most "
    f"{MAX_REFERENCE_DRAW_ITEMS} together. A selector can provide the whole list, "
    "but selectors inside an example are not resolved."
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


class ReferenceKeypoint(BaseModel):
    """One keypoint of a reference instance, as in a Roboflow keypoint prediction.

    ``class`` names the keypoint in its class skeleton. When the skeleton repeats a
    name, ``class_id`` picks the keypoint; without it, keypoints that share a name
    take that name's places in skeleton order.
    """

    model_config = ConfigDict(populate_by_name=True)

    x: float = Field(
        allow_inf_nan=False, description="Keypoint x, in pixels of the example image."
    )
    y: float = Field(
        allow_inf_nan=False, description="Keypoint y, in pixels of the example image."
    )
    class_name: str = Field(
        alias="class",
        min_length=1,
        description="Name of the keypoint in its class skeleton.",
        examples=["nose"],
    )
    class_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="Position of the keypoint in its class skeleton, from 0; picks "
        "between keypoints that share a name.",
        examples=[0],
    )
    occluded: bool = Field(
        default=False,
        description="Whether the keypoint is hidden, by another object or by the "
        "instance itself, and its position estimated.",
    )


class ReferenceAnnotation(BaseModel):
    """One annotation of a reference example, in the Roboflow prediction format.

    Object detection reads the class and the box, and keypoint detection also the
    keypoints. The same list can serve the other tasks, which read other fields of a
    prediction: `points` or `rle_mask` for instance segmentation, only `class` for
    classification. Fields a task does not read are ignored, so a whole prediction
    is a valid annotation.
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
    keypoints: Optional[List[ReferenceKeypoint]] = Field(
        default=None,
        description="Keypoints of the instance, which keypoint-detection examples "
        "need. A keypoint the instance does not show is left out.",
    )


class ReferenceExample(BaseModel):
    """An example image with every object of the block's labels annotated."""

    image: Union[str, Dict[str, Any]] = Field(
        description='Image as a URL, a base64 string or `{"type": ..., "value": ...}`.',
        examples=[{"type": "url", "value": "https://example.com/labeled.jpg"}],
    )
    annotations: List[ReferenceAnnotation] = Field(
        max_length=MAX_REFERENCE_ANNOTATIONS,
        description="Every object of the block's labels in the image.",
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


InstanceKeypoints = Tuple[Optional[Tuple[float, float, bool]], ...]


@dataclass(frozen=True)
class LoadedReferenceExample:
    """A decoded example image (BGR) with boxes as (label, x_min, y_min, x_max, y_max).

    For keypoint detection, ``keypoints`` holds one entry per box, in the same order:
    an ``(x, y, occluded)`` per keypoint of the class skeleton, ``None`` for a
    keypoint the instance does not show. Without skeletons it is empty.
    """

    image: np.ndarray
    boxes: Tuple[Tuple[str, float, float, float, float], ...]
    keypoints: Tuple[InstanceKeypoints, ...] = ()


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
    keypoint_skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
) -> List[ReferenceExample]:
    """Validate reference examples against the task and the labels of the block.

    Args:
        value: Manifest literal or resolved selector value; ``None`` and an empty
            list mean no examples.
        task_type: Task the block runs; examples are only supported for
            object-detection and keypoint-detection.
        labels: Labels the block returns, or ``None`` when they are only known at
            runtime (a selector).
        keypoint_skeletons: The skeleton of each class, for keypoint-detection, to
            check each instance's keypoints against; ``None`` when the skeletons
            are only known at runtime (a selector).

    Returns:
        Parsed examples, empty when there are none.

    Raises:
        ValueError: If the task takes no examples, an example is malformed, there
            are too many, an annotation uses a label the block does not have, or a
            keypoint-detection instance has no keypoints or one its class skeleton
            does not have, or the examples have more keypoints and edges than one
            request may draw.
    """
    if not value:
        return []

    if task_type not in REFERENCE_EXAMPLES_TASKS:
        raise ValueError(
            "`reference_examples` are only supported for the "
            f"{' and '.join(REFERENCE_EXAMPLES_TASKS)} tasks"
        )

    examples = _parse_reference_examples(value)
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

    if task_type == KEYPOINT_DETECTION_TASK:
        _check_reference_keypoints(examples, skeletons=keypoint_skeletons)

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
    keypoint_skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
) -> LoadedReferenceExample:
    """Decode an example image and convert its boxes to clipped corners.

    Images load like a workflow image input of an HTTP request: URLs follow the host's
    policy and a local path is never read. The block cannot see the engine's
    ``prevent_local_images_loading``, and these values come straight from the workflow
    definition or its inputs.

    Args:
        example: Validated example.
        index: Position of the example in its list, for error messages.
        keypoint_skeletons: The skeleton of each class, for keypoint detection. Each
            instance's keypoints are then placed in skeleton order, leaving out
            those outside the image.

    Returns:
        The decoded image with its boxes, and their keypoints with skeletons.

    Raises:
        RuntimeInputError: If the image cannot be loaded, which includes a local path.
        ValueError: If a box lies entirely outside its image, or an instance's
            keypoints do not fit its class skeleton.
    """
    image = deserialize_image_kind(
        f"reference_examples[{index}].image",
        example.image,
        prevent_local_images_loading=True,
    )
    pixels, _ = load_image(image.to_inference_format(numpy_preferred=True))
    height, width = pixels.shape[:2]
    boxes = []
    instance_keypoints = []
    for position, box in enumerate(example.annotations):
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
        if keypoint_skeletons is not None:
            slots = _instance_keypoint_slots(
                box,
                skeletons=keypoint_skeletons,
                where=f"`reference_examples[{index}].annotations[{position}]`",
            )
            instance_keypoints.append(
                tuple(
                    (
                        (keypoint.x, keypoint.y, keypoint.occluded)
                        if keypoint is not None
                        and 0 <= keypoint.x <= width
                        and 0 <= keypoint.y <= height
                        else None
                    )
                    for keypoint in slots
                )
            )
    loaded = LoadedReferenceExample(
        image=pixels, boxes=tuple(boxes), keypoints=tuple(instance_keypoints)
    )
    return loaded


def render_boxed_examples(
    examples: Sequence[ReferenceExample],
    *,
    labels: Sequence[str],
    numbers: Optional[Dict[str, str]] = None,
    keypoint_skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
) -> List[str]:
    """Load each example image in turn and draw its boxes and labels on it.

    An image is released once drawn, before the next one loads, so the examples
    never hold more than one full-size image.

    Args:
        examples: Validated examples.
        labels: Block labels; a label's position picks its box color.
        numbers: Numbers to write in place of the labels, from ``numbered_labels``.
            With skeletons, a box reads "class <number>", which keeps it apart
            from the keypoint numbers.
        keypoint_skeletons: The skeleton of each class, for keypoint detection; each
            instance's skeleton is drawn on top, its keypoints numbered from 1 in
            skeleton order.

    Returns:
        One base64 JPEG per example, downscaled to ``BOXED_IMAGE_MAX_EDGE``.

    Raises:
        RuntimeInputError: If an image cannot be loaded.
        ValueError: If a box lies entirely outside its image, or an instance's
            keypoints do not fit its class skeleton.
    """
    label_positions = {label: index for index, label in enumerate(labels)}
    numbers = numbers or {}
    number_tag = "{}" if keypoint_skeletons is None else "class {}"
    tags = {
        label: number_tag.format(numbers[label]) if label in numbers else label
        for label in labels
    }
    # Each loaded example goes straight into the drawing, so nothing keeps its image
    # once the JPEG is made.
    rendered = [
        _draw_boxes(
            load_reference_example(
                example, index=index, keypoint_skeletons=keypoint_skeletons
            ),
            label_positions=label_positions,
            tags=tags,
            skeletons=keypoint_skeletons,
        )
        for index, example in enumerate(examples)
    ]
    return rendered


def render_example_crops(
    examples: Sequence[ReferenceExample],
    *,
    labels: Sequence[str],
    keypoint_skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
) -> List[Tuple[str, List[str]]]:
    """Load each example image in turn and cut crops of its boxes, grouped by label.

    Boxes are considered largest first within each image, encoding at most
    ``MAX_CROPS_PER_LABEL`` crops per label per image. The image is released before
    the next one loads, so the examples never hold more than one full-size image.

    Args:
        examples: Validated examples.
        labels: Block labels; groups follow this order.
        keypoint_skeletons: The skeleton of each class, for keypoint detection. A
            crop then shows its instance's box and skeleton, its keypoints numbered
            from 1 in skeleton order, and its longest edge is between
            ``KEYPOINT_CROP_MIN_EDGE`` and ``KEYPOINT_CROP_MAX_EDGE``.

    Returns:
        ``(label, crops)`` pairs with up to ``MAX_CROPS_PER_LABEL`` base64 JPEG crops
        each, largest boxes first and equal ones in the order given. Labels without
        boxes are left out.

    Raises:
        RuntimeInputError: If an image cannot be loaded.
        ValueError: If a box lies entirely outside its image, or an instance's
            keypoints do not fit its class skeleton.
    """
    largest_crops = _LargestCrops(labels, skeletons=keypoint_skeletons)
    for index, example in enumerate(examples):
        # Passed straight in, so nothing keeps the image once its crops are cut.
        largest_crops.add(
            load_reference_example(
                example, index=index, keypoint_skeletons=keypoint_skeletons
            )
        )

    groups = largest_crops.groups()
    return groups


def gemini_reference_parts_for(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
    cache: Optional[ReferencePartsCache] = None,
    keypoint_skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
) -> List[dict]:
    """Build Gemini content parts for validated examples: boxed images.

    When ``numbered_labels`` picks numbers for the boxes, the text before the images
    maps them to the labels. With skeletons, each instance's skeleton is drawn too,
    and the text maps each class's keypoint numbers to the keypoint names.

    Args:
        value: Examples accepted by ``validate_reference_examples``.
        labels: Block labels.
        cache: Keeps the parts for later calls, as ``SHARED_REFERENCE_PARTS_CACHE``
            does for the block; without one the examples load on every call.
        keypoint_skeletons: The skeleton of each class, for keypoint detection.

    Returns:
        Content parts to place before the target image. A cache shares them with
        other callers, so they are not to be modified.
    """

    def render(examples: List[ReferenceExample]) -> List[dict]:
        numbers = numbered_labels(examples, labels=labels)
        boxed_images = render_boxed_examples(
            examples,
            labels=labels,
            numbers=numbers,
            keypoint_skeletons=keypoint_skeletons,
        )
        if keypoint_skeletons is None:
            rendered = _gemini_parts(boxed_images, numbers=numbers)
            return rendered

        intro = _keypoints_intro(
            (
                REFERENCE_NUMBERED_SKELETONS_INTRO
                if numbers
                else REFERENCE_SKELETONS_INTRO
            ),
            skeletons=keypoint_skeletons,
            labels=_used_labels(examples, labels=labels),
            numbers=numbers,
        )
        rendered = _gemini_parts(boxed_images, numbers=None, intro=intro)
        return rendered

    parts = _reference_parts(
        value,
        labels=labels,
        rendering="gemini" if keypoint_skeletons is None else "gemini-keypoints",
        cache=cache,
        render=render,
        skeletons=keypoint_skeletons,
    )
    return parts


def openai_reference_parts_for(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
    cache: Optional[ReferencePartsCache] = None,
    keypoint_skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
) -> List[dict]:
    """Build OpenAI Responses API content parts for validated examples: crops.

    With skeletons, each crop shows its instance's skeleton, and the text maps each
    class's keypoint numbers to the keypoint names.

    Args:
        value: Examples accepted by ``validate_reference_examples``.
        labels: Labels the model returns.
        cache: Keeps the parts for later calls, as ``SHARED_REFERENCE_PARTS_CACHE``
            does for the block; without one the examples load on every call.
        keypoint_skeletons: The skeleton of each class, for keypoint detection.

    Returns:
        Content parts to place before the target image. A cache shares them with
        other callers, so they are not to be modified.
    """

    def render(examples: List[ReferenceExample]) -> List[dict]:
        crop_groups = render_example_crops(
            examples, labels=labels, keypoint_skeletons=keypoint_skeletons
        )
        if keypoint_skeletons is None:
            rendered = _openai_parts(crop_groups)
            return rendered

        intro = _keypoints_intro(
            REFERENCE_SKELETON_CROPS_INTRO,
            skeletons=keypoint_skeletons,
            labels=[label for label, _ in crop_groups],
            numbers=None,
        )
        rendered = _openai_parts(crop_groups, intro=intro)
        return rendered

    parts = _reference_parts(
        value,
        labels=labels,
        rendering="openai" if keypoint_skeletons is None else "openai-keypoints",
        cache=cache,
        render=render,
        skeletons=keypoint_skeletons,
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


def _check_reference_keypoints(
    examples: Sequence[ReferenceExample],
    *,
    skeletons: Optional[Dict[str, KeypointSkeleton]],
) -> None:
    # Counted before any image loads, so a request too large to draw costs nothing.
    draw_items = 0
    edge_counts: Dict[str, int] = {}
    for index, example in enumerate(examples):
        for position, annotation in enumerate(example.annotations):
            where = f"`reference_examples[{index}].annotations[{position}]`"
            if annotation.keypoints is None:
                raise ValueError(
                    f"{where} has no `keypoints`, which keypoint-detection examples "
                    "need"
                )

            draw_items += len(annotation.keypoints)
            if skeletons is not None:
                _instance_keypoint_slots(annotation, skeletons=skeletons, where=where)
                label = annotation.class_name
                if label not in edge_counts:
                    edge_counts[label] = len(_unique_edges(skeletons[label]))
                draw_items += edge_counts[label]

    if draw_items > MAX_REFERENCE_DRAW_ITEMS:
        raise ValueError(
            f"`reference_examples` have {draw_items} keypoints and skeleton edges to "
            f"draw, more than the {MAX_REFERENCE_DRAW_ITEMS} allowed; pass fewer "
            "examples or fewer instances"
        )


def _instance_keypoint_slots(
    annotation: ReferenceAnnotation,
    *,
    skeletons: Dict[str, KeypointSkeleton],
    where: str,
) -> List[Optional[ReferenceKeypoint]]:
    skeleton = skeletons.get(annotation.class_name)
    if skeleton is None:
        raise ValueError(
            f"{where} is a {annotation.class_name!r}, which has no skeleton in "
            "`keypoint_schema`"
        )

    try:
        slots = _keypoint_slots(annotation.keypoints or [], skeleton=skeleton)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None

    return slots


def _unique_edges(skeleton: KeypointSkeleton) -> List[Tuple[int, int]]:
    """List each edge of the skeleton once, in the order given.

    An edge given twice, or once in each direction, draws the same line, and an edge
    from a keypoint to itself draws none.
    """
    edges = list(
        dict.fromkeys(
            (min(first, second), max(first, second))
            for first, second in skeleton.edges or []
            if first != second
        )
    )
    return edges


def _keypoint_slots(
    keypoints: Sequence[ReferenceKeypoint], *, skeleton: KeypointSkeleton
) -> List[Optional[ReferenceKeypoint]]:
    """Place each keypoint at its position in the skeleton.

    Keypoints with a ``class_id`` take that position first; each other keypoint then
    takes the first free position of its name.
    """
    names = skeleton.keypoints
    slots: List[Optional[ReferenceKeypoint]] = [None] * len(names)
    for keypoint in keypoints:
        if keypoint.class_id is None:
            continue

        if keypoint.class_id >= len(names) or names[keypoint.class_id] != (
            keypoint.class_name
        ):
            raise ValueError(
                f"keypoint {keypoint.class_name!r} has class_id {keypoint.class_id}, "
                "which is not its position in the skeleton"
            )
        if slots[keypoint.class_id] is not None:
            raise ValueError(
                f"keypoint {keypoint.class_name!r} (class_id {keypoint.class_id}) is "
                "given twice"
            )
        slots[keypoint.class_id] = keypoint

    for keypoint in keypoints:
        if keypoint.class_id is not None:
            continue

        free = [
            position
            for position, name in enumerate(names)
            if name == keypoint.class_name and slots[position] is None
        ]
        if not free:
            problem = (
                "is given more times than the skeleton has it"
                if keypoint.class_name in names
                else "is not a keypoint of the skeleton"
            )
            raise ValueError(f"keypoint {keypoint.class_name!r} {problem}")
        slots[free[0]] = keypoint

    return slots


def _used_labels(
    examples: Sequence[ReferenceExample], *, labels: Sequence[str]
) -> List[str]:
    used = {
        annotation.class_name
        for example in examples
        for annotation in example.annotations
    }
    ordered = [label for label in labels if label in used]
    return ordered


def _keypoints_intro(
    intro: str,
    *,
    skeletons: Dict[str, KeypointSkeleton],
    labels: Sequence[str],
    numbers: Optional[Dict[str, str]],
) -> str:
    lines = [f"{intro} {REFERENCE_KEYPOINTS_LEGEND}"]
    if numbers:
        classes = ", ".join(
            f"class {number} is {json.dumps(label, ensure_ascii=False)}"
            for label, number in numbers.items()
        )
        lines.append(f"Class numbers: {classes}.")

    for label in labels:
        keypoints = ", ".join(
            f"{number} {key}"
            for number, key in enumerate(skeletons[label].answer_keys, start=1)
        )
        lines.append(
            f"Keypoints of {json.dumps(label, ensure_ascii=False)} by number: "
            f"{keypoints}."
        )

    text = "\n".join(lines)
    return text


def _gemini_parts(
    boxed_images: Sequence[str],
    *,
    numbers: Optional[Dict[str, str]],
    intro: str = REFERENCE_BOXED_INTRO,
) -> List[dict]:
    if numbers:
        legend = "\n".join(
            f"{number}: {json.dumps(label, ensure_ascii=False)}"
            for label, number in numbers.items()
        )
        intro = f"{REFERENCE_NUMBERED_INTRO}\n{legend}"

    parts = [{"text": intro}]
    for index, data in enumerate(boxed_images, start=1):
        parts.append({"text": f"Reference image {index}:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": data}})
    return parts


def _openai_parts(
    crop_groups: Sequence[Tuple[str, Sequence[str]]],
    *,
    intro: str = REFERENCE_CROPS_INTRO,
) -> List[dict]:
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
    skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
) -> List[dict]:
    def build() -> List[dict]:
        built = render(_parse_reference_examples(value))
        return built

    if cache is None:
        parts = build()
        return parts

    key = _cache_key(value, labels=labels, rendering=rendering, skeletons=skeletons)
    parts = cache.get_or_build(key, build=build)
    return parts


def _cache_key(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
    rendering: str,
    skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
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
    if skeletons is not None:
        # The legend and the drawn numbers follow the skeletons.
        key_parts.append(
            {
                label: skeleton.model_dump(mode="json")
                for label, skeleton in skeletons.items()
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
        self,
        labels: Sequence[str],
        *,
        skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
    ) -> None:
        self._labels = list(labels)
        self._label_positions = {label: index for index, label in enumerate(labels)}
        self._skeletons = skeletons
        # Listed once per skeleton, not for each crop.
        self._edges = {
            label: _unique_edges(skeleton)
            for label, skeleton in (skeletons or {}).items()
        }
        # (-area, arrival, crop) sorts the largest box first, then the first given.
        self._kept: Dict[str, List[Tuple[float, int, str]]] = {
            label: [] for label in labels
        }
        self._arrivals = 0

    def add(self, example: LoadedReferenceExample) -> None:
        instances = list(
            zip(example.boxes, example.keypoints or [None] * len(example.boxes))
        )
        # Processing larger boxes first avoids replacing crops within this image.
        ordered_instances = sorted(
            instances, key=lambda instance: -_area(instance[0][1:])
        )
        for (label, x_min, y_min, x_max, y_max), points in ordered_instances:
            box = (x_min, y_min, x_max, y_max)
            rank = (-_area(box), self._arrivals)
            self._arrivals += 1
            kept = self._kept[label]
            if len(kept) == MAX_CROPS_PER_LABEL:
                last = max(kept)
                if rank > last[:2]:
                    continue

                kept.remove(last)
            kept.append(
                (*rank, self._crop(example.image, box=box, label=label, points=points))
            )

    def _crop(
        self,
        image: np.ndarray,
        *,
        box: Tuple[float, float, float, float],
        label: str,
        points: Optional[InstanceKeypoints],
    ) -> str:
        if self._skeletons is None or points is None:
            crop = _crop_base64(image, box=box)
            return crop

        crop = _skeleton_crop_base64(
            image,
            box=box,
            points=points,
            edges=self._edges[label],
            color=_BOX_COLORS[self._label_positions[label] % len(_BOX_COLORS)],
        )
        return crop

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
    skeletons: Optional[Dict[str, KeypointSkeleton]] = None,
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
        tag = tags[label]
        box = [x_min * scale, y_min * scale, x_max * scale, y_max * scale]
        draw.rectangle(box, outline=color, width=line_width)
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        tag_top = box[1] - (bottom - top) - 4
        if tag_top < 0:
            tag_top = box[1] + line_width
        draw.rectangle(
            [
                box[0],
                tag_top,
                box[0] + right - left + 4,
                tag_top + bottom - top + 4,
            ],
            fill=color,
        )
        draw.text(
            (box[0] + 2 - left, tag_top + 2 - top),
            tag,
            fill=_text_color(color),
            font=font,
        )

    if skeletons is not None and example.keypoints:
        # Drawn last, so no box or tag hides a keypoint.
        longest = max(canvas.size)
        number_font = _label_font(max(10, round(longest / 80)))
        # Listed once per skeleton, not for each instance.
        edges_by_label = {
            label: _unique_edges(skeletons[label])
            for label in {label for label, *_ in example.boxes}
        }
        for (label, *_), points in zip(example.boxes, example.keypoints):
            _draw_skeleton(
                draw,
                points=[
                    (
                        None
                        if point is None
                        else (point[0] * scale, point[1] * scale, point[2])
                    )
                    for point in points
                ],
                edges=edges_by_label[label],
                color=_BOX_COLORS[label_positions[label] % len(_BOX_COLORS)],
                radius=max(3, round(longest / 256)),
                line_width=max(1, round(longest / 512)),
                font=number_font,
            )

    bgr = np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1])
    encoded = _jpeg_base64(bgr)
    return encoded


def _draw_skeleton(
    draw: ImageDraw.ImageDraw,
    *,
    points: Sequence[Optional[Tuple[float, float, bool]]],
    edges: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
    radius: int,
    line_width: int,
    font: ImageFont.ImageFont,
) -> None:
    """Draw an instance's edges, then its keypoints numbered from 1 in skeleton order.

    A visible keypoint is a filled dot and an occluded one a hollow ring. Edges stop
    at the circles, so a ring stays hollow. A keypoint the instance does not show,
    and every edge that ends at one, is left out.
    """
    for first, second in edges:
        start, end = points[first], points[second]
        if start is None or end is None:
            continue

        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= 2 * radius:
            continue

        step_x = (end[0] - start[0]) * radius / length
        step_y = (end[1] - start[1]) * radius / length
        draw.line(
            [
                (start[0] + step_x, start[1] + step_y),
                (end[0] - step_x, end[1] - step_y),
            ],
            fill=color,
            width=line_width,
        )

    for number, point in enumerate(points, start=1):
        if point is None:
            continue

        x, y, occluded = point
        circle = [x - radius, y - radius, x + radius, y + radius]
        if occluded:
            draw.ellipse(circle, outline=color, width=2)
        else:
            draw.ellipse(circle, fill=color, outline=(255, 255, 255), width=1)
        _draw_outlined_text(draw, (x + radius + 1, y - radius), str(number), font=font)


def _draw_outlined_text(
    draw: ImageDraw.ImageDraw,
    position: Tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
) -> None:
    # A black outline drawn by hand keeps white text legible on any background and
    # works with the bitmap font too, which has no stroke support.
    x, y = position
    left, top, _, bottom = draw.textbbox((0, 0), text, font=font)
    origin_x, origin_y = x - left, y - top - (bottom - top) / 2
    for offset_x, offset_y in (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ):
        draw.text(
            (origin_x + offset_x, origin_y + offset_y), text, fill=(0, 0, 0), font=font
        )
    draw.text((origin_x, origin_y), text, fill=(255, 255, 255), font=font)


def _area(box: Tuple[float, float, float, float]) -> float:
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)


def _crop_region(
    image: np.ndarray, *, box: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = box
    margin_x = max(CROP_MIN_MARGIN, (x_max - x_min) * CROP_MARGIN)
    margin_y = max(CROP_MIN_MARGIN, (y_max - y_min) * CROP_MARGIN)
    left, top = max(0, int(x_min - margin_x)), max(0, int(y_min - margin_y))
    right = min(width, int(np.ceil(x_max + margin_x)))
    bottom = min(height, int(np.ceil(y_max + margin_y)))
    crop = image[top:bottom, left:right]
    return crop, left, top


def _skeleton_crop_base64(
    image: np.ndarray,
    *,
    box: Tuple[float, float, float, float],
    points: InstanceKeypoints,
    edges: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> str:
    crop, left, top = _crop_region(image, box=box)
    longest = max(crop.shape[:2])
    resized = crop
    if longest < KEYPOINT_CROP_MIN_EDGE:
        resized = _resize_longest_edge(
            crop, edge=KEYPOINT_CROP_MIN_EDGE, interpolation=cv2.INTER_CUBIC
        )
    elif longest > KEYPOINT_CROP_MAX_EDGE:
        resized = _resize_longest_edge(
            crop, edge=KEYPOINT_CROP_MAX_EDGE, interpolation=cv2.INTER_AREA
        )
    scale_x = resized.shape[1] / crop.shape[1]
    scale_y = resized.shape[0] / crop.shape[0]

    canvas = Image.fromarray(np.ascontiguousarray(resized[:, :, ::-1]))
    draw = ImageDraw.Draw(canvas)
    edge = max(canvas.size)
    line_width = max(1, round(edge / 256))
    x_min, y_min, x_max, y_max = box
    draw.rectangle(
        [
            (x_min - left) * scale_x,
            (y_min - top) * scale_y,
            (x_max - left) * scale_x,
            (y_max - top) * scale_y,
        ],
        outline=color,
        width=line_width,
    )
    _draw_skeleton(
        draw,
        points=[
            (
                None
                if point is None
                else ((point[0] - left) * scale_x, (point[1] - top) * scale_y, point[2])
            )
            for point in points
        ],
        edges=edges,
        color=color,
        radius=max(3, round(edge / 128)),
        line_width=line_width,
        font=_label_font(max(10, round(edge / 40))),
    )

    bgr = np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1])
    encoded = _jpeg_base64(bgr)
    return encoded


def _crop_base64(image: np.ndarray, *, box: Tuple[float, float, float, float]) -> str:
    crop, _, _ = _crop_region(image, box=box)

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

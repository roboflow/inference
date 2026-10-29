"""Labeled example images for VLM object detection ("label by example").

A reference example is an image with its objects boxed and labeled. Detection
blocks send the examples before the image to label, so the model sees what each
label looks like and how tightly it is boxed. There are two renderings because
the providers read them differently: whole images with the boxes drawn on them
(Gemini) and crops of the boxes grouped by label (OpenAI). The examples go first
in the request, so every image of a batch shares the same prefix and the
provider can cache it.
"""

import base64
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from roboflow_workflows.utils.images import encode_image_to_jpeg_bytes, load_image

REFERENCE_EXAMPLES_TASK = "object-detection"
MAX_REFERENCE_EXAMPLES = 10
MAX_CROPS_PER_LABEL = 3
BOXED_IMAGE_MAX_EDGE = 1024
CROP_MARGIN = 0.15
CROP_MIN_MARGIN = 8
CROP_MIN_EDGE = 128
CROP_MAX_EDGE = 384

REFERENCE_CROPS_INTRO = (
    "Reference examples: crops of labeled objects taken from other images of the "
    "same dataset, grouped by label. They show what each label looks like."
)
REFERENCE_BOXED_INTRO = (
    "Reference examples: other images of the same dataset where every object with "
    "one of the labels is outlined by a colored box with its label written on it. "
    "They show what each label looks like and how tightly it is boxed."
)
TARGET_IMAGE_INTRO = "Target image:"
TARGET_ONLY_RULE = (
    " The reference images are not part of the task: return detections for the "
    "target image only."
)

REFERENCE_EXAMPLES_DESCRIPTION = (
    "Labeled example images that show the model what each class looks like and how "
    "tightly it is boxed (object-detection only). Each example is "
    '`{"image": <URL, base64 or {"type": ..., "value": ...}>, "boxes": '
    '[{"x": ..., "y": ..., "width": ..., "height": ..., "class": ...}]}`, with box '
    "centers and sizes in pixels of that example image and `class` one of the "
    "block's labels. Box every object of those labels in each example. At most "
    f"{MAX_REFERENCE_EXAMPLES} examples. They are sent with every image, so each "
    "request costs more input tokens."
)
REFERENCE_EXAMPLES_EXAMPLE = [
    {
        "image": {"type": "url", "value": "https://example.com/labeled.jpg"},
        "boxes": [{"x": 120, "y": 80, "width": 40, "height": 60, "class": "cat"}],
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


class ReferenceBox(BaseModel):
    """One labeled box of a reference example, in the Roboflow prediction format."""

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


class ReferenceExample(BaseModel):
    """An example image with every object of the block's labels boxed."""

    image: Union[str, Dict[str, Any]] = Field(
        description="Image as a URL, a base64 string or an inference image dict.",
        examples=[{"type": "url", "value": "https://example.com/labeled.jpg"}],
    )
    boxes: List[ReferenceBox] = Field(
        description="Every object of the block's labels in the image."
    )


@dataclass(frozen=True)
class LoadedReferenceExample:
    """A decoded example image (BGR) with boxes as (label, x_min, y_min, x_max, y_max)."""

    image: np.ndarray
    boxes: Tuple[Tuple[str, float, float, float, float], ...]


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
            object-detection.
        labels: Labels the block returns, or ``None`` when they are only known at
            runtime (a selector).

    Returns:
        Parsed examples, empty when there are none.

    Raises:
        ValueError: If an example is malformed, there are too many, the task is not
            object-detection, or a box uses a label the block does not have.
    """
    examples = _parse_reference_examples(value)
    if not examples:
        return []

    if task_type != REFERENCE_EXAMPLES_TASK:
        raise ValueError(
            f"`reference_examples` are only supported for the {REFERENCE_EXAMPLES_TASK} task"
        )
    if labels is not None:
        used_labels = {box.class_name for example in examples for box in example.boxes}
        unknown = sorted(used_labels - set(labels))
        if unknown:
            raise ValueError(
                f"`reference_examples` use labels that the block does not have: {unknown}"
            )

    return examples


def load_reference_examples(
    examples: Sequence[ReferenceExample],
) -> List[LoadedReferenceExample]:
    """Decode the example images and convert their boxes to clipped corners.

    Images go through the installed image codec, so URLs follow the same policy as
    any other workflow image.

    Args:
        examples: Validated examples.

    Returns:
        One loaded example per input, in order.

    Raises:
        ValueError: If a box lies entirely outside its image.
    """
    loaded = []
    for index, example in enumerate(examples):
        pixels, _ = load_image(example.image)
        height, width = pixels.shape[:2]
        boxes = []
        for box in example.boxes:
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
        loaded.append(LoadedReferenceExample(image=pixels, boxes=tuple(boxes)))
    return loaded


def render_boxed_examples(
    examples: Sequence[LoadedReferenceExample],
    *,
    labels: Sequence[str],
) -> List[str]:
    """Draw every box and its label on each example image.

    Args:
        examples: Loaded examples.
        labels: Block labels; a label's position picks its box color.

    Returns:
        One base64 JPEG per example, downscaled to ``BOXED_IMAGE_MAX_EDGE``.
    """
    label_positions = {label: index for index, label in enumerate(labels)}
    rendered = []
    for example in examples:
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
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
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
                label,
                fill=_text_color(color),
                font=font,
            )

        bgr = np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1])
        rendered.append(_jpeg_base64(bgr))
    return rendered


def render_example_crops(
    examples: Sequence[LoadedReferenceExample],
    *,
    labels: Sequence[str],
) -> List[Tuple[str, List[str]]]:
    """Cut crops of the example boxes, grouped by label.

    Args:
        examples: Loaded examples.
        labels: Block labels; groups follow this order.

    Returns:
        ``(label, crops)`` pairs with up to ``MAX_CROPS_PER_LABEL`` base64 JPEG crops
        each, largest boxes first. Labels without boxes are left out.
    """
    by_label: Dict[str, List[Tuple[np.ndarray, Tuple[float, float, float, float]]]] = {
        label: [] for label in labels
    }
    for example in examples:
        for label, x_min, y_min, x_max, y_max in example.boxes:
            by_label[label].append((example.image, (x_min, y_min, x_max, y_max)))

    groups = []
    for label in labels:
        largest_first = sorted(by_label[label], key=lambda item: -_area(item[1]))
        chosen = largest_first[:MAX_CROPS_PER_LABEL]
        if chosen:
            groups.append(
                (label, [_crop_base64(image, box=box) for image, box in chosen])
            )
    return groups


def gemini_reference_parts_for(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
) -> List[dict]:
    """Build Gemini content parts for validated examples: boxed images.

    Results are cached, because a job runs the same examples on every image.

    Args:
        value: Examples accepted by ``validate_reference_examples``.
        labels: Block labels.

    Returns:
        Content parts to place before the target image.
    """
    parts = list(_cached_parts("gemini", _cache_key(value), tuple(labels)))
    return parts


def openai_reference_parts_for(
    value: Optional[Sequence[Any]],
    *,
    labels: Sequence[str],
) -> List[dict]:
    """Build OpenAI Responses API content parts for validated examples: crops.

    Results are cached, because a job runs the same examples on every image.

    Args:
        value: Examples accepted by ``validate_reference_examples``.
        labels: Labels the model returns.

    Returns:
        Content parts to place before the target image.
    """
    parts = list(_cached_parts("openai", _cache_key(value), tuple(labels)))
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


def _gemini_parts(boxed_images: Sequence[str]) -> List[dict]:
    parts = [{"text": REFERENCE_BOXED_INTRO}]
    for index, data in enumerate(boxed_images, start=1):
        parts.append({"text": f"Reference image {index}:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": data}})
    return parts


def _openai_parts(crop_groups: Sequence[Tuple[str, Sequence[str]]]) -> List[dict]:
    parts = [{"type": "input_text", "text": REFERENCE_CROPS_INTRO}]
    for label, crops in crop_groups:
        parts.append({"type": "input_text", "text": f'Examples of "{label}":'})
        parts.extend(
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{crop}"}
            for crop in crops
        )
    return parts


def _cache_key(value: Optional[Sequence[Any]]) -> str:
    items = [
        item.model_dump(by_alias=True) if isinstance(item, ReferenceExample) else item
        for item in value or []
    ]
    key = json.dumps(items, sort_keys=True)
    return key


# Without the cache every block run would fetch and render the same examples again.
# The key is the full examples JSON, so a hit needs the same images and boxes.
@lru_cache(maxsize=4)
def _cached_parts(
    provider: str, examples_json: str, labels: Tuple[str, ...]
) -> Tuple[dict, ...]:
    examples = load_reference_examples(
        _parse_reference_examples(json.loads(examples_json))
    )
    if provider == "gemini":
        parts = _gemini_parts(render_boxed_examples(examples, labels=labels))
    else:
        parts = _openai_parts(render_example_crops(examples, labels=labels))
    return tuple(parts)


def _area(box: Tuple[float, float, float, float]) -> float:
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)


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

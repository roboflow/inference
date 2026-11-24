"""A written guide to the classes of a VLM object-detection request.

A class guide says in words what each class looks like, how to tell it from the
other classes and from similar unlabeled things, and how its box is drawn. It can
stand in for labeled example images (see ``vlm_reference_examples``): Roboflow Auto
Label has a model write one from the examples once per job and sends the guide with
every image, which costs a few hundred tokens of text instead of about a thousand
per example image. The class names stay the labels the model answers with; the
guide goes after the detection instructions, under one intro line, with the scene
line first and one line per class.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CLASS_GUIDE_TASK = "object-detection"
CLASS_GUIDE_INTRO = "What the labels mean, from labeled examples of this dataset:"

CLASS_GUIDE_DESCRIPTION = (
    "A written guide to the classes (object-detection only): "
    '`{"scene": <what the images show>, "classes": {<class>: <guide>}}`. Each '
    "class guide says what the object looks like, how to tell it from the other "
    "classes and how its box is drawn; `scene` is optional. The text goes after the "
    "detection instructions of every request, and the class names stay the labels "
    "the model answers with. Every key of `classes` must be one of the block's "
    "classes; a class without a guide gets no line. Roboflow Auto Label writes the "
    "guide from labeled example images, so it can replace `reference_examples` at "
    "a fraction of their input tokens."
)
CLASS_GUIDE_EXAMPLE = {
    "scene": "Upright X-rays of one hand and wrist on a black background.",
    "classes": {
        "Radius": "The wider forearm bone, on the thumb side. Box its flared end "
        "and growth plate, from the carpal bones down to where the shaft narrows.",
    },
}


class ClassGuide(BaseModel):
    """What the images show and what each class means, as a model reads it."""

    model_config = ConfigDict(extra="forbid")

    scene: Optional[str] = Field(
        default=None,
        description="What the images show, and anything that holds for every class.",
        examples=[CLASS_GUIDE_EXAMPLE["scene"]],
    )
    classes: Dict[str, str] = Field(
        description="Guide per class: what the object looks like, how to tell it "
        "from the other classes and how its box is drawn.",
        examples=[CLASS_GUIDE_EXAMPLE["classes"]],
    )


def validate_class_guide(
    value: Any,
    *,
    task_type: str,
    labels: Optional[Sequence[str]],
) -> Optional[ClassGuide]:
    """Validate a class guide against the task and the labels of the block.

    Args:
        value: Manifest literal or resolved selector value; ``None`` means no guide.
        task_type: Task the block runs; a guide is only supported for
            object-detection.
        labels: Labels the block returns, or ``None`` when they are only known at
            runtime (a selector).

    Returns:
        The parsed guide, or ``None`` when there is none.

    Raises:
        ValueError: If the task is not object-detection, the guide is malformed, or
            it names a class the block does not have.
    """
    if value is None:
        return None

    if task_type != CLASS_GUIDE_TASK:
        raise ValueError(
            f"`class_guide` is only supported for the {CLASS_GUIDE_TASK} task"
        )

    try:
        guide = ClassGuide.model_validate(value)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in error.errors()
        )
        raise ValueError(f"`class_guide` is not a valid guide: {problems}") from None

    if labels is not None:
        unknown = sorted(set(guide.classes) - set(labels))
        if unknown:
            raise ValueError(
                f"`class_guide` describes classes that the block does not have: {unknown}"
            )

    return guide


def class_guide_text(guide: Optional[ClassGuide], *, labels: Sequence[str]) -> str:
    """The text a request appends to its detection instructions for the guide.

    Whitespace inside an entry, newlines included, collapses to single spaces, so
    each class keeps one line.

    Args:
        guide: Validated guide, or ``None``.
        labels: Block labels, which order the class lines.

    Returns:
        A blank line, the intro, the scene and one ``- <label>: <guide>`` line per
        class that has a guide; an empty string when the guide holds no text.
    """
    if guide is None:
        return ""

    lines = []
    scene = _one_line(guide.scene)
    if scene:
        lines.append(scene)
    for label in labels:
        text = _one_line(guide.classes.get(label))
        if text:
            lines.append(f"- {label}: {text}")
    if not lines:
        return ""

    text = "\n\n" + CLASS_GUIDE_INTRO + "\n" + "\n".join(lines)
    return text


def with_class_guide(
    parts: Sequence[dict],
    *,
    guide_text: str,
    is_text_part: Callable[[dict], bool] = lambda part: "text" in part,
) -> List[dict]:
    """Append the guide to the last text part, which holds the detection instructions.

    Args:
        parts: Content parts of the detection request, references included.
        guide_text: Text from ``class_guide_text``.
        is_text_part: Tells the text parts apart from image parts.

    Returns:
        New content parts; the same parts when there is no guide text.
    """
    combined = list(parts)
    if not guide_text:
        return combined

    text_indices = [index for index, part in enumerate(combined) if is_text_part(part)]
    if not text_indices:
        raise ValueError("The request has no text part to append the class guide to")
    last = text_indices[-1]
    combined[last] = {**combined[last], "text": combined[last]["text"] + guide_text}
    return combined


def _one_line(value: Optional[str]) -> str:
    return " ".join(value.split()) if value else ""

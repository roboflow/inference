"""Whole-image class labels: ``roboflow_core/classification_label_visualization@v1``.

Port of V1 ``roboflow_core/classification_label_visualization@v1``. Pixels
equal V1. Which classes are shown::

    ClassificationPrediction            the top class by confidence
    MultiLabelClassificationPrediction  every class in class_ids, by falling
                                        confidence

V1 lays the labels out as a stack of image-wide pseudo boxes and draws them
with ``sv.LabelAnnotator``; the same layout is reproduced here::

    spacing = 20 * text_scale + 2 * text_padding + 2 * text_thickness + 5
    TOP_*      box i = (0, spacing * (i + 1), W, H)
    BOTTOM_*   reversed order; box i = (0, 0, W, H - spacing * (i + 1))
    CENTER_*   rows centred vertically; CENTER_LEFT/RIGHT shift by an
               estimated label width (15 px per character at scale 1)

Colors: ``CLASS`` class id, ``INDEX`` position in the drawn order, ``TRACK``
palette color 0 (V1 gives every pseudo box tracker id 0). Host boundary: one
transfer of the confidences.
"""

from typing import Any, Dict, List, Literal, Tuple
import numpy as np
import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import (
    box_anchors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
    parse_color,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    check_literals,
    check_palette_literals,
    checked_choice,
    checked_int,
    checked_number,
    reject_copy_image,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    POSITIONS,
    TextStyle,
    detection_labels,
    paint_labels,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import CLASSIFICATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)


TEXT_OPTIONS = ("Class", "Confidence", "Class and Confidence")

_BASE_TEXT_HEIGHT = 20

_CHARACTER_WIDTH = 15

Shown = Tuple[int, str, float]

class ClassificationLabelVisualizationBlock(Block):
    """Write the predicted class (or classes) of the whole image onto a copy.

    Example: a single-label model predicting ``cat`` at 0.97 with
    ``text="Class and Confidence"`` writes ``cat 0.97`` in the top-left
    corner.
    """

    type = "roboflow_core/classification_label_visualization@v1"
    prunable = True
    metadata = {
        "name": "Classification Label Visualization",
        "port_of": "roboflow_core/classification_label_visualization@v1",
        "section": "visualization",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description=(
                "New annotated image with the input's identity, frame, source, "
                "time and parent row."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "RGB image to draw on; not modified. The output is a new image "
                "with the same identity, source and time."
            ),
            examples=["$inputs.image", "$steps.mask_visualization.image"],
        )
        color_palette: Literal[PALETTE_NAMES] | Ref(STRING_KIND) = Field(
            default="DEFAULT",
            description="Palette of the drawn elements.",
            examples=["DEFAULT", "ROBOFLOW", "Matplotlib Viridis", "$inputs.palette"],
        )
        palette_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Number of colors taken from a Matplotlib palette; >= 1.",
            examples=[10, "$inputs.palette_size"],
        )
        custom_colors: List[str] | Ref(LIST_OF_VALUES_KIND) = Field(
            default_factory=list,
            description=(
                "Colors of the CUSTOM palette: #RRGGBB, rgb(R, G, B), "
                "bgr(B, G, R) or a color name such as WHITE."
            ),
            examples=[["#FF0000", "#00FF00", "#0000FF"], "$inputs.custom_colors"],
        )
        color_axis: Literal[COLOR_AXES] | Ref(STRING_KIND) = Field(
            default="CLASS",
            description=(
                "How a row picks its palette color: CLASS (class id), INDEX (row "
                "position) or TRACK (bboxes_metadata tracker_id). Negative ids are "
                "grey."
            ),
            examples=["CLASS", "$inputs.color_axis"],
        )
        predictions: Ref(CLASSIFICATION_PREDICTION_KIND) = Field(
            description=(
                "Classification of this image (single- or multi-label), with "
                "image_metadata['class_names']."
            ),
            examples=["$steps.classification.predictions"],
        )
        text: Literal[TEXT_OPTIONS] | Ref(STRING_KIND) = Field(
            default="Class",
            description="What each label says.",
            examples=["Class", "Class and Confidence"],
        )
        text_position: Literal[POSITIONS] | Ref(STRING_KIND) = Field(
            default="TOP_LEFT",
            description="Where the labels stack in the image.",
            examples=["TOP_LEFT", "BOTTOM_RIGHT", "CENTER"],
        )
        text_color: str | Ref(STRING_KIND) = Field(
            default="WHITE",
            description="Text color: #RRGGBB, rgb(R, G, B), bgr(B, G, R) or a name.",
            examples=["WHITE", "#000000"],
        )
        text_scale: Number | Ref(FLOAT_KIND) = Field(
            default=1.0, description="Font scale; > 0.", examples=[1.0]
        )
        text_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1, description="Text stroke width; >= 1.", examples=[1]
        )
        text_padding: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10, description="Pixels around the text; >= 0.", examples=[10]
        )
        border_radius: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0, description="Label corner radius; >= 0.", examples=[0]
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_palette_literals(
            self,
        ) -> "ClassificationLabelVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

        @model_validator(mode="after")
        def _check_literals(self) -> "ClassificationLabelVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        color_palette: str,
        palette_size: int,
        custom_colors: Any,
        color_axis: str,
        text: str,
        text_position: str,
        text_color: str,
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        border_radius: int,
    ) -> Dict[str, Any]:
        """Write the predicted classes of ``predictions`` onto a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: Single- or multi-label classification.
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            text: ``Class``, ``Confidence`` or ``Class and Confidence``.
            text_position: Where the labels stack.
            text_color: Text color.
            text_scale: Font scale.
            text_thickness: Text stroke width.
            text_padding: Label padding.
            border_radius: Label corner radius.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters or a class id without a name.
        """
        values = dict(
            text=text,
            text_position=text_position,
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            border_radius=border_radius,
        )
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        shown = shown_classes(predictions)
        if not shown:
            result = annotated(image, canvas)
            return result

        height, width = image.size_hw
        boxes, ordered = _layout(
            shown,
            position=text_position,
            texts=_texts(shown, text),
            size_hw=(height, width),
            scale=text_scale,
            thickness=text_thickness,
            padding=text_padding,
        )
        class_ids = np.asarray([class_id for class_id, _, _ in ordered], dtype=np.int64)
        colors = row_colors(
            class_ids,
            np.zeros(len(ordered), dtype=np.int64),
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
        )
        labels = detection_labels(
            _texts(ordered, text),
            box_anchors(boxes, text_position),
            position=text_position,
            style=TextStyle(text_scale, text_thickness, text_padding, border_radius),
            background_rgb=[tuple(color) for color in colors.tolist()],
            text_rgb=parse_color(text_color, field="text_color"),
        )
        paint_labels(canvas, labels)
        result = annotated(image, canvas)

        return result

def shown_classes(predictions: Any) -> List[Shown]:
    """Return ``(class id, name, confidence)`` of the classes V1 shows, in order.

    Args:
        predictions: ``ClassificationPrediction`` or
            ``MultiLabelClassificationPrediction`` of one image.

    Returns:
        The top class for single-label; predicted classes by falling
        confidence (ties keep class id order) for multi-label.

    Raises:
        ValueError: On another type or a class id without a name.
    """
    if isinstance(predictions, ClassificationPrediction):
        confidence = predictions.confidence.reshape(-1).cpu().to(torch.float64).numpy()
        metadata = (predictions.images_metadata or [{}])[0] or {}
        if len(confidence) == 0:
            return []
        top = int(np.argmax(confidence))
        shown = [(top, _class_name(top, metadata), float(confidence[top]))]
        return shown
    if isinstance(predictions, MultiLabelClassificationPrediction):
        confidence = predictions.confidence.cpu().to(torch.float64).numpy()
        class_ids = predictions.class_ids.cpu().numpy().tolist()
        metadata = predictions.image_metadata or {}
        shown = [
            (
                int(class_id),
                _class_name(int(class_id), metadata),
                float(confidence[class_id]),
            )
            for class_id in class_ids
        ]
        shown.sort(key=lambda item: item[2], reverse=True)
        return shown

    raise ValueError(
        "expected ClassificationPrediction or MultiLabelClassificationPrediction, "
        f"got {type(predictions).__name__}"
    )

def _class_name(class_id: int, metadata: Dict[str, Any]) -> str:
    names = metadata.get("class_names")
    if names is not None and not isinstance(names, dict):
        names = dict(enumerate(names))
    if names is None or class_id not in names:
        raise ValueError(
            f"class id {class_id} has no name in image_metadata['class_names']"
        )

    return str(names[class_id])

def _texts(shown: List[Shown], text: str) -> List[str]:
    # V1 format_labels; V1 formatted float32 confidences.
    confidences = [f"{np.float32(confidence):.2f}" for _, _, confidence in shown]
    if text == "Class":
        return [name for _, name, _ in shown]
    if text == "Confidence":
        return confidences

    texts = [
        f"{name} {confidence}" for (_, name, _), confidence in zip(shown, confidences)
    ]

    return texts

def _layout(
    shown: List[Shown],
    *,
    position: str,
    texts: List[str],
    size_hw: Tuple[int, int],
    scale: float,
    thickness: int,
    padding: int,
) -> Tuple[np.ndarray, List[Shown]]:
    # V1 create_label_visualization: pseudo boxes whose anchors place labels.
    height, width = size_hw
    spacing = _BASE_TEXT_HEIGHT * scale + padding * 2 + thickness * 2 + 5
    count = len(shown)
    if position.startswith("BOTTOM"):
        ordered = shown[::-1]
        boxes = [
            [0, 0, width, height - (spacing + index * spacing)]
            for index in range(count)
        ]
    elif position.startswith("CENTER"):
        ordered = shown
        total = spacing * count
        start_y = max(0, min((height - total) / 2, height - total))
        label_width = (
            max(len(label) for label in texts) * _CHARACTER_WIDTH * scale + padding * 2
        )
        extra = 20 + max(0, 10 - padding) * 3
        left, right = 0, width
        if position == "CENTER_LEFT":
            left = label_width + extra
        elif position == "CENTER_RIGHT":
            right = width - (label_width + extra)
        boxes = [
            [left, start_y + index * spacing, right, start_y + (index + 1) * spacing]
            for index in range(count)
        ]
    else:
        ordered = shown
        boxes = [
            [0, spacing + index * spacing, width, height] for index in range(count)
        ]

    xyxy = np.array(boxes, dtype=np.float64)

    return xyxy, ordered

_CHECKS = {
    "text": lambda value: checked_choice(value, name="text", choices=TEXT_OPTIONS),
    "text_position": lambda value: checked_choice(
        value, name="text_position", choices=POSITIONS
    ),
    "text_color": lambda value: parse_color(value, field="text_color"),
    "text_scale": lambda value: checked_number(
        value, name="text_scale", minimum=0.0, above_minimum=True
    ),
    "text_thickness": lambda value: checked_int(
        value, name="text_thickness", minimum=1
    ),
    "text_padding": lambda value: checked_int(value, name="text_padding", minimum=0),
    "border_radius": lambda value: checked_int(value, name="border_radius", minimum=0),
}

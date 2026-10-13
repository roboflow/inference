"""Detection labels: ``v2/label_visualization@v2``.

Port of V1 ``roboflow_core/label_visualization@v2`` (``sv.LabelAnnotator``
with V1's label texts and adaptive text size). Pixels equal V1.

Text per row (``text``); "V2 producer" is the block that writes the key::

    Class                  row "class", else image_metadata class_names[class_id]
    Confidence             "0.93"
    Class and Confidence   "car 0.93" (confidence only when no row has a name)
    Index                  row position
    Dimensions             "cx, cy wxh" of the box, truncated
    Area                   mask pixel count for mask carriers, else box area
    Area (mask)            row "area_px"; no V2 producer yet ("Area (mask): N/A")
    Area (converted)       row "area_converted"; no V2 producer yet ("Area (conv): N/A")
    Tracker Id             row "tracker_id"; v2/trackers_* blocks ("No Tracker ID")
    Time In Zone           row "time_in_zone"; no V2 producer yet ("In zone: N/A")
    any other (selector)   that bboxes_metadata key; "" where a row lacks it

The text in brackets is drawn for each row without the key, as V1 does: a
batch may lack it on some rows. Masks are read only for ``Area`` and
``CENTER_OF_MASS`` on mask carriers, at ``mask_quality``. Text is drawn on the host in label-sized patches (``text``).
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import (
    CENTER_OF_MASS,
    box_anchors,
    mass_centres,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    parse_color,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    POSITIONS_WITH_MASS,
    BoxColorParams,
    MaskQuality,
    Number,
    check_literals,
    checked_choice,
    checked_int,
    checked_mask_quality,
    checked_number,
    mask_quality_field,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    DrawRows,
    boxes_of,
    dense_masks_of,
    has_masks,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    TextStyle,
    detection_labels,
    paint_labels,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY, row_class_name
from roboflow_workflows.v2.engine.declaration import Ref

TEXT_OPTIONS = (
    "Class",
    "Confidence",
    "Class and Confidence",
    "Index",
    "Dimensions",
    "Area",
    "Area (mask)",
    "Area (converted)",
    "Tracker Id",
    "Time In Zone",
)
TEXT_SIZE_MODES = ("Manual", "Automatic")
# V1 common/label_text.py: adaptive text scale.
_REFERENCE_MIN_DIMENSION_PX = 1080
_REFERENCE_TEXT_SCALE = 0.7
_TEXT_SCALE_LIMITS = (0.3, 5.0)


class LabelVisualizationBlock(AnnotationBlock):
    """Draw a text label at each detection onto a copy of the image.

    Full port of V1 roboflow_core/label_visualization@v2 with all its text
    options; not the limited prototype painter v2/label_visualization@v1.

    Example: ``text="Class and Confidence"`` writes ``"car 0.93"`` in white
    on the class color above each box's top-left corner.
    """

    type = "v2/label_visualization@v2"
    metadata = {
        "name": "Label Visualization",
        "port_of": "roboflow_core/label_visualization@v2",
        "section": "visualization",
    }

    class Params(BoxColorParams):
        text: Literal[TEXT_OPTIONS] | Ref(STRING_KIND) = Field(
            default="Class",
            description=(
                "What each label says. Class: class name. Confidence: e.g. 0.93. "
                "Class and Confidence: e.g. car 0.93. Index: row position. "
                "Dimensions: 'cx, cy wxh' of the box. Area: mask pixel count on "
                "instance masks (builds one full-image mask per detection, see "
                "mask_quality), box area otherwise. Area (mask), Area (converted) "
                "and Time In Zone read the area_px, area_converted and "
                "time_in_zone row keys, which no V2 block writes yet; rows "
                "without them show N/A. Tracker Id reads tracker_id from a "
                "tracker such as v2/trackers_bytetrack@v1; rows without it show "
                "'No Tracker ID'. A selector may also name any other "
                "bboxes_metadata key; rows without it get an empty label."
            ),
            examples=["Class", "Class and Confidence", "$inputs.label_text"],
        )
        text_position: Literal[POSITIONS_WITH_MASS] | Ref(STRING_KIND) = Field(
            default="TOP_LEFT",
            description=(
                "Anchor of the label on the box. CENTER_OF_MASS needs instance "
                "masks and builds one full-image mask per detection (see "
                "mask_quality)."
            ),
            examples=["TOP_LEFT", "CENTER", "$inputs.text_position"],
        )
        text_color: str | Ref(STRING_KIND) = Field(
            default="WHITE",
            description="Text color: #RRGGBB, rgb(R, G, B), bgr(B, G, R) or a name.",
            examples=["WHITE", "#000000", "$inputs.text_color"],
        )
        text_size_mode: Literal[TEXT_SIZE_MODES] | Ref(STRING_KIND) = Field(
            default="Manual",
            description=(
                "Manual uses text_scale as is; Automatic scales it with the image "
                "(text_scale 0.7 is the size at 1080 px)."
            ),
            examples=["Manual", "Automatic"],
        )
        text_scale: Number | Ref(FLOAT_KIND) = Field(
            default=1.0,
            description="Font scale; > 0.",
            examples=[1.0, "$inputs.text_scale"],
        )
        text_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1,
            description="Text stroke width; >= 1.",
            examples=[1, "$inputs.text_thickness"],
        )
        text_padding: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Pixels around the text inside its background; >= 0.",
            examples=[10, "$inputs.text_padding"],
        )
        border_radius: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description="Corner radius of the label background; >= 0.",
            examples=[0, 5],
        )
        mask_quality: MaskQuality = mask_quality_field(
            reads="text is Area or text_position is CENTER_OF_MASS, on instance masks"
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "LabelVisualizationBlock.Params":
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
        text_size_mode: str,
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        border_radius: int,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Draw one label per row of ``predictions`` onto a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``; label backgrounds.
            text: Text option or ``bboxes_metadata`` key.
            text_position: Anchor of each label.
            text_color: Text color.
            text_size_mode: ``Manual`` or ``Automatic``.
            text_scale: Font scale.
            text_thickness: Text stroke width.
            text_padding: Background padding.
            border_radius: Background corner radius.
            mask_quality: Mask quality for ``Area`` and ``CENTER_OF_MASS``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, a row without a class name for
                ``Class``, an unknown text key, ``CENTER_OF_MASS`` without
                masks, keypoints without boxes or predictions of another
                workflow input.
        """
        values = dict(
            text_position=text_position,
            text_color=text_color,
            text_size_mode=text_size_mode,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            border_radius=border_radius,
            mask_quality=mask_quality,
        )
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
        )
        rows = read_rows(boxes)
        if len(rows) == 0:
            result = annotated(image, canvas)
            return result

        needs_masks = text_position == CENTER_OF_MASS or text == "Area"
        masks = None
        if needs_masks and has_masks(predictions):
            dense = dense_masks_of(predictions, quality=mask_quality, block=self.type)
            masks = placed_in(dense, image, block=self.type).mask
        if text_position == CENTER_OF_MASS:
            if masks is None:
                raise ValueError(
                    f"{self.type}: text_position CENTER_OF_MASS needs instance "
                    "masks; these predictions have none"
                )
            anchors = mass_centres(masks)
        else:
            anchors = box_anchors(rows.xyxy, text_position)

        colors = row_colors(
            rows.class_id,
            rows.tracker_ids(),
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
        )
        style = TextStyle(
            scale=_effective_scale(
                text_scale, mode=text_size_mode, size_hw=image.size_hw
            ),
            thickness=text_thickness,
            padding=text_padding,
            border_radius=border_radius,
        )
        labels = detection_labels(
            label_texts(rows, text, masks=masks, block=self.type),
            anchors,
            position=text_position,
            style=style,
            background_rgb=[tuple(color) for color in colors.tolist()],
            text_rgb=parse_color(text_color, field="text_color"),
        )
        paint_labels(canvas, labels)
        result = annotated(image, canvas)

        return result


def label_texts(
    rows: DrawRows, text: str, *, masks: Optional[Any], block: str
) -> List[str]:
    """Return each row's label text for a V1 ``text`` option.

    Args:
        rows: Host rows in the image's pixels.
        text: One of ``TEXT_OPTIONS``, or a ``bboxes_metadata`` key.
        masks: ``(N, H, W)`` dense masks of the rows, when the carrier has
            masks (used by ``Area``).
        block: Block type for error messages.

    Returns:
        ``N`` texts.

    Raises:
        ValueError: When ``Class`` meets a row without a name, or no row has
            the key ``text``.
    """
    count = len(rows)
    metadata = rows.row_metadata
    if text == "Class":
        return rows.class_names()
    if text == "Confidence":
        return [f"{confidence:.2f}" for confidence in _confidences(rows)]
    if text == "Class and Confidence":
        confidences = [f"{confidence:.2f}" for confidence in _confidences(rows)]
        if not _has_any_name(rows):
            return confidences
        return [
            f"{name} {confidence}"
            for name, confidence in zip(rows.class_names(), confidences)
        ]
    if text == "Index":
        return [str(index) for index in range(count)]
    if text == "Dimensions":
        texts = []
        for x1, y1, x2, y2 in rows.xyxy:
            centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
            texts.append(
                f"{int(centre_x)}, {int(centre_y)} {int(x2 - x1)}x{int(y2 - y1)}"
            )
        return texts
    if text == "Area":
        if masks is not None:
            areas = masks.sum(dim=(1, 2)).cpu().numpy()
        else:
            xyxy = rows.xyxy
            areas = (xyxy[:, 3] - xyxy[:, 1]) * (xyxy[:, 2] - xyxy[:, 0])
        return [str(int(area)) for area in areas]
    if text == "Area (mask)":
        return _optional_numbers(metadata, key="area_px", prefix="Area (mask)")
    if text == "Area (converted)":
        return _optional_numbers(metadata, key="area_converted", prefix="Area (conv)")
    if text == "Tracker Id":
        return [
            (
                str(row["tracker_id"])
                if row.get("tracker_id") is not None
                else "No Tracker ID"
            )
            for row in metadata
        ]
    if text == "Time In Zone":
        texts = []
        for row in metadata:
            seconds = row.get("time_in_zone")
            texts.append(
                f"In zone: {round(seconds, 2)}s" if seconds else "In zone: N/A"
            )
        return texts

    if not any(text in row for row in metadata):
        raise ValueError(
            f"{block}: text {text!r} is neither one of {list(TEXT_OPTIONS)} nor a "
            "bboxes_metadata key of these predictions"
        )
    texts = [str(row[text]) if row.get(text) else "" for row in metadata]

    return texts


def _confidences(rows: DrawRows) -> np.ndarray:
    # V1 formatted float32 confidences; float32 -> float64 is exact.
    return rows.confidence.astype(np.float32)


def _has_any_name(rows: DrawRows) -> bool:
    has_name = CLASS_NAMES_KEY in rows.image_metadata or any(
        row_class_name(row) is not None for row in rows.row_metadata
    )

    return has_name


def _optional_numbers(metadata: List[dict], *, key: str, prefix: str) -> List[str]:
    texts = [
        f"{prefix}: {row[key]:.2f}" if row.get(key) is not None else f"{prefix}: N/A"
        for row in metadata
    ]

    return texts


def _effective_scale(scale: float, *, mode: str, size_hw: Any) -> float:
    # V1 compute_adaptive_label_text_scale.
    if mode != "Automatic":
        return scale

    resolution = min(size_hw) / _REFERENCE_MIN_DIMENSION_PX
    effective = _REFERENCE_TEXT_SCALE * resolution * (scale / _REFERENCE_TEXT_SCALE)
    clamped = max(_TEXT_SCALE_LIMITS[0], min(_TEXT_SCALE_LIMITS[1], effective))

    return clamped


_CHECKS = {
    "text_position": lambda value: checked_choice(
        value, name="text_position", choices=POSITIONS_WITH_MASS
    ),
    "text_color": lambda value: parse_color(value, field="text_color"),
    "text_size_mode": lambda value: checked_choice(
        value, name="text_size_mode", choices=TEXT_SIZE_MODES
    ),
    "text_scale": lambda value: checked_number(
        value, name="text_scale", minimum=0.0, above_minimum=True
    ),
    "text_thickness": lambda value: checked_int(
        value, name="text_thickness", minimum=1
    ),
    "text_padding": lambda value: checked_int(value, name="text_padding", minimum=0),
    "border_radius": lambda value: checked_int(value, name="border_radius", minimum=0),
    "mask_quality": checked_mask_quality,
}

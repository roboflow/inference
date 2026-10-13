"""Parameters shared by the annotation blocks, and their value checks.

Every literal may also be a selector. A literal is checked when the workflow
is compiled; a selected value is checked by the same function when the row
runs, so both fail with the same message::

    thickness: 0            -> compile error  "thickness must be an integer >= 1"
    thickness: $inputs.t    -> row error      same text, once t arrives

Each block declares its checks once, in a module ``_CHECKS`` dict of field
name -> check: its params validator passes it to ``check_literals``, and
``run`` applies every check to the received values.

``copy_image`` (V1) is rejected, whatever its value: these blocks always
return a new image and never change their input. To draw over an earlier
drawing, chain the blocks (``image: $steps.<previous>.image``).
"""

from typing import Any, List, Literal, Mapping, Optional, Sequence

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, ValidationError, model_validator
from pydantic_core import PydanticCustomError
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
    checked_palette_size,
    colors_of,
    palette_indices,
    parse_color,
    resolve_palette,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    MASK_QUALITIES,
)
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import BlockParams, Ref

POSITIONS_WITH_MASS = (
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
    "CENTER_OF_MASS",
)
_COPY_IMAGE_MESSAGE = (
    "copy_image is not a V2 parameter; delete the field from the step. V2 "
    "visualization blocks always return a new image and never change their "
    "input. To draw over an earlier drawing (V1 copy_image: false), chain the "
    "blocks: image: $steps.<previous>.image."
)
Number = StrictFloat | StrictInt


def is_selector(value: Any) -> bool:
    """Tell whether a parameter value is a selector string (``$inputs.x``)."""
    selector = isinstance(value, str) and value.startswith("$")

    return selector


def checked_int(value: Any, *, name: str, minimum: int) -> int:
    """Return ``value`` after checking it is an integer ``>= minimum``.

    Raises:
        ValueError: For booleans, non-integers and smaller values.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")

    return value


def checked_number(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: Optional[float] = None,
    above_minimum: bool = False,
) -> float:
    """Return ``value`` as a float after checking its range.

    Args:
        value: Candidate number.
        name: Parameter name for the error.
        minimum: Lower bound.
        maximum: Upper bound (inclusive), if any.
        above_minimum: Exclude ``minimum`` itself.

    Returns:
        The number as ``float``.

    Raises:
        ValueError: For booleans, non-numbers, NaN or values out of range.
    """
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    in_range = (
        is_number
        and value == value
        and (value > minimum if above_minimum else value >= minimum)
        and (maximum is None or value <= maximum)
    )
    if not in_range:
        bound = f"> {minimum}" if above_minimum else f">= {minimum}"
        if maximum is not None:
            bound = f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be a number {bound}, got {value!r}")

    number = float(value)

    return number


def checked_opacity(value: Any) -> float:
    """Return an opacity after checking it is a number in [0, 1]."""
    opacity = checked_number(value, name="opacity", minimum=0.0, maximum=1.0)

    return opacity


def checked_mask_quality(value: Any) -> str:
    """Return a ``mask_quality`` after checking it is ``accurate`` or ``coarse``."""
    quality = checked_choice(value, name="mask_quality", choices=MASK_QUALITIES)

    return quality


def checked_choice(value: Any, *, name: str, choices: Sequence[str]) -> str:
    """Return ``value`` after checking it is one of ``choices``.

    Raises:
        ValueError: Naming the allowed values.
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")

    return value


def check_literals(values: Mapping[str, Any], checks: Mapping[str, Any]) -> None:
    """Run each field's check on its literal value; skip selectors.

    Args:
        values: Field values of a validated ``BlockParams``.
        checks: Field name -> callable taking the value.
    """
    for name, check in checks.items():
        value = values.get(name)
        if value is not None and not is_selector(value):
            check(value)


class AnnotationParams(BlockParams):
    """``image`` and the rejection of V1 ``copy_image``."""

    image: Ref(IMAGE_KIND) = Field(
        description=(
            "RGB image to draw on; not modified. The output is a new image "
            "with the same identity, source and time."
        ),
        examples=["$inputs.image", "$steps.mask_visualization.image"],
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_copy_image(cls, values: Any) -> Any:
        # Located at the field, like other parameter errors, and saying why.
        if isinstance(values, Mapping) and "copy_image" in values:
            removed = PydanticCustomError("removed_parameter", _COPY_IMAGE_MESSAGE)
            raise ValidationError.from_exception_data(
                cls.__name__,
                [
                    {
                        "type": removed,
                        "loc": ("copy_image",),
                        "input": values["copy_image"],
                    }
                ],
            )

        return values


class PaletteParams(AnnotationParams):
    """Palette fields: V1 names and defaults."""

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
            "Colors of the CUSTOM palette: #RRGGBB, rgb(R, G, B), bgr(B, G, R) "
            "or a color name such as WHITE."
        ),
        examples=[["#FF0000", "#00FF00", "#0000FF"], "$inputs.custom_colors"],
    )

    @model_validator(mode="after")
    def _check_palette_literals(self) -> "PaletteParams":
        check_literals(
            {"palette_size": self.palette_size},
            {"palette_size": checked_palette_size},
        )
        if not is_selector(self.custom_colors):
            for index, color in enumerate(self.custom_colors):
                if not is_selector(color):
                    parse_color(color, field=f"custom_colors[{index}]")

        return self


class ColorParams(PaletteParams):
    """Palette fields plus how rows pick colors."""

    color_axis: Literal[COLOR_AXES] | Ref(STRING_KIND) = Field(
        default="CLASS",
        description=(
            "How a row picks its palette color: CLASS (class id), INDEX (row "
            "position) or TRACK (bboxes_metadata tracker_id). Negative ids are "
            "grey."
        ),
        examples=["CLASS", "$inputs.color_axis"],
    )


class BoxColorParams(ColorParams):
    """Colorable blocks that read boxes."""

    predictions: Ref(*BOX_KINDS) = Field(
        description=(
            "Detections, instance segmentation (dense, RLE or view) or keypoint "
            "predictions with boxes, of this image or of another frame of the "
            "same workflow input (for example a crop or its parent). Those are "
            "mapped into this image; parts outside it are clipped."
        ),
        examples=[
            "$steps.object_detection.predictions",
            "$steps.segmentation.segmentation",
        ],
    )


def mask_quality_field(*, reads: str) -> Any:
    """Return the ``mask_quality`` field declaration shared by mask readers.

    Args:
        reads: When this block reads masks, completing "Used when ...".

    Returns:
        The pydantic ``Field``.
    """
    field = Field(
        default="accurate",
        description=(
            f"How instance masks are drawn; used when {reads}. Each read builds "
            "one full-image mask per detection, at either quality. accurate: "
            "the masks the model defines (a low-resolution mask view is "
            "upscaled with the model's own interpolation). coarse: a mask view's "
            "grid cells are thresholded, then expanded, so edges can be blocky "
            "and small parts can be lost; it skips the accurate upscaling only. "
            "v2/instance_segmentation@v1 currently returns full-resolution "
            "masks, so both give the same result for it; they differ only for "
            "producers of compact mask views."
        ),
        examples=["accurate", "coarse", "$inputs.mask_quality"],
    )

    return field


MaskQuality = Literal[MASK_QUALITIES] | Ref(STRING_KIND)


def row_colors(
    class_id: np.ndarray,
    tracker_id: Optional[np.ndarray],
    *,
    color_palette: str,
    palette_size: int,
    custom_colors: Sequence[str],
    color_axis: str,
) -> np.ndarray:
    """Check the color parameters and return each row's RGB color.

    ``resolve_palette`` checks the palette parameters, ``palette_indices``
    the axis.

    Args:
        class_id: ``(N,)`` class ids.
        tracker_id: ``(N,)`` tracker ids, or ``None``.
        color_palette: Palette name.
        palette_size: Matplotlib sample count.
        custom_colors: ``CUSTOM`` colors.
        color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.

    Returns:
        ``(N, 3)`` uint8 RGB colors.

    Raises:
        ValueError: On an invalid parameter, or ``TRACK`` without ids.
    """
    palette = resolve_palette(
        color_palette, palette_size=palette_size, custom_colors=custom_colors
    )
    indices = palette_indices(color_axis, class_id=class_id, tracker_id=tracker_id)
    colors = colors_of(palette, indices)

    return colors

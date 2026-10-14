"""Value checks shared by the annotation blocks' parameters.

Each block writes its own ``Params`` fields and validators; the validators
call the named checks here.

Every literal may also be a selector. A literal is checked when the workflow
is compiled; a selected value is checked by the same function when the row
runs, so both fail with the same message::

    thickness: 0            -> compile error  "thickness must be an integer >= 1"
    thickness: $inputs.t    -> row error      same text, once t arrives

Each block declares its checks once, in a module ``_CHECKS`` dict of field
name -> check: its params validator passes it to ``check_literals``, and
``run`` applies every check to the received values.

``copy_image`` (V1) is rejected by ``reject_copy_image``, whatever its value:
these blocks always return a new image and never change their input. To draw
over an earlier drawing, chain the blocks (``image: $steps.<previous>.image``).
Palette blocks check literal ``palette_size`` and ``custom_colors`` with
``check_palette_literals`` before their own ``_CHECKS``.
"""

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from pydantic import StrictFloat, StrictInt, ValidationError
from pydantic_core import PydanticCustomError
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    checked_palette_size,
    colors_of,
    palette_indices,
    parse_color,
    resolve_palette,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import MASK_QUALITIES

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


def reject_copy_image(values: Any, *, title: str) -> Any:
    """Reject V1 ``copy_image`` before any field of a step is validated.

    Called by each annotation block's ``mode="before"`` params validator.
    The error is located at the field, like other parameter errors, and
    says why the field is gone.

    Args:
        values: Raw step values given to the validator.
        title: Name of the params class, used as the error title.

    Returns:
        ``values``, unchanged.

    Raises:
        ValidationError: When ``values`` holds ``copy_image``, whatever its
            value.
    """
    if isinstance(values, Mapping) and "copy_image" in values:
        removed = PydanticCustomError("removed_parameter", _COPY_IMAGE_MESSAGE)
        raise ValidationError.from_exception_data(
            title,
            [
                {
                    "type": removed,
                    "loc": ("copy_image",),
                    "input": values["copy_image"],
                }
            ],
        )

    return values


def check_palette_literals(*, palette_size: Any, custom_colors: Any) -> None:
    """Check the literal palette parameters of a step; skip selectors.

    Called by each palette block's ``mode="after"`` params validator, before
    the block's own ``_CHECKS``.

    Args:
        palette_size: Literal size or selector.
        custom_colors: Literal list of colors (items may be selectors), or a
            selector.

    Raises:
        ValueError: On a palette size below 1 or an unreadable color.
    """
    check_literals(
        {"palette_size": palette_size},
        {"palette_size": checked_palette_size},
    )
    if not is_selector(custom_colors):
        for index, color in enumerate(custom_colors):
            if not is_selector(color):
                parse_color(color, field=f"custom_colors[{index}]")


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

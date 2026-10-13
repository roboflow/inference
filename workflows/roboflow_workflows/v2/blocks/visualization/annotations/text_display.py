"""Text placed on an image: ``v2/text_display@v1``.

Port of V1 ``roboflow_core/text_display@v1``: one block of text with an
optional background, at a pixel position or at an image anchor. Pixels equal
V1 (OpenCV Hershey font, V1 layout and clipping).

Template: each ``{{ $parameters.<name> }}`` becomes ``str(value)`` of
``text_parameters[name]``, after that name's ``text_parameters_operations``
(query operations, as V1's UQL). A placeholder naming no parameter stays as
written, as in V1. Parameters no placeholder names are never transformed.

Layout (V1 ``text_display/utils.py``)::

    lines       text split on "\\n"
    advance     height("Ag") + baseline; consecutive lines 1/4 advance apart
    box         widest line and all lines, plus padding on every side
    position    absolute: (position_x, position_y)
                relative: anchor of the image - the same anchor of the box + offset
    clamp       the box moves inside the image; on an axis where it is larger
                than the image it starts at 0, and the image edge clips it
    align       each line left, center or right within the widest line

Host boundary: the layout is host arithmetic (``cv2.getTextSize``). Only the
rectangle the text and its background can touch is copied to the host,
drawn with OpenCV in frame coordinates (so clipping equals V1) and copied
back; the rest of the image stays on its device.

Differences from V1, all clearer errors:

- Selected ``text_align``, ``position_mode`` and ``anchor`` values are
  checked when the row runs. V1 drew an unknown ``text_align`` as left and
  an unknown ``position_mode`` as relative. ``anchor`` still ignores case.
- ``text_parameters_operations`` naming no ``text_parameters`` key fails
  compilation; V1 ignored it.
- ``copy_image`` is rejected (see ``params``).
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from roboflow_workflows.v2.blocks.query import (
    CompiledOperations,
    Operation,
    compile_operations,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    RGB,
    parse_color,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    AnnotationParams,
    Number,
    check_literals,
    checked_choice,
    checked_int,
    checked_number,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Ref

BLOCK_TYPE = "v2/text_display@v1"
PARAMETER_PATTERN = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")
TRANSPARENT = "transparent"
TEXT_ALIGNMENTS = ("left", "center", "right")
POSITION_MODES = ("absolute", "relative")
# Anchor -> its (x, y) fraction of the image (and of the box).
ANCHORS = {
    "top_left": (0.0, 0.0),
    "top_center": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "center_left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "center_right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom_center": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class TextDisplayBlock(AnnotationBlock):
    """Draw templated text, with an optional background, onto a copy of the image.

    Port of V1 roboflow_core/text_display@v1.

    Example: ``text="Cars: {{ $parameters.cars }}"`` with
    ``text_parameters={"cars": "$steps.model.predictions"}`` and
    ``text_parameters_operations={"cars": [{"type": "SequenceLength"}]}``
    writes ``Cars: 3`` in the top-left corner.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Text Display",
        "port_of": "roboflow_core/text_display@v1",
        "section": "visualization",
    }

    class Params(AnnotationParams):
        text: StrictStr = Field(
            description=(
                "Text to draw; '\\n' starts a new line. Each "
                "{{ $parameters.<name> }} is replaced by text_parameters[name]; "
                "a placeholder naming no parameter is drawn as written."
            ),
            examples=["Cars: {{ $parameters.cars }}", "Hello World"],
        )
        text_parameters: Dict[str, Any | Ref(allow_none=True)] = Field(
            default_factory=dict,
            description="Values of the placeholders, by name: literals or selectors.",
            examples=[{"cars": "$steps.model.predictions", "zone": "$inputs.zone"}],
        )
        text_parameters_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description=(
                "Query operations applied to a parameter before it is written, "
                "by parameter name, e.g. SequenceLength to count predictions. "
                "Every key must be a text_parameters key."
            ),
            examples=[{"cars": [{"type": "SequenceLength"}]}],
        )
        text_color: str | Ref(STRING_KIND) = Field(
            default="WHITE",
            description="Text color: #RRGGBB, rgb(R, G, B), bgr(B, G, R) or a name.",
            examples=["WHITE", "#FF0000", "$inputs.text_color"],
        )
        background_color: str | Ref(STRING_KIND) = Field(
            default="BLACK",
            description=(
                "Background color, in the text_color formats, or 'transparent' "
                "(any case) for no background."
            ),
            examples=["BLACK", "transparent", "$inputs.background_color"],
        )
        background_opacity: Number | Ref(FLOAT_KIND) = Field(
            default=1.0,
            description="Background weight in [0, 1]; 0 draws no background.",
            examples=[1.0, 0.5, "$inputs.background_opacity"],
        )
        font_scale: Number | Ref(FLOAT_KIND) = Field(
            default=1.0,
            description="OpenCV font scale in [0.1, 10].",
            examples=[1.0, 0.5, "$inputs.font_scale"],
        )
        font_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Text stroke width in [1, 10].",
            examples=[2, "$inputs.font_thickness"],
        )
        padding: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Pixels between the text and the background edge; >= 0.",
            examples=[10, "$inputs.padding"],
        )
        text_align: Literal[TEXT_ALIGNMENTS] | Ref(STRING_KIND) = Field(
            default="left",
            description="Alignment of each line within the widest line.",
            examples=["left", "center", "right"],
        )
        border_radius: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description="Corner radius of the background; >= 0.",
            examples=[0, 8, "$inputs.border_radius"],
        )
        position_mode: Literal[POSITION_MODES] | Ref(STRING_KIND) = Field(
            default="relative",
            description=(
                "absolute: the box's top-left corner is (position_x, "
                "position_y). relative: the box sits at anchor, moved by "
                "(offset_x, offset_y)."
            ),
            examples=["relative", "absolute"],
        )
        position_x: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description="Box left edge in pixels (absolute mode).",
            examples=[10, "$inputs.x"],
        )
        position_y: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description="Box top edge in pixels (absolute mode).",
            examples=[10, "$inputs.y"],
        )
        anchor: Literal[tuple(ANCHORS)] | Ref(STRING_KIND) = Field(
            default="top_left",
            description=(
                "Point of the image the box is attached to by the same point "
                "of the box (relative mode); case is ignored."
            ),
            examples=["top_left", "center", "bottom_right"],
        )
        offset_x: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Pixels the box moves right of the anchor (relative mode).",
            examples=[10, -20, "$inputs.offset_x"],
        )
        offset_y: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Pixels the box moves down from the anchor (relative mode).",
            examples=[10, -20, "$inputs.offset_y"],
        )

        # Not a model validator: the engine reruns those per invocation, and
        # this literal-only field and the parameter names never change.
        @field_validator("text_parameters_operations")
        @classmethod
        def _operations_compile(
            cls, value: Dict[str, List[Operation]], info: ValidationInfo
        ) -> Dict[str, List[Operation]]:
            if "text_parameters" in info.data:
                compile_parameter_operations(
                    value, parameter_names=frozenset(info.data["text_parameters"])
                )

            return value

        @model_validator(mode="after")
        def _check_literals(self) -> "TextDisplayBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def __init__(self) -> None:
        # (operations definition, compiled chains); replaced as a whole.
        self._compiled: Optional[Tuple[Any, Dict[str, CompiledOperations]]] = None

    def run(
        self,
        *,
        image: ImageData,
        text: str,
        text_parameters: Dict[str, Any],
        text_parameters_operations: Dict[str, List[Any]],
        text_color: str,
        background_color: str,
        background_opacity: float,
        font_scale: float,
        font_thickness: int,
        padding: int,
        text_align: str,
        border_radius: int,
        position_mode: str,
        position_x: int,
        position_y: int,
        anchor: str,
        offset_x: int,
        offset_y: int,
    ) -> Dict[str, Any]:
        """Fill the template and draw it onto a copy of ``image``.

        Args:
            image: RGB image; not modified.
            text: Template text.
            text_parameters: Resolved placeholder values by name.
            text_parameters_operations: Validated operations by name.
            text_color: Text color.
            background_color: Background color or ``transparent``.
            background_opacity: Background weight in [0, 1].
            font_scale: OpenCV font scale.
            font_thickness: Text stroke width.
            padding: Pixels between text and background edge.
            text_align: ``left``, ``center`` or ``right``.
            border_radius: Background corner radius.
            position_mode: ``absolute`` or ``relative``.
            position_x: Box left edge (absolute).
            position_y: Box top edge (absolute).
            anchor: Image and box anchor (relative).
            offset_x: Horizontal offset from the anchor (relative).
            offset_y: Vertical offset from the anchor (relative).

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters.
            QueryRuntimeError: When an operation cannot process its value.
        """
        values = dict(
            text_color=text_color,
            background_color=background_color,
            background_opacity=background_opacity,
            font_scale=font_scale,
            font_thickness=font_thickness,
            padding=padding,
            text_align=text_align,
            border_radius=border_radius,
            position_mode=position_mode,
            position_x=position_x,
            position_y=position_y,
            anchor=anchor,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        for name, check in _CHECKS.items():
            check(values[name])

        formatted = format_text(
            text,
            parameters=text_parameters,
            operations=self._operations(text_parameters_operations, text_parameters),
        )
        canvas = canvas_of(image, block=self.type)
        style = TextBoxStyle(
            scale=float(font_scale),
            thickness=font_thickness,
            padding=padding,
            align=text_align,
            border_radius=border_radius,
            text_rgb=parse_color(text_color, field="text_color"),
            background_rgb=(
                None
                if background_color.lower() == TRANSPARENT
                else parse_color(background_color, field="background_color")
            ),
            background_opacity=float(background_opacity),
        )
        box_xy = (position_x, position_y) if position_mode == "absolute" else None
        layout = text_layout(
            formatted,
            style=style,
            size_hw=image.size_hw,
            box_xy=box_xy,
            anchor=anchor.lower(),
            offset_xy=(offset_x, offset_y),
        )
        paint_text_box(canvas, layout, style=style)
        result = annotated(image, canvas)

        return result

    def _operations(
        self, definition: Dict[str, List[Any]], parameters: Dict[str, Any]
    ) -> Dict[str, CompiledOperations]:
        # Compiled once per definition; a concurrent call at worst compiles
        # the same chains again.
        cached = self._compiled
        if cached is None or cached[0] != definition:
            compiled = compile_parameter_operations(
                definition, parameter_names=frozenset(parameters)
            )
            cached = (definition, compiled)
            self._compiled = cached

        operations = cached[1]

        return operations


@dataclass(frozen=True)
class TextBoxStyle:
    """How the text block is drawn.

    Args:
        scale: OpenCV font scale.
        thickness: Text stroke width.
        padding: Pixels between text and background edge.
        align: ``left``, ``center`` or ``right``.
        border_radius: Background corner radius.
        text_rgb: Text color.
        background_rgb: Background color, or ``None`` for none.
        background_opacity: Background weight in [0, 1].
    """

    scale: float
    thickness: int
    padding: int
    align: str
    border_radius: int
    text_rgb: RGB
    background_rgb: Optional[RGB]
    background_opacity: float


@dataclass(frozen=True)
class TextLayout:
    """Where each line and the box go, in frame pixels.

    Args:
        lines: The text's lines.
        line_widths: Pixel width of each line; 0 for blank lines.
        max_width: Widest line.
        ref_height: Height of ``"Ag"``: the baseline offset of every line.
        line_advance: ``ref_height`` plus the baseline depth.
        line_spacing: Extra pixels between consecutive lines.
        box_xywh: Box left, top, width and height, before clipping.
        size_hw: Image height and width.
    """

    lines: Tuple[str, ...]
    line_widths: Tuple[int, ...]
    max_width: int
    ref_height: int
    line_advance: int
    line_spacing: int
    box_xywh: Tuple[int, int, int, int]
    size_hw: Tuple[int, int]

    def line_origins(self, style: TextBoxStyle) -> List[Tuple[str, int, int]]:
        """Return ``(line, x, y)`` of every line V1 draws; ``(x, y)`` is the origin.

        Args:
            style: The style the layout was computed with.

        Returns:
            Non-blank lines that start inside the image, in order.
        """
        height, width = self.size_hw
        box_x, box_y, _, _ = self.box_xywh
        current_y = box_y + style.padding
        origins = []
        for line, line_width in zip(self.lines, self.line_widths):
            if line.strip():
                text_x = (
                    box_x
                    + style.padding
                    + _align_offset(
                        style.align, max_width=self.max_width, line_width=line_width
                    )
                )
                text_y = current_y + self.ref_height
                if text_y > 0 and current_y < height and text_x < width:
                    origins.append((line, text_x, text_y))
            current_y += self.line_advance + self.line_spacing

        return origins


def format_text(
    template: str,
    *,
    parameters: Dict[str, Any],
    operations: Dict[str, CompiledOperations],
) -> str:
    """Replace each ``{{ $parameters.<name> }}`` with its parameter's text.

    Args:
        template: Text with placeholders.
        parameters: Placeholder values by name.
        operations: Compiled chains applied to a value before ``str``.

    Returns:
        The text; placeholders naming no parameter are kept as written.

    Raises:
        QueryRuntimeError: When an operation cannot process its value.
    """
    rendered: Dict[str, str] = {}

    def replace(match: re.Match) -> str:
        name = match.group(2)
        if name not in parameters:
            return match.group(0)
        if name not in rendered:
            value = parameters[name]
            if name in operations:
                value = operations[name].apply(value, parameters={})
            rendered[name] = str(value)

        return rendered[name]

    text = PARAMETER_PATTERN.sub(replace, template)

    return text


def compile_parameter_operations(
    definition: Dict[str, List[Any]], *, parameter_names: frozenset
) -> Dict[str, CompiledOperations]:
    """Compile per-parameter operation chains that name no runtime values.

    Args:
        definition: Operations by parameter name.
        parameter_names: Keys of ``text_parameters``.

    Returns:
        Compiled chains by parameter name.

    Raises:
        ValueError: When a key is not a ``text_parameters`` key.
        QueryDefinitionError: On an invalid chain.
    """
    compiled = {}
    for name, operations in definition.items():
        if name not in parameter_names:
            raise ValueError(
                f"{BLOCK_TYPE}: text_parameters_operations names `{name}`, which "
                f"is not a text_parameters key (text_parameters: "
                f"{sorted(parameter_names)})"
            )
        compiled[name] = compile_operations(
            operations,
            path=f"$.text_parameters_operations.{name}",
            declared_names=frozenset(),
        )

    return compiled


def text_layout(
    text: str,
    *,
    style: TextBoxStyle,
    size_hw: Tuple[int, int],
    box_xy: Optional[Tuple[int, int]],
    anchor: str,
    offset_xy: Tuple[int, int],
) -> TextLayout:
    """Measure the text and place its box as V1 ``compute_layout`` does.

    Args:
        text: Formatted text; ``\\n`` separates lines.
        style: Font and padding.
        size_hw: Image height and width.
        box_xy: Box top-left corner (absolute mode), or ``None`` to place it
            at ``anchor`` (relative mode).
        anchor: Lower-case ``ANCHORS`` key (relative mode).
        offset_xy: Offset from the anchor (relative mode).

    Returns:
        The layout, with the box clamped into the image.
    """
    height, width = size_hw
    lines = tuple(text.split("\n")) if text else ("",)
    (_, ref_height), baseline = cv2.getTextSize(
        "Ag", FONT, style.scale, style.thickness
    )
    line_advance = ref_height + baseline
    line_spacing = max(1, int(round(0.25 * line_advance)))
    line_widths = tuple(
        (
            cv2.getTextSize(line, FONT, style.scale, style.thickness)[0][0]
            if line.strip()
            else 0
        )
        for line in lines
    )
    max_width = max(line_widths, default=0)
    box_width = max_width + 2 * style.padding
    box_height = (
        len(lines) * line_advance
        + max(0, len(lines) - 1) * line_spacing
        + 2 * style.padding
    )

    if box_xy is None:
        fraction_x, fraction_y = ANCHORS[anchor]
        box_xy = (
            int(round(fraction_x * width))
            - int(round(fraction_x * box_width))
            + offset_xy[0],
            int(round(fraction_y * height))
            - int(round(fraction_y * box_height))
            + offset_xy[1],
        )
    box_x = _clamped(box_xy[0], size=box_width, limit=width)
    box_y = _clamped(box_xy[1], size=box_height, limit=height)

    layout = TextLayout(
        lines=lines,
        line_widths=line_widths,
        max_width=max_width,
        ref_height=ref_height,
        line_advance=line_advance,
        line_spacing=line_spacing,
        box_xywh=(box_x, box_y, box_width, box_height),
        size_hw=(height, width),
    )

    return layout


def paint_text_box(
    canvas: torch.Tensor, layout: TextLayout, *, style: TextBoxStyle
) -> None:
    """Draw the background and text lines into ``canvas``.

    Only the rectangle the drawing can touch travels to the host and back.

    Args:
        canvas: Contiguous ``(3, H, W)`` uint8 RGB image, changed in place.
        layout: The text layout in this image.
        style: The style the layout was computed with.
    """
    height, width = layout.size_hw
    box_x, box_y, box_width, box_height = layout.box_xywh
    background = (
        max(0, box_x),
        max(0, box_y),
        min(width, box_x + box_width),
        min(height, box_y + box_height),
    )
    origins = layout.line_origins(style)
    region = _touched_region(background, origins, style=style, size_hw=(height, width))
    if region is None:
        return

    x0, y0, x1, y1 = region
    # Frame coordinates up to the region's far corner, as text.paint_labels.
    host = np.zeros((y1, x1, 3), dtype=np.uint8)
    host[y0:y1, x0:x1] = canvas[:, y0:y1, x0:x1].permute(1, 2, 0).cpu().numpy()
    _draw_background(host, background, style=style)
    for line, x, y in origins:
        cv2.putText(
            host,
            line,
            (x, y),
            FONT,
            style.scale,
            tuple(int(channel) for channel in style.text_rgb),
            style.thickness,
            cv2.LINE_AA,
        )

    drawn = np.ascontiguousarray(host[y0:y1, x0:x1].transpose(2, 0, 1))
    canvas[:, y0:y1, x0:x1] = torch.from_numpy(drawn).to(canvas.device)


def _touched_region(
    background: Tuple[int, int, int, int],
    origins: List[Tuple[str, int, int]],
    *,
    style: TextBoxStyle,
    size_hw: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    # Half-open (x0, y0, x1, y1) holding the background and every stroke of
    # the text (anti-aliasing reaches past the measured size); None if empty.
    height, width = size_hw
    x0, y0, x1, y1 = background
    if style.background_rgb is None or style.background_opacity <= 0.0:
        x0, y0, x1, y1 = width, height, 0, 0
    reach = style.thickness + 3
    for line, x, y in origins:
        (line_width, line_height), baseline = cv2.getTextSize(
            line, FONT, style.scale, style.thickness
        )
        x0, y0 = min(x0, x - reach), min(y0, y - line_height - reach)
        x1, y1 = max(x1, x + line_width + reach), max(y1, y + baseline + reach)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, x1, y1


def _draw_background(
    host: np.ndarray, background: Tuple[int, int, int, int], *, style: TextBoxStyle
) -> None:
    # V1 draw_background: half-open background, opaque or blended in its ROI.
    x1, y1, x2, y2 = background
    color = style.background_rgb
    if color is None or style.background_opacity <= 0.0 or x2 <= x1 or y2 <= y1:
        return

    color = tuple(int(channel) for channel in color)
    if style.background_opacity >= 1.0:
        _fill_rounded(host, (x1, y1, x2 - 1, y2 - 1), color, style.border_radius)
        return

    roi = host[y1:y2, x1:x2]
    overlay = roi.copy()
    _fill_rounded(overlay, (0, 0, x2 - x1 - 1, y2 - y1 - 1), color, style.border_radius)
    opacity = style.background_opacity
    host[y1:y2, x1:x2] = cv2.addWeighted(overlay, opacity, roi, 1 - opacity, 0)


def _fill_rounded(
    image: np.ndarray, corners: Tuple[int, int, int, int], color: RGB, radius: int
) -> None:
    # V1 draw_rounded_rectangle: inclusive corners; the radius shrinks to fit.
    x1, y1, x2, y2 = corners
    if x2 <= x1 or y2 <= y1:
        return

    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius <= 0:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        return

    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for centre, start_angle in (
        ((x1 + radius, y1 + radius), 180),
        ((x2 - radius, y1 + radius), 270),
        ((x1 + radius, y2 - radius), 90),
        ((x2 - radius, y2 - radius), 0),
    ):
        cv2.ellipse(image, centre, (radius, radius), start_angle, 0, 90, color, -1)


def _align_offset(align: str, *, max_width: int, line_width: int) -> int:
    if align == "center":
        return (max_width - line_width) // 2
    if align == "right":
        return max_width - line_width

    return 0


def _clamped(start: int, *, size: int, limit: int) -> int:
    # V1 clamp_box on one axis: inside [0, limit - size]; 0 when too large.
    if size > limit:
        return 0

    clamped = max(0, min(start, limit - size))

    return clamped


def _checked_background(value: Any) -> Any:
    if isinstance(value, str) and value.lower() == TRANSPARENT:
        return value

    color = parse_color(value, field="background_color")

    return color


def _checked_anchor(value: Any) -> str:
    lowered = value.lower() if isinstance(value, str) else value
    anchor = checked_choice(lowered, name="anchor", choices=tuple(ANCHORS))

    return anchor


def _checked_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")

    return value


def _checked_thickness(value: Any) -> int:
    thickness = checked_int(value, name="font_thickness", minimum=1)
    if thickness > 10:
        raise ValueError(f"font_thickness must be at most 10, got {value!r}")

    return thickness


_CHECKS = {
    "text_color": lambda value: parse_color(value, field="text_color"),
    "background_color": _checked_background,
    "background_opacity": lambda value: checked_number(
        value, name="background_opacity", minimum=0.0, maximum=1.0
    ),
    "font_scale": lambda value: checked_number(
        value, name="font_scale", minimum=0.1, maximum=10.0
    ),
    "font_thickness": _checked_thickness,
    "padding": lambda value: checked_int(value, name="padding", minimum=0),
    "text_align": lambda value: checked_choice(
        value, name="text_align", choices=TEXT_ALIGNMENTS
    ),
    "border_radius": lambda value: checked_int(value, name="border_radius", minimum=0),
    "position_mode": lambda value: checked_choice(
        value, name="position_mode", choices=POSITION_MODES
    ),
    "position_x": lambda value: _checked_integer(value, name="position_x"),
    "position_y": lambda value: _checked_integer(value, name="position_y"),
    "anchor": _checked_anchor,
    "offset_x": lambda value: _checked_integer(value, name="offset_x"),
    "offset_y": lambda value: _checked_integer(value, name="offset_y"),
}

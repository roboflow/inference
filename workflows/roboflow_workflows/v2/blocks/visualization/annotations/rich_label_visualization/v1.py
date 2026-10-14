"""TrueType detection labels: ``roboflow_core/rich_label_visualization@v1``.

Port of V1 ``roboflow_core/rich_label_visualization@v1`` (``sv.RichLabelAnnotator``
with an approved font). Label texts, colors and anchors are those of
``roboflow_core/label_visualization@v2`` (see ``label``); only the font and layout
differ. Labels are drawn in row order, so a later row's label covers an
earlier one. Pixels equal V1.

Layout per row (``sv.RichLabelAnnotator``)::

    lines        the text's lines, each wrapped at max_line_length characters
    size         widest line x summed line heights, plus padding between lines
                 and on every side (Pillow textbbox of each line)
    background   placed at the anchor like sv.LabelAnnotator (text.background_rect);
                 rounded corners
    text         first line at background + padding - textbbox("Tg") origin;
                 each next line one line height + padding lower

Labels are not moved to stay inside the image: the image edge clips them.

Fonts: ``font_family`` names an approved font of the V1 registry (display
name or legacy snake_case id). V2 reads only fonts already on this machine:
the font installed with ``roboflow_workflows`` (``make download_fonts``) or
a copy V1 cached earlier whose SHA-256 matches the registry. It never
downloads a font while a workflow compiles or runs; a missing font fails
with the command that installs it.

Host boundary: Pillow draws on the host. The rectangle enclosing every
label is copied to the host once, drawn and copied back; the rest of the
image stays on its device. Labels spread over the whole image therefore copy
most of it.

Differences from V1:

- A missing font is an error naming ``make download_fonts``; V1 downloads it.
- ``text_padding`` must be >= 0 (V1 accepted any integer).
- Changing ``custom_colors`` takes effect at once. V1 cached its annotator
  without them, so a step kept the first colors it saw.
- ``copy_image`` is rejected (see ``params``).
"""

import textwrap
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.core_steps.visualizations.common import fonts
from roboflow_workflows.core_steps.visualizations.common.fonts.downloader import (
    compute_file_sha256,
)
from roboflow_workflows.core_steps.visualizations.common.fonts.registry import (
    FONTS_REGISTRY,
)
from roboflow_workflows.core_steps.visualizations.common.fonts.schema import (
    FontFamilyName,
    font_family_to_identifier,
    get_default_font_family_display_name,
    get_font_family_identifiers,
)
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import (
    CENTER_OF_MASS,
    box_anchors,
    mass_centres,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
    RGB,
    parse_color,
)
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization.v2 import (
    TEXT_OPTIONS,
    TEXT_SIZE_MODES,
    label_texts,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    POSITIONS_WITH_MASS,
    check_literals,
    check_palette_literals,
    checked_choice,
    checked_int,
    checked_mask_quality,
    reject_copy_image,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    MASK_QUALITIES,
    boxes_of,
    dense_masks_of,
    has_masks,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import background_rect
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

Rect = Tuple[int, int, int, int]

LegacyFontId = Literal[get_font_family_identifiers()]

_REFERENCE_MIN_DIMENSION_PX = 1080

_REFERENCE_FONT_SIZE_PT = 14

_FONT_SIZE_LIMITS = (8, 128)

_THREAD_FONTS = threading.local()

_THREAD_FONT_LIMIT = 32


class RichLabelVisualizationBlock(Block):
    """Draw an anti-aliased TrueType label at each detection onto a copy of the image.

    Port of V1 roboflow_core/rich_label_visualization@v1.

    Example: ``text="Class and Confidence"`` with ``font_family="Noto Sans"``
    writes ``"car 0.93"`` in white on the class color above each box's
    top-left corner.
    """

    type = "roboflow_core/rich_label_visualization@v1"
    prunable = True
    metadata = {
        "name": "Rich Label Visualization",
        "port_of": "roboflow_core/rich_label_visualization@v1",
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
        text: Literal[TEXT_OPTIONS] | Ref(STRING_KIND) = Field(
            default="Class",
            description=(
                "What each label says; the options of "
                "roboflow_core/label_visualization@v2: Class, Confidence, Class and "
                "Confidence, Index, Dimensions, Area, Area (mask), Area "
                "(converted), Tracker Id, Time In Zone. A selector may also "
                "name any other bboxes_metadata key; rows without it get an "
                "empty label."
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
        font_family: FontFamilyName | LegacyFontId | Ref(STRING_KIND) = Field(
            default=get_default_font_family_display_name(),
            description=(
                "Approved font, by display name (legacy snake_case ids are "
                "accepted). Noto Sans covers Latin, Greek and Cyrillic; other "
                "characters show the font's missing-glyph box. The font must "
                "be installed (make download_fonts); none is downloaded."
            ),
            examples=["Geist Mono", "Noto Sans", "$inputs.font_family"],
        )
        text_size_mode: Literal[TEXT_SIZE_MODES] | Ref(STRING_KIND) = Field(
            default="Manual",
            description=(
                "Manual draws font_size points. Automatic scales it with the "
                "image (font_size 14 is the size at 1080 px), within 8-128."
            ),
            examples=["Manual", "Automatic"],
        )
        font_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=14,
            description="Font size in points; >= 1.",
            examples=[14, "$inputs.font_size"],
        )
        text_padding: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Pixels around the text and between lines; >= 0.",
            examples=[10, "$inputs.text_padding"],
        )
        border_radius: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description="Corner radius of the label background; >= 0.",
            examples=[0, 5],
        )
        max_line_length: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=None,
            description=(
                "Characters per line before the text wraps; >= 1. Empty: no "
                "wrapping."
            ),
            examples=[30, "$inputs.max_line_length"],
        )
        mask_quality: Literal[MASK_QUALITIES] | Ref(STRING_KIND) = Field(
            default="accurate",
            description=(
                "How instance masks are drawn; used when text is Area or text_position "
                "is CENTER_OF_MASS, on instance masks. Each read builds one full-image "
                "mask per detection, at either quality. accurate: the masks the model "
                "defines (a low-resolution mask view is upscaled with the model's own "
                "interpolation). coarse: a mask view's grid cells are thresholded, "
                "then expanded, so edges can be blocky and small parts can be lost; it "
                "skips the accurate upscaling only. "
                "The model block's fast implementation retains compact scores when "
                "supported, so accurate and coarse can differ. Its accurate "
                "implementation and dense-only models already supply binary masks; "
                "both settings then use those same masks."
            ),
            examples=["accurate", "coarse", "$inputs.mask_quality"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_palette_literals(self) -> "RichLabelVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

        @model_validator(mode="after")
        def _check_literals(self) -> "RichLabelVisualizationBlock.Params":
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
        font_family: str,
        text_size_mode: str,
        font_size: int,
        text_padding: int,
        border_radius: int,
        max_line_length: Optional[int],
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
            font_family: Approved font display name or legacy id.
            text_size_mode: ``Manual`` or ``Automatic``.
            font_size: Font size in points.
            text_padding: Background padding and line gap.
            border_radius: Background corner radius.
            max_line_length: Wrap width in characters, or ``None``.
            mask_quality: Mask quality for ``Area`` and ``CENTER_OF_MASS``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, an unknown or uninstalled
                font, a row without a class name for ``Class``, an unknown
                text key, ``CENTER_OF_MASS`` without masks, keypoints without
                boxes or predictions of another workflow input.
        """
        values = dict(
            text_position=text_position,
            text_color=text_color,
            font_family=font_family,
            text_size_mode=text_size_mode,
            font_size=font_size,
            text_padding=text_padding,
            border_radius=border_radius,
            max_line_length=max_line_length,
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

        masks = None
        if (text_position == CENTER_OF_MASS or text == "Area") and has_masks(
            predictions
        ):
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
        font = _loaded_font(
            font_path(font_family),
            size=_effective_size(font_size, mode=text_size_mode, size_hw=image.size_hw),
        )
        labels = rich_labels(
            label_texts(rows, text, masks=masks, block=self.type),
            anchors,
            position=text_position,
            font=font,
            padding=text_padding,
            max_line_length=max_line_length,
            background_rgb=[tuple(color) for color in colors.tolist()],
        )
        paint_rich_labels(
            canvas,
            labels,
            font=font,
            text_rgb=parse_color(text_color, field="text_color"),
            border_radius=border_radius,
        )
        result = annotated(image, canvas)

        return result


@dataclass(frozen=True)
class RichLabel:
    """One label: a background and text lines, in frame pixels.

    Args:
        background: Inclusive ``(x1, y1, x2, y2)``, as Pillow fills it.
        background_rgb: Background color.
        lines: ``(text, x, y)`` per line; ``(x, y)`` is Pillow's text origin.
    """

    background: Rect
    background_rgb: RGB
    lines: Tuple[Tuple[str, int, int], ...]


def font_path(font_family: str) -> Path:
    """Return the local file of an approved font, without downloading it.

    Args:
        font_family: Display name (``"Geist Mono"``) or legacy id
            (``"geist_mono"``).

    Returns:
        The installed font, else a V1-cached copy with the registry checksum.

    Raises:
        ValueError: When ``font_family`` is not an approved font, or the font
            is not on this machine (the message names the install command).
    """
    if not isinstance(font_family, str):
        raise ValueError(f"font_family must be a font name, got {font_family!r}")

    identifier = font_family_to_identifier(font_family)
    path = _local_font(identifier)

    return path


def rich_labels(
    texts: Sequence[str],
    anchors: np.ndarray,
    *,
    position: str,
    font: ImageFont.FreeTypeFont,
    padding: int,
    max_line_length: Optional[int],
    background_rgb: Sequence[RGB],
) -> List[RichLabel]:
    """Lay out labels as ``sv.RichLabelAnnotator`` (no smart position) does.

    Args:
        texts: One text per label.
        anchors: ``(N, 2)`` anchor points, truncated to integers as V1 does.
        position: One of ``POSITIONS_WITH_MASS``.
        font: The loaded font.
        padding: Background padding and line gap.
        max_line_length: Wrap width in characters, or ``None``.
        background_rgb: Background color per label.

    Returns:
        The labels, in drawing order.
    """
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_left, text_top, _, _ = measure.textbbox((0, 0), "Tg", font=font)
    labels = []
    for text, anchor, background in zip(
        texts, np.asarray(anchors).astype(int), background_rgb
    ):
        lines = _wrapped(text, max_line_length)
        heights = []
        widest = 0
        for line in lines:
            left, top, right, bottom = measure.textbbox((0, 0), line, font=font)
            widest = max(widest, right - left)
            heights.append(bottom - top)
        width = int(widest + 2 * padding)
        height = int(sum(heights) + (len(lines) - 1) * padding + 2 * padding)
        x1, y1, x2, y2 = background_rect(
            int(anchor[0]),
            int(anchor[1]),
            width=width,
            height=height,
            position=position,
        )

        placed = []
        line_y = y1 + padding - text_top
        for line, line_height in zip(lines, heights):
            placed.append((line, x1 + padding - text_left, line_y))
            line_y += line_height + padding

        labels.append(
            RichLabel(
                background=(x1, y1, x2, y2),
                background_rgb=tuple(background),
                lines=tuple(placed),
            )
        )

    return labels


def paint_rich_labels(
    scene: torch.Tensor,
    labels: Sequence[RichLabel],
    *,
    font: ImageFont.FreeTypeFont,
    text_rgb: RGB,
    border_radius: int,
) -> None:
    """Draw labels into ``scene`` in order, exchanging one enclosing patch.

    Args:
        scene: Contiguous ``(3, H, W)`` uint8 RGB image, changed in place.
        labels: Labels in frame pixels.
        font: The font the labels were laid out with.
        text_rgb: Text color of every label.
        border_radius: Background corner radius.
    """
    height, width = int(scene.shape[1]), int(scene.shape[2])
    region = _enclosing_region(labels, font=font, height=height, width=width)
    if region is None:
        return

    x0, y0, x1, y1 = region
    patch = Image.fromarray(scene[:, y0:y1, x0:x1].permute(1, 2, 0).cpu().numpy())
    draw = ImageDraw.Draw(patch)
    for label in labels:
        bx1, by1, bx2, by2 = label.background
        draw.rounded_rectangle(
            (bx1 - x0, by1 - y0, bx2 - x0, by2 - y0),
            radius=border_radius,
            fill=label.background_rgb,
        )
        for line, x, y in label.lines:
            draw.text((x - x0, y - y0), line, font=font, fill=text_rgb)

    drawn = np.ascontiguousarray(np.asarray(patch).transpose(2, 0, 1))
    scene[:, y0:y1, x0:x1] = torch.from_numpy(drawn).to(scene.device)


def _enclosing_region(
    labels: Sequence[RichLabel],
    *,
    font: ImageFont.FreeTypeFont,
    height: int,
    width: int,
) -> Optional[Rect]:
    # Half-open (x0, y0, x1, y1) around every background and glyph, plus one
    # pixel for anti-aliasing, clipped to the image; None when nothing shows.
    if not labels:
        return None

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for label in labels:
        bx1, by1, bx2, by2 = label.background
        x0, y0 = min(x0, bx1), min(y0, by1)
        x1, y1 = max(x1, bx2 + 1), max(y1, by2 + 1)
        for line, x, y in label.lines:
            left, top, right, bottom = measure.textbbox((x, y), line, font=font)
            x0, y0 = min(x0, left - 1), min(y0, top - 1)
            x1, y1 = max(x1, right + 1), max(y1, bottom + 1)
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), width), min(int(y1), height)
    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, x1, y1


def _wrapped(text: Any, max_line_length: Optional[int]) -> List[str]:
    # sv.annotators.utils.wrap_text.
    if not text:
        return [""]

    text = str(text)
    if max_line_length is None:
        return text.splitlines() or [""]

    lines = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(
            paragraph,
            width=max_line_length,
            break_long_words=True,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        lines.extend(wrapped or [""])

    return lines or [""]


def _effective_size(size: int, *, mode: str, size_hw: Sequence[int]) -> int:
    # V1 compute_adaptive_rich_font_size.
    if mode != "Automatic":
        return size

    base = _REFERENCE_FONT_SIZE_PT * min(size_hw) / _REFERENCE_MIN_DIMENSION_PX
    effective = round(base * size / _REFERENCE_FONT_SIZE_PT)
    clamped = max(_FONT_SIZE_LIMITS[0], min(_FONT_SIZE_LIMITS[1], effective))

    return clamped


@lru_cache(maxsize=None)
def _local_font(identifier: str) -> Path:
    # Successful lookups only: lru_cache does not keep exceptions, so a font
    # installed later is found by the next call.
    metadata = FONTS_REGISTRY[identifier]
    installed = fonts.ASSETS_DIR / metadata.identifier / metadata.file_name
    if installed.is_file():
        return installed

    cached = fonts.FONTS_CACHE_DIR / metadata.identifier / metadata.file_name
    if cached.is_file() and compute_file_sha256(cached) == metadata.sha256:
        return cached

    raise ValueError(
        f"font {metadata.display_name!r} is not installed on this machine. "
        "Install the approved fonts with `make download_fonts` (or `python "
        "workflows/build_scripts/download_fonts.py`) from the repository root; "
        "V2 workflows do not download fonts."
    )


def _loaded_font(path: Path, *, size: int) -> ImageFont.FreeTypeFont:
    loaded = getattr(_THREAD_FONTS, "fonts", None)
    if loaded is None or len(loaded) >= _THREAD_FONT_LIMIT:
        loaded = _THREAD_FONTS.fonts = {}
    key = (path, size)
    if key not in loaded:
        loaded[key] = ImageFont.truetype(str(path), size)

    font = loaded[key]

    return font


def _checked_line_length(value: Any) -> Optional[int]:
    if value is None:
        return None

    length = checked_int(value, name="max_line_length", minimum=1)

    return length


_CHECKS = {
    "text_position": lambda value: checked_choice(
        value, name="text_position", choices=POSITIONS_WITH_MASS
    ),
    "text_color": lambda value: parse_color(value, field="text_color"),
    "font_family": font_path,
    "text_size_mode": lambda value: checked_choice(
        value, name="text_size_mode", choices=TEXT_SIZE_MODES
    ),
    "font_size": lambda value: checked_int(value, name="font_size", minimum=1),
    "text_padding": lambda value: checked_int(value, name="text_padding", minimum=0),
    "border_radius": lambda value: checked_int(value, name="border_radius", minimum=0),
    "max_line_length": _checked_line_length,
    "mask_quality": checked_mask_quality,
}

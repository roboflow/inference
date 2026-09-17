from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field, PositiveInt, field_validator

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.core_steps.visualizations.common.base_colorable import (
    ColorableVisualizationBlock,
    ColorableVisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.fonts import (
    resolve_font_path,
)
from inference.core.workflows.core_steps.visualizations.common.fonts.schema import (
    FontFamilyName,
    coerce_font_family_input,
    font_family_field_json_schema_extra,
    get_default_font_family_display_name,
)
from inference.core.workflows.core_steps.visualizations.common.label_text import (
    TEXT_SIZE_MODE_AUTOMATIC,
    TEXT_SIZE_MODE_MANUAL,
    build_detection_labels,
    compute_adaptive_rich_font_size,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/rich_label_visualization@v1"
SHORT_DESCRIPTION = (
    "Draw high-quality, anti-aliased text labels on an image using approved "
    "TrueType fonts."
)
LONG_DESCRIPTION = """
Draw text labels on detected objects using high-quality TrueType font rendering. This block is
the rich-text counterpart of the Label Visualization block: instead of OpenCV's built-in
bitmap font, it renders labels with Pillow through Supervision's `RichLabelAnnotator`,
producing anti-aliased, professional-looking text suitable for production UIs, reports and
customer-facing imagery.

## How This Block Works

This block takes an image and detection predictions and draws text labels on each detected
object. The block:

1. Takes an image and predictions as input
2. Extracts label text for each detection based on the selected text option (class name,
   confidence, tracker ID, dimensions, area, time in zone, or index)
3. Resolves the selected `font_family` identifier to an approved font shipped with `inference`
4. Determines label position based on the selected anchor point
5. Applies background color styling based on the selected color palette, with colors assigned
   by class, index, or track ID
6. Renders anti-aliased text labels with the selected font, size, color, padding and border
   radius using Supervision's `RichLabelAnnotator`
7. Returns an annotated image with text labels overlaid on the original image

## Approved Fonts

Fonts are selected with the `font_family` parameter. Only fonts approved and distributed with
`inference` can be used - arbitrary font files, filesystem paths and remote font URLs are
**not** supported. This keeps rendering deterministic and avoids parsing untrusted font files.
Official Docker images and wheels ship with all approved fonts included; on bare source
checkouts a missing font is fetched on first use from its pinned, checksum-verified source
(disable with `ALLOW_WORKFLOWS_FONTS_DOWNLOAD=False`). All approved fonts are licensed under
the SIL Open Font License 1.1 and ship with their license texts.

Fonts are selected by display name. 20 fonts are available:

Monospaced: Geist Mono (default, by Vercel), Anonymous Pro, Courier Prime, Fira Code,
IBM Plex Mono, Inconsolata, JetBrains Mono, PT Mono, Roboto Mono, Source Code Pro,
Space Mono.

Sans serif: Geist, Inter, Lato, Montserrat, Noto Sans, Nunito Sans, Open Sans, Roboto,
Work Sans.

Legacy snake_case identifiers (e.g. `geist_mono`) are still accepted and normalized to
the display name.

Noto Sans offers the broadest character coverage (Latin, Greek, Cyrillic). Characters not
covered by the selected font render as the font's missing-glyph symbol (typically an empty
box) - pick Noto Sans when annotating non-Latin text.

## Comparison with Label Visualization

- **Label Visualization** (`roboflow_core/label_visualization@v1`) uses OpenCV's Hershey
  fonts - fast, but aliased and pixelated, with no font choice.
- **Rich Label Visualization** (this block) uses TrueType fonts rendered by Pillow -
  higher-quality anti-aliased output, selectable fonts, Unicode support and optional text
  wrapping (`max_line_length`), at a small additional rendering cost per frame.

## Common Use Cases

- **Customer-facing visualizations**: Render detection overlays that match your product's
  typography (e.g. Geist Mono) for websites, dashboards and reports
- **High-quality reporting**: Produce publication-ready annotated images with readable,
  anti-aliased labels
- **Non-Latin text**: Render labels containing Greek or Cyrillic characters with Noto Sans
"""


class RichLabelManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Rich Label Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator", "font", "label", "text"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-font",
                "blockPriority": 3,
                "supervision": True,
                "warnings": [
                    {
                        "property": "copy_image",
                        "value": False,
                        "message": "This setting will mutate its input image. If the input is used by other blocks, it may cause unexpected behavior.",
                    }
                ],
            },
        }
    )

    text: Union[
        Literal[
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
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="Class",
        description="Content to display in text labels. Options: 'Class' (class name), 'Confidence' (confidence score), 'Class and Confidence' (both), 'Tracker Id' (tracking ID for tracked objects), 'Time In Zone' (time spent in zone), 'Dimensions' (center coordinates and width x height), 'Area' (bounding box area in pixels), 'Area (mask)' (mask area in pixels from Mask Area Measurement block), 'Area (converted)' (mask area in converted units from Mask Area Measurement block), or 'Index' (detection index).",
        examples=["LABEL", "$inputs.text"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    text_position: Union[
        Literal[
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
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="TOP_LEFT",
        description="Anchor position for placing labels relative to each detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object).",
        examples=["CENTER", "$inputs.text_position"],
    )

    text_color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the label text. Can be a color name (e.g., 'WHITE', 'BLACK') or color code in HEX format (e.g., '#FFFFFF') or RGB format (e.g., 'rgb(255, 255, 255)').",
        default="WHITE",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.text_color"],
    )

    font_family: Union[
        FontFamilyName,
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default=get_default_font_family_display_name(),
        title="Font",
        description="Font used to render label text. Pick from approved fonts shipped with `inference`. Arbitrary font files or URLs are not supported. Pick 'Noto Sans' for the broadest character coverage.",
        examples=["Geist Mono", "$inputs.font_family"],
        json_schema_extra=font_family_field_json_schema_extra(),
    )

    text_size_mode: Union[
        Literal["Manual", "Automatic"],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default=TEXT_SIZE_MODE_MANUAL,
        title="Size mode",
        description="How label text size is chosen. 'Manual' uses Size directly. 'Automatic' picks a readable size from image resolution and treats Size as a multiplier around the 1080p baseline (14 pt).",
        examples=["Manual", "Automatic", "$inputs.text_size_mode"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    font_size: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Size",
        description="Font size of the label text, in points. In Manual mode this is the rendered size. In Automatic mode this multiplies the resolution-derived baseline (14 pt at 1080p min dimension).",
        default=14,
        examples=[14, "$inputs.font_size"],
    )

    text_padding: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Padding around the text in pixels. Controls the spacing between the text and the label background border.",
        default=10,
        examples=[10, "$inputs.text_padding"],
    )

    border_radius: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Border radius of the label background in pixels. Set to 0 for square corners. Higher values create more rounded corners for a softer appearance.",
        default=0,
        examples=[0, "$inputs.border_radius"],
    )

    max_line_length: Union[Optional[PositiveInt], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=None,
        description="Maximum number of characters per line before the label text wraps. Leave empty to disable wrapping.",
        examples=[30, "$inputs.max_line_length"],
    )

    @field_validator("font_family", mode="before")
    @classmethod
    def _coerce_font_family(cls, value: object) -> object:
        if isinstance(value, str) and not value.startswith("$"):
            return coerce_font_family_input(value)
        return value

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RichLabelVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return RichLabelManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        text_position: str,
        text_color: str,
        font_family: str,
        font_size: int,
        text_padding: int,
        border_radius: int,
        max_line_length: Optional[int],
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    text_position,
                    text_color,
                    font_family,
                    font_size,
                    text_padding,
                    border_radius,
                    max_line_length,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            # validates against the approved registry; prevents supervision's
            # silent fallback to the default PIL font on invalid paths
            font_path = resolve_font_path(font_family)

            self.annotatorCache[key] = sv.RichLabelAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                text_position=getattr(sv.Position, text_position),
                text_color=str_to_color(text_color),
                font_path=str(font_path),
                font_size=font_size,
                text_padding=text_padding,
                border_radius=border_radius,
                max_line_length=max_line_length,
            )

        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        text: Optional[str],
        text_position: Optional[str],
        text_color: Optional[str],
        font_family: Optional[str],
        text_size_mode: Optional[str],
        font_size: Optional[int],
        text_padding: Optional[int],
        border_radius: Optional[int],
        max_line_length: Optional[int],
    ) -> BlockResult:
        if len(predictions) == 0:
            return {
                OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                    origin_image_data=image,
                    numpy_image=(
                        image.numpy_image.copy() if copy_image else image.numpy_image
                    ),
                )
            }

        height, width = image.numpy_image.shape[:2]
        effective_font_size = compute_adaptive_rich_font_size(
            height,
            width,
            manual_font_size=font_size,
            text_size_mode=text_size_mode or TEXT_SIZE_MODE_MANUAL,
        )

        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            text_position,
            text_color,
            font_family,
            effective_font_size,
            text_padding,
            border_radius,
            max_line_length,
        )
        labels = build_detection_labels(predictions, text)

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
            labels=labels,
        )

        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }

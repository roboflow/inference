import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.core_steps.visualizations.text_display.utils import (
    TextLayout,
    calculate_relative_position,
    compute_layout,
    draw_background,
    draw_text_lines,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    STRING_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

TYPE: str = "roboflow_core/text_display@v1"

PARAMETER_REGEX = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")

SHORT_DESCRIPTION = (
    "Display customizable text on an image with styling and positioning options."
)

LONG_DESCRIPTION = """
The **Text Display** block renders text on an image with full control over styling and positioning.

### Dynamic Text Content

Text content can be parameterized with workflow execution outcomes using the same templating syntax 
as Email and SMS notification blocks:

```
text = "Detected {{ $parameters.count }} objects of class {{ $parameters.class_name }}"
```

Parameters are provided via the `text_parameters` field:

```
text_parameters = {
    "count": "$steps.model.predictions",
    "class_name": "$inputs.target_class"
}
```

You can apply transformations to parameters using `text_parameters_operations`:

```
text_parameters_operations = {
    "count": [{"type": "SequenceLength"}]
}
```

### Styling Options

- **text_color**: Color of the text. Supports:
  - Supervision color names (uppercase): "WHITE", "BLACK", "RED", "GREEN", "BLUE", "YELLOW", "ROBOFLOW", etc.
  - Hex format: "#RRGGBB" (e.g., "#FF0000" for red)
  - RGB format: "rgb(R, G, B)" (e.g., "rgb(255, 0, 0)" for red)
  - BGR format: "bgr(B, G, R)" (e.g., "bgr(0, 0, 255)" for red)
- **background_color**: Background color behind the text. Supports the same color formats as `text_color`. Use "transparent" for no background.
- **background_opacity**: Transparency of the background (0.0 = fully transparent, 1.0 = fully opaque)
- **font_scale**: Scale factor for the font size
- **font_thickness**: Thickness of the text strokes
- **padding**: Padding around the text in pixels
- **text_align**: Horizontal text alignment ("left", "center", "right")
- **border_radius**: Radius for rounded corners on the background

### Positioning Options

The block supports both absolute and relative positioning:

**Absolute Positioning** (`position_mode = "absolute"`):
- `position_x`: X coordinate in pixels from the left edge
- `position_y`: Y coordinate in pixels from the top edge

**Relative Positioning** (`position_mode = "relative"`):
- `anchor`: Where to anchor the text ("center", "top_left", "top_center", "top_right", 
  "bottom_left", "bottom_center", "bottom_right", "center_left", "center_right")
- `offset_x`: Horizontal offset from the anchor point (positive = right)
- `offset_y`: Vertical offset from the anchor point (positive = down)
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Text Display",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-font",
                "blockPriority": 3,
            },
        }
    )
    type: Literal[f"{TYPE}"]

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The image to display text on.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    text: str = Field(
        description="The text content to display. Supports parameter interpolation using {{ $parameters.name }} syntax.",
        examples=[
            "Detection count: {{ $parameters.count }}",
            "Hello World",
        ],
        json_schema_extra={
            "multiline": True,
            "always_visible": True,
        },
    )

    text_parameters: Dict[
        str,
        Union[Selector(), Selector(), str, int, float, bool],
    ] = Field(
        description="Parameters to interpolate into the text.",
        examples=[
            {
                "count": "$steps.model.predictions",
                "class_name": "$inputs.target_class",
            }
        ],
        default_factory=dict,
        json_schema_extra={
            "always_visible": True,
        },
    )

    text_parameters_operations: Dict[str, List[AllOperationsType]] = Field(
        description="Operations to apply to text parameters before interpolation.",
        examples=[
            {
                "count": [{"type": "SequenceLength"}],
            }
        ],
        default_factory=dict,
    )

    # Styling options
    text_color: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="WHITE",
        description=(
            "Color of the text. Supports supervision color names (WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ROBOFLOW, etc.), "
            "hex format (#RRGGBB), rgb(R,G,B) format, or bgr(B,G,R) format."
        ),
        examples=[
            "WHITE",
            "ROBOFLOW",
            "#FF0000",
            "rgb(255,128,0)",
            "bgr(0,0,255)",
            "$inputs.text_color",
        ],
    )

    background_color: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="BLACK",
        description=(
            "Background color behind the text. Supports the same color formats as text_color. "
            "Use 'transparent' for no background."
        ),
        examples=[
            "BLACK",
            "transparent",
            "#000000",
            "rgb(0,0,0)",
            "bgr(0,0,0)",
            "$inputs.bg_color",
        ],
    )

    background_opacity: Union[
        FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=1.0,
        description="Opacity of the background (0.0 = fully transparent, 1.0 = fully opaque).",
        examples=[1.0, 0.5, 0.0, "$inputs.bg_opacity"],
    )

    font_scale: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=1.0,
        description="Scale factor for the font size.",
        examples=[1.0, 2.0, "$inputs.font_scale"],
        ge=0.1,
        le=10.0,
    )

    font_thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=2,
        description="Thickness of the text strokes.",
        examples=[1, 2, 3, "$inputs.font_thickness"],
        ge=1,
        le=10,
    )

    padding: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=10,
        description="Padding around the text in pixels.",
        examples=[5, 10, 20, "$inputs.padding"],
        ge=0,
    )

    text_align: Union[
        Literal["left", "center", "right"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="left",
        description="Horizontal alignment of the text within its bounding box.",
        examples=["left", "center", "right"],
    )

    border_radius: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="Radius for rounded corners on the background rectangle.",
        examples=[0, 5, 10, "$inputs.border_radius"],
        ge=0,
    )

    # Positioning options
    position_mode: Union[
        Literal["absolute", "relative"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="relative",
        description="Positioning mode: 'absolute' uses exact pixel coordinates, 'relative' uses anchor points with offsets.",
        examples=["absolute", "relative"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    # Absolute positioning
    position_x: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="X coordinate (pixels from left edge) when using absolute positioning.",
        examples=[10, 100, "$inputs.x"],
        json_schema_extra={
            "relevant_for": {
                "position_mode": {"values": ["absolute"]},
            },
        },
    )

    position_y: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="Y coordinate (pixels from top edge) when using absolute positioning.",
        examples=[10, 100, "$inputs.y"],
        json_schema_extra={
            "relevant_for": {
                "position_mode": {"values": ["absolute"]},
            },
        },
    )

    # Relative positioning
    anchor: Union[
        Literal[
            "center",
            "top_left",
            "top_center",
            "top_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
            "center_left",
            "center_right",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="top_left",
        description="Anchor point for relative positioning.",
        examples=["center", "top_left", "bottom_right"],
        json_schema_extra={
            "relevant_for": {
                "position_mode": {"values": ["relative"]},
            },
        },
    )

    offset_x: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=10,
        description="Horizontal offset from anchor point (positive = right).",
        examples=[0, 10, -20, "$inputs.offset_x"],
        json_schema_extra={
            "relevant_for": {
                "position_mode": {"values": ["relative"]},
            },
        },
    )

    offset_y: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=10,
        description="Vertical offset from anchor point (positive = down).",
        examples=[0, 10, -20, "$inputs.offset_y"],
        json_schema_extra={
            "relevant_for": {
                "position_mode": {"values": ["relative"]},
            },
        },
    )

    copy_image: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Whether to copy the input image before drawing (preserves original).",
        examples=[True, False],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class TextDisplayVisualizationBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        text: str,
        text_parameters: Dict[str, Any],
        text_parameters_operations: Dict[str, List[AllOperationsType]],
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
        copy_image: bool,
    ) -> BlockResult:
        formatted_text = format_text_with_parameters(
            text=text,
            text_parameters=text_parameters,
            text_parameters_operations=text_parameters_operations,
        )
        output_image = image.numpy_image.copy() if copy_image else image.numpy_image

        text_color_bgr = str_to_color(text_color).as_bgr()
        bg_color_bgr = (
            None
            if background_color.lower() == "transparent"
            else str_to_color(background_color).as_bgr()
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        img_h, img_w = output_image.shape[:2]

        layout = compute_layout(
            formatted_text=formatted_text,
            font=font,
            font_scale=font_scale,
            font_thickness=font_thickness,
            padding=padding,
            position_mode=position_mode,
            position_x=position_x,
            position_y=position_y,
            anchor=anchor,
            offset_x=offset_x,
            offset_y=offset_y,
            img_w=img_w,
            img_h=img_h,
        )

        # Background bounds (clipped)
        x1 = max(0, layout.box_x)
        y1 = max(0, layout.box_y)
        x2 = min(img_w, layout.box_x + layout.box_w)
        y2 = min(img_h, layout.box_y + layout.box_h)

        draw_background(
            output_image,
            x1,
            y1,
            x2,
            y2,
            bg_color_bgr,
            background_opacity,
            border_radius,
        )
        draw_text_lines(
            output_image,
            layout=layout,
            padding=padding,
            text_align=text_align,
            font=font,
            font_scale=font_scale,
            font_thickness=font_thickness,
            color_bgr=text_color_bgr,
        )

        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=output_image
            )
        }


def format_text_with_parameters(
    text: str,
    text_parameters: Dict[str, Any],
    text_parameters_operations: Dict[str, List[AllOperationsType]],
) -> str:
    """Format text by replacing parameter placeholders with actual values.

    Uses a single-pass regex substitution for efficiency and correctness.
    """
    # Cache for computed parameter values (with operations applied)
    computed_values: Dict[str, str] = {}

    def replace_placeholder(match: re.Match) -> str:
        parameter_name = match.group(2)
        if parameter_name not in text_parameters:
            return match.group(0)
        if parameter_name in computed_values:
            return computed_values[parameter_name]

        parameter_value = text_parameters[parameter_name]
        operations = text_parameters_operations.get(parameter_name)
        if operations:
            operations_chain = build_operations_chain(operations=operations)
            parameter_value = operations_chain(parameter_value, global_parameters={})

        # Cache and return
        computed_values[parameter_name] = str(parameter_value)
        return computed_values[parameter_name]

    return PARAMETER_REGEX.sub(replace_placeholder, text)

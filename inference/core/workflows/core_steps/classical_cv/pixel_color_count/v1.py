from typing import List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    RGB_COLOR_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Count the number of pixels that match a specific color within a given tolerance."
)


class ColorPixelCountManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/pixel_color_count@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Pixel Color Count",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": SHORT_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
        }
    )
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    target_color: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        StepOutputSelector(kind=[RGB_COLOR_KIND]),
        str,
        Tuple[int, int, int],
    ] = Field(
        description="Target color to count in the image. Can be a hex string "
        "(like '#431112') RGB string (like '(128, 32, 64)') or a RGB tuple "
        "(like (18, 17, 67)).",
        examples=["#431112", "$inputs.target_color", (18, 17, 67)],
    )
    tolerance: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        default=10,
        description="Tolerance for color matching.",
        examples=[10, "$inputs.tolerance"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="matching_pixels_count",
                kind=[INTEGER_KIND],
            ),
        ]


class PixelationCountBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[ColorPixelCountManifest]:
        return ColorPixelCountManifest

    def run(
        self,
        image: WorkflowImageData,
        target_color: Union[str, tuple],
        tolerance: int,
    ) -> BlockResult:
        color_pixel_count = count_specific_color_pixels(
            image.numpy_image, target_color, tolerance
        )
        return {"matching_pixels_count": color_pixel_count}


def count_specific_color_pixels(
    image: np.ndarray,
    target_color: Union[str, Tuple[int, int, int]],
    tolerance: int,
) -> int:
    """
    Counts the number of pixels that match the target color within the given tolerance.

    Args:
        image: Input image.
        target_color (Union[str, tuple]): Target color in hex format (e.g., '#431112') or BGR tuple (e.g., (18, 17, 67)).
        tolerance (int, optional): Tolerance for color matching. Defaults to 10.

    Returns:
        int: Number of pixels that match the target color.
    """
    target_color_bgr = convert_color_to_bgr_tuple(color=target_color)
    lower_bound = np.array(
        [
            target_color_bgr[0] - tolerance,
            target_color_bgr[1] - tolerance,
            target_color_bgr[2] - tolerance,
        ]
    )
    upper_bound = np.array(
        [
            target_color_bgr[0] + tolerance,
            target_color_bgr[1] + tolerance,
            target_color_bgr[2] + tolerance,
        ]
    )
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return int(np.sum(mask > 0))


def convert_color_to_bgr_tuple(
    color: Union[str, Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    if isinstance(color, str):
        return convert_string_color_to_bgr_tuple(color=color)
    if isinstance(color, tuple) and len(color) == 3:
        return color[::-1]
    raise ValueError(f"Invalid color format: {color}")


def convert_string_color_to_bgr_tuple(color: str) -> Tuple[int, int, int]:
    if color.startswith("#") and len(color) == 7:
        try:
            return tuple(int(color[i : i + 2], 16) for i in (5, 3, 1))
        except ValueError as e:
            raise ValueError(f"Invalid hex color format: {color}") from e
    if color.startswith("#") and len(color) == 4:
        try:
            return tuple(int(color[i] + color[i], 16) for i in (3, 2, 1))
        except ValueError as e:
            raise ValueError(f"Invalid hex color format: {color}") from e
    if color.startswith("(") and color.endswith(")"):
        try:
            return tuple(map(int, color[1:-1].split(",")))[::-1]
        except ValueError as e:
            raise ValueError(f"Invalid tuple color format: {color}") from e
    raise ValueError(f"Invalid hex color format: {color}")

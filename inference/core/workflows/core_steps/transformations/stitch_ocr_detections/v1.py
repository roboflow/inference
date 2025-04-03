from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, field_validator

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Combines OCR detection results into a coherent text string by organizing detections spatially. 
This transformation is perfect for turning individual OCR results into structured, readable text!

#### How It Works

This transformation reconstructs the original text from OCR detection results by:

1. 📐 **Grouping** text detections into rows based on their vertical (`y`) positions

2. 📏 **Sorting** detections within each row by horizontal (`x`) position

3. 📜 **Concatenating** the text in reading order (left-to-right, top-to-bottom)

#### Parameters

- **`tolerance`**: Controls how close detections need to be vertically to be considered part of the same line of text. 
A higher tolerance will group detections that are further apart vertically.

- **`reading_direction`**: Determines the order in which text is read. Available options:
  
    * **"left_to_right"**: Standard left-to-right reading (e.g., English) ➡️
  
    * **"right_to_left"**: Right-to-left reading (e.g., Arabic) ⬅️
  
    * **"vertical_top_to_bottom"**: Vertical reading from top to bottom ⬇️
  
    * **"vertical_bottom_to_top"**: Vertical reading from bottom to top ⬆️

    * **"auto"**: Automatically detects the reading direction based on the spatial arrangement of text elements.

#### Why Use This Transformation?

This is especially useful for:

- 📖 Converting individual character/word detections into a readable text block

- 📝 Reconstructing multi-line text from OCR results

- 🔀 Maintaining proper reading order for detected text elements

- 🌏 Supporting different writing systems and text orientations

#### Example Usage

Use this transformation after an OCR model that outputs individual words or characters, so you can reconstruct the 
original text layout in its intended format.
"""

SHORT_DESCRIPTION = "Combines OCR detection results into a coherent text string by organizing detections spatially."


class ReadingDirection(str, Enum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    VERTICAL_TOP_TO_BOTTOM = "vertical_top_to_bottom"
    VERTICAL_BOTTOM_TO_TOP = "vertical_bottom_to_top"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stitch OCR Detections",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-reel",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/stitch_ocr_detections@v1"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="OCR Detections",
        description="The output of an OCR detection model.",
        examples=["$steps.my_ocr_detection_model.predictions"],
    )
    reading_direction: Literal[
        "left_to_right",
        "right_to_left",
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
        "auto",
    ] = Field(
        title="Reading Direction",
        description="The direction of the text in the image.",
        examples=["right_to_left"],
        json_schema_extra={
            "values_metadata": {
                "left_to_right": {
                    "name": "Left To Right",
                    "description": "Standard left-to-right reading (e.g., English language)",
                },
                "right_to_left": {
                    "name": "Right To Left",
                    "description": "Right-to-left reading (e.g., Arabic)",
                },
                "vertical_top_to_bottom": {
                    "name": "Top To Bottom (Vertical)",
                    "description": "Vertical reading from top to bottom",
                },
                "vertical_bottom_to_top": {
                    "name": "Bottom To Top (Vertical)",
                    "description": "Vertical reading from bottom to top",
                },
                "auto": {
                    "name": "Auto",
                    "description": "Automatically detect the reading direction based on text arrangement.",
                },
            }
        },
    )
    tolerance: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        title="Tolerance",
        description="The tolerance for grouping detections into the same line of text.",
        default=10,
        examples=[10, "$inputs.tolerance"],
    )

    @field_validator("tolerance")
    @classmethod
    def ensure_tolerance_greater_than_zero(
        cls, value: Union[int, str]
    ) -> Union[int, str]:
        if isinstance(value, int) and value <= 0:
            raise ValueError(
                "Stitch OCR detections block expects `tollerance` to be greater than zero."
            )
        return value

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="ocr_text", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


def detect_reading_direction(detections: sv.Detections) -> str:
    if len(detections) == 0:
        return "left_to_right"

    xyxy = detections.xyxy
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]

    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    if avg_width > avg_height:
        return "left_to_right"
    else:
        return "vertical_top_to_bottom"


class StitchOCRDetectionsBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        reading_direction: str,
        tolerance: int,
    ) -> BlockResult:
        if reading_direction == "auto":
            reading_direction = detect_reading_direction(predictions[0])
        return [
            stitch_ocr_detections(
                detections=detections,
                reading_direction=reading_direction,
                tolerance=tolerance,
            )
            for detections in predictions
        ]


def stitch_ocr_detections(
    detections: sv.Detections,
    reading_direction: str = "left_to_right",
    tolerance: int = 10,
) -> Dict[str, str]:
    """
    Stitch OCR detections into coherent text based on spatial arrangement.

    Args:
        detections: Supervision Detections object containing OCR results
        reading_direction: Direction to read text ("left_to_right", "right_to_left",
                         "vertical_top_to_bottom", "vertical_bottom_to_top")
        tolerance: Vertical tolerance for grouping text into lines

    Returns:
        Dict containing stitched OCR text under 'ocr_text' key
    """
    if len(detections) == 0:
        return {"ocr_text": ""}

    xyxy = detections.xyxy.round().astype(dtype=int)
    class_names = detections.data["class_name"]

    # Prepare coordinates based on reading direction
    xyxy = prepare_coordinates(xyxy, reading_direction)

    # Group detections into lines
    boxes_by_line = group_detections_by_line(xyxy, reading_direction, tolerance)
    # Sort lines based on reading direction
    lines = sorted(
        boxes_by_line.keys(), reverse=reading_direction in ["vertical_bottom_to_top"]
    )

    # Build final text
    ordered_class_names = []
    for i, key in enumerate(lines):
        line_data = boxes_by_line[key]
        line_xyxy = np.array(line_data["xyxy"])
        line_idx = np.array(line_data["idx"])

        # Sort detections within line
        sort_idx = sort_line_detections(line_xyxy, reading_direction)

        # Add sorted class names for this line
        ordered_class_names.extend(class_names[line_idx[sort_idx]])

        # Add line separator if not last line
        if i < len(lines) - 1:
            ordered_class_names.append(get_line_separator(reading_direction))

    return {"ocr_text": "".join(ordered_class_names)}


def prepare_coordinates(
    xyxy: np.ndarray,
    reading_direction: str,
) -> np.ndarray:
    """Prepare coordinates based on reading direction."""
    if reading_direction in ["vertical_top_to_bottom", "vertical_bottom_to_top"]:
        # Swap x and y coordinates: [x1,y1,x2,y2] -> [y1,x1,y2,x2]
        return xyxy[:, [1, 0, 3, 2]]
    return xyxy


def group_detections_by_line(
    xyxy: np.ndarray,
    reading_direction: str,
    tolerance: int,
) -> Dict[float, Dict[str, List]]:
    """Group detections into lines based on primary coordinate."""
    # After prepare_coordinates swap, we always group by y ([:, 1])
    primary_coord = xyxy[:, 1]  # This is y for horizontal, swapped x for vertical

    # Round primary coordinate to group into lines
    rounded_primary = np.round(primary_coord / tolerance) * tolerance

    boxes_by_line = {}
    # Group bounding boxes and associated indices by line
    for i, (bbox, line_pos) in enumerate(zip(xyxy, rounded_primary)):
        if line_pos not in boxes_by_line:
            boxes_by_line[line_pos] = {"xyxy": [bbox], "idx": [i]}
        else:
            boxes_by_line[line_pos]["xyxy"].append(bbox)
            boxes_by_line[line_pos]["idx"].append(i)

    return boxes_by_line


def sort_line_detections(
    line_xyxy: np.ndarray,
    reading_direction: str,
) -> np.ndarray:
    """Sort detections within a line based on reading direction."""
    # After prepare_coordinates swap, we always sort by x ([:, 0])
    if reading_direction in ["left_to_right", "vertical_top_to_bottom"]:
        return line_xyxy[:, 0].argsort()  # Sort by x1 (original x or swapped y)
    else:  # right_to_left or vertical_bottom_to_top
        return (-line_xyxy[:, 0]).argsort()  # Sort by -x1 (original -x or swapped -y)


def get_line_separator(reading_direction: str) -> str:
    """Get the appropriate separator based on reading direction."""
    return "\n" if reading_direction in ["left_to_right", "right_to_left"] else " "

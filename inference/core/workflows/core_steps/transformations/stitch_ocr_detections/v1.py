from enum import Enum
from typing import Dict, List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field, field_validator

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
Combine individual OCR detection results (words, characters, or text regions) into coherent text strings by organizing detections spatially according to reading direction, grouping detections into lines, sorting them within lines, and concatenating text in proper reading order to reconstruct readable text from OCR model outputs.

## How This Block Works

This block reconstructs readable text from individual OCR detections by organizing them spatially and concatenating text in proper reading order. The block:

1. Receives OCR detection predictions containing individual text detections with bounding boxes and class names (text content)
2. Prepares coordinates based on reading direction:
   - For vertical reading directions, swaps x and y coordinates to enable vertical line processing
   - For horizontal reading directions, uses coordinates as-is
3. Groups detections into lines:
   - Groups detections based on vertical position (or horizontal position for vertical text) using the tolerance parameter
   - Detections within the tolerance distance are considered part of the same line
   - Higher tolerance values group detections that are further apart, useful for text with variable line spacing
4. Sorts lines based on reading direction:
   - For left-to-right and vertical top-to-bottom: sorts lines from top to bottom
   - For right-to-left and vertical bottom-to-top: sorts lines in reverse order (bottom to top)
5. Sorts detections within each line:
   - For left-to-right and vertical top-to-bottom: sorts detections by horizontal position (left to right, or top to bottom for vertical)
   - For right-to-left and vertical bottom-to-top: sorts detections in reverse order (right to left, or bottom to top for vertical)
6. Concatenates text in reading order:
   - Extracts class names (text content) from detections in sorted order
   - Adds line separators (newline for horizontal text, space for vertical text) between lines
   - Optionally inserts a delimiter between each text element if specified
   - Produces a single coherent text string with proper reading order
7. Handles automatic reading direction detection (if "auto" is selected):
   - Analyzes average width and height of detection bounding boxes
   - If average width > average height: detects horizontal text (left-to-right)
   - If average height >= average width: detects vertical text (top-to-bottom)
8. Returns the stitched text string:
   - Outputs a single text string under the `ocr_text` key
   - Text is formatted with proper line breaks and spacing according to reading direction

The block enables reconstruction of multi-line text from individual OCR detections, maintaining proper reading order for different languages and writing systems. It handles both horizontal (left-to-right, right-to-left) and vertical (top-to-bottom, bottom-to-top) text orientations, making it useful for processing text in various languages and formats.

## Common Use Cases

- **Text Reconstruction**: Convert individual word or character detections from OCR models into readable text blocks (e.g., reconstruct documents from word detections, combine character detections into words, stitch OCR results into paragraphs), enabling text reconstruction workflows
- **Multi-Line Text Processing**: Reconstruct multi-line text from OCR results with proper line breaks and formatting (e.g., extract paragraphs from OCR results, reconstruct formatted text, process multi-line documents), enabling multi-line text workflows
- **Multi-Language OCR**: Process OCR results from different languages and writing systems (e.g., process Arabic right-to-left text, handle vertical Chinese/Japanese text, support multiple reading directions), enabling multi-language OCR workflows
- **Document Processing**: Extract and reconstruct text from documents and images (e.g., extract text from scanned documents, process invoice text, extract text from forms), enabling document processing workflows
- **Text Extraction and Formatting**: Extract text from images and format it for downstream use (e.g., extract text for database storage, format text for API responses, prepare text for analysis), enabling text extraction workflows
- **OCR Result Post-Processing**: Post-process OCR model outputs to produce usable text strings (e.g., format OCR outputs, organize OCR results, prepare text for downstream blocks), enabling OCR post-processing workflows

## Connecting to Other Blocks

This block receives OCR detection predictions and produces stitched text strings:

- **After OCR model blocks** to convert detection results into readable text (e.g., OCR model to text string, OCR detections to formatted text, OCR results to text output), enabling OCR-to-text workflows
- **Before data storage blocks** to store extracted text (e.g., store OCR text in databases, save extracted text, log OCR results), enabling text storage workflows
- **Before notification blocks** to send extracted text in notifications (e.g., send OCR text in alerts, include extracted text in messages, notify with OCR results), enabling text notification workflows
- **Before text processing blocks** to process stitched text (e.g., process text with NLP models, analyze extracted text, apply text transformations), enabling text processing workflows
- **Before API output blocks** to provide text in API responses (e.g., return OCR text in API, format text for responses, provide extracted text output), enabling text output workflows
- **In workflow outputs** to provide stitched text as final output (e.g., text extraction workflows, OCR output workflows, document processing workflows), enabling text output workflows

## Requirements

This block requires OCR detection predictions (object detection format) with bounding boxes and class names containing text content. The `tolerance` parameter must be greater than zero and controls the vertical (or horizontal for vertical text) distance threshold for grouping detections into lines. The `reading_direction` parameter supports five modes: "left_to_right" (standard horizontal), "right_to_left" (Arabic-style), "vertical_top_to_bottom" (vertical), "vertical_bottom_to_top" (vertical reversed), and "auto" (automatic detection based on bounding box dimensions). The `delimiter` parameter is optional and inserts a delimiter between each text element (empty string by default, meaning no delimiter). The block outputs a single text string under the `ocr_text` key.
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
        description="OCR detection predictions from an OCR model. Should contain bounding boxes and class names with text content. Each detection represents a word, character, or text region that will be stitched together into coherent text. Supports object detection format with bounding boxes (xyxy) and class names in the data dictionary.",
        examples=["$steps.ocr_model.predictions", "$steps.my_ocr_detection_model.predictions"],
    )
    reading_direction: Literal[
        "left_to_right",
        "right_to_left",
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
        "auto",
    ] = Field(
        title="Reading Direction",
        description="Direction to read and organize text detections. 'left_to_right': Standard horizontal reading (English, most languages). 'right_to_left': Right-to-left reading (Arabic, Hebrew). 'vertical_top_to_bottom': Vertical reading from top to bottom (Traditional Chinese, Japanese). 'vertical_bottom_to_top': Vertical reading from bottom to top (rare vertical formats). 'auto': Automatically detects reading direction based on average bounding box dimensions (width > height = horizontal, height >= width = vertical). Determines how detections are grouped into lines and sorted within lines.",
        examples=["left_to_right", "right_to_left", "auto"],
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
        description="Vertical (or horizontal for vertical text) distance threshold in pixels for grouping detections into the same line. Detections within this tolerance distance are grouped into the same line. Higher values group detections that are further apart (useful for text with variable line spacing or slanted text). Lower values create more lines (useful for tightly spaced text). Must be greater than zero.",
        default=10,
        examples=[10, 20, 5],
    )
    delimiter: Union[str, Selector(kind=[STRING_KIND])] = Field(
        title="Delimiter",
        description="Optional delimiter string to insert between each text element (word/character) when stitching. Empty string (default) means no delimiter - text elements are concatenated directly. Useful for adding spaces between words, commas between elements, or custom separators. Example: use ' ' (space) to add spaces between words, or ',' to add commas.",
        default="",
        examples=["", " ", ",", "-"],
    )

    @field_validator("tolerance")
    @classmethod
    def ensure_tolerance_greater_than_zero(
        cls, value: Union[int, str]
    ) -> Union[int, str]:
        if isinstance(value, int) and value <= 0:
            raise ValueError(
                "Stitch OCR detections block expects `tolerance` to be greater than zero."
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
        delimiter: str = "",
    ) -> BlockResult:
        if reading_direction == "auto":
            reading_direction = detect_reading_direction(predictions[0])
        return [
            stitch_ocr_detections(
                detections=detections,
                reading_direction=reading_direction,
                tolerance=tolerance,
                delimiter=delimiter,
            )
            for detections in predictions
        ]


def stitch_ocr_detections(
    detections: sv.Detections,
    reading_direction: str = "left_to_right",
    tolerance: int = 10,
    delimiter: str = "",
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

    return {"ocr_text": delimiter.join(ordered_class_names)}


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

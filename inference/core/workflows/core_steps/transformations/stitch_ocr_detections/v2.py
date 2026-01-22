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
    FLOAT_KIND,
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
Combine individual OCR detection results (words, characters, or text regions) into coherent text strings by organizing detections spatially, grouping them into lines, and concatenating text in proper reading order.

## Stitching Algorithms

This block supports three algorithms for reconstructing text from OCR detections:

### Tolerance-based (default)
Groups detections into lines using a fixed pixel tolerance. Detections within the tolerance distance vertically (or horizontally for vertical text) are grouped into the same line, then sorted by position within each line.

- **Best for**: Consistent font sizes and well-aligned horizontal/vertical text
- **Parameters**: `tolerance` (pixel threshold for line grouping)

### Otsu Thresholding
Uses Otsu's method on normalized gap distances to automatically find the optimal threshold separating character gaps from word gaps. Gaps are normalized by local character width, making it resolution-invariant.

- **Best for**: Variable font sizes, automatic word boundary detection
- **Parameters**: `otsu_threshold_multiplier` (adjust threshold sensitivity)
- **Key feature**: Detects bimodal distributions to distinguish single words from multi-word text

### Collimate (Skewed Text)
Uses greedy parent-child traversal to follow text flow. Starting from the first detection, it finds subsequent detections that "follow" in reading order (similar alignment + correct direction), building lines through traversal rather than bucketing.

- **Best for**: Skewed, curved, or non-axis-aligned text
- **Parameters**: `collimate_tolerance` (alignment tolerance in pixels)
- **Note**: Does not detect word boundaries - use `delimiter` parameter if spacing is needed

## Reading Directions

All algorithms support multiple reading directions:
- `left_to_right`: Standard horizontal (English, most languages)
- `right_to_left`: Right-to-left (Arabic, Hebrew)
- `vertical_top_to_bottom`: Vertical top-to-bottom (Traditional Chinese, Japanese)
- `vertical_bottom_to_top`: Vertical bottom-to-top
- `auto`: Automatically detect based on bounding box dimensions

## Common Use Cases

- **Document OCR**: Reconstruct paragraphs and lines from character/word detections
- **Multi-language support**: Handle different reading directions and writing systems
- **Skewed text processing**: Use collimate algorithm for tilted or curved text
- **Word detection**: Use Otsu algorithm to automatically insert spaces between words
"""

SHORT_DESCRIPTION = "Combines OCR detection results into a coherent text string by organizing detections spatially."


class ReadingDirection(str, Enum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    VERTICAL_TOP_TO_BOTTOM = "vertical_top_to_bottom"
    VERTICAL_BOTTOM_TO_TOP = "vertical_bottom_to_top"


class StitchingAlgorithm(str, Enum):
    """Algorithm for grouping detections into words/lines.

    TOLERANCE: Uses fixed pixel tolerance for line grouping (original algorithm).
        Good for consistent font sizes and line spacing.

    OTSU: Uses Otsu's method on normalized gaps to find natural breaks.
        Resolution-invariant and works well with bimodal distributions
        (e.g., character-level vs word-level spacing).

    COLLIMATE: Uses greedy parent-child traversal to group detections.
        Good for skewed or curved text where bucket-based approaches fail.
    """

    TOLERANCE = "tolerance"
    OTSU = "otsu"
    COLLIMATE = "collimate"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stitch OCR Detections",
            "version": "v2",
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
    type: Literal["roboflow_core/stitch_ocr_detections@v2"]
    stitching_algorithm: Literal["tolerance", "otsu", "collimate"] = Field(
        title="Stitching Algorithm",
        description="Algorithm for grouping detections into words/lines. 'tolerance': Uses fixed pixel tolerance for line grouping (original algorithm). Good for consistent font sizes and line spacing. 'otsu': Uses Otsu's method on normalized gaps to find natural breaks between words. Resolution-invariant and works well with bimodal gap distributions. 'collimate': Uses greedy parent-child traversal to group detections. Good for skewed or curved text where bucket-based approaches fail.",
        examples=["tolerance", "otsu", "collimate"],
        json_schema_extra={
            "values_metadata": {
                "tolerance": {
                    "name": "Tolerance-based",
                    "description": "Uses fixed pixel tolerance for line grouping. Good for consistent font sizes.",
                },
                "otsu": {
                    "name": "Otsu Thresholding",
                    "description": "Resolution-invariant algorithm using normalized gaps. Works well with varying font sizes.",
                },
                "collimate": {
                    "name": "Collimate (Skewed Text)",
                    "description": "Greedy parent-child traversal for grouping. Best for skewed or curved text.",
                },
            }
        },
    )
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="OCR Detections",
        description="OCR detection predictions from an OCR model. Should contain bounding boxes and class names with text content. Each detection represents a word, character, or text region that will be stitched together into coherent text. Supports object detection format with bounding boxes (xyxy) and class names in the data dictionary.",
        examples=[
            "$steps.ocr_model.predictions",
            "$steps.my_ocr_detection_model.predictions",
        ],
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
        json_schema_extra={
            "relevant_for": {
                "stitching_algorithm": {
                    "values": ["tolerance"],
                    "required": True,
                },
            },
        },
    )
    delimiter: Union[str, Selector(kind=[STRING_KIND])] = Field(
        title="Delimiter",
        description="Optional delimiter string to insert between each text element (word/character) when stitching. Empty string (default) means no delimiter - text elements are concatenated directly. Useful for adding spaces between words, commas between elements, or custom separators. Example: use ' ' (space) to add spaces between words, or ',' to add commas.",
        default="",
        examples=["", " ", ",", "-"],
    )
    otsu_threshold_multiplier: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        title="Otsu Threshold Multiplier",
        description="Multiplier applied to the Otsu-computed threshold when using the 'otsu' stitching algorithm. Values > 1.0 make word breaks less frequent (more conservative, fewer splits), values < 1.0 make word breaks more frequent (more aggressive, more splits). Default is 1.0 (use Otsu threshold as-is). Try 1.3-1.5 if words are being incorrectly split, or 0.7-0.9 if words are being incorrectly merged.",
        default=1.0,
        examples=[1.0, 1.3, 1.5, 0.8],
        json_schema_extra={
            "relevant_for": {
                "stitching_algorithm": {
                    "values": ["otsu"],
                    "required": True,
                },
            },
        },
    )
    collimate_tolerance: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        title="Collimate Tolerance",
        description="Pixel tolerance for the 'collimate' stitching algorithm. Controls how much vertical (for horizontal text) or horizontal (for vertical text) deviation is allowed when determining if a detection follows another in reading order. Higher values handle more skewed text but may incorrectly merge separate lines. Default is 10 pixels.",
        default=10,
        examples=[5, 10, 15, 20],
        json_schema_extra={
            "relevant_for": {
                "stitching_algorithm": {
                    "values": ["collimate"],
                    "required": True,
                },
            },
        },
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


class StitchOCRDetectionsBlockV2(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        stitching_algorithm: str,
        reading_direction: str,
        tolerance: int,
        delimiter: str = "",
        otsu_threshold_multiplier: float = 1.0,
        collimate_tolerance: int = 10,
    ) -> BlockResult:
        if reading_direction == "auto":
            reading_direction = detect_reading_direction(predictions[0])

        if stitching_algorithm == "otsu":
            return [
                adaptive_word_grouping(
                    detections=detections,
                    reading_direction=reading_direction,
                    delimiter=delimiter,
                    threshold_multiplier=otsu_threshold_multiplier,
                )
                for detections in predictions
            ]
        elif stitching_algorithm == "collimate":
            return [
                collimate_word_grouping(
                    detections=detections,
                    reading_direction=reading_direction,
                    delimiter=delimiter,
                    tolerance=collimate_tolerance,
                )
                for detections in predictions
            ]
        else:
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


def find_otsu_threshold(gaps: np.ndarray) -> tuple[float, bool]:
    """Find natural break between intra-word and inter-word gaps using Otsu's method.

    This is a resolution-invariant approach that finds the optimal threshold
    to separate two classes of gaps (e.g., gaps within words vs gaps between words).

    Also detects whether the distribution is bimodal (two distinct groups) or
    unimodal (single group, suggesting single word or uniform spacing).

    Args:
        gaps: Array of normalized gap values

    Returns:
        Tuple of (threshold, is_bimodal):
        - threshold: Optimal threshold value that maximizes between-class variance
        - is_bimodal: True if distribution appears bimodal, False if unimodal
    """
    if len(gaps) < 2:
        return 0.0, False

    # Create histogram of gaps
    hist, bin_edges = np.histogram(gaps, bins=min(50, len(gaps)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    best_thresh = 0.0
    best_variance = 0.0
    best_below_mean = 0.0
    best_above_mean = 0.0

    for t in bin_centers:
        below = gaps[gaps <= t]
        above = gaps[gaps > t]

        if len(below) == 0 or len(above) == 0:
            continue

        # Between-class variance (Otsu's criterion)
        variance = len(below) * len(above) * (below.mean() - above.mean()) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh = t
            best_below_mean = below.mean()
            best_above_mean = above.mean()

    # Check if distribution is bimodal using several heuristics:
    # 1. The gap between class means should be significant relative to overall spread
    # 2. There should be meaningful absolute separation between classes

    overall_std = gaps.std()
    overall_mean = gaps.mean()

    # Separation ratio: how far apart are the two class means relative to overall std
    mean_separation = abs(best_above_mean - best_below_mean)
    separation_ratio = mean_separation / overall_std if overall_std > 0 else 0

    # Bimodality criteria - MUST have meaningful word gaps (not just outliers):
    # The key insight is that real word gaps are typically 0.5+ in normalized units.
    # A distribution with all gaps < 0.3 is unimodal (single word), even if there
    # are outliers (like overlapping characters with negative gaps) that inflate
    # the mean separation.
    #
    # Primary criterion: above-class mean must indicate actual word gaps exist
    has_positive_word_gaps = (
        best_above_mean > 0.3
    )  # Word gaps should be clearly positive

    # Secondary criterion: if we have good separation AND positive gaps
    has_good_relative_separation = separation_ratio > 1.5 and mean_separation > 0.3

    # Must have positive word gaps to be considered bimodal
    is_bimodal = has_positive_word_gaps and (
        mean_separation > 0.3 or has_good_relative_separation
    )

    return best_thresh, is_bimodal


def adaptive_word_grouping(
    detections: sv.Detections,
    reading_direction: str,
    delimiter: str = "",
    threshold_multiplier: float = 1.0,
) -> Dict[str, str]:
    """Stitch OCR detections using adaptive gap analysis with Otsu thresholding.

    This approach is resolution-invariant because it normalizes gaps by local
    character dimensions. It works well with bimodal gap distributions
    (e.g., character-level vs word-level spacing).

    The algorithm computes a global threshold across all lines to leverage
    the full dataset of gaps, which provides more robust Otsu thresholding
    than per-line computation.

    Args:
        detections: Supervision Detections object containing OCR results
        reading_direction: Direction to read text
        delimiter: String to insert between text elements
        threshold_multiplier: Multiplier applied to Otsu threshold (>1.0 = fewer word breaks, <1.0 = more word breaks)

    Returns:
        Dict containing stitched OCR text under 'ocr_text' key
    """
    if len(detections) == 0:
        return {"ocr_text": ""}

    xyxy = detections.xyxy
    class_names = detections.data["class_name"]

    # Determine if we're working with vertical text
    is_vertical = reading_direction in [
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
    ]

    # For vertical text, swap x/y for processing
    if is_vertical:
        # Swap coordinates: treat y as x for sorting
        x_centers = (xyxy[:, 1] + xyxy[:, 3]) / 2  # y becomes primary axis
        y_centers = (xyxy[:, 0] + xyxy[:, 2]) / 2  # x becomes secondary axis
        widths = xyxy[:, 3] - xyxy[:, 1]  # height becomes "width"
        heights = xyxy[:, 2] - xyxy[:, 0]  # width becomes "height"
    else:
        x_centers = (xyxy[:, 0] + xyxy[:, 2]) / 2
        y_centers = (xyxy[:, 1] + xyxy[:, 3]) / 2
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

    # First, group detections into lines based on y-coordinate clustering
    # Use adaptive threshold based on median height
    median_height = np.median(heights)
    line_tolerance = median_height * 0.5

    # Sort by y to group into lines
    y_sorted_indices = np.argsort(y_centers)

    lines = []
    current_line = [y_sorted_indices[0]]
    current_line_y = y_centers[y_sorted_indices[0]]

    for idx in y_sorted_indices[1:]:
        if abs(y_centers[idx] - current_line_y) <= line_tolerance:
            current_line.append(idx)
            # Update line y as running average
            current_line_y = np.mean([y_centers[i] for i in current_line])
        else:
            lines.append(current_line)
            current_line = [idx]
            current_line_y = y_centers[idx]
    lines.append(current_line)

    # Sort lines by y position
    line_y_positions = [np.mean([y_centers[i] for i in line]) for line in lines]
    if reading_direction in ["vertical_bottom_to_top"]:
        sorted_line_indices = np.argsort(line_y_positions)[::-1]
    else:
        sorted_line_indices = np.argsort(line_y_positions)

    # First pass: compute normalized gaps for ALL lines to get global threshold
    all_normalized_gaps = []
    line_data = []  # Store sorted line info for second pass

    for line_idx in sorted_line_indices:
        line = lines[line_idx]

        if len(line) == 1:
            line_data.append((line, None, None, None))
            continue

        # Sort detections in line by x position
        line_x_centers = x_centers[line]
        line_widths = widths[line]

        if reading_direction in ["right_to_left", "vertical_bottom_to_top"]:
            x_sorted_order = np.argsort(line_x_centers)[::-1]
        else:
            x_sorted_order = np.argsort(line_x_centers)

        sorted_line = [line[i] for i in x_sorted_order]
        sorted_x_centers = line_x_centers[x_sorted_order]
        sorted_widths = line_widths[x_sorted_order]

        # Compute normalized gaps for this line
        normalized_gaps = []
        for i in range(1, len(sorted_line)):
            prev_idx, curr_idx = i - 1, i
            # Raw gap between detection edges
            if reading_direction in ["right_to_left", "vertical_bottom_to_top"]:
                raw_gap = (
                    sorted_x_centers[prev_idx]
                    - sorted_x_centers[curr_idx]
                    - (sorted_widths[prev_idx] + sorted_widths[curr_idx]) / 2
                )
            else:
                raw_gap = (
                    sorted_x_centers[curr_idx]
                    - sorted_x_centers[prev_idx]
                    - (sorted_widths[prev_idx] + sorted_widths[curr_idx]) / 2
                )

            # Normalize by local character scale
            local_scale = (sorted_widths[prev_idx] + sorted_widths[curr_idx]) / 2
            if local_scale > 0:
                normalized_gaps.append(raw_gap / local_scale)
            else:
                normalized_gaps.append(0.0)

        normalized_gaps = np.array(normalized_gaps)
        all_normalized_gaps.extend(normalized_gaps.tolist())
        line_data.append(
            (sorted_line, sorted_x_centers, sorted_widths, normalized_gaps)
        )

    # Compute global threshold using all gaps, then apply multiplier
    all_normalized_gaps = np.array(all_normalized_gaps)
    global_threshold, is_bimodal = find_otsu_threshold(all_normalized_gaps)
    global_threshold *= threshold_multiplier

    # Second pass: use global threshold to group words
    all_text_parts = []

    for sorted_line, sorted_x_centers, sorted_widths, normalized_gaps in line_data:
        if normalized_gaps is None:
            # Single detection in line
            all_text_parts.append(class_names[sorted_line[0]])
            continue

        # If distribution is not bimodal (likely single word or uniform spacing),
        # treat all detections as a single word to avoid incorrect splitting
        if not is_bimodal:
            word_text = delimiter.join([class_names[idx] for idx in sorted_line])
            all_text_parts.append(word_text)
            continue

        # Group into words based on global threshold
        words = [[sorted_line[0]]]
        for i, det_idx in enumerate(sorted_line[1:]):
            if normalized_gaps[i] > global_threshold:
                words.append([det_idx])
            else:
                words[-1].append(det_idx)

        # Build text for this line
        line_text_parts = []
        for word in words:
            word_text = delimiter.join([class_names[idx] for idx in word])
            line_text_parts.append(word_text)

        # Join words with space (or delimiter if specified and non-empty)
        word_separator = " " if delimiter == "" else delimiter
        all_text_parts.append(word_separator.join(line_text_parts))

    # Join lines with appropriate separator
    line_separator = get_line_separator(reading_direction)
    return {"ocr_text": line_separator.join(all_text_parts)}


class CollimateDetection:
    """Helper class for collimate algorithm to store detection properties."""

    def __init__(self, xyxy: np.ndarray, class_name: str, idx: int):
        self.x = (xyxy[0] + xyxy[2]) / 2
        self.y = (xyxy[1] + xyxy[3]) / 2
        self.width = xyxy[2] - xyxy[0]
        self.height = xyxy[3] - xyxy[1]
        self.class_name = class_name
        self.idx = idx  # Original index for tracking

    def __repr__(self) -> str:
        return f"{self.class_name}"


def _detection_follows(
    parent: CollimateDetection,
    child: CollimateDetection,
    reading_direction: str,
    tolerance: int,
) -> bool:
    """Check if child detection follows parent in reading order within tolerance.

    For horizontal text: child should be roughly on the same line (similar y)
    and to the right of parent.

    For vertical text: child should be roughly in the same column (similar x)
    and below parent.

    Args:
        parent: The reference detection
        child: The detection to check
        reading_direction: Reading direction
        tolerance: Pixel tolerance for alignment

    Returns:
        True if child follows parent in reading order
    """
    is_vertical = reading_direction in [
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
    ]

    if is_vertical:
        # For vertical text, check if x-coordinates align and child is below/above
        # Use max width to handle narrow letters
        width_tolerance = max(parent.width, child.width) / 2
        x_aligned = abs(parent.x - child.x) < width_tolerance + tolerance

        if reading_direction == "vertical_top_to_bottom":
            return x_aligned and parent.y <= child.y
        else:  # vertical_bottom_to_top
            return x_aligned and parent.y >= child.y
    else:
        # For horizontal text, check if y-coordinates align and child is to the right/left
        # Use max height to handle varying letter sizes
        height_tolerance = max(parent.height, child.height) / 2
        y_aligned = abs(parent.y - child.y) < height_tolerance + tolerance

        if reading_direction == "left_to_right":
            return y_aligned and parent.x <= child.x
        else:  # right_to_left
            return y_aligned and parent.x >= child.x


def _sort_detections_for_collimate(
    detections: List[CollimateDetection],
    reading_direction: str,
) -> List[CollimateDetection]:
    """Sort detections by primary reading coordinate."""
    is_vertical = reading_direction in [
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
    ]

    if is_vertical:
        # Sort by y for vertical text
        reverse = reading_direction == "vertical_bottom_to_top"
        return sorted(detections, key=lambda d: d.y, reverse=reverse)
    else:
        # Sort by x for horizontal text
        reverse = reading_direction == "right_to_left"
        return sorted(detections, key=lambda d: d.x, reverse=reverse)


def _get_line_avg_coord(
    line: List[CollimateDetection],
    reading_direction: str,
) -> float:
    """Get average coordinate for sorting lines."""
    if len(line) == 0:
        return 0.0

    is_vertical = reading_direction in [
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
    ]

    if is_vertical:
        # For vertical text, lines are columns - sort by x
        return sum(d.x for d in line) / len(line)
    else:
        # For horizontal text, lines are rows - sort by y
        return sum(d.y for d in line) / len(line)


def collimate_word_grouping(
    detections: sv.Detections,
    reading_direction: str,
    delimiter: str = "",
    tolerance: int = 10,
) -> Dict[str, str]:
    """Stitch OCR detections using greedy parent-child traversal (collimate algorithm).

    This algorithm is good for skewed or curved text where traditional bucket-based
    line grouping may fail. It works by:
    1. Sorting detections by primary reading coordinate
    2. Starting with the first detection as a "parent"
    3. Finding all detections that "follow" the parent (within tolerance)
    4. Building lines/columns through greedy traversal

    Args:
        detections: Supervision Detections object containing OCR results
        reading_direction: Direction to read text
        delimiter: String to insert between characters within words
        tolerance: Pixel tolerance for alignment

    Returns:
        Dict containing stitched OCR text under 'ocr_text' key
    """
    if len(detections) == 0:
        return {"ocr_text": ""}

    xyxy = detections.xyxy
    class_names = detections.data["class_name"]

    # Convert to CollimateDetection objects
    coll_detections = [
        CollimateDetection(xyxy[i], class_names[i], i) for i in range(len(detections))
    ]

    # Sort by primary reading coordinate
    coll_detections = _sort_detections_for_collimate(coll_detections, reading_direction)

    if len(coll_detections) == 0:
        return {"ocr_text": ""}

    # Build lines through greedy parent-child traversal
    remaining = list(coll_detections)
    lines: List[List[CollimateDetection]] = [[remaining.pop(0)]]

    while len(remaining) > 0:
        found_child = False

        # Try to extend existing lines
        for line in lines:
            parent = line[-1]

            # Find children that follow parent
            for det in remaining.copy():
                if _detection_follows(parent, det, reading_direction, tolerance):
                    found_child = True
                    line.append(det)
                    parent = det  # New parent for next iteration
                    remaining.remove(det)

        # If no children found for any line, start a new line
        if not found_child and len(remaining) > 0:
            lines.append([remaining.pop(0)])

    # Sort lines by their average secondary coordinate
    is_vertical = reading_direction in [
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
    ]
    if is_vertical:
        # For vertical text, sort columns left-to-right (or right-to-left)
        reverse = reading_direction == "vertical_bottom_to_top"
    else:
        # For horizontal text, sort rows top-to-bottom
        reverse = False

    lines = sorted(
        lines,
        key=lambda line: _get_line_avg_coord(line, reading_direction),
        reverse=reverse,
    )

    # Build output text
    line_texts = []
    for line in lines:
        # Characters within a line are concatenated with delimiter
        line_text = delimiter.join(d.class_name for d in line)
        line_texts.append(line_text)

    # Join lines with appropriate separator
    line_separator = get_line_separator(reading_direction)
    return {"ocr_text": line_separator.join(line_texts)}

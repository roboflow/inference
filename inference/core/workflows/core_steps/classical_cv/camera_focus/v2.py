from typing import List, Literal, Optional, Tuple, Type

import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Calculate a score to indicate how well-focused a camera is."
LONG_DESCRIPTION = """
Calculate focus quality scores using the Tenengrad focus measure (Sobel gradient magnitudes) to assess image sharpness, detect blur, evaluate camera focus performance, enable auto-focus systems, perform image quality assessment, compute per-region focus measures for detected objects, and provide comprehensive visualization overlays including zebra pattern exposure warnings, focus peaking, heads-up display, composition grid, and center markers for professional camera control and image analysis workflows.

## How This Block Works

This block calculates the Tenengrad focus measure, which quantifies image sharpness by measuring gradient magnitudes using Sobel operators. The block:

1. Receives an input image (color or grayscale, automatically converts color to grayscale for processing)
2. Optionally receives detection bounding boxes to compute focus measures within specific regions
3. Converts the image to grayscale if it's in color format (Tenengrad measure works on single-channel images)
4. Calculates horizontal and vertical Sobel gradients:
   - Applies Sobel operator in horizontal direction (gradient X) to detect vertical edges
   - Applies Sobel operator in vertical direction (gradient Y) to detect horizontal edges
   - Uses 3x3 Sobel kernels for gradient computation
   - Computes gradient magnitude using squared values: sqrt(gx² + gy²) approximated as gx² + gy²
5. Calculates the focus measure:
   - Squares the horizontal and vertical gradient components
   - Sums the squared gradients to create a focus measure matrix
   - Higher values indicate stronger edges and finer detail (sharper, more focused regions)
   - Lower values indicate weaker edges and less detail (blurrier, less focused regions)
6. Computes overall focus value:
   - Calculates mean of focus measure matrix across entire image
   - Returns a single numerical focus score for the whole image
7. Computes per-region focus measures (if detections provided):
   - Extracts bounding box coordinates from detection predictions
   - Clips bounding boxes to image boundaries
   - Calculates mean focus measure within each bounding box region
   - Returns a list of focus values, one per detection region
8. Applies optional visualization overlays:
   - **Zebra Pattern Warnings**: Diagonal stripe overlay on under/overexposed regions (blue for underexposed, red for overexposed) to identify exposure issues
   - **Focus Peaking**: Green overlay highlighting in-focus areas (regions above focus threshold) to visualize sharp regions
   - **Heads-Up Display (HUD)**: Semi-transparent overlay showing focus value, brightness histogram (for each color channel and grayscale), and exposure information in top-left corner
   - **Composition Grid**: Overlay grid lines for composition assistance (2x2, 3x3 rule of thirds, 4x4, or 5x5 divisions)
   - **Center Marker**: Crosshair marker at frame center for alignment and framing reference
9. Preserves image structure and metadata
10. Returns the visualization image (if overlays enabled), overall focus measure value, and per-bounding-box focus measures list

The Tenengrad focus measure quantifies image sharpness by analyzing edge strength and gradient magnitudes. In-focus images contain many sharp edges with strong gradients, resulting in high Tenengrad scores. Out-of-focus images have blurred edges with weak gradients, resulting in low Tenengrad scores. The measure uses Sobel operators to compute gradients efficiently and is robust to noise. Higher Tenengrad values indicate better focus, with typical ranges varying based on image content, resolution, and edge density. The visualization overlays provide professional camera control aids, helping identify focus issues, exposure problems, and composition opportunities in real-time or during analysis.

## Common Use Cases

- **Auto-Focus Systems**: Assess focus quality to enable automatic camera focus adjustment with per-region focus analysis (e.g., evaluate focus during auto-focus operations, detect optimal focus position for specific objects, trigger focus adjustments based on Tenengrad scores), enabling advanced auto-focus workflows
- **Image Quality Assessment**: Evaluate image sharpness and detect blurry images with visualization overlays for quality control (e.g., assess image quality in capture pipelines with HUD display, detect out-of-focus images with focus peaking, filter low-quality images using focus thresholds), enabling comprehensive quality assessment workflows
- **Professional Camera Control**: Provide real-time focus and exposure feedback for manual camera operation (e.g., display focus peaking for manual focus, show zebra warnings for exposure adjustment, use composition grid for framing), enabling professional camera control workflows
- **Object-Specific Focus Analysis**: Evaluate focus quality for specific detected objects within images (e.g., assess focus on detected objects, analyze focus per bounding box region, optimize focus for specific object classes), enabling object-focused analysis workflows
- **Camera Calibration**: Evaluate focus performance during camera setup and calibration with comprehensive visualization (e.g., assess focus during camera calibration with overlays, optimize focus settings using HUD feedback, evaluate camera performance with visualization aids), enabling enhanced camera calibration workflows
- **Video Focus Tracking**: Monitor focus quality across video frames with per-object focus measures (e.g., track focus for moving objects, monitor focus quality in video streams, analyze focus consistency across frames), enabling video focus tracking workflows

## Connecting to Other Blocks

This block receives an image (and optionally detections) and produces a visualization image, overall focus_measure float value, and bbox_focus_measures list:

- **After object detection or instance segmentation blocks** to compute focus measures for detected objects (e.g., assess focus on detected objects, analyze focus per detection region, evaluate object-specific focus quality), enabling detection-to-focus workflows
- **After image capture or preprocessing blocks** to assess focus quality of captured or processed images (e.g., evaluate focus after image capture, assess sharpness after preprocessing with visualization, measure focus in image pipelines with overlays), enabling enhanced focus assessment workflows
- **Before logic blocks** like Continue If to make decisions based on focus quality (e.g., continue if focus is good, filter images based on Tenengrad scores, make decisions using focus measures or per-object focus values), enabling focus-based decision workflows
- **Before analysis blocks** to assess image quality before analysis (e.g., evaluate focus before analysis with HUD display, assess sharpness for processing, measure quality before analysis), enabling quality-based analysis workflows
- **In auto-focus systems** where focus measurement is part of a feedback loop with per-object analysis (e.g., measure focus for auto-focus with object prioritization, assess focus in feedback systems, evaluate focus in control loops), enabling advanced auto-focus system workflows
- **Before visualization blocks** to display focus quality information (e.g., visualize focus scores with overlays, display focus measures, show focus quality with professional camera aids), enabling comprehensive focus visualization workflows

## Version Differences

**Enhanced from v1:**

- **Different Focus Algorithm**: Uses Tenengrad focus measure (Sobel gradient magnitudes) instead of Brenner measure, providing more robust edge detection and focus assessment
- **Visualization Overlays**: Includes comprehensive visualization features including zebra pattern exposure warnings, focus peaking (green highlight on sharp areas), heads-up display with focus values and brightness histogram, composition grid overlays (2x2, 3x3, 4x4, 5x5), and center crosshair marker for professional camera control
- **Per-Region Focus Analysis**: Supports optional detection bounding boxes to compute focus measures within specific object regions, enabling object-specific focus assessment
- **Enhanced Outputs**: Returns three outputs - visualization image, overall focus_measure float, and bbox_focus_measures list (per-detection focus values)
- **Configurable Visualization**: All visualization overlays are configurable (zebra warnings, HUD, focus peaking, grid, center marker can be enabled/disabled independently)
- **Exposure Analysis**: Includes exposure assessment with configurable thresholds for underexposed/overexposed regions with visual zebra pattern warnings
- **Professional Camera Aids**: Provides tools similar to professional camera displays including focus peaking, histogram display, and composition guides

## Requirements

This block works on color or grayscale input images. Color images are automatically converted to grayscale before processing (Tenengrad measure works on single-channel images). The block outputs a visualization image (with optional overlays), an overall focus_measure float value, and a bbox_focus_measures list (if detections are provided). Higher Tenengrad values indicate better focus and sharper images, while lower values indicate blur and poor focus. The focus measure is sensitive to image content, resolution, and edge density, so threshold values for "good" focus should be calibrated based on specific use cases and image characteristics. All visualization overlays are optional and can be enabled or disabled independently based on workflow needs.
"""


class CameraFocusManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/camera_focus@v2"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Camera Focus",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-aperture",
                "blockPriority": 8,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image (color or grayscale) to calculate focus quality for. Color images are automatically converted to grayscale before processing (Tenengrad focus measure works on single-channel images). The block calculates the Tenengrad focus measure using Sobel gradient magnitudes to assess image sharpness. The output includes a visualization image (with optional overlays if enabled), an overall focus_measure float value, and a bbox_focus_measures list (per-detection focus values if detections are provided). Higher Tenengrad values indicate better focus and sharper images (stronger edges and gradients), while lower values indicate blur and poor focus (weaker gradients). The focus measure uses Sobel operators to compute gradient magnitudes efficiently. Original image metadata is preserved. Use this block to assess focus quality, detect blur, enable auto-focus systems, perform object-specific focus analysis, or perform image quality assessment with professional camera control visualization aids.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    underexposed_threshold_percent: float = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description="Brightness percentage threshold below which pixels are marked as underexposed. Must be between 0.0 and 100.0. Default is 3.0%, meaning pixels with brightness below 3% (approximately value 8 in 0-255 range) are considered underexposed. Pixels below this threshold will show blue zebra pattern overlay when show_zebra_warnings is enabled. Lower values are stricter (fewer pixels marked as underexposed), higher values are more lenient (more pixels marked as underexposed). Adjust based on exposure tolerance and image requirements.",
    )

    overexposed_threshold_percent: float = Field(
        default=97.0,
        ge=0.0,
        le=100.0,
        description="Brightness percentage threshold above which pixels are marked as overexposed. Must be between 0.0 and 100.0. Default is 97.0%, meaning pixels with brightness above 97% (approximately value 247 in 0-255 range) are considered overexposed. Pixels above this threshold will show red zebra pattern overlay when show_zebra_warnings is enabled. Higher values are stricter (fewer pixels marked as overexposed), lower values are more lenient (more pixels marked as overexposed). Adjust based on exposure tolerance and image requirements.",
    )

    show_zebra_warnings: bool = Field(
        default=True,
        description="Display diagonal zebra pattern overlay on under/overexposed regions. When enabled (default True), pixels below underexposed_threshold_percent show blue zebra stripes, and pixels above overexposed_threshold_percent show red zebra stripes. This provides visual feedback for exposure issues similar to professional camera zebra pattern displays. The zebra pattern helps identify regions with exposure problems (too dark or too bright) that may need adjustment. Disable if you don't want exposure warnings or want cleaner visualization.",
    )

    grid_overlay: Literal["None", "2x2", "3x3", "4x4", "5x5"] = Field(
        default="3x3",
        description="Composition grid overlay for framing assistance. Options: 'None' (no grid), '2x2' (four quadrants), '3x3' (default, rule of thirds with 9 sections), '4x4' (16 sections), or '5x5' (25 sections). The grid helps with composition and framing by dividing the image into sections. The 3x3 grid (rule of thirds) is commonly used for balanced composition. Grid lines are drawn in gray color. Choose based on composition needs: rule of thirds (3x3) for general use, 2x2 for simple quadrant composition, or higher divisions for more detailed composition guides.",
    )

    show_hud: bool = Field(
        default=True,
        description="Display heads-up display (HUD) overlay with focus scores and brightness histogram. When enabled (default True), shows a semi-transparent black overlay in the top-left corner displaying: focus value (labeled 'TFM Focus' with numerical score), brightness histogram showing distribution for each color channel (red, green, blue) and grayscale, and exposure label. The HUD provides comprehensive focus and exposure information for professional camera control. Disable if you don't need the HUD display or want cleaner visualization.",
    )

    show_focus_peaking: bool = Field(
        default=True,
        description="Display green overlay highlighting in-focus areas (focus peaking). When enabled (default True), regions with focus measures above a threshold (top 30% by default) are highlighted with a semi-transparent green overlay. This helps visualize which areas of the image are in sharp focus, similar to professional camera focus peaking displays. The green highlight makes it easy to see sharp regions at a glance. Disable if you don't want focus peaking overlay or want cleaner visualization.",
    )

    show_center_marker: bool = Field(
        default=True,
        description="Display crosshair marker at the center of the frame. When enabled (default True), shows a white crosshair at the image center for alignment and framing reference. The crosshair size scales with image dimensions for visibility. This helps with composition alignment and center framing, similar to professional camera center markers. Disable if you don't need the center marker or want cleaner visualization.",
    )

    detections: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        description="Optional detection predictions (object detection or instance segmentation) to compute focus measures within bounding box regions. When provided, the block calculates a separate focus measure for each detection's bounding box region and returns them in the bbox_focus_measures list output. This enables object-specific focus analysis, allowing you to assess focus quality for individual detected objects rather than just the overall image. Useful for evaluating focus on specific objects of interest, analyzing focus per object class, or optimizing focus for detected regions. Each bbox_focus_measure value corresponds to the mean Tenengrad focus measure within that object's bounding box. Leave as None if you only need overall image focus assessment.",
        examples=["$steps.object_detection_model.predictions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
            OutputDefinition(
                name="focus_measure",
                kind=[FLOAT_KIND],
            ),
            OutputDefinition(
                name="bbox_focus_measures",
                kind=[LIST_OF_VALUES_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CameraFocusBlockV2(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[CameraFocusManifest]:
        return CameraFocusManifest

    def run(
        self,
        image: WorkflowImageData,
        underexposed_threshold_percent: float,
        overexposed_threshold_percent: float,
        show_zebra_warnings: bool,
        grid_overlay: str,
        show_hud: bool,
        show_focus_peaking: bool,
        show_center_marker: bool,
        detections: Optional[sv.Detections],
        *args,
        **kwargs,
    ) -> BlockResult:
        underexposed_threshold = int(underexposed_threshold_percent * 255 / 100)
        overexposed_threshold = int(overexposed_threshold_percent * 255 / 100)
        result_image, focus_value, bbox_focus_values = visualize_tenengrad_measure(
            image.numpy_image,
            underexposed_threshold=underexposed_threshold,
            overexposed_threshold=overexposed_threshold,
            show_zebra_warnings=show_zebra_warnings,
            grid_overlay=grid_overlay,
            show_hud=show_hud,
            show_focus_peaking=show_focus_peaking,
            show_center_marker=show_center_marker,
            detections=detections,
        )
        if result_image is image.numpy_image:
            output = image
        else:
            output = WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=result_image,
            )
        return {
            OUTPUT_IMAGE_KEY: output,
            "focus_measure": focus_value,
            "bbox_focus_measures": bbox_focus_values,
        }


def _create_zebra_mask(shape: Tuple[int, int], spacing: int = 8) -> np.ndarray:
    """Create diagonal zebra stripe pattern mask."""
    height, width = shape
    y, x = np.ogrid[:height, :width]
    return ((x + y) // spacing) % 2 == 0


def _apply_zebra_warnings(
    image: np.ndarray,
    gray: np.ndarray,
    under_thresh: int = 16,
    over_thresh: int = 239,
    opacity: float = 0.5,
) -> np.ndarray:
    """Overlay zebra pattern on under/overexposed regions."""
    output = image.copy()
    zebra = _create_zebra_mask(gray.shape)

    underexposed = (gray < under_thresh) & zebra
    overexposed = (gray > over_thresh) & zebra

    blue = np.array([255, 0, 0], dtype=np.uint8)
    red = np.array([0, 0, 255], dtype=np.uint8)

    output[underexposed] = (
        output[underexposed] * (1 - opacity) + blue * opacity
    ).astype(np.uint8)
    output[overexposed] = (output[overexposed] * (1 - opacity) + red * opacity).astype(
        np.uint8
    )

    return output


def _apply_focus_peaking(
    image: np.ndarray,
    focus_measure: np.ndarray,
    threshold_percent: float = 30.0,
    opacity: float = 0.6,
) -> np.ndarray:
    """Overlay green highlight on in-focus areas."""
    output = image.copy()
    max_val = focus_measure.max()
    if max_val == 0:
        return output
    normalized = (focus_measure / max_val * 255).astype(np.uint8)
    threshold = int(threshold_percent * 255 / 100)
    mask = normalized > threshold
    green = np.array([0, 255, 0], dtype=np.uint8)
    output[mask] = (output[mask] * (1 - opacity) + green * opacity).astype(np.uint8)
    return output


def _draw_center_marker(
    image: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw crosshair at frame center."""
    height, width = image.shape[:2]

    reference_size = 720
    scale = min(height, width) / reference_size
    scale = max(0.4, min(scale, 2.5))

    size = int(20 * scale)
    thickness = max(1, int(1.6 * scale))

    cx, cy = width // 2, height // 2
    cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)
    return image


def _draw_grid(
    image: np.ndarray,
    divisions: int,
    color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Draw composition grid lines."""
    height, width = image.shape[:2]
    for i in range(1, divisions):
        x = width * i // divisions
        y = height * i // divisions
        cv2.line(image, (x, 0), (x, height), color, 1)
        cv2.line(image, (0, y), (width, y), color, 1)
    return image


def _draw_text_with_outline(
    image: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw text with dark outline for better legibility."""
    x, y = pos
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)


def _draw_hud_overlay(
    image: np.ndarray,
    focus_value: float,
    gray: np.ndarray,
    original_image: np.ndarray,
) -> np.ndarray:
    """Draw focus value and histogram overlay."""
    output = image.copy()
    height, width = image.shape[:2]

    reference_size = 720
    scale = min(height, width) / reference_size
    scale = max(0.4, min(scale, 2.5))

    padding = int(14 * scale)
    hist_width = int(180 * scale)
    hist_height = int(50 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * scale
    font_scale_small = 0.4 * scale
    thickness = max(1, int(1.6 * scale))
    line_spacing = int(6 * scale)
    margin = int(12 * scale)

    focus_label = "TFM Focus"
    focus_value_text = f"{focus_value:.1f}"
    exposure_label = "Exposure"

    focus_label_size = cv2.getTextSize(focus_label, font, font_scale, thickness)[0]
    focus_value_size = cv2.getTextSize(focus_value_text, font, font_scale, thickness)[0]
    label_size = cv2.getTextSize(exposure_label, font, font_scale_small, thickness)[0]
    section_spacing = int(12 * scale)

    content_width = max(
        focus_label_size[0] + focus_value_size[0] + int(20 * scale),
        label_size[0],
        hist_width,
    )
    content_height = (
        focus_label_size[1]
        + section_spacing
        + label_size[1]
        + line_spacing
        + hist_height
    )

    hud_width = content_width + padding * 2
    hud_height = content_height + padding * 2
    hud_x, hud_y = margin, margin

    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (hud_x, hud_y),
        (hud_x + hud_width, hud_y + hud_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

    cv2.rectangle(
        output,
        (hud_x, hud_y),
        (hud_x + hud_width, hud_y + hud_height),
        (80, 80, 80),
        1,
    )

    text_x = hud_x + padding
    cursor_y = hud_y + padding + focus_label_size[1]
    _draw_text_with_outline(
        output,
        focus_label,
        (text_x, cursor_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    value_x = hud_x + hud_width - padding - focus_value_size[0]
    _draw_text_with_outline(
        output,
        focus_value_text,
        (value_x, cursor_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    divider_y = cursor_y + int(section_spacing * 0.5)
    cv2.line(
        output,
        (hud_x + padding, divider_y),
        (hud_x + hud_width - padding, divider_y),
        (60, 60, 60),
        1,
    )

    cursor_y += section_spacing + label_size[1]
    _draw_text_with_outline(
        output,
        exposure_label,
        (text_x, cursor_y),
        font,
        font_scale_small,
        (180, 180, 180),
        thickness,
    )

    hist_x = text_x
    hist_y = cursor_y + line_spacing
    hist_bottom = hist_y + hist_height - 1

    cv2.rectangle(
        output,
        (hist_x, hist_y),
        (hist_x + hist_width, hist_bottom),
        (0, 0, 0),
        -1,
    )

    x_coords = np.linspace(hist_x, hist_x + hist_width - 1, 256).astype(np.int32)
    line_thickness = max(1, thickness)

    if len(original_image.shape) == 3:
        channel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for ch, color in enumerate(channel_colors):
            hist = cv2.calcHist([original_image], [ch], None, [256], [0, 256])
            hist_max = hist.max()
            if hist_max > 0:
                hist_normalized = (
                    (hist / hist_max * hist_height).astype(np.int32).flatten()
                )
                pts = np.column_stack([x_coords, hist_bottom - hist_normalized]).astype(
                    np.int32
                )
                cv2.polylines(output, [pts], False, color, line_thickness)

    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    gray_hist_max = gray_hist.max()
    if gray_hist_max > 0:
        gray_hist_normalized = (
            (gray_hist / gray_hist_max * hist_height).astype(np.int32).flatten()
        )
        pts = np.column_stack([x_coords, hist_bottom - gray_hist_normalized]).astype(
            np.int32
        )
        cv2.polylines(output, [pts], False, (255, 255, 255), line_thickness)

    return output


GRID_DIVISIONS = {
    "None": 0,
    "2x2": 2,
    "3x3": 3,
    "4x4": 4,
    "5x5": 5,
}


def _compute_tenengrad(
    input_image: np.ndarray,
    detections: Optional[sv.Detections] = None,
) -> Tuple[np.ndarray, np.ndarray, float, List[Optional[float]]]:
    """
    Compute Tenengrad focus measure using Sobel gradients.

    Returns grayscale image, focus measure array, overall focus value,
    and per-bbox focus values.
    """
    if len(input_image.shape) == 3:
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = input_image

    height, width = gray.shape

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    focus_measure = gx
    np.square(focus_measure, out=focus_measure)
    np.square(gy, out=gy)
    np.add(focus_measure, gy, out=focus_measure)

    focus_value = float(focus_measure.mean())

    bbox_focus_measures: List[Optional[float]] = []
    if detections is not None and len(detections) > 0:
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                region_focus = focus_measure[y1:y2, x1:x2].mean()
                bbox_focus_measures.append(float(region_focus))
            else:
                bbox_focus_measures.append(None)

    return gray, focus_measure, focus_value, bbox_focus_measures


def visualize_tenengrad_measure(
    input_image: np.ndarray,
    underexposed_threshold: int = 16,
    overexposed_threshold: int = 239,
    show_zebra_warnings: bool = True,
    grid_overlay: str = "3x3",
    show_hud: bool = True,
    show_focus_peaking: bool = True,
    show_center_marker: bool = True,
    detections: Optional[sv.Detections] = None,
) -> Tuple[np.ndarray, float, List[Optional[float]]]:
    """
    Tenengrad focus measure with visualization overlay.

    Uses Sobel operators to compute gradient magnitudes as a focus metric.
    Higher values indicate sharper/more in-focus images.

    Returns the input image unchanged if no visualizations are enabled.
    """
    grid_divisions = GRID_DIVISIONS.get(grid_overlay, 0)
    any_visualization_enabled = (
        show_zebra_warnings
        or show_hud
        or show_focus_peaking
        or show_center_marker
        or grid_divisions > 0
    )

    gray, focus_measure, focus_value, bbox_focus_measures = _compute_tenengrad(
        input_image, detections
    )

    if not any_visualization_enabled:
        return input_image, focus_value, bbox_focus_measures

    if len(input_image.shape) == 3:
        output = input_image.copy()
    else:
        output = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    if show_zebra_warnings:
        output = _apply_zebra_warnings(
            output, gray, underexposed_threshold, overexposed_threshold
        )
    if show_focus_peaking:
        output = _apply_focus_peaking(output, focus_measure)
    if show_center_marker:
        output = _draw_center_marker(output)
    if grid_divisions > 0:
        output = _draw_grid(output, grid_divisions)
    if show_hud:
        output = _draw_hud_overlay(output, focus_value, gray, input_image)

    return output, focus_value, bbox_focus_measures

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
LONG_DESCRIPTION: str = """
Calculates image sharpness using the Tenengrad focus measure (Sobel gradient magnitudes).
Higher scores indicate sharper, more in-focus images.

Includes visualization overlays: zebra pattern warnings for exposure issues, focus peaking
to highlight sharp areas, a heads-up display with focus values and histogram, composition
grid, and center crosshair.

Optionally accepts bounding box detections to compute focus measures within specific regions.
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
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    underexposed_threshold_percent: float = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description="Brightness percentage below which pixels are marked as underexposed.",
    )

    overexposed_threshold_percent: float = Field(
        default=97.0,
        ge=0.0,
        le=100.0,
        description="Brightness percentage above which pixels are marked as overexposed.",
    )

    show_zebra_warnings: bool = Field(
        default=True,
        description="Display zebra pattern overlay on under/overexposed regions.",
    )

    grid_overlay: Literal["None", "2x2", "3x3", "4x4", "5x5"] = Field(
        default="3x3",
        description="Composition grid overlay (3x3 is rule of thirds).",
    )

    show_hud: bool = Field(
        default=True,
        description="Display heads-up overlay with focus scores and brightness histogram.",
    )

    show_focus_peaking: bool = Field(
        default=True,
        description="Display green overlay highlighting in-focus areas (focus peaking).",
    )

    show_center_marker: bool = Field(
        default=True,
        description="Display crosshair marker at the center of the frame.",
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
        description="Optional detections to compute focus measures within bounding boxes.",
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
        result_image, focus_value, bbox_focus_values = calculate_tenengrad_measure(
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
    thickness = max(1, int(1.5 * scale))

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


def _draw_hud_overlay(
    image: np.ndarray,
    focus_value: float,
    gray: np.ndarray,
) -> np.ndarray:
    """Draw focus value and histogram overlay."""
    output = image.copy()
    height, width = image.shape[:2]

    reference_size = 720
    scale = min(height, width) / reference_size
    scale = max(0.4, min(scale, 2.5))

    padding = int(12 * scale)
    hist_width = int(200 * scale)
    hist_height = int(60 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 * scale
    thickness = max(1, int(1.5 * scale))
    line_height = int(20 * scale)
    radius = int(8 * scale)
    margin = int(12 * scale)

    text1 = f"Focus: {focus_value:.1f}"

    text_width = cv2.getTextSize(text1, font, font_scale, thickness)[0][0]
    text_height = cv2.getTextSize(text1, font, font_scale, thickness)[0][1]
    content_width = max(text_width, hist_width)
    content_height = line_height + hist_height

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
    corner_radius = min(radius, hud_width // 4, hud_height // 4)
    if corner_radius > 0:
        cv2.rectangle(
            overlay,
            (hud_x + corner_radius, hud_y),
            (hud_x + hud_width - corner_radius, hud_y + hud_height),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            overlay,
            (hud_x, hud_y + corner_radius),
            (hud_x + hud_width, hud_y + hud_height - corner_radius),
            (0, 0, 0),
            -1,
        )
        for cx, cy in [
            (hud_x + corner_radius, hud_y + corner_radius),
            (hud_x + hud_width - corner_radius, hud_y + corner_radius),
            (hud_x + corner_radius, hud_y + hud_height - corner_radius),
            (hud_x + hud_width - corner_radius, hud_y + hud_height - corner_radius),
        ]:
            cv2.circle(overlay, (cx, cy), corner_radius, (0, 0, 0), -1)

    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

    text_x = hud_x + padding
    text_y = hud_y + padding + text_height
    cv2.putText(
        output, text1, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
    )

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = (hist / hist.max() * hist_height).astype(np.int32).flatten()

    hist_x = hud_x + padding
    hist_y = text_y + int(8 * scale)
    x_coords = np.linspace(hist_x, hist_x + hist_width - 1, 256).astype(np.int32)
    pts = np.column_stack(
        [x_coords, hist_y + hist_height - 1 - hist_normalized]
    ).astype(np.int32)
    cv2.polylines(output, [pts], False, (255, 255, 255), max(1, thickness - 1))

    return output


GRID_DIVISIONS = {
    "None": 0,
    "2x2": 2,
    "3x3": 3,
    "4x4": 4,
    "5x5": 5,
}


def calculate_tenengrad_measure(
    input_image: np.ndarray,
    underexposed_threshold: int = 16,
    overexposed_threshold: int = 239,
    show_zebra_warnings: bool = True,
    grid_overlay: str = "3x3",
    show_hud: bool = True,
    show_focus_peaking: bool = True,
    show_center_marker: bool = True,
    detections: Optional[sv.Detections] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Tenengrad focus measure with visualization overlay.

    Uses Sobel operators to compute gradient magnitudes as a focus metric.
    Higher values indicate sharper/more in-focus images.

    Parameters
    ----------
    input_image : np.ndarray
        The input image (color or grayscale).
    underexposed_threshold : int
        Pixel intensity below which areas are marked as underexposed.
    overexposed_threshold : int
        Pixel intensity above which areas are marked as overexposed.
    show_zebra_warnings : bool
        Whether to display zebra pattern overlay on under/overexposed regions.
    grid_overlay : str
        Grid overlay type: "None", "2x2", "3x3", "4x4", or "5x5".
    show_hud : bool
        Whether to display heads-up overlay with focus values and histogram.
    show_focus_peaking : bool
        Whether to display green overlay on in-focus areas.
    show_center_marker : bool
        Whether to display crosshair at frame center.
    detections : Optional[sv.Detections]
        Optional detections to compute focus measures within bounding boxes.

    Returns
    -------
    Tuple[np.ndarray, float, List[float]]
        Visualization image, whole-image focus value, and list of per-bbox focus values.
    """
    if len(input_image.shape) == 3:
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        output = input_image.copy()
    else:
        gray = input_image
        output = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    height, width = gray.shape

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    focus_measure = gx**2 + gy**2
    focus_value = float(focus_measure.mean())

    bbox_focus_measures: List[float] = []
    if detections is not None and len(detections) > 0:
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                region_focus = focus_measure[y1:y2, x1:x2].mean()
                bbox_focus_measures.append(float(region_focus))

    if show_zebra_warnings:
        output = _apply_zebra_warnings(
            output, gray, underexposed_threshold, overexposed_threshold
        )
    if show_focus_peaking:
        output = _apply_focus_peaking(output, focus_measure)
    if show_center_marker:
        output = _draw_center_marker(output)
    grid_divisions = GRID_DIVISIONS.get(grid_overlay, 0)
    if grid_divisions > 0:
        output = _draw_grid(output, grid_divisions)
    if show_hud:
        output = _draw_hud_overlay(output, focus_value, gray)

    return output, focus_value, bbox_focus_measures

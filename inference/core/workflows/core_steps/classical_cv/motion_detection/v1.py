import json
from typing import List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field
from shapely.geometry import Polygon

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ZONE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class MotionDetectionManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/motion_detection@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Motion Detection",
            "version": "v1",
            "short_description": "Detect motion in a video using OpenCV.",
            "long_description": (
                """
Detect motion in video streams using OpenCV's background subtraction algorithm.

## How This Block Works

This block uses background subtraction (specifically the MOG2 algorithm) to detect motion in video frames. The block maintains state across frames to build a background model and track motion patterns:

1. **Initializes background model** - on the first frame, creates a background subtractor using the specified history and threshold parameters
2. **Processes each frame** - applies background subtraction to identify pixels that differ from the learned background model
3. **Filters noise** - applies morphological operations to remove noise and combine nearby motion regions into coherent contours
4. **Extracts motion regions** - finds contours representing motion areas, filters them by minimum size, and optionally clips them to a detection zone
5. **Simplifies contours** - reduces contour complexity to keep detection data manageable
6. **Generates outputs** - creates object detection predictions with bounding boxes, determines motion status, triggers alarms when motion starts, and provides motion zone polygons

The block tracks motion state across frames - the **alarm** output becomes true only when motion transitions from not detected to detected, making it useful for triggering actions when motion first appears.

## Common Use Cases

- **Security Monitoring**: Detect motion in surveillance cameras to trigger alerts, recordings, or notifications when activity is detected
- **Resource Optimization**: Conditionally run expensive inference operations (e.g., object detection, classification) only when motion is detected to save computational resources
- **Activity Detection**: Monitor areas for movement to track occupancy, identify entry/exit events, or detect unauthorized access
- **Video Analytics**: Analyze video streams to identify motion patterns, track activity levels, or detect anomalies in monitored areas
- **Smart Recording**: Trigger video recording or snapshot capture when motion is detected, reducing storage requirements compared to continuous recording
- **Zone Monitoring**: Monitor specific areas within a frame using detection zones to focus motion detection on relevant regions while ignoring busy but irrelevant areas

## Connecting to Other Blocks

The motion detection outputs from this block can be connected to:

- **Conditional logic blocks** (e.g., Continue If) to execute workflow steps only when motion is detected or when alarms trigger
- **Object detection blocks** to run detection models only on frames with motion, saving computational resources
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when motion is detected or alarms trigger
- **Data storage blocks** (e.g., Roboflow Dataset Upload, CSV Formatter) to log motion events, timestamps, and detection data for analytics
- **Visualization blocks** to draw motion zones, bounding boxes, or annotations on frames showing detected motion
- **Filter blocks** to filter images or data based on motion status before passing to downstream processing
"""
            ),
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-bell-exclamation",
                "blockPriority": 8,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image or video frame to analyze for motion. The block processes frames sequentially to build a background model - each frame updates the background model and detects motion relative to learned background patterns. Can be connected from workflow inputs or previous steps.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    minimum_contour_area: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Minimum Contour Area",
        description="Minimum area in square pixels for a motion region to be detected. Contours smaller than this threshold are filtered out to ignore noise, small shadows, or minor pixel variations. Lower values increase sensitivity but may detect more false positives (e.g., 100 for very sensitive detection, 500 for only large objects). Default is 200 square pixels.",
        gt=0,
        examples=[200, 100, 500],
        default=200,
    )

    morphological_kernel_size: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Morphological Kernel Size",
        description="Size of the morphological kernel in pixels used to combine nearby motion regions and filter noise. Larger values merge more distant motion regions into single contours but may also merge separate objects. Smaller values preserve more detail but may leave fragmented detections. The kernel uses an elliptical shape. Default is 3 pixels.",
        gt=0,
        examples=[3, 5, 7],
        default=3,
    )

    threshold: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Threshold",
        description="Threshold value for the squared Mahalanobis distance used by the MOG2 background subtraction algorithm. Controls sensitivity to motion - smaller values increase sensitivity (detect smaller changes) but may produce more false positives, larger values decrease sensitivity (only detect significant changes) but may miss subtle motion. Recommended range is 8-32. Default is 16.",
        gt=0,
        examples=[16, 8, 24, 32],
        default=16,
    )

    history: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="History",
        description="Number of previous frames used to build the background model. Controls how quickly the background adapts to changes - larger values (e.g., 50-100) create a more stable background model that's less sensitive to temporary changes but adapts slowly to permanent background changes. Smaller values (e.g., 10-20) allow faster adaptation but may treat moving objects as background if they stop moving. Default is 30 frames.",
        gt=0,
        examples=[30, 50, 100],
        default=30,
    )

    detection_zone: Union[list, str, Selector(kind=[ZONE_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        title="Detection Zone",
        description="Optional polygon zone to limit motion detection to a specific area of the frame. Motion is only detected within this zone, ignoring activity outside. Format: [[x1, y1], [x2, y2], [x3, y3], ...] where coordinates are in pixels. The polygon must have more than 3 points. Can be provided as a list, JSON string, or selector referencing zone outputs from other blocks. Useful for focusing on specific regions (e.g., doorways, windows, restricted areas) while ignoring busy but irrelevant areas. If not provided, motion is detected across the entire frame.",
        default=None,
    )

    suppress_first_detections: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(  # type: ignore
        title="Don't Detect Until History is Full",
        description="If true, suppresses motion detections until the background model has been initialized with enough frames (specified by the history parameter). This prevents false positives from early frames where the background model hasn't learned the scene yet. When false, the block attempts to detect motion immediately, which may produce unreliable results during initialization. Default is true (recommended for most use cases).",
        examples=[True, False],
        default=True,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="motion",
                kind=[
                    BOOLEAN_KIND,
                ],
                description="Boolean flag indicating whether motion was detected in the current frame. True if any motion regions were found, false otherwise. This flag is true for every frame with detected motion.",
            ),
            OutputDefinition(
                name="alarm",
                kind=[
                    BOOLEAN_KIND,
                ],
                description="Boolean flag that becomes true only when motion transitions from not detected (previous frame) to detected (current frame). Useful for triggering actions when motion first appears. Returns false if motion was already detected in the previous frame, even if motion continues in the current frame.",
            ),
            OutputDefinition(
                name="detections",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                ],
                description="Object detection predictions containing bounding boxes for all detected motion regions. Each detection has class name 'motion', confidence 1.0, and bounding box coordinates. Empty detections if no motion is detected. Compatible with other blocks that accept object detection predictions.",
            ),
            OutputDefinition(
                name="motion_zones",
                kind=[
                    LIST_OF_VALUES_KIND,
                ],
                description="List of polygon coordinates representing the exact shapes of detected motion regions. Each polygon is a list of [x, y] coordinate pairs defining the contour of a motion region. Useful for visualization or precise motion area analysis. Empty list if no motion is detected.",
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MotionDetectionBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_motion = False
        self.back_sub = None
        self.frame_count = 0
        self._kernel_cache = {}  # Cache morphological kernels by size

    @classmethod
    def get_manifest(cls) -> Type[MotionDetectionManifest]:
        return MotionDetectionManifest

    def run(
        self,
        image: WorkflowImageData,
        minimum_contour_area: int,
        morphological_kernel_size: int,
        threshold: int,
        history: int,
        suppress_first_detections: bool,
        detection_zone: Optional[Union[str, List[Tuple[int, int]]]],
        *args,
        **kwargs,
    ) -> BlockResult:

        if isinstance(detection_zone, str):
            try:
                detection_zone = json.loads(detection_zone)
            except Exception as e:
                raise ValueError(f"Could not parse detection zone as a valid json")

        if not self.back_sub:
            self.frame_count = 0
            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=threshold, detectShadows=True
            )

        frame = image.numpy_image

        # apply background subtraction
        mask = self.back_sub.apply(frame)

        # if frames aren't initialized yet, return no motion
        if self.frame_count < history and suppress_first_detections:
            self.frame_count += 1
            return {
                "motion": False,
                "detections": sv.Detections.empty(),
                "alarm": False,
                "motion_zones": [],
            }

        # apply morphological filtering to ignore changes due to noise
        # Use cached kernel to avoid recreating the same kernel repeatedly
        if morphological_kernel_size not in self._kernel_cache:
            self._kernel_cache[morphological_kernel_size] = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (morphological_kernel_size, morphological_kernel_size),
            )
        kernel = self._kernel_cache[morphological_kernel_size]
        mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # create contours around filtered areas
        contours, hierarchy = cv2.findContours(
            mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # apply minimum contour size and filter out 0 length contours
        # Check length first (cheaper) before computing area
        filtered_contours = [
            contour
            for contour in contours
            if len(contour) > 2 and cv2.contourArea(contour) > minimum_contour_area
        ]

        # clip contours if a detection zone is provided
        if detection_zone and len(detection_zone) > 0:
            filtered_contours = clip_contours_to_contour(
                filtered_contours, detection_zone
            )

        # simplify contours by 1% of their perimeter
        # this is ideal for keeping the detections to a reasonable size
        simplified_contours = []
        for contour in filtered_contours:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Only keep contours with at least 3 vertices
            if len(approx) >= 3:
                simplified_contours.append(approx)

        # get bounding boxes and polygons
        xyxy_boxes = []
        polygons = []
        for cnt in simplified_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            xyxy_boxes.append([int(x), int(y), int(x + w), int(y + h)])
            # Extract polygon coordinates, handling both squeezed and unsqueezed formats
            polygon = np.squeeze(cnt)
            if polygon.ndim == 1:  # Single point case, skip
                continue
            polygons.append(polygon.tolist())

        # convert to sv detections
        detections = (
            sv.Detections(
                xyxy=np.array(xyxy_boxes),
                confidence=np.array([1] * len(xyxy_boxes)),
                class_id=np.array([0] * len(xyxy_boxes)),
                data={"class_name": np.array(["motion"] * len(xyxy_boxes))},
            )
            if len(xyxy_boxes) > 0
            else sv.Detections.empty()
        )

        # if contours exist, there's motion
        motion = len(filtered_contours) > 0

        # alarm flips to true only if there was no motion before and motion now
        alarm = not self.last_motion and motion
        self.last_motion = motion

        return {
            "motion": motion,
            "detections": detections,
            "alarm": alarm,
            "motion_zones": polygons,
        }


def clip_contours_to_contour(
    contours: List[np.ndarray], clip_contour: np.ndarray
) -> List[np.ndarray]:
    """
    Clip OpenCV contours to another contour and return clipped OpenCV contours.

    Args:
        contours: List of OpenCV contours, each as numpy array of shape (N, 1, 2)
        clip_contour: Clip contour as numpy array of shape (M, 2) with xy points

    Returns:
        List of clipped OpenCV contours as numpy arrays of shape (N, 1, 2).
        Only includes contours that overlap with the clip contour.
    """

    clip_poly = Polygon(clip_contour)
    result = []

    for contour in contours:
        # Convert OpenCV contour (N, 1, 2) to xy points (N, 2)
        points = contour.reshape(-1, 2)

        if len(points) < 3:
            continue

        try:
            poly = Polygon(points)
            clipped = poly.intersection(clip_poly)

            if clipped.is_empty:
                continue

            # Extract coordinates based on geometry type
            if clipped.geom_type == "Polygon":
                coords = list(clipped.exterior.coords[:-1])
                if len(coords) >= 3:
                    result.append(list_to_contour(coords))

            elif clipped.geom_type == "MultiPolygon":
                for geom in clipped.geoms:
                    coords = list(geom.exterior.coords[:-1])
                    if len(coords) >= 3:
                        result.append(list_to_contour(coords))

        except Exception:
            # Silently skip contours that fail shapely operations
            # (e.g., self-intersecting polygons)
            continue

    return result


def list_to_contour(list_of_tuples: List[Tuple]) -> np.ndarray:
    """
    Convert a list of (x, y) tuples to an OpenCV contour format.

    Args:
        list_of_tuples: List of coordinate tuples [(x1, y1), (x2, y2), ...]

    Returns:
        NumPy array of shape (N, 1, 2) suitable for OpenCV operations
    """
    points = np.array(
        [[int(xy[0]), int(xy[1])] for xy in list_of_tuples], dtype=np.int32
    )
    return points.reshape(-1, 1, 2)

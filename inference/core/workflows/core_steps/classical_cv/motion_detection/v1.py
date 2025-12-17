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

SHORT_DESCRIPTION: str = "Detect motion in an image using OpenCV."
LONG_DESCRIPTION: str = """
This block uses background subtraction to detect motion in an image. The block draws the contours
of the detected motion, as well as outputs the bounding boxes as an object detection. Two flags are
provided for use in workflows - one to indicate motion, and an alarm to indicate when the motion
changed from no motion to motion detected. Additionally a zone can be provided to limit the scope
of the motion detection to a specific area of the image.

Motion detection is extremely useful for generating alerts and file uploads. Additionally, inference
can be conditionally run based on motion detection to save compute resources.
"""


class MotionDetectionManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/motion_detection@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Motion Detection",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
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
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    minimum_contour_area: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Minimum Contour Area",
        description="Motion in areas smaller than this in square pixels will not be counted as motion.",
        examples=[200],
        default=200,
    )

    morphological_kernel_size: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Morphological Kernel Size",
        description="The size of the kernel used for morphological operations to combine contours.",
        examples=[3],
        default=3,
    )

    threshold: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Threshold",
        description="The threshold value for the squared Mahalanobis distance for background subtraction."
        " Smaller values increase sensitivity to motion. Recommended values are 8-32.",
        examples=[16],
        default=16,
    )

    history: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="History",
        description="The number of previous frames to use for background subtraction. Larger values make the model"
        " less sensitive to quick changes in the background, smaller values allow for more adaptation.",
        examples=[30],
        default=30,
    )

    detection_zone: Union[list, str, Selector(kind=[ZONE_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        title="Detection Zone",
        description="An optional polygon zone in a format [[x1, y1], [x2, y2], [x3, y3], ...];"
        " each zone must consist of more than 3 points",
        default=None,
    )

    suppress_first_detections: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(  # type: ignore
        title="Don't Detect Until History is Full",
        description="Suppress motion detections until the background history is fully initialized.",
        examples=[True],
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
            ),
            OutputDefinition(
                name="alarm",
                kind=[
                    BOOLEAN_KIND,
                ],
            ),
            OutputDefinition(
                name="detections",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name="motion_zones",
                kind=[
                    LIST_OF_VALUES_KIND,
                ],
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

        frames_initialized = (
            self.frame_count >= history or not suppress_first_detections
        )
        if not frames_initialized:
            self.frame_count += 1
            return {
                "motion": False,
                "detections": sv.Detections.empty(),
                "alarm": False,
                "motion_zones": [],
            }

        frame = image.numpy_image

        # apply background subtraction
        mask = self.back_sub.apply(frame)

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
            if len(contour) > 0 and cv2.contourArea(contour) > minimum_contour_area
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
        mask_height, mask_width, _ = frame.shape
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

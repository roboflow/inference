import copy
import json
from typing import List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field
from shapely.geometry import Polygon

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
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

SHORT_DESCRIPTION: str = "Detect motion in an image using traditional CV."
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
        validation_alias=AliasChoices("minimum_contour_area"),
        default=200,
    )

    morphological_kernel_size: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Morphological Kernel Size",
        description="The size of the kernel used for morphological operations to combine contours.",
        examples=[3],
        validation_alias=AliasChoices("morphological_kernel_size"),
        default=3,
    )

    threshold: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Threshold",
        description="The threshold value for the squared Mahalanobis distance for background subtraction."
        " Smaller values increase sensitivity to motion.",
        examples=[16],
        validation_alias=AliasChoices("threshold"),
        default=16,
    )

    history: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="History",
        description="The number of previous frames to use for background subtraction.",
        examples=[30],
        validation_alias=AliasChoices("history"),
        default=30,
    )

    detection_zone: Union[list, str, Selector(kind=[ZONE_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        title="Detection Zone",
        description="An optional polygon zone in a format [[x1, y1], [x2, y2], [x3, y3], ...];"
        " each zone must consist of more than 3 points",
        examples=["$inputs.zones"],
        default=None,
    )

    suppress_first_detections: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(  # type: ignore
        title="Don't Detect Until History is Full",
        description="Suppress motion detections until the background history is fully initialized.",
        examples=["$inputs.zones"],
        default=True,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
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
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MotionDetectionBlockV1(WorkflowBlock):
    NOISE_FILTER_THRESHOLD = 32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_motion = False
        self.backSub = None
        self.threshold = None
        self.history = None
        self.frame_count = 0

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

        if type(detection_zone) == str:
            try:
                detection_zone = json.loads(detection_zone)
            except Exception as e:
                raise ValueError(
                    f"Could not parse zone as a valid json: {detection_zone}"
                )

        if not self.backSub or self.threshold != threshold or self.history != history:
            self.threshold = threshold
            self.history = history
            self.backSub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=threshold, detectShadows=True
            )

        frames_initialized = (
            self.frame_count >= history or not suppress_first_detections
        )
        if not frames_initialized:
            self.frame_count += 1
            return {
                OUTPUT_IMAGE_KEY: copy.copy(image),
                "motion": False,
                "detections": sv.Detections.empty(),
                "alarm": False,
            }

        frame = image.numpy_image

        # apply background subtraction
        mask = self.backSub.apply(frame)

        # filter out the minimal grayscale values to reduce noise
        # not exposing this as a param for simplicity - overall sensitivity can be adjusted via the main threshold param
        _, mask_thresh = cv2.threshold(
            mask, self.NOISE_FILTER_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        # apply morphological filtering to ignore changes due to noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morphological_kernel_size, morphological_kernel_size)
        )
        mask_morph = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        # create contours around filtered areas
        contours, hierarchy = cv2.findContours(
            mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # apply minimum contour size and filter out 0 length contours
        filtered_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) > minimum_contour_area and len(contour) > 0
        ]

        # clip contours if a detection zone is provided
        if detection_zone and len(detection_zone) > 0:
            filtered_contours = clip_contours_to_contour(
                filtered_contours, detection_zone
            )

        if len(filtered_contours) > 0:
            # draw contours on output image
            frame_ct = cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 2)
            # create output workflow image
            output_image = WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=frame_ct,
            )
        else:
            # if no contours, output a copy of the input image
            output_image = copy.copy(image)

        # get bounding boxes
        xyxy_boxes = []
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            xyxy_boxes.append([x, y, x + w, y + h])

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
        alarm = True if not self.last_motion and motion else False
        self.last_motion = motion

        return {
            OUTPUT_IMAGE_KEY: output_image,
            "motion": motion,
            "detections": detections,
            "alarm": alarm,
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

        except:
            continue

    return result


def list_to_contour(list_of_tuples):
    points = [[int(n) for n in xy_tuple] for xy_tuple in list_of_tuples]
    return np.array(points).reshape(-1, 1, 2)

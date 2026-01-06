from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.gaze import GazeDetectionInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    add_inference_keypoints_to_sv_detections,
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    load_core_model,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Detect faces and estimate gaze direction (where people are looking) using the L2CS gaze detection model, returning facial landmarks and gaze angles.

## How This Block Works

This block uses the L2CS gaze detection model to analyze faces and determine where a person is looking. The block:

1. Takes images as input (supports batch processing)
2. Optionally detects faces in the image (can be disabled if input images are pre-cropped to faces)
3. Detects facial landmarks (keypoints) for each face
4. Estimates gaze direction using yaw (horizontal angle) and pitch (vertical angle) measurements
5. Returns face predictions with landmarks, yaw angles, and pitch angles for each detected face

Gaze direction is represented by two angles: **yaw** (horizontal gaze, -180° to 180°, negative values indicate looking left) and **pitch** (vertical gaze, -90° to 90°, negative values indicate looking down). These angles tell you where each person is looking relative to the camera. The block can work with full images (detecting faces automatically) or with pre-cropped face images for more precise control.

## Common Use Cases

- **Attention and Engagement Tracking**: Monitor where people are looking in video calls, meetings, or presentations to measure attention and engagement levels
- **Driver Monitoring Systems**: Detect driver distraction, drowsiness, or if a driver is looking away from the road in automotive safety applications
- **User Interface Interaction**: Track where users are looking on screens or displays to improve UI/UX design or enable gaze-based controls
- **Retail and Marketing Analytics**: Analyze customer attention in retail environments, tracking which products or displays capture gaze to optimize store layouts
- **Accessibility Applications**: Build gaze-controlled interfaces for users with mobility limitations, allowing them to interact with devices using eye movement
- **Security and Surveillance**: Detect suspicious behavior by monitoring gaze patterns, such as people repeatedly looking in specific directions or avoiding eye contact

## Connecting to Other Blocks

The face predictions and gaze angles from this block can be connected to:

- **Conditional logic blocks** (e.g., Continue If) to trigger actions based on gaze direction thresholds (e.g., alert if driver looks away for too long, or trigger UI changes when user looks at specific areas)
- **Filter blocks** to filter faces based on gaze angles (e.g., only process faces looking in a specific direction)
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log gaze data for analytics or compliance purposes
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when specific gaze patterns are detected (e.g., driver distraction warnings)
- **Visualization blocks** (e.g., Keypoint Visualization) to overlay facial landmarks and gaze direction vectors on images for analysis or debugging
- **Object detection blocks** combined with crop blocks to first detect faces, then crop them before running gaze detection for improved accuracy

## Requirements

This block only supports local execution mode. Remote execution is not currently supported.
"""


def convert_gaze_detections_to_sv_detections_and_angles(
    images: Batch[WorkflowImageData],
    gaze_predictions: List[dict],
) -> Tuple[List[sv.Detections], List[List[float]], List[List[float]]]:
    """Convert gaze detection results to supervision detections and angle lists."""
    face_predictions = []
    yaw_degrees = []
    pitch_degrees = []

    for single_image, predictions in zip(images, gaze_predictions):
        height, width = single_image.numpy_image.shape[:2]

        # Format predictions for this image
        image_face_preds = {
            "predictions": [],
            "image": {"width": width, "height": height},
        }
        batch_yaw = []
        batch_pitch = []

        for p in predictions:  # predictions is already a list
            p_dict = p.model_dump(by_alias=True, exclude_none=True)
            for pred in p_dict["predictions"]:
                face = pred["face"]

                # Face detection with landmarks
                face_pred = {
                    "x": face["x"],
                    "y": face["y"],
                    "width": face["width"],
                    "height": face["height"],
                    "confidence": face["confidence"],
                    "class": "face",
                    "class_id": 0,
                    "keypoints": [
                        {
                            "x": l["x"],
                            "y": l["y"],
                            "confidence": face["confidence"],
                            "class": str(i),
                            "class_id": i,
                        }
                        for i, l in enumerate(face["landmarks"])
                    ],
                }

                image_face_preds["predictions"].append(face_pred)

                # Store angles in degrees
                batch_yaw.append(pred["yaw"] * 180 / np.pi)
                batch_pitch.append(pred["pitch"] * 180 / np.pi)

        face_predictions.append(image_face_preds)
        yaw_degrees.append(batch_yaw)
        pitch_degrees.append(batch_pitch)

    # Process predictions
    face_preds = convert_inference_detections_batch_to_sv_detections(face_predictions)

    # Add keypoints to supervision detections
    for prediction, detections in zip(face_predictions, face_preds):
        add_inference_keypoints_to_sv_detections(
            inference_prediction=prediction["predictions"],
            detections=detections,
        )

    face_preds = attach_prediction_type_info_to_sv_detections_batch(
        predictions=face_preds,
        prediction_type="facial-landmark",
    )
    face_preds = attach_parents_coordinates_to_batch_of_sv_detections(
        images=images,
        predictions=face_preds,
    )

    return face_preds, yaw_degrees, pitch_degrees


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Gaze Detection",
            "version": "v1",
            "short_description": "Detect faces and estimate gaze direction",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["gaze", "face"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-eyes",
                "blockPriority": 13.5,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/gaze@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    do_run_face_detection: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Whether to run face detection on input images. Set to True to automatically detect faces in full images. Set to False if input images are already pre-cropped to contain only face regions, which can improve accuracy and processing speed. When False, the block assumes the entire image contains a single face and skips face detection.",
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="face_predictions",
                kind=[KEYPOINT_DETECTION_PREDICTION_KIND],
                description="Facial landmark predictions",
            ),
            OutputDefinition(
                name="yaw_degrees",
                kind=[FLOAT_KIND],
                description="Yaw angle in degrees (-180 to 180, negative is left)",
            ),
            OutputDefinition(
                name="pitch_degrees",
                kind=[FLOAT_KIND],
                description="Pitch angle in degrees (-90 to 90, negative is down)",
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class GazeBlockV1(WorkflowBlock):
    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        do_run_face_detection: bool,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                do_run_face_detection=do_run_face_detection,
            )
        else:
            raise ValueError(
                f"Unsupported step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        do_run_face_detection: bool,
    ) -> BlockResult:
        predictions = []

        for single_image in images:
            inference_request = GazeDetectionInferenceRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                do_run_face_detection=do_run_face_detection,
                api_key=self._api_key,
            )
            gaze_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="gaze",
            )
            prediction = self._model_manager.infer_from_request_sync(
                gaze_model_id, inference_request
            )
            predictions.append(prediction)

        # Convert predictions to supervision format and get angles
        face_preds, yaw_degrees, pitch_degrees = (
            convert_gaze_detections_to_sv_detections_and_angles(
                images=images,
                gaze_predictions=predictions,
            )
        )

        return [
            {
                "face_predictions": face_pred,
                "yaw_degrees": yaw,
                "pitch_degrees": pitch,
            }
            for face_pred, yaw, pitch in zip(face_preds, yaw_degrees, pitch_degrees)
        ]

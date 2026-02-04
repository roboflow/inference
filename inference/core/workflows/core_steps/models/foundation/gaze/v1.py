from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.gaze import GazeDetectionInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference_sdk import InferenceHTTPClient
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
Run L2CS Gaze detection model on faces in images.

This block can:
1. Detect faces in images and estimate their gaze direction
2. Estimate gaze direction on pre-cropped face images

The gaze direction is represented by yaw and pitch angles in degrees.
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
        description="Whether to run face detection. Set to False if input images are pre-cropped face images.",
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
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                do_run_face_detection=do_run_face_detection,
            )
        else:
            raise ValueError(
                f"Unsupported step execution mode: {self._step_execution_mode}"
            )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        do_run_face_detection: bool,
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        else:
            client.select_api_v1()

        inference_images = [i.base64_image for i in images]
        predictions = client.detect_gazes(inference_input=inference_images)

        if not isinstance(predictions, list):
            predictions = [predictions]

        # Process remote predictions into the expected format
        return self._process_remote_predictions(
            images=images,
            predictions=predictions,
        )

    def _process_remote_predictions(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        """Process predictions from remote execution into the expected format."""
        face_predictions = []
        yaw_degrees = []
        pitch_degrees = []

        for single_image, prediction in zip(images, predictions):
            height, width = single_image.numpy_image.shape[:2]

            image_face_preds = {
                "predictions": [],
                "image": {"width": width, "height": height},
            }
            batch_yaw = []
            batch_pitch = []

            for pred in prediction.get("predictions", []):
                face = pred.get("face", {})

                face_pred = {
                    "x": face.get("x", 0),
                    "y": face.get("y", 0),
                    "width": face.get("width", 0),
                    "height": face.get("height", 0),
                    "confidence": face.get("confidence", 0),
                    "class": "face",
                    "class_id": 0,
                    "keypoints": [
                        {
                            "x": l.get("x", 0),
                            "y": l.get("y", 0),
                            "confidence": face.get("confidence", 0),
                            "class": str(i),
                            "class_id": i,
                        }
                        for i, l in enumerate(face.get("landmarks", []))
                    ],
                }

                image_face_preds["predictions"].append(face_pred)

                # Store angles in degrees (remote already returns radians)
                batch_yaw.append(pred.get("yaw", 0) * 180 / np.pi)
                batch_pitch.append(pred.get("pitch", 0) * 180 / np.pi)

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

        return [
            {
                "face_predictions": face_pred,
                "yaw_degrees": yaw,
                "pitch_degrees": pitch,
            }
            for face_pred, yaw, pitch in zip(face_preds, yaw_degrees, pitch_degrees)
        ]

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

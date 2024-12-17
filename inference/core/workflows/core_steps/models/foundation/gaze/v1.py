from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.gaze import GazeDetectionInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
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
    IMAGE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Run gaze detection on faces in images.

This block can:
1. Detect faces in images and estimate their gaze direction
2. Estimate gaze direction on pre-cropped face images

The gaze direction is represented by yaw and pitch angles in radians.
"""

class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Gaze Detection Model",
            "version": "v1", 
            "short_description": "Detect faces and estimate gaze direction",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["gaze", "face"],
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
                name="predictions",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
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
            raise NotImplementedError(
                "Remote execution is not supported for Gaze Detection. Run a local or dedicated inference server to use this block."
            )
        else:
            raise ValueError(f"Unknown step execution mode: {self._step_execution_mode}")

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
            height, width = single_image.numpy_image.shape[:2]
            
            # Format predictions for supervision - one flat list per image
            image_predictions = {
                "predictions": [],
                "image": {
                    "width": width,
                    "height": height
                }
            }
            
            for p in prediction:
                p_dict = p.model_dump(by_alias=True, exclude_none=True)
                for pred in p_dict["predictions"]:
                    face = pred["face"]
                    detection = {
                        "x": face["x"],
                        "y": face["y"],
                        "width": face["width"],
                        "height": face["height"],
                        "confidence": face["confidence"],
                        "class": "face",
                        "class_id": 0,
                        "gaze": {
                            "yaw": pred["yaw"],
                            "pitch": pred["pitch"]
                        }
                    }
                    image_predictions["predictions"].append(detection)
            
            predictions.append(image_predictions)

        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="gaze-detection",
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        
        return [{"predictions": prediction} for prediction in predictions]

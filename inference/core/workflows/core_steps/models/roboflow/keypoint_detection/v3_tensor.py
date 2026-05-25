from typing import List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field, PositiveInt, model_validator

from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

from inference.core.env import (
    HOSTED_DETECT_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.remote_response_converters import (
    class_id_to_name_from_responses,
    dict_response_to_key_points,
)
from inference.core.workflows.core_steps.common.tensor_prediction_metadata import (
    attach_prediction_metadata,
)
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run inference on a keypoint detection model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Keypoint Detection Model",
            "version": "v3",
            "short_description": "Predict skeletons on objects.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["yolo"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 4,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_keypoint_detection_model@v3"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence_mode: Union[
        Literal["default", "custom"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="default",
        json_schema_extra={
            "always_visible": True,
            "values_metadata": {
                "default": {
                    "name": "Default",
                    "description": "Use the model's built-in default threshold.",
                },
                "custom": {
                    "name": "Custom",
                    "description": "Specify a custom confidence threshold.",
                },
            },
        },
    )
    custom_confidence: Union[
        Optional[FloatZeroToOne],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        json_schema_extra={
            "relevant_for": {
                "confidence_mode": {
                    "values": ["custom"],
                    "required": True,
                },
            },
        },
    )
    keypoint_confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.0,
        description="Confidence threshold to predict a keypoint as visible.",
        examples=[0.3, "$inputs.keypoint_confidence"],
    )
    class_filter: Union[Optional[List[str]], Selector(kind=[LIST_OF_VALUES_KIND])] = (
        Field(
            default=None,
            description="List of accepted classes. Classes must exist in the model's training set.",
            examples=[["a", "b", "c"], "$inputs.class_filter"],
        )
    )
    iou_threshold: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Minimum overlap threshold between boxes to combine them into a single detection, used in NMS.",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=300,
        description="Maximum number of detections to return.",
        examples=[300, "$inputs.max_detections"],
    )
    class_agnostic_nms: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to specify if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    max_candidates: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )
    disable_active_learning: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to disable project-level active learning for this block.",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        Selector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for active learning, if enabled.",
        examples=["my_project", "$inputs.al_target_project"],
    )

    @model_validator(mode="after")
    def validate_custom_confidence(self) -> "BlockManifest":
        if self.confidence_mode == "custom" and self.custom_confidence is None:
            raise ValueError(
                "custom_confidence must be provided when confidence_mode is 'custom'"
            )
        return self

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["keypoint-detection"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(
                name="predictions", kind=[KEYPOINT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowKeypointDetectionModelBlockV3(WorkflowBlock):

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
        model_id: str,
        confidence_mode: str,
        custom_confidence: Optional[float],
        keypoint_confidence: Optional[float],
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        confidence = (
            custom_confidence if confidence_mode == "custom" else "default"
        )
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                confidence=confidence,
                keypoint_confidence=keypoint_confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                confidence=confidence,
                keypoint_confidence=keypoint_confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Union[None, float, Literal["default"]],
        keypoint_confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        tensor_inputs = [img.tensor_image for img in images]
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        # Adapter returns Tuple[List[KeyPoints], Optional[List[Detections]]].
        # The bbox list is informational alongside keypoints — block emits
        # KeyPoints as the canonical `predictions` output; bbox info per
        # instance lands in key_points_metadata via the remote-path converter,
        # and on the local path the bbox is already exposed through the
        # adapter's KeyPoints.key_points_metadata (model post_process fills it).
        result: Tuple[List[KeyPoints], Optional[List[Detections]]] = (
            self._model_manager.run_tensor_native_inference(
                model_id=model_id,
                images=tensor_inputs,
                input_color_format="rgb",
                confidence=confidence,
                key_points_threshold=keypoint_confidence,
                iou_threshold=iou_threshold,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                max_detections=max_detections,
                max_candidates=max_candidates,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        )
        predictions, _ = result
        class_names = dict(
            enumerate(self._model_manager.get_class_names(model_id=model_id))
        )
        results: BlockResult = []
        for image, prediction in zip(images, predictions):
            inference_id = attach_prediction_metadata(
                prediction,
                image=image,
                model_id=model_id,
                prediction_type="keypoint-detection",
                class_names=class_names,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": prediction,
                    "model_id": model_id,
                }
            )
        return results

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Union[None, float, Literal["default"]],
        keypoint_confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_DETECT_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        client_config = InferenceConfiguration(
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence_threshold=confidence,
            keypoint_confidence_threshold=keypoint_confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        non_empty_inference_images = [i.base64_image for i in images]
        responses = client.infer(
            inference_input=non_empty_inference_images,
            model_id=model_id,
        )
        if not isinstance(responses, list):
            responses = [responses]
        predictions = [dict_response_to_key_points(r) for r in responses]
        class_names = class_id_to_name_from_responses(responses)
        results: BlockResult = []
        for image, prediction in zip(images, predictions):
            inference_id = attach_prediction_metadata(
                prediction,
                image=image,
                model_id=model_id,
                prediction_type="keypoint-detection",
                class_names=class_names,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": prediction,
                    "model_id": model_id,
                }
            )
        return results

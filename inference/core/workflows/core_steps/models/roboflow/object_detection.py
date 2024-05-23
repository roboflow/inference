from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.utils import (
    anchor_prediction_detections_in_parent_coordinates,
    attach_parent_info,
    attach_prediction_type_info,
    filter_out_unwanted_classes_from_predictions_detections,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    FloatZeroToOne,
    FlowControl,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run inference on a object-detection model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Detect objects using an object detection model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["RoboflowObjectDetectionModel", "ObjectDetectionModel"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    model_id: Union[WorkflowParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            description="Roboflow model identifier",
            examples=["my_project/3", "$inputs.model"],
        )
    )
    class_agnostic_nms: Union[
        Optional[bool], WorkflowParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Value to decide if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    class_filter: Union[
        Optional[List[str]], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=None,
        description="List of classes to retrieve from predictions (to define subset of those which was used while model training)",
        examples=[["a", "b", "c"], "$inputs.class_filter"],
    )
    confidence: Union[
        FloatZeroToOne,
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    iou_threshold: Union[
        FloatZeroToOne,
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Parameter of NMS, to decide on minimum box intersection over union to merge boxes",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=300,
        description="Maximum number of detections to return",
        examples=[300, "$inputs.max_detections"],
    )
    max_candidates: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )
    disable_active_learning: Union[
        bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Parameter to decide if Active Learning data sampling is disabled for the model",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        WorkflowParameterSelector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for Active Learning data sampling - see Roboflow Active Learning "
        "docs for more information",
        examples=["my_project", "$inputs.al_target_project"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="prediction_type", kind=[BATCH_OF_PREDICTION_TYPE_KIND]
            ),
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
        ]


class RoboflowObjectDetectionModelBlock(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
    ):
        self._model_manager = model_manager
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        images: List[dict],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        request = ObjectDetectionInferenceRequest(
            api_key=self._api_key,
            model_id=model_id,
            image=images,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            source="workflow-execution",
        )
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        predictions = await self._model_manager.infer_from_request(
            model_id=model_id, request=request
        )
        if isinstance(predictions, list):
            predictions = [
                e.dict(by_alias=True, exclude_none=True) for e in predictions
            ]
        else:
            predictions = [predictions.dict(by_alias=True, exclude_none=True)]
        return self._post_process_result(
            images=images,
            predictions=predictions,
            class_filter=class_filter,
        )

    async def run_remotely(
        self,
        images: List[Dict[str, Any]],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CLASSIFICATION_URL
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
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        inference_input = [i["value"] for i in images]
        predictions = await client.infer_async(
            inference_input=inference_input,
            model_id=model_id,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        return self._post_process_result(
            images=images,
            predictions=predictions,
            class_filter=class_filter,
        )

    def _post_process_result(
        self,
        images: List[dict],
        predictions: List[dict],
        class_filter: Optional[List[str]],
    ) -> List[dict]:
        predictions = attach_prediction_type_info(
            predictions=predictions,
            prediction_type="object-detection",
        )
        predictions = filter_out_unwanted_classes_from_predictions_detections(
            predictions=predictions,
            classes_to_accept=class_filter,
        )
        predictions = attach_parent_info(
            images=images,
            predictions=predictions,
        )
        return anchor_prediction_detections_in_parent_coordinates(
            image=images,
            predictions=predictions,
        )

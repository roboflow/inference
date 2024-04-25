from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field, PositiveInt

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.steps_executors.models import (
    attach_parent_info,
    attach_prediction_type_info,
)
from inference.enterprise.workflows.core_steps.common.utils import (
    anchor_detections_in_parent_coordinates,
    filter_out_unwanted_classes,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_METADATA_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    FloatZeroToOne,
    FlowControl,
    InferenceImageSelector,
    InferenceParameterSelector,
    OutputStepImageSelector,
)
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run inference on a multi-label classification model hosted on or uploaded to Roboflow.

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
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    model_id: Union[InferenceParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            description="Roboflow model identifier",
            examples=["my_project/3", "$inputs.model"],
        )
    )
    class_agnostic_nms: Union[
        Optional[bool], InferenceParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Value to decide if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    class_filter: Union[
        Optional[List[str]], InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=None,
        description="List of classes to retrieve from predictions (to define subset of those which was used while model training)",
        examples=[["a", "b", "c"], "$inputs.class_filter"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    iou_threshold: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Parameter of NMS, to decide on minimum box intersection over union to merge boxes",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[
        Optional[PositiveInt], InferenceParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=300,
        description="Maximum number of detections to return",
        examples=[300, "$inputs.max_detections"],
    )
    max_candidates: Union[
        Optional[PositiveInt], InferenceParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )
    disable_active_learning: Union[
        Optional[bool], InferenceParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Parameter to decide if Active Learning data sampling is disabled for the model",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        InferenceParameterSelector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for Active Learning data sampling - see Roboflow Active Learning "
        "docs for more information",
        examples=["my_project", "$inputs.al_target_project"],
    )


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
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name="predictions_parent_coordinates",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
        ]

    async def run_locally(
        self,
        image: List[dict],
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
            image=image,
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
        result = await self._model_manager.infer_from_request(
            model_id=model_id, request=request
        )
        if issubclass(type(result), list):
            serialised_result = [
                e.dict(by_alias=True, exclude_none=True) for e in result
            ]
        else:
            serialised_result = [result.dict(by_alias=True, exclude_none=True)]
        return self._post_process_result(
            image=image,
            class_filter=class_filter,
            serialised_result=serialised_result,
        )

    async def run_remotely(
        self,
        image: List[Dict[str, Any]],
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
        inference_input = [i["value"] for i in image]
        results = await client.infer_async(
            inference_input=inference_input,
            model_id=model_id,
        )
        if not issubclass(type(results), list):
            results = [results]
        return self._post_process_result(
            image=image, class_filter=class_filter, serialised_result=results
        )

    def _post_process_result(
        self,
        image: List[dict],
        class_filter: Optional[List[str]],
        serialised_result: List[dict],
    ) -> List[dict]:
        serialised_result = attach_prediction_type_info(
            results=serialised_result,
            prediction_type="object-detection",
        )
        serialised_result = filter_out_unwanted_classes(
            serialised_result=serialised_result,
            classes_to_accept=class_filter,
        )
        serialised_result = attach_parent_info(image=image, results=serialised_result)
        return anchor_detections_in_parent_coordinates(
            image=image,
            serialised_result=serialised_result,
        )

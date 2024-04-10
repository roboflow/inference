from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.core.entities.requests.inference import ClassificationInferenceRequest
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
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    TOP_CLASS_KIND,
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


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "This block represents inference from Roboflow multi-class classification model.",
            "docs": "https://inference.roboflow.com/workflows/classify_objects",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["ClassificationModel"]
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
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
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


class RoboflowClassificationBlock(WorkflowBlock):

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
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="top", kind=[TOP_CLASS_KIND]),
            OutputDefinition(name="confidence", kind=[FLOAT_ZERO_TO_ONE_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    async def run_locally(
        self,
        image: List[dict],
        model_id: str,
        confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        request = ClassificationInferenceRequest(
            api_key=self._api_key,
            model_id=model_id,
            image=image,
            confidence=confidence,
            disable_active_learning=disable_active_learning,
            source="workflow-execution",
            active_learning_target_dataset=active_learning_target_dataset,
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
            serialised_result=serialised_result,
            image=image,
        )

    async def run_remotely(
        self,
        image: List[dict],
        model_id: str,
        confidence: Optional[float],
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
            confidence_threshold=confidence,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
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
        return self._post_process_result(image=image, serialised_result=results)

    def _post_process_result(
        self,
        image: List[dict],
        serialised_result: List[dict],
    ) -> List[dict]:
        serialised_result = attach_prediction_type_info(
            results=serialised_result,
            prediction_type="classification",
        )
        serialised_result = attach_parent_info(
            image=image, results=serialised_result, nested_key=None
        )
        return serialised_result

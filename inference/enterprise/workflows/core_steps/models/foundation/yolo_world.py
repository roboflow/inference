from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.steps_executors.models import (
    attach_parent_info,
    attach_prediction_type_info,
    load_core_model,
)
from inference.enterprise.workflows.complier.steps_executors.utils import make_batches
from inference.enterprise.workflows.core_steps.common.utils import (
    anchor_detections_in_parent_coordinates,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_METADATA_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    STRING_KIND,
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
            "description": "Block that make it possible to use YoloWorld model within `workflows` - providing real-time, zero-shot object detection.",
            "docs": "https://inference.roboflow.com/workflows/yolo_world",
            "block_type": "model",
        }
    )
    type: Literal["YoloWorldModel", "YoloWorld"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    class_names: Union[
        InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]
    ] = Field(
        description="List of classes to use YoloWorld model against",
        examples=[["a", "b", "c"], "$inputs.class_names"],
    )
    version: Union[
        Literal["s", "m", "l", "x", "v2-s", "v2-m", "v2-l", "v2-x"],
        InferenceParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        default="l",
        description="Variant of YoloWorld model",
        examples=["l", "$inputs.variant"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for detections",
        examples=[0.3, "$inputs.confidence"],
    )


class YoloWorldModelBlock(WorkflowBlock):

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
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="predictions_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    async def run_locally(
        self,
        image: List[dict],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        serialised_result = []
        for single_image in image:
            inference_request = YOLOWorldInferenceRequest(
                image=single_image,
                yolo_world_version_id=version,
                confidence=confidence,
                text=class_names,
            )
            yolo_world_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="yolo_world",
                api_key=self._api_key,
            )
            result = await self._model_manager.infer_from_request(
                yolo_world_model_id, inference_request
            )
            serialised_result.append(result.dict(by_alias=True, exclude_none=True))
        return self._post_process_result(
            image=image, serialised_result=serialised_result
        )

    async def run_remotely(
        self,
        image: List[dict],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
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
        configuration = InferenceConfiguration(
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        client.configure(inference_configuration=configuration)
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        image_batches = list(
            make_batches(
                iterable=image,
                batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            )
        )
        serialised_result = []
        for single_batch in image_batches:
            batch_results = await client.infer_from_yolo_world_async(
                inference_input=[i["value"] for i in single_batch],
                class_names=class_names,
                model_version=version,
                confidence=confidence,
            )
            serialised_result.extend(batch_results)
        return self._post_process_result(
            image=image, serialised_result=serialised_result
        )

    def _post_process_result(
        self,
        image: List[dict],
        serialised_result: List[dict],
    ) -> List[dict]:
        serialised_result = attach_prediction_type_info(
            results=serialised_result,
            prediction_type="object-detection",
        )
        serialised_result = attach_parent_info(image=image, results=serialised_result)
        return anchor_detections_in_parent_coordinates(
            image=image,
            serialised_result=serialised_result,
        )

import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.steps_executors.models import (
    load_core_model,
)
from inference.enterprise.workflows.complier.steps_executors.utils import make_batches
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    LIST_OF_VALUES_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    FlowControl,
    InferenceImageSelector,
    InferenceParameterSelector,
    OutputStepImageSelector,
)
from inference.enterprise.workflows.steps.common.utils import (
    attach_parent_info,
    attach_prediction_type_info,
)
from inference_sdk import InferenceHTTPClient


class BlockManifest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Block to execute comparison of Clip embeddings between image and text.",
            "docs": "https://inference.roboflow.com/workflows/compare_clip_vectors",
            "block_type": "model",
        }
    )
    type: Literal["ClipComparison"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    text: Union[InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]] = (
        Field(
            description="List of texts to calculate similarity against each input image",
            examples=[["a", "b", "c"], "$inputs.texts"],
        )
    )


class ClipComparisonBlock:

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
    def get_input_manifest(cls) -> Type[BaseModel]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="similarity", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="predictions_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    async def run_locally(
        self,
        image: List[dict],
        text: List[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        serialised_result = []
        for single_image in image:
            inference_request = ClipCompareRequest(
                subject=single_image,
                subject_type="image",
                prompt=text,
                prompt_type="text",
            )
            doctr_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="clip",
                api_key=self._api_key,
            )
            result = await self._model_manager.infer_from_request(
                doctr_model_id, inference_request
            )
            serialised_result.append(result.dict())
        return self._post_process_result(
            image=image, serialised_result=serialised_result
        )

    async def run_remotely(
        self,
        image: List[dict],
        text: List[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
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
        image_batches = list(
            make_batches(
                iterable=image,
                batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            )
        )
        serialised_result = []
        for single_batch in image_batches:
            coroutines = []
            for single_image in single_batch:
                coroutine = client.clip_compare_async(
                    subject=single_image["value"],
                    prompt=text,
                )
                coroutines.append(coroutine)
            batch_results = list(await asyncio.gather(*coroutines))
            serialised_result.extend(batch_results)
        return self._post_process_result(
            image=image, serialised_result=serialised_result
        )

    def _post_process_result(
        self,
        image: List[dict],
        serialised_result: List[dict],
    ) -> List[dict]:
        serialised_result = attach_parent_info(
            image=image,
            results=serialised_result,
            nested_key=None,
        )
        return attach_prediction_type_info(
            results=serialised_result,
            prediction_type="embeddings-comparison",
        )

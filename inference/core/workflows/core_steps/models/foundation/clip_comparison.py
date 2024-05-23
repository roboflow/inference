import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import AliasChoices, ConfigDict, Field

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.utils import (
    attach_parent_info,
    attach_prediction_type_info,
    load_core_model,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    LIST_OF_VALUES_KIND,
    FlowControl,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.utils.iterables import make_batches

LONG_DESCRIPTION = """
Use the OpenAI CLIP zero-shot classification model to classify images.

This block accepts an image and a list of text prompts. The block then returns the 
similarity of each text label to the provided image.

This block is useful for classifying images without having to train a fine-tuned 
classification model. For example, you could use CLIP to classify the type of vehicle 
in an image, or if an image contains NSFW material.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Compare CLIP image and text embeddings.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["ClipComparison"]
    name: str = Field(description="Unique name of step in workflows")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    texts: Union[WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]] = (
        Field(
            description="List of texts to calculate similarity against each input image",
            examples=[["a", "b", "c"], "$inputs.texts"],
            validation_alias=AliasChoices("texts", "text"),
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="similarity", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(
                name="prediction_type", kind=[BATCH_OF_PREDICTION_TYPE_KIND]
            ),
        ]


class ClipComparisonBlock(WorkflowBlock):

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
        texts: List[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        predictions = []
        for single_image in images:
            inference_request = ClipCompareRequest(
                subject=single_image,
                subject_type="image",
                prompt=texts,
                prompt_type="text",
                api_key=self._api_key,
            )
            clip_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="clip",
            )
            prediction = await self._model_manager.infer_from_request(
                clip_model_id, inference_request
            )
            predictions.append(prediction.dict())
        return self._post_process_result(image=images, predictions=predictions)

    async def run_remotely(
        self,
        images: List[dict],
        texts: List[str],
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
        image_sub_batches = list(
            make_batches(
                iterable=images,
                batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            )
        )
        predictions = []
        for single_sub_batch in image_sub_batches:
            coroutines = []
            for single_image in single_sub_batch:
                coroutine = client.clip_compare_async(
                    subject=single_image["value"],
                    prompt=texts,
                )
                coroutines.append(coroutine)
            sub_batch_predictions = list(await asyncio.gather(*coroutines))
            predictions.extend(sub_batch_predictions)
        return self._post_process_result(image=images, predictions=predictions)

    def _post_process_result(
        self,
        image: List[dict],
        predictions: List[dict],
    ) -> List[dict]:
        predictions = attach_parent_info(
            images=image,
            predictions=predictions,
            nested_key=None,
        )
        return attach_prediction_type_info(
            predictions=predictions,
            prediction_type="embeddings-comparison",
        )

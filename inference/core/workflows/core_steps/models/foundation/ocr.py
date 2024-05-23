from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import AliasChoices, ConfigDict, Field

from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
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
    BATCH_OF_STRING_KIND,
    FlowControl,
    StepOutputImageSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
 Retrieve the characters in an image using Optical Character Recognition (OCR).

This block returns the text within an image.

You may want to use this block in combination with a detections-based block (i.e. 
ObjectDetectionBlock). An object detection model could isolate specific regions from an 
image (i.e. a shipping container ID in a logistics use case) for further processing. 
You can then use a CropBlock to crop the region of interest before running OCR.

Using a detections model then cropping detections allows you to isolate your analysis 
on particular regions of an image.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Run Optical Character Recognition on a model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["OCRModel"]
    name: str = Field(description="Unique name of step in workflows")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="result", kind=[BATCH_OF_STRING_KIND]),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(
                name="prediction_type", kind=[BATCH_OF_PREDICTION_TYPE_KIND]
            ),
        ]


class OCRModelBlock(WorkflowBlock):

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
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        serialised_result = []
        for single_image in images:
            inference_request = DoctrOCRInferenceRequest(
                image=single_image,
                api_key=self._api_key,
            )
            doctr_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="doctr",
            )
            result = await self._model_manager.infer_from_request(
                doctr_model_id, inference_request
            )
            serialised_result.append(result.dict())
        return self._post_process_result(
            predictions=serialised_result,
            image=images,
        )

    async def run_remotely(
        self,
        images: List[dict],
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
        configuration = InferenceConfiguration(
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        client.configure(configuration)
        predictions = await client.ocr_image_async(
            inference_input=[i["value"] for i in images],
        )
        if len(images) == 1:
            predictions = [predictions]
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
            prediction_type="ocr",
        )

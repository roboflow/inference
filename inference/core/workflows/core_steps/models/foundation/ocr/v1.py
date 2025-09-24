from typing import List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    load_core_model,
    remove_unexpected_keys_from_dictionary,
)
from inference.core.workflows.execution_engine.constants import (
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
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
You can then use a DynamicCropBlock to crop the region of interest before running OCR.

Using a detections model then cropping detections allows you to isolate your analysis 
on particular regions of an image.
"""

EXPECTED_OUTPUT_KEYS = {"result", "parent_id", "root_parent_id", "prediction_type"}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OCR Model",
            "version": "v1",
            "short_description": "Extract text from an image using optical character recognition.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inDevelopment": True,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/ocr_model@v1", "OCRModel"]
    name: str = Field(description="Unique name of step in workflows")
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="result", kind=[STRING_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OCRModelBlockV1(WorkflowBlock):
    # TODO: we need data model for OCR predictions

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
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(images=images)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(images=images)
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
    ) -> BlockResult:
        predictions = []
        for single_image in images:
            inference_request = DoctrOCRInferenceRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                api_key=self._api_key,
            )
            doctr_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="doctr",
            )
            result = self._model_manager.infer_from_request_sync(
                doctr_model_id, inference_request
            )
            predictions.append(result.model_dump())
        return self._post_process_result(
            predictions=predictions,
            images=images,
        )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
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
        configuration = InferenceConfiguration(
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        client.configure(configuration)
        non_empty_inference_images = [i.base64_image for i in images]
        predictions = client.ocr_image(
            inference_input=non_empty_inference_images,
        )
        if len(images) == 1:
            predictions = [predictions]
        return self._post_process_result(
            predictions=predictions,
            images=images,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        for prediction, image in zip(predictions, images):
            prediction[PREDICTION_TYPE_KEY] = "ocr"
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
            _ = remove_unexpected_keys_from_dictionary(
                dictionary=prediction,
                expected_keys=EXPECTED_OUTPUT_KEYS,
            )
        return predictions

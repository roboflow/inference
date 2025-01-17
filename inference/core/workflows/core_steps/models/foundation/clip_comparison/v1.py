from functools import partial
from typing import List, Literal, Optional, Type, Union

from pydantic import AliasChoices, ConfigDict, Field

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    load_core_model,
    remove_unexpected_keys_from_dictionary,
    run_in_parallel,
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
    LIST_OF_VALUES_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient

LONG_DESCRIPTION = """
Use the OpenAI CLIP zero-shot classification model to classify images.

This block accepts an image and a list of text prompts. The block then returns the 
similarity of each text label to the provided image.

This block is useful for classifying images without having to train a fine-tuned 
classification model. For example, you could use CLIP to classify the type of vehicle 
in an image, or if an image contains NSFW material.
"""

EXPECTED_OUTPUT_KEYS = {"similarity", "parent_id", "root_parent_id", "prediction_type"}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Clip Comparison",
            "version": "v1",
            "short_description": "Compare CLIP image and text embeddings.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "fak fa-message-image",
                "blockPriority": 10,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/clip_comparison@v1", "ClipComparison"]
    name: str = Field(description="Unique name of step in workflows")
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    texts: Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]] = Field(
        description="List of texts to calculate similarity against each input image",
        examples=[["a", "b", "c"], "$inputs.texts"],
        validation_alias=AliasChoices("texts", "text"),
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="similarity", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ClipComparisonBlockV1(WorkflowBlock):

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
        texts: List[str],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(images=images, texts=texts)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(images=images, texts=texts)
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        texts: List[str],
    ) -> BlockResult:
        predictions = []
        for single_image in images:
            inference_request = ClipCompareRequest(
                subject=single_image.to_inference_format(numpy_preferred=True),
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
            prediction = self._model_manager.infer_from_request_sync(
                clip_model_id, inference_request
            )
            predictions.append(prediction.model_dump())
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        texts: List[str],
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
        tasks = [
            partial(
                client.clip_compare,
                subject=single_image.numpy_image,
                prompt=texts,
            )
            for single_image in images
        ]
        predictions = run_in_parallel(
            tasks=tasks,
            max_workers=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        return self._post_process_result(images=images, predictions=predictions)

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> List[dict]:
        for prediction, image in zip(predictions, images):
            prediction[PREDICTION_TYPE_KEY] = "classification"
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
            # removing fields from `inference` response model
            # that are not registered as outputs
            _ = remove_unexpected_keys_from_dictionary(
                dictionary=prediction,
                expected_keys=EXPECTED_OUTPUT_KEYS,
            )
        return predictions

from functools import partial
from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

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
    run_in_parallel,
)
from inference.core.workflows.execution_engine.constants import (
    PARENT_ID_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    PARENT_ID_KIND,
    STRING_KIND,
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


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Clip Comparison",
            "version": "v2",
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
    type: Literal["roboflow_core/clip_comparison@v2"]
    name: str = Field(description="Unique name of step in workflows")
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    classes: Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]] = Field(
        description="List of classes to calculate similarity against each input image",
        examples=[["a", "b", "c"], "$inputs.texts"],
        min_items=1,
    )
    version: Union[
        Literal[
            "RN101",
            "RN50",
            "RN50x16",
            "RN50x4",
            "RN50x64",
            "ViT-B-16",
            "ViT-B-32",
            "ViT-L-14-336px",
            "ViT-L-14",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="ViT-B-16",
        description="Variant of CLIP model",
        examples=["ViT-B-16", "$inputs.variant"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="similarities", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="max_similarity", kind=[FLOAT_ZERO_TO_ONE_KIND]),
            OutputDefinition(name="most_similar_class", kind=[STRING_KIND]),
            OutputDefinition(name="min_similarity", kind=[FLOAT_ZERO_TO_ONE_KIND]),
            OutputDefinition(name="least_similar_class", kind=[STRING_KIND]),
            OutputDefinition(
                name="classification_predictions",
                kind=[CLASSIFICATION_PREDICTION_KIND],
            ),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ClipComparisonBlockV2(WorkflowBlock):

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
        classes: List[str],
        version: str,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(images=images, classes=classes, version=version)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(images=images, classes=classes, version=version)
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        classes: List[str],
        version: str,
    ) -> BlockResult:
        predictions = []
        for single_image in images:
            inference_request = ClipCompareRequest(
                clip_version_id=version,
                subject=single_image.to_inference_format(numpy_preferred=True),
                subject_type="image",
                prompt=classes,
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
            classes=classes,
        )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        classes: List[str],
        version: str,
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
                subject=single_image.base64_image,
                prompt=classes,
                clip_version=version,
            )
            for single_image in images
        ]
        predictions = run_in_parallel(
            tasks=tasks,
            max_workers=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        return self._post_process_result(
            images=images,
            predictions=predictions,
            classes=classes,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        classes: List[str],
    ) -> List[dict]:
        results = []
        for prediction, image in zip(predictions, images):
            similarities = prediction["similarity"]
            max_similarity = np.max(similarities)
            max_similarity_id = np.argmax(similarities)
            min_similarity = np.min(similarities)
            min_similarity_id = np.argmin(similarities)
            most_similar_class_name = classes[max_similarity_id]
            least_similar_class_name = classes[min_similarity_id]
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
            classification_predictions = {
                "predictions": [
                    {
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                    }
                    for class_id, (class_name, confidence) in enumerate(
                        zip(classes, similarities)
                    )
                ],
                "top": most_similar_class_name,
                "confidence": max_similarity,
                "parent_id": image.parent_metadata.parent_id,
            }
            result = {
                PARENT_ID_KEY: image.parent_metadata.parent_id,
                ROOT_PARENT_ID_KEY: image.workflow_root_ancestor_metadata.parent_id,
                "similarities": similarities,
                "max_similarity": max_similarity,
                "most_similar_class": most_similar_class_name,
                "min_similarity": min_similarity,
                "least_similar_class": least_similar_class_name,
                "classification_predictions": classification_predictions,
            }
            results.append(result)
        return results

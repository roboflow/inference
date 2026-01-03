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
Use CLIP to perform zero-shot image classification by comparing images with text labels and returning detailed similarity scores and classification predictions.

## How This Block Works

This block takes one or more images and a list of text labels (class names) as input, then uses OpenAI's CLIP model to compare the semantic similarity between each image and each text label. The block:

1. Takes images and a list of class names (e.g., ["car", "truck", "bicycle", "motorcycle"])
2. Generates embeddings for both the images and text labels using the specified CLIP model variant
3. Calculates similarity scores between each image and each class label
4. Identifies the most similar and least similar classes for each image
5. Returns structured classification predictions with confidence scores for all classes

The block provides enhanced outputs compared to v1, including the maximum and minimum similarity scores, the most and least similar class names, and structured classification predictions that can be directly used by other blocks. You can also select from multiple CLIP model variants (e.g., ViT-B-16, RN50, ViT-L-14) to balance accuracy and performance for your use case.

## Common Use Cases

- **Zero-Shot Classification**: Classify images into categories using text labels without training a custom classification model (e.g., classify vehicles as "car", "truck", "motorcycle", "bicycle")
- **Content Filtering**: Check if images contain specific content by comparing against text descriptions (e.g., "NSFW content", "violence", "safe for work")
- **Multi-Class Classification**: Classify images into multiple categories simultaneously by providing a list of class names and getting similarity scores for each
- **Custom Category Detection**: Detect custom object categories or attributes by describing them in text (e.g., "red car", "person wearing helmet", "outdoor scene")
- **Quality Assessment**: Evaluate image content against quality criteria described in text (e.g., "high quality", "blurry", "well-lit")
- **Content Moderation**: Automatically flag images that match certain text descriptions for moderation purposes

## Connecting to Other Blocks

The similarity scores and classification predictions from this block can be connected to:

- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on the most similar class or similarity thresholds
- **Filter blocks** to filter images based on classification results, max similarity scores, or specific class matches
- **Analytics blocks** (e.g., Data Aggregator) to analyze classification patterns over time
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log classification results with structured prediction data
- **Notification blocks** to send alerts when specific content is detected based on similarity scores or class matches
- **Classification blocks** that accept classification predictions as input, since this block outputs structured classification predictions

## Version Differences (v2 vs v1)

v2 introduces several enhancements over v1:

- **Enhanced Outputs**: Provides structured outputs including `max_similarity`, `most_similar_class`, `min_similarity`, `least_similar_class`, and `classification_predictions` in addition to raw similarity scores
- **Model Selection**: Adds a `version` parameter to select from multiple CLIP model variants (RN101, RN50, RN50x16, RN50x4, RN50x64, ViT-B-16, ViT-B-32, ViT-L-14-336px, ViT-L-14) for different accuracy/performance trade-offs
- **Structured Predictions**: Outputs `classification_predictions` in a format compatible with other classification blocks, including class names, IDs, confidences, and top prediction
- **Parameter Rename**: The `texts` parameter has been renamed to `classes` for clarity
- **Better Integration**: The structured outputs make it easier to connect to other workflow blocks that expect classification predictions
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
        description="List of class names (text labels) to compare against each input image. CLIP will calculate similarity scores between the image and each class, enabling zero-shot classification. Provide descriptive class names (e.g., ['car', 'truck', 'bicycle'] or ['NSFW content', 'safe content']). The block returns similarity scores for each class, with higher scores indicating better matches. At least one class must be provided.",
        examples=[["car", "truck", "bicycle"], ["NSFW", "safe"], "$inputs.classes"],
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
        description="CLIP model variant to use for embeddings. Different variants offer different trade-offs between accuracy and performance. ViT-B-16 (default) provides a good balance. ViT-L-14 variants offer higher accuracy but slower inference. RN (ResNet) variants are faster but may be less accurate. ViT-L-14-336px uses higher resolution input for better accuracy on detailed images.",
        examples=["ViT-B-16", "ViT-L-14", "RN50", "$inputs.clip_version"],
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

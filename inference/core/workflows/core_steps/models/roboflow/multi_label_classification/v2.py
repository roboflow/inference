from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import ClassificationInferenceRequest
from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import attach_prediction_type_info
from inference.core.workflows.execution_engine.constants import (
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    FloatZeroToOne,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run inference on a multi-label classification model hosted on or uploaded to Roboflow.

## How This Block Works

This block takes one or more images as input and runs them through a trained multi-label classification model. Unlike single-label classification (which assigns exactly one class per image), multi-label classification can assign multiple classes to a single image simultaneously. The model analyzes each image and returns all applicable class labels with their confidence scores. Each class has its own confidence score indicating how certain the model is about that specific label, and classes are not mutually exclusive - multiple tags can apply to the same image. The block returns:

- Multiple predicted classes (all applicable categories from the model's training classes)
- Confidence scores for each predicted class (how certain the model is about each label, typically from 0.0 to 1.0)
- Additional metadata including class IDs, prediction type, inference ID, and model ID

The model processes the entire image and outputs all applicable class labels per image, making it ideal for scenarios where images can belong to multiple categories simultaneously. For example, an image of a dog in a park could be labeled with multiple tags: ["dog", "outdoor", "daytime", "grass"] all at once.

## Common Use Cases

- **Content Tagging**: Tagging images with multiple attributes (e.g., ["sunset", "beach", "people", "summer"] for a beach photo)
- **Medical Imaging**: Identifying multiple conditions or features in medical scans (e.g., ["fracture", "edema", "foreign_object"])
- **Product Attributes**: Tagging products with multiple characteristics (e.g., ["red", "cotton", "long-sleeve", "casual"] for a shirt)
- **Scene Understanding**: Identifying multiple elements in a scene (e.g., ["indoor", "kitchen", "daytime", "person", "food"])
- **Quality Inspection**: Marking multiple quality attributes or defects (e.g., ["scratch", "discoloration", "acceptable"] for a manufactured part)
- **Social Media Content**: Categorizing posts with multiple tags (e.g., ["food", "restaurant", "vegetarian", "dinner"])

## Model Sources

You can use:

- Models from your private Roboflow account (requires authentication)
- Public models from [Roboflow Universe](https://universe.roboflow.com) (no authentication needed for public models)

## Requirements

You need to set your Roboflow API key in your Inference environment to use private models. To learn more about setting your Roboflow API key, [refer to the Inference documentation](https://inference.roboflow.com/quickstart/configure_api_key/).

## Connecting to Other Blocks

The classification results from this block can be connected to:

- **Filter blocks** to filter images or workflows based on the presence of specific labels or combinations of labels
- **Conditional logic blocks** to route workflow execution based on which labels are present (e.g., if "urgent" label exists, send notification)
- **Visualization blocks** to display all classification labels on images
- **Data storage blocks** to log multi-label results for analytics and searching
- **Search/retrieval blocks** to find images matching specific label combinations
- **Aggregation blocks** to count or analyze label distributions across batches of images

## Version Differences (v2 vs v1)

This version (v2) includes the `model_id` in the output, making it easier to track which model was used when chaining multiple classification models in a workflow. The `inference_id` output also uses the `INFERENCE_ID_KIND` instead of `STRING_KIND` for better type checking.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Multi-Label Classification Model",
            "version": "v2",
            "short_description": "Apply multiple tags to an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 3,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_multi_label_classification_model@v2"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Minimum confidence threshold (0.0-1.0) for predictions. Classes with confidence below this threshold are filtered out.",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    disable_active_learning: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to disable project-level active learning for this block.",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        Selector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for active learning, if enabled.",
        examples=["my_project", "$inputs.al_target_project"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowMultiLabelClassificationModelBlockV2(WorkflowBlock):

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
        model_id: str,
        confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                confidence=confidence,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                confidence=confidence,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
        request = ClassificationInferenceRequest(
            api_key=self._api_key,
            model_id=model_id,
            image=inference_images,
            confidence=confidence,
            disable_active_learning=disable_active_learning,
            source="workflow-execution",
            active_learning_target_dataset=active_learning_target_dataset,
        )
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        predictions = self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request, confidence=confidence
        )
        if isinstance(predictions, list):
            predictions = [
                e.dict(by_alias=True, exclude_none=True) for e in predictions
            ]
        else:
            predictions = [predictions.dict(by_alias=True, exclude_none=True)]
        return self._post_process_result(
            predictions=predictions,
            images=images,
            model_id=model_id,
        )

    def run_remotely(
        self,
        images: Batch[Optional[WorkflowImageData]],
        model_id: str,
        confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
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
        non_empty_inference_images = [i.base64_image for i in images]
        predictions = client.infer(
            inference_input=non_empty_inference_images,
            model_id=model_id,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        return self._post_process_result(
            images=images, predictions=predictions, model_id=model_id
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        model_id: str,
    ) -> List[dict]:
        predictions = attach_prediction_type_info(
            predictions=predictions,
            prediction_type="classification",
        )
        for prediction, image in zip(predictions, images):
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return [
            {
                "inference_id": prediction.get(INFERENCE_ID_KEY),
                "predictions": prediction,
                "model_id": model_id,
            }
            for prediction in predictions
        ]

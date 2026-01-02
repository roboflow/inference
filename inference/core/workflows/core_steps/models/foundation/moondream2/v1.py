import json
from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.moondream2 import Moondream2InferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    # SmolVLM needs an image and a text prompt.
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Union[
        Selector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Optional text prompt describing the objects you want to detect. Use natural language to describe object classes (e.g., 'person', 'car', 'dog', 'red bicycle'). If not provided, the model will use its default behavior. For multiple object types, you can describe them in a single prompt.",
        examples=["person", "car", "dog", "$inputs.prompt"],
        default=None,
    )

    # Standard model configuration for UI, schema, etc.
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Moondream2",
            "version": "v1",
            "short_description": "Run Moondream2 on an image.",
            "long_description": (
                """
Run Moondream2, a multimodal vision-language model that performs zero-shot object detection by understanding natural language prompts.

## How This Block Works

This block takes one or more images as input and processes them through Moondream2, a vision language model (VLM) that can understand both images and text simultaneously. Unlike most VLMs that output text responses, Moondream2 is specialized for **zero-shot object detection** - it can detect objects in images based on natural language descriptions without being specifically trained on those object classes. The block:

1. Encodes images for processing by the model
2. Applies your optional text prompt describing the objects you want to detect (e.g., "person", "car", "dog")
3. Processes the image(s) with the prompt to identify and localize the described objects
4. Returns object detection predictions with bounding boxes, class names, and confidence scores

Moondream2 is designed to be a lightweight, efficient model that provides flexible object detection capabilities. You can detect any object type described in your prompt without needing to train or configure a traditional object detection model for each class.

## Common Use Cases

- **Zero-Shot Detection**: Detect objects in images using natural language descriptions without training models for specific classes
- **Flexible Object Detection**: Detect custom object categories by simply describing them in prompts (e.g., "red car", "person wearing hat", "open door")
- **Rapid Prototyping**: Quickly test object detection for new categories without dataset preparation or model training
- **Multi-Class Detection**: Detect multiple object types in a single pass by describing them in your prompt
- **Content Filtering**: Identify specific content or objects in images for moderation or filtering purposes
- **Inventory Management**: Detect and count items in images using descriptive prompts (e.g., "product on shelf", "package on conveyor")

## Requirements

**⚠️ Important: Dedicated Inference Server Required**

This block requires **local execution** (cannot run remotely). A **GPU is recommended** for best performance. The model requires appropriate dependencies to be installed.

## Connecting to Other Blocks

The object detection predictions from this block can be connected to:

- **Visualization blocks** (e.g., Bounding Box Visualization) to draw detection results on images
- **Filter blocks** (e.g., Detections Filter) to filter detections based on confidence, class, or other criteria
- **Analytics blocks** (e.g., Data Aggregator) to count or aggregate detection results over time
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on detection results
- **Transformation blocks** (e.g., Dynamic Crop) to extract regions based on detected objects
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log detection results
"""
            ),
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Moondream2",
                "moondream",
                "vision language model",
                "VLM",
                "object detection",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/moondream2@v1"]

    model_version: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="moondream2/moondream2_2b_jul24",
        description="The Moondream2 model to use for inference. Default is 'moondream2/moondream2_2b_jul24'. Can also use other Moondream2 model variants or Roboflow model IDs for custom or fine-tuned models.",
        examples=["moondream2/moondream2_2b_jul24", "moondream2/moondream2-2b"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        # Only images can be passed in as a list/batch
        return ["images"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class Moondream2BlockV1(WorkflowBlock):
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
        model_version: str,
        prompt: Optional[str],
    ) -> BlockResult:
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_version=model_version,
                prompt=prompt,
            )
        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for moondream2. Please use a local or dedicated inference server."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
    ) -> BlockResult:
        # Convert each image to the format required by the model.
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]

        # Use the provided prompt (or an empty string if None) for every image.
        prompt = prompt or ""

        prompts = [prompt] * len(inference_images)

        # Register Moondream2 with the model manager.
        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image, single_prompt in zip(inference_images, prompts):
            request = Moondream2InferenceRequest(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                text=[],
                prompt=single_prompt,
            )
            # Run inference.
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            predictions.append(prediction.model_dump(by_alias=True, exclude_none=True))

        return self._post_process_result(images=images, predictions=predictions)

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="object-detection",
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [{"predictions": prediction} for prediction in predictions]

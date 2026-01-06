import json
from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


##########################################################################
# Qwen2.5-VL Workflow Block Manifest
##########################################################################
class BlockManifest(WorkflowBlockManifest):
    # Qwen2.5-VL only needs an image and an optional text prompt.
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text prompt or question to ask about the image(s). This is the main instruction for Qwen2.5-VL - you can ask questions, request descriptions, or provide specific analysis instructions. Examples: 'What is in this image?', 'Describe the scene', 'Are there any people?', 'Count the number of objects'. If not provided (None), the model will generate a general description of the image content. The prompt is combined with the system prompt before being sent to the model.",
        examples=[
            "What is in this image?",
            "Describe the scene",
            "Are there any people?",
            "Count the number of objects",
        ],
    )

    # Standard model configuration for UI, schema, etc.
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Qwen2.5-VL",
            "version": "v1",
            "short_description": "Run Qwen2.5-VL on an image.",
            "long_description": (
                """
Run Alibaba's Qwen2.5-VL model to analyze images and answer questions using natural language prompts.

## How This Block Works

This block takes one or more images as input and processes them through Alibaba's Qwen2.5-VL vision language model. The block processes each image individually:

1. **Receives images and prompts** - takes one or more images along with an optional text prompt and optional system prompt
2. **Combines prompts** - merges the text prompt and system prompt into a single combined prompt using a special separator
3. **Converts images** - transforms each image into the format required by the Qwen2.5-VL model
4. **Registers the model** - ensures the specified Qwen2.5-VL model version is loaded and ready for inference
5. **Runs inference** - processes each image with the combined prompt through the Qwen2.5-VL model using a conversation template
6. **Returns responses** - provides the model's text answer for each image as a dictionary output containing the parsed text response

The block supports flexible, free-form prompts - you can ask any question about the image, request descriptions, ask for analysis, or give specific instructions. If no prompt is provided, the model will generate a general description of the image content.

## Common Use Cases

- **Visual Question Answering**: Ask questions about image content - "What objects are in this image?", "How many people are visible?", "What is the person doing?"
- **Image Description**: Generate descriptions of images for accessibility, content indexing, or documentation
- **Content Analysis**: Analyze images for safety, quality, or compliance - "Does this image contain inappropriate content?", "Is this product damaged?"
- **Object Recognition**: Identify objects, landmarks, or scenes in images with natural language descriptions
- **Document Understanding**: Extract and understand text from images, analyze document structure, or answer questions about document content
- **Scene Understanding**: Understand complex scenes and relationships - "What activities are happening?", "What is the relationship between objects in this image?"

## Requirements

**⚠️ Important: Dedicated Inference Server Required**

This block requires **local execution** (cannot run remotely). A **GPU is highly recommended** for acceptable performance. You may want to use a dedicated deployment for Qwen2.5-VL models. The model requires appropriate dependencies to be installed (typically transformers, torch, and related packages).

## Connecting to Other Blocks

The text outputs from this block can be connected to:

- **Parser blocks** (e.g., JSON Parser v1) to extract structured information from Qwen2.5-VL's text responses if you prompt it to return JSON
- **Conditional logic blocks** to route workflow execution based on Qwen2.5-VL's responses
- **Filter blocks** to filter images or data based on the model's analysis
- **Visualization blocks** to display text overlays or annotations on images
- **Data storage blocks** to log responses for analytics or audit trails
- **Notification blocks** to send alerts based on Qwen2.5-VL's findings (e.g., specific content detected, quality issues identified)
"""
            ),
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Qwen2.5",
                "qwen2.5-vl",
                "vision language model",
                "VLM",
                "Alibaba",
            ],
            "is_vlm_block": True,
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/qwen25vl@v1"]

    model_version: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="qwen25-vl-7b",
        description="The Qwen2.5-VL model version to use for inference. Default is 'qwen25-vl-7b', which provides a good balance of performance and accuracy. You can also use Roboflow model IDs (format: 'workspace/model/version') for custom or fine-tuned Qwen2.5-VL models. The model will be registered with the model manager when the block runs.",
        examples=["qwen25-vl-7b"],
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to provide additional context or instructions that set the behavior, tone, or style for Qwen2.5-VL. This is combined with the main prompt before being sent to the model. Useful for controlling response format (e.g., 'Answer in one sentence', 'Use technical language'), setting the model's role (e.g., 'You are a helpful assistant.'), or providing domain-specific context. If not provided (None), only the main prompt is used.",
        examples=[
            "You are a helpful assistant.",
            "Answer concisely.",
            "Use technical language",
            "Answer in one sentence",
        ],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="parsed_output",
                kind=[DICTIONARY_KIND],
                description="A parsed version of the output, provided as a dictionary containing the text.",
            ),
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        # Only images can be passed in as a list/batch
        return ["images"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


##########################################################################
# Qwen2.5-VL Workflow Block
##########################################################################
class Qwen25VLBlockV1(WorkflowBlock):
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
        system_prompt: Optional[str],
    ) -> BlockResult:
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_version=model_version,
                prompt=prompt,
                system_prompt=system_prompt,
            )
        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for Qwen2.5-VL. Please use a local or dedicated inference server."
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
        system_prompt: Optional[str],
    ) -> BlockResult:
        # Convert each image to the format required by the model.
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        # Use the provided prompt or default to a generic image description request.
        prompt = prompt or "Describe what's in this image."
        system_prompt = (
            system_prompt
            or "You are a Qwen2.5-VL model that can answer questions about any image."
        )
        prompts = [prompt + "<system_prompt>" + system_prompt] * len(inference_images)
        # Register Qwen2.5-VL with the model manager.
        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image, single_prompt in zip(inference_images, prompts):
            # Build an LMMInferenceRequest with both prompt and image.
            request = LMMInferenceRequest(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                source="workflow-execution",
                prompt=single_prompt,
            )
            # Run inference.
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            response_text = prediction.response
            predictions.append(
                {
                    "parsed_output": response_text,
                }
            )
        return predictions

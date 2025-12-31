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
        description="Optional text prompt to provide additional context to Qwen2.5-VL. Otherwise it will just be None",
        examples=["What is in this image?"],
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

## What is a Vision Language Model (VLM)?

A Vision Language Model (VLM) is an AI model that can understand both **images and text** simultaneously. Unlike traditional computer vision models that are trained for a single task (like object detection or classification), VLMs like Qwen2.5-VL:
- **Understand natural language prompts** - you can ask questions or give instructions in plain English
- **Process visual content** - analyze images to understand what's in them
- **Generate flexible text responses** - provide natural language answers based on the image content
- **Support conversational interactions** - can follow up on questions and maintain context

This makes VLMs incredibly versatile and useful when you need flexible, natural language-driven computer vision without training separate models for each task.

## How This Block Works

This block takes one or more images as input and processes them through Alibaba's Qwen2.5-VL model. The block:
1. **Encodes images** for processing by the model
2. **Combines prompts** - merges your optional text prompt with an optional system prompt to provide context
3. **Sends the request to Qwen2.5-VL** - processes the image(s) with the combined prompt using a conversation template
4. **Returns the response** - provides the model's text answer as a parsed dictionary output

The block supports flexible, free-form prompts - you can ask any question about the image, request descriptions, ask for analysis, or give specific instructions.

## Inputs and Outputs

**Input:**
- **images**: One or more images to analyze (can be from workflow inputs or previous steps)
- **prompt**: Optional text prompt/question to ask about the image (e.g., "What is in this image?", "Describe the scene", "Are there any people?")
- **system_prompt**: Optional system prompt to provide additional context or instructions to the model (e.g., "You are a helpful assistant.", "Answer concisely.")
- **model_version**: Qwen2.5-VL model to use (default: "qwen25-vl-7b") - can also use Roboflow model IDs for custom/fine-tuned models

**Output:**
- **parsed_output**: A dictionary containing the text response from Qwen2.5-VL

## Key Configuration Options

- **prompt**: Your question or instruction in natural language - be specific about what you want Qwen2.5-VL to analyze or describe from the image. If not provided, the model will generate a general description
- **system_prompt**: Optional instructions that set the context or behavior for the model - useful for controlling the tone, style, or format of responses (e.g., "Answer in one sentence", "Use technical language")
- **model_version**: Choose the Qwen2.5-VL model - "qwen25-vl-7b" (default, good balance of performance and accuracy). Can also use Roboflow model IDs for custom or fine-tuned Qwen2.5-VL models

## Common Use Cases

- **Visual Question Answering**: Ask questions about image content - "What objects are in this image?", "How many people are visible?", "What is the person doing?"
- **Image Description**: Generate descriptions of images for accessibility, content indexing, or documentation
- **Content Analysis**: Analyze images for safety, quality, or compliance - "Does this image contain inappropriate content?", "Is this product damaged?"
- **Object Recognition**: Identify objects, landmarks, or scenes in images with natural language descriptions
- **Document Understanding**: Extract and understand text from images, analyze document structure, or answer questions about document content
- **Scene Understanding**: Understand complex scenes and relationships - "What activities are happening?", "What is the relationship between objects in this image?"

## Requirements

**⚠️ Important: Dedicated Inference Server Required**

This block requires **local execution** (cannot run remotely). A **GPU is highly recommended** for acceptable performance. You may want to use a dedicated deployment for Qwen2.5-VL models. The model requires appropriate dependencies to be installed.

## Connecting to Other Blocks

The text outputs from this block can be connected to:
- **Conditional logic blocks** to route workflow execution based on Qwen2.5-VL's responses
- **Filter blocks** to filter images or data based on the model's analysis
- **Visualization blocks** to display text overlays or annotations on images
- **Data storage blocks** to log responses for analytics or audit trails
- **Notification blocks** to send alerts based on Qwen2.5-VL's findings (e.g., specific content detected, quality issues identified)
- **Parser blocks** (e.g., JSON Parser) to extract structured information from Qwen2.5-VL's text responses if you prompt it to return JSON
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
        description="The Qwen2.5-VL model to be used for inference.",
        examples=["qwen25-vl-7b"],
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to provide additional context to Qwen2.5-VL.",
        examples=["You are a helpful assistant."],
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
        # Use the provided prompt (or an empty string if None) for every image.
        prompt = prompt or ""
        system_prompt = system_prompt or ""
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

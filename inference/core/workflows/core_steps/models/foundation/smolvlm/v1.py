from typing import List, Literal, Optional, Type, Union

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
    ROBOFLOW_MODEL_ID_KIND,
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
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text prompt or question to ask about the image. If not provided, the model will generate a general description. SmolVLM2 is particularly good at document OCR, visual question answering, and object counting tasks. Be specific about what you want the model to analyze or describe from the image.",
        examples=["What is in this image?", "How many objects are there?", "Extract text from this document"],
    )

    # Standard model configuration for UI, schema, etc.
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SmolVLM2",
            "version": "v1",
            "short_description": "Run SmolVLM2 on an image.",
            "long_description": (
                """
Run Hugging Face's SmolVLM2 model to analyze images and answer questions using natural language prompts with a lightweight, efficient vision language model.

## How This Block Works

This block takes one or more images as input and processes them through Hugging Face's SmolVLM2 model. SmolVLM2 is a vision language model (VLM) that can understand both images and text simultaneously, allowing you to ask questions or give instructions in plain English. Unlike traditional computer vision models trained for a single task, SmolVLM2 provides flexible text responses based on image content. The block:

1. Encodes images for processing by the model
2. Applies your optional text prompt (or empty string if not provided) to guide the model's analysis
3. Processes the image(s) with the prompt using the model's chat template format
4. Returns the model's text answer as a parsed dictionary output

SmolVLM2 is specifically designed to be a smaller, more efficient VLM (2.2 billion parameters) that provides strong performance while requiring fewer computational resources than larger VLMs. This makes it ideal for scenarios where you need VLM capabilities but have limited GPU memory or want faster inference times. The block supports flexible, free-form prompts - you can ask any question about the image, request descriptions, ask for analysis, or give specific instructions.

## Common Use Cases

- **Visual Question Answering**: Ask questions about image content - "What objects are in this image?", "How many dogs are visible?", "What is the person doing?"
- **Document OCR**: Extract and understand text from documents, forms, or images containing text
- **Document Question Answering**: Answer specific questions about document content - "What is the total amount on this invoice?", "What date is on this form?"
- **Object Counting**: Count objects in images - "How many people are in this crowd?", "How many items are on this shelf?"
- **Image Description**: Generate descriptions of images for accessibility, content indexing, or documentation
- **Content Analysis**: Analyze images for safety, quality, or compliance - "Does this image contain text?", "Is this document complete?"

## Requirements

**⚠️ Important: Dedicated Inference Server Required**

This block requires **local execution** (cannot run remotely). A **GPU is recommended** for best performance, though SmolVLM2's smaller size means it can run more efficiently than larger VLMs. The model requires appropriate dependencies (transformers library) to be installed.

## Connecting to Other Blocks

The text outputs from this block can be connected to:

- **Conditional logic blocks** to route workflow execution based on SmolVLM2's responses
- **Filter blocks** to filter images or data based on the model's analysis
- **Visualization blocks** to display text overlays or annotations on images
- **Data storage blocks** to log responses for analytics or audit trails
- **Notification blocks** to send alerts based on SmolVLM2's findings (e.g., specific content detected, object counts reached thresholds)
- **Parser blocks** (e.g., JSON Parser) to extract structured information from SmolVLM2's text responses if you prompt it to return structured data
"""
            ),
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "SmolVLM2",
                "smolvlm",
                "vision language model",
                "VLM",
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
    type: Literal["roboflow_core/smolvlm2@v1"]

    model_version: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="smolvlm2/smolvlm-2.2b-instruct",
        description="The SmolVLM2 model to use for inference. Default is 'smolvlm2/smolvlm-2.2b-instruct' (2.2B parameter model optimized for instruction following). Can also use Roboflow model IDs for custom or fine-tuned SmolVLM2 models.",
        examples=["smolvlm2/smolvlm-2.2b-instruct"],
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


class SmolVLM2BlockV1(WorkflowBlock):
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
                "Remote execution is not supported for SmolVLM2. Please use a local or dedicated inference server."
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

        # Register SmolVLM2 with the model manager.
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

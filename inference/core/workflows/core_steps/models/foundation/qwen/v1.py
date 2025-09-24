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
                "This workflow block runs Qwen2.5-VL—a vision language model that accepts an image "
                "and an optional text prompt—and returns a text answer based on a conversation template."
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

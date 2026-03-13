import json
from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
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
from inference_sdk import InferenceHTTPClient


##########################################################################
# Qwen3.5-VL Workflow Block Manifest
##########################################################################
class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Qwen3.5-VL",
            "version": "v1",
            "short_description": "Run Qwen3.5-VL on an image.",
            "long_description": (
                "This workflow block runs Qwen3.5-VL—a vision language model that accepts an image "
                "and an optional text prompt—and returns a text answer based on a conversation template."
            ),
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Qwen3.5",
                "qwen3.5-vl",
                "vision language model",
                "VLM",
                "Alibaba",
            ],
            "is_vlm_block": True,
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.7,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/qwen3_5vl@v1"]

    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text prompt to provide additional context to Qwen3.5-VL. Otherwise it will just be a default one, which may affect the desired model behavior.",
        examples=["What is in this image?"],
    )
    model_version: Union[
        Literal["qwen3_5-0.8b", "qwen3_5-2b"],
        Selector(kind=[ROBOFLOW_MODEL_ID_KIND]),
        str,
    ] = Field(
        default="qwen3_5-0.8b",
        description="The Qwen3.5-VL model to be used for inference.",
        examples=["qwen3_5-0.8b", "qwen3_5-2b"],
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to provide additional context to Qwen3.5-VL.",
        examples=["You are a helpful assistant."],
    )

    enable_thinking: bool = Field(
        default=False,
        description="If true, enables Qwen3.5-VL's thinking mode, which allows the model to generate reasoning tokens before answering. The thinking output will be returned in the 'thinking' field.",
        json_schema_extra={
            "relevant_for": {
                "model_version": {
                    "values": ["qwen3_5-2b", "qwen3_5-2b-peft"],
                },
            },
        },
    )

    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate. If not set, the model's default will be used. Consider increasing for thinking mode.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="parsed_output",
                kind=[DICTIONARY_KIND],
                description="A parsed version of the output, provided as a dictionary containing the text.",
            ),
            OutputDefinition(
                name="thinking",
                kind=[STRING_KIND],
                description="The model's thinking/reasoning output when enable_thinking is True. Empty string when enable_thinking is False.",
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
# Qwen3.5-VL Workflow Block
##########################################################################
class Qwen35VLBlockV1(WorkflowBlock):
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
        enable_thinking: bool = False,
        max_new_tokens: Optional[int] = None,
    ) -> BlockResult:
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_version=model_version,
                prompt=prompt,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_version=model_version,
                prompt=prompt,
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
        system_prompt: Optional[str],
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

        prompt = prompt or "Describe what's in this image."
        system_prompt = (
            system_prompt
            or "You are a Qwen3.5-VL model that can answer questions about any image."
        )
        combined_prompt = prompt + "<system_prompt>" + system_prompt

        predictions = []
        for image in images:
            result = client.infer_lmm(
                inference_input=image.base64_image,
                model_id=model_version,
                prompt=combined_prompt,
                model_id_in_path=True,
            )
            response_text = result.get("response", result)
            predictions.append({"parsed_output": response_text, "thinking": ""})

        return predictions

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
        system_prompt: Optional[str],
        enable_thinking: bool = False,
        max_new_tokens: Optional[int] = None,
    ) -> BlockResult:
        # Convert each image to the format required by the model.
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        # Use the provided prompt or default to a generic image description request.
        prompt = prompt or "Describe what's in this image."
        system_prompt = system_prompt or "You are a helpful assistant."
        prompts = [prompt + "<system_prompt>" + system_prompt] * len(inference_images)
        # Register Qwen3.5-VL with the model manager.
        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image, single_prompt in zip(inference_images, prompts):
            # Build an LMMInferenceRequest with both prompt and image.
            request_kwargs = dict(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                source="workflow-execution",
                prompt=single_prompt,
                enable_thinking=enable_thinking,
            )
            if max_new_tokens is not None:
                request_kwargs["max_new_tokens"] = max_new_tokens
            request = LMMInferenceRequest(**request_kwargs)
            # Run inference.
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            response_text = prediction.response
            # When enable_thinking is used and the response contains
            # thinking data (dict with 'thinking' and 'answer' keys),
            # extract them separately.
            if enable_thinking and isinstance(response_text, dict):
                thinking = response_text.get("thinking", "")
                answer = response_text.get("answer", "")
                predictions.append(
                    {
                        "parsed_output": answer,
                        "thinking": thinking,
                    }
                )
            else:
                predictions.append(
                    {
                        "parsed_output": response_text,
                        "thinking": "",
                    }
                )
        return predictions

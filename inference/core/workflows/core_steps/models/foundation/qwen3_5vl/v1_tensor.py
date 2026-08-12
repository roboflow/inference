"""Tensor-native `roboflow_core/qwen3_5vl@v1` block."""

from typing import List, Optional, Type

from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v1 import (
    BlockManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient


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
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
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
        enable_thinking: bool = False,
        max_new_tokens: Optional[int] = None,
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
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
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
        prompt = prompt or "Describe what's in this image."
        system_prompt = system_prompt or "You are a helpful assistant."
        # The model splits the combined prompt back apart on "<system_prompt>".
        combined_prompt = prompt + "<system_prompt>" + system_prompt

        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image in images:
            kwargs = dict(
                prompt=combined_prompt,
                enable_thinking=enable_thinking,
            )
            if max_new_tokens is not None:
                kwargs["max_new_tokens"] = max_new_tokens
            if image.is_tensor_materialised():
                model_image, image_color_format = image.tensor_image, "rgb"
            else:
                model_image, image_color_format = image.numpy_image, "bgr"
            result = self._model_manager.run_tensor_native_inference(
                model_version,
                images=[model_image],
                input_color_format=image_color_format,
                **kwargs,
            )
            # One entry per image: str, or a {"thinking", "answer"} dict when
            # thinking is enabled.
            response_text = result[0]
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

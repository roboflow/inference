"""Tensor-native sibling of `roboflow_core/qwen3_5vl@v2`.

SCRATCH — first pass for review. v2 is a variant of v1 with a different model
selector / variant list and NO thinking mode: it always runs with
`enable_thinking=False` and emits only `parsed_output` (no `thinking` output).

Following florence2/v2_tensor: subclass the tensor-native `Qwen35VLBlockV1`
(from `v1_tensor`) and reuse the verbatim v2 `BlockManifest`. v2's run signature
has no `enable_thinking`, and its output dict has no `thinking` key, so the
`run` / `run_locally` / `run_remotely` shapes are overridden here (they cannot be
delegated to v1's, which always carry a `thinking` key). The LOCAL path still
routes through the inference_models adapter exactly as v1_tensor does, with
`enable_thinking=False` hard-wired.
"""

from typing import List, Optional, Type

from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v1_tensor import (
    Qwen35VLBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v2 import (
    BlockManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest
from inference_sdk import InferenceHTTPClient


class Qwen35VLBlockV2(Qwen35VLBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
        system_prompt: Optional[str],
        max_new_tokens: Optional[int] = None,
    ) -> BlockResult:
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_version=model_version,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
            )
        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_version=model_version,
                prompt=prompt,
                system_prompt=system_prompt,
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
            or "You are a Qwen3.5 model that can answer questions about any image."
        )
        combined_prompt = prompt + "<system_prompt>" + system_prompt

        predictions = []
        for image in images:
            result = client.infer_lmm(
                inference_input=image.base64_image,
                model_id=model_version,
                prompt=combined_prompt,
                model_id_in_path=True,
                enable_thinking=False,
                max_new_tokens=max_new_tokens,
            )
            response_text = result.get("response", result)
            predictions.append({"parsed_output": response_text})

        return predictions

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
        system_prompt: Optional[str],
        max_new_tokens: Optional[int] = None,
    ) -> BlockResult:
        # Local exec goes through the inference_models adapter with the CHW RGB
        # tensor_image. v2 has no thinking mode, so enable_thinking is hard-wired
        # to False — the adapter then returns one str per image.
        prompt = prompt or "Describe what's in this image."
        system_prompt = system_prompt or "You are a helpful assistant."
        combined_prompt = prompt + "<system_prompt>" + system_prompt

        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image in images:
            kwargs = dict(
                prompt=combined_prompt,
                enable_thinking=False,
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
            response_text = result[0]
            predictions.append({"parsed_output": response_text})
        return predictions

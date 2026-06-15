"""Tensor-native sibling of `roboflow_core/qwen3_5vl@v1`.

SCRATCH — first pass for review. Qwen3.5-VL's *outputs* are TEXT/dict
(parsed_output / thinking), NOT prediction kinds, so this block does not PRODUCE
tensor-native predictions. The only thing that changes under
ENABLE_TENSOR_DATA_REPRESENTATION is the LOCAL image path: instead of
`to_inference_format` + `LMMInferenceRequest` + `infer_from_request_sync`, local
inference routes through the inference_models adapter
(`run_tensor_native_inference`) with the block's CHW RGB `tensor_image`.

Manifest, class name and type literal are imported verbatim from v1. Remote
inference is IDENTICAL to v1 (`infer_lmm`; text output) — there is nothing
tensor-native to convert on the output side. `describe_outputs` is inherited
unchanged via the imported manifest.

Adapter contract (inference/models/qwen3_5vl/qwen3_5vl_inference_models.py ->
Qwen35HF.prompt): `run_tensor_native_inference(model_id, images=[...],
input_color_format="rgb", prompt=..., enable_thinking=..., max_new_tokens=...)`
returns one entry per image —
    - enable_thinking=False -> List[str]
    - enable_thinking=True  -> List[Dict[str, str]] with {"thinking", "answer"}
The prompt/system_prompt are combined into a single string with the
`<system_prompt>` separator, exactly as the numpy local path did; the HF model
splits them back apart in `pre_process_generation`.
"""

from typing import List, Optional, Type

from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
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

# Unchanged from v1 — verbatim manifest, class name and type literal.
from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v1 import (
    BlockManifest,
)


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
        # Local exec goes through the inference_models adapter with the CHW RGB
        # tensor_image. prompt/system_prompt are combined exactly as the numpy
        # local path did; the HF model splits them on "<system_prompt>" inside
        # pre_process_generation. The adapter returns one entry per image:
        # str (thinking off) or {"thinking","answer"} dict (thinking on).
        prompt = prompt or "Describe what's in this image."
        system_prompt = system_prompt or "You are a helpful assistant."
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
            result = self._model_manager.run_tensor_native_inference(
                model_version,
                images=[image.tensor_image],
                input_color_format="rgb",
                **kwargs,
            )
            response_text = result[0]
            # When enable_thinking is used the adapter yields a dict with
            # 'thinking' and 'answer' keys; split them out as the numpy block did.
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

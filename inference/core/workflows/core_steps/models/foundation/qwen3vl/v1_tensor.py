"""Tensor-native sibling of `roboflow_core/qwen3vl@v1`.

SCRATCH — first pass for review. Qwen3-VL's *output* is TEXT (parsed_output is a
dict wrapping the model's response string), NOT a prediction kind, so this block
does not PRODUCE tensor-native predictions. The only tensor-native surface is the
local image path: under ENABLE_TENSOR_DATA_REPRESENTATION the block reads
`image.tensor_image` (CHW uint8 RGB on WORKFLOWS_IMAGE_TENSOR_DEVICE) and routes
local inference through the inference_models adapter's run_tensor_native_inference
instead of LMMInferenceRequest.

Manifest, class name, type literal and describe_outputs are IDENTICAL to v1 and
imported verbatim. run_remotely is unchanged from v1 (the model output is text
either way; there is nothing tensor-native to convert on the remote side).
"""

from typing import List, Optional, Type

from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode

# Unchanged from v1 — verbatim manifest, class name, type literal, outputs.
from inference.core.workflows.core_steps.models.foundation.qwen3vl.v1 import (
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


class Qwen3VLBlockV1(WorkflowBlock):
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
            or "You are a Qwen3-VL model that can answer questions about any image."
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
            predictions.append({"parsed_output": response_text})

        return predictions

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
        system_prompt: Optional[str],
    ) -> BlockResult:
        # Use the provided prompt or default to a generic image description request.
        prompt = prompt or "Describe what's in this image."
        system_prompt = (
            system_prompt
            or "You are a Qwen3-VL model that can answer questions about any image."
        )
        # The adapter forwards `prompt` straight to Qwen3VLHF.prompt(), which splits
        # on "<system_prompt>" internally — same combined-prompt convention as the
        # numpy block's LMMInferenceRequest path.
        combined_prompt = prompt + "<system_prompt>" + system_prompt

        # Register Qwen3-VL with the model manager.
        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image in images:
            # Local exec goes through the inference_models adapter (CHW RGB tensor).
            # run_tensor_native_inference -> Qwen3VLHF.prompt returns List[str],
            # one entry per image; we pass one image at a time.
            if image.is_tensor_materialised():
                model_image, image_color_format = image.tensor_image, "rgb"
            else:
                model_image, image_color_format = image.numpy_image, "bgr"
            result = self._model_manager.run_tensor_native_inference(
                model_version,
                images=[model_image],
                input_color_format=image_color_format,
                prompt=combined_prompt,
            )
            response_text = result[0]
            predictions.append(
                {
                    "parsed_output": response_text,
                }
            )
        return predictions

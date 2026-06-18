"""Tensor-native sibling of `roboflow_core/smolvlm2@v1`.

SCRATCH — first pass for review. SmolVLM2's *output* is text wrapped in a dict
(`parsed_output`), NOT a prediction kind, so this block does not PRODUCE
tensor-native predictions and takes no tensor-native prediction INPUT — the only
change is the local inference route.

Manifest, class name, type literal and `describe_outputs` are identical to v1 and
imported verbatim. Local inference now routes through the inference_models adapter
(`run_tensor_native_inference`, CHW uint8 RGB tensor straight off
`image.tensor_image`) instead of `LMMInferenceRequest`; `run_remotely` is byte-for
-byte identical to v1 (the model output is text either way).
"""

from typing import List, Optional, Type

from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode

# Unchanged from v1 — verbatim manifest, class name and type literal.
from inference.core.workflows.core_steps.models.foundation.smolvlm.v1 import (
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
            return self.run_remotely(
                images=images,
                model_version=model_version,
                prompt=prompt,
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
        predictions = []
        for image in images:
            result = client.infer_lmm(
                inference_input=image.base64_image,
                model_id=model_version,
                prompt=prompt,
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
    ) -> BlockResult:
        # Use the provided prompt or default to a generic image description request.
        prompt = prompt or "Describe what's in this image."

        # Register SmolVLM2 with the model manager.
        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image in images:
            # Local exec goes through the inference_models adapter (CHW uint8 RGB
            # tensor straight off image.tensor_image). The adapter's
            # run_tensor_native_inference returns one generated string per image.
            if image.is_tensor_materialised():
                model_image, image_color_format = image.tensor_image, "rgb"
            else:
                model_image, image_color_format = image.numpy_image, "bgr"
            result = self._model_manager.run_tensor_native_inference(
                model_version,
                images=[model_image],
                input_color_format=image_color_format,
                prompt=prompt,
            )
            response_text = result[0]
            predictions.append(
                {
                    "parsed_output": response_text,
                }
            )
        return predictions

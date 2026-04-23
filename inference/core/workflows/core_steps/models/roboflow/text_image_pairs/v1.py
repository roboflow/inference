from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run inference on a fine-tuned text-image-pairs (VLM) model hosted on or uploaded to
Roboflow. Text-image-pairs projects let you fine-tune vision-language models such as
PaliGemma 2, Florence 2, Qwen 2.5 VL, Qwen 3 VL, Qwen 3.5, SmolVLM2, and SmolVLM 256M.

The block returns the raw VLM response as `LANGUAGE_MODEL_OUTPUT_KIND` so it can be
composed with downstream formatter blocks (e.g. `vlm_as_detector`, `vlm_as_classifier`)
depending on how you want to interpret the output.

You can query any model that is private to your account, or any public text-image-pairs
model available on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Multimodal Model",
            "version": "v1",
            "short_description": "Run a fine-tuned multimodal (VLM) model hosted on Roboflow.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "VLM",
                "text-image-pairs",
                "PaliGemma",
                "Florence-2",
                "Qwen",
                "SmolVLM",
            ],
            "is_vlm_block": True,
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-comment-dots",
                "blockPriority": 5.6,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_text_image_pairs_model@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    prompt: Union[Selector(kind=[STRING_KIND]), Optional[str]] = Field(
        default=None,
        description="Optional text prompt forwarded to the VLM.",
        examples=["Describe the image.", "$inputs.prompt"],
    )
    disable_active_learning: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to disable project-level active learning for this block.",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        Selector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for active learning, if enabled.",
        examples=["my_project", "$inputs.al_target_project"],
    )

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["text-image-pairs"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="response", kind=[LANGUAGE_MODEL_OUTPUT_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowTextImagePairsModelBlockV1(WorkflowBlock):

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
        model_id: str,
        prompt: Optional[str],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                prompt=prompt,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                prompt=prompt,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        prompt: Optional[str],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        predictions = []
        for image in inference_images:
            request = LMMInferenceRequest(
                api_key=self._api_key,
                model_id=model_id,
                image=image,
                source="workflow-execution",
                prompt=prompt or "",
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_id, request=request
            )
            predictions.append({"response": prediction.response})
        return predictions

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        prompt: Optional[str],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
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
        # NOTE: `infer_lmm` in inference_sdk does not currently honor
        # `max_batch_size` / `max_concurrent_requests` (it issues one request per
        # call) and does not forward `source` to the `/infer/lmm` payload. We
        # still call `client.configure(...)` so the `source` tag propagates
        # through the shared client state for any header/telemetry usage.
        client_config = InferenceConfiguration(
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)

        predictions = []
        for image in images:
            # NOTE: `infer_lmm` does not currently accept
            # `disable_active_learning` / `active_learning_target_dataset`, so
            # the remote path does not propagate AL controls to `/infer/lmm`.
            # The local execution path honors them via `LMMInferenceRequest`.
            result = client.infer_lmm(
                inference_input=image.base64_image,
                model_id=model_id,
                prompt=prompt or "",
                model_id_in_path=True,
            )
            if isinstance(result, list):
                result = result[0] if result else {}
            predictions.append({"response": result.get("response")})
        return predictions

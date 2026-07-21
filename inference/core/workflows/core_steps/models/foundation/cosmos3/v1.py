from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.env import (
    COSMOS3_ENABLED,
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
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient

DEFAULT_PROMPT = "Describe what's in this image."
DEFAULT_SYSTEM_PROMPT = (
    "You are Cosmos, a helpful assistant that understands physical scenes "
    "and answers questions about images and videos."
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Cosmos 3",
            "version": "v1",
            "short_description": "Reason about physical scenes with NVIDIA Cosmos 3.",
            "long_description": (
                "This workflow block runs the NVIDIA Cosmos 3 reasoner — a world model "
                "tuned for physical scene understanding—on an image with an optional text "
                "prompt, and returns a text answer. The model is well suited to questions "
                "about spatial relations, safety, and what will happen next in a scene."
            ),
            "license": "OpenMDW-1.1",
            "block_type": "model",
            "search_keywords": [
                "Cosmos",
                "cosmos3",
                "NVIDIA",
                "world model",
                "vision language model",
                "VLM",
                "robotics",
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
    type: Literal["roboflow_core/cosmos3_edge@v1"]

    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Optional text prompt to pass to Cosmos 3 Edge. Otherwise a default "
        "scene-description prompt is used, which may affect the desired model behavior.",
        examples=["Is the walkway free of obstacles?", "$inputs.prompt"],
    )
    model_version: Union[
        Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), Literal["nvidia/cosmos-3-edge"]
    ] = Field(
        default="nvidia/cosmos-3-edge",
        description="The Cosmos 3 Edge model to be used for inference.",
        examples=["nvidia/cosmos-3-edge"],
    )
    system_prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Optional system prompt to provide additional context to Cosmos 3 Edge.",
        examples=["You are a safety inspector.", "$inputs.system_prompt"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output",
                kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND],
                description="The model's text answer for each input image.",
            ),
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        restrictions = [
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Requires a GPU; run_locally() loads a model that needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
        ]
        if not COSMOS3_ENABLED:
            restrictions.append(
                RuntimeRestriction(
                    severity=Severity.HARD,
                    note=(
                        "COSMOS3_ENABLED=False: the Cosmos 3 Edge endpoint is not "
                        "registered, so run_remotely() returns 404."
                    ),
                    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
                    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
                )
            )
        return restrictions

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        """Return list of model_id variants that can satisfy this block."""
        return ["cosmos-3-edge"]


class Cosmos3EdgeBlockV1(WorkflowBlock):
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

        combined_prompt = _combine_prompt(prompt=prompt, system_prompt=system_prompt)
        predictions = []
        for image in images:
            result = client.infer_lmm(
                inference_input=image.base64_image,
                model_id=model_version,
                prompt=combined_prompt,
                model_id_in_path=True,
            )
            predictions.append({"output": result.get("response", result)})
        return predictions

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
        system_prompt: Optional[str],
    ) -> BlockResult:
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        combined_prompt = _combine_prompt(prompt=prompt, system_prompt=system_prompt)
        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image in inference_images:
            request = LMMInferenceRequest(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                source="workflow-execution",
                prompt=combined_prompt,
            )
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            predictions.append({"output": prediction.response})
        return predictions


def _combine_prompt(prompt: Optional[str], system_prompt: Optional[str]) -> str:
    prompt = prompt or DEFAULT_PROMPT
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    return prompt + "<system_prompt>" + system_prompt

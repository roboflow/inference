from typing import List, Literal, Optional, Type
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
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
**Dedicated inference server required (GPU recommended) - you may want to use dedicated deployment**

This Workflow block introduces **Qwen 2.5 VL**, a Visual Language Model (VLM) capable of understanding
and answering questions about images. The model can:

* Generate natural language descriptions of images
* Answer questions about image content
* Understand and reason about visual information
"""

class BaseManifest(WorkflowBlockManifest):
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt/question for the Qwen model",
        examples=["What is in this image?", "$inputs.prompt"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="raw_output", 
                kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

class BlockManifest(BaseManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Qwen 2.5 VL Model",
            "version": "v1",
            "short_description": "Run Qwen 2.5 VL on an image",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["Qwen", "Qwen-VL", "VLM"],
            "is_vlm_block": True,
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-brain",
                "blockPriority": 5.5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/qwen_vl@v1"]
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal["qwen25vl-3b"],
    ] = Field(
        default="qwen25vl-3b",
        description="Model to be used",
        examples=["qwen25vl-3b"],
    )

class QwenBlockV1(WorkflowBlock):
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
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_version=model_version,
                prompt=prompt,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for Qwen. Run a local or dedicated inference server to use this block (GPU recommended)."
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
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        prompts = [prompt or "Describe this image"] * len(images)
        
        self._model_manager.add_model(
            model_id=model_version,
            api_key=self._api_key,
        )
        
        predictions = []
        for image, single_prompt in zip(inference_images, prompts):
            request = LMMInferenceRequest(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                source="workflow-execution",
                prompt=single_prompt,
            )
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            predictions.append(
                {
                    "raw_output": prediction.response,
                }
            )
        return predictions
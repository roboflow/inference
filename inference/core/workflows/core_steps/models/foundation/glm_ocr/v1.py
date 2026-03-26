from typing import List, Literal, Optional, Type, Union

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


PROMPT_OPTIONS = {
    "text_recognition": "Text Recognition:",
    "formula_recognition": "Formula Recognition:",
    "table_recognition": "Table Recognition:",
}

DEFAULT_PROMPT = "Text Recognition:"

LONG_DESCRIPTION = """
Recognize text in images using GLM-OCR, a vision language model by Zhipu AI specialized
for optical character recognition.

GLM-OCR supports three built-in recognition modes controlled by the prompt:

- **Text Recognition** (`Text Recognition:`) — General-purpose text recognition for
  serial numbers, labels, scene text, and documents.
- **Formula Recognition** (`Formula Recognition:`) — Recognizes mathematical formulas
  and equations.
- **Table Recognition** (`Table Recognition:`) — Recognizes table structures and content.

You can also provide a custom prompt for specialized recognition tasks.

This block pairs well with detection models and DynamicCropBlock to isolate regions of
interest before running OCR. For example, use an object detection model to find labels
or text regions, crop them, then pass the crops to GLM-OCR.

Note: GLM-OCR requires a GPU for inference.
"""


class BlockManifest(WorkflowBlockManifest):
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Optional[str] = Field(
        default=None,
        description=(
            "Text prompt to guide GLM-OCR recognition. "
            "Use 'Text Recognition:' for general text, "
            "'Formula Recognition:' for math formulas, "
            "'Table Recognition:' for tables, "
            "or provide a custom prompt."
        ),
        examples=["Text Recognition:", "Formula Recognition:", "Table Recognition:"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "name": "GLM-OCR",
            "version": "v1",
            "short_description": "Run GLM-OCR on an image to recognize text.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "GLM-OCR",
                "glm",
                "OCR",
                "text recognition",
                "formula recognition",
                "table recognition",
                "Zhipu",
            ],
            "is_vlm_block": True,
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inDevelopment": False,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/glm_ocr@v1"]

    model_version: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="glm-ocr",
        description="The GLM-OCR model to be used for inference.",
        examples=["glm-ocr"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="parsed_output",
                kind=[STRING_KIND],
                description="The recognized text from the image.",
            ),
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class GLMOCRBlockV1(WorkflowBlock):
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

        prompt = prompt or DEFAULT_PROMPT

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
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        prompt = prompt or DEFAULT_PROMPT

        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image in inference_images:
            request = LMMInferenceRequest(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                source="workflow-execution",
                prompt=prompt,
            )
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            response_text = prediction.response
            predictions.append({"parsed_output": response_text})

        return predictions

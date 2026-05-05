import json
from typing import Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, model_validator

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

STRUCTURED_ANSWERING_PROMPT_TEMPLATE = (
    "You are supposed to produce responses in JSON wrapped in Markdown markers: "
    "```json\nyour-response\n```. Below is a dictionary with keys and values. "
    "Each key must be present in your response. Values represent descriptions "
    "for JSON fields to be generated. Provide only JSON Markdown in response.\n\n"
    "Specification of requirements regarding output fields:\n{output_structure}"
)

TASK_TYPE_TO_PROMPT = {
    "text-recognition": "Text Recognition:",
    "table-recognition": "Table Recognition:",
    "formula-recognition": "Formula Recognition:",
    "structured-answering": None,
    "custom": None,
}

TaskType = Literal[
    "text-recognition",
    "table-recognition",
    "formula-recognition",
    "structured-answering",
    "custom",
]

TASKS_METADATA = {
    "text-recognition": {
        "name": "Text Recognition",
        "description": "General-purpose text recognition for serial numbers, labels, scene text, and documents.",
    },
    "table-recognition": {
        "name": "Table Recognition",
        "description": "Recognizes table structures and content.",
    },
    "formula-recognition": {
        "name": "Formula Recognition",
        "description": "Recognizes mathematical formulas and equations.",
    },
    "structured-answering": {
        "name": "Structured Output",
        "description": "Extract values into a JSON document with a user-defined schema.",
    },
    "custom": {
        "name": "Custom Prompt",
        "description": "Provide your own prompt for specialized recognition tasks.",
    },
}

TASKS_REQUIRING_PROMPT = {"custom"}
TASKS_REQUIRING_OUTPUT_STRUCTURE = {"structured-answering"}

LONG_DESCRIPTION = """
Recognize text in images using GLM-OCR, a vision language model by Zhipu AI specialized
for optical character recognition.

GLM-OCR supports three built-in recognition modes:

- **Text Recognition** — General-purpose text recognition for
  serial numbers, labels, scene text, and documents.
- **Formula Recognition** — Recognizes mathematical formulas
  and equations.
- **Table Recognition** — Recognizes table structures and content.

You can also select **Custom Prompt** to provide your own prompt for specialized
recognition tasks, or **Structured Output** to extract values from the image
into a JSON document with a user-defined schema (pair with the JSON Parser
block to materialize the keys as workflow outputs).

This block pairs well with detection models and DynamicCropBlock to isolate regions of
interest before running OCR. For example, use an object detection model to find labels
or text regions, crop them, then pass the crops to GLM-OCR.

Note: GLM-OCR requires a GPU for inference.
"""


class BlockManifest(WorkflowBlockManifest):
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    task_type: TaskType = Field(
        default="text-recognition",
        description="Recognition task to perform. Determines the prompt sent to GLM-OCR.",
        json_schema_extra={
            "values_metadata": TASKS_METADATA,
            "recommended_parsers": {
                "structured-answering": "roboflow_core/json_parser@v1",
            },
            "always_visible": True,
        },
    )

    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Custom text prompt for GLM-OCR. Only used when task_type is 'custom'.",
        examples=["Describe the text in the image."],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": TASKS_REQUIRING_PROMPT, "required": True},
            },
        },
    )
    output_structure: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary describing the structure of the expected JSON response. "
        "Keys are the JSON field names; values describe what the model should put in each field.",
        examples=[{"my_key": "description"}, "$inputs.output_structure"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": TASKS_REQUIRING_OUTPUT_STRUCTURE,
                    "required": True,
                },
            },
        },
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate. If not set, the model default will be used.",
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

    @model_validator(mode="after")
    def validate_prompt(self) -> "BlockManifest":
        if self.task_type == "custom" and not self.prompt:
            raise ValueError("`prompt` is required when task_type is 'custom'.")
        if (
            self.task_type in TASKS_REQUIRING_OUTPUT_STRUCTURE
            and not self.output_structure
        ):
            raise ValueError(
                f"`output_structure` is required when task_type is '{self.task_type}'."
            )
        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="parsed_output",
                kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND],
                description="The recognized text from the image. For "
                "`structured-answering` this is a JSON-in-Markdown document "
                "ready to be fed into the JSON Parser block.",
            ),
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def _resolve_prompt(
    task_type: str,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
) -> str:
    if task_type == "custom":
        return prompt
    if task_type == "structured-answering":
        return STRUCTURED_ANSWERING_PROMPT_TEMPLATE.format(
            output_structure=json.dumps(output_structure, indent=4),
        )
    return TASK_TYPE_TO_PROMPT[task_type]


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
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> BlockResult:
        resolved_prompt = _resolve_prompt(task_type, prompt, output_structure)
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_version=model_version,
                prompt=resolved_prompt,
                max_new_tokens=max_new_tokens,
            )
        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_version=model_version,
                prompt=resolved_prompt,
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
        prompt: str,
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

        predictions = []
        for image in images:
            result = client.infer_lmm(
                inference_input=image.base64_image,
                model_id=model_version,
                prompt=prompt,
                model_id_in_path=True,
                max_new_tokens=max_new_tokens,
            )
            response_text = result.get("response", result)
            predictions.append({"parsed_output": response_text})

        return predictions

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> BlockResult:
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]

        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image in inference_images:
            request_kwargs = dict(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                source="workflow-execution",
                prompt=prompt,
            )
            if max_new_tokens is not None:
                request_kwargs["max_new_tokens"] = max_new_tokens
            request = LMMInferenceRequest(**request_kwargs)
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            response_text = prediction.response
            predictions.append({"parsed_output": response_text})

        return predictions

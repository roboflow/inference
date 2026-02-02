import base64
import json
import re
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

import requests
from pydantic import ConfigDict, Field, field_validator, model_validator
from requests import Response

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.workflows.core_steps.common.utils import run_in_parallel
from inference.core.workflows.core_steps.common.vlms import VLM_TASKS_METADATA
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    BOOLEAN_KIND,
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

GOOGLE_API_KEY_PATTERN = re.compile(r"key=(.[^&]*)")
GOOGLE_API_KEY_VALUE_GROUP = 1
MIN_KEY_LENGTH_TO_REVEAL_PREFIX = 8

MODEL_ALIASES = {
    "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-03-25": "gemini-2.5-pro",
    "gemini-2.0-flash-exp": "gemini-2.0-flash",
}

GEMINI_MODELS = [
    {
        "id": "gemini-3-pro-preview",
        "name": "Gemini 3 Pro",
        "supports_thinking_level": True,
        "supports_native_code_execution": True,
    },
    {
        "id": "gemini-3-flash-preview",
        "name": "Gemini 3 Flash",
        "supports_thinking_level": True,
        "supports_native_code_execution": True,
    },
    {
        "id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "supports_thinking_level": False,
        "supports_native_code_execution": False,
    },
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "supports_thinking_level": False,
        "supports_native_code_execution": False,
    },
    {
        "id": "gemini-2.5-flash-lite",
        "name": "Gemini 2.5 Flash-Lite",
        "supports_thinking_level": False,
        "supports_native_code_execution": False,
    },
    {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "supports_thinking_level": False,
        "supports_native_code_execution": False,
    },
    {
        "id": "gemini-2.0-flash-lite",
        "name": "Gemini 2.0 Flash-Lite",
        "supports_thinking_level": False,
        "supports_native_code_execution": False,
    },
]

MODEL_VERSION_IDS = [model["id"] for model in GEMINI_MODELS]

MODEL_VERSION_METADATA = {
    model["id"]: {"name": model["name"]} for model in GEMINI_MODELS
}

MODELS_SUPPORTING_THINKING_LEVEL = [
    model["id"] for model in GEMINI_MODELS if model["supports_thinking_level"]
]

MODELS_SUPPORTING_NATIVE_CODE_EXECUTION = [
    model["id"] for model in GEMINI_MODELS if model["supports_native_code_execution"]
]

MODELS_NOT_SUPPORTING_THINKING_LEVEL = [
    model["id"] for model in GEMINI_MODELS if not model["supports_thinking_level"]
]

SUPPORTED_TASK_TYPES_LIST = [
    "unconstrained",
    "ocr",
    "structured-answering",
    "classification",
    "multi-label-classification",
    "visual-question-answering",
    "caption",
    "detailed-caption",
    "object-detection",
]
SUPPORTED_TASK_TYPES = set(SUPPORTED_TASK_TYPES_LIST)

RELEVANT_TASKS_METADATA = {
    k: v for k, v in VLM_TASKS_METADATA.items() if k in SUPPORTED_TASK_TYPES
}
RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)

LONG_DESCRIPTION = f"""
Ask a question to Google's Gemini model with vision capabilities.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

### API Key Options

This block supports two API key modes:

1. **Roboflow Managed API Key (Default)** - Use `rf_key:account` to proxy requests through Roboflow's API:
   * **Simplified setup** - no Google AI API key required
   * **Secure** - your workflow API key is used for authentication
   * **Usage-based billing** - charged per token based on the model used

2. **Custom Google AI API Key** - Provide your own Google AI API key:
   * Full control over API usage
   * You pay Google directly

**WARNING!**

This block makes use of `/v1beta` API of Google Gemini model - the implementation may change
in the future, without guarantee of backward compatibility.
"""

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

TASKS_REQUIRING_PROMPT = {
    "unconstrained",
    "visual-question-answering",
}

TASKS_REQUIRING_CLASSES = {
    "classification",
    "multi-label-classification",
    "object-detection",
}

TASKS_REQUIRING_OUTPUT_STRUCTURE = {
    "structured-answering",
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Google Gemini",
            "version": "v3",
            "short_description": "Run Google's Gemini model with vision capabilities.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Gemini", "Google"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fa-brands fa-google",
                "blockPriority": 5,
                "popular": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/google_gemini@v3"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    task_type: TaskType = Field(
        default="unconstrained",
        description="Task type to be performed by model. Value determines required parameters and output response.",
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
            "recommended_parsers": {
                "structured-answering": "roboflow_core/json_parser@v1",
                "classification": "roboflow_core/vlm_as_classifier@v2",
                "multi-label-classification": "roboflow_core/vlm_as_classifier@v2",
                "object-detection": "roboflow_core/vlm_as_detector@v2",
            },
            "always_visible": True,
        },
    )
    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to the Gemini model",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": TASKS_REQUIRING_PROMPT, "required": True},
            },
            "multiline": True,
        },
    )
    output_structure: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary with structure of expected JSON response",
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
    classes: Optional[Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]]] = Field(
        default=None,
        description="List of classes to be used",
        examples=[["class-a", "class-b"], "$inputs.classes"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": TASKS_REQUIRING_CLASSES,
                    "required": True,
                },
            },
        },
    )
    api_key: Union[
        Selector(kind=[STRING_KIND, SECRET_KIND, ROBOFLOW_MANAGED_KEY]), str
    ] = Field(
        default="rf_key:account",
        description="Your Google AI API key or 'rf_key:account' to use Roboflow's managed API key",
        examples=["rf_key:account", "xxx-xxx", "$inputs.google_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal[tuple(MODEL_VERSION_IDS)],
    ] = Field(
        default="gemini-3-pro-preview",
        description="Model to be used",
        examples=["gemini-3-pro-preview", "$inputs.gemini_model"],
        json_schema_extra={
            "values_metadata": MODEL_VERSION_METADATA,
        },
    )
    thinking_level: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            Literal["low", "high"],
        ]
    ] = Field(
        default=None,
        description="Controls the depth of internal reasoning for Gemini 3+ models. "
        "'low' minimizes latency and cost (best for simple tasks), 'high' maximizes reasoning depth (default). "
        "Only supported by Gemini 3 and newer models.",
        json_schema_extra={
            "relevant_for": {
                "model_version": {
                    "values": MODELS_SUPPORTING_THINKING_LEVEL,
                    "required": False,
                },
            },
        },
    )
    temperature: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Temperature to sample from the model - value in range 0.0-2.0, the higher - the more "
        'random / "creative" the generations are.',
        ge=0.0,
        le=2.0,
        json_schema_extra={
            "relevant_for": {
                "model_version": {
                    "values": MODELS_NOT_SUPPORTING_THINKING_LEVEL,
                    "required": False,
                },
            },
        },
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens the model can generate in it's response. "
        "If not specified, the model will use its default limit.",
    )
    google_code_execution: Optional[Union[bool, Selector(kind=[BOOLEAN_KIND])]] = Field(
        default=False,
        description="Enable native code execution for the Gemini model.",
        json_schema_extra={
            "relevant_for": {
                "model_version": {
                    "values": MODELS_SUPPORTING_NATIVE_CODE_EXECUTION,
                    "required": False,
                },
            },
        },
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description="Number of concurrent requests that can be executed by block when batch of input images provided. "
        "If not given - block defaults to value configured globally in Workflows Execution Engine. "
        "Please restrict if you hit Google Gemini API limits.",
    )

    @field_validator("model_version", mode="before")
    @classmethod
    def validate_model_version(cls, value):
        if isinstance(value, str) and value in MODEL_ALIASES:
            return MODEL_ALIASES[value]
        return value

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.task_type in TASKS_REQUIRING_PROMPT and self.prompt is None:
            raise ValueError(
                f"`prompt` parameter required to be set for task `{self.task_type}`"
            )
        if self.task_type in TASKS_REQUIRING_CLASSES and self.classes is None:
            raise ValueError(
                f"`classes` parameter required to be set for task `{self.task_type}`"
            )
        if (
            self.task_type in TASKS_REQUIRING_OUTPUT_STRUCTURE
            and self.output_structure is None
        ):
            raise ValueError(
                f"`output_structure` parameter required to be set for task `{self.task_type}`"
            )
        return self

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
            OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class GoogleGeminiBlockV3(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
    ):
        self._model_manager = model_manager
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        task_type: TaskType,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        model_version: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        thinking_level: Optional[str],
        google_code_execution: Optional[bool],
        max_concurrent_requests: Optional[int],
        api_key: str = "rf_key:account",
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_outputs = run_gemini_prompting(
            roboflow_api_key=self._api_key,
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            google_api_key=api_key,
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            google_code_execution=google_code_execution,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]


def run_gemini_prompting(
    roboflow_api_key: Optional[str],
    images: List[Dict[str, Any]],
    task_type: TaskType,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    google_api_key: str,
    model_version: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    thinking_level: Optional[str],
    google_code_execution: Optional[bool],
    max_concurrent_requests: Optional[int],
) -> List[str]:
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    gemini_prompts = []
    for image in images:
        loaded_image, _ = load_image(image)
        base64_image = base64.b64encode(
            encode_image_to_jpeg_bytes(loaded_image)
        ).decode("ascii")
        generated_prompt = PROMPT_BUILDERS[task_type](
            base64_image=base64_image,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            model_version=model_version,
            temperature=temperature,
            thinking_level=thinking_level,
            max_tokens=max_tokens,
        )

        if (
            google_code_execution
            and model_version in MODELS_SUPPORTING_NATIVE_CODE_EXECUTION
        ):
            generated_prompt["tools"] = [{"code_execution": {}}]

        gemini_prompts.append(generated_prompt)
    return execute_gemini_requests(
        roboflow_api_key=roboflow_api_key,
        google_api_key=google_api_key,
        gemini_prompts=gemini_prompts,
        model_version=model_version,
        max_concurrent_requests=max_concurrent_requests,
    )


def execute_gemini_requests(
    roboflow_api_key: Optional[str],
    google_api_key: str,
    gemini_prompts: List[dict],
    model_version: str,
    max_concurrent_requests: Optional[int],
) -> List[str]:
    tasks = [
        partial(
            execute_gemini_request,
            roboflow_api_key=roboflow_api_key,
            google_api_key=google_api_key,
            prompt=prompt,
            model_version=model_version,
        )
        for prompt in gemini_prompts
    ]
    max_workers = (
        max_concurrent_requests
        or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    )
    return run_in_parallel(
        tasks=tasks,
        max_workers=max_workers,
    )


def execute_gemini_request(
    roboflow_api_key: Optional[str],
    google_api_key: str,
    prompt: dict,
    model_version: str,
) -> str:
    """Route to proxied or direct execution based on API key format."""
    if google_api_key.startswith(("rf_key:account", "rf_key:user:")):
        return _execute_proxied_gemini_request(
            roboflow_api_key=roboflow_api_key,
            google_api_key=google_api_key,
            prompt=prompt,
            model_version=model_version,
        )
    else:
        return _execute_direct_gemini_request(
            google_api_key=google_api_key,
            prompt=prompt,
            model_version=model_version,
        )


def _execute_proxied_gemini_request(
    roboflow_api_key: str,
    google_api_key: str,
    prompt: dict,
    model_version: str,
) -> str:
    """Execute Gemini request via Roboflow proxy."""
    payload = {
        "model": model_version,
        "google_api_key": google_api_key,
        **prompt,  # Contains contents, generationConfig, systemInstruction
    }

    endpoint = "apiproxy/gemini"

    try:
        response_data = post_to_roboflow_api(
            endpoint=endpoint,
            api_key=roboflow_api_key,
            payload=payload,
        )
        return _extract_gemini_response_text(response_data)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to Roboflow proxy: {e}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Invalid response structure from Roboflow proxy: {e}"
        ) from e


def _execute_direct_gemini_request(
    google_api_key: str,
    prompt: dict,
    model_version: str,
) -> str:
    """Execute Gemini request directly to Google API."""
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_version}:generateContent",
        headers={
            "Content-Type": "application/json",
        },
        params={
            "key": google_api_key,
        },
        json=prompt,
    )
    response_data = response.json()
    google_api_key_safe_raise_for_status(response=response)
    return _extract_gemini_response_text(response_data)


def _extract_gemini_response_text(response_data: dict) -> str:
    """Extract text content from Gemini API response."""
    if "candidates" not in response_data or not response_data["candidates"]:
        raise ValueError("Gemini API returned no response candidates.")

    candidate = response_data["candidates"][0]
    finish_reason = candidate.get("finishReason", "FINISH_REASON_UNSPECIFIED")

    if finish_reason == "MAX_TOKENS":
        raise ValueError(
            "Gemini API stopped generation because the max_tokens limit was reached. "
            "Please increase the max_tokens parameter to allow for a complete response."
        )

    # Check for values different than natural stop or unspecified
    if finish_reason not in ["STOP", "FINISH_REASON_UNSPECIFIED"]:
        raise ValueError(
            f"Gemini API stopped generation with finish reason: {finish_reason}."
        )

    try:
        parts = candidate["content"]["parts"]
        text_parts = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "executableCode" in part:
                code = part["executableCode"].get("code", "")
                language = part["executableCode"].get("language", "python").lower()
                text_parts.append(f"\n```{language}\n{code}\n```\n")
            elif "codeExecutionResult" in part:
                output = part["codeExecutionResult"].get("output", "")
                outcome = part["codeExecutionResult"].get("outcome", "")
                text_parts.append(
                    f"\n**Code Execution Result ({outcome}):**\n```\n{output}\n```\n"
                )

        if not text_parts:
            # Fallback if no parts are recognized
            raise ValueError("No recognizable content found in Gemini API response.")

        return "".join(text_parts)
    except (KeyError, IndexError, TypeError):
        raise ValueError("Unable to parse Gemini API response.")


def prepare_unconstrained_prompt(
    base64_image: str,
    prompt: str,
    model_version: str,
    temperature: Optional[float],
    thinking_level: Optional[str],
    max_tokens: Optional[int],
    **kwargs,
) -> dict:
    return {
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {
                    "text": prompt,
                },
            ],
            "role": "user",
        },
        "generationConfig": prepare_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            model_version=model_version,
        ),
    }


def prepare_classification_prompt(
    base64_image: str,
    classes: List[str],
    model_version: str,
    temperature: Optional[float],
    thinking_level: Optional[str],
    max_tokens: Optional[int],
    **kwargs,
) -> dict:
    serialised_classes = ", ".join(classes)
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You act as single-class classification model. You must provide reasonable predictions. "
                    "You are only allowed to produce JSON document. "
                    'Expected structure of json: {"class_name": "class-name", "confidence": 0.4}. '
                    "`class-name` must be one of the class names defined by user. You are only allowed to return "
                    "single JSON document, even if there are potentially multiple classes. You are not allowed to "
                    "return list.",
                }
            ],
        },
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
            "role": "user",
        },
        "generationConfig": prepare_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            model_version=model_version,
            response_mime_type="application/json",
        ),
    }


def prepare_multi_label_classification_prompt(
    base64_image: str,
    classes: List[str],
    model_version: str,
    temperature: Optional[float],
    thinking_level: Optional[str],
    max_tokens: Optional[int],
    **kwargs,
) -> dict:
    serialised_classes = ", ".join(classes)
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You act as multi-label classification model. You must provide reasonable predictions. "
                    "You are only allowed to produce JSON document. "
                    'Expected structure of json: {"predicted_classes": [{"class": "class-name-1", "confidence": 0.9}, '
                    '{"class": "class-name-2", "confidence": 0.7}]}. '
                    "`class-name-X` must be one of the class names defined by user and `confidence` is a float value "
                    "in range 0.0-1.0 that represents how sure you are that the class is present in the image. "
                    "Only return class names that are visible.",
                }
            ],
        },
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
            "role": "user",
        },
        "generationConfig": prepare_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            model_version=model_version,
            response_mime_type="application/json",
        ),
    }


def prepare_vqa_prompt(
    base64_image: str,
    prompt: str,
    model_version: str,
    temperature: Optional[float],
    thinking_level: Optional[str],
    max_tokens: Optional[int],
    **kwargs,
) -> dict:
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You act as Visual Question Answering model. Your task is to provide answer to question"
                    "submitted by user. If this is open-question - answer with few sentences, for ABCD question, "
                    "return only the indicator of the answer.",
                }
            ],
        },
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {
                    "text": f"Question: {prompt}",
                },
            ],
            "role": "user",
        },
        "generationConfig": prepare_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            model_version=model_version,
        ),
    }


def prepare_ocr_prompt(
    base64_image: str,
    model_version: str,
    temperature: Optional[float],
    thinking_level: Optional[str],
    max_tokens: Optional[int],
    **kwargs,
) -> dict:
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You act as OCR model. Your task is to read text from the image and return it in "
                    "paragraphs representing the structure of texts in the image. You should only return "
                    "recognised text, nothing else.",
                }
            ],
        },
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {
                    "text": f"Read the text",
                },
            ],
            "role": "user",
        },
        "generationConfig": prepare_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            model_version=model_version,
        ),
    }


def prepare_caption_prompt(
    base64_image: str,
    short_description: bool,
    model_version: str,
    temperature: Optional[float],
    thinking_level: Optional[str],
    max_tokens: Optional[int],
    **kwargs,
) -> dict:
    caption_detail_level = "Caption should be short."
    if not short_description:
        caption_detail_level = "Caption should be extensive."
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": f"You act as image caption model. Your task is to provide description of the image. "
                    f"{caption_detail_level}",
                }
            ],
        },
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {
                    "text": f"Caption the image",
                },
            ],
            "role": "user",
        },
        "generationConfig": prepare_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            model_version=model_version,
        ),
    }


def prepare_structured_answering_prompt(
    base64_image: str,
    output_structure: Dict[str, str],
    model_version: str,
    temperature: Optional[float],
    thinking_level: Optional[str],
    max_tokens: Optional[int],
    **kwargs,
) -> dict:
    output_structure_serialised = json.dumps(output_structure, indent=4)
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You are supposed to produce responses in JSON. User is to provide you dictionary with "
                    "keys and values. Each key must be present in your response. Values in user dictionary "
                    "represent descriptions for JSON fields to be generated. Provide only JSON in response.",
                }
            ],
        },
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {
                    "text": f"Specification of requirements regarding output fields: \n"
                    f"{output_structure_serialised}",
                },
            ],
            "role": "user",
        },
        "generationConfig": prepare_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            model_version=model_version,
            response_mime_type="application/json",
        ),
    }


def prepare_object_detection_prompt(
    base64_image: str,
    classes: List[str],
    model_version: str,
    temperature: Optional[float],
    thinking_level: Optional[str],
    max_tokens: Optional[int],
    **kwargs,
) -> dict:
    serialised_classes = ", ".join(classes)
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You act as object-detection model. You must provide reasonable predictions. "
                    "You are only allowed to produce JSON document. "
                    'Expected structure of json: {"detections": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4, "class_name": "my-class-X", "confidence": 0.7}]}. '
                    "`my-class-X` must be one of the class names defined by user. All coordinates must be in range 0.0-1.0, representing percentage of image dimensions. "
                    "`confidence` is a value in range 0.0-1.0 representing your confidence in prediction. You should detect all instances of classes provided by user.",
                }
            ],
        },
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
            "role": "user",
        },
        "generationConfig": prepare_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            model_version=model_version,
            response_mime_type="application/json",
        ),
    }


def prepare_generation_config(
    max_tokens: Optional[int],
    temperature: Optional[float],
    thinking_level: Optional[str],
    model_version: str,
    response_mime_type: str = "text/plain",
) -> dict:
    result = {
        "response_mime_type": response_mime_type,
        "candidate_count": 1,
    }

    if max_tokens is not None:
        result["max_output_tokens"] = max_tokens

    supports_thinking_level = model_version in MODELS_SUPPORTING_THINKING_LEVEL

    if thinking_level is not None and supports_thinking_level:
        result["thinking_config"] = {"thinking_level": thinking_level}

    if temperature is not None and not supports_thinking_level:
        result["temperature"] = temperature

    return result


def google_api_key_safe_raise_for_status(response: Response) -> None:
    request_is_successful = response.status_code < 400
    if request_is_successful:
        return None
    response.url = GOOGLE_API_KEY_PATTERN.sub(deduct_api_key, response.url)
    response.raise_for_status()


def deduct_api_key(match: re.Match) -> str:
    key_value = match.group(GOOGLE_API_KEY_VALUE_GROUP)
    if len(key_value) < MIN_KEY_LENGTH_TO_REVEAL_PREFIX:
        return f"key=***"
    key_prefix = key_value[:2]
    key_postfix = key_value[-2:]
    return f"key={key_prefix}***{key_postfix}"


PROMPT_BUILDERS = {
    "unconstrained": prepare_unconstrained_prompt,
    "ocr": prepare_ocr_prompt,
    "visual-question-answering": prepare_vqa_prompt,
    "caption": partial(prepare_caption_prompt, short_description=True),
    "detailed-caption": partial(prepare_caption_prompt, short_description=False),
    "classification": prepare_classification_prompt,
    "multi-label-classification": prepare_multi_label_classification_prompt,
    "structured-answering": prepare_structured_answering_prompt,
    "object-detection": prepare_object_detection_prompt,
}

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
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
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

GEMINI_MODEL_ALIASES = {
    "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-03-25": "gemini-2.5-pro",
    "gemini-2.0-flash-exp": "gemini-2.0-flash",
}

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
Run Google's Gemini model with vision capabilities to perform various computer vision tasks.

## What is a Vision Language Model (VLM)?

A Vision Language Model (VLM) is an AI model that can understand both **images and text** simultaneously. Unlike traditional computer vision models that are trained for a single task (like object detection or classification), VLMs like Gemini:
- **Understand natural language prompts** - you can ask questions or give instructions in plain English
- **Process visual content** - analyze images to understand what's in them
- **Generate flexible outputs** - provide text responses, structured data, or formatted results based on your needs
- **Support multiple tasks** - the same model can perform classification, detection, OCR, question answering, and more just by changing the prompt

This makes VLMs incredibly versatile and useful when you need flexible, natural language-driven computer vision without training separate models for each task.

## How This Block Works

This block takes one or more images as input and processes them through Google's Gemini model. Based on the **task type** you select, the block:
1. **Prepares the appropriate prompt** for Gemini based on the task type (e.g., OCR, classification, object detection)
2. **Encodes images** to base64 format for API transmission
3. **Sends the request to Google's Gemini API** with the image and task-specific instructions
4. **Returns the response** as text output, which can be structured JSON or natural language depending on the task

The block supports multiple predefined task types, each optimized for specific use cases, or you can use "unconstrained" mode for completely custom prompts.

## Supported Task Types

The block supports the following task types:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

## Inputs and Outputs

**Input:**
- **images**: One or more images to analyze (can be from workflow inputs or previous steps)
- **task_type**: The type of task to perform (determines how the prompt is structured and what output format to expect)
- **prompt**: Text prompt/question (required for "unconstrained" and "visual-question-answering" tasks)
- **classes**: List of classes for classification or detection tasks (required for "classification", "multi-label-classification", "object-detection" tasks)
- **output_structure**: Dictionary defining the expected JSON structure (required for "structured-answering" task)
- **api_key**: Your Google AI API key (required)
- **model_version**: Gemini model version to use (default: "gemini-2.0-flash")
- **max_tokens**: Maximum number of tokens in the response (default: 450)
- **temperature**: Sampling temperature (0.0-2.0, optional) - controls randomness/creativity
- **max_concurrent_requests**: Number of concurrent API requests when processing batches (optional, uses global default if not specified)

**Output:**
- **output**: Text response from Gemini (string) - format depends on task type (may be JSON for structured tasks)
- **classes**: The list of classes that were provided (for classification/detection tasks)

## Key Configuration Options

- **task_type**: Select the task type that best matches your use case - this determines what prompt is sent to Gemini and what output format to expect
- **model_version**: Choose the Gemini model - newer models (gemini-2.5-pro, gemini-2.5-flash) are more capable; older models (gemini-1.5-flash) are faster but less capable. Flash models are optimized for speed, while Pro models prioritize accuracy
- **max_tokens**: Control the maximum response length - increase for longer responses (e.g., detailed captions), decrease for shorter responses (e.g., classification)
- **temperature**: Control output randomness - lower values (0.0-0.5) produce more deterministic, focused responses; higher values (1.0-2.0) produce more creative, varied responses
- **max_concurrent_requests**: Limit concurrent API calls to stay within Google Gemini API rate limits

## Common Use Cases

- **Content Analysis**: Analyze images for safety, quality, or compliance - ask questions like "Does this image contain inappropriate content?"
- **Document Processing**: Extract text from documents, forms, or receipts using OCR, then structure the data using structured-answering
- **Product Cataloging**: Classify product images into categories or extract product attributes like color, style, material
- **Accessibility**: Generate detailed image descriptions for visually impaired users using captioning tasks
- **Data Extraction**: Extract structured information from images (e.g., extract fields from forms, receipts, or documents)
- **Visual Q&A**: Build chatbots that can answer questions about images (e.g., "What brand is this product?", "Is this person wearing a mask?")

## Requirements

You need to provide your Google AI API key to use this block. The API key is used to authenticate requests to Google's Gemini API. You can get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey) or [Google Cloud Console](https://console.cloud.google.com/). Note that API usage is subject to Google's pricing and rate limits.

**⚠️ Beta API Warning**

This block uses Google's `/v1beta` API endpoint for Gemini. The implementation may change in the future without guarantee of backward compatibility as Google continues to develop their API.

## Connecting to Other Blocks

The text outputs from this block can be connected to:
- **Parser blocks** (e.g., JSON Parser v1, VLM as Classifier v1, VLM as Detector v1) to convert text responses into structured data formats
- **Conditional logic blocks** to route workflow execution based on Gemini's responses
- **Filter blocks** to filter images or detections based on Gemini's analysis
- **Visualization blocks** to display text overlays or annotations on images
- **Data storage blocks** to log responses for analytics or audit trails
- **Notification blocks** to send alerts based on Gemini's findings
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
            "version": "v1",
            "short_description": "Run Google's Gemini model with vision capabilities.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Gemini", "Google"],
            "beta": True,
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
    type: Literal["roboflow_core/google_gemini@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    task_type: TaskType = Field(
        default="unconstrained",
        description="Task type to be performed by model. Value determines required parameters and output response.",
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
            "recommended_parsers": {
                "structured-answering": "roboflow_core/json_parser@v1",
                "classification": "roboflow_core/vlm_as_classifier@v1",
                "multi-label-classification": "roboflow_core/vlm_as_classifier@v1",
                "object-detection": "roboflow_core/vlm_as_detector@v1",
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
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your Google AI API key",
        examples=["xxx-xxx", "$inputs.google_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    ] = Field(
        default="gemini-2.0-flash",
        description="Model to be used",
        examples=["gemini-2.5-pro", "$inputs.gemini_model"],
    )
    max_tokens: int = Field(
        default=450,
        description="Maximum number of tokens the model can generate in it's response.",
    )
    temperature: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Temperature to sample from the model - value in range 0.0-2.0, the higher - the more "
        'random / "creative" the generations are.',
        ge=0.0,
        le=2.0,
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
        if isinstance(value, str) and value in GEMINI_MODEL_ALIASES:
            return GEMINI_MODEL_ALIASES[value]
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


class GoogleGeminiBlockV1(WorkflowBlock):

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
        api_key: str,
        model_version: str,
        max_tokens: int,
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_outputs = run_gemini_prompting(
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            google_api_key=api_key,
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]


def run_gemini_prompting(
    images: List[Dict[str, Any]],
    task_type: TaskType,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    google_api_key: Optional[str],
    model_version: str,
    max_tokens: int,
    temperature: Optional[float],
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
            temperature=temperature,
            max_tokens=max_tokens,
        )
        gemini_prompts.append(generated_prompt)
    return execute_gemini_requests(
        google_api_key=google_api_key,
        gemini_prompts=gemini_prompts,
        model_version=model_version,
        max_concurrent_requests=max_concurrent_requests,
    )


def execute_gemini_requests(
    google_api_key: str,
    gemini_prompts: List[dict],
    model_version: str,
    max_concurrent_requests: Optional[int],
) -> List[str]:
    tasks = [
        partial(
            execute_gemini_request,
            prompt=prompt,
            model_version=model_version,
            google_api_key=google_api_key,
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
    prompt: dict,
    model_version: str,
    google_api_key: str,
) -> str:
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
    return response_data["candidates"][0]["content"]["parts"][0]["text"]


def prepare_unconstrained_prompt(
    base64_image: str,
    prompt: str,
    temperature: Optional[float],
    max_tokens: int,
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
        ),
    }


def prepare_classification_prompt(
    base64_image: str,
    classes: List[str],
    temperature: Optional[float],
    max_tokens: int,
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
            response_mime_type="application/json",
        ),
    }


def prepare_multi_label_classification_prompt(
    base64_image: str,
    classes: List[str],
    temperature: Optional[float],
    max_tokens: int,
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
            response_mime_type="application/json",
        ),
    }


def prepare_vqa_prompt(
    base64_image: str,
    prompt: str,
    temperature: Optional[float],
    max_tokens: int,
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
        ),
    }


def prepare_ocr_prompt(
    base64_image: str,
    temperature: Optional[float],
    max_tokens: int,
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
        ),
    }


def prepare_caption_prompt(
    base64_image: str,
    short_description: bool,
    temperature: Optional[float],
    max_tokens: int,
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
        ),
    }


def prepare_structured_answering_prompt(
    base64_image: str,
    output_structure: Dict[str, str],
    temperature: Optional[float],
    max_tokens: int,
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
            response_mime_type="application/json",
        ),
    }


def prepare_object_detection_prompt(
    base64_image: str,
    classes: List[str],
    temperature: Optional[float],
    max_tokens: int,
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
            response_mime_type="application/json",
        ),
    }


def prepare_generation_config(
    max_tokens: int,
    temperature: Optional[float],
    response_mime_type: str = "text/plain",
) -> dict:
    result = {
        "max_output_tokens": max_tokens,
        "response_mime_type": response_mime_type,
        "candidate_count": 1,
    }
    if temperature is not None:
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

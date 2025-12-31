import base64
import json
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

import requests
from openai import OpenAI
from openai._types import NOT_GIVEN
from pydantic import ConfigDict, Field, model_validator

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

SUPPORTED_TASK_TYPES_LIST = [
    "unconstrained",
    "ocr",
    "structured-answering",
    "classification",
    "multi-label-classification",
    "visual-question-answering",
    "caption",
    "detailed-caption",
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
Run OpenAI's GPT models with vision capabilities to perform various computer vision tasks.

## What is a Vision Language Model (VLM)?

A Vision Language Model (VLM) is an AI model that can understand both **images and text** simultaneously. Unlike traditional computer vision models that are trained for a single task (like object detection or classification), VLMs like GPT-4 Vision:
- **Understand natural language prompts** - you can ask questions or give instructions in plain English
- **Process visual content** - analyze images to understand what's in them
- **Generate flexible outputs** - provide text responses or structured data based on your needs
- **Support multiple tasks** - the same model can perform classification, detection, OCR, question answering, and more just by changing the prompt

This makes VLMs incredibly versatile and useful when you need flexible, natural language-driven computer vision without training separate models for each task.

## How This Block Works

This block takes one or more images as input and processes them through OpenAI's GPT models (including GPT-5 and GPT-4o). Based on the **task type** you select, the block:
1. **Prepares the appropriate prompt** for the GPT model based on the task type (e.g., OCR, classification, caption)
2. **Encodes images** to base64 format for API transmission
3. **Sends the request to OpenAI's API** (directly or via Roboflow proxy) with the image and task-specific instructions
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
- **classes**: List of classes for classification tasks (required for "classification" and "multi-label-classification" tasks)
- **output_structure**: Dictionary defining the expected JSON structure (required for "structured-answering" task)
- **api_key**: Your OpenAI API key, or use "rf_key:account" (default) to proxy requests through Roboflow's API
- **model_version**: GPT model version to use (default: "gpt-5") - options include gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini, o3, o4-mini
- **image_detail**: Image processing quality - "auto" (default, model decides), "high" (high fidelity, processes fine details), or "low" (faster, lower cost, lower detail)
- **max_tokens**: Maximum number of tokens in the response (default: 450)
- **temperature**: Sampling temperature (0.0-2.0, optional) - controls randomness/creativity
- **max_concurrent_requests**: Number of concurrent API requests when processing batches (optional, uses global default if not specified)

**Output:**
- **output**: Text response from GPT (string) - format depends on task type (may be JSON for structured tasks)
- **classes**: The list of classes that were provided (for classification tasks)

## Key Configuration Options

- **task_type**: Select the task type that best matches your use case - this determines what prompt is sent to GPT and what output format to expect
- **model_version**: Choose the GPT model - newer models (gpt-5, gpt-4.1) are more capable; older models (gpt-4o-mini) are faster but less capable
- **api_key**: Use your OpenAI API key directly, or set to "rf_key:account" (default) to route requests through Roboflow's API proxy, which can help with rate limits and API key management
- **image_detail**: Control image processing quality - "auto" lets the model decide (good default), "high" for tasks requiring fine detail (slower, higher cost), "low" for simple tasks (faster, lower cost)
- **max_tokens**: Control maximum response length - increase for longer, detailed responses (e.g., comprehensive image analysis), decrease for shorter responses
- **temperature**: Control output randomness - lower values (0.0-0.5) produce more deterministic, focused responses; higher values (1.0-2.0) produce more creative, varied responses
- **max_concurrent_requests**: Limit concurrent API calls to stay within OpenAI API rate limits

## Common Use Cases

- **Content Analysis**: Analyze images for safety, quality, or compliance - ask questions like "Does this image contain inappropriate content?"
- **Document Processing**: Extract text from documents, forms, or receipts using OCR, then structure the data using structured-answering
- **Product Cataloging**: Classify product images into categories or extract product attributes like color, style, material
- **Accessibility**: Generate detailed image descriptions for visually impaired users using captioning tasks
- **Data Extraction**: Extract structured information from images (e.g., extract fields from forms, receipts, or documents)
- **Visual Q&A**: Build chatbots that can answer questions about images (e.g., "What brand is this product?", "Is this person wearing a mask?")

## Requirements

You need to provide your OpenAI API key to use this block. The API key is used to authenticate requests to OpenAI's GPT API. You can get your API key from [OpenAI's platform](https://platform.openai.com/api-keys). Alternatively, you can use "rf_key:account" (the default) to proxy requests through Roboflow's API, which can help with rate limit management. Note that API usage is subject to OpenAI's pricing and rate limits.

## Connecting to Other Blocks

The text outputs from this block can be connected to:
- **Parser blocks** (e.g., JSON Parser v1, VLM as Classifier v1) to convert text responses into structured data formats
- **Conditional logic blocks** to route workflow execution based on GPT's responses
- **Filter blocks** to filter images or data based on GPT's analysis
- **Visualization blocks** to display text overlays or annotations on images
- **Data storage blocks** to log responses for analytics or audit trails
- **Notification blocks** to send alerts based on GPT's findings

## Version Differences (v3 vs v2)

This version (v3) includes several enhancements over v2:
- **Roboflow API Proxy**: Supports proxying requests through Roboflow's API using "rf_key:account" (default) or "rf_key:user:<id>", which can help manage rate limits and API key security
- **Expanded Model Support**: Added support for additional models including o3 and o4-mini
- **Updated Default Model**: Default model changed to "gpt-5" (from "gpt-4o") to take advantage of the latest capabilities
"""


TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

TASKS_REQUIRING_PROMPT = {
    "unconstrained",
    "visual-question-answering",
}

TASKS_REQUIRING_CLASSES = {
    "classification",
    "multi-label-classification",
}

TASKS_REQUIRING_OUTPUT_STRUCTURE = {
    "structured-answering",
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenAI",
            "version": "v3",
            "short_description": "Run OpenAI's GPT models with vision capabilities.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "ChatGPT", "GPT", "OpenAI"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5,
                "popular": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/open_ai@v3"]
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
            },
            "always_visible": True,
        },
    )
    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to the OpenAI model",
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
        description="Your OpenAI API key",
        examples=["xxx-xxx", "$inputs.openai_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "o3",
            "o4-mini",
        ],
    ] = Field(
        default="gpt-5",
        description="Model to be used",
        examples=["gpt-5", "$inputs.openai_model"],
    )
    image_detail: Union[
        Selector(kind=[STRING_KIND]), Literal["auto", "high", "low"]
    ] = Field(
        default="auto",
        description="Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity.",
        examples=["auto", "high", "low"],
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
        "Please restrict if you hit OpenAI limits.",
    )

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


class OpenAIBlockV3(WorkflowBlock):

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
        image_detail: Literal["low", "high", "auto"],
        max_tokens: int,
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
        api_key: str = "rf_key:account",
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_outputs = run_gpt_4v_llm_prompting(
            roboflow_api_key=self._api_key,
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            openai_api_key=api_key,
            gpt_model_version=model_version,
            gpt_image_detail=image_detail,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]


def run_gpt_4v_llm_prompting(
    images: List[Dict[str, Any]],
    task_type: TaskType,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    roboflow_api_key: Optional[str],
    openai_api_key: Optional[str],
    gpt_model_version: str,
    gpt_image_detail: Literal["auto", "high", "low"],
    max_tokens: int,
    temperature: Optional[int],
    max_concurrent_requests: Optional[int],
) -> List[str]:
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    gpt4_prompts = []
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
            gpt_image_detail=gpt_image_detail,
        )
        gpt4_prompts.append(generated_prompt)
    return execute_gpt_4v_requests(
        roboflow_api_key=roboflow_api_key,
        openai_api_key=openai_api_key,
        gpt4_prompts=gpt4_prompts,
        gpt_model_version=gpt_model_version,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent_requests=max_concurrent_requests,
    )


def execute_gpt_4v_requests(
    roboflow_api_key: str,
    openai_api_key: str,
    gpt4_prompts: List[List[dict]],
    gpt_model_version: str,
    max_tokens: int,
    temperature: Optional[float],
    max_concurrent_requests: Optional[int],
) -> List[str]:
    tasks = [
        partial(
            execute_gpt_4v_request,
            roboflow_api_key=roboflow_api_key,
            openai_api_key=openai_api_key,
            prompt=prompt,
            gpt_model_version=gpt_model_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for prompt in gpt4_prompts
    ]
    max_workers = (
        max_concurrent_requests
        or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    )
    return run_in_parallel(
        tasks=tasks,
        max_workers=max_workers,
    )


def _execute_proxied_openai_request(
    roboflow_api_key: str,
    openai_api_key: str,
    prompt: List[dict],
    gpt_model_version: str,
    max_tokens: int,
    temperature: Optional[float],
) -> str:
    """Executes OpenAI request via Roboflow proxy."""
    # Build payload and endpoint outside error handling.
    payload = {
        "model": gpt_model_version,
        "messages": prompt,
        "max_completion_tokens": max_tokens,
        "openai_api_key": openai_api_key,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    endpoint = "apiproxy/openai"  # Use relative endpoint

    try:
        # Use the Roboflow API post function (this enures proper auth headers used based on invocation context)
        response_data = post_to_roboflow_api(
            endpoint=endpoint,
            api_key=roboflow_api_key,
            payload=payload,
        )
        return response_data["choices"][0]["message"]["content"]
    except (
        requests.exceptions.RequestException
    ) as e:  # Keep existing broad exception for now
        raise RuntimeError(f"Failed to connect to Roboflow proxy: {e}") from e
    except (KeyError, IndexError) as e:
        # Consider if specific error from post_to_roboflow_api should be caught
        # or if current error handling is sufficient
        raise RuntimeError(
            f"Invalid response structure from Roboflow proxy: {e} - Response: {response_data if 'response_data' in locals() else 'Unknown (request failed)'}"
        ) from e


def _execute_openai_request(
    openai_api_key: str,
    prompt: List[dict],
    gpt_model_version: str,
    max_tokens: int,
    temperature: Optional[float],
) -> str:
    """Executes OpenAI request directly."""
    # Use NOT_GIVEN only if needed, right away.
    temp_value = temperature if temperature is not None else NOT_GIVEN

    try:
        client = _get_openai_client(openai_api_key)
        # Required params tight together
        response = client.chat.completions.create(
            model=gpt_model_version,
            messages=prompt,
            max_completion_tokens=max_tokens,
            temperature=temp_value,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Don't do any extra logic except what is necessary.
        raise RuntimeError(f"OpenAI API request failed: {e}") from e


def execute_gpt_4v_request(
    roboflow_api_key: str,
    openai_api_key: str,
    prompt: List[dict],
    gpt_model_version: str,
    max_tokens: int,
    temperature: Optional[float],
) -> str:
    # Tuple-of-prefixes is faster for multiple startswith checks
    if openai_api_key.startswith(("rf_key:account", "rf_key:user:")):
        return _execute_proxied_openai_request(
            roboflow_api_key=roboflow_api_key,
            openai_api_key=openai_api_key,
            prompt=prompt,
            gpt_model_version=gpt_model_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        return _execute_openai_request(
            openai_api_key=openai_api_key,
            prompt=prompt,
            gpt_model_version=gpt_model_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def prepare_unconstrained_prompt(
    base64_image: str,
    prompt: str,
    gpt_image_detail: str,
    **kwargs,
) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": gpt_image_detail,
                    },
                },
            ],
        }
    ]


def prepare_classification_prompt(
    base64_image: str, classes: List[str], gpt_image_detail: str, **kwargs
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": "You act as single-class classification model. You must provide reasonable predictions. "
            "You are only allowed to produce JSON document in Markdown ```json [...]``` markers. "
            'Expected structure of json: {"class_name": "class-name", "confidence": 0.4}. '
            "`class-name` must be one of the class names defined by user. You are only allowed to return "
            "single JSON document, even if there are potentially multiple classes. You are not allowed to return list.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": gpt_image_detail,
                    },
                },
            ],
        },
    ]


def prepare_multi_label_classification_prompt(
    base64_image: str, classes: List[str], gpt_image_detail: str, **kwargs
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": "You act as multi-label classification model. You must provide reasonable predictions. "
            "You are only allowed to produce JSON document in Markdown ```json``` markers. "
            'Expected structure of json: {"predicted_classes": [{"class": "class-name-1", "confidence": 0.9}, '
            '{"class": "class-name-2", "confidence": 0.7}]}. '
            "`class-name-X` must be one of the class names defined by user and `confidence` is a float value in range "
            "0.0-1.0 that represent how sure you are that the class is present in the image. Only return class names "
            "that are visible.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": gpt_image_detail,
                    },
                },
            ],
        },
    ]


def prepare_vqa_prompt(
    base64_image: str, prompt: str, gpt_image_detail: str, **kwargs
) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "You act as Visual Question Answering model. Your task is to provide answer to question"
            "submitted by user. If this is open-question - answer with few sentences, for ABCD question, "
            "return only the indicator of the answer.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Question: {prompt}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": gpt_image_detail,
                    },
                },
            ],
        },
    ]


def prepare_ocr_prompt(
    base64_image: str, gpt_image_detail: str, **kwargs
) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "You act as OCR model. Your task is to read text from the image and return it in "
            "paragraphs representing the structure of texts in the image. You should only return "
            "recognised text, nothing else.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": gpt_image_detail,
                    },
                },
            ],
        },
    ]


def prepare_caption_prompt(
    base64_image: str, gpt_image_detail: str, short_description: bool, **kwargs
) -> List[dict]:
    caption_detail_level = "Caption should be short."
    if not short_description:
        caption_detail_level = "Caption should be extensive."
    return [
        {
            "role": "system",
            "content": f"You act as image caption model. Your task is to provide description of the image. "
            f"{caption_detail_level}",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": gpt_image_detail,
                    },
                },
            ],
        },
    ]


def prepare_structured_answering_prompt(
    base64_image: str, output_structure: Dict[str, str], gpt_image_detail: str, **kwargs
) -> List[dict]:
    output_structure_serialised = json.dumps(output_structure, indent=4)
    return [
        {
            "role": "system",
            "content": "You are supposed to produce responses in JSON wrapped in Markdown markers: "
            "```json\nyour-response\n```. User is to provide you dictionary with keys and values. "
            "Each key must be present in your response. Values in user dictionary represent "
            "descriptions for JSON fields to be generated. Provide only JSON Markdown in response.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Specification of requirements regarding output fields: \n"
                    f"{output_structure_serialised}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": gpt_image_detail,
                    },
                },
            ],
        },
    ]


def _get_openai_client(api_key: str):
    client = _openai_client_cache.get(api_key)
    if client is None:
        client = OpenAI(api_key=api_key)
        _openai_client_cache[api_key] = client
    return client


PROMPT_BUILDERS = {
    "unconstrained": prepare_unconstrained_prompt,
    "ocr": prepare_ocr_prompt,
    "visual-question-answering": prepare_vqa_prompt,
    "caption": partial(prepare_caption_prompt, short_description=True),
    "detailed-caption": partial(prepare_caption_prompt, short_description=False),
    "classification": prepare_classification_prompt,
    "multi-label-classification": prepare_multi_label_classification_prompt,
    "structured-answering": prepare_structured_answering_prompt,
}

_openai_client_cache = {}

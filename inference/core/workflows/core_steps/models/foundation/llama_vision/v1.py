import base64
import json
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

from openai import OpenAI
from pydantic import ConfigDict, Field, field_validator, model_validator

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
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

MODEL_VERSION_MAPPING = {
    "11B (Free) - OpenRouter": "meta-llama/llama-3.2-11b-vision-instruct:free",
    "11B (Regular) - OpenRouter": "meta-llama/llama-3.2-11b-vision-instruct",
    "90B (Free) - OpenRouter": "meta-llama/llama-3.2-90b-vision-instruct:free",
    "90B (Regular) - OpenRouter": "meta-llama/llama-3.2-90b-vision-instruct",
}

ModelVersion = Literal[
    "11B (Free) - OpenRouter",
    "11B (Regular) - OpenRouter",
    "90B (Free) - OpenRouter",
    "90B (Regular) - OpenRouter",
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
Run Meta's Llama 3.2 Vision model via OpenRouter API to perform various computer vision tasks including OCR, classification, captioning, visual question answering, and more.

## How This Block Works

This block takes one or more images as input and processes them through Meta's Llama 3.2 Vision model via the OpenRouter API. Based on the **task type** you select, the block:

1. Takes images and task-specific parameters (prompts, classes, or output structure) as input
2. Prepares the appropriate prompt for the Llama model based on the selected task type
3. Encodes images to base64 format for API transmission
4. Sends the request to OpenRouter API with the image and task-specific instructions
5. Returns the model's response as text output, which can be structured JSON or natural language depending on the task

The block supports multiple predefined task types, each optimized for specific use cases, or you can use "unconstrained" mode for completely custom prompts. The model is accessed through OpenRouter, which provides both free and paid API access to different model variants (11B and 90B parameter versions).

## Supported Task Types

The block supports the following task types:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

**Note:** The model can be unpredictable when structured output (JSON) is expected, particularly for tasks like `structured-answering`, `classification`, or `multi-label-classification`. This is due to content filtering mechanisms in the model. For more reliable structured outputs, consider using other VLM blocks like OpenAI GPT-4 Vision or Anthropic Claude.

## Common Use Cases

- **Image Analysis and Understanding**: Ask questions about images, get descriptions, or analyze visual content using natural language prompts
- **Text Extraction**: Extract text from images using OCR capabilities for document processing, receipt scanning, or text recognition
- **Image Classification**: Classify images into custom categories (single-label or multi-label) without training a model
- **Content Generation**: Generate captions or descriptions of images for accessibility, content indexing, or automated tagging
- **Visual Question Answering**: Answer specific questions about image content for interactive applications or content analysis
- **Structured Data Extraction**: Extract structured information from images using JSON output schemas (with noted reliability limitations)

## Connecting to Other Blocks

The outputs from this block can be connected to:

- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on the model's responses or extracted data
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log analysis results or extracted information
- **Expression blocks** to parse, validate, or transform the text or structured outputs
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts based on analysis results
- **Parser blocks** (e.g., JSON Parser, VLM as Classifier) to process structured outputs for specific task types
- **Additional workflow blocks** for further processing or analysis

## Requirements

This block requires an **OpenRouter API key**. The model is accessed through OpenRouter's API, which is a third-party service. OpenRouter provides access to multiple model variants:

- **11B versions**: Smaller, faster, and more cost-effective. Good for basic tasks and development.
- **90B versions**: Larger model with better quality results, suitable for production use cases requiring higher accuracy.

Both versions are available as:
- **Free**: Free tier available for OpenRouter clients (subject to availability and rate limits)
- **Regular**: Paid API access, typically faster and more reliable

Please review [OpenRouter pricing](https://openrouter.ai/) before use, as charges apply for paid API usage. Check the [model license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) and [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/) before using this model. See the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md) for more information.
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
            "name": "Llama 3.2 Vision",
            "version": "v1",
            "short_description": "Run Llama model with Vision capabilities",
            "long_description": LONG_DESCRIPTION,
            "license": "Llama 3.2 Community",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Llama", "Vision", "Meta"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-brands fa-meta",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/llama_3_2_vision@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    task_type: TaskType = Field(
        default="unconstrained",
        description="Task type to be performed by the model. Select from: unconstrained (custom prompts), ocr (text recognition), structured-answering (JSON output), classification (single-label), multi-label-classification, visual-question-answering (answer questions about images), caption (short description), or detailed-caption (long description). The task type determines which other parameters are required (e.g., prompt for VQA, classes for classification, output_structure for structured-answering) and the format of the response.",
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
        description="Text prompt for the Llama model. Required for 'unconstrained' and 'visual-question-answering' task types. Can be any question, instruction, or request about the image. For unconstrained mode, use any custom prompt. For VQA, ask specific questions about the image content (e.g., 'What objects are in this image?', 'Describe the scene').",
        examples=["What objects are in this image?", "Describe the scene in detail", "$inputs.prompt"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": TASKS_REQUIRING_PROMPT, "required": True},
            },
            "multiline": True,
        },
    )
    output_structure: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary defining the expected JSON output structure for 'structured-answering' task type. Maps field names to their descriptions (e.g., {'count': 'number of objects', 'color': 'dominant color'}). The model will attempt to return JSON matching this structure. Note: The model can be unpredictable with structured outputs due to content filtering. Required for structured-answering task type.",
        examples=[{"count": "number of objects", "color": "dominant color"}, {"objects": "list of detected objects"}, "$inputs.output_structure"],
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
        description="List of class names for classification tasks. Required for 'classification' (single-label) and 'multi-label-classification' task types. Provide descriptive class names (e.g., ['cat', 'dog', 'bird'] for single-label or ['sunny', 'cloudy', 'rainy'] for multi-label). For single-label classification, the model selects one class. For multi-label, the model can select multiple classes. Note: Classification results can be unpredictable due to model content filtering.",
        examples=[["cat", "dog", "bird"], ["happy", "sad", "neutral"], "$inputs.classes"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": TASKS_REQUIRING_CLASSES,
                    "required": True,
                },
            },
        },
    )
    api_key: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="OpenRouter API key required to access the Llama 3.2 Vision model. You can obtain an API key from https://openrouter.ai/docs/api-keys. OpenRouter provides access to the model via their API service. This field is kept private for security.",
        examples=["sk-or-v1-xxx-xxx", "$inputs.openrouter_api_key", "$secrets.openrouter_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "11B (Free) - OpenRouter",
            "11B (Regular) - OpenRouter",
            "90B (Free) - OpenRouter",
            "90B (Regular) - OpenRouter",
        ],
    ] = Field(
        default="11B (Free) - OpenRouter",
        description="Llama 3.2 Vision model variant to use. Options: '11B' (11 billion parameters - faster, cheaper, good for basic tasks) or '90B' (90 billion parameters - higher quality, better for production). Both are available as 'Free' (free tier, subject to availability) or 'Regular' (paid, faster and more reliable). Default is 11B Free. For production use cases requiring higher accuracy, consider 90B Regular.",
        examples=["11B (Free) - OpenRouter", "90B (Regular) - OpenRouter", "$inputs.model_version"],
    )
    max_tokens: int = Field(
        default=500,
        description="Maximum number of tokens (words/characters) the model can generate in its response. Higher values allow longer responses but may increase cost and processing time. Must be greater than 1. Default is 500 tokens.",
        gt=1,
    )
    temperature: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.1,
        description="Temperature parameter controlling randomness in the model's responses. Range 0.0-2.0. Lower values (closer to 0) produce more deterministic, focused responses. Higher values (closer to 2) produce more random and creative responses. Default is 0.1 for consistent, focused outputs. Use higher values for creative tasks and lower values for factual or classification tasks.",
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description="Number of concurrent requests that can be executed by block when batch of input images provided. "
        "If not given - block defaults to value configured globally in Workflows Execution Engine. "
        "Please restrict if you hit limits.",
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

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: Union[str, float]) -> Union[str, float]:
        if isinstance(value, str):
            return value
        if value < 0.0 or value > 2.0:
            raise ValueError(
                "'temperature' parameter required to be in range [0.0, 2.0]"
            )
        return value

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
        return ">=1.3.0,<2.0.0"


class LlamaVisionBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
    ):
        self._model_manager = model_manager

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager"]

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
        model_version: ModelVersion,
        max_tokens: int,
        temperature: float,
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_outputs = run_llama_vision_32_llm_prompting(
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            llama_api_key=api_key,
            llama_model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]


def run_llama_vision_32_llm_prompting(
    images: List[Dict[str, Any]],
    task_type: TaskType,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    llama_api_key: Optional[str],
    llama_model_version: ModelVersion,
    max_tokens: int,
    temperature: float,
    max_concurrent_requests: Optional[int],
) -> List[str]:
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    model_version_id = MODEL_VERSION_MAPPING.get(llama_model_version)
    if not llama_model_version:
        raise ValueError(
            f"Invalid model name: '{llama_model_version}'. Please use one of {list(MODEL_VERSION_MAPPING.keys())}."
        )
    llama_prompts = []
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
        )
        llama_prompts.append(generated_prompt)
    return execute_llama_vision_32_requests(
        llama_api_key=llama_api_key,
        llama_prompts=llama_prompts,
        model_version_id=model_version_id,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent_requests=max_concurrent_requests,
    )


def execute_llama_vision_32_requests(
    llama_api_key: str,
    llama_prompts: List[List[dict]],
    model_version_id: str,
    max_tokens: int,
    temperature: float,
    max_concurrent_requests: Optional[int],
) -> List[str]:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=llama_api_key)
    tasks = [
        partial(
            execute_llama_vision_32_request,
            client=client,
            prompt=prompt,
            llama_model_version=model_version_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for prompt in llama_prompts
    ]
    max_workers = (
        max_concurrent_requests
        or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    )
    return run_in_parallel(
        tasks=tasks,
        max_workers=max_workers,
    )


def execute_llama_vision_32_request(
    client: OpenAI,
    prompt: List[dict],
    llama_model_version: str,
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=llama_model_version,
        messages=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response.choices is None:
        error_detail = getattr(response, "error", {}).get("message", "N/A")
        raise RuntimeError(
            "OpenRouter provider failed in delivering response. This issue happens from time "
            "to time, especially when using Free offering - raise issue to OpenRouter if that's "
            f"problematic for you. Details: {error_detail}"
        )
    return response.choices[0].message.content


def prepare_unconstrained_prompt(
    base64_image: str,
    prompt: str,
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
                    },
                },
            ],
        }
    ]


def prepare_classification_prompt(
    base64_image: str, classes: List[str], **kwargs
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": "You act as single-class classification model. You must provide reasonable predictions. "
            "You are only allowed to produce JSON document in Markdown ```json [...]``` markers. "
            'Expected structure of json: {"class_name": "class-name", "confidence": 0.4}. '
            "`class-name` must be one of the class names defined by user. You are only allowed to return "
            "single JSON document, even if there are potentially multiple classes. You are not allowed to return list. "
            "You cannot discuss the result, you are only allowed to return JSON document.",
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
                    },
                },
            ],
        },
    ]


def prepare_multi_label_classification_prompt(
    base64_image: str, classes: List[str], **kwargs
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
            "that are visible. You cannot discuss the result, you are only allowed to return JSON document.",
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
                    },
                },
            ],
        },
    ]


def prepare_vqa_prompt(base64_image: str, prompt: str, **kwargs) -> List[dict]:
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
                    },
                },
            ],
        },
    ]


def prepare_ocr_prompt(base64_image: str, **kwargs) -> List[dict]:
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
                    },
                },
            ],
        },
    ]


def prepare_caption_prompt(
    base64_image: str, short_description: bool, **kwargs
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
                    },
                },
            ],
        },
    ]


def prepare_structured_answering_prompt(
    base64_image: str, output_structure: Dict[str, str], **kwargs
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
                    },
                },
            ],
        },
    ]


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

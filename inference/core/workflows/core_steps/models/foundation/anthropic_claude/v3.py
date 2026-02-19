import base64
import json
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import anthropic
import requests
from anthropic import NOT_GIVEN
from pydantic import ConfigDict, Field, model_validator

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio
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
    INTEGER_KIND,
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

CLAUDE_MODELS = [
    {
        "id": "claude-opus-4-6",
        "name": "Claude Opus 4.6",
        "exact_version": "claude-opus-4-6",
        "max_output_tokens": 128000,
    },
    {
        "id": "claude-sonnet-4-5",
        "name": "Claude Sonnet 4.5",
        "exact_version": "claude-sonnet-4-5-20250929",
        "max_output_tokens": 64000,
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "exact_version": "claude-haiku-4-5-20251001",
        "max_output_tokens": 64000,
    },
    {
        "id": "claude-opus-4-5",
        "name": "Claude Opus 4.5",
        "exact_version": "claude-opus-4-5-20251101",
        "max_output_tokens": 64000,
    },
    {
        "id": "claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "exact_version": "claude-sonnet-4-20250514",
        "max_output_tokens": 64000,
    },
    {
        "id": "claude-opus-4-1",
        "name": "Claude Opus 4.1",
        "exact_version": "claude-opus-4-1-20250805",
        "max_output_tokens": 32000,
    },
    {
        "id": "claude-opus-4",
        "name": "Claude Opus 4",
        "exact_version": "claude-opus-4-20250514",
        "max_output_tokens": 32000,
    },
]

MODEL_VERSION_IDS = [model["id"] for model in CLAUDE_MODELS]
EXACT_MODEL_VERSIONS = {model["id"]: model["exact_version"] for model in CLAUDE_MODELS}

MODEL_VERSION_METADATA = {
    model["id"]: {"name": model["name"]} for model in CLAUDE_MODELS
}

MAX_OUTPUT_TOKENS = {model["id"]: model["max_output_tokens"] for model in CLAUDE_MODELS}
DEFAULT_MAX_OUTPUT_TOKENS = 64000

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
Ask a question to Anthropic Claude model with vision capabilities.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

### API Key Options

This block supports two API key modes:

1. **Roboflow Managed API Key (Default)** - Use `rf_key:account` to proxy requests through Roboflow's API:
   * **Simplified setup** - no Anthropic API key required
   * **Secure** - your workflow API key is used for authentication
   * **Usage-based billing** - charged per token based on the model used

2. **Custom Anthropic API Key** - Provide your own Anthropic API key:
   * Full control over API usage
   * You pay Anthropic directly
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
            "name": "Anthropic Claude",
            "version": "v3",
            "short_description": "Run Anthropic Claude model with vision capabilities.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Claude", "Anthropic"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-a",
                "blockPriority": 5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/anthropic_claude@v3"]
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
        description="Text prompt to the Claude model",
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
        description="Your Anthropic API key or 'rf_key:account' to use Roboflow's managed API key",
        examples=["rf_key:account", "xxx-xxx", "$inputs.anthropic_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal[tuple(MODEL_VERSION_IDS)],
    ] = Field(
        default="claude-sonnet-4-5",
        description="Model to be used",
        examples=["claude-sonnet-4-5", "$inputs.claude_model"],
        json_schema_extra={
            "values_metadata": MODEL_VERSION_METADATA,
        },
    )
    extended_thinking: Optional[bool] = Field(
        default=None,
        description="Enable extended thinking for deeper reasoning on complex tasks. "
        "Note: temperature cannot be used when extended thinking is enabled.",
    )
    thinking_budget_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens for internal thinking when extended thinking is enabled. "
        "Higher values allow deeper reasoning but increase latency and cost. "
        "Must be less than max_tokens. Minimum: 1024.",
        ge=1024,
        json_schema_extra={
            "relevant_for": {
                "extended_thinking": {
                    "values": [True],
                    "required": False,
                },
            },
        },
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens the model can generate in its response.",
    )
    temperature: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Temperature to sample from the model - value in range 0.0-1.0, the higher - the more "
        'random / "creative" the generations are. Cannot be used when extended_thinking is enabled.',
        ge=0.0,
        le=1.0,
    )
    max_image_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        description="Maximum size of the image - if input has larger side, it will be downscaled, keeping aspect ratio",
        default=1024,
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description="Number of concurrent requests that can be executed by block when batch of input images provided. "
        "If not given - block defaults to value configured globally in Workflows Execution Engine. "
        "Please restrict if you hit Anthropic API limits.",
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
        if self.extended_thinking:
            if self.temperature is not None:
                raise ValueError(
                    "`temperature` cannot be used when `extended_thinking` is enabled"
                )
            budget_tokens = self.thinking_budget_tokens
            max_tokens = self.max_tokens
            if budget_tokens and max_tokens and budget_tokens >= max_tokens:
                raise ValueError(
                    f"`thinking_budget_tokens` ({budget_tokens}) must be less than `max_tokens` ({max_tokens})"
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


class AnthropicClaudeBlockV3(WorkflowBlock):

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
        extended_thinking: Optional[bool],
        thinking_budget_tokens: Optional[int],
        max_image_size: int,
        max_concurrent_requests: Optional[int],
        api_key: str = "rf_key:account",
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_outputs = run_claude_prompting(
            roboflow_api_key=self._api_key,
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            anthropic_api_key=api_key,
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            max_image_size=max_image_size,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]


def run_claude_prompting(
    roboflow_api_key: Optional[str],
    images: List[Dict[str, Any]],
    task_type: TaskType,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    anthropic_api_key: str,
    model_version: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    extended_thinking: Optional[bool],
    thinking_budget_tokens: Optional[int],
    max_image_size: int,
    max_concurrent_requests: Optional[int],
) -> List[str]:
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    prompts = []
    for image in images:
        loaded_image, _ = load_image(image)
        loaded_image = downscale_image_keeping_aspect_ratio(
            image=loaded_image, desired_size=(max_image_size, max_image_size)
        )
        base64_image = base64.b64encode(
            encode_image_to_jpeg_bytes(loaded_image)
        ).decode("ascii")
        generated_prompt = PROMPT_BUILDERS[task_type](
            base64_image=base64_image,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        prompts.append(generated_prompt)
    return execute_claude_requests(
        roboflow_api_key=roboflow_api_key,
        anthropic_api_key=anthropic_api_key,
        prompts=prompts,
        model_version=model_version,
        max_tokens=max_tokens,
        temperature=temperature,
        extended_thinking=extended_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        max_concurrent_requests=max_concurrent_requests,
    )


def execute_claude_requests(
    roboflow_api_key: Optional[str],
    anthropic_api_key: str,
    prompts: List[Tuple[Optional[str], List[dict]]],
    model_version: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    extended_thinking: Optional[bool],
    thinking_budget_tokens: Optional[int],
    max_concurrent_requests: Optional[int],
) -> List[str]:
    tasks = [
        partial(
            execute_claude_request,
            roboflow_api_key=roboflow_api_key,
            anthropic_api_key=anthropic_api_key,
            system_prompt=prompt[0],
            messages=prompt[1],
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        for prompt in prompts
    ]
    max_workers = (
        max_concurrent_requests
        or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    )
    return run_in_parallel(
        tasks=tasks,
        max_workers=max_workers,
    )


def execute_claude_request(
    roboflow_api_key: Optional[str],
    anthropic_api_key: str,
    system_prompt: Optional[str],
    messages: List[dict],
    model_version: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    extended_thinking: Optional[bool],
    thinking_budget_tokens: Optional[int],
) -> str:
    """Route to proxied or direct execution based on API key format."""
    if anthropic_api_key.startswith(("rf_key:account", "rf_key:user:")):
        return _execute_proxied_claude_request(
            roboflow_api_key=roboflow_api_key,
            anthropic_api_key=anthropic_api_key,
            system_prompt=system_prompt,
            messages=messages,
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
    else:
        return _execute_direct_claude_request(
            anthropic_api_key=anthropic_api_key,
            system_prompt=system_prompt,
            messages=messages,
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )


def _execute_proxied_claude_request(
    roboflow_api_key: str,
    anthropic_api_key: str,
    system_prompt: Optional[str],
    messages: List[dict],
    model_version: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    extended_thinking: Optional[bool],
    thinking_budget_tokens: Optional[int],
) -> str:
    """Execute Claude request via Roboflow proxy."""
    model_max_output = MAX_OUTPUT_TOKENS.get(model_version, DEFAULT_MAX_OUTPUT_TOKENS)
    effective_max_tokens = max_tokens if max_tokens is not None else model_max_output

    payload = {
        "model": model_version,
        "anthropic_api_key": anthropic_api_key,
        "messages": messages,
        "max_tokens": effective_max_tokens,
    }

    if system_prompt is not None:
        payload["system"] = system_prompt

    if temperature is not None and not extended_thinking:
        payload["temperature"] = temperature

    if extended_thinking:
        effective_budget = (
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else model_max_output // 2
        )
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": effective_budget,
        }

    endpoint = "apiproxy/anthropic"

    try:
        response_data = post_to_roboflow_api(
            endpoint=endpoint,
            api_key=roboflow_api_key,
            payload=payload,
        )
        return _extract_claude_response_text(response_data)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to Roboflow proxy: {e}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Invalid response structure from Roboflow proxy: {e}"
        ) from e


def _execute_direct_claude_request(
    anthropic_api_key: str,
    system_prompt: Optional[str],
    messages: List[dict],
    model_version: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    extended_thinking: Optional[bool],
    thinking_budget_tokens: Optional[int],
) -> str:
    """Execute Claude request directly to Anthropic API."""
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    if system_prompt is None:
        system_prompt = NOT_GIVEN

    if temperature is None or extended_thinking:
        temperature = NOT_GIVEN

    model_max_output = MAX_OUTPUT_TOKENS.get(model_version, DEFAULT_MAX_OUTPUT_TOKENS)
    effective_max_tokens = max_tokens if max_tokens is not None else model_max_output

    request_params = {
        "system": system_prompt,
        "messages": messages,
        "max_tokens": effective_max_tokens,
        "model": EXACT_MODEL_VERSIONS.get(model_version, model_version),
        "temperature": temperature,
    }

    if extended_thinking:
        effective_budget = (
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else model_max_output // 2
        )
        request_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": effective_budget,
        }

    # Stream response to avoid max_tokens limitation
    with client.messages.stream(**request_params) as stream:
        result = stream.get_final_message()

    return _validate_and_extract_direct_response(result)


def _validate_and_extract_direct_response(result) -> str:
    """Validate and extract text from direct Anthropic API response."""
    stop_reason = result.stop_reason

    if stop_reason == "max_tokens":
        raise ValueError(
            "Claude API stopped generation because the max_tokens limit was reached. "
            "Please increase the max_tokens parameter to allow for a complete response."
        )

    if stop_reason not in ["end_turn", "stop_sequence"]:
        raise ValueError(
            f"Claude API stopped generation with unexpected stop reason: {stop_reason}."
        )

    # Ignore thinking blocks and return text content
    for block in result.content:
        if block.type == "text":
            return block.text

    raise ValueError("Claude API returned no text content in response.")


def _extract_claude_response_text(response_data: dict) -> str:
    """Extract text content from Claude API response (proxied)."""
    stop_reason = response_data.get("stop_reason")

    if stop_reason == "max_tokens":
        raise ValueError(
            "Claude API stopped generation because the max_tokens limit was reached. "
            "Please increase the max_tokens parameter to allow for a complete response."
        )

    if stop_reason not in ["end_turn", "stop_sequence", None]:
        raise ValueError(
            f"Claude API stopped generation with unexpected stop reason: {stop_reason}."
        )

    content = response_data.get("content", [])
    if not content:
        raise ValueError("Claude API returned no content in response.")

    # Ignore thinking blocks and return text content
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")

    raise ValueError("Claude API returned no text content in response.")


def prepare_unconstrained_prompt(
    base64_image: str,
    prompt: str,
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]
    return None, messages


def prepare_classification_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    serialised_classes = ", ".join(classes)
    system_prompt = (
        "You act as single-class classification model. You must provide reasonable predictions. "
        "You are only allowed to produce JSON document. "
        'Expected structure of json: {"class_name": "class-name", "confidence": 0.4}. '
        "`class-name` must be one of the class names defined by user. You are only allowed to return "
        "single JSON document, even if there are potentially multiple classes. You are not allowed to "
        "return list."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_multi_label_classification_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    serialised_classes = ", ".join(classes)
    system_prompt = (
        "You act as multi-label classification model. You must provide reasonable predictions. "
        "You are only allowed to produce JSON document. "
        'Expected structure of json: {"predicted_classes": [{"class": "class-name-1", "confidence": 0.9}, '
        '{"class": "class-name-2", "confidence": 0.7}]}.'
        "`class-name-X` must be one of the class names defined by user and `confidence` is a float value "
        "in range 0.0-1.0 that represents how sure you are that the class is present in the image. "
        "Only return class names that are visible."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_vqa_prompt(
    base64_image: str,
    prompt: str,
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    system_prompt = (
        "You act as Visual Question Answering model. Your task is to provide answer to question"
        "submitted by user. If this is open-question - answer with few sentences, for ABCD question, "
        "return only the indicator of the answer."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"Question: {prompt}",
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_ocr_prompt(
    base64_image: str,
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    system_prompt = (
        "You act as OCR model. Your task is to read text from the image and return it in "
        "paragraphs representing the structure of texts in the image. You should only return "
        "recognised text, nothing else."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_caption_prompt(
    base64_image: str,
    short_description: bool,
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    caption_detail_level = "Caption should be short."
    if not short_description:
        caption_detail_level = "Caption should be extensive."
    system_prompt = (
        f"You act as image caption model. Your task is to provide description of the image. "
        f"{caption_detail_level}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_structured_answering_prompt(
    base64_image: str,
    output_structure: Dict[str, str],
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    output_structure_serialised = json.dumps(output_structure, indent=4)
    system_prompt = (
        "You are supposed to produce responses in JSON. User is to provide you dictionary with "
        "keys and values. Each key must be present in your response. Values in user dictionary "
        "represent descriptions for JSON fields to be generated. Provide only JSON in response."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"Specification of requirements regarding output fields: \n"
                    f"{output_structure_serialised}",
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_object_detection_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    serialised_classes = ", ".join(classes)
    system_prompt = (
        "You act as object-detection model. You must provide reasonable predictions. "
        "You are only allowed to produce JSON document. "
        'Expected structure of json: {"detections": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4, "class_name": "my-class-X", "confidence": 0.7}]} '
        "- remember to close top-level dictionary at the end. "
        "`my-class-X` must be one of the class names defined by user. All coordinates must be in range 0.0-1.0, representing percentage of image dimensions. "
        "`confidence` is a value in range 0.0-1.0 representing your confidence in prediction. You should detect all instances of classes provided by user."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
        }
    ]
    return system_prompt, messages


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

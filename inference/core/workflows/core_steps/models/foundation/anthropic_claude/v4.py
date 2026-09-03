"""Anthropic Claude workflow block (v4).

Extends v3 with token-usage outputs and an object-detection path aligned with
the vlm-exam benchmark contract for Claude models: images for that task are
pre-resized to the exact dimensions Claude's internal resize would produce
(high-resolution tier) and sent as lossless PNG, and the prompt asks for a
JSON list of ``box_2d``/``label`` entries with ``[x_min, y_min, x_max, y_max]``
in absolute pixel coordinates of the uploaded image, with the image placed
before the text and no system prompt. Use ``roboflow_core/vlm_as_detector@v2``
with ``model_type="anthropic-claude"`` to parse the output.
"""

import base64
import json
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import anthropic
import cv2
import numpy as np
import requests
from anthropic import NOT_GIVEN
from pydantic import ConfigDict, Field, model_validator

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio
from inference.core.workflows.core_steps.common.token_usage import (
    TOKEN_OUTPUT_DEFINITIONS,
    parse_responses_api_usage,
)
from inference.core.workflows.core_steps.common.utils import (
    compute_anthropic_upload_dimensions,
    run_in_parallel,
)
from inference.core.workflows.core_steps.common.vlms import VLM_TASKS_METADATA
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.model_capabilities import (
    build_thinking_config,
    resolve_temperature,
)
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
    AirGappedAvailability,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    is_workflow_selector,
    third_party_model,
)

CLAUDE_MODELS = [
    {
        "id": "claude-fable-5-1",
        "name": "Claude Fable 5.1",
        "exact_version": "claude-fable-5-1",
        "max_output_tokens": 128000,
    },
    {
        "id": "claude-fable-5",
        "name": "Claude Fable 5",
        "exact_version": "claude-fable-5",
        "max_output_tokens": 128000,
    },
    {
        "id": "claude-opus-5",
        "name": "Claude Opus 5",
        "exact_version": "claude-opus-5",
        "max_output_tokens": 128000,
    },
    {
        "id": "claude-sonnet-5",
        "name": "Claude Sonnet 5",
        "exact_version": "claude-sonnet-5",
        "max_output_tokens": 128000,
    },
    {
        "id": "claude-opus-4-8",
        "name": "Claude Opus 4.8",
        "exact_version": "claude-opus-4-8",
        "max_output_tokens": 128000,
    },
    {
        "id": "claude-opus-4-7",
        "name": "Claude Opus 4.7",
        "exact_version": "claude-opus-4-7",
        "max_output_tokens": 128000,
    },
    {
        "id": "claude-opus-4-6",
        "name": "Claude Opus 4.6",
        "exact_version": "claude-opus-4-6",
        "max_output_tokens": 128000,
    },
    {
        "id": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "exact_version": "claude-sonnet-4-6",
        "max_output_tokens": 64000,
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

DETECTION_MAX_PNG_PAYLOAD_BYTES = 2_500_000
"""Largest PNG payload sent for object detection, before base64 growth.

Lossless PNG of a large photographic upload can exceed the request body
limits of the Roboflow proxy and Anthropic's per-image maximum. Above this
size the block re-encodes the image as JPEG (quality 95) at the same
resolution, which keeps the coordinate contract intact.
"""

DETECTION_JPEG_FALLBACK_QUALITY = 95

OBJECT_DETECTION_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners in absolute pixel coordinates "
    "of the {width}x{height} pixel image. "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
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

The `object-detection` task asks Claude for a JSON list of
`{{"box_2d": [x_min, y_min, x_max, y_max], "label": ...}}` entries where
coordinates are absolute pixels of the uploaded image. The image is
pre-resized to the exact dimensions Claude's internal resize would produce
and sent as lossless PNG, matching the vlm-exam benchmark setup for Claude
models; the `max_image_size` parameter is not applied to this task. Use
`roboflow_core/vlm_as_detector@v2` with `model_type="anthropic-claude"` to
convert the output into predictions. Confidence scores are not requested;
the parser assigns `1.0`.

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
            "version": "v4",
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
    type: Literal["roboflow_core/anthropic_claude@v4"]
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
        "Note: temperature cannot be used when extended thinking is enabled. Models that "
        "only support adaptive thinking (Claude Opus 4.7 and newer) ignore `thinking_budget_tokens`.",
    )
    thinking_budget_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens for internal thinking when extended thinking is enabled. "
        "Higher values allow deeper reasoning but increase latency and cost. "
        "Must be less than max_tokens. Minimum: 1024. Ignored by models that only support "
        "adaptive thinking (Claude Opus 4.7 and newer).",
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
        'random / "creative" the generations are. Cannot be used when extended_thinking is enabled. '
        "Ignored by models that no longer accept sampling parameters (Claude Opus 4.7 and newer).",
        ge=0.0,
        le=1.0,
    )
    max_image_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        description="Maximum size of the image - if input has larger side, it will be downscaled, keeping aspect ratio. "
        "Not applied to the `object-detection` task, which pre-resizes images to Claude's native resolution instead.",
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
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

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
            *TOKEN_OUTPUT_DEFINITIONS,
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        if is_workflow_selector(self.model_version):
            # Selector returned verbatim; the attached resolver performs the
            # EXACT_MODEL_VERSIONS lookup once the input value is substituted.
            return [
                third_party_model(
                    provider="anthropic",
                    model_id=self.model_version,
                    model_id_resolver=lambda label: EXACT_MODEL_VERSIONS.get(
                        label, label
                    ),
                )
            ]
        return [
            third_party_model(
                provider="anthropic",
                model_id=EXACT_MODEL_VERSIONS.get(
                    self.model_version, self.model_version
                ),
            )
        ]


class AnthropicClaudeBlockV4(WorkflowBlock):

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
        return ">=1.4.0,<2.0.0"

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
            {
                "output": content,
                "classes": classes,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            for content, input_tokens, output_tokens in raw_outputs
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
) -> List[Tuple[str, Optional[int], Optional[int]]]:
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    prompts = []
    for image in images:
        loaded_image, _ = load_image(image)
        base64_image, media_type, image_width, image_height = encode_image_for_task(
            loaded_image, task_type=task_type, max_image_size=max_image_size
        )
        generated_prompt = PROMPT_BUILDERS[task_type](
            base64_image=base64_image,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            image_width=image_width,
            image_height=image_height,
            media_type=media_type,
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


def encode_image_for_task(
    image: np.ndarray, *, task_type: TaskType, max_image_size: int
) -> Tuple[str, int, int]:
    """Encode an image as base64 using task-appropriate preprocessing.

    The ``object-detection`` task pre-resizes the image to the exact
    dimensions Claude's internal resize would produce (high-resolution tier)
    and encodes it as lossless PNG, so pixel coordinates returned by the
    model map one-to-one onto the uploaded image - matching the vlm-exam
    benchmark setup the absolute-pixel contract was validated with. All other
    tasks downscale to ``max_image_size`` and send JPEG, as previous block
    versions did.

    Args:
        image: BGR image to be encoded.
        task_type: Task type determining the preprocessing applied.
        max_image_size: Maximum longest edge applied to non-detection tasks.

    Returns:
        Tuple of the base64-encoded image payload (without a data URL prefix),
        its media type, and the ``(width, height)`` of the encoded image.
    """
    if task_type == "object-detection":
        encoded_image = _resize_image_to_anthropic_upload_dimensions(image)
        image_bytes = _encode_image_to_png_bytes(encoded_image)
        media_type = "image/png"
        if len(image_bytes) > DETECTION_MAX_PNG_PAYLOAD_BYTES:
            image_bytes = _encode_image_to_jpeg_bytes_with_quality(
                encoded_image, quality=DETECTION_JPEG_FALLBACK_QUALITY
            )
            media_type = "image/jpeg"
    else:
        encoded_image = downscale_image_keeping_aspect_ratio(
            image=image, desired_size=(max_image_size, max_image_size)
        )
        image_bytes = encode_image_to_jpeg_bytes(encoded_image)
        media_type = "image/jpeg"

    base64_image = base64.b64encode(image_bytes).decode("ascii")
    encoded_height, encoded_width = encoded_image.shape[:2]

    return base64_image, media_type, encoded_width, encoded_height


def _resize_image_to_anthropic_upload_dimensions(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target_width, target_height = compute_anthropic_upload_dimensions(width, height)
    if (target_width, target_height) == (width, height):
        return image

    return cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )


def _encode_image_to_png_bytes(image: np.ndarray) -> bytes:
    _, encoded_image = cv2.imencode(".png", image)
    return encoded_image.tobytes()


def _encode_image_to_jpeg_bytes_with_quality(
    image: np.ndarray, *, quality: int
) -> bytes:
    _, encoded_image = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded_image.tobytes()


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
) -> List[Tuple[str, Optional[int], Optional[int]]]:
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
) -> Tuple[str, Optional[int], Optional[int]]:
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
) -> Tuple[str, Optional[int], Optional[int]]:
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

    temperature = resolve_temperature(
        temperature,
        model_version=model_version,
        extended_thinking=extended_thinking,
    )
    if temperature is not None:
        payload["temperature"] = temperature

    thinking = build_thinking_config(
        extended_thinking=extended_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        model_version=model_version,
        max_tokens=effective_max_tokens,
    )
    if thinking is not None:
        payload["thinking"] = thinking

    endpoint = "apiproxy/anthropic"

    try:
        response_data = post_to_roboflow_api(
            endpoint=endpoint,
            api_key=roboflow_api_key,
            payload=payload,
        )
        text = _extract_claude_response_text(response_data)
        input_tokens, output_tokens = parse_responses_api_usage(
            response_data.get("usage")
        )
        return text, input_tokens, output_tokens
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
) -> Tuple[str, Optional[int], Optional[int]]:
    """Execute Claude request directly to Anthropic API."""
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    if system_prompt is None:
        system_prompt = NOT_GIVEN

    temperature = resolve_temperature(
        temperature,
        model_version=model_version,
        extended_thinking=extended_thinking,
    )
    if temperature is None:
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

    thinking = build_thinking_config(
        extended_thinking=extended_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        model_version=model_version,
        max_tokens=effective_max_tokens,
    )
    if thinking is not None:
        request_params["thinking"] = thinking

    # Stream response to avoid max_tokens limitation
    with client.messages.stream(**request_params) as stream:
        result = stream.get_final_message()

    text = _validate_and_extract_direct_response(result)
    input_tokens, output_tokens = parse_responses_api_usage(
        getattr(result, "usage", None)
    )
    return text, input_tokens, output_tokens


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
    image_width: int,
    image_height: int,
    media_type: str = "image/png",
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    """Build the absolute-pixel detection request used by Claude models.

    Matches the vlm-exam benchmark setup: the image placed before the text
    prompt, no system prompt, and coordinates requested as absolute pixels of
    the uploaded ``image_width`` x ``image_height`` image. The image is
    lossless PNG unless its payload exceeded the size limit, in which case
    it is JPEG at the same resolution.

    Args:
        base64_image: Base64-encoded image.
        classes: Class names the model may predict.
        image_width: Width of the uploaded image in pixels.
        image_height: Height of the uploaded image in pixels.
        media_type: Media type of the encoded image.
        **kwargs: Ignored builder arguments shared across task types.

    Returns:
        Tuple of the system prompt (``None``) and the request messages.
    """
    class_list = ", ".join(classes)
    prompt_text = OBJECT_DETECTION_PROMPT_TEMPLATE.format(
        width=image_width,
        height=image_height,
        class_list=class_list,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]
    return None, messages


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

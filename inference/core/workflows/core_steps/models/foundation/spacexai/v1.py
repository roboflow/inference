"""SpaceXAI (Grok) workflow block.

Calls Grok vision models via xAI's OpenAI-compatible Responses API, either
directly with a user-provided xAI key or through Roboflow's ``apiproxy/xai``
managed-key proxy. Object-detection prompting uses the percent-of-image
``box_2d`` contract validated in the vlm-exam benchmark for Grok 4.5/4.6.
"""

import base64
import json
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import requests
from openai import OpenAI
from pydantic import ConfigDict, Field, model_validator

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.workflows.core_steps.common.utils import (
    DETECTION_MAX_EDGE_PIXELS,
    run_in_parallel,
    scale_dimensions_to_max_edge,
)
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
    AirGappedAvailability,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    third_party_model,
)

XAI_BASE_URL = "https://api.x.ai/v1"

GROK_MODELS = [
    {
        "id": "grok-4.6",
        "name": "Grok 4.6",
    },
    {
        "id": "grok-4.5",
        "name": "Grok 4.5",
    },
]

MODEL_VERSION_IDS = [model["id"] for model in GROK_MODELS]

MODEL_VERSION_METADATA = {model["id"]: {"name": model["name"]} for model in GROK_MODELS}

OBJECT_DETECTION_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the text label in the key "
    '"label" and the 2D bounding box in the key "box_2d". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max] as percentages '
    "of image width and height (floats between 0 and 100). "
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
Ask a question to SpaceXAI Grok models with vision capabilities.

You can specify arbitrary text prompts or predefined ones, the block supports
the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

The `object-detection` task asks Grok for a JSON list of
`{{"label": ..., "box_2d": [x_min, y_min, x_max, y_max]}}` entries where
coordinates are percentages of image width and height (floats 0-100). Use
`roboflow_core/vlm_as_detector@v2` with `model_type="spacexai"` to convert the
output into predictions. Confidence scores are optional; when absent the
parser assigns `1.0`.

Images for object detection are downscaled so that their longest edge does not
exceed {DETECTION_MAX_EDGE_PIXELS}px and are sent as lossless PNG with
`detail: "high"`.

### API Key Options

1. **Roboflow Managed API Key (Default)** - Use `rf_key:account` to proxy
   requests through Roboflow's API. Usage is billed against Roboflow credits.
2. **Custom xAI API Key** - Provide your own xAI API key and pay xAI directly.
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
            "name": "SpaceXAI",
            "version": "v1",
            "short_description": "Run SpaceXAI Grok models with vision capabilities.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Grok", "xAI", "SpaceXAI"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-rocket",
                "blockPriority": 5.1,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/spacexai@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    task_type: TaskType = Field(
        default="unconstrained",
        description=(
            "Task type to be performed by model. Value determines required "
            "parameters and output response."
        ),
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
        description="Text prompt to the Grok model",
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
        description=(
            "Your xAI API key or 'rf_key:account' to use Roboflow's managed API key"
        ),
        examples=["rf_key:account", "xxx-xxx", "$inputs.xai_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal[tuple(MODEL_VERSION_IDS)],
    ] = Field(
        default="grok-4.6",
        description="Model to be used",
        examples=["grok-4.6", "grok-4.5", "$inputs.grok_model"],
        json_schema_extra={
            "values_metadata": MODEL_VERSION_METADATA,
        },
    )
    reasoning_effort: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            Literal["low", "high"],
        ]
    ] = Field(
        default=None,
        description=(
            "Optional reasoning effort passed to xAI as "
            '`reasoning: {"effort": ...}`. For requests with a direct xAI key, '
            "the request is retried without reasoning when the model rejects "
            "the parameter."
        ),
        examples=["low", "high"],
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of tokens the model can generate in its response. "
            "If not specified, the model will use its default limit. Minimum value is 16."
        ),
        ge=16,
    )
    temperature: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description=(
            "Temperature to sample from the model - value in range 0.0-2.0, the "
            'higher - the more random / "creative" the generations are.'
        ),
        ge=0.0,
        le=2.0,
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description=(
            "Number of concurrent requests that can be executed by block when "
            "batch of input images provided. If not given - block defaults to "
            "value configured globally in Workflows Execution Engine. Please "
            "restrict if you hit xAI limits."
        ),
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
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [third_party_model(provider="xai", model_id=self.model_version)]


class SpaceXAIBlockV1(WorkflowBlock):

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
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
        api_key: str = "rf_key:account",
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_outputs = run_spacexai_prompting(
            roboflow_api_key=self._api_key,
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            xai_api_key=api_key,
            model_version=model_version,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]


def run_spacexai_prompting(
    roboflow_api_key: Optional[str],
    images: List[Dict[str, Any]],
    task_type: TaskType,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    xai_api_key: str,
    model_version: str,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
    max_concurrent_requests: Optional[int],
) -> List[str]:
    """Encode images, build per-task prompts and execute xAI requests.

    Args:
        roboflow_api_key: Roboflow API key for proxied execution.
        images: Input images in loadable form.
        task_type: Task determining preprocessing and prompt construction.
        prompt: Free-form text prompt for tasks that accept one.
        output_structure: Field descriptions for structured answering.
        classes: Class names for classification and detection tasks.
        xai_api_key: xAI API key or Roboflow-proxied ``rf_key:`` key.
        model_version: Grok model identifier.
        reasoning_effort: Optional reasoning effort.
        max_tokens: Maximum number of output tokens.
        temperature: Sampling temperature.
        max_concurrent_requests: Cap on concurrent xAI requests.

    Returns:
        Raw text outputs, one per input image.

    Raises:
        ValueError: If the task type has no registered prompt builder.
    """
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    spacexai_prompts = []
    for image in images:
        loaded_image, _ = load_image(image)
        base64_image, image_width, image_height = encode_image_for_task(
            loaded_image, task_type=task_type
        )
        generated_prompt = PROMPT_BUILDERS[task_type](
            base64_image=base64_image,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            image_width=image_width,
            image_height=image_height,
        )
        spacexai_prompts.append(generated_prompt)
    return execute_spacexai_requests(
        roboflow_api_key=roboflow_api_key,
        xai_api_key=xai_api_key,
        spacexai_prompts=spacexai_prompts,
        model_version=model_version,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent_requests=max_concurrent_requests,
    )


def encode_image_for_task(
    image: np.ndarray, *, task_type: TaskType
) -> Tuple[str, int, int]:
    """Encode an image as base64 using task-appropriate preprocessing.

    The ``object-detection`` task downscales so the longest edge does not
    exceed ``DETECTION_MAX_EDGE_PIXELS`` and encodes as lossless PNG. All
    other tasks send the image unchanged as JPEG.

    Args:
        image: BGR image to be encoded.
        task_type: Task type determining the preprocessing applied.

    Returns:
        Tuple of the base64-encoded image payload (without a data URL prefix)
        and the ``(width, height)`` of the encoded image.
    """
    if task_type == "object-detection":
        encoded_image = _downscale_image_to_max_edge(
            image, max_edge=DETECTION_MAX_EDGE_PIXELS
        )
        image_bytes = _encode_image_to_png_bytes(encoded_image)
    else:
        encoded_image = image
        image_bytes = encode_image_to_jpeg_bytes(encoded_image)

    base64_image = base64.b64encode(image_bytes).decode("ascii")
    encoded_height, encoded_width = encoded_image.shape[:2]

    return base64_image, encoded_width, encoded_height


def _downscale_image_to_max_edge(image: np.ndarray, *, max_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    target_width, target_height = scale_dimensions_to_max_edge(width, height, max_edge)
    if (target_width, target_height) == (width, height):
        return image

    resized_image = cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )

    return resized_image


def _encode_image_to_png_bytes(image: np.ndarray) -> bytes:
    _, encoded_image = cv2.imencode(".png", image)
    return encoded_image.tobytes()


def execute_spacexai_requests(
    roboflow_api_key: Optional[str],
    xai_api_key: str,
    spacexai_prompts: List[dict],
    model_version: str,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
    max_concurrent_requests: Optional[int],
) -> List[str]:
    """Execute prepared xAI request payloads in parallel.

    Args:
        roboflow_api_key: Roboflow API key for proxied execution.
        xai_api_key: xAI API key or Roboflow-proxied ``rf_key:`` key.
        spacexai_prompts: Prompt payloads with ``input`` and optionally
            ``instructions`` keys.
        model_version: Grok model identifier.
        reasoning_effort: Optional reasoning effort.
        max_tokens: Maximum number of output tokens.
        temperature: Sampling temperature.
        max_concurrent_requests: Cap on concurrent requests.

    Returns:
        Raw text outputs in the order of the input prompts.
    """
    tasks = [
        partial(
            execute_spacexai_request,
            roboflow_api_key=roboflow_api_key,
            xai_api_key=xai_api_key,
            instructions=prompt.get("instructions"),
            input_content=prompt["input"],
            model_version=model_version,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for prompt in spacexai_prompts
    ]
    max_workers = (
        max_concurrent_requests
        or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    )
    return run_in_parallel(
        tasks=tasks,
        max_workers=max_workers,
    )


def execute_spacexai_request(
    roboflow_api_key: Optional[str],
    xai_api_key: str,
    instructions: Optional[str],
    input_content: List[dict],
    model_version: str,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> str:
    """Execute a single xAI request, routing to direct or proxied mode.

    Args:
        roboflow_api_key: Roboflow API key, required for proxied execution.
        xai_api_key: xAI API key or Roboflow-proxied ``rf_key:`` key.
        instructions: Optional system instructions.
        input_content: ``input`` entries of the Responses API payload.
        model_version: Grok model identifier.
        reasoning_effort: Optional reasoning effort.
        max_tokens: Maximum number of output tokens.
        temperature: Sampling temperature.

    Returns:
        Raw text output of the model.
    """
    if xai_api_key.startswith(("rf_key:account", "rf_key:user:")):
        if not roboflow_api_key:
            raise ValueError(
                "Roboflow API key is required when using a Roboflow-managed xAI API key."
            )

        return _execute_proxied_spacexai_request(
            roboflow_api_key=roboflow_api_key,
            xai_api_key=xai_api_key,
            instructions=instructions,
            input_content=input_content,
            model_version=model_version,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return _execute_direct_spacexai_request(
        xai_api_key=xai_api_key,
        instructions=instructions,
        input_content=input_content,
        model_version=model_version,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _execute_proxied_spacexai_request(
    roboflow_api_key: str,
    xai_api_key: str,
    instructions: Optional[str],
    input_content: List[dict],
    model_version: str,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> str:
    """Execute xAI request via Roboflow proxy."""
    payload = {
        "model": model_version,
        "input": input_content,
        "xai_api_key": xai_api_key,
        "store": False,
    }

    if instructions is not None:
        payload["instructions"] = instructions

    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens

    if temperature is not None:
        payload["temperature"] = temperature

    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}

    try:
        response_data = post_to_roboflow_api(
            endpoint="apiproxy/xai",
            api_key=roboflow_api_key,
            payload=payload,
        )
        return _extract_output_text(response_data)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to Roboflow proxy: {e}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Invalid response structure from Roboflow proxy: {e}"
        ) from e


def _is_unsupported_reasoning_error(error: Exception) -> bool:
    message = str(error).lower()
    return "reasoning" in message and (
        "unsupported" in message
        or "not supported" in message
        or "unknown" in message
        or "invalid" in message
    )


def _execute_direct_spacexai_request(
    xai_api_key: str,
    instructions: Optional[str],
    input_content: List[dict],
    model_version: str,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> str:
    """Execute xAI request directly against api.x.ai."""
    client = OpenAI(base_url=XAI_BASE_URL, api_key=xai_api_key)

    request_params: Dict[str, Any] = {
        "model": model_version,
        "input": input_content,
        # xAI rejects responses exceeding its server-side storage limit;
        # we never retrieve stored responses, so disable storage entirely.
        "store": False,
    }

    if instructions is not None:
        request_params["instructions"] = instructions

    if max_tokens is not None:
        request_params["max_output_tokens"] = max_tokens

    if temperature is not None:
        request_params["temperature"] = temperature

    if reasoning_effort is not None:
        request_params["reasoning"] = {"effort": reasoning_effort}

    try:
        response = client.responses.create(**request_params)
    except Exception as error:
        if reasoning_effort is None or not _is_unsupported_reasoning_error(error):
            raise
        request_params.pop("reasoning", None)
        response = client.responses.create(**request_params)

    status = response.status
    if status == "failed":
        error_message = "Unknown error"
        if response.error:
            error_message = f"{response.error.code}: {response.error.message}"
        raise ValueError(f"xAI API request failed: {error_message}")

    if status == "cancelled":
        raise ValueError("xAI API request was cancelled.")

    if status == "incomplete":
        reason = "Unknown reason"
        if response.incomplete_details:
            reason = response.incomplete_details.reason
        if reason == "max_output_tokens":
            raise ValueError(
                "xAI API stopped generation because the max_tokens limit was reached. "
                "Please increase the max_tokens parameter to allow for a complete response."
            )
        raise ValueError(f"xAI API returned an incomplete response. Reason: {reason}")

    if status not in ["completed", "in_progress", "queued"]:
        raise ValueError(f"xAI API returned unexpected status: {status}")

    output_text = response.output_text
    if not output_text:
        raise ValueError("xAI API returned no text content in response.")

    return output_text


def _extract_output_text(response_data: dict) -> str:
    """Extract output text from xAI / OpenAI Responses API response."""
    status = response_data.get("status")

    if status == "failed":
        error = response_data.get("error", {})
        error_message = (
            f"{error.get('code', 'Unknown')}: {error.get('message', 'Unknown error')}"
        )
        raise ValueError(f"xAI API request failed: {error_message}")

    if status == "cancelled":
        raise ValueError("xAI API request was cancelled.")

    if status == "incomplete":
        incomplete_details = response_data.get("incomplete_details", {})
        reason = incomplete_details.get("reason", "Unknown reason")
        if reason == "max_output_tokens":
            raise ValueError(
                "xAI API stopped generation because the max_tokens limit was reached. "
                "Please increase the max_tokens parameter to allow for a complete response."
            )
        raise ValueError(f"xAI API returned an incomplete response. Reason: {reason}")

    if status not in ["completed", "in_progress", "queued", None]:
        raise ValueError(f"xAI API returned unexpected status: {status}")

    output_items = response_data.get("output", [])
    texts = []
    for item in output_items:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))

    output_text = "".join(texts)
    if not output_text:
        raise ValueError("xAI API returned no text content in response.")

    return output_text


def _image_content(base64_image: str, *, media_type: str, detail: str = "auto") -> dict:
    return {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,{base64_image}",
        "detail": detail,
    }


def prepare_unconstrained_prompt(
    base64_image: str,
    prompt: str,
    **kwargs,
) -> dict:
    """Build a request forwarding the user's prompt without instructions."""
    return {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    _image_content(base64_image, media_type="image/jpeg"),
                ],
            }
        ],
    }


def prepare_classification_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> dict:
    """Build a single-label classification request."""
    serialised_classes = ", ".join(classes)
    return {
        "instructions": (
            "You act as single-class classification model. You must provide reasonable predictions. "
            "You are only allowed to produce JSON document in Markdown ```json [...]``` markers. "
            'Expected structure of json: {"class_name": "class-name", "confidence": 0.4}. '
            "`class-name` must be one of the class names defined by user. You are only allowed to return "
            "single JSON document, even if there are potentially multiple classes. You are not allowed to return list."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"List of all classes to be recognised by model: {serialised_classes}",
                    },
                    _image_content(base64_image, media_type="image/jpeg"),
                ],
            }
        ],
    }


def prepare_multi_label_classification_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> dict:
    """Build a multi-label classification request."""
    serialised_classes = ", ".join(classes)
    return {
        "instructions": (
            "You act as multi-label classification model. You must provide reasonable predictions. "
            "You are only allowed to produce JSON document in Markdown ```json``` markers. "
            'Expected structure of json: {"predicted_classes": [{"class": "class-name-1", "confidence": 0.9}, '
            '{"class": "class-name-2", "confidence": 0.7}]}. '
            "`class-name-X` must be one of the class names defined by user and `confidence` is a float value in range "
            "0.0-1.0 that represent how sure you are that the class is present in the image. Only return class names "
            "that are visible."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"List of all classes to be recognised by model: {serialised_classes}",
                    },
                    _image_content(base64_image, media_type="image/jpeg"),
                ],
            }
        ],
    }


def prepare_vqa_prompt(
    base64_image: str,
    prompt: str,
    **kwargs,
) -> dict:
    """Build a visual-question-answering request."""
    return {
        "instructions": (
            "You act as Visual Question Answering model. Your task is to provide answer to question "
            "submitted by user. If this is open-question - answer with few sentences, for ABCD question, "
            "return only the indicator of the answer."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Question: {prompt}"},
                    _image_content(base64_image, media_type="image/jpeg"),
                ],
            }
        ],
    }


def prepare_ocr_prompt(
    base64_image: str,
    **kwargs,
) -> dict:
    """Build an OCR request returning recognised text as paragraphs."""
    return {
        "instructions": (
            "You act as OCR model. Your task is to read text from the image and return it in "
            "paragraphs representing the structure of texts in the image. You should only return "
            "recognised text, nothing else."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    _image_content(base64_image, media_type="image/jpeg"),
                ],
            }
        ],
    }


def prepare_caption_prompt(
    base64_image: str,
    short_description: bool,
    **kwargs,
) -> dict:
    """Build an image captioning request."""
    caption_detail_level = "Caption should be short."
    if not short_description:
        caption_detail_level = "Caption should be extensive."
    return {
        "instructions": (
            f"You act as image caption model. Your task is to provide description of the image. "
            f"{caption_detail_level}"
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    _image_content(base64_image, media_type="image/jpeg"),
                ],
            }
        ],
    }


def prepare_structured_answering_prompt(
    base64_image: str,
    output_structure: Dict[str, str],
    **kwargs,
) -> dict:
    """Build a structured-answering request producing user-defined JSON."""
    output_structure_serialised = json.dumps(output_structure, indent=4)
    return {
        "instructions": (
            "You are supposed to produce responses in JSON wrapped in Markdown markers: "
            "```json\nyour-response\n```. User is to provide you dictionary with keys and values. "
            "Each key must be present in your response. Values in user dictionary represent "
            "descriptions for JSON fields to be generated. Provide only JSON Markdown in response."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Specification of requirements regarding output fields: \n"
                        f"{output_structure_serialised}",
                    },
                    _image_content(base64_image, media_type="image/jpeg"),
                ],
            }
        ],
    }


def prepare_object_detection_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> dict:
    """Build the percent-format detection request used by Grok 4.5/4.6.

    Args:
        base64_image: Base64-encoded PNG image.
        classes: Class names the model may predict.
        **kwargs: Ignored builder arguments shared across task types.

    Returns:
        Request payload with an ``input`` key containing the detection prompt
        and a high-detail PNG image.
    """
    class_list = ", ".join(classes)
    prompt_text = OBJECT_DETECTION_PROMPT_TEMPLATE.format(class_list=class_list)
    return {
        "input": [
            {
                "role": "user",
                "content": [
                    _image_content(base64_image, media_type="image/png", detail="high"),
                    {"type": "input_text", "text": prompt_text},
                ],
            }
        ],
    }


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

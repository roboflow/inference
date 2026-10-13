"""Ask Anthropic Claude about each image.

Port of V1 ``roboflow_core/anthropic_claude@v5`` and its
``anthropic_claude/model_capabilities.py``. Claude reads the image first, then
the task text::

    per image: task ──▶ (system?, user message: image, text?)
        object-detection: resized to the size Claude reads (anthropic_upload_size),
                          PNG (JPEG 95 above 2.5 MB), no system prompt,
                          xyxy_absolute prompt with the upload size
        other tasks:      longest edge ≤ max_image_size, JPEG 90
    api_key rf_key:* ─▶ platform_client.post("apiproxy/anthropic", payload)
    api_key literal  ─▶ Anthropic(base_url=anthropic_base_url,
                                  http_client=anthropic_http_client)
                        .messages.stream(...).get_final_message()
    answer ──▶ decode_vlm_output(xyxy_absolute, upload size) ──▶ predictions

The direct call streams, as V1 does: the SDK refuses a non-streaming request
whose ``max_tokens`` could take longer than ten minutes, which includes every
model's default.

Since Claude Opus 4.7, Anthropic rejects ``temperature`` and manual extended
thinking. ``MODELS_WITH_LEGACY_CONTROLS`` names the models that still accept
them; other models, including unknown ids, drop ``temperature`` and get
``{"type": "adaptive"}`` thinking.

Differences from V1:

* A failed request raises ``ChatRequestError`` with the keys scrubbed. V1
  raised the vendor's exception or ``ValueError``.
* ``max_concurrent_requests``, ``max_tokens`` and ``max_image_size`` must be
  at least 1; unset concurrency means 8 instead of an environment variable.
* A dropped ``temperature`` or thinking budget is logged once per step run,
  not once per process.
* Predictions are native V2 carriers; see ``decoding``.
"""

import base64
import json
import logging
import re
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import cv2
import httpx
import numpy as np
import requests
import torch
from anthropic import NOT_GIVEN, Anthropic, DefaultHttpxClient
from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)
from roboflow_workflows.v2.blocks.language.chat import (
    DEFAULT_MANAGED_KEY,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    ChatAnswer,
    ChatKeys,
    ChatRequestError,
    EncodedImage,
    check_task_fields,
    encode_image,
    reveal_keys,
    run_concurrently,
    scrubbed_call,
    third_party_model,
    token_count,
    token_outputs,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    anthropic_upload_size,
    build_object_detection_prompt,
    decode_vlm_output,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.blocks.platform import (
    OFFLINE_PLATFORM_CLIENT,
    PlatformClient,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

logger = logging.getLogger(__name__)

PROVIDER = "Anthropic"
ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_PROXY_ENDPOINT = "apiproxy/anthropic"
DETECTION_TASK = "object-detection"
DETECTION_BOX_FORMAT = "xyxy_absolute"
"""Claude answers boxes in pixels of the image as uploaded."""

DETECTION_MAX_PNG_PAYLOAD_BYTES = 2_500_000
"""Largest detection PNG sent; above it the same pixels go as JPEG (V1)."""

DETECTION_JPEG_FALLBACK_QUALITY = 95
DEFAULT_MAX_IMAGE_SIZE = 1024
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_OUTPUT_TOKENS = 64000
"""Output limit of a model missing from ``CLAUDE_MODELS``."""


@dataclass(frozen=True)
class ClaudeModel:
    """One entry of V1's Claude model table.

    Args:
        name: Display name.
        exact_version: Model id sent directly to Anthropic.
        max_output_tokens: The model's output limit; ``max_tokens`` default.
    """

    name: str
    exact_version: str
    max_output_tokens: int


CLAUDE_MODELS: Dict[str, ClaudeModel] = {
    "claude-opus-5-5": ClaudeModel("Claude Opus 5.5", "claude-opus-5-5", 128000),
    "claude-sonnet-5-5": ClaudeModel("Claude Sonnet 5.5", "claude-sonnet-5-5", 128000),
    "claude-fable-5-1": ClaudeModel("Claude Fable 5.1", "claude-fable-5-1", 128000),
    "claude-fable-5": ClaudeModel("Claude Fable 5", "claude-fable-5", 128000),
    "claude-opus-5": ClaudeModel("Claude Opus 5", "claude-opus-5", 128000),
    "claude-sonnet-5": ClaudeModel("Claude Sonnet 5", "claude-sonnet-5", 128000),
    "claude-opus-4-8": ClaudeModel("Claude Opus 4.8", "claude-opus-4-8", 128000),
    "claude-opus-4-7": ClaudeModel("Claude Opus 4.7", "claude-opus-4-7", 128000),
    "claude-opus-4-6": ClaudeModel("Claude Opus 4.6", "claude-opus-4-6", 128000),
    "claude-sonnet-4-6": ClaudeModel("Claude Sonnet 4.6", "claude-sonnet-4-6", 64000),
    "claude-sonnet-4-5": ClaudeModel(
        "Claude Sonnet 4.5", "claude-sonnet-4-5-20250929", 64000
    ),
    "claude-haiku-4-5": ClaudeModel(
        "Claude Haiku 4.5", "claude-haiku-4-5-20251001", 64000
    ),
    "claude-opus-4-5": ClaudeModel(
        "Claude Opus 4.5", "claude-opus-4-5-20251101", 64000
    ),
    "claude-sonnet-4": ClaudeModel(
        "Claude Sonnet 4", "claude-sonnet-4-20250514", 64000
    ),
    "claude-opus-4-1": ClaudeModel(
        "Claude Opus 4.1", "claude-opus-4-1-20250805", 32000
    ),
    "claude-opus-4": ClaudeModel("Claude Opus 4", "claude-opus-4-20250514", 32000),
}
"""V1 ``CLAUDE_MODELS``: block label -> model, in V1's display order."""

ClaudeModelId = Literal[tuple(CLAUDE_MODELS)]
ClaudeTemperature = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
ImageEdge = Annotated[StrictInt, Field(ge=1)]

MODELS_WITH_LEGACY_CONTROLS = frozenset(
    {
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-opus-4",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "claude-3-7-sonnet",
        "claude-3-5-haiku",
        "claude-3-opus",
        "claude-3-haiku",
        "claude-3-5-sonnet",
        "claude-3-5-sonnet-v2",
        "claude-4-5-sonnet",
        "claude-4-5-sonnet-v2",
    }
)
"""Normalized ids that still accept ``temperature`` and a thinking budget (V1)."""

_DATED_SNAPSHOT_SUFFIX = re.compile(r"-\d{8}$")
_LATEST_SUFFIX = re.compile(r"-latest$")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")

CLAUDE_TASKS = (
    "unconstrained",
    "ocr",
    "structured-answering",
    "classification",
    "multi-label-classification",
    "visual-question-answering",
    "caption",
    "detailed-caption",
    DETECTION_TASK,
)
ClaudeTask = Literal[CLAUDE_TASKS]

CLASSIFICATION_SYSTEM_PROMPT = (
    "You act as single-class classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document. Expected structure "
    'of json: {"class_name": "class-name", "confidence": 0.4}. `class-name` must be '
    "one of the class names defined by user. You are only allowed to return single "
    "JSON document, even if there are potentially multiple classes. You are not "
    "allowed to return list."
)
MULTI_LABEL_SYSTEM_PROMPT = (
    "You act as multi-label classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document. Expected structure "
    'of json: {"predicted_classes": [{"class": "class-name-1", "confidence": 0.9}, '
    '{"class": "class-name-2", "confidence": 0.7}]}.`class-name-X` must be one of '
    "the class names defined by user and `confidence` is a float value in range "
    "0.0-1.0 that represents how sure you are that the class is present in the "
    "image. Only return class names that are visible."
)
# V1 joins "question" and "submitted" without a space; kept byte for byte.
VQA_SYSTEM_PROMPT = (
    "You act as Visual Question Answering model. Your task is to provide answer to "
    "questionsubmitted by user. If this is open-question - answer with few "
    "sentences, for ABCD question, return only the indicator of the answer."
)
OCR_SYSTEM_PROMPT = (
    "You act as OCR model. Your task is to read text from the image and return it in "
    "paragraphs representing the structure of texts in the image. You should only "
    "return recognised text, nothing else."
)
CAPTION_SYSTEM_PROMPT = (
    "You act as image caption model. Your task is to provide description of the "
    "image. Caption should be {length}."
)
STRUCTURED_ANSWERING_SYSTEM_PROMPT = (
    "You are supposed to produce responses in JSON. User is to provide you "
    "dictionary with keys and values. Each key must be present in your response. "
    "Values in user dictionary represent descriptions for JSON fields to be "
    "generated. Provide only JSON in response."
)
CLASS_LIST_TEXT = "List of all classes to be recognised by model: {class_list}"
OUTPUT_STRUCTURE_TEXT = "Specification of requirements regarding output fields: \n"


@dataclass(frozen=True)
class ClaudeRequest:
    """One image's request, before a key or transport is chosen.

    Args:
        system: System prompt, or ``None``.
        messages: Anthropic ``messages``: one user turn, image first.
        upload_size_wh: Uploaded ``(width, height)``.
    """

    system: Optional[str]
    messages: List[Dict[str, Any]]
    upload_size_wh: Tuple[int, int]


@dataclass(frozen=True)
class ClaudeOptions:
    """Request settings shared by every image of one step run.

    Args:
        model_version: Block label; the proxy receives it as is.
        exact_version: Model id sent directly to Anthropic.
        max_tokens: Output token limit.
        temperature: Sampling temperature, or ``None`` to omit it.
        thinking: Anthropic ``thinking`` object, or ``None``.
    """

    model_version: str
    exact_version: str
    max_tokens: int
    temperature: Optional[float]
    thinking: Optional[Dict[str, Any]]


def normalize_model_id(model_version: str) -> str:
    """Slugify a model id and strip ``-latest`` and dated snapshot suffixes.

    Args:
        model_version: Block label or wire id, e.g. ``claude-sonnet-4-5-20250929``.

    Returns:
        The normalized id, e.g. ``claude-sonnet-4-5``.
    """
    slug = _NON_ALPHANUMERIC.sub("-", model_version.strip().lower()).strip("-")
    slug = _LATEST_SUFFIX.sub("", slug)
    normalized = _DATED_SNAPSHOT_SUFFIX.sub("", slug)

    return normalized


def has_legacy_controls(model_version: str) -> bool:
    """Whether a model still accepts ``temperature`` and a thinking budget.

    Args:
        model_version: Block label or wire id.

    Returns:
        ``False`` for Claude Opus 4.7 and newer and for unknown ids.
    """
    legacy = normalize_model_id(model_version) in MODELS_WITH_LEGACY_CONTROLS

    return legacy


def claude_options(
    model_version: str,
    *,
    max_tokens: Optional[int],
    temperature: Optional[float],
    extended_thinking: Optional[bool],
    thinking_budget_tokens: Optional[int],
) -> ClaudeOptions:
    """Resolve the request settings of a model, as V1 and the Roboflow proxy do.

    ``temperature`` is dropped with thinking on and for models without legacy
    controls. Thinking is ``{"type": "enabled", "budget_tokens": N}`` (``N``
    defaults to half the output limit) on legacy models, ``{"type":
    "adaptive"}`` otherwise. Dropping a configured value logs a warning.

    Args:
        model_version: Block label.
        max_tokens: Configured output limit, or ``None`` for the model's.
        temperature: Configured temperature, or ``None``.
        extended_thinking: Whether thinking was requested.
        thinking_budget_tokens: Configured budget, or ``None``.

    Returns:
        The settings of every request of the run.
    """
    model = CLAUDE_MODELS.get(model_version)
    model_max_output = model.max_output_tokens if model else DEFAULT_MAX_OUTPUT_TOKENS
    legacy = has_legacy_controls(model_version)

    resolved_temperature = None if extended_thinking else temperature
    if resolved_temperature is not None and not legacy:
        resolved_temperature = None
        logger.warning(
            "Anthropic model `%s` does not accept `temperature` (Anthropic dropped "
            "sampling controls starting with Claude Opus 4.7). Ignoring "
            "temperature=%s; remove it from the step to silence this warning.",
            model_version,
            temperature,
        )

    thinking = None
    if extended_thinking and legacy:
        budget = thinking_budget_tokens
        if budget is None:
            budget = model_max_output // 2
        thinking = {"type": "enabled", "budget_tokens": budget}
    elif extended_thinking:
        thinking = {"type": "adaptive"}
        if thinking_budget_tokens is not None:
            logger.warning(
                "Anthropic model `%s` only supports adaptive thinking; ignoring "
                "thinking_budget_tokens=%s and requesting `thinking.type=adaptive`.",
                model_version,
                thinking_budget_tokens,
            )

    options = ClaudeOptions(
        model_version=model_version,
        exact_version=model.exact_version if model else model_version,
        max_tokens=max_tokens if max_tokens is not None else model_max_output,
        temperature=resolved_temperature,
        thinking=thinking,
    )

    return options


def downscaled_size_wh(
    size_wh: Tuple[int, int], *, max_image_size: int
) -> Tuple[int, int]:
    """Size V1 downscales a non-detection upload to.

    V1 ``downscale_image_keeping_aspect_ratio`` arithmetic: never upscales;
    the shorter side is truncated, not rounded.

    Args:
        size_wh: ``(width, height)`` of the image.
        max_image_size: Longest allowed edge.

    Returns:
        ``(width, height)`` of the upload.
    """
    width, height = size_wh
    if width <= max_image_size and height <= max_image_size:
        return size_wh

    ratio = width / height
    if ratio >= 1:
        scaled = (max_image_size, int(max_image_size / ratio))
    else:
        scaled = (int(max_image_size * ratio), max_image_size)

    return scaled


def claude_upload(
    image: ImageData, *, task_type: str, max_image_size: int
) -> EncodedImage:
    """Encode an image as V1 uploads it for the task.

    Detection: resized with Lanczos to ``anthropic_upload_size`` and sent as
    PNG, or as JPEG 95 at the same size when the PNG exceeds
    ``DETECTION_MAX_PNG_PAYLOAD_BYTES``. Other tasks: downscaled with OpenCV's
    default bilinear filter to ``max_image_size`` and sent as JPEG 90.

    Args:
        image: The image to upload.
        task_type: The block's task.
        max_image_size: Longest edge of a non-detection upload.

    Returns:
        The encoded upload with its pixel size.

    Raises:
        ValueError: When OpenCV cannot encode the image.
    """
    if task_type == DETECTION_TASK:
        size_wh = anthropic_upload_size((image.width, image.height))
        upload = encode_image(image, image_format="png", size_wh=size_wh)
        png = base64.b64decode(upload.base64)
        if len(png) > DETECTION_MAX_PNG_PAYLOAD_BYTES:
            upload = _lossless_png_as_jpeg(png, size_wh=upload.size_wh)
        return upload

    size_wh = downscaled_size_wh(
        (image.width, image.height), max_image_size=max_image_size
    )
    if size_wh != (image.width, image.height):
        image = _bilinear_resized(image, size_wh=size_wh)
    upload = encode_image(image)

    return upload


def claude_request(
    task_type: str,
    *,
    upload: EncodedImage,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> ClaudeRequest:
    """Build V1's system prompt and user message of one task.

    Args:
        task_type: One of ``CLAUDE_TASKS``.
        upload: The encoded image.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        The request.

    Raises:
        ValueError: On an unknown task or invalid detection classes.
    """
    class_text = CLASS_LIST_TEXT.format(class_list=", ".join(classes or []))
    prompts = {
        "unconstrained": lambda: (None, prompt),
        "ocr": lambda: (OCR_SYSTEM_PROMPT, None),
        "visual-question-answering": lambda: (VQA_SYSTEM_PROMPT, f"Question: {prompt}"),
        "caption": lambda: (CAPTION_SYSTEM_PROMPT.format(length="short"), None),
        "detailed-caption": lambda: (
            CAPTION_SYSTEM_PROMPT.format(length="extensive"),
            None,
        ),
        "classification": lambda: (CLASSIFICATION_SYSTEM_PROMPT, class_text),
        "multi-label-classification": lambda: (MULTI_LABEL_SYSTEM_PROMPT, class_text),
        "structured-answering": lambda: (
            STRUCTURED_ANSWERING_SYSTEM_PROMPT,
            OUTPUT_STRUCTURE_TEXT + json.dumps(output_structure, indent=4),
        ),
        DETECTION_TASK: lambda: (
            None,
            build_object_detection_prompt(
                DETECTION_BOX_FORMAT, classes=classes, upload_size_wh=upload.size_wh
            ),
        ),
    }
    if task_type not in prompts:
        raise ValueError(f"Task type: {task_type} not supported.")

    system, text = prompts[task_type]()
    content: List[Dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": upload.media_type,
                "data": upload.base64,
            },
        }
    ]
    if text is not None:
        content.append({"type": "text", "text": text})
    request = ClaudeRequest(
        system=system,
        messages=[{"role": "user", "content": content}],
        upload_size_wh=upload.size_wh,
    )

    return request


def read_claude_message(data: Dict[str, Any], *, via_proxy: bool) -> ChatAnswer:
    """Read a Claude message with V1's failure messages; thinking is ignored.

    Args:
        data: The message as a JSON dictionary.
        via_proxy: Whether it came through the Roboflow proxy, which may omit
            ``stop_reason``.

    Returns:
        The first text block and token usage.

    Raises:
        ChatRequestError: When generation hit ``max_tokens``, stopped for
            another reason, or returned no text.
    """
    stop_reason = data.get("stop_reason")
    if stop_reason == "max_tokens":
        raise ChatRequestError(
            "Claude API stopped generation because the max_tokens limit was reached. "
            "Please increase the max_tokens parameter to allow for a complete response."
        )
    accepted = ("end_turn", "stop_sequence")
    if stop_reason not in accepted and not (via_proxy and stop_reason is None):
        raise ChatRequestError(
            f"Claude API stopped generation with unexpected stop reason: {stop_reason}."
        )

    content = data.get("content") or []
    if via_proxy and not content:
        raise ChatRequestError("Claude API returned no content in response.")
    texts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if not texts:
        raise ChatRequestError("Claude API returned no text content in response.")

    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    answer = ChatAnswer(
        text=texts[0],
        input_tokens=token_count(usage.get("input_tokens")),
        output_tokens=token_count(usage.get("output_tokens")),
    )

    return answer


class AnthropicClaudeBlock(Block):
    """Ask an Anthropic Claude model about each image.

    Example: ``task_type="object-detection"`` with ``classes=["car"]`` sends
    each image at the size Claude reads it and returns the answered pixel
    boxes as detections on the original image. ``api_key="rf_key:account"``
    (the default) bills the host's Roboflow account through its platform
    client; any other key calls Anthropic directly.
    """

    type = "v2/anthropic_claude@v5"
    metadata = {
        "name": "Anthropic Claude",
        "port_of": "roboflow_core/anthropic_claude@v5",
        "section": "model",
        "task_metadata": {task: TASK_METADATA[task] for task in CLAUDE_TASKS},
        "model_version_metadata": {
            model_id: {"name": model.name} for model_id, model in CLAUDE_MODELS.items()
        },
        "recommended_parsers": {"structured-answering": "v2/json_parser@v1"},
    }
    outputs = {
        "output": Output(
            STRING_KIND,
            LANGUAGE_MODEL_OUTPUT_KIND,
            source="images",
            description="Claude's raw answer to this image.",
        ),
        "classes": Output(
            LIST_OF_VALUES_KIND, source="images", description="The step's classes."
        ),
        **token_outputs(source="images"),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to ask about; one request per image."
        )
        task_type: ClaudeTask = Field(
            default="unconstrained",
            description="What to ask: a free prompt, OCR, captions, classification, structured answers or detection.",
            examples=["object-detection"],
        )
        prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description="Prompt for 'unconstrained' and question for 'visual-question-answering'.",
            examples=["my prompt", "$inputs.prompt"],
        )
        output_structure: Optional[Dict[str, StrictStr]] = Field(
            default=None,
            description="For 'structured-answering': answer field names and their descriptions.",
            examples=[{"my_key": "description"}],
        )
        classes: Optional[List[StrictStr]] | Ref(
            LIST_OF_VALUES_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Class names for classification and detection.",
            examples=[["class-a", "class-b"], "$inputs.classes"],
        )
        api_key: ApiKey = Field(
            default=DEFAULT_MANAGED_KEY,
            description=(
                "Anthropic key as a secret or string, or 'rf_key:account' / "
                "'rf_key:user:ID' to use a key stored with Roboflow."
            ),
            examples=["rf_key:account", "$inputs.anthropic_api_key"],
        )
        model_version: ClaudeModelId | Ref(STRING_KIND) = Field(
            default=DEFAULT_MODEL,
            description="Claude model label; see the block's model_version_metadata.",
            examples=["claude-sonnet-4-5", "$inputs.claude_model"],
        )
        extended_thinking: Optional[StrictBool] = Field(
            default=None,
            description=(
                "Let the model think before answering. Excludes temperature. "
                "Claude Opus 4.7 and newer think adaptively and ignore the budget."
            ),
            examples=[True],
        )
        thinking_budget_tokens: Optional[StrictInt] = Field(
            default=None,
            ge=1024,
            description=(
                "Thinking token budget; below max_tokens. None means half the "
                "model's output limit."
            ),
            examples=[4096],
        )
        max_tokens: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most output tokens; None means the model's output limit.",
            examples=[2048],
        )
        temperature: Optional[ClaudeTemperature] | Ref(FLOAT_KIND, allow_none=True) = (
            Field(
                default=None,
                description=(
                    "Sampling temperature. Not with extended thinking; ignored by "
                    "Claude Opus 4.7 and newer."
                ),
                examples=[0.2],
            )
        )
        max_image_size: ImageEdge | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_MAX_IMAGE_SIZE,
            description=(
                "Longest uploaded edge for every task but object-detection, "
                "which uploads at the size Claude reads."
            ),
            examples=[1024],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most requests in flight for one batch; None means 8.",
            examples=[4],
        )

        @model_validator(mode="after")
        def _fields_consistent(self) -> "AnthropicClaudeBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )
            if self.extended_thinking:
                if self.temperature is not None:
                    raise ValueError(
                        "`temperature` cannot be used when `extended_thinking` is enabled"
                    )
                budget, limit = self.thinking_budget_tokens, self.max_tokens
                if budget and limit and budget >= limit:
                    raise ValueError(
                        f"`thinking_budget_tokens` ({budget}) must be less than "
                        f"`max_tokens` ({limit})"
                    )

            return self

    def __init__(
        self,
        *,
        platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT,
        api_key: Any = None,
        anthropic_base_url: str = ANTHROPIC_API_URL,
        anthropic_http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._platform_client = platform_client
        self._roboflow_api_key = api_key
        self._base_url = anthropic_base_url
        self._http_client = anthropic_http_client or DefaultHttpxClient()

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Dict[str, Output]:
        """Narrow ``predictions`` to the kind the configured task decodes."""
        outputs = {
            **cls.outputs,
            **vlm_prediction_outputs(params.task_type, source="images"),
        }

        return outputs

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the Claude model the step calls, by its exact Anthropic id."""
        model = CLAUDE_MODELS.get(params.model_version)
        resources = [
            third_party_model(
                model.exact_version if model else params.model_version,
                provider="anthropic",
                resource="anthropic_http_client",
            )
        ]

        return resources

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[Any]:
        """Report remote inference and image encoding."""
        operations = list(WORK_OPERATIONS)

        return operations

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> List[Any]:
        """Declare that the step has no runtime restrictions."""
        return []

    def run(
        self,
        *,
        images: Batch,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Any,
        model_version: str,
        extended_thinking: Optional[bool],
        thinking_budget_tokens: Optional[int],
        max_tokens: Optional[int],
        temperature: Optional[float],
        max_image_size: int,
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Send one request per image and decode each answer.

        Every request is built before the first is sent.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: Anthropic key, secret or managed ``rf_key:`` reference.
            model_version: Claude model label.
            extended_thinking: Whether the model thinks first.
            thinking_budget_tokens: Thinking budget, or ``None``.
            max_tokens: Output token limit, or ``None`` for the model's.
            temperature: Sampling temperature, or ``None``.
            max_image_size: Longest edge of a non-detection upload.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes", "input_tokens", "output_tokens",
            "predictions", "error_status", "inference_id"}`` per image.

        Raises:
            ValueError: On a limit below 1, invalid detection classes or a
                managed key without the host's Roboflow ``api_key`` resource,
                before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        _check_at_least_one(
            max_concurrent_requests=max_concurrent_requests,
            max_image_size=max_image_size,
        )
        options = claude_options(
            model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        claude_requests = [
            claude_request(
                task_type,
                upload=claude_upload(
                    image, task_type=task_type, max_image_size=max_image_size
                ),
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
            for image in images
        ]

        keys = reveal_keys(api_key, roboflow_api_key=self._roboflow_api_key)
        if keys.managed and not keys.roboflow_key:
            raise ValueError(
                "Roboflow API key is required when using a Roboflow-managed Anthropic "
                "API key; the host must provide the 'api_key' resource."
            )

        calls = [
            partial(self._ask, keys, request, options) for request in claude_requests
        ]
        answers = run_concurrently(
            calls, max_concurrent_requests=max_concurrent_requests
        )

        results = []
        for image, request, answer in zip(images, claude_requests, answers):
            inference_id = uuid.uuid4().hex
            error_status, predictions = decode_vlm_output(
                task_type,
                answer.text,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=DETECTION_BOX_FORMAT,
                upload_size_wh=request.upload_size_wh,
            )
            results.append(
                {
                    "output": answer.text,
                    "classes": classes,
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    "predictions": predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )

        return results

    def _ask(
        self, keys: ChatKeys, request: ClaudeRequest, options: ClaudeOptions
    ) -> ChatAnswer:
        send = self._send_proxied if keys.managed else self._send_direct
        answer = scrubbed_call(
            partial(send, keys, request, options), keys=keys, provider=PROVIDER
        )

        return answer

    def _send_proxied(
        self, keys: ChatKeys, request: ClaudeRequest, options: ClaudeOptions
    ) -> ChatAnswer:
        payload: Dict[str, Any] = {
            "model": options.model_version,
            "anthropic_api_key": keys.provider_key,
            "messages": request.messages,
            "max_tokens": options.max_tokens,
        }
        if request.system is not None:
            payload["system"] = request.system
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.thinking is not None:
            payload["thinking"] = options.thinking
        try:
            data = self._platform_client.post(
                endpoint=ANTHROPIC_PROXY_ENDPOINT,
                api_key=keys.roboflow_key,
                payload=payload,
            )
        except requests.exceptions.RequestException as error:
            message = f"Failed to connect to Roboflow proxy: {error}"
        else:
            answer = read_claude_message(data, via_proxy=True)
            return answer

        raise ChatRequestError(message)

    def _send_direct(
        self, keys: ChatKeys, request: ClaudeRequest, options: ClaudeOptions
    ) -> ChatAnswer:
        # Without a key the SDK would read ANTHROPIC_API_KEY from the environment.
        if keys.provider_key is None:
            raise ChatRequestError("Anthropic API key is missing.")
        client = Anthropic(
            api_key=keys.provider_key,
            base_url=self._base_url,
            http_client=self._http_client,
        )
        arguments: Dict[str, Any] = {
            "system": request.system if request.system is not None else NOT_GIVEN,
            "messages": request.messages,
            "max_tokens": options.max_tokens,
            "model": options.exact_version,
            "temperature": (
                options.temperature if options.temperature is not None else NOT_GIVEN
            ),
        }
        if options.thinking is not None:
            arguments["thinking"] = options.thinking
        with client.messages.stream(**arguments) as stream:
            message = stream.get_final_message()
        answer = read_claude_message(message.to_dict(), via_proxy=False)

        return answer


def _check_at_least_one(**values: Optional[int]) -> None:
    for name, value in values.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")


def _bilinear_resized(image: ImageData, *, size_wh: Tuple[int, int]) -> ImageData:
    pixels = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    resized = cv2.resize(np.ascontiguousarray(pixels), size_wh)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    resized_image = ImageData.from_tensor(torch.from_numpy(resized).permute(2, 0, 1))

    return resized_image


def _lossless_png_as_jpeg(png: bytes, *, size_wh: Tuple[int, int]) -> EncodedImage:
    # PNG is lossless, so decoding it gives back exactly the resized pixels.
    pixels = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    encoded, data = cv2.imencode(
        ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, DETECTION_JPEG_FALLBACK_QUALITY]
    )
    if not encoded:
        raise ValueError(f"OpenCV could not encode an image of shape {pixels.shape}")

    upload = EncodedImage(
        base64=base64.b64encode(data.tobytes()).decode("ascii"),
        media_type="image/jpeg",
        size_wh=size_wh,
    )

    return upload


BLOCKS = (AnthropicClaudeBlock,)

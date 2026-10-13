"""Ask Google Gemini about each image.

Port of V1 ``roboflow_core/google_gemini@v6``. V1 posts JSON to Gemini's
``/v1beta`` REST API; no Google SDK is involved::

    per image: task ──▶ {systemInstruction?, contents: image (JPEG 90, full size)
                         + text, generationConfig, tools?}
    api_key rf_key:* ─▶ platform_client.post("apiproxy/gemini",
                                             {model, google_api_key, **request})
    api_key literal  ─▶ gemini_http_client.post(
                            "<gemini_base_url>/models/<model>:generateContent",
                            header x-goog-api-key)
    answer ──▶ decode_vlm_output(yxyx_0_1000) ──▶ predictions

``thinking_level`` is sent only to models with thinking levels, and
``temperature`` only to models without them. Detection asks for Gemini's
native ``box_2d`` boxes, ``[y_min, x_min, y_max, x_max]`` in 0-1000, under a
response schema whose labels are the classes.

Differences from V1:

* Direct requests go through an ``httpx`` client the host can inject
  (``gemini_http_client``, ``gemini_base_url``) instead of ``requests``. An
  HTTP error is raised before the body is read, so a non-JSON error body
  gives the status, not a JSON decoding error.
* A failed request raises ``ChatRequestError`` with the keys scrubbed.
* ``max_concurrent_requests`` and ``max_tokens`` must be at least 1; unset
  concurrency means 8 instead of an environment variable.
* Predictions are native V2 carriers; see ``decoding``.
"""

import json
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple

import httpx
import requests
from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
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
    build_object_detection_prompt,
    decode_vlm_output,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.blocks.platform import (
    OFFLINE_PLATFORM_CLIENT,
    PlatformClient,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
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

PROVIDER = "Gemini"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_PROXY_ENDPOINT = "apiproxy/gemini"
REQUEST_TIMEOUT_S = 120
"""V1's timeout of a direct request."""

DETECTION_TASK = "object-detection"
DETECTION_BOX_FORMAT = "yxyx_0_1000"
"""Gemini's native box format; resolution independent, so no upload size."""

DEFAULT_MODEL = "gemini-3.1-pro-preview"
THINKING_LEVELS = ("minimal", "low", "medium", "high")
JSON_MIME_TYPE = "application/json"
TEXT_MIME_TYPE = "text/plain"


@dataclass(frozen=True)
class GeminiModel:
    """One entry of V1's Gemini model table.

    Args:
        name: Display name.
        thinking_levels: Levels the model accepts; empty means it takes
            ``temperature`` instead.
        native_code_execution: Whether ``google_code_execution`` applies.
    """

    name: str
    thinking_levels: Tuple[str, ...]
    native_code_execution: bool


_NO_MINIMAL = ("low", "medium", "high")

GEMINI_MODELS: Dict[str, GeminiModel] = {
    "gemini-3.8-flash": GeminiModel("Gemini 3.8 Flash", _NO_MINIMAL, True),
    "gemini-3.7-flash": GeminiModel("Gemini 3.7 Flash", _NO_MINIMAL, True),
    "gemini-3.6-flash": GeminiModel("Gemini 3.6 Flash", THINKING_LEVELS, True),
    "gemini-3.5-flash": GeminiModel("Gemini 3.5 Flash", THINKING_LEVELS, True),
    "gemini-3.5-flash-lite": GeminiModel(
        "Gemini 3.5 Flash-Lite", THINKING_LEVELS, True
    ),
    "gemini-3.1-pro-preview": GeminiModel("Gemini 3.1 Pro", _NO_MINIMAL, True),
    "gemini-3.1-flash-lite": GeminiModel(
        "Gemini 3.1 Flash-Lite", THINKING_LEVELS, True
    ),
    "gemini-3-flash-preview": GeminiModel("Gemini 3 Flash", THINKING_LEVELS, True),
    "gemini-2.5-pro": GeminiModel("Gemini 2.5 Pro", (), False),
    "gemini-2.5-flash": GeminiModel("Gemini 2.5 Flash", (), False),
    "gemini-2.5-flash-lite": GeminiModel("Gemini 2.5 Flash-Lite", (), False),
}
"""V1 ``GEMINI_MODELS`` in V1's display order."""

MODEL_ALIASES = {
    "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-03-25": "gemini-2.5-pro",
}
"""Retired ids a literal ``model_version`` is renamed from (V1)."""

GeminiModelId = Literal[tuple(GEMINI_MODELS)]
ThinkingLevel = Literal[THINKING_LEVELS]
GeminiTemperature = Annotated[StrictFloat, Field(ge=0.0, le=2.0)]

GEMINI_TASKS = (
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
GeminiTask = Literal[GEMINI_TASKS]

DETECTION_TASK_METADATA = {
    "name": "Object Detection",
    "description": (
        "Model detects bounding boxes for the provided classes, returning a "
        "Gemini-native `box_2d` JSON list (y_min, x_min, y_max, x_max integers "
        "normalized to 0-1000), decoded into predictions by the block."
    ),
}
"""V1 replaces the shared detection task description with this one."""

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
    '{"class": "class-name-2", "confidence": 0.7}]}. `class-name-X` must be one of '
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
class GeminiPrompt:
    """What one task asks, before it becomes a Gemini request.

    Args:
        system: System instruction, or ``None``.
        text: User text after the image.
        response_mime_type: ``text/plain`` or ``application/json``.
    """

    system: Optional[str]
    text: str
    response_mime_type: str = TEXT_MIME_TYPE


def check_thinking_level(model: Any, level: Any) -> None:
    """Reject a thinking level the selected model does not support (V1 rule).

    Unset values, selectors and models outside ``GEMINI_MODELS`` pass.

    Args:
        model: Model id, a selector or ``None``.
        level: Thinking level, a selector or ``None``.

    Raises:
        ValueError: With V1's message when the model rejects the level.
    """
    if model is None or level is None or _is_selector(model) or _is_selector(level):
        return
    if model not in GEMINI_MODELS:
        return

    supported = list(GEMINI_MODELS[model].thinking_levels)
    if level in supported:
        return
    if supported:
        raise ValueError(
            f"Model {model} supports thinking_level values {supported}, got {level!r}."
        )
    raise ValueError(
        f"Model {model} does not support configurable thinking_level (got {level!r})."
    )


def generation_config(
    model_version: str,
    *,
    max_tokens: Optional[int],
    temperature: Optional[float],
    thinking_level: Optional[str],
    response_mime_type: str = TEXT_MIME_TYPE,
) -> Dict[str, Any]:
    """Build V1's ``generationConfig``.

    Args:
        model_version: Model id.
        max_tokens: Output token limit, or ``None`` for Gemini's.
        temperature: Sent only to models without thinking levels.
        thinking_level: Sent only to models with thinking levels.
        response_mime_type: ``text/plain`` or ``application/json``.

    Returns:
        The ``generationConfig`` object.

    Raises:
        ValueError: When the model rejects ``thinking_level``.
    """
    check_thinking_level(model_version, thinking_level)
    model = GEMINI_MODELS.get(model_version)
    thinks = model is not None and bool(model.thinking_levels)

    config: Dict[str, Any] = {
        "response_mime_type": response_mime_type,
        "candidate_count": 1,
    }
    if max_tokens is not None:
        config["max_output_tokens"] = max_tokens
    if thinking_level is not None and thinks:
        config["thinking_config"] = {"thinking_level": thinking_level}
    if temperature is not None and not thinks:
        config["temperature"] = temperature

    return config


def gemini_prompt(
    task_type: str,
    *,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> GeminiPrompt:
    """Return V1's system instruction and user text of one task.

    Args:
        task_type: One of ``GEMINI_TASKS``.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        The prompt.

    Raises:
        ValueError: On an unknown task or invalid detection classes.
    """
    class_text = CLASS_LIST_TEXT.format(class_list=", ".join(classes or []))
    prompts = {
        "unconstrained": lambda: GeminiPrompt(None, prompt),
        "ocr": lambda: GeminiPrompt(OCR_SYSTEM_PROMPT, "Read the text"),
        "visual-question-answering": lambda: GeminiPrompt(
            VQA_SYSTEM_PROMPT, f"Question: {prompt}"
        ),
        "caption": lambda: GeminiPrompt(
            CAPTION_SYSTEM_PROMPT.format(length="short"), "Caption the image"
        ),
        "detailed-caption": lambda: GeminiPrompt(
            CAPTION_SYSTEM_PROMPT.format(length="extensive"), "Caption the image"
        ),
        "classification": lambda: GeminiPrompt(
            CLASSIFICATION_SYSTEM_PROMPT, class_text, JSON_MIME_TYPE
        ),
        "multi-label-classification": lambda: GeminiPrompt(
            MULTI_LABEL_SYSTEM_PROMPT, class_text, JSON_MIME_TYPE
        ),
        "structured-answering": lambda: GeminiPrompt(
            STRUCTURED_ANSWERING_SYSTEM_PROMPT,
            OUTPUT_STRUCTURE_TEXT + json.dumps(output_structure, indent=4),
            JSON_MIME_TYPE,
        ),
        DETECTION_TASK: lambda: GeminiPrompt(
            None,
            build_object_detection_prompt(DETECTION_BOX_FORMAT, classes=classes),
            JSON_MIME_TYPE,
        ),
    }
    if task_type not in prompts:
        raise ValueError(f"Task type: {task_type} not supported.")

    task_prompt = prompts[task_type]()

    return task_prompt


def detection_response_schema(classes: Sequence[str]) -> Dict[str, Any]:
    """Response schema of a detection answer: ``box_2d`` in 0-1000 and a class.

    Args:
        classes: Class names, the allowed labels.

    Returns:
        Gemini's ``response_schema`` object (V1).
    """
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "box_2d": {
                    "type": "ARRAY",
                    "items": {"type": "INTEGER", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "label": {"type": "STRING", "enum": list(classes)},
            },
            "required": ["box_2d", "label"],
            "propertyOrdering": ["box_2d", "label"],
        },
    }

    return schema


def gemini_request(
    task_prompt: GeminiPrompt,
    *,
    upload: EncodedImage,
    generation: Dict[str, Any],
    response_schema: Optional[Dict[str, Any]] = None,
    code_execution: bool = False,
) -> Dict[str, Any]:
    """Assemble one ``generateContent`` request body.

    Args:
        task_prompt: The task's system instruction and text.
        upload: The encoded image; it precedes the text.
        generation: ``generationConfig`` without a response schema.
        response_schema: Schema added to ``generationConfig``, or ``None``.
        code_execution: Whether to offer Gemini's code execution tool.

    Returns:
        The request body.
    """
    request: Dict[str, Any] = {}
    if task_prompt.system is not None:
        request["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": task_prompt.system}],
        }
    request["contents"] = {
        "parts": [
            {"inline_data": {"mime_type": upload.media_type, "data": upload.base64}},
            {"text": task_prompt.text},
        ],
        "role": "user",
    }
    request["generationConfig"] = dict(generation)
    if response_schema is not None:
        request["generationConfig"]["response_schema"] = response_schema
    if code_execution:
        request["tools"] = [{"code_execution": {}}]

    return request


def read_gemini_response(data: Dict[str, Any]) -> ChatAnswer:
    """Read a ``generateContent`` response with V1's failure messages.

    Args:
        data: The response as a JSON dictionary.

    Returns:
        The first text part. Output tokens include thinking tokens.

    Raises:
        ChatRequestError: When there is no candidate, generation hit the
            token limit or stopped for another reason, or no text is found.
    """
    candidates = data.get("candidates")
    if not candidates:
        raise ChatRequestError("Gemini API returned no response candidates.")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "FINISH_REASON_UNSPECIFIED")
    if finish_reason == "MAX_TOKENS":
        raise ChatRequestError(
            "Gemini API stopped generation because the max_tokens limit was reached. "
            "Please increase the max_tokens parameter to allow for a complete response."
        )
    if finish_reason not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
        raise ChatRequestError(
            f"Gemini API stopped generation with finish reason: {finish_reason}."
        )

    # With code execution, text parts sit between executableCode parts.
    try:
        parts = candidate["content"]["parts"]
        texts = [part["text"] for part in parts if "text" in part]
        text = texts[0] if texts else parts[0]["text"]
    except (KeyError, IndexError, TypeError):
        raise ChatRequestError("Unable to parse Gemini API response.") from None

    input_tokens, output_tokens = gemini_usage(data.get("usageMetadata"))
    answer = ChatAnswer(
        text=text, input_tokens=input_tokens, output_tokens=output_tokens
    )

    return answer


def gemini_usage(usage_metadata: Any) -> Tuple[Optional[int], Optional[int]]:
    """Read ``usageMetadata``: prompt tokens, and answer plus thinking tokens.

    Args:
        usage_metadata: The response's ``usageMetadata``, or ``None``.

    Returns:
        ``(input_tokens, output_tokens)``; ``None`` for a missing count.
    """
    if not isinstance(usage_metadata, dict):
        return None, None

    prompt = token_count(usage_metadata.get("promptTokenCount"))
    candidates = token_count(usage_metadata.get("candidatesTokenCount"))
    thoughts = token_count(usage_metadata.get("thoughtsTokenCount"))
    output = None
    if candidates is not None or thoughts is not None:
        output = (candidates or 0) + (thoughts or 0)

    return prompt, output


class GoogleGeminiBlock(Block):
    """Ask a Google Gemini model about each image.

    Example: ``task_type="object-detection"`` with ``classes=["car"]`` asks
    for Gemini's 0-1000 ``box_2d`` boxes and returns them as detections in
    image pixels. ``api_key="rf_key:account"`` (the default) bills the host's
    Roboflow account through its platform client; any other key calls
    Google directly.
    """

    type = "v2/google_gemini@v6"
    metadata = {
        "name": "Google Gemini",
        "port_of": "roboflow_core/google_gemini@v6",
        "section": "model",
        "task_metadata": {
            **{task: TASK_METADATA[task] for task in GEMINI_TASKS},
            DETECTION_TASK: DETECTION_TASK_METADATA,
        },
        "model_version_metadata": {
            model_id: {
                "name": model.name,
                "reasoning_levels": list(model.thinking_levels),
            }
            for model_id, model in GEMINI_MODELS.items()
        },
        "recommended_parsers": {"structured-answering": "v2/json_parser@v1"},
    }
    outputs = {
        "output": Output(
            STRING_KIND,
            LANGUAGE_MODEL_OUTPUT_KIND,
            source="images",
            description="Gemini's raw answer to this image.",
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
        task_type: GeminiTask = Field(
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
                "Google AI key as a secret or string, or 'rf_key:account' / "
                "'rf_key:user:ID' to use a key stored with Roboflow."
            ),
            examples=["rf_key:account", "$inputs.google_api_key"],
        )
        model_version: GeminiModelId | Ref(STRING_KIND) = Field(
            default=DEFAULT_MODEL,
            description=(
                "Gemini model id; see the block's model_version_metadata. Retired "
                "gemini-2.5-pro previews are renamed to gemini-2.5-pro."
            ),
            examples=["gemini-3.1-pro-preview", "$inputs.gemini_model"],
        )
        thinking_level: Optional[ThinkingLevel] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=(
                    "Reasoning depth of Gemini 3+ models; supported values differ per "
                    "model. None keeps Google's default. Not for the Gemini 2.5 series."
                ),
                examples=["low", "$inputs.thinking_level"],
            )
        )
        temperature: Optional[GeminiTemperature] | Ref(FLOAT_KIND, allow_none=True) = (
            Field(
                default=None,
                description="Sampling temperature; sent only to models without thinking levels.",
                examples=[0.4],
            )
        )
        max_tokens: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most output tokens; None keeps the model's limit.",
            examples=[2048],
        )
        google_code_execution: Optional[StrictBool] | Ref(
            BOOLEAN_KIND, allow_none=True
        ) = Field(
            default=False,
            description="Let models that support it run code while answering.",
            examples=[True],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most requests in flight for one batch; None means 8.",
            examples=[4],
        )

        @field_validator("model_version", mode="before")
        @classmethod
        def _renamed_model(cls, value: Any) -> Any:
            renamed = (
                MODEL_ALIASES.get(value, value) if isinstance(value, str) else value
            )

            return renamed

        @model_validator(mode="after")
        def _fields_consistent(self) -> "GoogleGeminiBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )
            check_thinking_level(self.model_version, self.thinking_level)

            return self

    def __init__(
        self,
        *,
        platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT,
        api_key: Any = None,
        gemini_base_url: str = GEMINI_API_URL,
        gemini_http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._platform_client = platform_client
        self._roboflow_api_key = api_key
        self._base_url = gemini_base_url.rstrip("/")
        self._http_client = gemini_http_client or httpx.Client()

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
        """Report the Gemini model the step calls."""
        resources = [
            third_party_model(
                params.model_version, provider="google", resource="gemini_http_client"
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
        thinking_level: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        google_code_execution: Optional[bool],
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
            api_key: Google key, secret or managed ``rf_key:`` reference.
            model_version: Gemini model id.
            thinking_level: Reasoning depth, or ``None``.
            temperature: Sampling temperature, or ``None``.
            max_tokens: Output token limit, or ``None``.
            google_code_execution: Whether to offer code execution.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes", "input_tokens", "output_tokens",
            "predictions", "error_status", "inference_id"}`` per image.

        Raises:
            ValueError: On a concurrency below 1, a thinking level the model
                rejects, invalid detection classes or a managed key without the
                host's Roboflow ``api_key`` resource, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        if max_concurrent_requests is not None and max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be at least 1, got {max_concurrent_requests}"
            )
        task_prompt = gemini_prompt(
            task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        generation = generation_config(
            model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            response_mime_type=task_prompt.response_mime_type,
        )
        response_schema = None
        if task_type == DETECTION_TASK:
            response_schema = detection_response_schema(classes)
        model = GEMINI_MODELS.get(model_version)
        code_execution = bool(
            google_code_execution and model is not None and model.native_code_execution
        )
        gemini_requests = [
            gemini_request(
                task_prompt,
                upload=encode_image(image),
                generation=generation,
                response_schema=response_schema,
                code_execution=code_execution,
            )
            for image in images
        ]

        keys = reveal_keys(api_key, roboflow_api_key=self._roboflow_api_key)
        if keys.managed and not keys.roboflow_key:
            raise ValueError(
                "Roboflow API key is required when using a Roboflow-managed Gemini "
                "API key; the host must provide the 'api_key' resource."
            )

        calls = [
            partial(self._ask, keys, request, model_version)
            for request in gemini_requests
        ]
        answers = run_concurrently(
            calls, max_concurrent_requests=max_concurrent_requests
        )

        results = []
        for image, answer in zip(images, answers):
            inference_id = uuid.uuid4().hex
            error_status, predictions = decode_vlm_output(
                task_type,
                answer.text,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=DETECTION_BOX_FORMAT,
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
        self, keys: ChatKeys, request: Dict[str, Any], model_version: str
    ) -> ChatAnswer:
        send = self._send_proxied if keys.managed else self._send_direct
        answer = scrubbed_call(
            partial(send, keys, request, model_version), keys=keys, provider=PROVIDER
        )

        return answer

    def _send_proxied(
        self, keys: ChatKeys, request: Dict[str, Any], model_version: str
    ) -> ChatAnswer:
        payload = {
            "model": model_version,
            "google_api_key": keys.provider_key,
            **request,
        }
        try:
            data = self._platform_client.post(
                endpoint=GEMINI_PROXY_ENDPOINT,
                api_key=keys.roboflow_key,
                payload=payload,
            )
        except requests.exceptions.RequestException as error:
            message = f"Failed to connect to Roboflow proxy: {error}"
        else:
            answer = read_gemini_response(data)
            return answer

        raise ChatRequestError(message)

    def _send_direct(
        self, keys: ChatKeys, request: Dict[str, Any], model_version: str
    ) -> ChatAnswer:
        if keys.provider_key is None:
            raise ChatRequestError("Google API key is missing.")
        response = self._http_client.post(
            f"{self._base_url}/models/{model_version}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": keys.provider_key,
            },
            json=request,
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        answer = read_gemini_response(response.json())

        return answer


def _is_selector(value: Any) -> bool:
    selector = isinstance(value, str) and value.startswith("$")

    return selector


BLOCKS = (GoogleGeminiBlock,)

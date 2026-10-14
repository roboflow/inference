"""Run any OpenRouter vision model by its slug.

Port of V1 ``roboflow_core/openrouter@v3`` and the request machinery of V1
``core_steps/common/openrouter.py``. OpenRouter speaks OpenAI chat
completions::

    per image: task ──▶ chat messages (system?, text + JPEG image, original size)
    api_key rf_key:* ─▶ platform_client.post("apiproxy/openrouter", payload)
    api_key sk-or-.. ─▶ OpenAI(base_url=openrouter_base_url,
                               http_client=openrouter_http_client)
                        .chat.completions.create(extra_body={provider, reasoning})
    a 4xx naming the reasoning config ─▶ retried once without it, as V1
    answer ──▶ decode_vlm_output(detection_format) ──▶ predictions, error_status

``privacy_level`` becomes OpenRouter's ``provider`` filter: ``allow`` adds
nothing, ``deny`` blocks data collection, ``zdr`` also requires zero data
retention. Models in ``MODEL_NATIVE_QUANTIZATIONS`` are also restricted to
their native precision or higher.

``OpenRouterBlockBase`` owns the resources and the request loop, so other
OpenRouter-served blocks only build their prompts.

Differences from V1:

* A failed request raises ``ChatRequestError`` with the keys scrubbed.
* ``max_concurrent_requests`` must be at least 1; unset means 8 instead of an
  environment variable.
* Predictions are native V2 carriers; see ``decoding``.
"""

import json
import logging
import uuid
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)
import httpx
from openai import APIStatusError, DefaultHttpxClient, OpenAI
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.models.vision_language_models.chat import (
    DEFAULT_MANAGED_KEY,
    REASONING_EFFORT_OPTIONS,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    ChatAnswer,
    ChatKeys,
    ChatPrompt,
    ChatRequestError,
    check_task_fields,
    encode_image,
    reveal_keys,
    run_concurrently,
    scrubbed_call,
    third_party_model,
    token_outputs,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.chat_completions import (
    chat_completion_usage,
    chat_messages,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.decoding import (
    BoxFormatName,
    build_object_detection_prompt,
    decode_vlm_output,
    detection_box_format,
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

PROVIDER = "OpenRouter"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

OPENROUTER_PROXY_ENDPOINT = "apiproxy/openrouter"

LEGACY_DETECTION_BOX_FORMAT = "named_normalized"

DETECTION_TASK = "object-detection"

PrivacyLevel = Literal["allow", "deny", "zdr"]

PRIVACY_LEVEL_METADATA = {
    "allow": {
        "name": "Allow data collection",
        "description": (
            "Routes to any provider, including those that may train on your data. "
            "Cheapest pool of providers; choose only if you don't mind your inputs "
            "being used to train future models."
        ),
    },
    "deny": {
        "name": "No data collection (recommended)",
        "description": (
            "Restricts to providers that do not train on your data. Default for "
            "the Roboflow-managed key. Providers may still retain inputs for "
            "short-term abuse monitoring per their own policies."
        ),
    },
    "zdr": {
        "name": "Zero data retention",
        "description": (
            "Restricts to providers that retain nothing. Smaller pool of providers; "
            "some models may be unavailable. Use when regulatory or compliance "
            "requirements forbid any retention."
        ),
    },
}

DETECTION_FORMAT_METADATA = {
    "named_normalized": {
        "name": "Named keys, 0-1 (default)",
        "description": "`x_min`/`y_min`/`x_max`/`y_max` floats normalized to 0-1.",
    },
    "xyxy_0_1000": {
        "name": "xyxy, 0-1000",
        "description": "`box_2d` list of integers normalized to 0-1000.",
    },
    "yxyx_0_1000": {
        "name": "yxyx, 0-1000",
        "description": "`box_2d` list of integers normalized to 0-1000, y first.",
    },
    "xyxy_absolute": {
        "name": "xyxy, pixels",
        "description": "`box_2d` list in pixels of the uploaded image.",
    },
    "xyxy_percent": {
        "name": "xyxy, percent",
        "description": "`box_2d` list of floats as percent of image size.",
    },
    "named_0_1000": {
        "name": "Named keys, 0-1000",
        "description": "`x_min`/`y_min`/`x_max`/`y_max` integers normalized to 0-1000.",
    },
}

BF16_NATIVE = ("bf16", "fp32")

FP8_NATIVE = ("fp8", "bf16", "fp32")

BF16_NATIVE_FP8_FLOOR = FP8_NATIVE

MODEL_NATIVE_QUANTIZATIONS: Dict[str, Tuple[str, ...]] = {
    "meta/muse-glimmer-30b": BF16_NATIVE,
    "qwen/qwen3.5-9b": BF16_NATIVE,
    "qwen/qwen3.5-27b": BF16_NATIVE,
    "qwen/qwen3.5-35b-a3b": BF16_NATIVE_FP8_FLOOR,
    "qwen/qwen3.5-122b-a10b": BF16_NATIVE,
    "qwen/qwen3.5-397b-a17b": BF16_NATIVE_FP8_FLOOR,
    "qwen/qwen3.6-27b": BF16_NATIVE_FP8_FLOOR,
    "qwen/qwen3.6-35b-a3b": BF16_NATIVE_FP8_FLOOR,
    "qwen/qwen3.8-27b": BF16_NATIVE_FP8_FLOOR,
    "qwen/qwen3-vl-8b-instruct": BF16_NATIVE,
    "qwen/qwen3-vl-30b-a3b-instruct": BF16_NATIVE,
    "qwen/qwen3-vl-30b-a3b-thinking": BF16_NATIVE_FP8_FLOOR,
    "qwen/qwen3-vl-235b-a22b-instruct": BF16_NATIVE,
    "qwen/qwen3-vl-235b-a22b-thinking": BF16_NATIVE,
    "z-ai/glm-5.3-flash": FP8_NATIVE,
    "google/gemma-4-31b-it": BF16_NATIVE,
    "google/gemma-4-26b-a4b-it": BF16_NATIVE,
    "meta-llama/llama-3.2-11b-vision-instruct": BF16_NATIVE,
}

PROXY_ERROR_STATUSES = (400, 401, 403, 413, 429, 502, 503)

_REASONING_REJECTION_MARKERS = (
    "unsupported",
    "not supported",
    "unknown",
    "invalid",
    "mandatory",
    "cannot be disabled",
)

OPENROUTER_TASKS = (
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

OpenRouterTask = Literal[OPENROUTER_TASKS]

_NO_DISCUSSION = (
    " You cannot discuss the result, you are only allowed to return JSON document."
)

OCR_SYSTEM_PROMPT = (
    "You act as OCR model. Your task is to read text from the image and return it in "
    "paragraphs representing the structure of texts in the image. You should only "
    "return recognised text, nothing else."
)

VQA_SYSTEM_PROMPT = (
    "You act as Visual Question Answering model. Your task is to provide answer to "
    "question submitted by user. If this is open-question - answer with few "
    "sentences, for ABCD question, return only the indicator of the answer."
)

CAPTION_SYSTEM_PROMPT = (
    "You act as image caption model. Your task is to provide description of the "
    "image. Caption should be {length}."
)

CLASSIFICATION_SYSTEM_PROMPT = (
    "You act as single-class classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in Markdown "
    '```json [...]``` markers. Expected structure of json: {"class_name": '
    '"class-name", "confidence": 0.4}. `class-name` must be one of the class names '
    "defined by user. You are only allowed to return single JSON document, even if "
    "there are potentially multiple classes. You are not allowed to return list."
    + _NO_DISCUSSION
)

MULTI_LABEL_SYSTEM_PROMPT = (
    "You act as multi-label classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in Markdown "
    '```json``` markers. Expected structure of json: {"predicted_classes": '
    '[{"class": "class-name-1", "confidence": 0.9}, {"class": "class-name-2", '
    '"confidence": 0.7}]}. `class-name-X` must be one of the class names defined by '
    "user and `confidence` is a float value in range 0.0-1.0 that represent how "
    "sure you are that the class is present in the image. Only return class names "
    "that are visible." + _NO_DISCUSSION
)

STRUCTURED_ANSWERING_SYSTEM_PROMPT = (
    "You are supposed to produce responses in JSON wrapped in Markdown markers: "
    "```json\nyour-response\n```. User is to provide you dictionary with keys and "
    "values. Each key must be present in your response. Values in user dictionary "
    "represent descriptions for JSON fields to be generated. Provide only JSON "
    "Markdown in response."
)

DETECTION_SYSTEM_PROMPT = (
    "You act as object-detection model. You must provide reasonable predictions. "
    "You are only allowed to produce JSON document in Markdown ```json``` markers. "
    'Expected structure of json: {"detections": [{"x_min": 0.1, "y_min": 0.2, '
    '"x_max": 0.3, "y_max": 0.4, "class_name": "my-class-X", "confidence": 0.7}]}. '
    "`my-class-X` must be one of the class names defined by user. All coordinates "
    "must be in range 0.0-1.0, representing percentage of image dimensions. "
    "`confidence` is a value in range 0.0-1.0 representing your confidence in "
    "prediction. You should detect all instances of classes provided by user."
    + _NO_DISCUSSION
)

CLASS_LIST_TEXT = "List of all classes to be recognised by model: {class_list}"

class OpenRouterProxyError(RuntimeError):
    """The Roboflow proxy answered an OpenRouter request with an HTTP error.

    Args:
        message: The proxy's ``details`` or ``error`` text.
        status_code: The HTTP status.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code

def build_provider_routing(
    privacy_level: str, *, quantizations: Optional[Sequence[str]] = None
) -> Optional[Dict[str, Any]]:
    """Build OpenRouter's ``provider`` object from privacy and quantizations.

    Args:
        privacy_level: ``allow``, ``deny`` or ``zdr``.
        quantizations: Allowed provider quantizations, or ``None``.

    Returns:
        The ``provider`` object, or ``None`` when nothing applies.

    Raises:
        ValueError: On an unknown privacy level.
    """
    if privacy_level == "allow":
        provider: Dict[str, Any] = {}
    elif privacy_level == "deny":
        provider = {"data_collection": "deny"}
    elif privacy_level == "zdr":
        provider = {"data_collection": "deny", "zdr": True}
    else:
        raise ValueError(f"unknown privacy_level: {privacy_level}")
    if quantizations:
        provider["quantizations"] = list(quantizations)

    routing = provider or None

    return routing

def reasoning_config(reasoning_effort: Optional[str]) -> Optional[Dict[str, Any]]:
    """Translate ``reasoning_effort`` into OpenRouter's ``reasoning`` object.

    Args:
        reasoning_effort: ``None`` (provider default) or a level; ``"none"``
            disables reasoning.

    Returns:
        ``None``, ``{"enabled": False}`` or ``{"effort": level}``.
    """
    if reasoning_effort is None:
        return None
    if reasoning_effort == "none":
        return {"enabled": False}

    config = {"effort": reasoning_effort}

    return config

def proxy_error_handlers() -> Dict[int, Callable[[Exception], None]]:
    """``http_errors_handlers`` raising ``OpenRouterProxyError`` with the API text.

    Returns:
        Handlers for ``PROXY_ERROR_STATUSES``.
    """

    def handler(status_code: int) -> Callable[[Exception], None]:
        def handle(http_error: Exception) -> None:
            try:
                data = http_error.response.json()
                message = data.get("details") or data.get("error") or str(http_error)
            except Exception:
                message = str(http_error)
            raise OpenRouterProxyError(str(message), status_code=status_code) from None

        return handle

    handlers = {status: handler(status) for status in PROXY_ERROR_STATUSES}

    return handlers

def is_unsupported_reasoning_error(error: Exception) -> bool:
    """Whether a client error (4xx) rejected the request's reasoning config.

    Connection errors and upstream 5xx never qualify, so a transient failure
    is not paid for twice.

    Args:
        error: The request's exception.

    Returns:
        ``True`` when a retry without ``reasoning`` is warranted.
    """
    if isinstance(error, APIStatusError):
        status = error.status_code
    elif isinstance(error, OpenRouterProxyError):
        status = error.status_code
    else:
        return False
    if not 400 <= status < 500:
        return False

    message = str(error).lower()
    rejected = "reasoning" in message and any(
        marker in message for marker in _REASONING_REJECTION_MARKERS
    )

    return rejected

def read_chat_completion(data: Dict[str, Any], *, via_proxy: bool) -> ChatAnswer:
    """Read an OpenRouter chat completion with V1's failure messages.

    Args:
        data: The completion as a JSON dictionary.
        via_proxy: Whether it came through the Roboflow proxy.

    Returns:
        The answer, its reasoning trace and token usage.

    Raises:
        ChatRequestError: On no choices or a missing ``message.content``.
    """
    choices = data.get("choices") or []
    if not choices:
        error = data.get("error")
        details = error.get("message") if isinstance(error, dict) else None
        if via_proxy:
            raise ChatRequestError(
                "OpenRouter returned no completion via Roboflow proxy. "
                f"Details: {details or 'no choices returned'}"
            )
        raise ChatRequestError(
            "OpenRouter provider failed in delivering response. This issue happens "
            "from time to time - raise issue to OpenRouter if that's problematic for "
            f"you. Details: {details or 'N/A'}"
        )

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        length_hint = (
            " The model may be a reasoning model that ran out of tokens during its "
            "internal thinking step. Try increasing `max_tokens` (e.g. 1000+) or pick "
            "a non-reasoning model variant."
        )
        ran_out = choices[0].get("finish_reason") == "length"
        if via_proxy:
            hint = length_hint if ran_out else ""
            raise ChatRequestError(
                "OpenRouter response missing message.content via Roboflow proxy." + hint
            )
        hint = (
            length_hint
            if ran_out
            else " This can happen when the model returns only tool calls or "
            "reasoning tokens. Try a different prompt or model."
        )
        raise ChatRequestError("OpenRouter response missing message.content." + hint)

    thinking = message.get("reasoning")
    input_tokens, output_tokens = chat_completion_usage(data.get("usage"))
    answer = ChatAnswer(
        text=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking=thinking if isinstance(thinking, str) else "",
    )

    return answer

class OpenRouterBlockBase(Block):
    """Resources and request loop shared by OpenRouter-served blocks.

    Subclasses build one ``messages`` list per image and call ``ask_all``.
    """

    def __init__(
        self,
        *,
        platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT,
        api_key: Any = None,
        openrouter_base_url: str = OPENROUTER_API_URL,
        openrouter_http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._platform_client = platform_client
        self._roboflow_api_key = api_key
        self._base_url = openrouter_base_url
        self._http_client = openrouter_http_client or DefaultHttpxClient()

    def ask_all(
        self,
        prompts: List[List[Dict[str, Any]]],
        *,
        api_key: Any,
        model: str,
        max_tokens: int,
        temperature: Optional[float],
        privacy_level: str,
        reasoning_effort: Optional[str],
        max_concurrent_requests: Optional[int],
    ) -> List[ChatAnswer]:
        """Send one chat completion per ``messages`` list; answers in order.

        Args:
            prompts: One ``messages`` list per request.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            model: OpenRouter model slug.
            max_tokens: Output token limit.
            temperature: Sampling temperature; ``None`` keeps the provider's.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            reasoning_effort: Reasoning level or ``None``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One answer per request.

        Raises:
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails.
        """
        keys = reveal_keys(api_key, roboflow_api_key=self._roboflow_api_key)
        if keys.provider_key is None:
            raise ValueError("OpenRouter API key is required.")

        if keys.managed and not keys.roboflow_key:
            raise ValueError(
                "Roboflow API key is required when using a Roboflow-managed OpenRouter "
                "API key; the host must provide the 'api_key' resource."
            )

        build_provider_routing(privacy_level)  # an unknown level fails before sending
        options = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "privacy_level": privacy_level,
            "reasoning": reasoning_config(reasoning_effort),
            "quantizations": MODEL_NATIVE_QUANTIZATIONS.get(model),
        }
        calls = [partial(self._ask, keys, messages, **options) for messages in prompts]
        answers = run_concurrently(
            calls, max_concurrent_requests=max_concurrent_requests
        )

        return answers

    def _ask(
        self, keys: ChatKeys, messages: List[Dict[str, Any]], **options: Any
    ) -> ChatAnswer:
        request = partial(self._request, keys, messages, **options)
        answer = scrubbed_call(request, keys=keys, provider=PROVIDER)

        return answer

    def _request(
        self,
        keys: ChatKeys,
        messages: List[Dict[str, Any]],
        *,
        reasoning: Optional[Dict[str, Any]],
        **options: Any,
    ) -> ChatAnswer:
        send = self._send_proxied if keys.managed else self._send_direct
        try:
            data = send(keys, messages, reasoning=reasoning, **options)
        except Exception as error:
            if reasoning is None or not is_unsupported_reasoning_error(error):
                raise
            logger.warning(
                "OpenRouter rejected the reasoning config %s for model %s "
                "(details: %s). Retrying without it - the model will use its "
                "provider-default reasoning behavior.",
                reasoning,
                options["model"],
                keys.scrub(str(error)),
            )
            data = send(keys, messages, reasoning=None, **options)
        answer = read_chat_completion(data, via_proxy=keys.managed)

        return answer

    def _send_proxied(
        self,
        keys: ChatKeys,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
        temperature: Optional[float],
        privacy_level: str,
        reasoning: Optional[Dict[str, Any]],
        quantizations: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "openrouter_api_key": keys.provider_key,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "privacy_level": privacy_level,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if quantizations is not None:
            payload["quantizations"] = list(quantizations)
        data = self._platform_client.post(
            endpoint=OPENROUTER_PROXY_ENDPOINT,
            api_key=keys.roboflow_key,
            payload=payload,
            http_errors_handlers=proxy_error_handlers(),
        )

        return data

    def _send_direct(
        self,
        keys: ChatKeys,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
        temperature: Optional[float],
        privacy_level: str,
        reasoning: Optional[Dict[str, Any]],
        quantizations: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {}
        provider = build_provider_routing(privacy_level, quantizations=quantizations)
        if provider is not None:
            extra_body["provider"] = provider
        if reasoning is not None:
            extra_body["reasoning"] = reasoning
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if temperature is not None:
            request["temperature"] = temperature
        client = OpenAI(
            base_url=self._base_url,
            api_key=keys.provider_key,
            http_client=self._http_client,
        )
        data = client.chat.completions.create(**request).to_dict()

        return data

def openrouter_task_messages(
    task_type: str,
    *,
    image: ImageData,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    detection_format: str,
) -> List[Dict[str, Any]]:
    """Build V1's chat messages of one task; the image goes as JPEG, full size.

    Args:
        task_type: One of ``OPENROUTER_TASKS``.
        image: The image to ask about.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.
        detection_format: Box format a detection prompt asks for;
            ``named_normalized`` keeps V1's legacy system prompt.

    Returns:
        The ``messages`` list.

    Raises:
        ValueError: On an unknown task.
    """
    upload = encode_image(image)
    class_text = CLASS_LIST_TEXT.format(class_list=", ".join(classes or []))
    if task_type == DETECTION_TASK and detection_format != LEGACY_DETECTION_BOX_FORMAT:
        text = build_object_detection_prompt(
            detection_format, classes=classes, upload_size_wh=upload.size_wh
        )
        messages = chat_messages(ChatPrompt(None, text), [upload.data_url])
        return messages

    prompts = {
        "unconstrained": lambda: ChatPrompt(None, prompt),
        "ocr": lambda: ChatPrompt(OCR_SYSTEM_PROMPT, None),
        "visual-question-answering": lambda: ChatPrompt(
            VQA_SYSTEM_PROMPT, f"Question: {prompt}"
        ),
        "caption": lambda: ChatPrompt(
            CAPTION_SYSTEM_PROMPT.format(length="short"), None
        ),
        "detailed-caption": lambda: ChatPrompt(
            CAPTION_SYSTEM_PROMPT.format(length="extensive"), None
        ),
        "classification": lambda: ChatPrompt(CLASSIFICATION_SYSTEM_PROMPT, class_text),
        "multi-label-classification": lambda: ChatPrompt(
            MULTI_LABEL_SYSTEM_PROMPT, class_text
        ),
        "structured-answering": lambda: ChatPrompt(
            STRUCTURED_ANSWERING_SYSTEM_PROMPT,
            "Specification of requirements regarding output fields: \n"
            + json.dumps(output_structure, indent=4),
        ),
        DETECTION_TASK: lambda: ChatPrompt(DETECTION_SYSTEM_PROMPT, class_text),
    }
    if task_type not in prompts:
        raise ValueError(f"Task type: {task_type} not supported.")

    messages = chat_messages(prompts[task_type](), [upload.data_url])

    return messages

class OpenRouterBlock(OpenRouterBlockBase):
    """Ask any OpenRouter vision model about each image by pasting its slug.

    Example: ``model_id="qwen/qwen3.6-27b"``, ``task_type="object-detection"``,
    ``classes=["car"]`` and ``detection_format="xyxy_0_1000"`` asks for
    0-1000 boxes and returns them as detections in image pixels.
    ``api_key="rf_key:account"`` (the default) bills the host's Roboflow
    account through its platform client.
    """

    type = "roboflow_core/openrouter@v3"
    metadata = {
        "name": "OpenRouter",
        "port_of": "roboflow_core/openrouter@v3",
        "section": "model",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "detection_format_metadata": DETECTION_FORMAT_METADATA,
        "recommended_parsers": {"structured-answering": "roboflow_core/json_parser@v1"},
    }
    outputs = {
        "output": Output(
            STRING_KIND,
            LANGUAGE_MODEL_OUTPUT_KIND,
            source="images",
            description="The model's raw answer to this image.",
        ),
        "classes": Output(
            LIST_OF_VALUES_KIND, source="images", description="The step's classes."
        ),
        "thinking": Output(
            STRING_KIND,
            source="images",
            description="Reasoning trace when OpenRouter returns one, else empty.",
        ),
        **token_outputs(source="images"),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to ask about; one request per image."
        )
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "OpenRouter model slug, e.g. `openai/gpt-4o-mini`, "
                "`anthropic/claude-3.5-sonnet`, `qwen/qwen3.6-27b`. See "
                "https://openrouter.ai/models for the full list."
            ),
            examples=["qwen/qwen3.6-27b", "$inputs.openrouter_model_id"],
        )
        task_type: OpenRouterTask = Field(
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
        detection_format: BoxFormatName | Ref(STRING_KIND) = Field(
            default=LEGACY_DETECTION_BOX_FORMAT,
            description="Box format object-detection asks for and decodes.",
            examples=["xyxy_0_1000"],
        )
        api_key: ApiKey = Field(
            default=DEFAULT_MANAGED_KEY,
            description=(
                "OpenRouter key as a secret or string, or 'rf_key:account' / "
                "'rf_key:user:ID' to use a key stored with Roboflow."
            ),
            examples=["rf_key:account", "sk-or-...", "$inputs.openrouter_api_key"],
        )
        privacy_level: PrivacyLevel = Field(
            default="deny",
            description=(
                "Provider privacy filter. Stricter levels reduce the pool of "
                "providers and may increase per-call cost on the managed key."
            ),
            examples=["deny"],
        )
        max_tokens: StrictInt = Field(
            default=2048,
            gt=1,
            description="Most output tokens; reasoning models spend some on thinking.",
            examples=[2048],
        )
        temperature: Annotated[StrictFloat, Field(ge=0.0, le=2.0)] | Ref(FLOAT_KIND) = (
            Field(
                default=0.1,
                description="Sampling temperature.",
                examples=[0.1],
            )
        )
        reasoning_effort: Optional[Literal[REASONING_EFFORT_OPTIONS]] | Ref(
            STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Sent as OpenRouter `reasoning`; 'none' disables it. None keeps the "
                "model's default. A model rejecting it is retried without it."
            ),
            examples=["low", "$inputs.reasoning_effort"],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most requests in flight for one batch; None means 8.",
            examples=[4],
        )

        @model_validator(mode="after")
        def _task_fields_present(self) -> "OpenRouterBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )

            return self

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
        """Report the OpenRouter model the step calls."""
        resources = [
            third_party_model(
                params.model_id,
                provider="openrouter",
                resource="openrouter_http_client",
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
        model_id: str,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        detection_format: str,
        api_key: Any,
        privacy_level: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str],
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Send one request per image and decode each answer.

        Args:
            images: The batch of images.
            model_id: OpenRouter model slug.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            detection_format: Box format of a detection task.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_tokens: Output token limit.
            temperature: Sampling temperature.
            reasoning_effort: Reasoning level or ``None``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes", "thinking", "input_tokens",
            "output_tokens", "predictions", "error_status", "inference_id"}``
            per image.

        Raises:
            ValueError: On an unknown detection format, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2.")

        detection_contract = None
        if task_type == DETECTION_TASK:
            detection_contract = detection_box_format(detection_format)

        prompts = [
            openrouter_task_messages(
                task_type,
                image=image,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
                detection_format=detection_format,
            )
            for image in images
        ]
        answers = self.ask_all(
            prompts,
            api_key=api_key,
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            reasoning_effort=reasoning_effort,
            max_concurrent_requests=max_concurrent_requests,
        )

        results = []
        for image, answer in zip(images, answers):
            inference_id = uuid.uuid4().hex
            needs_size = (
                detection_contract is not None
                and detection_contract.requires_upload_size
            )
            error_status, predictions = decode_vlm_output(
                task_type,
                answer.text,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=detection_format if detection_contract is not None else None,
                upload_size_wh=(image.width, image.height) if needs_size else None,
            )
            results.append(
                {
                    "output": answer.text,
                    "classes": classes,
                    "thinking": answer.thinking,
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    "predictions": predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )

        return results

BLOCKS = (OpenRouterBlock,)

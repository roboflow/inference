"""Shared utilities for workflow blocks that route through OpenRouter.

This module owns the OpenRouter API Key Passthrough plumbing so individual
blocks (Gemma, Llama Vision, Kimi, unified Qwen) only need to declare their
manifest specifics (model dropdown, search keywords, icon) and call into the
shared base class for execution.

Two key paths are supported per call:
    1. **Roboflow-managed key** (default for new blocks): the user's `api_key`
       starts with ``rf_key:account`` (or ``rf_key:user:<id>``) and is sent to
       Roboflow's ``apiproxy/openrouter`` route, which resolves to the managed
       OpenRouter key, applies privacy filters, bills credits, and returns the
       upstream response.
    2. **Custom user key**: any other ``api_key`` value (e.g. ``sk-or-...``) is
       passed straight to ``openrouter.ai`` via the OpenAI SDK with no Roboflow
       proxy in the loop.

Both paths honor a user-selected ``privacy_level`` of ``allow``, ``deny``, or
``zdr`` (zero data retention). Both also attach a per-model provider
``quantizations`` allowlist (:data:`MODEL_NATIVE_QUANTIZATIONS`) so OpenRouter
only routes to providers serving the model at native precision or higher.
Full task-type prompt builders shared across the VLM blocks live here too so
the per-block files stay small.
"""

import base64
import json
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from openai import APIStatusError, OpenAI
from pydantic import ConfigDict, Field

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.exceptions import (
    RoboflowAPIForbiddenError,
    RoboflowAPIUnsuccessfulRequestError,
)
from inference.core.logger import logger
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.workflows.core_steps.common.token_usage import (
    parse_chat_completion_usage,
)
from inference.core.workflows.core_steps.common.utils import run_in_parallel
from inference.core.workflows.core_steps.common.vlms import VLM_TASKS_METADATA
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

# ---------------------------------------------------------------------------
# Privacy level
# ---------------------------------------------------------------------------

PRIVACY_LEVEL_LITERAL = Literal["allow", "deny", "zdr"]

PRIVACY_LEVEL_METADATA = {
    "allow": {
        "name": "Allow data collection",
        "description": (
            "Routes to any provider, including those that may train on your "
            "data. Cheapest pool of providers; choose only if you don't mind "
            "your inputs being used to train future models."
        ),
    },
    "deny": {
        "name": "No data collection (recommended)",
        "description": (
            "Restricts to providers that do not train on your data. Default "
            "for the Roboflow-managed key. Providers may still retain inputs "
            "for short-term abuse monitoring per their own policies."
        ),
    },
    "zdr": {
        "name": "Zero data retention",
        "description": (
            "Restricts to providers that retain nothing. Smaller pool of "
            "providers; some models may be unavailable. Use when regulatory "
            "or compliance requirements forbid any retention."
        ),
    },
}


def build_provider_routing(
    privacy_level: str,
    quantizations: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Build the ``provider`` object for an OpenRouter request.

    Merges the ``privacy_level`` filter (``allow`` adds nothing, ``deny``
    blocks data collection, ``zdr`` additionally requires zero data
    retention) with an optional ``quantizations`` allowlist. Returns ``None``
    when neither applies.

    Raises:
        ValueError: If ``privacy_level`` is not a known level.
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

    return provider or None


# ---------------------------------------------------------------------------
# Native model precision (provider quantization filter)
# ---------------------------------------------------------------------------

# Allowlists for OpenRouter's `provider.quantizations` filter, meaning "the
# model's native precision or higher". Endpoints reporting any other
# quantization (including `unknown`) are excluded from routing.
BF16_NATIVE = ("bf16", "fp32")
FP8_NATIVE = ("fp8", "bf16", "fp32")
# BF16-native models with no BF16 endpoint on OpenRouter today; FP8 keeps
# them routable at the highest precision actually served.
BF16_NATIVE_FP8_FLOOR = FP8_NATIVE

# OpenRouter model slug -> acceptable provider quantization labels, from the
# model's native release precision and what OpenRouter providers serve today.
#
# Absent models get no filter, because a filter would leave them with zero
# providers: single-provider hosted SKUs (Muse Spark 1.1/1.2, Qwen
# Flash/Plus/Max, the Qwen 3.7 line, GLM-5V-Turbo, DeepSeek V4 Flash Vision
# Exp) report `unknown` precision; DeepSeek V4 is natively FP4+FP8; the only
# endpoints for Qwen3 VL 8B Thinking and 32B Instruct report `unknown`.
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


def get_native_quantizations(model: str) -> Optional[Tuple[str, ...]]:
    """Return the quantization allowlist for an OpenRouter model slug.

    Returns ``None`` for models absent from
    :data:`MODEL_NATIVE_QUANTIZATIONS`; absence means the model must not
    be filtered by quantization.
    """
    return MODEL_NATIVE_QUANTIZATIONS.get(model)


# ---------------------------------------------------------------------------
# Manifest mixin
# ---------------------------------------------------------------------------


class OpenRouterBlockManifestMixin(WorkflowBlockManifest):
    """Pydantic mixin contributing the OpenRouter-specific manifest fields.

    Concrete block manifests inherit from this AND declare their own ``type``,
    ``model_version``, ``task_type``, ``images``, ``prompt``, etc.
    """

    api_key: Union[
        Selector(kind=[STRING_KIND, SECRET_KIND, ROBOFLOW_MANAGED_KEY]), str
    ] = Field(
        default="rf_key:account",
        description=(
            "OpenRouter API key. Defaults to Roboflow's managed key, billed in "
            "credits via Roboflow. Provide your own `sk-or-...` key to call "
            "OpenRouter directly without Roboflow billing."
        ),
        examples=["rf_key:account", "sk-or-...", "$inputs.openrouter_api_key"],
        private=True,
    )
    privacy_level: PRIVACY_LEVEL_LITERAL = Field(
        default="deny",
        description=(
            "Provider privacy filter. Stricter levels reduce the pool of "
            "providers and may increase per-call cost on the managed key."
        ),
        json_schema_extra={"values_metadata": PRIVACY_LEVEL_METADATA},
    )
    max_tokens: int = Field(
        default=500,
        description="Maximum number of tokens the model can generate in its response.",
        gt=1,
    )
    temperature: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.1,
        description=(
            "Temperature to sample from the model - value in range 0.0-2.0, "
            'the higher - the more random / "creative" the generations are.'
        ),
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description=(
            "Number of concurrent requests for batches of images. If not "
            "given - block defaults to value configured globally in Workflows "
            "Execution Engine. Restrict if you hit rate limits."
        ),
    )


# ---------------------------------------------------------------------------
# Base class for block runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenRouterResult:
    """One chat-completion result with its reasoning trace and token usage.

    ``reasoning_trace`` is the ``message.reasoning`` string OpenRouter returns
    for reasoning models (empty string when absent). Token counts are parsed
    from the provider ``usage`` object; ``None`` means usage was omitted,
    never a real zero.
    """

    content: str
    reasoning_trace: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class OpenRouterWorkflowBlockBase(WorkflowBlock):
    """Shared base class for blocks that route through OpenRouter.

    Subclasses provide manifest + prompt-building; this class owns the
    routing/execution machinery.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
    ):
        self._model_manager = model_manager
        self._roboflow_api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key"]

    def execute_openrouter_batch(
        self,
        openrouter_api_key: str,
        model: str,
        prompts: List[List[dict]],
        max_tokens: int,
        temperature: Optional[float],
        privacy_level: str,
        max_concurrent_requests: Optional[int],
        reasoning: Optional[dict] = None,
        include_reasoning: bool = False,
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """Run a batch of OpenRouter chat-completion calls in parallel.

        Legacy result shapes, kept stable for shipped block versions: bare
        content strings by default, ``(content, reasoning_trace)`` tuples
        with ``include_reasoning=True``. New blocks that need token usage
        should call :meth:`execute_openrouter_batch_with_usage` instead.

        See :meth:`execute_openrouter_batch_with_usage` for routing,
        ``temperature`` and ``reasoning`` semantics.
        """
        results = self.execute_openrouter_batch_with_usage(
            openrouter_api_key=openrouter_api_key,
            model=model,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            max_concurrent_requests=max_concurrent_requests,
            reasoning=reasoning,
        )
        if include_reasoning:
            return [(r.content, r.reasoning_trace) for r in results]
        return [r.content for r in results]

    def execute_openrouter_batch_with_usage(
        self,
        openrouter_api_key: str,
        model: str,
        prompts: List[List[dict]],
        max_tokens: int,
        temperature: Optional[float],
        privacy_level: str,
        max_concurrent_requests: Optional[int],
        reasoning: Optional[dict] = None,
    ) -> List[OpenRouterResult]:
        """Run a batch of OpenRouter chat-completion calls in parallel.

        Routes through the Roboflow proxy when ``openrouter_api_key`` starts
        with ``rf_key:`` (managed/user-stored), otherwise calls the OpenRouter
        API directly using the OpenAI SDK with the provided key. Works on
        both paths: the reasoning trace and token usage are always populated
        on the returned :class:`OpenRouterResult` objects.

        ``temperature`` set to ``None`` omits the parameter so the provider
        default applies. ``reasoning`` is an optional OpenRouter reasoning
        config object (e.g. ``{"effort": "low"}`` or ``{"enabled": False}``)
        forwarded verbatim; when the target model rejects the config, the
        request is retried once without it.

        Note: honoring ``reasoning`` on managed ``rf_key:`` keys requires a
        Roboflow platform proxy version that forwards the key upstream
        (older proxy versions strip it and the model applies its
        provider-default reasoning behavior).

        Models registered in :data:`MODEL_NATIVE_QUANTIZATIONS` get a
        provider ``quantizations`` allowlist on every request, restricting
        routing to native precision or higher. On the proxied path the
        Roboflow proxy must forward the field upstream.
        """
        quantizations = get_native_quantizations(model=model)

        is_managed = openrouter_api_key.startswith(("rf_key:account", "rf_key:user:"))
        if is_managed:
            single = partial(
                _execute_proxied_openrouter_request,
                roboflow_api_key=self._roboflow_api_key,
                openrouter_api_key=openrouter_api_key,
                model=model,
                privacy_level=privacy_level,
                reasoning=reasoning,
                quantizations=quantizations,
            )
        else:
            single = partial(
                _execute_direct_openrouter_request,
                api_key=openrouter_api_key,
                model=model,
                privacy_level=privacy_level,
                reasoning=reasoning,
                quantizations=quantizations,
            )
        tasks = [
            partial(
                single,
                messages=p,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for p in prompts
        ]
        max_workers = (
            max_concurrent_requests
            or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
        )
        return run_in_parallel(tasks=tasks, max_workers=max_workers)


# ---------------------------------------------------------------------------
# Per-request helpers
# ---------------------------------------------------------------------------


def _build_proxy_error_handler(
    status_code: int,
) -> Callable[[Exception], None]:
    def _handler(http_error: Exception) -> None:
        try:
            error_data = http_error.response.json()
            api_msg = (
                error_data.get("details") or error_data.get("error") or str(http_error)
            )
        except Exception:
            api_msg = str(http_error)
        if status_code == 403:
            error = RoboflowAPIForbiddenError(api_msg)
        else:
            error = RoboflowAPIUnsuccessfulRequestError(api_msg)
        # The exception types carry no HTTP status; attach it so callers (the
        # retry-without-reasoning heuristic) can tell a client rejection from
        # a relayed upstream 5xx.
        error.status_code = status_code
        raise error from http_error

    return _handler


_PROXY_ERROR_HANDLERS: Dict[int, Callable[[Exception], None]] = {
    400: _build_proxy_error_handler(400),
    401: _build_proxy_error_handler(401),
    403: _build_proxy_error_handler(403),
    413: _build_proxy_error_handler(413),
    429: _build_proxy_error_handler(429),
    # 502 = OpenRouter upstream failure surfaced by the Roboflow proxy. Map
    # it through the same handler so the upstream provider's error message
    # is preserved in the workflow result.
    502: _build_proxy_error_handler(502),
    503: _build_proxy_error_handler(503),
}


# Substrings observed in real OpenRouter rejections of a `reasoning` config.
# Captured live against qwen/qwen3.8-max and qwen/qwen3.7-flash:
#   "Reasoning is mandatory for this endpoint and cannot be disabled." (400)
#   'reasoning.effort: Invalid option: expected one of "max"|"xhigh"|...' (400)
_REASONING_REJECTION_MARKERS = (
    "unsupported",
    "not supported",
    "unknown",
    "invalid",
    "mandatory",
    "cannot be disabled",
)


def _is_unsupported_reasoning_error(error: Exception) -> bool:
    """Whether the error is a client-error rejection of the reasoning config.

    Only client (HTTP 4xx) errors qualify — connection/timeout failures and
    relayed upstream 5xx must never trigger the retry-without-reasoning
    fallback, as that would issue a duplicate billed request for a transient
    problem. On the direct path the OpenAI SDK raises ``APIStatusError`` with
    a status code; on the proxied path ``_PROXY_ERROR_HANDLERS`` attaches
    ``status_code`` to the raised Roboflow exception. An exception without a
    status (raised outside those handlers) is treated as not retryable.
    """
    if isinstance(error, APIStatusError):
        status = error.status_code
    elif isinstance(
        error, (RoboflowAPIUnsuccessfulRequestError, RoboflowAPIForbiddenError)
    ):
        status = getattr(error, "status_code", None)
    else:
        return False
    if status is None or not 400 <= status < 500:
        return False
    message = str(error).lower()
    return "reasoning" in message and any(
        marker in message for marker in _REASONING_REJECTION_MARKERS
    )


def _execute_proxied_openrouter_request(
    roboflow_api_key: Optional[str],
    openrouter_api_key: str,
    model: str,
    messages: List[dict],
    max_tokens: int,
    temperature: Optional[float],
    privacy_level: str,
    reasoning: Optional[dict] = None,
    quantizations: Optional[Sequence[str]] = None,
) -> OpenRouterResult:
    payload = {
        "openrouter_api_key": openrouter_api_key,
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
    try:
        response_data = post_to_roboflow_api(
            endpoint="apiproxy/openrouter",
            api_key=roboflow_api_key,
            payload=payload,
            http_errors_handlers=_PROXY_ERROR_HANDLERS,
        )
    except Exception as error:
        if reasoning is None or not _is_unsupported_reasoning_error(error):
            raise
        logger.warning(
            "OpenRouter rejected the reasoning config %s for model %s "
            "(details: %s). Retrying without it - the model will use its "
            "provider-default reasoning behavior.",
            reasoning,
            model,
            error,
        )
        retry_payload = {k: v for k, v in payload.items() if k != "reasoning"}
        response_data = post_to_roboflow_api(
            endpoint="apiproxy/openrouter",
            api_key=roboflow_api_key,
            payload=retry_payload,
            http_errors_handlers=_PROXY_ERROR_HANDLERS,
        )
    choices = response_data.get("choices") or []
    if not choices:
        err_msg = (
            (response_data.get("error") or {}).get("message")
            if isinstance(response_data.get("error"), dict)
            else None
        ) or "no choices returned"
        raise RuntimeError(
            "OpenRouter returned no completion via Roboflow proxy. "
            f"Details: {err_msg}"
        )
    message = choices[0].get("message") or {}
    content = message.get("content")
    input_tokens, output_tokens = parse_chat_completion_usage(
        response_data.get("usage")
    )
    if content is None:
        # Reasoning models (Kimi K2.x, some Qwen 3.5/3.6 variants) emit all
        # of their tokens as `reasoning` and run out before producing visible
        # `content` if `max_tokens` is too low. Surface a hint so users know
        # how to recover instead of just raising "missing content".
        finish = choices[0].get("finish_reason")
        hint = (
            " The model may be a reasoning model that ran out of tokens "
            "during its internal thinking step. Try increasing `max_tokens` "
            "(e.g. 1000+) or pick a non-reasoning model variant."
            if finish == "length"
            else ""
        )
        raise RuntimeError(
            "OpenRouter response missing message.content via Roboflow proxy." + hint
        )
    reasoning_trace = message.get("reasoning")
    return OpenRouterResult(
        content=content,
        reasoning_trace=reasoning_trace if isinstance(reasoning_trace, str) else "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _execute_direct_openrouter_request(
    api_key: str,
    model: str,
    messages: List[dict],
    max_tokens: int,
    temperature: Optional[float],
    privacy_level: str,
    reasoning: Optional[dict] = None,
    quantizations: Optional[Sequence[str]] = None,
) -> OpenRouterResult:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    extra_body: Dict[str, Any] = {}
    provider = build_provider_routing(privacy_level, quantizations=quantizations)
    if provider is not None:
        extra_body["provider"] = provider
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    try:
        response = client.chat.completions.create(
            **request_kwargs,
            extra_body=extra_body,
        )
    except Exception as error:
        if reasoning is None or not _is_unsupported_reasoning_error(error):
            raise
        logger.warning(
            "OpenRouter rejected the reasoning config %s for model %s "
            "(details: %s). Retrying without it - the model will use its "
            "provider-default reasoning behavior.",
            reasoning,
            model,
            error,
        )
        retry_extra_body = {k: v for k, v in extra_body.items() if k != "reasoning"}
        response = client.chat.completions.create(
            **request_kwargs,
            extra_body=retry_extra_body,
        )
    if not response.choices:
        error_detail = getattr(response, "error", {}) or {}
        if isinstance(error_detail, dict):
            error_detail = error_detail.get("message", "N/A")
        raise RuntimeError(
            "OpenRouter provider failed in delivering response. This issue "
            "happens from time to time - raise issue to OpenRouter if that's "
            f"problematic for you. Details: {error_detail}"
        )
    content = response.choices[0].message.content
    if content is None:
        # Reasoning models can burn all of `max_tokens` on internal thinking
        # before producing user-visible content. The OpenAI SDK populates
        # `finish_reason` on the choice; "length" means we ran out.
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        hint = (
            " The model may be a reasoning model that ran out of tokens "
            "during its internal thinking step. Try increasing `max_tokens` "
            "(e.g. 1000+) or pick a non-reasoning model variant."
            if finish_reason == "length"
            else " This can happen when the model returns only tool calls "
            "or reasoning tokens. Try a different prompt or model."
        )
        raise RuntimeError("OpenRouter response missing message.content." + hint)
    # OpenRouter returns the reasoning trace as an extra `reasoning`
    # field on the message; the OpenAI SDK surfaces unknown fields as
    # attributes on its pydantic models.
    reasoning_trace = getattr(response.choices[0].message, "reasoning", None)
    input_tokens, output_tokens = parse_chat_completion_usage(
        getattr(response, "usage", None)
    )
    return OpenRouterResult(
        content=content,
        reasoning_trace=reasoning_trace if isinstance(reasoning_trace, str) else "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ---------------------------------------------------------------------------
# Shared task-type surface (mirrors Gemma/Kimi/Llama v1 builders)
# ---------------------------------------------------------------------------


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

RECOMMENDED_PARSERS = {
    "structured-answering": "roboflow_core/json_parser@v1",
    "classification": "roboflow_core/vlm_as_classifier@v2",
    "multi-label-classification": "roboflow_core/vlm_as_classifier@v2",
    "object-detection": "roboflow_core/vlm_as_detector@v2",
}


def _prepare_unconstrained_prompt(base64_image: str, prompt: str, **_) -> List[dict]:
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


def _prepare_classification_prompt(
    base64_image: str, classes: List[str], **_
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": (
                "You act as single-class classification model. You must "
                "provide reasonable predictions. You are only allowed to "
                "produce JSON document in Markdown ```json [...]``` markers. "
                'Expected structure of json: {"class_name": "class-name", '
                '"confidence": 0.4}. `class-name` must be one of the class '
                "names defined by user. You are only allowed to return single "
                "JSON document, even if there are potentially multiple "
                "classes. You are not allowed to return list. You cannot "
                "discuss the result, you are only allowed to return JSON "
                "document."
            ),
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


def _prepare_multi_label_classification_prompt(
    base64_image: str, classes: List[str], **_
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": (
                "You act as multi-label classification model. You must "
                "provide reasonable predictions. You are only allowed to "
                "produce JSON document in Markdown ```json``` markers. "
                'Expected structure of json: {"predicted_classes": '
                '[{"class": "class-name-1", "confidence": 0.9}, '
                '{"class": "class-name-2", "confidence": 0.7}]}. '
                "`class-name-X` must be one of the class names defined by "
                "user and `confidence` is a float value in range 0.0-1.0 "
                "that represent how sure you are that the class is present "
                "in the image. Only return class names that are visible. "
                "You cannot discuss the result, you are only allowed to "
                "return JSON document."
            ),
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


def _prepare_vqa_prompt(base64_image: str, prompt: str, **_) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You act as Visual Question Answering model. Your task is "
                "to provide answer to question submitted by user. If this "
                "is open-question - answer with few sentences, for ABCD "
                "question, return only the indicator of the answer."
            ),
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


def _prepare_ocr_prompt(base64_image: str, **_) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You act as OCR model. Your task is to read text from the "
                "image and return it in paragraphs representing the structure "
                "of texts in the image. You should only return recognised "
                "text, nothing else."
            ),
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


def _prepare_caption_prompt(
    base64_image: str, short_description: bool, **_
) -> List[dict]:
    caption_detail_level = (
        "Caption should be short."
        if short_description
        else "Caption should be extensive."
    )
    return [
        {
            "role": "system",
            "content": (
                "You act as image caption model. Your task is to provide "
                f"description of the image. {caption_detail_level}"
            ),
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


def _prepare_structured_answering_prompt(
    base64_image: str, output_structure: Dict[str, str], **_
) -> List[dict]:
    output_structure_serialised = json.dumps(output_structure, indent=4)
    return [
        {
            "role": "system",
            "content": (
                "You are supposed to produce responses in JSON wrapped in "
                "Markdown markers: ```json\nyour-response\n```. User is to "
                "provide you dictionary with keys and values. Each key must "
                "be present in your response. Values in user dictionary "
                "represent descriptions for JSON fields to be generated. "
                "Provide only JSON Markdown in response."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Specification of requirements regarding output "
                        f"fields: \n{output_structure_serialised}"
                    ),
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


def _prepare_object_detection_prompt(
    base64_image: str, classes: List[str], **_
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": (
                "You act as object-detection model. You must provide "
                "reasonable predictions. You are only allowed to produce "
                "JSON document in Markdown ```json``` markers. Expected "
                'structure of json: {"detections": [{"x_min": 0.1, '
                '"y_min": 0.2, "x_max": 0.3, "y_max": 0.4, '
                '"class_name": "my-class-X", "confidence": 0.7}]}. '
                "`my-class-X` must be one of the class names defined by "
                "user. All coordinates must be in range 0.0-1.0, "
                "representing percentage of image dimensions. `confidence` "
                "is a value in range 0.0-1.0 representing your confidence "
                "in prediction. You should detect all instances of classes "
                "provided by user. You cannot discuss the result, you are "
                "only allowed to return JSON document."
            ),
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


PROMPT_BUILDERS = {
    "unconstrained": _prepare_unconstrained_prompt,
    "ocr": _prepare_ocr_prompt,
    "visual-question-answering": _prepare_vqa_prompt,
    "caption": partial(_prepare_caption_prompt, short_description=True),
    "detailed-caption": partial(_prepare_caption_prompt, short_description=False),
    "classification": _prepare_classification_prompt,
    "multi-label-classification": _prepare_multi_label_classification_prompt,
    "structured-answering": _prepare_structured_answering_prompt,
    "object-detection": _prepare_object_detection_prompt,
}


def build_prompts_from_images(
    images: List[Dict[str, Any]],
    task_type: str,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> List[List[dict]]:
    """Build a list of OpenRouter ``messages`` arrays, one per input image.

    ``images`` items are inference-format image dicts as produced by
    ``WorkflowImageData.to_inference_format()``.
    """
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    builder = PROMPT_BUILDERS[task_type]
    built: List[List[dict]] = []
    for image in images:
        loaded_image, _ = load_image(image)
        base64_image = base64.b64encode(
            encode_image_to_jpeg_bytes(loaded_image)
        ).decode("ascii")
        built.append(
            builder(
                base64_image=base64_image,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
        )
    return built


def validate_task_type_required_fields(
    task_type: str,
    prompt: Optional[str],
    classes: Optional[List[str]],
    output_structure: Optional[Dict[str, str]],
) -> None:
    """Raise ``ValueError`` if a required field for ``task_type`` is missing.

    Used by block manifests' ``model_validator`` to surface a clear error
    before the workflow runs.
    """
    if task_type in TASKS_REQUIRING_PROMPT and prompt is None:
        raise ValueError(
            f"`prompt` parameter required to be set for task `{task_type}`"
        )
    if task_type in TASKS_REQUIRING_CLASSES and classes is None:
        raise ValueError(
            f"`classes` parameter required to be set for task `{task_type}`"
        )
    if task_type in TASKS_REQUIRING_OUTPUT_STRUCTURE and output_structure is None:
        raise ValueError(
            f"`output_structure` parameter required to be set for task `{task_type}`"
        )

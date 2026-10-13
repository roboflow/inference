"""Vendor-named vision models served over OpenAI-compatible APIs.

Ports of V1 ``roboflow_core/google_gemma@v4``,
``roboflow_core/qwen3_5_openrouter@v1``, ``roboflow_core/qwen3_6_openrouter@v1``
and ``roboflow_core/spacexai@v3``. Each block fixes one vendor's model table
and prompts; the requests reuse the shared chat helpers::

    google_gemma@v4 (OpenRouterBlockBase, the request loop of openrouter@v3)
      model label ──▶ OpenRouter slug; openrouter@v3 task messages
      api_key rf_key:* ─▶ platform proxy;  sk-or-.. ─▶ OpenRouter directly
      answer ──▶ decode_vlm_output(named_normalized) ──▶ predictions

    qwen3_5_openrouter@v1, qwen3_6_openrouter@v1 (own OpenRouter key only)
      model label ──▶ OpenRouter slug; V1 Qwen messages
      ──▶ chat.completions.create(model, messages, max_tokens, temperature)
      raw answer only; no provider routing, quantization filter or reasoning

    spacexai@v3 (xAI Responses API)
      task ──▶ JPEG, detail "auto"
      object-detection ──▶ lossless PNG at full size, detail "high", image first
      api_key rf_key:* ─▶ platform_client.post("apiproxy/xai", payload)
      xAI key          ─▶ OpenAI(base_url=xai_base_url).responses.create(store=False)
      a direct error naming the reasoning config ─▶ retried once without it, as V1
      answer ──▶ decode_vlm_output(xyxy_percent) ──▶ predictions

Differences from V1:

* A failed request raises ``ChatRequestError`` with the keys scrubbed and no
  chained exception. V1 raised the vendor's exception, whose text may contain
  the key.
* ``max_concurrent_requests`` must be at least 1; unset means 8 instead of an
  environment variable. After a failure, requests not yet started are skipped.
* ``inference_id`` is a hex UUID, as in every V2 block; V1 used the dashed form.
* Predictions are native V2 carriers; see ``decoding``.
* Qwen: a Roboflow-managed ``rf_key:`` key fails before any request (V1 sent
  it to OpenRouter as a bearer token). An answer without ``message.content``
  fails with the OpenRouter message instead of returning ``None`` as output.
"""

import base64
import json
import logging
import uuid
from functools import partial
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional

import cv2
import httpx
import numpy as np
from openai import DefaultHttpxClient, OpenAI
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.language.chat import (
    DEFAULT_MANAGED_KEY,
    REASONING_EFFORT_OPTIONS,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    ChatAnswer,
    ChatKeys,
    ChatPrompt,
    ChatRequestError,
    EncodedImage,
    check_reasoning_effort,
    check_task_fields,
    encode_image,
    reveal_keys,
    run_concurrently,
    scrubbed_call,
    third_party_model,
    token_outputs,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    build_object_detection_prompt,
    decode_vlm_output,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.blocks.language.openai import (
    CAPTION_INSTRUCTIONS,
    CLASS_LIST_TEXT,
    CLASSIFICATION_INSTRUCTIONS,
    MULTI_LABEL_INSTRUCTIONS,
    OCR_INSTRUCTIONS,
    STRUCTURED_ANSWERING_INSTRUCTIONS,
    VQA_INSTRUCTIONS,
    chat_messages,
    read_responses_answer,
    responses_fields,
)
from roboflow_workflows.v2.blocks.language.openrouter import (
    DETECTION_TASK,
    LEGACY_DETECTION_BOX_FORMAT,
    OPENROUTER_API_URL,
    OPENROUTER_TASKS,
    PRIVACY_LEVEL_METADATA,
    OpenRouterBlockBase,
    OpenRouterTask,
    PrivacyLevel,
    openrouter_task_messages,
    read_chat_completion,
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

GEMMA_MODELS = {
    "Gemma 4 31B - OpenRouter": "google/gemma-4-31b-it",
    "Gemma 4 26B A4B - OpenRouter": "google/gemma-4-26b-a4b-it",
}
"""V1 google_gemma@v4 model labels and their OpenRouter slugs."""

QWEN3_5_MODELS = {
    "Qwen 3.5 9B - OpenRouter": "qwen/qwen3.5-9b",
    "Qwen 3.5 27B - OpenRouter": "qwen/qwen3.5-27b",
    "Qwen 3.5 122B A10B - OpenRouter": "qwen/qwen3.5-122b-a10b",
    "Qwen 3.5 397B A17B - OpenRouter": "qwen/qwen3.5-397b-a17b",
    "Qwen 3.5 Flash 02-23 - OpenRouter": "qwen/qwen3.5-flash-02-23",
    "Qwen 3.5 Plus 20260420 - OpenRouter": "qwen/qwen3.5-plus-20260420",
}
QWEN3_6_MODELS = {
    "Qwen 3.6 27B - OpenRouter": "qwen/qwen3.6-27b",
    "Qwen 3.6 35B A3B - OpenRouter": "qwen/qwen3.6-35b-a3b",
    "Qwen 3.6 Flash - OpenRouter": "qwen/qwen3.6-flash",
    "Qwen 3.6 Plus - OpenRouter": "qwen/qwen3.6-plus",
    "Qwen 3.6 Max Preview - OpenRouter": "qwen/qwen3.6-max-preview",
}

QWEN_VQA_SYSTEM_PROMPT = (
    "You act as Visual Question Answering model. Your task is to provide answer to "
    "questionsubmitted by user. If this is open-question - answer with few "
    "sentences, for ABCD question, return only the indicator of the answer."
)
"""V1 Qwen's VQA prompt; its missing space ("questionsubmitted") is V1's text."""

QWEN_RECOMMENDED_PARSERS = {
    "structured-answering": "v2/json_parser@v1",
    "classification": "v2/vlm_as_classifier@v2",
    "multi-label-classification": "v2/vlm_as_classifier@v2",
}
"""V1's recommendations without vlm_as_detector: its "qwen" format reads 0-1000
``box_2d`` boxes, while this block asks for 0-1 ``x_min``/``y_min`` boxes."""

XAI_PROVIDER = "xAI"
XAI_API_URL = "https://api.x.ai/v1"
XAI_PROXY_ENDPOINT = "apiproxy/xai"
XAI_DETECTION_BOX_FORMAT = "xyxy_percent"
XAI_PNG_COMPRESSION = 9
"""V1's lossless PNG level; OpenCV's default (1) put large frames over xAI's
25 MB upload limit."""

GROK_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
GROK_MODELS = {
    "grok-4.7": ("Grok 4.7", ("low", "medium", "high", "xhigh")),
    "grok-4.6": ("Grok 4.6", ("low", "medium", "high", "xhigh")),
    "grok-4.5": ("Grok 4.5", ("low", "medium", "high")),
}
"""V1 model id -> (name, reasoning levels); xAI downgrades grok-4.5 ``xhigh``."""
GROK_REASONING_LEVELS = {model: levels for model, (_, levels) in GROK_MODELS.items()}

_XAI_REASONING_REJECTION_MARKERS = (
    "unsupported",
    "not supported",
    "unknown",
    "invalid",
)
"""V1 spacexai@v3: an error naming ``reasoning`` and one of these is retried."""


def model_slug(label: str, *, models: Mapping[str, str]) -> str:
    """Return the provider slug of a model label, as V1 maps it.

    Args:
        label: The label chosen in ``model_version``.
        models: Label to slug table of the block.

    Returns:
        The slug.

    Raises:
        ValueError: With V1 Qwen's message when the label is unknown.
    """
    slug = models.get(label)
    if slug is None:
        raise ValueError(
            f"Invalid model name: '{label}'. Please use one of {list(models)}."
        )

    return slug


def _model_resource(
    model_version: Any, *, models: Mapping[str, str]
) -> DependentResource:
    model_id = models.get(model_version, model_version)
    resource = third_party_model(
        model_id, provider="openrouter", resource="openrouter_http_client"
    )

    return resource


class GoogleGemmaBlock(OpenRouterBlockBase):
    """Ask Google's Gemma 4 about each image through OpenRouter.

    Example: ``task_type="object-detection"`` with ``classes=["car"]`` returns
    the raw ``output`` and ``predictions`` as detections in image pixels.
    ``api_key="rf_key:account"`` (the default) bills the host's Roboflow
    account through its platform client; an ``sk-or-...`` key or a secret
    calls OpenRouter directly.
    """

    type = "v2/google_gemma@v4"
    metadata = {
        "name": "Google Gemma",
        "port_of": "roboflow_core/google_gemma@v4",
        "section": "model",
        "license": "Gemma Terms of Use",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "recommended_parsers": {"structured-answering": "v2/json_parser@v1"},
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
        **token_outputs(source="images"),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to ask about; one request per image."
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
        model_version: Literal[tuple(GEMMA_MODELS)] | Ref(STRING_KIND) = Field(
            default="Gemma 4 31B - OpenRouter",
            description="Gemma model served by OpenRouter.",
            examples=["Gemma 4 31B - OpenRouter", "$inputs.gemma_model"],
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
        temperature: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=0.1,
            ge=0.0,
            le=2.0,
            description="Sampling temperature.",
            examples=[0.1],
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
        def _task_fields_present(self) -> "GoogleGemmaBlock.Params":
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
        resources = [_model_resource(params.model_version, models=GEMMA_MODELS)]

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
        model_version: str,
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
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            model_version: A ``GEMMA_MODELS`` label.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_tokens: Output token limit.
            temperature: Sampling temperature.
            reasoning_effort: Reasoning level or ``None``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes", "input_tokens", "output_tokens",
            "predictions", "error_status", "inference_id"}`` per image.

        Raises:
            ValueError: On an unknown model label, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        model = model_slug(model_version, models=GEMMA_MODELS)
        prompts = [
            openrouter_task_messages(
                task_type,
                image=image,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
                detection_format=LEGACY_DETECTION_BOX_FORMAT,
            )
            for image in images
        ]
        answers = self.ask_all(
            prompts,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            reasoning_effort=reasoning_effort,
            max_concurrent_requests=max_concurrent_requests,
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
                box_format=(
                    LEGACY_DETECTION_BOX_FORMAT if task_type == DETECTION_TASK else None
                ),
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


def qwen_task_messages(
    task_type: str,
    *,
    image: ImageData,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Build V1 Qwen's chat messages of one task; the image goes as JPEG, full size.

    V1 Qwen's prompts are the OpenRouter ones except the VQA system prompt.

    Args:
        task_type: One of ``OPENROUTER_TASKS``.
        image: The image to ask about.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        The ``messages`` list.

    Raises:
        ValueError: On an unknown task.
    """
    if task_type == "visual-question-answering":
        upload = encode_image(image)
        chat_prompt = ChatPrompt(QWEN_VQA_SYSTEM_PROMPT, f"Question: {prompt}")
        messages = chat_messages(chat_prompt, [upload.data_url])
        return messages

    messages = openrouter_task_messages(
        task_type,
        image=image,
        prompt=prompt,
        output_structure=output_structure,
        classes=classes,
        detection_format=LEGACY_DETECTION_BOX_FORMAT,
    )

    return messages


class _QwenParams(BlockParams):
    images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
        description="Images to ask about; one request per image."
    )
    task_type: OpenRouterTask = Field(
        default="unconstrained",
        description="What to ask: a free prompt, OCR, captions, classification, structured answers or detection.",
        examples=["structured-answering"],
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
    classes: Optional[List[StrictStr]] | Ref(LIST_OF_VALUES_KIND, allow_none=True) = (
        Field(
            default=None,
            description="Class names for classification and detection.",
            examples=[["class-a", "class-b"], "$inputs.classes"],
        )
    )
    api_key: ApiKey = Field(
        description="Your OpenRouter API key, as a secret or string.",
        examples=["$inputs.open_router_api_key"],
    )
    max_tokens: StrictInt = Field(
        default=500,
        gt=1,
        description="Most output tokens.",
        examples=[500],
    )
    temperature: StrictFloat | Ref(FLOAT_KIND) = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
        examples=[0.1],
    )
    max_concurrent_requests: Optional[StrictInt] = Field(
        default=None,
        ge=1,
        description="Most requests in flight for one batch; None means 8.",
        examples=[4],
    )

    @model_validator(mode="after")
    def _task_fields_present(self) -> "_QwenParams":
        check_task_fields(
            self.task_type,
            prompt=self.prompt,
            classes=self.classes,
            output_structure=self.output_structure,
        )

        return self


class QwenOpenRouterBlockBase(Block):
    """Qwen vision models on OpenRouter with the caller's own key, as V1 Qwen.

    Subclasses set ``type``, ``models``, ``metadata`` and a ``Params`` adding
    ``model_version``. The request carries only ``model``, ``messages``,
    ``max_tokens`` and ``temperature``; use ``v2/openrouter@v3`` for managed
    keys, privacy routing or reasoning settings.
    """

    models: ClassVar[Mapping[str, str]] = {}
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
    }

    def __init__(
        self,
        *,
        openrouter_base_url: str = OPENROUTER_API_URL,
        openrouter_http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = openrouter_base_url
        self._http_client = openrouter_http_client or DefaultHttpxClient()

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the OpenRouter model the step calls."""
        resources = [_model_resource(params.model_version, models=cls.models)]

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
        max_tokens: int,
        temperature: float,
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Send one request per image; return the raw answers.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key as a string or secret.
            model_version: A label of ``models``.
            max_tokens: Output token limit.
            temperature: Sampling temperature.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes"}`` per image.

        Raises:
            ValueError: On an unknown model label, a missing provider key or
                a Roboflow-managed key, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails.
        """
        model = model_slug(model_version, models=self.models)
        keys = reveal_keys(api_key, roboflow_api_key=None)
        if keys.provider_key is None:
            raise ValueError(f"{self.type} requires an OpenRouter API key.")

        if keys.managed:
            raise ValueError(
                f"{self.type} calls OpenRouter with your own key; it cannot use a "
                "Roboflow-managed 'rf_key:' key. Use v2/openrouter@v3 for that."
            )

        prompts = [
            qwen_task_messages(
                task_type,
                image=image,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
            for image in images
        ]
        request = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        calls = [partial(self._ask, keys, messages, request) for messages in prompts]
        answers = run_concurrently(
            calls, max_concurrent_requests=max_concurrent_requests
        )

        results = [{"output": answer.text, "classes": classes} for answer in answers]

        return results

    def _ask(
        self, keys: ChatKeys, messages: List[Dict[str, Any]], request: Dict[str, Any]
    ) -> ChatAnswer:
        answer = scrubbed_call(
            partial(self._request, keys, messages, request),
            keys=keys,
            provider="OpenRouter",
        )

        return answer

    def _request(
        self, keys: ChatKeys, messages: List[Dict[str, Any]], request: Dict[str, Any]
    ) -> ChatAnswer:
        client = OpenAI(
            base_url=self._base_url,
            api_key=keys.provider_key,
            http_client=self._http_client,
        )
        data = client.chat.completions.create(messages=messages, **request).to_dict()
        answer = read_chat_completion(data, via_proxy=False)

        return answer


def _qwen_metadata(name: str, *, port_of: str) -> Dict[str, Any]:
    metadata = {
        "name": name,
        "port_of": port_of,
        "section": "model",
        "license": "Qwen License",
        "deprecated": True,
        "deprecation_message": (
            "Use v2/openrouter@v3 with a Qwen model_id for V2 workflows, "
            "including decoded detections, managed keys, privacy routing and "
            "reasoning. V1-only replacement: roboflow_core/qwen_vlm@v1."
        ),
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "recommended_parsers": QWEN_RECOMMENDED_PARSERS,
    }

    return metadata


class Qwen35OpenRouterBlock(QwenOpenRouterBlockBase):
    """Ask a Qwen 3.5 vision model on OpenRouter about each image.

    Object detection returns raw normalized 0-1 ``x_min``/``y_min``/``x_max``/
    ``y_max`` fields in ``output``. There is no matching supported Qwen
    ``model_type`` in ``v2/vlm_as_detector@v2``; its ``qwen`` format expects
    0-1000 ``box_2d`` fields. Use ``v2/openrouter@v3`` with
    ``model_id="qwen/qwen3.5-27b"`` for native decoded ``predictions``.

    Example: ``task_type="structured-answering"`` with
    ``output_structure={"count": "number of cars"}`` and
    ``api_key="$inputs.openrouter_key"`` returns the raw JSON answer, which
    ``v2/json_parser@v1`` turns into fields.
    """

    type = "v2/qwen3_5_openrouter@v1"
    models = QWEN3_5_MODELS
    metadata = _qwen_metadata(
        "Qwen 3.5 API", port_of="roboflow_core/qwen3_5_openrouter@v1"
    )

    class Params(_QwenParams):
        model_version: Literal[tuple(QWEN3_5_MODELS)] | Ref(STRING_KIND) = Field(
            default="Qwen 3.5 27B - OpenRouter",
            description="Qwen 3.5 model served by OpenRouter.",
            examples=["Qwen 3.5 27B - OpenRouter", "$inputs.qwen_model"],
        )


class Qwen36OpenRouterBlock(QwenOpenRouterBlockBase):
    """Ask a Qwen 3.6 vision model on OpenRouter about each image.

    Object detection returns raw normalized 0-1 ``x_min``/``y_min``/``x_max``/
    ``y_max`` fields in ``output``. There is no matching supported Qwen
    ``model_type`` in ``v2/vlm_as_detector@v2``; its ``qwen`` format expects
    0-1000 ``box_2d`` fields. Use ``v2/openrouter@v3`` with
    ``model_id="qwen/qwen3.6-35b-a3b"`` for native decoded ``predictions``.

    Example: ``task_type="classification"`` with ``classes=["cat", "dog"]``
    returns the raw JSON answer, which ``v2/vlm_as_classifier@v2`` decodes.
    """

    type = "v2/qwen3_6_openrouter@v1"
    models = QWEN3_6_MODELS
    metadata = _qwen_metadata(
        "Qwen 3.6 API", port_of="roboflow_core/qwen3_6_openrouter@v1"
    )

    class Params(_QwenParams):
        model_version: Literal[tuple(QWEN3_6_MODELS)] | Ref(STRING_KIND) = Field(
            default="Qwen 3.6 35B A3B - OpenRouter",
            description="Qwen 3.6 model served by OpenRouter.",
            examples=["Qwen 3.6 35B A3B - OpenRouter", "$inputs.qwen_model"],
        )


def xai_task_request(
    task_type: str,
    *,
    image: ImageData,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> Dict[str, Any]:
    """Build V1 spacexai@v3's Responses API ``input`` and ``instructions``.

    Object detection uploads a lossless PNG at full size with ``detail``
    ``"high"`` and the image before the percent-box prompt. Every other task
    uploads a JPEG with ``detail`` ``"auto"`` and V1's OpenAI instructions.

    Args:
        task_type: One of ``OPENROUTER_TASKS`` (the same nine tasks).
        image: The image to ask about.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        ``{"input": [...]}``, with ``"instructions"`` when the task has them.

    Raises:
        ValueError: On an unknown task.
    """
    if task_type == DETECTION_TASK:
        text = build_object_detection_prompt(XAI_DETECTION_BOX_FORMAT, classes=classes)
        chat_prompt = ChatPrompt(None, text, image_first=True)
        fields = responses_fields(chat_prompt, lossless_png(image), detail="high")
        return fields

    class_text = CLASS_LIST_TEXT.format(class_list=", ".join(classes or []))
    prompts = {
        "unconstrained": lambda: ChatPrompt(None, prompt),
        "ocr": lambda: ChatPrompt(OCR_INSTRUCTIONS, None),
        "visual-question-answering": lambda: ChatPrompt(
            VQA_INSTRUCTIONS, f"Question: {prompt}"
        ),
        "caption": lambda: ChatPrompt(
            CAPTION_INSTRUCTIONS.format(length="short"), None
        ),
        "detailed-caption": lambda: ChatPrompt(
            CAPTION_INSTRUCTIONS.format(length="extensive"), None
        ),
        "classification": lambda: ChatPrompt(CLASSIFICATION_INSTRUCTIONS, class_text),
        "multi-label-classification": lambda: ChatPrompt(
            MULTI_LABEL_INSTRUCTIONS, class_text
        ),
        "structured-answering": lambda: ChatPrompt(
            STRUCTURED_ANSWERING_INSTRUCTIONS,
            "Specification of requirements regarding output fields: \n"
            + json.dumps(output_structure, indent=4),
        ),
    }
    if task_type not in prompts:
        raise ValueError(f"Task type: {task_type} not supported.")

    fields = responses_fields(prompts[task_type](), encode_image(image), detail="auto")

    return fields


def lossless_png(image: ImageData) -> EncodedImage:
    """Encode an image as V1 spacexai@v3 uploads detections: PNG, level 9.

    ``chat.encode_image`` writes PNG at OpenCV's default level, which V1
    spacexai@v3 avoided; the pixels are the same, the file is smaller.

    Args:
        image: RGB or single-channel image.

    Returns:
        The encoded image at its own size.

    Raises:
        ValueError: When OpenCV cannot encode the image.
    """
    pixels = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    if pixels.shape[2] == 3:
        pixels = cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_RGB2BGR)
    else:
        pixels = np.ascontiguousarray(pixels[:, :, 0])
    encoded, data = cv2.imencode(
        ".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, XAI_PNG_COMPRESSION]
    )
    if not encoded:
        raise ValueError(f"OpenCV could not encode an image of shape {pixels.shape}")

    height, width = pixels.shape[:2]
    upload = EncodedImage(
        base64=base64.b64encode(data.tobytes()).decode("ascii"),
        media_type="image/png",
        size_wh=(width, height),
    )

    return upload


def xai_options(
    model: str,
    *,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> Dict[str, Any]:
    """Return ``model``, ``store`` and the set generation options of a request.

    Args:
        model: Grok model id.
        reasoning_effort: Reasoning level or ``None``.
        max_tokens: Output token limit or ``None``.
        temperature: Sampling temperature or ``None``.

    Returns:
        The request options; ``store`` is always ``False``, as in V1.

    Raises:
        ValueError: When a known model does not support ``reasoning_effort``.
    """
    check_reasoning_effort(
        model, reasoning_effort, levels_by_model=GROK_REASONING_LEVELS
    )
    options: Dict[str, Any] = {"model": model, "store": False}
    if max_tokens is not None:
        options["max_output_tokens"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if reasoning_effort is not None:
        options["reasoning"] = {"effort": reasoning_effort}

    return options


def read_xai_answer(data: Dict[str, Any]) -> ChatAnswer:
    """Read an xAI Responses API result with V1 spacexai@v3's messages.

    V1's xAI reader is V1's OpenAI reader with "xAI API" in its messages, so
    this reuses ``read_responses_answer`` and renames the API.

    Args:
        data: The response as a JSON dictionary.

    Returns:
        The concatenated ``output_text`` parts and token usage.

    Raises:
        ChatRequestError: On a failed, cancelled, incomplete or unexpected
            status, or an answer without text.
    """
    try:
        answer = read_responses_answer(data)
    except ChatRequestError as error:
        message = str(error).replace("OpenAI API", "xAI API", 1)
    else:
        return answer

    raise ChatRequestError(message)


def rejects_reasoning(error: Exception) -> bool:
    """Whether an xAI error rejected the request's reasoning config (V1 rule).

    Args:
        error: The direct request's exception.

    Returns:
        ``True`` when the message names ``reasoning`` and a rejection marker.
    """
    message = str(error).lower()
    rejected = "reasoning" in message and any(
        marker in message for marker in _XAI_REASONING_REJECTION_MARKERS
    )

    return rejected


class SpaceXAIBlock(Block):
    """Ask a SpaceXAI Grok model about each image through xAI's Responses API.

    Example: ``task_type="object-detection"`` with ``classes=["car"]`` sends
    the full-size image as lossless PNG and returns ``predictions`` as
    detections in image pixels. ``api_key="rf_key:account"`` (the default)
    bills the host's Roboflow account through its platform client; an xAI key
    or a secret calls xAI directly.
    """

    type = "v2/spacexai@v3"
    metadata = {
        "name": "SpaceXAI",
        "port_of": "roboflow_core/spacexai@v3",
        "section": "model",
        "license": "Apache-2.0",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "model_metadata": {
            model: {"name": name, "reasoning_levels": list(levels)}
            for model, (name, levels) in GROK_MODELS.items()
        },
        "recommended_parsers": {"structured-answering": "v2/json_parser@v1"},
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
        **token_outputs(source="images"),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to ask about; one request per image."
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
        api_key: ApiKey = Field(
            default=DEFAULT_MANAGED_KEY,
            description=(
                "xAI key as a secret or string, or 'rf_key:account' / "
                "'rf_key:user:ID' to use a key stored with Roboflow."
            ),
            examples=["rf_key:account", "$inputs.xai_api_key"],
        )
        model_version: Literal[tuple(GROK_MODELS)] | Ref(STRING_KIND) = Field(
            default="grok-4.6",
            description="Grok model.",
            examples=["grok-4.6", "$inputs.grok_model"],
        )
        reasoning_effort: Optional[Literal[GROK_REASONING_EFFORTS]] | Ref(
            STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Sent as xAI `reasoning: {effort}`. Grok defaults to 'high' and "
                "cannot disable reasoning; grok-4.5 has no 'xhigh'. With a direct "
                "key, a model rejecting it is retried without it."
            ),
            examples=["low"],
        )
        max_tokens: Optional[StrictInt] = Field(
            default=None,
            ge=16,
            description="Most output tokens; None keeps xAI's limit.",
            examples=[2048],
        )
        temperature: Optional[StrictFloat] | Ref(FLOAT_KIND, allow_none=True) = Field(
            default=None,
            ge=0.0,
            le=2.0,
            description="Sampling temperature; None keeps xAI's default.",
            examples=[0.2],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most requests in flight for one batch; None means 8.",
            examples=[4],
        )

        @model_validator(mode="after")
        def _task_fields_present(self) -> "SpaceXAIBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )
            check_reasoning_effort(
                self.model_version,
                self.reasoning_effort,
                levels_by_model=GROK_REASONING_LEVELS,
            )

            return self

    def __init__(
        self,
        *,
        platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT,
        api_key: Any = None,
        xai_base_url: str = XAI_API_URL,
        xai_http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._platform_client = platform_client
        self._roboflow_api_key = api_key
        self._base_url = xai_base_url
        self._http_client = xai_http_client or DefaultHttpxClient()

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
        """Report the Grok model the step calls."""
        resources = [
            third_party_model(
                params.model_version, provider="xai", resource="xai_http_client"
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
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Send one request per image and decode each answer.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: xAI key, secret or managed ``rf_key:`` reference.
            model_version: Grok model id.
            reasoning_effort: Reasoning level or ``None``.
            max_tokens: Output token limit or ``None``.
            temperature: Sampling temperature or ``None``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes", "input_tokens", "output_tokens",
            "predictions", "error_status", "inference_id"}`` per image.

        Raises:
            ValueError: On a missing provider key, a managed key without the
                host's Roboflow ``api_key`` resource, or a reasoning level the
                model does not support, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        options = xai_options(
            model_version,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        keys = reveal_keys(api_key, roboflow_api_key=self._roboflow_api_key)
        if keys.provider_key is None:
            raise ValueError(f"{self.type} requires an xAI API key.")

        if keys.managed and not keys.roboflow_key:
            raise ValueError(
                "Roboflow API key is required when using a Roboflow-managed xAI "
                "API key; the host must provide the 'api_key' resource."
            )

        requests = [
            xai_task_request(
                task_type,
                image=image,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
            for image in images
        ]
        calls = [
            partial(self._ask, keys, {**options, **request}) for request in requests
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
                box_format=(
                    XAI_DETECTION_BOX_FORMAT if task_type == DETECTION_TASK else None
                ),
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

    def _ask(self, keys: ChatKeys, payload: Dict[str, Any]) -> ChatAnswer:
        answer = scrubbed_call(
            partial(self._request, keys, payload), keys=keys, provider=XAI_PROVIDER
        )

        return answer

    def _request(self, keys: ChatKeys, payload: Dict[str, Any]) -> ChatAnswer:
        if keys.managed:
            data = self._platform_client.post(
                endpoint=XAI_PROXY_ENDPOINT,
                api_key=keys.roboflow_key,
                payload={**payload, "xai_api_key": keys.provider_key},
            )
        else:
            data = self._send_direct(keys, payload)
        answer = read_xai_answer(data)

        return answer

    def _send_direct(self, keys: ChatKeys, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = OpenAI(
            base_url=self._base_url,
            api_key=keys.provider_key,
            http_client=self._http_client,
        )
        try:
            response = client.responses.create(**payload)
        except Exception as error:
            if "reasoning" not in payload or not rejects_reasoning(error):
                raise
            logger.warning(
                "xAI rejected the reasoning config %s for model %s (details: %s). "
                "Retrying without it - the model will use its default effort.",
                payload["reasoning"],
                payload["model"],
                keys.scrub(str(error)),
            )
            without_reasoning = {
                name: value for name, value in payload.items() if name != "reasoning"
            }
            response = client.responses.create(**without_reasoning)
        data = response.to_dict()

        return data


BLOCKS = (
    GoogleGemmaBlock,
    Qwen35OpenRouterBlock,
    Qwen36OpenRouterBlock,
    SpaceXAIBlock,
)

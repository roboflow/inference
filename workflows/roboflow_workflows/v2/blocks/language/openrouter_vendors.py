"""Active OpenRouter vendors: Kimi, Meta Muse and Z.ai GLM.

LlamaVisionBlock is retired and excluded from BLOCKS. Its implementation and
shared helpers remain for existing internal references, without a drop-in promise.

Historical implementation context: vendor model blocks served through OpenRouter: Kimi, Meta Muse, Z.ai GLM, Llama.

Ports of V1 ``roboflow_core/kimi_openrouter@v3``, ``roboflow_core/meta_vlm@v3``,
``roboflow_core/zai_vlm@v2`` and ``roboflow_core/llama_vision@v2``. Each block
picks a model by its V1 label and sends OpenRouter chat completions through
``OpenRouterBlockBase``, which owns keys, privacy routing, quantization
filters, the reasoning retry and the Roboflow proxy::

    model_version label ──▶ VendorModel (slug, box format, reasoning rules)
    per image:
      kimi, llama ─▶ openrouter_task_messages      (V1 system + text + image)
      meta, zai   ─▶ image_first_task_messages     (one user message: image, text)
                     image as JPEG; shrunk 10% at a time over 9.5 MB base64
    OpenRouterBlockBase.ask_all ─▶ answers in image order
    kimi, meta, zai: answer ─▶ decode_vlm_output(model box format) ─▶ predictions
    llama: raw answer only (V1 v2 decodes nothing)

Reasoning, as V1: Kimi and Llama send none. Meta always sends an effort
(Muse rejects disabled reasoning). Z.ai disables reasoning by default; a
model that requires reasoning gets ``low`` instead of ``none``.

Differences from V1:

* A failed request raises ``ChatRequestError`` with the keys scrubbed.
* ``max_concurrent_requests`` must be at least 1; unset means 8 instead of an
  environment variable.
* Predictions are native V2 carriers; see ``decoding``.
* Every check runs again on the resolved values before any request: a key
  that resolved to ``None`` (V1 failed on it too; the shared OpenRouter base
  would send the ``OPENAI_API_KEY`` environment variable instead), an
  unknown model label (V1 Kimi and Llama raised ``KeyError``), a reasoning
  level the model rejects, a task field that resolved to ``None``, and
  invalid or duplicate ``classes`` of a decoding task (V1 sent the request
  first).
* A ``model_version`` selector is reported to workload introspection as the
  selector; V1 attached a label-to-slug resolver.
"""

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import cv2
import torch
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.language.chat import (
    DEFAULT_MANAGED_KEY,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    ChatAnswer,
    EncodedImage,
    check_reasoning_effort,
    check_task_fields,
    encode_image,
    third_party_model,
    token_outputs,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    CLASSIFICATION_TASKS,
    DETECTION_TASKS,
    build_object_detection_prompt,
    checked_classes,
    decode_vlm_output,
    detection_box_format,
    max_edge_upload_size,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.blocks.language.openrouter import (
    CAPTION_SYSTEM_PROMPT,
    CLASS_LIST_TEXT,
    CLASSIFICATION_SYSTEM_PROMPT,
    LEGACY_DETECTION_BOX_FORMAT,
    MULTI_LABEL_SYSTEM_PROMPT,
    OCR_SYSTEM_PROMPT,
    OPENROUTER_TASKS,
    PRIVACY_LEVEL_METADATA,
    STRUCTURED_ANSWERING_SYSTEM_PROMPT,
    VQA_SYSTEM_PROMPT,
    OpenRouterBlockBase,
    OpenRouterTask,
    PrivacyLevel,
    openrouter_task_messages,
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
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

OPENROUTER_MAX_BASE64_BYTES = 9_500_000
"""Largest base64 image V1 Meta and Z.ai upload; larger images are shrunk."""

SHRINK_FACTOR = 0.9
"""Longest-edge factor of each V1 shrink step."""

DECODING_TASKS = CLASSIFICATION_TASKS + DETECTION_TASKS

ZAI_FLASH_DETECTION_PROMPT = (
    "Detect all objects in this image and return their locations in the "
    "form of coordinates. The format of output should be like "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}}. '
    "bbox_2d is [xmin, ymin, xmax, ymax] as integers between 0 and 1000, "
    "normalized to image width and height. "
    "Only use these labels: {class_list}. Return a JSON array only."
)
"""V1 GLM 5.3 Flash detection prompt; formatted with ``{class_list}``."""


@dataclass(frozen=True)
class VendorModel:
    """One selectable model of a vendor block.

    Args:
        slug: OpenRouter model slug.
        box_format: Box format detection answers are decoded with; must be
            normalized (no upload size).
        reasoning_levels: Accepted ``reasoning_effort`` values; empty when the
            block has no reasoning setting.
        detection_prompt: Template overriding the box format's own detection
            prompt; formatted with ``{class_list}``.
        reasoning_required: Whether the model rejects disabled reasoning, so
            ``none`` is sent as ``low``.

    Raises:
        ValueError: On an unknown box format or one needing the upload size.
    """

    slug: str
    box_format: str = LEGACY_DETECTION_BOX_FORMAT
    reasoning_levels: Tuple[str, ...] = ()
    detection_prompt: Optional[str] = None
    reasoning_required: bool = False

    def __post_init__(self) -> None:
        if detection_box_format(self.box_format).requires_upload_size:
            raise ValueError(
                f"Box format {self.box_format!r} needs the upload size; vendor "
                "blocks decode normalized formats only"
            )

    def detection_text(self, classes: Sequence[str]) -> str:
        """Render this model's object-detection prompt.

        Args:
            classes: Class names the model may answer with.

        Returns:
            The prompt text.
        """
        if self.detection_prompt is not None:
            text = self.detection_prompt.format(class_list=", ".join(classes))
            return text

        text = build_object_detection_prompt(self.box_format, classes=classes)

        return text


KIMI_MODELS: Dict[str, VendorModel] = {
    "Kimi K2.5 - OpenRouter": VendorModel("moonshotai/kimi-k2.5"),
    "Kimi K2.6 - OpenRouter": VendorModel("moonshotai/kimi-k2.6"),
}
LLAMA_MODELS: Dict[str, VendorModel] = {
    "11B - OpenRouter": VendorModel("meta-llama/llama-3.2-11b-vision-instruct"),
}
"""Deprecated in V1: OpenRouter has no live Llama 3.2 Vision endpoint left."""

MUSE_SPARK_LEVELS = ("minimal", "low", "medium", "high", "xhigh")
META_MODELS: Dict[str, VendorModel] = {
    "Muse Spark 1.1": VendorModel(
        "meta/muse-spark-1.1",
        box_format="named_0_1000",
        reasoning_levels=MUSE_SPARK_LEVELS,
    ),
    "Muse Spark 1.2": VendorModel(
        "meta/muse-spark-1.2",
        box_format="named_0_1000",
        reasoning_levels=MUSE_SPARK_LEVELS,
    ),
    "Muse Spark 1.3": VendorModel(
        "meta/muse-spark-1.3",
        box_format="named_0_1000",
        reasoning_levels=MUSE_SPARK_LEVELS,
    ),
    "Muse Glimmer": VendorModel(
        "meta/muse-glimmer-30b",
        box_format="named_0_1000",
        reasoning_levels=("low", "medium", "high", "xhigh"),
    ),
}

ZAI_LEVELS = ("none", "low", "medium", "high")
ZAI_MODELS: Dict[str, VendorModel] = {
    "GLM 5V Turbo": VendorModel(
        "z-ai/glm-5v-turbo", box_format="xyxy_0_1000", reasoning_levels=ZAI_LEVELS
    ),
    # Served earlier as the OpenRouter stealth model "Ox Alpha" (V1 note).
    "GLM 5.3 Flash": VendorModel(
        "z-ai/glm-5.3-flash",
        box_format="xyxy_0_1000",
        reasoning_levels=ZAI_LEVELS,
        detection_prompt=ZAI_FLASH_DETECTION_PROMPT,
        reasoning_required=True,
    ),
}
ZAI_REQUIRED_REASONING_FALLBACK = "low"

KimiModelVersion = Literal[tuple(KIMI_MODELS)]
LlamaModelVersion = Literal[tuple(LLAMA_MODELS)]
MetaModelVersion = Literal[tuple(META_MODELS)]
MetaReasoningEffort = Literal[MUSE_SPARK_LEVELS]
ZaiModelVersion = Literal[tuple(ZAI_MODELS)]
ZaiReasoningEffort = Literal[ZAI_LEVELS]

MUSE_REASONING_METADATA = {
    "minimal": {
        "name": "Minimal",
        "description": (
            "Shortest reasoning pass. Supported by Muse Spark models only — "
            "Muse Glimmer starts at low."
        ),
    },
    "low": {
        "name": "Low (recommended)",
        "description": (
            "Small reasoning budget. Muse models require reasoning; this is "
            "the lowest effort every variant accepts."
        ),
    },
    "medium": {
        "name": "Medium",
        "description": "Moderate reasoning budget before answering.",
    },
    "high": {
        "name": "High",
        "description": (
            "Large reasoning budget. Slow and expensive; consider raising "
            "`max_tokens` so reasoning does not crowd out the answer."
        ),
    },
    "xhigh": {
        "name": "Extra high",
        "description": (
            "Maximum reasoning depth. Slowest and most expensive; raise "
            "`max_tokens` accordingly."
        ),
    },
}
ZAI_REASONING_METADATA = {
    "none": {
        "name": "Disabled (recommended)",
        "description": (
            "Turns extended reasoning off. GLM models default to extended "
            "reasoning on OpenRouter, which bloats latency and can consume "
            "the whole token budget before a visible answer is produced. "
            "This is the configuration validated in the vlm-exam benchmarks. "
            "GLM 5.3 Flash requires reasoning and falls back to low effort "
            "instead."
        ),
    },
    "low": {
        "name": "Low",
        "description": "Small reasoning budget before answering.",
    },
    "medium": {
        "name": "Medium",
        "description": "Moderate reasoning budget before answering.",
    },
    "high": {
        "name": "High",
        "description": (
            "Large reasoning budget. Slowest and most expensive; consider "
            "raising `max_tokens` so reasoning does not crowd out the answer."
        ),
    },
}

DECODED_PARSERS = {"structured-answering": "v2/json_parser@v1"}
"""Parsers still worth chaining after a block that decodes in-block."""

LLAMA_PARSERS = {
    "structured-answering": "v2/json_parser@v1",
    "classification": "v2/vlm_as_classifier@v2",
    "multi-label-classification": "v2/vlm_as_classifier@v2",
}
"""Raw-answer parsers; no supported Llama model_type reads its detection format."""


def resolve_model(
    label: str, *, models: Mapping[str, VendorModel], family: str
) -> VendorModel:
    """Look up a model by its V1 label.

    Args:
        label: The resolved ``model_version``.
        models: The block's model table.
        family: Name in the error message, e.g. ``"Muse variant"``.

    Returns:
        The model.

    Raises:
        ValueError: When the label is not in ``models``.
    """
    model = models.get(label)
    if model is None:
        raise ValueError(f"Unknown {family} '{label}'. Pick one of: {list(models)}")

    return model


def encode_capped_jpeg(image: ImageData) -> EncodedImage:
    """Encode an image as JPEG, shrinking it until its base64 fits OpenRouter.

    V1 Meta and Z.ai rule: while the base64 text exceeds
    ``OPENROUTER_MAX_BASE64_BYTES``, the previous step's pixels are resized
    with ``INTER_AREA`` so the longest edge is 90% of what it was. Their box
    formats are normalized, so the upload size does not affect decoding.

    Args:
        image: The image to upload.

    Returns:
        The encoded upload with its pixel size.
    """
    upload = encode_image(image)
    if len(upload.base64) <= OPENROUTER_MAX_BASE64_BYTES:
        return upload

    pixels = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    while len(upload.base64) > OPENROUTER_MAX_BASE64_BYTES and max(upload.size_wh) > 1:
        longest_edge = max(int(max(upload.size_wh) * SHRINK_FACTOR), 1)
        size_wh = max_edge_upload_size(upload.size_wh, max_edge=longest_edge)
        pixels = cv2.resize(pixels, size_wh, interpolation=cv2.INTER_AREA)
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        shrunk = ImageData.from_tensor(torch.from_numpy(pixels).permute(2, 0, 1))
        upload = encode_image(shrunk)

    return upload


def image_first_task_messages(
    task_type: str,
    *,
    upload: EncodedImage,
    model: VendorModel,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Build V1 Meta/Z.ai messages: one user message, image first, no system.

    Args:
        task_type: One of ``OPENROUTER_TASKS``.
        upload: The encoded image.
        model: The selected model; gives the detection prompt.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        The ``messages`` list.

    Raises:
        ValueError: On an unknown task.
    """
    class_text = CLASS_LIST_TEXT.format(class_list=", ".join(classes or []))
    texts = {
        "unconstrained": lambda: prompt,
        "ocr": lambda: OCR_SYSTEM_PROMPT,
        "visual-question-answering": lambda: (
            f"{VQA_SYSTEM_PROMPT}\n\nQuestion: {prompt}"
        ),
        "caption": lambda: CAPTION_SYSTEM_PROMPT.format(length="short"),
        "detailed-caption": lambda: CAPTION_SYSTEM_PROMPT.format(length="extensive"),
        "classification": lambda: f"{CLASSIFICATION_SYSTEM_PROMPT}\n\n{class_text}",
        "multi-label-classification": lambda: (
            f"{MULTI_LABEL_SYSTEM_PROMPT}\n\n{class_text}"
        ),
        "structured-answering": lambda: (
            f"{STRUCTURED_ANSWERING_SYSTEM_PROMPT}\n\n"
            "Specification of requirements regarding output fields: \n"
            f"{json.dumps(output_structure, indent=4)}"
        ),
        "object-detection": lambda: model.detection_text(classes),
    }
    if task_type not in texts:
        raise ValueError(f"Task type: {task_type} not supported.")

    content = [
        {"type": "image_url", "image_url": {"url": upload.data_url}},
        {"type": "text", "text": texts[task_type]()},
    ]
    messages = [{"role": "user", "content": content}]

    return messages


def _raw_outputs(*, with_reasoning: bool) -> Dict[str, Output]:
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
    if with_reasoning:
        outputs["thinking"] = Output(
            STRING_KIND,
            source="images",
            description="Reasoning trace when OpenRouter returns one, else empty.",
        )
        outputs.update(token_outputs(source="images"))

    return outputs


class _VendorParams(BlockParams):
    """Fields every vendor block shares, with V1's task-field check."""

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
    classes: Optional[List[StrictStr]] | Ref(LIST_OF_VALUES_KIND, allow_none=True) = (
        Field(
            default=None,
            description="Class names for classification and detection.",
            examples=[["class-a", "class-b"], "$inputs.classes"],
        )
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
    max_concurrent_requests: Optional[StrictInt] = Field(
        default=None,
        ge=1,
        description="Most requests in flight for one batch; None means 8.",
        examples=[4],
    )

    @model_validator(mode="after")
    def _task_fields_present(self) -> "_VendorParams":
        check_task_fields(
            self.task_type,
            prompt=self.prompt,
            classes=self.classes,
            output_structure=self.output_structure,
        )

        return self


class _LegacyVendorParams(_VendorParams):
    """Kimi and Llama: V1 mixin defaults, no reasoning setting."""

    max_tokens: StrictInt = Field(
        default=500,
        gt=1,
        description="Most output tokens the model may generate.",
        examples=[500],
    )
    temperature: StrictFloat | Ref(FLOAT_KIND) = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
        examples=[0.1],
    )


class _ReasoningVendorParams(_VendorParams):
    """Meta and Z.ai: 2048 tokens, provider temperature, a reasoning level."""

    max_tokens: Optional[StrictInt] = Field(
        default=2048,
        gt=1,
        description=(
            "Most output tokens; None means 2048. Raise it (e.g. 8192) for long "
            "answers; billing counts generated tokens only."
        ),
        examples=[2048],
    )
    temperature: Optional[StrictFloat] | Ref(FLOAT_KIND, allow_none=True) = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; None keeps the provider default.",
        examples=[0.2],
    )


class OpenRouterVendorBlock(OpenRouterBlockBase):
    """Request and decoding steps the vendor blocks share.

    Subclasses declare a model table and call ``ask_images`` and
    ``decode_answers`` from ``run``.
    """

    models: Mapping[str, VendorModel] = {}
    model_family = "model"
    image_first = False
    """Whether requests use ``image_first_task_messages`` (Meta, Z.ai)."""

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the OpenRouter model the step calls; a selector as is."""
        model_id = params.model_version
        if model_id in cls.models:
            model_id = cls.models[model_id].slug

        resources = [
            third_party_model(
                model_id, provider="openrouter", resource="openrouter_http_client"
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

    def ask_images(
        self,
        images: Batch,
        *,
        model_version: str,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Any,
        privacy_level: str,
        max_tokens: int,
        temperature: Optional[float],
        reasoning_effort: Optional[str],
        max_concurrent_requests: Optional[int],
    ) -> Tuple[VendorModel, List[ChatAnswer]]:
        """Check the resolved values, then send one request per image.

        Args:
            images: The batch of images.
            model_version: V1 model label.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_tokens: Output token limit.
            temperature: Sampling temperature; ``None`` keeps the provider's.
            reasoning_effort: Level from the block, or ``None`` for no
                reasoning config.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            The selected model and one answer per image, in order.

        Raises:
            ValueError: On a missing key, an unknown model, a rejected
                reasoning level, a missing task field or invalid classes;
                nothing is sent.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails.
        """
        if api_key is None:
            # The OpenAI SDK would fall back to the OPENAI_API_KEY environment
            # variable and send that key to OpenRouter.
            raise ValueError(
                "`api_key` resolved to None; give an OpenRouter key or "
                "'rf_key:account'"
            )

        model = resolve_model(
            model_version, models=self.models, family=self.model_family
        )
        levels = {model_version: model.reasoning_levels}
        check_reasoning_effort(model_version, reasoning_effort, levels_by_model=levels)
        check_task_fields(
            task_type,
            prompt=prompt,
            classes=classes,
            output_structure=output_structure,
        )
        if task_type in DECODING_TASKS:
            classes = checked_classes(classes)

        prompts = [
            self._messages(
                task_type,
                image=image,
                model=model,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
            for image in images
        ]
        if model.reasoning_required and reasoning_effort == "none":
            reasoning_effort = ZAI_REQUIRED_REASONING_FALLBACK
        answers = self.ask_all(
            prompts,
            api_key=api_key,
            model=model.slug,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            reasoning_effort=reasoning_effort,
            max_concurrent_requests=max_concurrent_requests,
        )

        return model, answers

    def _messages(
        self, task_type: str, *, image: ImageData, model: VendorModel, **fields: Any
    ) -> List[Dict[str, Any]]:
        if self.image_first:
            upload = encode_capped_jpeg(image)
            messages = image_first_task_messages(
                task_type, upload=upload, model=model, **fields
            )
            return messages

        messages = openrouter_task_messages(
            task_type, image=image, detection_format=model.box_format, **fields
        )

        return messages


def decode_answers(
    images: Batch,
    answers: List[ChatAnswer],
    *,
    model: VendorModel,
    task_type: str,
    classes: Optional[List[str]],
    with_reasoning: bool,
) -> List[Dict[str, Any]]:
    """Decode each answer with the model's box format.

    Args:
        images: The images, in request order.
        answers: One answer per image.
        model: The selected model.
        task_type: What was asked.
        classes: Class names.
        with_reasoning: Whether to add ``thinking`` and token counts.

    Returns:
        One ``{"output", "classes", ["thinking", "input_tokens",
        "output_tokens",] "predictions", "error_status", "inference_id"}``
        per image.
    """
    results = []
    for image, answer in zip(images, answers):
        inference_id = str(uuid.uuid4())
        error_status, predictions = decode_vlm_output(
            task_type,
            answer.text,
            image=image,
            classes=classes,
            inference_id=inference_id,
            box_format=model.box_format,
        )
        result: Dict[str, Any] = {"output": answer.text, "classes": classes}
        if with_reasoning:
            result["thinking"] = answer.thinking
            result["input_tokens"] = answer.input_tokens
            result["output_tokens"] = answer.output_tokens
        result["predictions"] = predictions
        result["error_status"] = error_status
        result["inference_id"] = inference_id
        results.append(result)

    return results


class KimiOpenRouterBlock(OpenRouterVendorBlock):
    """Ask Moonshot AI Kimi vision models about each image.

    Example: ``task_type="object-detection"``, ``classes=["car"]`` asks Kimi
    K2.6 for 0-1 boxes and returns them as detections in image pixels.
    ``api_key="rf_key:account"`` (the default) bills the host's Roboflow
    account through its platform client.
    """

    type = "v2/kimi_openrouter@v3"
    metadata = {
        "name": "MoonshotAI Kimi",
        "port_of": "roboflow_core/kimi_openrouter@v3",
        "section": "model",
        "license": "Moonshot AI Kimi License",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "recommended_parsers": DECODED_PARSERS,
    }
    outputs = {
        **_raw_outputs(with_reasoning=False),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)
    models = KIMI_MODELS
    model_family = "Kimi model"

    class Params(_LegacyVendorParams):
        model_version: KimiModelVersion | Ref(STRING_KIND) = Field(
            default="Kimi K2.6 - OpenRouter",
            description="Kimi model to run.",
            examples=["Kimi K2.6 - OpenRouter", "$inputs.kimi_model"],
        )

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Dict[str, Output]:
        """Narrow ``predictions`` to the kind the configured task decodes."""
        outputs = {
            **cls.outputs,
            **vlm_prediction_outputs(params.task_type, source="images"),
        }

        return outputs

    def run(
        self,
        *,
        images: Batch,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Any,
        privacy_level: str,
        max_concurrent_requests: Optional[int],
        max_tokens: int,
        temperature: float,
        model_version: str,
    ) -> List[Dict[str, Any]]:
        """Send one request per image and decode each answer.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.
            max_tokens: Output token limit.
            temperature: Sampling temperature.
            model_version: Kimi model label.

        Returns:
            One ``{"output", "classes", "predictions", "error_status",
            "inference_id"}`` per image.

        Raises:
            ValueError: On an invalid resolved value, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        model, answers = self.ask_images(
            images,
            model_version=model_version,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            api_key=api_key,
            privacy_level=privacy_level,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=None,
            max_concurrent_requests=max_concurrent_requests,
        )
        results = decode_answers(
            images,
            answers,
            model=model,
            task_type=task_type,
            classes=classes,
            with_reasoning=False,
        )

        return results


class LlamaVisionBlock(OpenRouterVendorBlock):
    """Ask Llama 3.2 Vision about each image (deprecated, as in V1).

    OpenRouter no longer hosts Llama 3.2 Vision, so real requests are
    expected to fail; the block stays for existing workflows. It returns the
    raw answer only. Classification can use ``v2/vlm_as_classifier@v2``.
    Object detection returns raw normalized 0-1 ``x_min``/``y_min``/``x_max``/
    ``y_max`` fields in ``output``; no matching supported Llama ``model_type``
    exists in ``v2/vlm_as_detector@v2``. Use ``v2/meta_vlm@v3`` or
    ``v2/openrouter@v3`` for native decoded ``predictions``.
    """

    type = "v2/llama_vision@v2"
    metadata = {
        "name": "Llama 3.2 Vision",
        "port_of": "roboflow_core/llama_vision@v2",
        "section": "model",
        "license": "Llama 3.2 Community License",
        "deprecated": True,
        "deprecation_message": (
            "OpenRouter no longer hosts Llama 3.2 Vision. Use the Meta block "
            "(`v2/meta_vlm@v3`) for Muse Spark and Muse Glimmer; it decodes "
            "detection and classification predictions in-block."
        ),
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "recommended_parsers": LLAMA_PARSERS,
    }
    outputs = _raw_outputs(with_reasoning=False)
    models = LLAMA_MODELS
    model_family = "Llama Vision model"

    class Params(_LegacyVendorParams):
        model_version: LlamaModelVersion | Ref(STRING_KIND) = Field(
            default="11B - OpenRouter",
            description="Llama 3.2 Vision model to run.",
            examples=["11B - OpenRouter", "$inputs.llama_model"],
        )

    def run(
        self,
        *,
        images: Batch,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Any,
        privacy_level: str,
        max_concurrent_requests: Optional[int],
        max_tokens: int,
        temperature: float,
        model_version: str,
    ) -> List[Dict[str, Any]]:
        """Send one request per image; return the raw answers.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.
            max_tokens: Output token limit.
            temperature: Sampling temperature.
            model_version: Llama model label.

        Returns:
            One ``{"output", "classes"}`` per image.

        Raises:
            ValueError: On an invalid resolved value, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails.
        """
        _, answers = self.ask_images(
            images,
            model_version=model_version,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            api_key=api_key,
            privacy_level=privacy_level,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=None,
            max_concurrent_requests=max_concurrent_requests,
        )
        results = [{"output": answer.text, "classes": classes} for answer in answers]

        return results


class _ReasoningVendorBlock(OpenRouterVendorBlock):
    """Meta and Z.ai: image-first requests, a reasoning level, token outputs."""

    image_first = True
    outputs = {
        **_raw_outputs(with_reasoning=True),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Dict[str, Output]:
        """Narrow ``predictions`` to the kind the configured task decodes."""
        outputs = {
            **cls.outputs,
            **vlm_prediction_outputs(params.task_type, source="images"),
        }

        return outputs

    def run(
        self,
        *,
        images: Batch,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Any,
        privacy_level: str,
        max_concurrent_requests: Optional[int],
        max_tokens: Optional[int],
        temperature: Optional[float],
        model_version: str,
        reasoning_effort: str,
    ) -> List[Dict[str, Any]]:
        """Send one request per image and decode each answer.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.
            max_tokens: Output token limit; ``None`` means 2048.
            temperature: Sampling temperature; ``None`` keeps the provider's.
            model_version: Model label.
            reasoning_effort: Reasoning level the model accepts.

        Returns:
            One ``{"output", "classes", "thinking", "input_tokens",
            "output_tokens", "predictions", "error_status", "inference_id"}``
            per image.

        Raises:
            ValueError: On an invalid resolved value, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        model, answers = self.ask_images(
            images,
            model_version=model_version,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            api_key=api_key,
            privacy_level=privacy_level,
            max_tokens=2048 if max_tokens is None else max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_concurrent_requests=max_concurrent_requests,
        )
        results = decode_answers(
            images,
            answers,
            model=model,
            task_type=task_type,
            classes=classes,
            with_reasoning=True,
        )

        return results


def _model_metadata(models: Mapping[str, VendorModel]) -> Dict[str, Dict[str, Any]]:
    metadata = {
        label: {"name": label, "reasoning_levels": list(model.reasoning_levels)}
        for label, model in models.items()
    }

    return metadata


def _check_reasoning(params: BlockParams, models: Mapping[str, VendorModel]) -> None:
    levels = {label: model.reasoning_levels for label, model in models.items()}
    check_reasoning_effort(
        params.model_version, params.reasoning_effort, levels_by_model=levels
    )


class MetaVlmBlock(_ReasoningVendorBlock):
    """Ask Meta Muse vision models about each image.

    Example: ``model_version="Muse Glimmer"``, ``task_type="object-detection"``
    and ``classes=["car"]`` asks for 0-1000 ``x_min``/``y_min``/``x_max``/
    ``y_max`` boxes and returns them as detections in image pixels. Muse
    always reasons; ``reasoning_effort`` defaults to ``low``.
    """

    type = "v2/meta_vlm@v3"
    metadata = {
        "name": "Meta",
        "port_of": "roboflow_core/meta_vlm@v3",
        "section": "model",
        "license": "Apache-2.0",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "model_version_metadata": _model_metadata(META_MODELS),
        "reasoning_effort_metadata": MUSE_REASONING_METADATA,
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "recommended_parsers": DECODED_PARSERS,
    }
    models = META_MODELS
    model_family = "Muse variant"

    class Params(_ReasoningVendorParams):
        model_version: MetaModelVersion | Ref(STRING_KIND) = Field(
            default="Muse Spark 1.3",
            description=(
                "Muse model to run. Spark 1.1, Spark 1.2, Spark 1.3, and Glimmer "
                "are the current image-capable Muse chat models on OpenRouter."
            ),
            examples=["Muse Spark 1.3", "Muse Glimmer"],
        )
        reasoning_effort: MetaReasoningEffort = Field(
            default="low",
            description=(
                "Reasoning budget, always sent: Muse rejects disabled reasoning. "
                "'minimal' is Spark-only."
            ),
            examples=["low"],
        )

        @model_validator(mode="after")
        def _reasoning_supported(self) -> "MetaVlmBlock.Params":
            _check_reasoning(self, META_MODELS)

            return self


class ZaiVlmBlock(_ReasoningVendorBlock):
    """Ask Z.ai GLM vision models about each image.

    Example: ``task_type="object-detection"`` and ``classes=["car"]`` asks
    GLM 5V Turbo for 0-1000 xyxy boxes and returns them as detections in
    image pixels. Reasoning is off by default; GLM 5.3 Flash requires it, so
    ``none`` becomes ``low`` for that model.
    """

    type = "v2/zai_vlm@v2"
    metadata = {
        "name": "Z.ai",
        "port_of": "roboflow_core/zai_vlm@v2",
        "section": "model",
        "license": "MIT",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "model_version_metadata": _model_metadata(ZAI_MODELS),
        "reasoning_effort_metadata": ZAI_REASONING_METADATA,
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "recommended_parsers": DECODED_PARSERS,
    }
    models = ZAI_MODELS
    model_family = "Z.ai model"

    class Params(_ReasoningVendorParams):
        model_version: ZaiModelVersion | Ref(STRING_KIND) = Field(
            default="GLM 5V Turbo",
            description="Z.ai GLM model to run.",
            examples=["GLM 5V Turbo", "GLM 5.3 Flash"],
        )
        reasoning_effort: ZaiReasoningEffort = Field(
            default="none",
            description=(
                "Extended-reasoning budget; 'none' (the default) disables it. "
                "GLM 5.3 Flash requires reasoning and gets 'low' instead."
            ),
            examples=["none"],
        )

        @model_validator(mode="after")
        def _reasoning_supported(self) -> "ZaiVlmBlock.Params":
            _check_reasoning(self, ZAI_MODELS)

            return self


# LlamaVisionBlock is retained as implementation history, outside discovery.
BLOCKS = (KimiOpenRouterBlock, MetaVlmBlock, ZaiVlmBlock)

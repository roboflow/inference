"""Ask Qwen vision-language models about images, locally or through OpenRouter.

Port of V1 ``roboflow_core/qwen_vlm@v4``. One step picks a backend; both ask
the task's prompt and decode the answer with the same box format::

    every check, on literal and resolved selector values   (before any effect)
    backend "native"
      model_version label ──▶ inference_models id, or fine_tuned_model_id
      task ──▶ "<user text><system_prompt><system text>"   (V1 native prompts)
      models resource ──▶ ModelRequest(model id, model_package_id) ──▶ model
      per image: model.prompt([RGB tensor], prompt=..., enable_thinking=...)
    backend "openrouter"
      openrouter_model_version label ──▶ VendorModel (slug, reasoning rules)
      per image: one user message, JPEG image first, then the task text
                 (shrunk 10% at a time while its base64 exceeds 9.5 MB)
      OpenRouterBlockBase.ask_all ──▶ answers in image order
    answer ──▶ decode_vlm_output("xyxy_0_1000") ──▶ predictions, error_status

The OpenRouter half is the request path of ``openrouter_vendors`` (V1 Qwen
v4 shares it with Z.ai): image-first messages, the capped JPEG upload,
reasoning disabled by default and ``low`` for models that require it,
``temperature`` left out unless set, privacy and quantization routing, and
the retry without a rejected reasoning config. The native half reuses the
``ModelCache`` resource, the ``<system_prompt>`` marker of
``v2/qwen3_5vl@v2`` and the generation options of the local VLM helpers.
Native token counts are ``None``: no backend reports them.

Differences from V1:

* The native backend runs only on the host's ``models`` resource. V1's
  remote native mode (Roboflow-hosted inference) is not ported.
* ``model_package_id`` picks an exact native package, as in every V2 local
  model block.
* A native prompt containing ``<system_prompt>`` fails the step; V1 split
  it in the wrong place.
* A native answer must be one text or one ``{"thinking", "answer"}`` dict
  per image; anything else fails the step. V1 returned it as JSON text. A
  thinking answer with an empty ``answer`` gives ``output=""`` and keeps
  ``thinking``; V1 returned the whole dict as JSON text in ``output``.
* Every check runs again on resolved selector values before any model load
  or request: labels, the fine-tuned id, text fields, task fields, decoding
  ``classes`` and the ``temperature`` range.
* A failed OpenRouter request raises ``ChatRequestError`` with the keys
  scrubbed; ``max_concurrent_requests`` must be at least 1, unset means 8.
* A ``model_version`` selector is reported to workload introspection as the
  selector; V1 attached a label-to-id resolver.
* Predictions are native V2 carriers; see ``decoding``.
"""

import json
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.blocks.language.chat import (
    DEFAULT_MANAGED_KEY,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    ChatAnswer,
    check_task_fields,
    third_party_model,
    token_outputs,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    build_object_detection_prompt,
    checked_classes,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.blocks.language.openrouter import (
    CAPTION_SYSTEM_PROMPT,
    CLASS_LIST_TEXT,
    CLASSIFICATION_SYSTEM_PROMPT,
    MULTI_LABEL_SYSTEM_PROMPT,
    OCR_SYSTEM_PROMPT,
    OPENROUTER_API_URL,
    OPENROUTER_TASKS,
    PRIVACY_LEVEL_METADATA,
    STRUCTURED_ANSWERING_SYSTEM_PROMPT,
    VQA_SYSTEM_PROMPT,
    OpenRouterTask,
    PrivacyLevel,
)
from roboflow_workflows.v2.blocks.language.openrouter_vendors import (
    DECODED_PARSERS,
    DECODING_TASKS,
    OpenRouterVendorBlock,
    VendorModel,
    decode_answers,
)
from roboflow_workflows.v2.blocks.models._local_vlm import (
    checked_text,
    generation_options,
    require_rgb,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import MODELS_RESOURCE, models_cache
from roboflow_workflows.v2.blocks.models.vlm import SYSTEM_PROMPT_MARKER, TASK_TYPE
from roboflow_workflows.v2.blocks.platform import (
    OFFLINE_PLATFORM_CLIENT,
    PlatformClient,
)
from roboflow_workflows.v2.data.batch import Batch
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

QWEN_BOX_FORMAT = "xyxy_0_1000"
"""Qwen's grounding format: ``box_2d`` integers 0-1000; both backends use it."""

DEFAULT_MAX_TOKENS = 2048
"""V1's output limit on both backends; native ``None`` keeps the backend's."""

FINE_TUNED_LABEL = "Fine-tuned model"
DEFAULT_NATIVE_MODEL_VERSION = "Qwen 3.5 VL 2B"
DEFAULT_OPENROUTER_MODEL_VERSION = "Qwen 3.7 Plus"

NATIVE_MODELS: Dict[str, str] = {
    "Qwen 2.5 VL 7B": "qwen25-vl-7b",
    "Qwen 3 VL 2B": "qwen3vl-2b-instruct",
    "Qwen 3.5 VL 0.8B": "qwen3_5-0.8b",
    "Qwen 3.5 VL 2B": "qwen3_5-2b",
    "Qwen 3.8 VL 27B": "qwen3_8-27b",
}
"""V1 native labels and their ``inference_models`` model ids."""

FINE_TUNED_FAMILIES = ("qwen-pretrains/2",)
"""Workspace fine-tune families V1 lists next to the native ids."""

THINKING_MODEL_VERSIONS = ("Qwen 3.5 VL 2B", "Qwen 3.8 VL 27B", FINE_TUNED_LABEL)
"""Native labels that honour ``enable_thinking``; fine-tunes derive from Qwen 3.5."""

REASONING_LEVELS = ("none", "low", "medium", "high")


def _openrouter_model(slug: str, *, reasoning_required: bool = False) -> VendorModel:
    model = VendorModel(
        slug,
        box_format=QWEN_BOX_FORMAT,
        reasoning_levels=REASONING_LEVELS,
        reasoning_required=reasoning_required,
    )

    return model


OPENROUTER_MODELS: Dict[str, VendorModel] = {
    "Qwen 3.8 Max": _openrouter_model("qwen/qwen3.8-max", reasoning_required=True),
    "Qwen 3.8 Flash": _openrouter_model("qwen/qwen3.8-flash"),
    "Qwen 3.7 Plus": _openrouter_model("qwen/qwen3.7-plus"),
    "Qwen 3.7 Flash": _openrouter_model("qwen/qwen3.7-flash"),
    "Qwen 3.8 27B": _openrouter_model("qwen/qwen3.8-27b"),
    "Qwen 3.6 Plus": _openrouter_model("qwen/qwen3.6-plus"),
    "Qwen 3.6 Flash": _openrouter_model("qwen/qwen3.6-flash"),
    "Qwen 3.6 27B": _openrouter_model("qwen/qwen3.6-27b"),
    "Qwen 3.6 35B A3B": _openrouter_model("qwen/qwen3.6-35b-a3b"),
    "Qwen 3.5 Plus": _openrouter_model("qwen/qwen3.5-plus-20260420"),
    "Qwen 3.5 Plus 02-15": _openrouter_model("qwen/qwen3.5-plus-02-15"),
    "Qwen 3.5 Flash": _openrouter_model("qwen/qwen3.5-flash-02-23"),
    "Qwen 3.5 9B": _openrouter_model("qwen/qwen3.5-9b"),
    "Qwen 3.5 27B": _openrouter_model("qwen/qwen3.5-27b"),
    "Qwen 3.5 35B A3B": _openrouter_model("qwen/qwen3.5-35b-a3b"),
    "Qwen 3.5 122B A10B": _openrouter_model("qwen/qwen3.5-122b-a10b"),
    "Qwen 3.5 397B A17B": _openrouter_model("qwen/qwen3.5-397b-a17b"),
    "Qwen 3 VL 8B Instruct": _openrouter_model("qwen/qwen3-vl-8b-instruct"),
    "Qwen 3 VL 8B Thinking": _openrouter_model(
        "qwen/qwen3-vl-8b-thinking", reasoning_required=True
    ),
    "Qwen 3 VL 30B A3B Instruct": _openrouter_model("qwen/qwen3-vl-30b-a3b-instruct"),
    "Qwen 3 VL 30B A3B Thinking": _openrouter_model(
        "qwen/qwen3-vl-30b-a3b-thinking", reasoning_required=True
    ),
    "Qwen 3 VL 32B Instruct": _openrouter_model("qwen/qwen3-vl-32b-instruct"),
    "Qwen 3 VL 235B A22B": _openrouter_model("qwen/qwen3-vl-235b-a22b-instruct"),
    "Qwen 3 VL 235B A22B Thinking": _openrouter_model(
        "qwen/qwen3-vl-235b-a22b-thinking", reasoning_required=True
    ),
}
"""V1 OpenRouter labels; ``reasoning_required`` models get ``low`` for ``none``."""

Backend = Literal["native", "openrouter"]
NativeModelVersion = Literal[(*NATIVE_MODELS, FINE_TUNED_LABEL)]
OpenRouterModelVersion = Literal[tuple(OPENROUTER_MODELS)]
ReasoningEffort = Literal[REASONING_LEVELS]

NATIVE_UNCONSTRAINED_SYSTEM_PROMPT = (
    "You are a Qwen vision-language model that can answer questions about any image."
)
NATIVE_STRUCTURE_TEXT = "Specification of requirements regarding output fields: \n"

BACKEND_METADATA = {
    "native": {
        "name": "Native (Roboflow)",
        "description": (
            "Runs a Qwen-VL model from the host's 'models' resource. Smaller "
            "models, no external request."
        ),
    },
    "openrouter": {
        "name": "OpenRouter",
        "description": (
            "Routes to large hosted Qwen models via OpenRouter. Defaults to a "
            "Roboflow-managed key (billed in credits)."
        ),
    },
}
REASONING_EFFORT_METADATA = {
    "none": {
        "name": "Disabled (recommended)",
        "description": (
            "Turns extended reasoning off. Qwen models default to extended "
            "reasoning on OpenRouter, which bloats latency and can consume the "
            "whole token budget before a visible answer is produced. Models that "
            "require reasoning (Qwen 3.8 Max and the Qwen 3 VL Thinking variants) "
            "fall back to low effort instead."
        ),
    },
    "low": {"name": "Low", "description": "Small reasoning budget before answering."},
    "medium": {
        "name": "Medium",
        "description": "Moderate reasoning budget before answering.",
    },
    "high": {
        "name": "High",
        "description": (
            "Large reasoning budget. Slowest and most expensive; consider raising "
            "`max_tokens` so reasoning does not crowd out the answer."
        ),
    },
}


def native_prompt(
    task_type: str,
    *,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> str:
    """Build V1's native prompt: user text, the marker, then the system text.

    Object detection sends an empty system text, so the backend uses its
    default system prompt, as V1 does.

    Args:
        task_type: One of ``OPENROUTER_TASKS``.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        ``"<user text><system_prompt><system text>"``.

    Raises:
        ValueError: On an unknown task, or user text containing the
            ``<system_prompt>`` marker, which the backend would split on.
    """
    class_text = CLASS_LIST_TEXT.format(class_list=", ".join(classes or []))
    texts = {
        "unconstrained": lambda: (prompt or "", NATIVE_UNCONSTRAINED_SYSTEM_PROMPT),
        "ocr": lambda: ("Extract the text from this image.", OCR_SYSTEM_PROMPT),
        "visual-question-answering": lambda: (
            f"Question: {prompt}" if prompt else "Describe this image.",
            VQA_SYSTEM_PROMPT,
        ),
        "caption": lambda: (
            "Caption this image.",
            CAPTION_SYSTEM_PROMPT.format(length="short"),
        ),
        "detailed-caption": lambda: (
            "Describe this image in detail.",
            CAPTION_SYSTEM_PROMPT.format(length="extensive"),
        ),
        "classification": lambda: (class_text, CLASSIFICATION_SYSTEM_PROMPT),
        "multi-label-classification": lambda: (class_text, MULTI_LABEL_SYSTEM_PROMPT),
        "object-detection": lambda: (
            build_object_detection_prompt(QWEN_BOX_FORMAT, classes=classes or []),
            "",
        ),
        "structured-answering": lambda: (
            NATIVE_STRUCTURE_TEXT + json.dumps(output_structure or {}, indent=4),
            STRUCTURED_ANSWERING_SYSTEM_PROMPT,
        ),
    }
    if task_type not in texts:
        raise ValueError(f"Task type: {task_type} not supported.")

    user_text, system_text = texts[task_type]()
    if SYSTEM_PROMPT_MARKER in user_text:
        raise ValueError(
            f"The native Qwen prompt contains {SYSTEM_PROMPT_MARKER!r}, which the "
            "model reads as the start of the system prompt; remove it from prompt, "
            "classes and output_structure"
        )

    full_prompt = f"{user_text}{SYSTEM_PROMPT_MARKER}{system_text}"

    return full_prompt


def native_model_id(model_version: Any, *, fine_tuned_model_id: Any) -> str:
    """Resolve a native label to the model id the ``models`` resource loads.

    Args:
        model_version: A ``NATIVE_MODELS`` label or ``FINE_TUNED_LABEL``.
        fine_tuned_model_id: Workspace model id used with ``FINE_TUNED_LABEL``.

    Returns:
        The model id.

    Raises:
        ValueError: On an unknown label, or a fine-tuned label without a
            non-empty ``fine_tuned_model_id``.
    """
    if model_version == FINE_TUNED_LABEL:
        if not isinstance(fine_tuned_model_id, str) or not fine_tuned_model_id:
            raise ValueError(
                "`fine_tuned_model_id` is required when "
                f"`model_version='{FINE_TUNED_LABEL}'`."
            )
        return fine_tuned_model_id

    if not isinstance(model_version, str) or model_version not in NATIVE_MODELS:
        raise ValueError(
            f"Unknown pre-trained Qwen variant '{model_version}'. Pick one of: "
            f"{list(NATIVE_MODELS)} or '{FINE_TUNED_LABEL}'."
        )

    model_id = NATIVE_MODELS[model_version]

    return model_id


def _native_answer(answers: Any, *, request: ModelRequest) -> ChatAnswer:
    # Qwen3.5 and Qwen3.8 answer with {"thinking", "answer"} in thinking mode.
    if isinstance(answers, list) and len(answers) == 1:
        answer = answers[0]
        if isinstance(answer, str):
            return ChatAnswer(text=answer)

        if (
            isinstance(answer, dict)
            and isinstance(answer.get("answer"), str)
            and isinstance(answer.get("thinking", ""), str)
        ):
            return ChatAnswer(
                text=answer["answer"], thinking=answer.get("thinking", "")
            )

    raise ValueError(
        f"v2/qwen_vlm@v4: model {str(request)!r} must answer one image with a list "
        f"of one text or one {{'thinking', 'answer'}} dict, got {answers!r}"
    )


def _checked_temperature(temperature: Any) -> None:
    if temperature is None:
        return

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(f"temperature must be a number or None, got {temperature!r}")

    if not 0.0 <= temperature <= 2.0:
        raise ValueError("'temperature' parameter required to be in range [0.0, 2.0]")


class _QwenOpenRouter(OpenRouterVendorBlock):
    """OpenRouter request path of ``v2/qwen_vlm@v4``; not a registered block."""

    models = OPENROUTER_MODELS
    model_family = "OpenRouter Qwen variant"
    image_first = True


class QwenVLMBlock(Block):
    """Ask a Qwen vision-language model about each image, natively or via OpenRouter.

    Example: ``backend="openrouter"``, ``openrouter_model_version="Qwen 3.7
    Plus"``, ``task_type="object-detection"`` and ``classes=["car"]`` asks
    for 0-1000 ``box_2d`` boxes and returns them as detections in image
    pixels. With ``backend="native"`` and ``model_version="Qwen 3.5 VL 2B"``
    the same step runs ``qwen3_5-2b`` from the host's ``models`` resource
    and decodes the same answer format.
    """

    type = "v2/qwen_vlm@v4"
    metadata = {
        "name": "Qwen",
        "port_of": "roboflow_core/qwen_vlm@v4",
        "section": "model",
        "license": "Apache-2.0",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "backend_metadata": BACKEND_METADATA,
        "openrouter_model_version_metadata": {
            label: {"name": label, "reasoning_levels": list(model.reasoning_levels)}
            for label, model in OPENROUTER_MODELS.items()
        },
        "reasoning_effort_metadata": REASONING_EFFORT_METADATA,
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "recommended_parsers": DECODED_PARSERS,
        "supported_model_variants": [*NATIVE_MODELS.values(), *FINE_TUNED_FAMILIES],
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
            description=(
                "Reasoning trace: native with enable_thinking on a thinking "
                "model, OpenRouter when the model returns one; else empty."
            ),
        ),
        **token_outputs(source="images"),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to ask about; one model call or request per image."
        )
        backend: Backend = Field(
            default="native",
            description=(
                "'native' runs a model from the host's 'models' resource; "
                "'openrouter' sends requests to OpenRouter."
            ),
            examples=["native", "openrouter"],
        )
        model_version: NativeModelVersion | Ref(STRING_KIND) = Field(
            default=DEFAULT_NATIVE_MODEL_VERSION,
            description=(
                f"Native Qwen-VL variant, or {FINE_TUNED_LABEL!r} to use "
                "fine_tuned_model_id. Only used when backend='native'."
            ),
            examples=[DEFAULT_NATIVE_MODEL_VERSION, FINE_TUNED_LABEL],
        )
        fine_tuned_model_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=(
                    "Fine-tuned Qwen model id in 'workspace/version' form; required "
                    f"when model_version={FINE_TUNED_LABEL!r}."
                ),
                examples=["your-workspace/3", "$inputs.qwen_finetune"],
            )
        )
        model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=(
                    "Exact native model package to load; None lets the loader "
                    "choose. Only used when backend='native'."
                ),
                examples=[None, "$inputs.model_package_id"],
            )
        )
        openrouter_model_version: OpenRouterModelVersion | Ref(STRING_KIND) = Field(
            default=DEFAULT_OPENROUTER_MODEL_VERSION,
            description=(
                "OpenRouter-hosted Qwen vision variant. Only used when "
                "backend='openrouter'."
            ),
            examples=[DEFAULT_OPENROUTER_MODEL_VERSION, "Qwen 3.8 Max"],
        )
        task_type: OpenRouterTask = Field(
            default="unconstrained",
            description=(
                "What to ask: a free prompt, OCR, captions, classification, "
                "structured answers or detection."
            ),
            examples=["object-detection"],
        )
        prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description=(
                "Prompt for 'unconstrained' and question for "
                "'visual-question-answering'; one value for the whole step."
            ),
            examples=["my prompt", "$inputs.prompt"],
        )
        enable_thinking: StrictBool = Field(
            default=False,
            description=(
                "Native thinking mode; the trace goes to 'thinking'. Used by "
                f"{list(THINKING_MODEL_VERSIONS)}, ignored by other variants."
            ),
            examples=[False],
        )
        reasoning_effort: ReasoningEffort = Field(
            default="none",
            description=(
                "OpenRouter reasoning budget; 'none' (default) disables it. Models "
                "that require reasoning get 'low' instead."
            ),
            examples=["none", "low"],
        )
        output_structure: Optional[Dict[str, StrictStr]] = Field(
            default=None,
            description=(
                "For 'structured-answering': answer field names and their "
                "descriptions."
            ),
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
                "OpenRouter key as a secret or string, or 'rf_key:account' / "
                "'rf_key:user:ID' to use a key stored with Roboflow. Only used "
                "when backend='openrouter'."
            ),
            examples=["rf_key:account", "sk-or-...", "$inputs.openrouter_api_key"],
        )
        privacy_level: PrivacyLevel = Field(
            default="deny",
            description=(
                "OpenRouter provider privacy filter. Stricter levels reduce the "
                "pool of providers and may increase per-call cost."
            ),
            examples=["deny"],
        )
        max_tokens: Optional[StrictInt] = Field(
            default=DEFAULT_MAX_TOKENS,
            gt=1,
            description=(
                f"Most generated tokens; {DEFAULT_MAX_TOKENS} by default. None "
                f"means {DEFAULT_MAX_TOKENS} on OpenRouter and the backend default "
                "natively."
            ),
            examples=[DEFAULT_MAX_TOKENS, 8192],
        )
        temperature: Optional[StrictFloat] | Ref(FLOAT_KIND, allow_none=True) = Field(
            default=None,
            ge=0.0,
            le=2.0,
            description=(
                "OpenRouter sampling temperature; None leaves it out, so the "
                "provider's per-model default applies."
            ),
            examples=[0.2],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most OpenRouter requests in flight per batch; None means 8.",
            examples=[4],
        )

        @model_validator(mode="after")
        def _required_fields_present(self) -> "QwenVLMBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )
            fine_tuned = self.model_version == FINE_TUNED_LABEL
            if self.backend == "native" and fine_tuned and not self.fine_tuned_model_id:
                raise ValueError(
                    "`fine_tuned_model_id` is required when "
                    f"`model_version='{FINE_TUNED_LABEL}'`."
                )

            return self

    def __init__(
        self,
        *,
        models: Union[None, Mapping[Any, Any], ModelCache] = None,
        platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT,
        api_key: Any = None,
        openrouter_base_url: str = OPENROUTER_API_URL,
        openrouter_http_client: Any = None,
    ) -> None:
        # models is needed by the native backend only.
        self._models = None if models is None else models_cache(models, block=self.type)
        self._openrouter = _QwenOpenRouter(
            platform_client=platform_client,
            api_key=api_key,
            openrouter_base_url=openrouter_base_url,
            openrouter_http_client=openrouter_http_client,
        )

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
        """Report the native model or the OpenRouter model; a selector as is."""
        if params.backend == "openrouter":
            model_id = params.openrouter_model_version
            if model_id in OPENROUTER_MODELS:
                model_id = OPENROUTER_MODELS[model_id].slug
            resources = [
                third_party_model(
                    model_id, provider="openrouter", resource="openrouter_http_client"
                )
            ]
            return resources

        model_id = params.model_version
        if model_id == FINE_TUNED_LABEL:
            model_id = params.fine_tuned_model_id
        elif model_id in NATIVE_MODELS:
            model_id = NATIVE_MODELS[model_id]
        resources = [
            DependentResource(
                resource_type="model",
                identifier=model_id,
                details={
                    "resource": MODELS_RESOURCE,
                    "task_type": TASK_TYPE,
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return resources

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[Any]:
        """Report local inference, or remote inference and image encoding."""
        if params.backend == "openrouter":
            operations = list(WORK_OPERATIONS)
            return operations

        return [WorkOperation.MODEL_INFERENCE]

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> List[Any]:
        """Declare that the step has no runtime restrictions."""
        return []

    def run(
        self,
        *,
        images: Batch,
        backend: str,
        model_version: str,
        fine_tuned_model_id: Optional[str],
        model_package_id: Optional[str],
        openrouter_model_version: str,
        task_type: str,
        prompt: Optional[str],
        enable_thinking: bool,
        reasoning_effort: str,
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Any,
        privacy_level: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Ask every image of the batch, then decode each answer.

        Args:
            images: The batch of RGB images.
            backend: ``"native"`` or ``"openrouter"``.
            model_version: Native label or ``FINE_TUNED_LABEL``.
            fine_tuned_model_id: Model id used with ``FINE_TUNED_LABEL``.
            model_package_id: Exact native package, or ``None``.
            openrouter_model_version: OpenRouter label.
            task_type: What to ask.
            prompt: Prompt or question.
            enable_thinking: Native thinking mode, for thinking variants.
            reasoning_effort: OpenRouter reasoning level.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_tokens: Output token limit; see ``Params``.
            temperature: OpenRouter temperature; ``None`` is left out.
            max_concurrent_requests: OpenRouter requests in flight.

        Returns:
            One ``{"output", "classes", "thinking", "input_tokens",
            "output_tokens", "predictions", "error_status", "inference_id"}``
            per image, in batch order.

        Raises:
            ValueError: On an invalid resolved value, a missing ``models``
                resource or an unexpected native answer; checks run before
                any model load or request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When an OpenRouter request fails.
        """
        prompt = checked_text(prompt, name="prompt")
        if backend == "openrouter":
            results = self._ask_openrouter(
                images,
                model_version=openrouter_model_version,
                task_type=task_type,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
                api_key=api_key,
                privacy_level=privacy_level,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_concurrent_requests=max_concurrent_requests,
            )
            return results

        results = self._ask_native(
            images,
            model_version=model_version,
            fine_tuned_model_id=fine_tuned_model_id,
            model_package_id=model_package_id,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
        )

        return results

    def _ask_openrouter(
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
        max_tokens: Optional[int],
        temperature: Optional[float],
        reasoning_effort: str,
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        _checked_temperature(temperature)
        # ask_images checks the key, label, reasoning level, task fields and
        # decoding classes before it encodes an image or sends a request.
        model, answers = self._openrouter.ask_images(
            images,
            model_version=model_version,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            api_key=api_key,
            privacy_level=privacy_level,
            max_tokens=DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
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

    def _ask_native(
        self,
        images: Batch,
        *,
        model_version: str,
        fine_tuned_model_id: Optional[str],
        model_package_id: Optional[str],
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        enable_thinking: bool,
        max_tokens: Optional[int],
    ) -> List[Dict[str, Any]]:
        if self._models is None:
            raise ValueError(
                f"{self.type} with backend='native' needs the '{MODELS_RESOURCE}' "
                "resource: a ModelCache or a mapping of model id to loaded model"
            )

        check_task_fields(
            task_type, prompt=prompt, classes=classes, output_structure=output_structure
        )
        if task_type in DECODING_TASKS:
            classes = checked_classes(classes)
        model_id = native_model_id(
            model_version, fine_tuned_model_id=fine_tuned_model_id
        )
        request = ModelRequest(model_id, model_package_id)
        full_prompt = native_prompt(
            task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        thinking = enable_thinking and model_version in THINKING_MODEL_VERSIONS
        options = {"enable_thinking": thinking, **generation_options(max_tokens)}
        require_rgb(images)
        if len(images) == 0:
            return []

        model = self._models.get(request)
        answers = [
            _native_answer(
                model.prompt(
                    [member.tensor_image],
                    prompt=full_prompt,
                    input_color_format="rgb",
                    **options,
                ),
                request=request,
            )
            for member in images
        ]
        # The native model id stands in for a slug; decoding reads the box format.
        decoding = VendorModel(model_id, box_format=QWEN_BOX_FORMAT)
        results = decode_answers(
            images,
            answers,
            model=decoding,
            task_type=task_type,
            classes=classes,
            with_reasoning=True,
        )

        return results


BLOCKS = (QwenVLMBlock,)

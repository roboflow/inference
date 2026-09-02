"""Z.ai GLM vision-language block, v2.

Same surface as ``zai_vlm@v1``, plus in-block decoding of the model
answer: ``predictions``, ``error_status`` and ``inference_id`` outputs sit
next to the raw ``output`` string, so no separate "VLM as Detector" /
"VLM as Classifier" step is needed.

OpenRouter-only. Uses the vlm-exam request contract validated for the GLM
models: image-first user message, no system role, extended reasoning
disabled by default, and a per-model ``box_2d`` / 0-1000 detection
contract - ``xyxy_0_1000`` for GLM 5V Turbo and ``yxyx_0_1000`` for
GLM 5.3 Flash.
"""

import base64
import json
from typing import Any, Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

import cv2
import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.core_steps.common.openrouter import (
    PRIVACY_LEVEL_LITERAL,
    PRIVACY_LEVEL_METADATA,
    RELEVANT_TASKS_METADATA,
    SUPPORTED_TASK_TYPES_LIST,
    OpenRouterBlockManifestMixin,
    OpenRouterWorkflowBlockBase,
    validate_task_type_required_fields,
)
from inference.core.workflows.core_steps.common.reasoning import (
    attach_reasoning_levels,
    build_openrouter_reasoning_config,
    validate_reasoning_level,
)
from inference.core.workflows.core_steps.common.token_usage import (
    TOKEN_OUTPUT_DEFINITIONS,
)
from inference.core.workflows.core_steps.common.utils import (
    scale_dimensions_to_max_edge,
)
from inference.core.workflows.core_steps.common.vlm_decoding import (
    actual_vlm_prediction_outputs,
    build_object_detection_prompt,
    decode_vlm_output,
    describe_vlm_prediction_outputs,
)
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
    WorkflowBlockManifest,
    is_workflow_selector,
    third_party_model,
)

# Detection and classification answers are decoded in-block now, so only
# `structured-answering` still points at a downstream parser.
RECOMMENDED_PARSERS = {
    "structured-answering": "roboflow_core/json_parser@v1",
}

# Default detection contract, used when a caller does not pass a per-model
# override: the vlm-exam `xyxy_normalized_0_to_1000` format pinned for
# GLM 5V Turbo.
DEFAULT_DETECTION_BOX_FORMAT = "xyxy_0_1000"

# One row per model; add future Z.ai models here. A row may override
# `reasoning_levels` if a model diverges from the shared set;
# `reasoning_required` marks models that reject `reasoning: {enabled:
# false}` and fall back to low effort when the user disables reasoning.
# `box_format` is the shared coordinate contract used to both prompt for
# and decode detections; the two GLM models emit the same "box_2d" key with
# different axis orders, so each pins its own format. GLM 5.3 Flash's
# `yxyx_0_1000` scores 0.331 dataset mAP@50 on the full 250-image benchmark
# vs 0.219 for absolute-pixel bbox_2d prompting, the next best format.
MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "GLM 5V Turbo": {
        "model_id": "z-ai/glm-5v-turbo",
        "box_format": "xyxy_0_1000",
    },
    # GLM 5.3 Flash ran as the OpenRouter stealth model "Ox Alpha"; the
    # retirement notice confirms they are the same model.
    "GLM 5.3 Flash": {
        "model_id": "z-ai/glm-5.3-flash",
        "box_format": "yxyx_0_1000",
        "reasoning_required": True,
    },
}

MODEL_IDS = {label: variant["model_id"] for label, variant in MODEL_VARIANTS.items()}

ModelVersion = Literal[tuple(MODEL_VARIANTS.keys())]
DEFAULT_MODEL_VERSION = "GLM 5V Turbo"

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high"]
ReasoningEffort = Literal[tuple(REASONING_EFFORT_OPTIONS)]
DEFAULT_REASONING_EFFORT = "none"

REASONING_EFFORT_METADATA = {
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

MODEL_REASONING_LEVELS = {
    label: variant.get("reasoning_levels", REASONING_EFFORT_OPTIONS)
    for label, variant in MODEL_VARIANTS.items()
}

MODEL_VERSION_METADATA = attach_reasoning_levels(
    {label: {"name": label} for label in MODEL_VARIANTS},
    MODEL_REASONING_LEVELS,
)

# Fallback effort applied when reasoning is disabled by the user but the
# selected model rejects `reasoning: {"enabled": false}`.
REASONING_REQUIRED_FALLBACK_EFFORT = "low"


def build_zai_reasoning_config(
    reasoning_effort: str,
    *,
    reasoning_required: bool,
) -> Optional[dict]:
    """Translate the block's reasoning effort into OpenRouter's config.

    Mirrors the vlm-exam benchmark behavior: reasoning is explicitly
    disabled unless an effort is requested, except for models that require
    reasoning and reject ``enabled: false``; those always receive an
    effort (falling back to low when the user disabled reasoning).

    Args:
        reasoning_effort: One of ``REASONING_EFFORT_OPTIONS``.
        reasoning_required: Whether the model rejects disabled reasoning.

    Returns:
        OpenRouter ``reasoning`` payload object.
    """
    if reasoning_required and reasoning_effort == "none":
        return {"effort": REASONING_REQUIRED_FALLBACK_EFFORT}
    return build_openrouter_reasoning_config(reasoning_effort)


DEFAULT_MAX_TOKENS = 2048
OPENROUTER_MAX_BASE64_BYTES = 9_500_000
OPENROUTER_JPEG_QUALITY = 90

_TASK_CLASSIFICATION = (
    "You act as single-class classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in Markdown "
    "```json [...]``` markers. Expected structure of json: "
    '{"class_name": "class-name", "confidence": 0.4}. `class-name` must be one '
    "of the class names defined by user. You are only allowed to return single "
    "JSON document, even if there are potentially multiple classes. You are not "
    "allowed to return list. You cannot discuss the result, you are only "
    "allowed to return JSON document."
)

_TASK_MULTI_LABEL = (
    "You act as multi-label classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in Markdown "
    '```json``` markers. Expected structure of json: {"predicted_classes": '
    '[{"class": "class-name-1", "confidence": 0.9}, {"class": "class-name-2", '
    '"confidence": 0.7}]}. `class-name-X` must be one of the class names '
    "defined by user and `confidence` is a float value in range 0.0-1.0 that "
    "represent how sure you are that the class is present in the image. Only "
    "return class names that are visible. You cannot discuss the result, you "
    "are only allowed to return JSON document."
)

_TASK_VQA = (
    "You act as Visual Question Answering model. Your task is to provide "
    "answer to question submitted by user. If this is open-question - answer "
    "with few sentences, for ABCD question, return only the indicator of "
    "the answer."
)

_TASK_OCR = (
    "You act as OCR model. Your task is to read text from the image and "
    "return it in paragraphs representing the structure of texts in the "
    "image. You should only return recognised text, nothing else."
)

_TASK_STRUCTURED = (
    "You are supposed to produce responses in JSON wrapped in Markdown "
    "markers: ```json\nyour-response\n```. User is to provide you "
    "dictionary with keys and values. Each key must be present in your "
    "response. Values in user dictionary represent descriptions for JSON "
    "fields to be generated. Provide only JSON Markdown in response."
)


def encode_image_for_zai_openrouter(numpy_image: np.ndarray) -> str:
    """Encode a BGR image as base64 JPEG under the OpenRouter payload cap.

    Encodes at fixed JPEG quality and, when the base64 payload exceeds
    ``OPENROUTER_MAX_BASE64_BYTES``, iteratively downscales the longest
    edge by 10% until it fits. Detection coordinates are normalized to
    0-1000, so downscaling does not affect parsing.

    Args:
        numpy_image: Image in BGR channel order.

    Returns:
        Base64-encoded JPEG payload.
    """
    working = numpy_image
    while True:
        jpeg_bytes = encode_image_to_jpeg_bytes(
            working, jpeg_quality=OPENROUTER_JPEG_QUALITY
        )
        base64_image = base64.b64encode(jpeg_bytes).decode("ascii")
        if len(base64_image) <= OPENROUTER_MAX_BASE64_BYTES:
            return base64_image

        height, width = working.shape[:2]
        if max(height, width) <= 1:
            return base64_image

        target_max_edge = max(int(max(height, width) * 0.9), 1)
        target_width, target_height = scale_dimensions_to_max_edge(
            width, height, target_max_edge
        )
        working = cv2.resize(
            working, (target_width, target_height), interpolation=cv2.INTER_AREA
        )


def _user_message(base64_image: str, text: str) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
                {"type": "text", "text": text},
            ],
        }
    ]


def _prepare_unconstrained_prompt(base64_image: str, prompt: str, **_) -> List[dict]:
    return _user_message(base64_image=base64_image, text=prompt)


def _prepare_ocr_prompt(base64_image: str, **_) -> List[dict]:
    return _user_message(base64_image=base64_image, text=_TASK_OCR)


def _prepare_vqa_prompt(base64_image: str, prompt: str, **_) -> List[dict]:
    return _user_message(
        base64_image=base64_image, text=f"{_TASK_VQA}\n\nQuestion: {prompt}"
    )


def _prepare_caption_prompt(
    base64_image: str, short_description: bool, **_
) -> List[dict]:
    detail = (
        "Caption should be short."
        if short_description
        else "Caption should be extensive."
    )
    text = (
        "You act as image caption model. Your task is to provide "
        f"description of the image. {detail}"
    )
    return _user_message(base64_image=base64_image, text=text)


def _prepare_classification_prompt(
    base64_image: str, classes: List[str], **_
) -> List[dict]:
    text = (
        f"{_TASK_CLASSIFICATION}\n\n"
        f"List of all classes to be recognised by model: {', '.join(classes)}"
    )
    return _user_message(base64_image=base64_image, text=text)


def _prepare_multi_label_classification_prompt(
    base64_image: str, classes: List[str], **_
) -> List[dict]:
    text = (
        f"{_TASK_MULTI_LABEL}\n\n"
        f"List of all classes to be recognised by model: {', '.join(classes)}"
    )
    return _user_message(base64_image=base64_image, text=text)


def _prepare_structured_answering_prompt(
    base64_image: str, output_structure: Dict[str, str], **_
) -> List[dict]:
    text = (
        f"{_TASK_STRUCTURED}\n\n"
        "Specification of requirements regarding output fields: \n"
        f"{json.dumps(output_structure, indent=4)}"
    )
    return _user_message(base64_image=base64_image, text=text)


def _prepare_object_detection_prompt(
    base64_image: str, classes: List[str], box_format: str, **_
) -> List[dict]:
    text = build_object_detection_prompt(box_format=box_format, classes=classes)
    return _user_message(base64_image=base64_image, text=text)


PROMPT_BUILDERS = {
    "unconstrained": _prepare_unconstrained_prompt,
    "ocr": _prepare_ocr_prompt,
    "visual-question-answering": _prepare_vqa_prompt,
    "caption": lambda **kwargs: _prepare_caption_prompt(
        short_description=True, **kwargs
    ),
    "detailed-caption": lambda **kwargs: _prepare_caption_prompt(
        short_description=False, **kwargs
    ),
    "classification": _prepare_classification_prompt,
    "multi-label-classification": _prepare_multi_label_classification_prompt,
    "structured-answering": _prepare_structured_answering_prompt,
    "object-detection": _prepare_object_detection_prompt,
}


def build_zai_openrouter_prompts(
    images: List[np.ndarray],
    task_type: str,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    box_format: str = DEFAULT_DETECTION_BOX_FORMAT,
) -> List[List[dict]]:
    """Build one OpenRouter ``messages`` array per input image, GLM-style.

    Every task sends a single user message with the image part first and
    the instruction text second, with no system role, matching the
    vlm-exam GLM request contract.

    Args:
        images: BGR numpy images.
        task_type: One of the supported VLM task types.
        prompt: User prompt for unconstrained / VQA tasks.
        output_structure: Output spec for structured-answering.
        classes: Class list for classification / detection tasks.
        box_format: Per-model detection coordinate contract, from
            ``MODEL_VARIANTS``.

    Returns:
        List of ``messages`` arrays, one per image.

    Raises:
        ValueError: If the task type is not supported.
    """
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    builder = PROMPT_BUILDERS[task_type]
    built: List[List[dict]] = []
    for image in images:
        built.append(
            builder(
                base64_image=encode_image_for_zai_openrouter(numpy_image=image),
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
                box_format=box_format,
            )
        )
    return built


RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)

LONG_DESCRIPTION = f"""
Run Z.ai GLM vision-language models via [OpenRouter](https://openrouter.ai/).

Supported models: GLM 5V Turbo and GLM 5.3 Flash (previously served as the
OpenRouter stealth model "Ox Alpha").

You can specify arbitrary text prompts or predefined ones. The block supports:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

Object detection uses the per-model format validated in the vlm-exam
benchmarks. Both models emit `box_2d`/`label` entries normalized to
0-1000, but with different axis orders: GLM 5V Turbo uses
`[x_min, y_min, x_max, y_max]` and GLM 5.3 Flash uses
`[y_min, x_min, y_max, x_max]`. The block prompts for and decodes the
right one for the selected model.

## Version Differences

This version (v2) decodes the model answer inside the block, adding
`predictions`, `error_status` and `inference_id` outputs next to the raw
`output` string:

* `predictions` holds object detections for the `object-detection` task and a
classification prediction for the `classification` /
`multi-label-classification` tasks - the kind of the output follows the
selected task type.
* `predictions` is `None` for every other task (unconstrained prompting, OCR,
captioning, structured answering, visual question answering).
* `error_status` is `True` when the answer could not be decoded.
* `inference_id` is generated per image.

Separate `roboflow_core/vlm_as_detector@v2` and
`roboflow_core/vlm_as_classifier@v2` steps are no longer needed.

Every request sends the image before the instruction text in a single user
message. GLM models default to extended reasoning on OpenRouter; the block
disables it by default (the benchmarked configuration) and exposes a
`reasoning_effort` knob to turn it back on. GLM 5.3 Flash requires
reasoning and falls back to low effort when disabled. `max_tokens`
defaults to 2048; raise it (e.g. 8192) when you need a longer answer.

By default the block uses the Roboflow-managed OpenRouter key and bills
your Roboflow credits. Paste your own `sk-or-...` key to call OpenRouter
directly.
"""


class BlockManifest(OpenRouterBlockManifestMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Z.ai",
            "version": "v2",
            "short_description": "Run Z.ai GLM vision models via OpenRouter.",
            "long_description": LONG_DESCRIPTION,
            "license": "MIT",
            "block_type": "model",
            "search_keywords": [
                "LMM",
                "VLM",
                "GLM",
                "GLM-5V",
                "Z.ai",
                "Zhipu",
                "OpenRouter",
            ],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.56,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/zai_vlm@v2"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_version: Union[Selector(kind=[STRING_KIND]), ModelVersion] = Field(
        default=DEFAULT_MODEL_VERSION,
        description="Z.ai GLM model to run.",
        examples=[DEFAULT_MODEL_VERSION, "GLM 5.3 Flash"],
        json_schema_extra={
            "values_metadata": MODEL_VERSION_METADATA,
        },
    )
    task_type: TaskType = Field(
        default="unconstrained",
        description=(
            "Task type to be performed by model. Value determines required "
            "parameters and output response."
        ),
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
            "recommended_parsers": RECOMMENDED_PARSERS,
            "always_visible": True,
        },
    )
    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to send to the model.",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["unconstrained", "visual-question-answering"],
                    "required": True,
                },
            },
            "multiline": True,
        },
    )
    output_structure: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary with structure of expected JSON response.",
        examples=[{"my_key": "description"}, "$inputs.output_structure"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": ["structured-answering"], "required": True},
            },
        },
    )
    classes: Optional[Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]]] = Field(
        default=None,
        description="List of classes to be used.",
        examples=[["class-a", "class-b"], "$inputs.classes"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": [
                        "classification",
                        "multi-label-classification",
                        "object-detection",
                    ],
                    "required": True,
                },
            },
        },
    )
    reasoning_effort: ReasoningEffort = Field(
        default=DEFAULT_REASONING_EFFORT,
        description=(
            "Extended-reasoning budget. GLM models default to extended "
            "reasoning on OpenRouter; `none` (the default) disables it, "
            "matching the configuration validated in the vlm-exam benches."
        ),
        json_schema_extra={"values_metadata": REASONING_EFFORT_METADATA},
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        description=(
            "Maximum number of tokens the model can generate in its response. "
            f"Defaults to {DEFAULT_MAX_TOKENS}. Raise it explicitly "
            "(e.g. 8192) when a task needs a longer answer. Billing is based "
            "on tokens actually generated, not on this limit."
        ),
        gt=1,
    )
    temperature: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description=(
            "Sampling temperature. Left unset by default so the provider "
            "default applies. Range 0.0-2.0."
        ),
    )
    api_key: Union[
        Selector(kind=[STRING_KIND, SECRET_KIND, ROBOFLOW_MANAGED_KEY]), str
    ] = Field(
        default="rf_key:account",
        description=(
            "OpenRouter API key. Defaults to Roboflow's managed key. Provide "
            "your own `sk-or-...` key to call OpenRouter directly without "
            "Roboflow billing."
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

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        validate_task_type_required_fields(
            task_type=self.task_type,
            prompt=self.prompt,
            classes=self.classes,
            output_structure=self.output_structure,
        )
        validate_reasoning_level(
            model=self.model_version,
            level=self.reasoning_effort,
            levels_by_model=MODEL_REASONING_LEVELS,
        )
        return self

    @field_validator("temperature")
    @classmethod
    def validate_temperature(
        cls, value: Optional[Union[str, float]]
    ) -> Optional[Union[str, float]]:
        if value is None or isinstance(value, str):
            return value
        if value < 0.0 or value > 2.0:
            raise ValueError(
                "'temperature' parameter required to be in range [0.0, 2.0]"
            )
        return value

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            *cls._describe_raw_outputs(),
            *describe_vlm_prediction_outputs(),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        return [
            *self._describe_raw_outputs(),
            *actual_vlm_prediction_outputs(self.task_type),
        ]

    @classmethod
    def _describe_raw_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
            OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(
                name="thinking",
                kind=[STRING_KIND],
                description=(
                    "Reasoning trace from the model when OpenRouter returns "
                    "one. Empty string otherwise."
                ),
            ),
            *TOKEN_OUTPUT_DEFINITIONS,
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        if is_workflow_selector(self.model_version):
            return [
                third_party_model(
                    provider="openrouter",
                    model_id=self.model_version,
                    model_id_resolver=lambda label: MODEL_IDS.get(label),
                )
            ]
        return [
            third_party_model(
                provider="openrouter",
                model_id=MODEL_IDS[self.model_version],
            )
        ]


class ZaiVlmBlockV2(OpenRouterWorkflowBlockBase):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        reasoning_effort: str,
        api_key: str,
        privacy_level: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        variant = MODEL_VARIANTS.get(model_version)
        if variant is None:
            raise ValueError(
                f"Unknown Z.ai model '{model_version}'. "
                f"Pick one of: {list(MODEL_VARIANTS)}"
            )
        validate_reasoning_level(
            model=model_version,
            level=reasoning_effort,
            levels_by_model=MODEL_REASONING_LEVELS,
        )
        prompts = build_zai_openrouter_prompts(
            images=[image.numpy_image for image in images],
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            box_format=variant["box_format"],
        )
        results = self.execute_openrouter_batch_with_usage(
            openrouter_api_key=api_key,
            model=variant["model_id"],
            prompts=prompts,
            max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
            temperature=temperature,
            privacy_level=privacy_level,
            max_concurrent_requests=max_concurrent_requests,
            reasoning=build_zai_reasoning_config(
                reasoning_effort,
                reasoning_required=variant.get("reasoning_required", False),
            ),
        )
        outputs = []
        for image, result in zip(images, results):
            inference_id = str(uuid4())
            error_status, predictions = decode_vlm_output(
                task_type=task_type,
                raw_output=result.content,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=variant["box_format"],
            )
            outputs.append(
                {
                    "output": result.content,
                    "classes": classes,
                    "thinking": result.reasoning_trace,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "predictions": predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )
        return outputs

"""Meta Muse vision-language block, v3.

Same surface as ``meta_vlm@v2``, plus in-block decoding of the model
answer: ``predictions``, ``error_status`` and ``inference_id`` outputs sit
next to the raw ``output`` string, so no separate "VLM as Detector" /
"VLM as Classifier" step is needed.

OpenRouter-only. Three Muse models, using the vlm-exam request contract:
image-first user message, no system role, reasoning always on, and the
shared ``named_0_1000`` detection contract (flat ``x_min`` / ``y_min`` /
``x_max`` / ``y_max`` integers normalized to 0-1000). Llama Vision stays
on its own block.
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

# Spark `none` is HTTP 400. Glimmer has no `minimal`.
MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "Muse Spark 1.1": {
        "model_id": "meta/muse-spark-1.1",
        "reasoning_levels": ["minimal", "low", "medium", "high", "xhigh"],
    },
    "Muse Spark 1.2": {
        "model_id": "meta/muse-spark-1.2",
        "reasoning_levels": ["minimal", "low", "medium", "high", "xhigh"],
    },
    "Muse Glimmer": {
        "model_id": "meta/muse-glimmer-30b",
        "reasoning_levels": ["low", "medium", "high", "xhigh"],
    },
}

MODEL_IDS = {label: variant["model_id"] for label, variant in MODEL_VARIANTS.items()}

MODEL_REASONING_LEVELS = {
    label: variant["reasoning_levels"] for label, variant in MODEL_VARIANTS.items()
}

MODEL_VERSION_METADATA = attach_reasoning_levels(
    {label: {"name": label} for label in MODEL_VARIANTS},
    MODEL_REASONING_LEVELS,
)

ModelVersion = Literal[tuple(MODEL_VARIANTS.keys())]
DEFAULT_MODEL_VERSION = "Muse Spark 1.2"

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

REASONING_EFFORT_OPTIONS = ["minimal", "low", "medium", "high", "xhigh"]
ReasoningEffort = Literal[tuple(REASONING_EFFORT_OPTIONS)]
DEFAULT_REASONING_EFFORT = "low"

REASONING_EFFORT_METADATA = {
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

DEFAULT_MAX_TOKENS = 2048
OPENROUTER_MAX_BASE64_BYTES = 9_500_000
OPENROUTER_JPEG_QUALITY = 90

# Muse grounds objects with flat `x_min`/`y_min`/`x_max`/`y_max` integers
# normalized to 0-1000 - the shared `named_0_1000` contract, whose prompt
# wording is the vlm-exam `_META_FLAT_NORMALIZED_PROMPT_TEMPLATE` this block
# was benchmarked with.
DETECTION_BOX_FORMAT = "named_0_1000"

# Detection and classification answers are decoded in-block now, so only
# `structured-answering` still points at a downstream parser.
RECOMMENDED_PARSERS = {
    "structured-answering": "roboflow_core/json_parser@v1",
}

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


def encode_image_for_muse_openrouter(numpy_image: np.ndarray) -> str:
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
    base64_image: str, classes: List[str], **_
) -> List[dict]:
    text = build_object_detection_prompt(
        box_format=DETECTION_BOX_FORMAT,
        classes=classes,
    )
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


def build_muse_openrouter_prompts(
    images: List[np.ndarray],
    task_type: str,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> List[List[dict]]:
    """Build one OpenRouter ``messages`` array per input image, Muse-style.

    Every task sends a single user message with the image part first and
    the instruction text second, with no system role, matching the
    vlm-exam Muse request contract.

    Args:
        images: BGR numpy images.
        task_type: One of the supported VLM task types.
        prompt: User prompt for unconstrained / VQA tasks.
        output_structure: Output spec for structured-answering.
        classes: Class list for classification / detection tasks.

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
                base64_image=encode_image_for_muse_openrouter(numpy_image=image),
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
        )
    return built


def build_reasoning_config(reasoning_effort: str) -> Dict[str, Any]:
    """Build the OpenRouter ``reasoning`` config for Muse models.

    Muse models reject disabled reasoning, so an ``effort`` is always sent.

    Args:
        reasoning_effort: One of ``REASONING_EFFORT_OPTIONS``.

    Returns:
        Reasoning config to attach to the OpenRouter request.
    """
    return {"effort": reasoning_effort}


RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)

LONG_DESCRIPTION = f"""
Run Meta Muse vision-language models via [OpenRouter](https://openrouter.ai/).

Supported models: Muse Spark 1.1, Muse Spark 1.2, and Muse Glimmer.
Llama 3.2 Vision stays on its own block.

You can specify arbitrary text prompts or predefined ones. The block supports:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

Object detection uses Muse's native grounding format: a JSON array of
`label` / `x_min` / `y_min` / `x_max` / `y_max` fields with coordinates
normalized to 0-1000.

## Version Differences

This version (v3) decodes the model answer inside the block, adding
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
message. Reasoning stays on; the `reasoning_effort` knob defaults to low.
`max_tokens` defaults to 2048; raise it (e.g. 8192) when you need a longer
answer.

By default the block uses the Roboflow-managed OpenRouter key and bills
your Roboflow credits. Paste your own `sk-or-...` key to call OpenRouter
directly.
"""


class BlockManifest(OpenRouterBlockManifestMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Meta",
            "version": "v3",
            "short_description": "Run Meta Muse vision models via OpenRouter.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "LMM",
                "VLM",
                "Meta",
                "Muse",
                "Spark",
                "Glimmer",
                "OpenRouter",
            ],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fa-brands fa-meta",
                "blockPriority": 5.55,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/meta_vlm@v3"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_version: Union[Selector(kind=[STRING_KIND]), ModelVersion] = Field(
        default=DEFAULT_MODEL_VERSION,
        description=(
            "Muse model to run. Spark 1.1, Spark 1.2, and Glimmer are the "
            "current image-capable Muse chat models on OpenRouter."
        ),
        examples=[DEFAULT_MODEL_VERSION, "Muse Glimmer"],
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
            "Reasoning budget. Muse models reject disabled reasoning, so this "
            "is always sent as an effort. Low is the default used in the "
            "vlm-exam benches. Supported values differ per model (see the "
            "model dropdown): 'minimal' is Spark-only."
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


class MetaVlmBlockV3(OpenRouterWorkflowBlockBase):
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
        model_id = MODEL_IDS.get(model_version)
        if model_id is None:
            raise ValueError(
                f"Unknown Muse variant '{model_version}'. "
                f"Pick one of: {list(MODEL_VARIANTS)}"
            )
        validate_reasoning_level(
            model=model_version,
            level=reasoning_effort,
            levels_by_model=MODEL_REASONING_LEVELS,
        )
        prompts = build_muse_openrouter_prompts(
            images=[image.numpy_image for image in images],
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        results = self.execute_openrouter_batch_with_usage(
            openrouter_api_key=api_key,
            model=model_id,
            prompts=prompts,
            max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
            temperature=temperature,
            privacy_level=privacy_level,
            max_concurrent_requests=max_concurrent_requests,
            reasoning=build_reasoning_config(reasoning_effort),
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
                box_format=DETECTION_BOX_FORMAT,
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

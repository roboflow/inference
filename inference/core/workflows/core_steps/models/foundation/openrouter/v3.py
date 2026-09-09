"""Generic OpenRouter workflow block.

Like the Qwen-VL / Kimi / Gemma OpenRouter blocks, but the model is a free-form
string instead of a fixed dropdown. The user pastes any OpenRouter model slug
(e.g. ``openai/gpt-4o-mini``, ``anthropic/claude-3.5-sonnet``,
``qwen/qwen3.6-27b``) and the block routes through Roboflow's
``apiproxy/openrouter`` proxy by default, or directly to OpenRouter when the
user provides their own ``sk-or-...`` key.

The task-type surface (unconstrained, OCR, classification, detection, etc.)
is the one shared via ``common.openrouter`` with the other VLM blocks.
Detection and classification answers are decoded in-block via
``common.vlm_decoding``.
"""

from typing import Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.workflows.core_steps.common.openrouter import (
    LEGACY_DETECTION_BOX_FORMAT,
    RELEVANT_TASKS_METADATA,
    SUPPORTED_TASK_TYPES_LIST,
    OpenRouterBlockManifestMixin,
    OpenRouterWorkflowBlockBase,
    build_prompts_from_images,
    validate_task_type_required_fields,
)
from inference.core.workflows.core_steps.common.reasoning import (
    REASONING_EFFORT_METADATA,
    REASONING_EFFORT_OPTIONS,
    build_openrouter_reasoning_config,
)
from inference.core.workflows.core_steps.common.token_usage import (
    TOKEN_OUTPUT_DEFINITIONS,
)
from inference.core.workflows.core_steps.common.vlm_decoding import (
    BoxFormatName,
    actual_vlm_prediction_outputs,
    decode_vlm_output,
    describe_vlm_prediction_outputs,
    get_detection_box_format,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    DependentResource,
    WorkflowBlockManifest,
    third_party_model,
)

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

# Default object-detection contract: the legacy OpenRouter system message in
# `common.openrouter`, which asks for `x_min`/`y_min`/`x_max`/`y_max` floats
# normalized to 0.0-1.0 (`named_normalized` in the shared decoding package).
# `detection_format` lets the user switch to any other registered contract,
# so a new model can be prompted and decoded the way it was trained without
# a block change.
DETECTION_BOX_FORMAT = LEGACY_DETECTION_BOX_FORMAT

DETECTION_FORMAT_METADATA = {
    "named_normalized": {
        "name": "Named keys, normalized 0-1 (default)",
        "description": (
            'Asks for `{"detections": [{"x_min", "y_min", "x_max", '
            '"y_max", "class_name", "confidence"}]}` with coordinates as '
            "floats in 0.0-1.0. Generic contract most chat models follow; "
            "the block's behaviour before this option existed."
        ),
    },
    "xyxy_0_1000": {
        "name": "box_2d [x_min, y_min, x_max, y_max], 0-1000",
        "description": (
            "Asks for `box_2d` integers normalized to 0-1000, x before y. "
            "The native grounding contract of Qwen-VL and Z.ai GLM models. "
            "A reported `confidence` is ignored."
        ),
    },
    "yxyx_0_1000": {
        "name": "box_2d [y_min, x_min, y_max, x_max], 0-1000",
        "description": (
            "Asks for `box_2d` integers normalized to 0-1000, y before x. "
            "The native grounding contract of Google Gemini models."
        ),
    },
    "xyxy_absolute": {
        "name": "box_2d [x_min, y_min, x_max, y_max], pixels",
        "description": (
            "Asks for `box_2d` in absolute pixels of the uploaded image, whose "
            "size is stated in the prompt. Used by the OpenAI GPT-5.6 / GPT-6 "
            "and Anthropic Claude blocks."
        ),
    },
    "xyxy_percent": {
        "name": "box_2d [x_min, y_min, x_max, y_max], percent",
        "description": (
            "Asks for `box_2d` as percentages of image width and height "
            "(floats 0-100). Used by the SpaceXAI Grok block."
        ),
    },
    "named_0_1000": {
        "name": "Named keys, 0-1000",
        "description": (
            "Asks for `label`, `x_min`, `y_min`, `x_max`, `y_max` integers "
            "normalized to 0-1000. The native grounding contract of Meta Muse "
            "models. A reported `confidence` is ignored."
        ),
    },
}

# Detections and classifications are decoded in-block from v3 on, so only the
# structured-answering parser stays relevant.
RECOMMENDED_PARSERS = {
    "structured-answering": "roboflow_core/json_parser@v1",
}


RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)


LONG_DESCRIPTION = f"""
Run **any** vision-language model available on [OpenRouter](https://openrouter.ai/) by
pasting its model slug into the `model_id` field — e.g.
`openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`, `google/gemini-2.5-pro`,
`qwen/qwen3.6-27b`.

This is the generic escape hatch for OpenRouter — when you want a model that
doesn't have a dedicated block (Qwen-VL, Kimi, Gemma, Llama Vision) and you
want to try it out without waiting for a new block to be added.

The block supports the standard VLM task-type surface:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

#### 🛠️ API key

By default the block uses the **Roboflow-managed OpenRouter key** and bills your
Roboflow credits — no extra setup needed. To bypass Roboflow billing, paste your
own `sk-or-...` key into the `api_key` field.

#### 🔒 Privacy filter

* **No data collection** *(default)* – providers may not train on your inputs.
* **Allow data collection** – broader provider pool.
* **Zero data retention** – strictest, restricts to providers that retain nothing.

!!! warning "Model availability"

    OpenRouter exposes hundreds of models with different capabilities. Not every
    model supports image inputs, and some are text-only or reasoning-only. If
    the model can't return a visible response (e.g. a reasoning model that
    burns all of `max_tokens` on internal thinking), try increasing
    `max_tokens` or pick a different model.

## Version Differences

This version (v3) decodes model answers inside the block:

* **`predictions`** - classification and object-detection answers are parsed here,
  so the deprecated `VLM as Detector` / `VLM as Classifier` blocks are no longer
  needed. The output kind follows `task_type`: object detection predictions for
  `object-detection`, classification predictions for `classification` and
  `multi-label-classification`, and `None` for every other task.
* **`error_status`** - `True` when the model answer could not be parsed.
* **`inference_id`** - identifier generated per image and attached to the decoded
  predictions.
* **`detection_format`** - for `object-detection`, picks the bounding-box contract
  the model is prompted for and decoded with. The default is the generic
  normalized-floats JSON every chat model understands; pick the model family's
  native contract (e.g. `box_2d` 0-1000 for Qwen / GLM, `y`-first for Gemini,
  pixels for GPT-5.6+) to match the wording it was trained on, so a new model
  can be used without a block change.
"""


def _base_outputs() -> List[OutputDefinition]:
    """Outputs the block produces regardless of the selected task."""
    return [
        OutputDefinition(name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]),
        OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND]),
        OutputDefinition(
            name="thinking",
            kind=[STRING_KIND],
            description=(
                "Reasoning trace when OpenRouter returns one (reasoning "
                "models with reasoning enabled). Empty string otherwise."
            ),
        ),
        *TOKEN_OUTPUT_DEFINITIONS,
    ]


class BlockManifest(OpenRouterBlockManifestMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenRouter",
            "version": "v3",
            "short_description": "Run any OpenRouter model by pasting its model slug.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "OpenRouter",
                "VLM",
                "LMM",
                "Qwen",
                "Llama",
                "generic",
            ],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-globe",
                "blockPriority": 5.6,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/openrouter@v3"]

    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    model_id: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description=(
            "OpenRouter model slug, e.g. `openai/gpt-4o-mini`, "
            "`anthropic/claude-3.5-sonnet`, `qwen/qwen3.6-27b`. See "
            "https://openrouter.ai/models for the full list."
        ),
        examples=[
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.5-pro",
            "qwen/qwen3.6-27b",
            "$inputs.openrouter_model_id",
        ],
    )

    # Overrides the mixin default of 500, which reasoning models can burn
    # entirely on internal thinking and fail with missing content.
    max_tokens: int = Field(
        default=2048,
        description=(
            "Maximum number of tokens the model can generate in its response. "
            "Defaults to 2048. Raise it explicitly (e.g. 8192) when a task "
            "needs a longer answer. Billing is based on tokens actually "
            "generated, not on this limit."
        ),
        gt=1,
    )

    reasoning_effort: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            Literal[tuple(REASONING_EFFORT_OPTIONS)],
        ]
    ] = Field(
        default=None,
        description=(
            "Extended-reasoning budget forwarded to OpenRouter as "
            '`reasoning: {"effort": ...}`. Unset keeps the model\'s '
            "provider-default behavior; `none` explicitly disables reasoning. "
            "Models that reject the config are retried without it. Reasoning "
            "tokens count toward `max_tokens`, so raise it for medium/high."
        ),
        examples=["low", "$inputs.reasoning_effort"],
        json_schema_extra={"values_metadata": REASONING_EFFORT_METADATA},
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

    detection_format: Union[Selector(kind=[STRING_KIND]), BoxFormatName] = Field(
        default=DETECTION_BOX_FORMAT,
        description=(
            "Bounding-box contract used for `object-detection`: both the "
            "prompt wording the model receives and the decoding of its "
            "answer. Keep the default for models without a known native "
            "format, or pick the family's contract for best accuracy."
        ),
        examples=["named_normalized", "xyxy_0_1000", "$inputs.detection_format"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": ["object-detection"], "required": False},
            },
            "values_metadata": DETECTION_FORMAT_METADATA,
        },
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        validate_task_type_required_fields(
            task_type=self.task_type,
            prompt=self.prompt,
            classes=self.classes,
            output_structure=self.output_structure,
        )
        return self

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: Union[str, float]) -> Union[str, float]:
        if isinstance(value, str):
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
        return _base_outputs() + describe_vlm_prediction_outputs()

    def get_actual_outputs(self) -> List[OutputDefinition]:
        return _base_outputs() + actual_vlm_prediction_outputs(self.task_type)

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [third_party_model(provider="openrouter", model_id=self.model_id)]


class OpenRouterBlockV3(OpenRouterWorkflowBlockBase):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: str,
        privacy_level: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str],
        max_concurrent_requests: Optional[int],
        detection_format: str = DETECTION_BOX_FORMAT,
    ) -> BlockResult:
        if task_type == "object-detection":
            # Fail on an unknown format before any paid request is sent.
            box_format = get_detection_box_format(detection_format)
        else:
            box_format = None
        inference_images = [i.to_inference_format() for i in images]
        prompts = build_prompts_from_images(
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            detection_box_format=detection_format,
        )
        results = self.execute_openrouter_batch_with_usage(
            openrouter_api_key=api_key,
            model=model_id,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            max_concurrent_requests=max_concurrent_requests,
            reasoning=build_openrouter_reasoning_config(reasoning_effort),
        )
        predictions = []
        for image, result in zip(images, results):
            inference_id = str(uuid4())
            upload_width = upload_height = None
            if box_format is not None and box_format.requires_upload_dimensions:
                # The OpenRouter path uploads the image at its original size.
                upload_height, upload_width = image.numpy_image.shape[:2]
            error_status, decoded_predictions = decode_vlm_output(
                task_type=task_type,
                raw_output=result.content,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=detection_format,
                upload_width=upload_width,
                upload_height=upload_height,
            )
            predictions.append(
                {
                    "output": result.content,
                    "classes": classes,
                    "thinking": result.reasoning_trace,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "predictions": decoded_predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )
        return predictions

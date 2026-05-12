"""Generic OpenRouter workflow block.

Like the Qwen-VL / Kimi / Gemma OpenRouter blocks, but the model is a free-form
string instead of a fixed dropdown. The user pastes any OpenRouter model slug
(e.g. ``openai/gpt-4o-mini``, ``anthropic/claude-3.5-sonnet``,
``qwen/qwen3.6-27b``) and the block routes through Roboflow's
``apiproxy/openrouter`` proxy by default, or directly to OpenRouter when the
user provides their own ``sk-or-...`` key.

The task-type surface (unconstrained, OCR, classification, detection, etc.)
is the one shared via ``common.openrouter`` with the other VLM blocks.
"""

from typing import Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.workflows.core_steps.common.openrouter import (
    RECOMMENDED_PARSERS,
    RELEVANT_TASKS_METADATA,
    SUPPORTED_TASK_TYPES_LIST,
    OpenRouterBlockManifestMixin,
    OpenRouterWorkflowBlockBase,
    build_prompts_from_images,
    validate_task_type_required_fields,
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
    WorkflowBlockManifest,
)

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]


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
"""


class BlockManifest(OpenRouterBlockManifestMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenRouter",
            "version": "v1",
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
    type: Literal["roboflow_core/openrouter@v1"]

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
        return [
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
            OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OpenRouterBlockV1(OpenRouterWorkflowBlockBase):

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
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        prompts = build_prompts_from_images(
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        raw_outputs = self.execute_openrouter_batch(
            openrouter_api_key=api_key,
            model=model_id,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]

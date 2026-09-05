from typing import Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.workflows.core_steps.common.openrouter import (
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
    actual_vlm_prediction_outputs,
    decode_vlm_output,
    describe_vlm_prediction_outputs,
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
    is_workflow_selector,
    third_party_model,
)

MODEL_VERSION_MAPPING = {
    "Gemma 4 31B - OpenRouter": "google/gemma-4-31b-it",
    "Gemma 4 26B A4B - OpenRouter": "google/gemma-4-26b-a4b-it",
}

ModelVersion = Literal[
    "Gemma 4 31B - OpenRouter",
    "Gemma 4 26B A4B - OpenRouter",
]

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

# The object-detection prompt this block sends (the legacy OpenRouter JSON
# contract in `common.openrouter`) asks for `x_min`/`y_min`/`x_max`/`y_max`
# floats normalized to 0.0-1.0 - the `named_normalized` box format of the
# shared decoding package.
DETECTION_BOX_FORMAT = "named_normalized"

# Detections and classifications are decoded in-block from this version on, so
# only the structured-answering parser stays relevant.
RECOMMENDED_PARSERS = {
    "structured-answering": "roboflow_core/json_parser@v1",
}

RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)

LONG_DESCRIPTION = f"""
Ask a question to Google's Gemma model with vision capabilities.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

#### 🛠️ API providers and model variants

Gemma is exposed via [OpenRouter](https://openrouter.ai/). By default this block uses
the **Roboflow-managed OpenRouter key** and bills your Roboflow credits — no extra
setup needed. To bypass Roboflow billing, paste your own `sk-or-...` key into the
`api_key` field.

The `privacy_level` field controls which OpenRouter providers may serve the request:

* **No data collection** *(default)* – providers may not train on your inputs.
* **Allow data collection** – broader provider pool, including providers that train on inputs.
* **Zero data retention** – strictest, restricts to providers that retain nothing.

#### 💡 Further reading and Acceptable Use Policy

!!! warning "Model license"

    Check the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) before use.

## Version Differences

This version (v4) decodes model answers inside the block:

* **`predictions`** - classification and object-detection answers are parsed here,
  so the deprecated `VLM as Detector` / `VLM as Classifier` blocks are no longer
  needed. The output kind follows `task_type`: object detection predictions for
  `object-detection`, classification predictions for `classification` and
  `multi-label-classification`, and `None` for every other task.
* **`error_status`** - `True` when the model answer could not be parsed.
* **`inference_id`** - identifier generated per image and attached to the decoded
  predictions.
"""


def _base_outputs() -> List[OutputDefinition]:
    """Outputs the block produces regardless of the selected task."""
    return [
        OutputDefinition(name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]),
        OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND]),
        *TOKEN_OUTPUT_DEFINITIONS,
    ]


class BlockManifest(OpenRouterBlockManifestMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Google Gemma",
            "version": "v4",
            "short_description": "Run Google's Gemma model with vision capabilities via OpenRouter.",
            "long_description": LONG_DESCRIPTION,
            "license": "Gemma Terms of Use",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Gemma", "Google", "OpenRouter"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fa-brands fa-google",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/google_gemma@v4"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    task_type: TaskType = Field(
        default="unconstrained",
        description="Task type to be performed by model. Value determines required parameters and output response.",
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
            "recommended_parsers": RECOMMENDED_PARSERS,
            "always_visible": True,
        },
    )
    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to the Gemma model",
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
        description="Dictionary with structure of expected JSON response",
        examples=[{"my_key": "description"}, "$inputs.output_structure"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": ["structured-answering"], "required": True},
            },
        },
    )
    classes: Optional[Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]]] = Field(
        default=None,
        description="List of classes to be used",
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
    model_version: Union[Selector(kind=[STRING_KIND]), ModelVersion] = Field(
        default="Gemma 4 31B - OpenRouter",
        description="Model to be used",
        examples=["Gemma 4 31B - OpenRouter", "$inputs.gemma_model"],
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
            "Models that reject the config are retried without it."
        ),
        examples=["low", "$inputs.reasoning_effort"],
        json_schema_extra={"values_metadata": REASONING_EFFORT_METADATA},
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
        if is_workflow_selector(self.model_version):
            # Friendly-label selector returned verbatim; the attached resolver
            # performs the MODEL_VERSION_MAPPING lookup once the input value
            # is substituted.
            return [
                third_party_model(
                    provider="openrouter",
                    model_id=self.model_version,
                    model_id_resolver=lambda label: MODEL_VERSION_MAPPING[label],
                )
            ]
        return [
            third_party_model(
                provider="openrouter",
                model_id=MODEL_VERSION_MAPPING[self.model_version],
            )
        ]


class GoogleGemmaBlockV4(OpenRouterWorkflowBlockBase):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: str,
        privacy_level: str,
        model_version: ModelVersion,
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str],
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
        results = self.execute_openrouter_batch_with_usage(
            openrouter_api_key=api_key,
            model=MODEL_VERSION_MAPPING[model_version],
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
            error_status, decoded_predictions = decode_vlm_output(
                task_type=task_type,
                raw_output=result.content,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=DETECTION_BOX_FORMAT,
            )
            predictions.append(
                {
                    "output": result.content,
                    "classes": classes,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "predictions": decoded_predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )
        return predictions

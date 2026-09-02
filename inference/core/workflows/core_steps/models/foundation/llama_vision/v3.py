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

# OpenRouter no longer serves any Llama 3.2 vision variant - the paid 11B that
# v2 shipped, the :free tier and the 90B variant that v1 listed have all been
# removed from OpenRouter's catalog. From v3 on the block runs on Llama 4
# instead, which OpenRouter serves as the Scout and Maverick variants.
MODEL_VERSION_MAPPING = {
    "Llama 4 Scout - OpenRouter": "meta-llama/llama-4-scout",
    "Llama 4 Maverick - OpenRouter": "meta-llama/llama-4-maverick",
}

ModelVersion = Literal["Llama 4 Scout - OpenRouter", "Llama 4 Maverick - OpenRouter"]

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
Ask a question to Llama 4 Vision model.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

#### 🛠️ API providers and model variants

Llama 4 Vision is exposed via [OpenRouter](https://openrouter.ai/) in two variants:

* **Llama 4 Scout** *(default)* – `meta-llama/llama-4-scout`.
* **Llama 4 Maverick** – `meta-llama/llama-4-maverick`, the larger variant.

By default this block uses the **Roboflow-managed OpenRouter key** and bills your
Roboflow credits — no extra setup needed. To bypass Roboflow billing, paste your own
`sk-or-...` key into the `api_key` field.

The `privacy_level` field controls which OpenRouter providers may serve the request:

* **No data collection** *(default)* – providers may not train on your inputs.
* **Allow data collection** – broader provider pool.
* **Zero data retention** – strictest, restricts to providers that retain nothing.

#### 💡 Further reading and Acceptable Use Policy

!!! warning "Model license"

    Check the [Llama 4 license](https://www.llama.com/llama4/license/) before use.

## Version Differences

This version (v3) moves to Llama 4 and decodes model answers inside the block:

* **model** - v3 swaps the delisted Llama 3.2 11B model, which OpenRouter no longer
  serves, for Llama 4 Scout (default) and Llama 4 Maverick.
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
    ]


class BlockManifest(OpenRouterBlockManifestMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Llama 4 Vision",
            "version": "v3",
            "short_description": "Run Llama 4 Scout / Maverick via OpenRouter.",
            "long_description": LONG_DESCRIPTION,
            "license": "Llama 4 Community License",
            "block_type": "model",
            "search_keywords": [
                "LMM",
                "VLM",
                "Llama",
                "Llama 4",
                "Scout",
                "Maverick",
                "Meta",
                "OpenRouter",
            ],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fa-brands fa-meta",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/llama_vision@v3"]
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
        description="Text prompt to the Llama model",
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
        default="Llama 4 Scout - OpenRouter",
        description="Model to be used",
        examples=["Llama 4 Scout - OpenRouter", "$inputs.llama_model"],
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


class LlamaVisionBlockV3(OpenRouterWorkflowBlockBase):

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
            model=MODEL_VERSION_MAPPING[model_version],
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            max_concurrent_requests=max_concurrent_requests,
        )
        predictions = []
        for image, raw_output in zip(images, raw_outputs):
            inference_id = str(uuid4())
            error_status, decoded_predictions = decode_vlm_output(
                task_type=task_type,
                raw_output=raw_output,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=DETECTION_BOX_FORMAT,
            )
            predictions.append(
                {
                    "output": raw_output,
                    "classes": classes,
                    "predictions": decoded_predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )
        return predictions

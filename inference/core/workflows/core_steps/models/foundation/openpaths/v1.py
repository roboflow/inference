"""Generic OpenPaths workflow block.

Mirrors the generic OpenRouter block: the model is a free-form string instead
of a fixed dropdown. The user pastes any OpenPaths model slug (e.g.
``openpaths/auto``, ``openpaths/auto-vision``) and the block calls the
OpenAI-compatible OpenPaths gateway (https://openpaths.io) directly.

OpenPaths (https://openpaths.io) is an OpenAI-compatible gateway (like
OpenRouter). The block talks to it through the OpenAI SDK with the base URL
set explicitly to ``https://openpaths.io/v1`` — there is no Roboflow proxy in
the loop. The ``api_key`` defaults to the ``OPENPATHS_API_KEY`` environment
variable.

The task-type surface (unconstrained, OCR, classification, detection, etc.) is
the one shared via ``common.openrouter`` with the other VLM blocks.
"""

import os
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

from openai import OpenAI
from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.env import (
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.workflows.core_steps.common.openrouter import (
    RECOMMENDED_PARSERS,
    RELEVANT_TASKS_METADATA,
    SUPPORTED_TASK_TYPES_LIST,
    build_prompts_from_images,
    validate_task_type_required_fields,
)
from inference.core.workflows.core_steps.common.utils import run_in_parallel
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
    SECRET_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OPENPATHS_BASE_URL = "https://openpaths.io/v1"
OPENPATHS_API_KEY_ENV = "OPENPATHS_API_KEY"
DEFAULT_OPENPATHS_MODEL = "openpaths/auto"

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]


RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)


LONG_DESCRIPTION = f"""
Run **any** vision-language model available on
[OpenPaths](https://openpaths.io/) by pasting its model slug into the
`model_id` field — e.g. `openpaths/auto`, `openpaths/auto-vision`.

OpenPaths is an OpenAI-compatible gateway (like OpenRouter). This block talks to
it directly via the OpenAI SDK with the base URL set to
`https://openpaths.io/v1`, so you can try any model OpenPaths exposes without
waiting for a dedicated block.

The block supports the standard VLM task-type surface:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

#### 🛠️ API key

Provide your OpenPaths API key in the `api_key` field, or leave it empty to use
the `OPENPATHS_API_KEY` environment variable.

!!! warning "Model availability"

    OpenPaths exposes many models with different capabilities. Not every model
    supports image inputs, and some are text-only or reasoning-only. If the
    model can't return a visible response (e.g. a reasoning model that burns all
    of `max_tokens` on internal thinking), try increasing `max_tokens` or pick a
    different model. For vision tasks use a vision-capable model such as
    `openpaths/auto-vision`.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenPaths",
            "version": "v1",
            "short_description": "Run any OpenPaths model by pasting its model slug.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "OpenPaths",
                "VLM",
                "LMM",
                "OpenAI",
                "gateway",
                "generic",
            ],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-globe",
                "blockPriority": 5.61,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/open_paths@v1"]

    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    model_id: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_OPENPATHS_MODEL,
        description=(
            "OpenPaths model slug, e.g. `openpaths/auto`, "
            "`openpaths/auto-vision`. See https://openpaths.io/v1/models for "
            "the full list."
        ),
        examples=[
            "openpaths/auto",
            "openpaths/auto-vision",
            "$inputs.openpaths_model_id",
        ],
    )

    api_key: Optional[
        Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str]
    ] = Field(
        default=None,
        description=(
            "OpenPaths API key. If not given, the `OPENPATHS_API_KEY` "
            "environment variable is used."
        ),
        examples=["sk-...", "$inputs.openpaths_api_key"],
        private=True,
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


def execute_openpaths_request(
    client: OpenAI,
    model: str,
    messages: List[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    """Run a single OpenPaths chat-completion call via the OpenAI SDK."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response.choices is None:
        error_detail = getattr(response, "error", {}) or {}
        if isinstance(error_detail, dict):
            error_detail = error_detail.get("message", "N/A")
        raise RuntimeError(
            "OpenPaths provider failed in delivering response. "
            f"Details: {error_detail}"
        )
    content = response.choices[0].message.content
    if content is None:
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        hint = (
            " The model may be a reasoning model that ran out of tokens during "
            "its internal thinking step. Try increasing `max_tokens` (e.g. "
            "1000+) or pick a non-reasoning model variant."
            if finish_reason == "length"
            else " This can happen when the model returns only tool calls or "
            "reasoning tokens. Try a different prompt or model."
        )
        raise RuntimeError("OpenPaths response missing message.content." + hint)
    return content


class OpenPathsBlockV1(WorkflowBlock):

    def __init__(self):
        self._client_cache: Dict[str, OpenAI] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def _get_client(self, api_key: str) -> OpenAI:
        client = self._client_cache.get(api_key)
        if client is None:
            client = OpenAI(base_url=OPENPATHS_BASE_URL, api_key=api_key)
            self._client_cache[api_key] = client
        return client

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Optional[str],
        max_tokens: int,
        temperature: float,
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        resolved_api_key = api_key or os.getenv(OPENPATHS_API_KEY_ENV)
        if not resolved_api_key:
            raise ValueError(
                "OpenPaths block requires an API key. Provide `api_key` or set "
                f"the `{OPENPATHS_API_KEY_ENV}` environment variable."
            )
        client = self._get_client(resolved_api_key)
        inference_images = [i.to_inference_format() for i in images]
        prompts = build_prompts_from_images(
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        tasks = [
            partial(
                execute_openpaths_request,
                client=client,
                model=model_id,
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
        raw_outputs = run_in_parallel(tasks=tasks, max_workers=max_workers)
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]

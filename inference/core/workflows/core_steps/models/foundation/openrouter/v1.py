from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, model_validator

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.openrouter import (
    execute_openrouter_requests,
)
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.models.foundation.openai_compatible.v1 import (
    _build_messages,
    _build_prompt_content,
    _resolve_parameters,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

CUSTOM_MODEL_CHOICE = "Custom model slug"

OPENROUTER_MODEL_VERSION_MAPPING = {
    "Gemma 4 31B - OpenRouter": "google/gemma-4-31b-it",
    "Gemma 4 26B A4B - OpenRouter": "google/gemma-4-26b-a4b-it",
    "Kimi K2.5 - OpenRouter": "moonshotai/kimi-k2.5",
    "Kimi K2.6 - OpenRouter": "moonshotai/kimi-k2.6",
    "Llama 3.2 Vision 11B (Free) - OpenRouter": "meta-llama/llama-3.2-11b-vision-instruct:free",
    "Llama 3.2 Vision 11B - OpenRouter": "meta-llama/llama-3.2-11b-vision-instruct",
    "Llama 3.2 Vision 90B (Free) - OpenRouter": "meta-llama/llama-3.2-90b-vision-instruct:free",
    "Llama 3.2 Vision 90B - OpenRouter": "meta-llama/llama-3.2-90b-vision-instruct",
    "Qwen 3.5 9B - OpenRouter": "qwen/qwen3.5-9b",
    "Qwen 3.5 27B - OpenRouter": "qwen/qwen3.5-27b",
    "Qwen 3.5 122B A10B - OpenRouter": "qwen/qwen3.5-122b-a10b",
    "Qwen 3.5 397B A17B - OpenRouter": "qwen/qwen3.5-397b-a17b",
    "Qwen 3.5 Flash 02-23 - OpenRouter": "qwen/qwen3.5-flash-02-23",
    "Qwen 3.5 Plus 20260420 - OpenRouter": "qwen/qwen3.5-plus-20260420",
    "Qwen 3.6 27B - OpenRouter": "qwen/qwen3.6-27b",
    "Qwen 3.6 35B A3B - OpenRouter": "qwen/qwen3.6-35b-a3b",
    "Qwen 3.6 Flash - OpenRouter": "qwen/qwen3.6-flash",
    "Qwen 3.6 Plus - OpenRouter": "qwen/qwen3.6-plus",
    "Qwen 3.6 Max Preview - OpenRouter": "qwen/qwen3.6-max-preview",
}

OPENROUTER_MODEL_VERSION_IDS = [
    *OPENROUTER_MODEL_VERSION_MAPPING.keys(),
    CUSTOM_MODEL_CHOICE,
]

OPENROUTER_MODEL_VERSION_METADATA = {
    name: {"model_id": model_id}
    for name, model_id in OPENROUTER_MODEL_VERSION_MAPPING.items()
}
OPENROUTER_MODEL_VERSION_METADATA[CUSTOM_MODEL_CHOICE] = {
    "description": "Enter any OpenRouter chat-completions-compatible model slug."
}

OpenRouterModelVersion = Literal[tuple(OPENROUTER_MODEL_VERSION_IDS)]
PrivacyLevel = Literal["deny", "allow", "zdr"]

LONG_DESCRIPTION = """
Send a prompt to an OpenRouter chat-completions-compatible model.

The block uses OpenRouter's OpenAI-compatible chat completions API. It can use
your direct OpenRouter API key, or `rf_key:account` to proxy requests through
Roboflow and bill usage as Roboflow credits.

Pick a model from the curated list, or choose `Custom model slug` to provide an
OpenRouter model slug directly. The Roboflow proxy validates managed-key
requests against OpenRouter's model catalog at request time.

Prompt parameters work the same way as the OpenAI-Compatible LLM block:
non-image values are substituted into the prompt template, and image values are
sent as `image_url` content parts for models that support vision input.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenRouter",
            "version": "v1",
            "short_description": "Run chat-completions-compatible models via OpenRouter.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "LLM",
                "VLM",
                "OpenRouter",
                "OpenAI-compatible",
                "Qwen",
                "Kimi",
                "Gemma",
                "Llama",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/openrouter@v1"]
    api_key: Union[
        Selector(kind=[STRING_KIND, SECRET_KIND, ROBOFLOW_MANAGED_KEY]), str
    ] = Field(
        default="rf_key:account",
        description="Your OpenRouter API key or 'rf_key:account' to use Roboflow's managed API key",
        examples=["rf_key:account", "sk-or-...", "$inputs.open_router_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        OpenRouterModelVersion,
    ] = Field(
        default="Qwen 3.6 35B A3B - OpenRouter",
        description="OpenRouter model to use.",
        examples=["Qwen 3.6 35B A3B - OpenRouter", "$inputs.openrouter_model"],
        json_schema_extra={
            "values_metadata": OPENROUTER_MODEL_VERSION_METADATA,
            "always_visible": True,
        },
    )
    custom_model_slug: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="OpenRouter model slug to use when model_version is 'Custom model slug'.",
        examples=["qwen/qwen3.6-35b-a3b", "$inputs.openrouter_model_slug"],
        json_schema_extra={
            "relevant_for": {
                "model_version": {
                    "values": [CUSTOM_MODEL_CHOICE],
                    "required": True,
                },
            },
        },
    )
    system_prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Optional system prompt to set model behavior.",
        examples=["You are a helpful assistant.", "$inputs.system_prompt"],
        json_schema_extra={
            "multiline": True,
        },
    )
    prompt: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Prompt template with optional {{ $parameters.param_name }} placeholders.",
        examples=[
            "Describe what you see in the image.",
            "Count the {{ $parameters.object_type }} in the image.",
        ],
        json_schema_extra={
            "multiline": True,
            "always_visible": True,
        },
    )
    prompt_parameters: Dict[
        str,
        Union[Selector(), str, int, float, bool],
    ] = Field(
        description="Dictionary mapping parameter names to workflow data sources.",
        examples=[
            {
                "detections": "$steps.model.predictions",
                "frames": "$steps.image_stack.frames",
            }
        ],
        default_factory=dict,
        json_schema_extra={
            "always_visible": True,
        },
    )
    prompt_parameters_operations: Dict[str, List[AllOperationsType]] = Field(
        description="Optional UQL operation chains to transform parameter values before insertion.",
        examples=[
            {
                "detections": [
                    {
                        "type": "DetectionsPropertyExtract",
                        "property_name": "class_name",
                    }
                ]
            }
        ],
        default_factory=dict,
    )
    privacy_level: Union[Selector(kind=[STRING_KIND]), PrivacyLevel] = Field(
        default="deny",
        description="Provider privacy routing for OpenRouter requests.",
        examples=["deny", "allow", "zdr", "$inputs.openrouter_privacy_level"],
    )
    max_tokens: int = Field(
        default=500,
        description="Maximum number of tokens the model can generate.",
        ge=1,
        le=8192,
    )
    temperature: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Sampling temperature (0.0-2.0). Higher = more random.",
        ge=0.0,
        le=2.0,
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description="Number of concurrent requests that can be executed by block. "
        "If not given, the block defaults to the globally configured Workflows value.",
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.model_version == CUSTOM_MODEL_CHOICE and not self.custom_model_slug:
            raise ValueError(
                "`custom_model_slug` must be set when `model_version` is "
                f"`{CUSTOM_MODEL_CHOICE}`."
            )
        return self

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
            OutputDefinition(name="error_status", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class OpenRouterBlockV1(WorkflowBlock):

    def __init__(self, api_key: Optional[str]):
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    def run(
        self,
        api_key: str,
        model_version: str,
        custom_model_slug: Optional[str],
        system_prompt: Optional[str],
        prompt: str,
        prompt_parameters: Dict[str, Any],
        prompt_parameters_operations: Dict[str, List[AllOperationsType]],
        privacy_level: str,
        max_tokens: int,
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        model_id = resolve_model_id(
            model_version=model_version,
            custom_model_slug=custom_model_slug,
        )
        resolved_params = _resolve_parameters(
            prompt_parameters=prompt_parameters,
            prompt_parameters_operations=prompt_parameters_operations,
        )
        text_prompt, image_parts = _build_prompt_content(
            prompt=prompt,
            resolved_params=resolved_params,
        )
        messages = _build_messages(
            system_prompt=system_prompt,
            text_prompt=text_prompt,
            image_parts=image_parts,
        )
        try:
            output = execute_openrouter_requests(
                roboflow_api_key=self._api_key,
                openrouter_api_key=api_key,
                prompts=[messages],
                model_version_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                max_concurrent_requests=max_concurrent_requests,
                privacy_level=privacy_level,
            )[0]
            return {"output": output, "error_status": ""}
        except Exception as e:
            logger.warning(
                f"OpenRouter request to {model_id} failed: {e}",
                exc_info=True,
            )
            return {"output": "", "error_status": str(e)}


def resolve_model_id(model_version: str, custom_model_slug: Optional[str]) -> str:
    if model_version == CUSTOM_MODEL_CHOICE:
        if not custom_model_slug:
            raise ValueError(
                f"`custom_model_slug` must be set when using `{CUSTOM_MODEL_CHOICE}`."
            )
        return custom_model_slug
    return OPENROUTER_MODEL_VERSION_MAPPING.get(model_version, model_version)

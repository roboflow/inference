import base64
import re
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union

from openai import OpenAI
from pydantic import ConfigDict, Field

from inference.core.logger import logger
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
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

PARAMETER_REGEX = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")

LONG_DESCRIPTION = """
Send a prompt to any OpenAI-compatible API endpoint (e.g. local Qwen, vLLM, Ollama,
LM Studio, or any service that implements the OpenAI chat completions API).

## How This Block Works

1. You provide a **base URL** (e.g. `http://localhost:8000/v1`) and a **model name**
2. Write a **prompt template** using `{{ $parameters.param_name }}` placeholders
3. Supply **prompt_parameters** mapping parameter names to workflow data selectors
4. Parameters that resolve to images — either `WorkflowImageData` objects or raw JPEG
   `bytes` (e.g. from the Image Stack block) — are automatically base64-encoded and
   sent as `image_url` content parts in the OpenAI message
5. Lists of images are supported: each image in the list becomes a separate content part
6. Non-image parameters are converted to strings and substituted into the prompt text
7. Optionally apply **UQL operations** to transform parameter values before insertion

## Image Handling

The block detects image values automatically:

- **`WorkflowImageData`** — the numpy image is JPEG-encoded then base64-encoded
- **`bytes`** — assumed to be JPEG blobs already (e.g. from the Image Stack block),
  base64-encoded directly
- **`list`** of either type — each element becomes a separate `image_url` content part

Image parameters referenced in the prompt template have their placeholders removed from
the text (the images are sent as vision content parts, not inline text).

## Example

```
prompt: "Describe the activity across these frames: {{ $parameters.context }}"
prompt_parameters:
  context: "$steps.some_step.output"
  frames: "$steps.image_stack.frames"
```

`frames` (list of JPEG bytes) becomes multiple vision content parts.
`context` (string) gets substituted into the prompt text.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenAI-Compatible LLM",
            "version": "v1",
            "short_description": "Send prompts to any OpenAI-compatible API endpoint.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "LLM",
                "VLM",
                "OpenAI",
                "vLLM",
                "Ollama",
                "Qwen",
                "compatible",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/openai_compatible@v1"]
    base_url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Base URL of the OpenAI-compatible API (e.g. http://localhost:8000/v1).",
        examples=["http://localhost:8000/v1", "$inputs.base_url"],
    )
    model_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Model name to pass in the API request.",
        examples=["Qwen/Qwen2.5-VL-7B-Instruct", "$inputs.model_name"],
    )
    api_key: Optional[Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str]] = Field(
        default=None,
        description="API key for the endpoint (if required).",
        examples=["xxx-xxx", "$inputs.api_key"],
        private=True,
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
        description="Prompt template with optional {{ $parameters.param_name }} placeholders. "
        "Non-image parameters are substituted as text. Image parameters are sent as "
        "vision content parts.",
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
        Union[Selector(), Selector(), str, int, float, bool],
    ] = Field(
        description="Dictionary mapping parameter names to workflow data sources. "
        "Keys are referenced in prompt as {{ $parameters.key }}. "
        "Values that resolve to images (WorkflowImageData or JPEG bytes) are sent as "
        "vision content parts. Lists of images are supported. "
        "All other values are converted to strings and substituted into the prompt text.",
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
        description="Optional UQL operation chains to transform parameter values before "
        "insertion. Keys must match parameter names in prompt_parameters.",
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
    max_tokens: int = Field(
        default=500,
        description="Maximum number of tokens the model can generate.",
        gt=1,
    )
    temperature: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Sampling temperature (0.0-2.0). Higher = more random.",
        ge=0.0,
        le=2.0,
    )

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


class OpenAICompatibleBlockV1(WorkflowBlock):

    def __init__(self):
        self._client_cache: Dict[Tuple[str, str], OpenAI] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    def _get_client(self, base_url: str, api_key: str) -> OpenAI:
        cache_key = (base_url, api_key)
        client = self._client_cache.get(cache_key)
        if client is None:
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
            self._client_cache[cache_key] = client
        return client

    def run(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str],
        system_prompt: Optional[str],
        prompt: str,
        prompt_parameters: Dict[str, Any],
        prompt_parameters_operations: Dict[str, List[AllOperationsType]],
        max_tokens: int,
        temperature: Optional[float],
    ) -> BlockResult:
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
            output = _execute_request(
                client=self._get_client(base_url, api_key or "no-key"),
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {"output": output, "error_status": ""}
        except Exception as e:
            logger.warning(
                f"OpenAI-compatible request to {base_url} failed: {e}",
                exc_info=True,
            )
            return {"output": "", "error_status": str(e)}


def _resolve_parameters(
    prompt_parameters: Dict[str, Any],
    prompt_parameters_operations: Dict[str, List[AllOperationsType]],
) -> Dict[str, Any]:
    resolved = {}
    for name, value in prompt_parameters.items():
        operations = prompt_parameters_operations.get(name)
        if operations:
            chain = build_operations_chain(operations=operations)
            value = chain(value, global_parameters={})
        resolved[name] = value
    return resolved


def _is_image_value(value: Any) -> bool:
    if isinstance(value, (WorkflowImageData, bytes)):
        return True
    return False


def _is_image_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return any(_is_image_value(item) for item in value)


def _encode_single_image(value: Any) -> str:
    if isinstance(value, WorkflowImageData):
        jpeg_bytes = encode_image_to_jpeg_bytes(value.numpy_image)
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    elif isinstance(value, bytes):
        b64 = base64.b64encode(value).decode("ascii")
    else:
        raise ValueError(f"Cannot encode value of type {type(value)} as image")
    return f"data:image/jpeg;base64,{b64}"


def _collect_image_data_urls(value: Any) -> List[str]:
    if _is_image_value(value):
        return [_encode_single_image(value)]
    if _is_image_list(value):
        return [_encode_single_image(item) for item in value if _is_image_value(item)]
    return []


def _build_prompt_content(
    prompt: str,
    resolved_params: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Build text prompt and image content parts from template + params.

    Returns (text_prompt, image_parts) where image_parts is a list of
    base64 data URLs for any image parameters found.
    """
    image_parts: List[str] = []
    image_param_names: Set[str] = set()

    matching_parameters = PARAMETER_REGEX.findall(prompt)
    parameters_to_substitute = {
        p[1] for p in matching_parameters if p[1] in resolved_params
    }

    # Collect images from params referenced in the template
    for name in parameters_to_substitute:
        value = resolved_params[name]
        urls = _collect_image_data_urls(value)
        if urls:
            image_parts.extend(urls)
            image_param_names.add(name)

    # Collect images from params NOT referenced in the template
    for name, value in resolved_params.items():
        if name in image_param_names or name in parameters_to_substitute:
            continue
        urls = _collect_image_data_urls(value)
        if urls:
            image_parts.extend(urls)
            image_param_names.add(name)

    # Substitute non-image params into the prompt text
    parameter_to_placeholders: Dict[str, List[str]] = defaultdict(list)
    for placeholder, param_name in matching_parameters:
        if param_name not in parameters_to_substitute:
            continue
        parameter_to_placeholders[param_name].append(placeholder)

    for param_name, placeholders in parameter_to_placeholders.items():
        for placeholder in placeholders:
            if param_name in image_param_names:
                prompt = prompt.replace(placeholder, "")
            else:
                prompt = prompt.replace(placeholder, str(resolved_params[param_name]))

    return prompt.strip(), image_parts


def _build_messages(
    system_prompt: Optional[str],
    text_prompt: str,
    image_parts: List[str],
) -> List[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: List[dict] = []
    if text_prompt:
        content.append({"type": "text", "text": text_prompt})
    for image_url in image_parts:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )

    messages.append({"role": "user", "content": content})
    return messages


def _execute_request(
    client: OpenAI,
    model_name: str,
    messages: List[dict],
    max_tokens: int,
    temperature: Optional[float],
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    if response.choices is None or len(response.choices) == 0:
        error_detail = getattr(response, "error", {})
        if isinstance(error_detail, dict):
            error_detail = error_detail.get("message", "No response choices returned")
        raise RuntimeError(f"API returned no choices. Details: {error_detail}")
    return response.choices[0].message.content

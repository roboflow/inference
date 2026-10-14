"""Send a prompt, with optional images, to any OpenAI-compatible server.

Canonical type: ``roboflow_core/openai_compatible@v1``.
"""

from functools import partial
from typing import Annotated, Any, Collection, Dict, List, Optional
import httpx
from openai import DefaultHttpxClient, OpenAI
from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
)
from roboflow_workflows.v2.blocks.language._shared.openai import (
    Scalar,
    compatible_prompt,
    compile_parameter_operations,
)
from roboflow_workflows.v2.blocks.language.chat import (
    WORK_OPERATIONS,
    ApiKey,
    ChatKeys,
    ChatPrompt,
    ChatRequestError,
    reveal_keys,
    scrubbed_call,
    third_party_model,
)
from roboflow_workflows.v2.blocks.language.chat_completions import chat_messages
from roboflow_workflows.v2.blocks.query import Operation
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

COMPATIBLE_TIMEOUT_SECONDS = 120.0


class OpenAICompatibleBlock(Block):
    """Send a prompt, with optional images, to any OpenAI-compatible server.

    Example: ``base_url="http://localhost:8000/v1"``, ``model_name="qwen3-vl"``,
    ``prompt="Describe {{ $parameters.photo }}"`` with
    ``prompt_parameters={"photo": "$inputs.image"}`` sends the text
    "Describe" and the image as a vision part. Failures do not raise: the
    step returns ``output=""`` and the message in ``error_status``.
    """

    type = "roboflow_core/openai_compatible@v1"
    metadata = {
        "name": "OpenAI-Compatible LLM",
        "port_of": "roboflow_core/openai_compatible@v1",
        "section": "model",
    }
    outputs = {
        "output": Output(
            STRING_KIND,
            LANGUAGE_MODEL_OUTPUT_KIND,
            description="The model's answer; empty when the request failed.",
        ),
        "error_status": Output(
            STRING_KIND,
            description="Empty on success, else the scrubbed failure message.",
        ),
    }

    class Params(BlockParams):
        base_url: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "API base URL, e.g. http://localhost:8000/v1. The host's "
                "allowed_base_urls must allow it; redirects are not followed."
            ),
            examples=["http://localhost:8000/v1"],
        )
        model_name: StrictStr | Ref(STRING_KIND) = Field(
            description="Model name the server knows.",
            examples=["Qwen/Qwen3-VL-8B-Instruct"],
        )
        api_key: Optional[ApiKey] = Field(
            default=None,
            description="Server key as a secret or string; None sends 'no-key'.",
            examples=["$steps.secrets.llm_api_key"],
        )
        system_prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description="System message; None or empty sends none.",
            examples=["You are a careful inspector."],
        )
        prompt: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "Instruction text. {{ $parameters.NAME }} inserts that prompt "
                "parameter; an image parameter is removed from the text and sent "
                "as an image."
            ),
            examples=["Count the people in {{ $parameters.photo }}"],
        )
        prompt_parameters: Dict[str, Scalar | Ref(allow_none=True)] = (
            Field(  # noqa: F821
                default_factory=dict,
                description=(
                    "Values for the prompt: literals or selectors. Images, JPEG bytes "
                    "and lists of them become image parts, whether referenced or not."
                ),
                examples=[{"photo": "$inputs.image", "count": "$steps.count.output"}],
            )
        )
        prompt_parameters_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description="UQL operations per prompt parameter, applied first.",
            examples=[{"count": [{"type": "SequenceLength"}]}],
        )
        max_tokens: StrictInt = Field(
            default=500, gt=1, description="Most output tokens.", examples=[500]
        )
        temperature: Optional[Annotated[StrictFloat, Field(ge=0.0, le=2.0)]] | Ref(
            FLOAT_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Sampling temperature; None keeps the server's default.",
            examples=[0.2],
        )
        extra_body: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Extra JSON sent as the OpenAI SDK extra_body.",
            examples=[{"chat_template_kwargs": {"enable_thinking": False}}],
        )

        @field_validator("prompt_parameters_operations")
        @classmethod
        def _operations_compile(
            cls, value: Dict[str, List[Operation]], info: ValidationInfo
        ) -> Dict[str, List[Operation]]:
            if "prompt_parameters" in info.data:
                compile_parameter_operations(
                    value, keys=tuple(info.data["prompt_parameters"])
                )

            return value

    def __init__(
        self,
        *,
        allowed_base_urls: Optional[Collection[str]] = None,
        openai_compatible_http_client: Optional[httpx.Client] = None,
    ) -> None:
        if allowed_base_urls is None:
            from roboflow_workflows.environment import (
                OPENAI_COMPATIBLE_ALLOWED_BASE_URLS,
            )

            allowed_base_urls = OPENAI_COMPATIBLE_ALLOWED_BASE_URLS
        self._allowed_base_urls = frozenset(
            url.rstrip("/") for url in allowed_base_urls
        )
        self._allow_any = "*" in allowed_base_urls
        self._http_client = openai_compatible_http_client or DefaultHttpxClient(
            follow_redirects=False
        )

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the server and model the step calls."""
        resources = [
            third_party_model(
                params.model_name,
                provider=params.base_url.rstrip("/"),
                resource="openai_compatible_http_client",
            )
        ]

        return resources

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[Any]:
        """Report remote inference and image encoding."""
        operations = list(WORK_OPERATIONS)

        return operations

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> List[Any]:
        """Declare that the step has no runtime restrictions."""
        return []

    def run(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: Any,
        system_prompt: Optional[str],
        prompt: str,
        prompt_parameters: Dict[str, Any],
        prompt_parameters_operations: Dict[str, List[Any]],
        max_tokens: int,
        temperature: Optional[float],
        extra_body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Render the prompt and send one chat-completions request.

        Args:
            base_url: Server API base URL.
            model_name: Model name.
            api_key: Server key, secret or ``None``.
            system_prompt: System message or ``None``.
            prompt: Instruction template.
            prompt_parameters: Placeholder values and images.
            prompt_parameters_operations: Operations per parameter.
            max_tokens: Output token limit.
            temperature: Sampling temperature or ``None``.
            extra_body: Extra request JSON or ``None``.

        Returns:
            ``{"output": answer, "error_status": ""}``, or ``output=""`` and
            the scrubbed failure message.

        Raises:
            QueryRuntimeError: When an operation cannot process a value.
            SecretUnavailableError: When the key was read back from a recording.
        """
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2.")

        chains = compile_parameter_operations(
            prompt_parameters_operations, keys=tuple(prompt_parameters)
        )
        values = {
            name: (
                value
                if name not in chains
                else chains[name].apply(value, parameters={})
            )
            for name, value in prompt_parameters.items()
        }
        text, image_urls = compatible_prompt(prompt, values)
        messages = chat_messages(ChatPrompt(system_prompt, text), image_urls)
        keys = reveal_keys(api_key, roboflow_api_key=None)
        request = {"model": model_name, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            request["temperature"] = temperature
        if extra_body is not None:
            request["extra_body"] = extra_body

        complete = partial(self._complete, base_url, keys=keys, request=request)
        try:
            output = scrubbed_call(complete, keys=keys, provider="OpenAI-compatible")
        except ChatRequestError as error:
            result = {"output": "", "error_status": str(error)}
            return result

        result = {"output": output, "error_status": ""}

        return result

    def _complete(
        self, base_url: str, *, keys: ChatKeys, request: Dict[str, Any]
    ) -> str:
        url = self._approved_url(base_url)
        client = OpenAI(
            base_url=url,
            api_key=keys.provider_key or "no-key",
            timeout=COMPATIBLE_TIMEOUT_SECONDS,
            http_client=self._http_client,
        )
        response = client.chat.completions.create(**request)
        if not response.choices:
            details = getattr(response, "error", None) or {}
            if isinstance(details, dict):
                details = details.get("message", "No response choices returned")
            raise ChatRequestError(f"API returned no choices. Details: {details}")

        content = response.choices[0].message.content
        if content is None:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            raise ChatRequestError(
                f"API returned empty message content (finish_reason={finish_reason})."
            )

        return content

    def _approved_url(self, base_url: str) -> str:
        url = base_url.rstrip("/")
        if not self._allow_any and url not in self._allowed_base_urls:
            raise ChatRequestError(
                "OpenAI-compatible endpoint is not approved by the server. Configure "
                "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS to allow this base URL."
            )

        parsed = httpx.URL(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or parsed.userinfo
            or parsed.query
            or parsed.fragment
        ):
            raise ChatRequestError(
                "OpenAI-compatible base URL must be an HTTP(S) URL without "
                "credentials, query strings, or fragments."
            )

        return url

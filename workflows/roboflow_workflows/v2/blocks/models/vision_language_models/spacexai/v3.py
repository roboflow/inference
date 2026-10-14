"""Ask a SpaceXAI Grok model about each image through xAI's Responses API.

Canonical type: ``roboflow_core/spacexai@v3``.
"""

from functools import partial
from typing import Any, Dict, List, Literal, Optional
import uuid
import httpx
from openai import DefaultHttpxClient
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.models.vision_language_models._shared import (
    compatible_vendors,
)
from roboflow_workflows.v2.blocks.models.vision_language_models._shared.compatible_vendors import (
    XAI_DETECTION_BOX_FORMAT,
    logger,
    read_xai_answer,
    rejects_reasoning,
    xai_task_request,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.chat import (
    DEFAULT_MANAGED_KEY,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    ChatAnswer,
    ChatKeys,
    check_reasoning_effort,
    check_task_fields,
    reveal_keys,
    run_concurrently,
    scrubbed_call,
    third_party_model,
    token_outputs,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.decoding import (
    decode_vlm_output,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.openrouter.v3 import (
    DETECTION_TASK,
    OPENROUTER_TASKS,
    OpenRouterTask,
)
from roboflow_workflows.v2.blocks.platform import (
    OFFLINE_PLATFORM_CLIENT,
    PlatformClient,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

XAI_PROVIDER = "xAI"

XAI_API_URL = "https://api.x.ai/v1"

XAI_PROXY_ENDPOINT = "apiproxy/xai"

GROK_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")

GROK_MODELS = {
    "grok-4.7": ("Grok 4.7", ("low", "medium", "high", "xhigh")),
    "grok-4.6": ("Grok 4.6", ("low", "medium", "high", "xhigh")),
    "grok-4.5": ("Grok 4.5", ("low", "medium", "high")),
}

GROK_REASONING_LEVELS = {model: levels for model, (_, levels) in GROK_MODELS.items()}


def xai_options(
    model: str,
    *,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> Dict[str, Any]:
    """Return ``model``, ``store`` and the set generation options of a request.

    Args:
        model: Grok model id.
        reasoning_effort: Reasoning level or ``None``.
        max_tokens: Output token limit or ``None``.
        temperature: Sampling temperature or ``None``.

    Returns:
        The request options; ``store`` is always ``False``, as in V1.

    Raises:
        ValueError: When a known model does not support ``reasoning_effort``.
    """
    check_reasoning_effort(
        model, reasoning_effort, levels_by_model=GROK_REASONING_LEVELS
    )
    options: Dict[str, Any] = {"model": model, "store": False}
    if max_tokens is not None:
        options["max_output_tokens"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if reasoning_effort is not None:
        options["reasoning"] = {"effort": reasoning_effort}

    return options


class SpaceXAIBlock(Block):
    """Ask a SpaceXAI Grok model about each image through xAI's Responses API.

    Example: ``task_type="object-detection"`` with ``classes=["car"]`` sends
    the full-size image as lossless PNG and returns ``predictions`` as
    detections in image pixels. ``api_key="rf_key:account"`` (the default)
    bills the host's Roboflow account through its platform client; an xAI key
    or a secret calls xAI directly.
    """

    type = "roboflow_core/spacexai@v3"
    metadata = {
        "name": "SpaceXAI",
        "port_of": "roboflow_core/spacexai@v3",
        "section": "model",
        "license": "Apache-2.0",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "model_metadata": {
            model: {"name": name, "reasoning_levels": list(levels)}
            for model, (name, levels) in GROK_MODELS.items()
        },
        "recommended_parsers": {"structured-answering": "roboflow_core/json_parser@v1"},
    }
    outputs = {
        "output": Output(
            STRING_KIND,
            LANGUAGE_MODEL_OUTPUT_KIND,
            source="images",
            description="The model's raw answer to this image.",
        ),
        "classes": Output(
            LIST_OF_VALUES_KIND, source="images", description="The step's classes."
        ),
        **token_outputs(source="images"),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to ask about; one request per image."
        )
        task_type: OpenRouterTask = Field(
            default="unconstrained",
            description="What to ask: a free prompt, OCR, captions, classification, structured answers or detection.",
            examples=["object-detection"],
        )
        prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description="Prompt for 'unconstrained' and question for 'visual-question-answering'.",
            examples=["my prompt", "$inputs.prompt"],
        )
        output_structure: Optional[Dict[str, StrictStr]] = Field(
            default=None,
            description="For 'structured-answering': answer field names and their descriptions.",
            examples=[{"my_key": "description"}],
        )
        classes: Optional[List[StrictStr]] | Ref(
            LIST_OF_VALUES_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Class names for classification and detection.",
            examples=[["class-a", "class-b"], "$inputs.classes"],
        )
        api_key: ApiKey = Field(
            default=DEFAULT_MANAGED_KEY,
            description=(
                "xAI key as a secret or string, or 'rf_key:account' / "
                "'rf_key:user:ID' to use a key stored with Roboflow."
            ),
            examples=["rf_key:account", "$inputs.xai_api_key"],
        )
        model_version: Literal[tuple(GROK_MODELS)] | Ref(STRING_KIND) = Field(
            default="grok-4.6",
            description="Grok model.",
            examples=["grok-4.6", "$inputs.grok_model"],
        )
        reasoning_effort: Optional[Literal[GROK_REASONING_EFFORTS]] | Ref(
            STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Sent as xAI `reasoning: {effort}`. Grok defaults to 'high' and "
                "cannot disable reasoning; grok-4.5 has no 'xhigh'. With a direct "
                "key, a model rejecting it is retried without it."
            ),
            examples=["low"],
        )
        max_tokens: Optional[StrictInt] = Field(
            default=None,
            ge=16,
            description="Most output tokens; None keeps xAI's limit.",
            examples=[2048],
        )
        temperature: Optional[StrictFloat] | Ref(FLOAT_KIND, allow_none=True) = Field(
            default=None,
            ge=0.0,
            le=2.0,
            description="Sampling temperature; None keeps xAI's default.",
            examples=[0.2],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most requests in flight for one batch; None means 8.",
            examples=[4],
        )

        @model_validator(mode="after")
        def _task_fields_present(self) -> "SpaceXAIBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )
            check_reasoning_effort(
                self.model_version,
                self.reasoning_effort,
                levels_by_model=GROK_REASONING_LEVELS,
            )

            return self

    def __init__(
        self,
        *,
        platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT,
        api_key: Any = None,
        xai_base_url: str = XAI_API_URL,
        xai_http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._platform_client = platform_client
        self._roboflow_api_key = api_key
        self._base_url = xai_base_url
        self._http_client = xai_http_client or DefaultHttpxClient()

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Dict[str, Output]:
        """Narrow ``predictions`` to the kind the configured task decodes."""
        outputs = {
            **cls.outputs,
            **vlm_prediction_outputs(params.task_type, source="images"),
        }

        return outputs

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the Grok model the step calls."""
        resources = [
            third_party_model(
                params.model_version, provider="xai", resource="xai_http_client"
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
        images: Batch,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Any,
        model_version: str,
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Send one request per image and decode each answer.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: xAI key, secret or managed ``rf_key:`` reference.
            model_version: Grok model id.
            reasoning_effort: Reasoning level or ``None``.
            max_tokens: Output token limit or ``None``.
            temperature: Sampling temperature or ``None``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes", "input_tokens", "output_tokens",
            "predictions", "error_status", "inference_id"}`` per image.

        Raises:
            ValueError: On a missing provider key, a managed key without the
                host's Roboflow ``api_key`` resource, or a reasoning level the
                model does not support, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        options = xai_options(
            model_version,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        keys = reveal_keys(api_key, roboflow_api_key=self._roboflow_api_key)
        if keys.provider_key is None:
            raise ValueError(f"{self.type} requires an xAI API key.")

        if keys.managed and not keys.roboflow_key:
            raise ValueError(
                "Roboflow API key is required when using a Roboflow-managed xAI "
                "API key; the host must provide the 'api_key' resource."
            )

        requests = [
            xai_task_request(
                task_type,
                image=image,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
            for image in images
        ]
        calls = [
            partial(self._ask, keys, {**options, **request}) for request in requests
        ]
        answers = run_concurrently(
            calls, max_concurrent_requests=max_concurrent_requests
        )

        results = []
        for image, answer in zip(images, answers):
            inference_id = uuid.uuid4().hex
            error_status, predictions = decode_vlm_output(
                task_type,
                answer.text,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=(
                    XAI_DETECTION_BOX_FORMAT if task_type == DETECTION_TASK else None
                ),
            )
            results.append(
                {
                    "output": answer.text,
                    "classes": classes,
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    "predictions": predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )

        return results

    def _ask(self, keys: ChatKeys, payload: Dict[str, Any]) -> ChatAnswer:
        answer = scrubbed_call(
            partial(self._request, keys, payload), keys=keys, provider=XAI_PROVIDER
        )

        return answer

    def _request(self, keys: ChatKeys, payload: Dict[str, Any]) -> ChatAnswer:
        if keys.managed:
            data = self._platform_client.post(
                endpoint=XAI_PROXY_ENDPOINT,
                api_key=keys.roboflow_key,
                payload={**payload, "xai_api_key": keys.provider_key},
            )
        else:
            data = self._send_direct(keys, payload)
        answer = read_xai_answer(data)

        return answer

    def _send_direct(self, keys: ChatKeys, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Module attribute read at call time: tests patch compatible_vendors.OpenAI.
        client = compatible_vendors.OpenAI(
            base_url=self._base_url,
            api_key=keys.provider_key,
            http_client=self._http_client,
        )
        try:
            response = client.responses.create(**payload)
        except Exception as error:
            if "reasoning" not in payload or not rejects_reasoning(error):
                raise
            logger.warning(
                "xAI rejected the reasoning config %s for model %s (details: %s). "
                "Retrying without it - the model will use its default effort.",
                payload["reasoning"],
                payload["model"],
                keys.scrub(str(error)),
            )
            without_reasoning = {
                name: value for name, value in payload.items() if name != "reasoning"
            }
            response = client.responses.create(**without_reasoning)
        data = response.to_dict()

        return data

"""Ask a Qwen 3.5 vision model on OpenRouter about each image.

Canonical type: ``roboflow_core/qwen3_5_openrouter@v1``.
"""

from functools import partial
from typing import Any, Dict, List, Literal, Optional
import httpx
from openai import DefaultHttpxClient, OpenAI
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.models.vision_language_models._shared.compatible_vendors import (
    model_slug,
    qwen_task_messages,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.chat import (
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    ChatAnswer,
    ChatKeys,
    check_task_fields,
    reveal_keys,
    run_concurrently,
    scrubbed_call,
    third_party_model,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.openrouter.v3 import (
    OPENROUTER_API_URL,
    OPENROUTER_TASKS,
    OpenRouterTask,
    read_chat_completion,
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

QWEN3_5_MODELS = {
    "Qwen 3.5 9B - OpenRouter": "qwen/qwen3.5-9b",
    "Qwen 3.5 27B - OpenRouter": "qwen/qwen3.5-27b",
    "Qwen 3.5 122B A10B - OpenRouter": "qwen/qwen3.5-122b-a10b",
    "Qwen 3.5 397B A17B - OpenRouter": "qwen/qwen3.5-397b-a17b",
    "Qwen 3.5 Flash 02-23 - OpenRouter": "qwen/qwen3.5-flash-02-23",
    "Qwen 3.5 Plus 20260420 - OpenRouter": "qwen/qwen3.5-plus-20260420",
}


class Qwen35OpenRouterBlock(Block):
    """Ask a Qwen 3.5 vision model on OpenRouter about each image.

    Object detection returns raw normalized 0-1 ``x_min``/``y_min``/``x_max``/
    ``y_max`` fields in ``output``. There is no matching supported Qwen
    ``model_type`` in ``roboflow_core/vlm_as_detector@v2``; its ``qwen`` format expects
    0-1000 ``box_2d`` fields. Use ``roboflow_core/openrouter@v3`` with
    ``model_id="qwen/qwen3.5-27b"`` for native decoded ``predictions``.

    Example: ``task_type="structured-answering"`` with
    ``output_structure={"count": "number of cars"}`` and
    ``api_key="$inputs.openrouter_key"`` returns the raw JSON answer, which
    ``roboflow_core/json_parser@v1`` turns into fields.
    """

    type = "roboflow_core/qwen3_5_openrouter@v1"
    metadata = {
        "name": "Qwen 3.5 API",
        "port_of": "roboflow_core/qwen3_5_openrouter@v1",
        "section": "model",
        "license": "Qwen License",
        "deprecated": True,
        "deprecation_message": (
            "Use roboflow_core/openrouter@v3 with a Qwen model_id for V2 workflows, "
            "including decoded detections, managed keys, privacy routing and "
            "reasoning. V1-only replacement: roboflow_core/qwen_vlm@v1."
        ),
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "recommended_parsers": {
            "structured-answering": "roboflow_core/json_parser@v1",
            "classification": "roboflow_core/vlm_as_classifier@v2",
            "multi-label-classification": "roboflow_core/vlm_as_classifier@v2",
        },
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
    }

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to ask about; one request per image."
        )
        task_type: OpenRouterTask = Field(
            default="unconstrained",
            description="What to ask: a free prompt, OCR, captions, classification, structured answers or detection.",
            examples=["structured-answering"],
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
            description="Your OpenRouter API key, as a secret or string.",
            examples=["$inputs.open_router_api_key"],
        )
        max_tokens: StrictInt = Field(
            default=500,
            gt=1,
            description="Most output tokens.",
            examples=[500],
        )
        temperature: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=0.1,
            ge=0.0,
            le=2.0,
            description="Sampling temperature.",
            examples=[0.1],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most requests in flight for one batch; None means 8.",
            examples=[4],
        )
        model_version: Literal[tuple(QWEN3_5_MODELS)] | Ref(STRING_KIND) = Field(
            default="Qwen 3.5 27B - OpenRouter",
            description="Qwen 3.5 model served by OpenRouter.",
            examples=["Qwen 3.5 27B - OpenRouter", "$inputs.qwen_model"],
        )

        @model_validator(mode="after")
        def _task_fields_present(self) -> "Qwen35OpenRouterBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )

            return self

    def __init__(
        self,
        *,
        openrouter_base_url: str = OPENROUTER_API_URL,
        openrouter_http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = openrouter_base_url
        self._http_client = openrouter_http_client or DefaultHttpxClient()

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the OpenRouter model the step calls."""
        model_id = QWEN3_5_MODELS.get(params.model_version, params.model_version)
        resources = [
            third_party_model(
                model_id, provider="openrouter", resource="openrouter_http_client"
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
        max_tokens: int,
        temperature: float,
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Send one request per image; return the raw answers.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key as a string or secret.
            model_version: A label of ``QWEN3_5_MODELS``.
            max_tokens: Output token limit.
            temperature: Sampling temperature.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes"}`` per image.

        Raises:
            ValueError: On an unknown model label, a missing provider key or
                a Roboflow-managed key, before any request.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails.
        """
        model = model_slug(model_version, models=QWEN3_5_MODELS)
        keys = reveal_keys(api_key, roboflow_api_key=None)
        if keys.provider_key is None:
            raise ValueError(f"{self.type} requires an OpenRouter API key.")

        if keys.managed:
            raise ValueError(
                f"{self.type} calls OpenRouter with your own key; it cannot use a "
                "Roboflow-managed 'rf_key:' key. Use roboflow_core/openrouter@v3 for that."
            )

        prompts = [
            qwen_task_messages(
                task_type,
                image=image,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
            for image in images
        ]
        request = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        calls = [partial(self._ask, keys, messages, request) for messages in prompts]
        answers = run_concurrently(
            calls, max_concurrent_requests=max_concurrent_requests
        )

        results = [{"output": answer.text, "classes": classes} for answer in answers]

        return results

    def _ask(
        self, keys: ChatKeys, messages: List[Dict[str, Any]], request: Dict[str, Any]
    ) -> ChatAnswer:
        answer = scrubbed_call(
            partial(self._request, keys, messages, request),
            keys=keys,
            provider="OpenRouter",
        )

        return answer

    def _request(
        self, keys: ChatKeys, messages: List[Dict[str, Any]], request: Dict[str, Any]
    ) -> ChatAnswer:
        client = OpenAI(
            base_url=self._base_url,
            api_key=keys.provider_key,
            http_client=self._http_client,
        )
        data = client.chat.completions.create(messages=messages, **request).to_dict()
        answer = read_chat_completion(data, via_proxy=False)

        return answer

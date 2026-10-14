"""Ask an OpenAI GPT model about each image, optionally decoding the answer.

Canonical type: ``roboflow_core/open_ai@v7``.
"""

from functools import partial
from typing import Annotated, Any, Dict, List, Literal, Optional
import uuid
import httpx
from openai import DefaultHttpxClient, OpenAI
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.models.vision_language_models._shared.openai import (
    DEFAULT_MODEL,
    DETECTION_TASK,
    OPENAI_MODELS,
    REASONING_LEVELS,
    SEGMENTATION_DEFAULT_MODEL,
    SEGMENTATION_TASK,
    box_format_for_model,
    default_model,
    openai_task_request,
    read_responses_answer,
    resolved_labels,
    responses_options,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.chat import (
    DEFAULT_MANAGED_KEY,
    REASONING_EFFORT_OPTIONS,
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
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

PROVIDER = "OpenAI"

OPENAI_API_URL = "https://api.openai.com/v1"

OPENAI_PROXY_ENDPOINT = "apiproxy/openai/v2"

OPENAI_TASKS = (
    "unconstrained",
    "ocr",
    "structured-answering",
    "classification",
    "multi-label-classification",
    "visual-question-answering",
    "caption",
    "detailed-caption",
    DETECTION_TASK,
    SEGMENTATION_TASK,
)

OpenAITask = Literal[OPENAI_TASKS]

TASKS_SUPPORTING_OUTPUT_CLASSES = (DETECTION_TASK, SEGMENTATION_TASK)

ANSWER_OUTPUT = Output(
    STRING_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    source="images",
    description="The model's raw answer to this image.",
)

CLASSES_OUTPUT = Output(
    LIST_OF_VALUES_KIND,
    source="images",
    description="Labels the answer was decoded with: output_classes, else classes.",
)


def _all_prediction_outputs() -> Dict[str, Output]:
    outputs = dict(vlm_prediction_outputs(SEGMENTATION_TASK, source="images"))
    outputs["predictions"] = Output(
        OBJECT_DETECTION_PREDICTION_KIND,
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
        source="images",
        description="Decoded prediction of the task; None when unreadable or not decoded.",
    )

    return outputs


class OpenAIBlock(Block):
    """Ask an OpenAI GPT model about each image, optionally decoding the answer.

    Example: ``task_type="object-detection"`` with ``classes=["car", "person"]``
    returns the raw ``output`` and ``predictions`` as native detections in the
    image's pixels. ``api_key="rf_key:account"`` (the default) bills the host's
    Roboflow account through its platform client; an ``sk-...`` key or a
    secret calls OpenAI directly.
    """

    type = "roboflow_core/open_ai@v7"
    metadata = {
        "name": "OpenAI",
        "port_of": "roboflow_core/open_ai@v7",
        "section": "model",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENAI_TASKS},
        "model_metadata": {
            model_id: {
                "name": model.name,
                "reasoning_levels": list(model.reasoning_levels),
            }
            for model_id, model in OPENAI_MODELS.items()
        },
    }
    outputs = {
        "output": ANSWER_OUTPUT,
        "classes": CLASSES_OUTPUT,
        **token_outputs(source="images"),
        **_all_prediction_outputs(),
    }
    output_fields = ("task_type",)

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to ask about; one request per image."
        )
        task_type: OpenAITask = Field(
            default="unconstrained",
            description=(
                "What to ask: a free prompt, OCR, captions, classification, "
                "structured answers, object detection or instance segmentation."
            ),
            examples=["object-detection"],
        )
        prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description="Prompt for 'unconstrained' and question for 'visual-question-answering'.",
            examples=["How many cars are there?", "$inputs.prompt"],
        )
        output_structure: Optional[Dict[str, StrictStr]] = Field(
            default=None,
            description="For 'structured-answering': answer field names and their descriptions.",
            examples=[{"count": "number of cars"}],
        )
        classes: Optional[List[StrictStr]] | Ref(
            LIST_OF_VALUES_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Class names (or visual descriptions, with output_classes) for "
                "classification, detection and segmentation."
            ),
            examples=[["car", "person"], "$inputs.classes"],
        )
        output_classes: Optional[List[StrictStr]] | Ref(
            LIST_OF_VALUES_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Detection/segmentation labels, one per entry of classes; the "
                "model sees classes as descriptions and answers with these labels."
            ),
            examples=[["car", "person"]],
        )
        api_key: ApiKey = Field(
            default=DEFAULT_MANAGED_KEY,
            description=(
                "OpenAI key as a secret or string, or 'rf_key:account' / "
                "'rf_key:user:ID' to use a key stored with Roboflow."
            ),
            examples=["rf_key:account", "$steps.secrets.openai_api_key"],
        )
        model_version: Optional[Literal[tuple(OPENAI_MODELS)]] | Ref(
            STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                f"OpenAI model. None: {DEFAULT_MODEL!r}, or "
                f"{SEGMENTATION_DEFAULT_MODEL!r} for instance-segmentation."
            ),
            examples=["gpt-5.1", "$inputs.model_version"],
        )
        reasoning_effort: Optional[Literal[REASONING_EFFORT_OPTIONS]] | Ref(
            STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Reasoning level for models that support it; each model accepts "
                "the levels in its metadata. None keeps OpenAI's default."
            ),
            examples=["low"],
        )
        image_detail: Literal["auto", "high", "low"] | Ref(STRING_KIND) = Field(
            default="auto",
            description="Image detail OpenAI processes; detection uploads ignore it, as in V1.",
            examples=["auto"],
        )
        max_tokens: Optional[StrictInt] = Field(
            default=None,
            ge=16,
            description="Most output tokens; None keeps OpenAI's limit.",
            examples=[2048],
        )
        temperature: Optional[Annotated[StrictFloat, Field(ge=0.0, le=2.0)]] | Ref(
            FLOAT_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Sampling temperature; None keeps OpenAI's default.",
            examples=[0.2],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most requests in flight for one batch; None means 8.",
            examples=[4],
        )

        @model_validator(mode="after")
        def _task_fields_present(self) -> "OpenAIBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )
            if self.output_classes is not None:
                if self.task_type not in TASKS_SUPPORTING_OUTPUT_CLASSES:
                    raise ValueError(
                        "`output_classes` is only supported for object-detection and "
                        "instance-segmentation"
                    )
                if isinstance(self.classes, list) and isinstance(
                    self.output_classes, list
                ):
                    resolved_labels(self.classes, self.output_classes)
            check_reasoning_effort(
                default_model(self.model_version, task_type=self.task_type),
                self.reasoning_effort,
                levels_by_model=REASONING_LEVELS,
            )

            return self

    def __init__(
        self,
        *,
        platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT,
        api_key: Any = None,
        openai_base_url: str = OPENAI_API_URL,
        openai_http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._platform_client = platform_client
        self._roboflow_api_key = api_key
        self._base_url = openai_base_url
        self._http_client = openai_http_client or DefaultHttpxClient()

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Dict[str, Output]:
        """Narrow ``predictions`` to the kind the configured task decodes."""
        outputs = {
            "output": ANSWER_OUTPUT,
            "classes": CLASSES_OUTPUT,
            **token_outputs(source="images"),
            **vlm_prediction_outputs(params.task_type, source="images"),
        }

        return outputs

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the OpenAI model the step calls."""
        model = default_model(params.model_version, task_type=params.task_type)
        resources = [
            third_party_model(model, provider="openai", resource="openai_http_client")
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
        output_classes: Optional[List[str]],
        api_key: Any,
        model_version: Optional[str],
        reasoning_effort: Optional[str],
        image_detail: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Send one request per image and decode each answer.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question, for the tasks that take one.
            output_structure: Fields of a structured answer.
            classes: Class names or descriptions.
            output_classes: Labels paired with ``classes``.
            api_key: OpenAI key, secret or managed ``rf_key:`` reference.
            model_version: Model, ``None`` for the task's default.
            reasoning_effort: Reasoning level or ``None``.
            image_detail: ``auto``, ``high`` or ``low``.
            max_tokens: Output token limit or ``None``.
            temperature: Sampling temperature or ``None``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.

        Returns:
            One ``{"output", "classes", "input_tokens", "output_tokens",
            "predictions", "error_status", "inference_id"}`` per image.

        Raises:
            ValueError: On a managed key without the host's Roboflow
                ``api_key`` resource, or invalid classes.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2.")

        model = default_model(model_version, task_type=task_type)
        labels = resolved_labels(classes, output_classes)
        keys = reveal_keys(api_key, roboflow_api_key=self._roboflow_api_key)
        if keys.provider_key is None:
            raise ValueError("OpenAI API key is required.")

        if keys.managed and not keys.roboflow_key:
            raise ValueError(
                "Roboflow API key is required when using a Roboflow-managed OpenAI "
                "API key; the host must provide the 'api_key' resource."
            )

        requests = [
            openai_task_request(
                task_type,
                image=image,
                model=model,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
                output_classes=output_classes,
                image_detail=image_detail,
            )
            for image in images
        ]
        options = responses_options(
            model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        calls = [
            partial(self._ask, keys, {**request.payload, **options})
            for request in requests
        ]
        answers = run_concurrently(
            calls, max_concurrent_requests=max_concurrent_requests
        )

        box_format = (
            box_format_for_model(model) if task_type == DETECTION_TASK else None
        )
        results = []
        for image, request, answer in zip(images, requests, answers):
            inference_id = uuid.uuid4().hex
            error_status, predictions = decode_vlm_output(
                task_type,
                answer.text,
                image=image,
                classes=labels,
                inference_id=inference_id,
                box_format=box_format,
                upload_size_wh=request.decoding_size_wh,
            )
            results.append(
                {
                    "output": answer.text,
                    "classes": labels,
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
            partial(self._request, keys, payload), keys=keys, provider=PROVIDER
        )

        return answer

    def _request(self, keys: ChatKeys, payload: Dict[str, Any]) -> ChatAnswer:
        if keys.managed:
            data = self._platform_client.post(
                endpoint=OPENAI_PROXY_ENDPOINT,
                api_key=keys.roboflow_key,
                payload={**payload, "openai_api_key": keys.provider_key},
            )
        else:
            client = OpenAI(
                api_key=keys.provider_key,
                base_url=self._base_url,
                http_client=self._http_client,
            )
            data = client.responses.create(**payload).to_dict()
        answer = read_responses_answer(data)

        return answer

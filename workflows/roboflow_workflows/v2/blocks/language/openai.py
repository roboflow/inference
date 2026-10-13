"""OpenAI GPT vision models and OpenAI-compatible chat servers.

Ports of V1 ``roboflow_core/open_ai@v7`` and ``roboflow_core/openai_compatible@v1``.
They speak different OpenAI APIs::

    open_ai@v7 (one request per image, run_concurrently)
      task ──▶ ChatPrompt + upload (JPEG, or PNG ≤2048 px for detection)
           ──▶ Responses API payload {model, input, instructions?, text?, ...}
      api_key rf_key:* ─▶ platform_client.post("apiproxy/openai/v2", payload)
      api_key sk-...   ─▶ OpenAI(base_url=openai_base_url,
                                 http_client=openai_http_client).responses.create
      answer text ──▶ decode_vlm_output ──▶ predictions, error_status

    openai_compatible@v1 (one request per call)
      prompt + {{ $parameters.NAME }} ──▶ text; image parameters ──▶ image parts
      ──▶ chat completions at the step's base_url (allowlist checked first)
      failure ──▶ output "", error_status = scrubbed message

Both send through an ``httpx.Client`` resource; a test injects one with an
``httpx.MockTransport`` and nothing leaves the process. Images are encoded on
the host, once per request (``chat.encode_image``).

``model_version=None`` (the default) means V1's default for the task:
``gpt-5.1``, or ``gpt-6-astra`` for instance segmentation. V1 replaced an
omitted ``model_version`` the same way.

Differences from V1:

* A failed open_ai request raises ``ChatRequestError`` with the keys
  scrubbed; openai_compatible reports the scrubbed message in
  ``error_status``. V1 texts could contain the key.
* The OpenAI-compatible allowlist is the ``allowed_base_urls`` resource. Its
  default reads V1's ``OPENAI_COMPATIBLE_ALLOWED_BASE_URLS`` setting.
* ``max_concurrent_requests`` must be at least 1; unset means 8 (V1's
  default) instead of an environment variable.
* Predictions are native V2 carriers; see ``decoding``.
"""

import base64
import json
import re
import uuid
from copy import deepcopy
from functools import partial
from typing import (
    Annotated,
    Any,
    Collection,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import httpx
from openai import DefaultHttpxClient, OpenAI
from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from roboflow_workflows.v2.blocks.language.chat import (
    DEFAULT_MANAGED_KEY,
    REASONING_EFFORT_OPTIONS,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    ChatAnswer,
    ChatKeys,
    ChatPrompt,
    ChatRequestError,
    EncodedImage,
    SizeWH,
    check_reasoning_effort,
    check_task_fields,
    encode_image,
    reveal_keys,
    run_concurrently,
    scrubbed_call,
    third_party_model,
    token_count,
    token_outputs,
)
from roboflow_workflows.v2.blocks.language.chat_completions import (  # noqa: F401
    chat_completion_usage,
    chat_messages,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    DETECTION_MAX_EDGE_PIXELS,
    build_instance_segmentation_prompt,
    build_object_detection_prompt,
    decode_vlm_output,
    max_edge_upload_size,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.blocks.platform import (
    OFFLINE_PLATFORM_CLIENT,
    PlatformClient,
)
from roboflow_workflows.v2.blocks.query import (
    CompiledOperations,
    Operation,
    compile_operations,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
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
COMPATIBLE_TIMEOUT_SECONDS = 120.0
PARAMETER_PATTERN = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")
"""V1's ``{{ $parameters.NAME }}`` placeholder."""

Scalar = Union[StrictStr, StrictInt, StrictFloat, StrictBool]

STRUCTURED_ABSOLUTE_STYLE = "structured-absolute"
NORMALIZED_LEGACY_STYLE = "normalized-legacy"
PLAIN_ABSOLUTE_STYLE = "plain-absolute"
ABSOLUTE_BOX_FORMAT = "xyxy_absolute"
NORMALIZED_BOX_FORMAT = "named_normalized"
BOX_FORMAT_BY_STYLE = {
    STRUCTURED_ABSOLUTE_STYLE: ABSOLUTE_BOX_FORMAT,
    PLAIN_ABSOLUTE_STYLE: ABSOLUTE_BOX_FORMAT,
    NORMALIZED_LEGACY_STYLE: NORMALIZED_BOX_FORMAT,
}

DETECTION_TASK = "object-detection"
SEGMENTATION_TASK = "instance-segmentation"
DEFAULT_MODEL = "gpt-5.1"
SEGMENTATION_DEFAULT_MODEL = "gpt-6-astra"


class OpenAIModel(NamedTuple):
    """A known OpenAI model: display name, reasoning levels, detection prompt."""

    name: str
    reasoning_levels: Tuple[str, ...]
    detection_style: str


_FRONTIER_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")
_GPT5_2_LEVELS = ("none", "low", "medium", "high", "xhigh")
_GPT5_LEVELS = ("minimal", "low", "medium", "high")

OPENAI_MODELS: Dict[str, OpenAIModel] = {
    "gpt-6-astra": OpenAIModel(
        "GPT-6 Astra",
        ("low", "medium", "high", "xhigh", "max"),
        STRUCTURED_ABSOLUTE_STYLE,
    ),
    "gpt-6-sol": OpenAIModel("GPT-6 Sol", _FRONTIER_LEVELS, STRUCTURED_ABSOLUTE_STYLE),
    "gpt-6-luna": OpenAIModel(
        "GPT-6 Luna", _FRONTIER_LEVELS, STRUCTURED_ABSOLUTE_STYLE
    ),
    "gpt-5.6-sol": OpenAIModel(
        "GPT-5.6 Sol", _FRONTIER_LEVELS, STRUCTURED_ABSOLUTE_STYLE
    ),
    "gpt-5.6-terra": OpenAIModel(
        "GPT-5.6 Terra", _FRONTIER_LEVELS, STRUCTURED_ABSOLUTE_STYLE
    ),
    "gpt-5.6-luna": OpenAIModel(
        "GPT-5.6 Luna", _FRONTIER_LEVELS, STRUCTURED_ABSOLUTE_STYLE
    ),
    "gpt-5.5": OpenAIModel("GPT-5.5", _GPT5_2_LEVELS, STRUCTURED_ABSOLUTE_STYLE),
    "gpt-5.4": OpenAIModel("GPT-5.4", _GPT5_2_LEVELS, STRUCTURED_ABSOLUTE_STYLE),
    "gpt-5.4-mini": OpenAIModel(
        "GPT-5.4 mini", _GPT5_2_LEVELS, STRUCTURED_ABSOLUTE_STYLE
    ),
    "gpt-5.4-nano": OpenAIModel("GPT-5.4 nano", _GPT5_2_LEVELS, PLAIN_ABSOLUTE_STYLE),
    "gpt-5.2": OpenAIModel("GPT-5.2", _GPT5_2_LEVELS, STRUCTURED_ABSOLUTE_STYLE),
    "gpt-5.1": OpenAIModel(
        "GPT-5.1", ("none", "low", "medium", "high"), NORMALIZED_LEGACY_STYLE
    ),
    "gpt-5": OpenAIModel("GPT-5", _GPT5_LEVELS, NORMALIZED_LEGACY_STYLE),
    "gpt-5-mini": OpenAIModel("GPT-5 mini", _GPT5_LEVELS, NORMALIZED_LEGACY_STYLE),
    "gpt-5-nano": OpenAIModel("GPT-5 nano", _GPT5_LEVELS, NORMALIZED_LEGACY_STYLE),
    "gpt-4.1": OpenAIModel("GPT-4.1", (), PLAIN_ABSOLUTE_STYLE),
    "gpt-4.1-mini": OpenAIModel("GPT-4.1 mini", (), PLAIN_ABSOLUTE_STYLE),
    "gpt-4.1-nano": OpenAIModel("GPT-4.1 nano", (), PLAIN_ABSOLUTE_STYLE),
    "gpt-4o": OpenAIModel("GPT-4o", (), PLAIN_ABSOLUTE_STYLE),
    "gpt-4o-mini": OpenAIModel("GPT-4o mini", (), PLAIN_ABSOLUTE_STYLE),
}
"""V1 v7 model table, newest first. Unknown models use the structured prompt."""

REASONING_LEVELS = {
    model_id: model.reasoning_levels for model_id, model in OPENAI_MODELS.items()
}

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

OCR_INSTRUCTIONS = (
    "You act as OCR model. Your task is to read text from the image and return it in "
    "paragraphs representing the structure of texts in the image. You should only "
    "return recognised text, nothing else."
)
VQA_INSTRUCTIONS = (
    "You act as Visual Question Answering model. Your task is to provide answer to "
    "question submitted by user. If this is open-question - answer with few "
    "sentences, for ABCD question, return only the indicator of the answer."
)
CAPTION_INSTRUCTIONS = (
    "You act as image caption model. Your task is to provide description of the "
    "image. Caption should be {length}."
)
CLASSIFICATION_INSTRUCTIONS = (
    "You act as single-class classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in Markdown "
    '```json [...]``` markers. Expected structure of json: {"class_name": '
    '"class-name", "confidence": 0.4}. `class-name` must be one of the class names '
    "defined by user. You are only allowed to return single JSON document, even if "
    "there are potentially multiple classes. You are not allowed to return list."
)
MULTI_LABEL_INSTRUCTIONS = (
    "You act as multi-label classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in Markdown "
    '```json``` markers. Expected structure of json: {"predicted_classes": '
    '[{"class": "class-name-1", "confidence": 0.9}, {"class": "class-name-2", '
    '"confidence": 0.7}]}. `class-name-X` must be one of the class names defined by '
    "user and `confidence` is a float value in range 0.0-1.0 that represent how "
    "sure you are that the class is present in the image. Only return class names "
    "that are visible."
)
STRUCTURED_ANSWERING_INSTRUCTIONS = (
    "You are supposed to produce responses in JSON wrapped in Markdown markers: "
    "```json\nyour-response\n```. User is to provide you dictionary with keys and "
    "values. Each key must be present in your response. Values in user dictionary "
    "represent descriptions for JSON fields to be generated. Provide only JSON "
    "Markdown in response."
)
NORMALIZED_DETECTION_INSTRUCTIONS = (
    "You act as object-detection model. You must provide reasonable predictions. "
    "You are only allowed to produce JSON document. "
    'Expected structure of json: {"detections": [{"x_min": 0.1, "y_min": 0.2, '
    '"x_max": 0.3, "y_max": 0.4, "class_name": "my-class-X", "confidence": 0.7}]}. '
    "`my-class-X` must be one of the class names defined by user. All coordinates "
    "must be in range 0.0-1.0, representing percentage of image dimensions. "
    "`confidence` is a value in range 0.0-1.0 representing your confidence in "
    "prediction. You should detect all instances of classes provided by user."
)
STRUCTURED_DETECTION_PROMPT = (
    "Detect all objects in this image. "
    'Output JSON with the key "detections" holding a list where each entry '
    'contains the 2D bounding box in the key "box_2d" and the text label '
    'in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners in absolute pixel coordinates "
    "of the {width}x{height} pixel image. "
    "Only use these labels: {class_list}"
)
CLASS_LIST_TEXT = "List of all classes to be recognised by model: {class_list}"


def _json_schema_format(name: str, item_properties: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "type": "object",
        "properties": item_properties,
        "required": list(item_properties),
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {name: {"type": "array", "items": entry}},
        "required": [name],
        "additionalProperties": False,
    }
    text_format = {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    }

    return text_format


DETECTION_OUTPUT_FORMAT = _json_schema_format(
    "detections",
    {
        "label": {"type": "string"},
        "box_2d": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
)
"""Responses API structured output of the structured detection prompt."""

SEGMENTATION_OUTPUT_FORMAT = _json_schema_format(
    "segmentations",
    {
        "label": {"type": "string"},
        "polygon": {"type": "array", "items": {"type": "integer"}},
    },
)
"""Responses API structured output of the segmentation prompt."""

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

    type = "v2/open_ai@v7"
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


class OpenAITaskRequest(NamedTuple):
    """Responses API fields of one image's request.

    Args:
        payload: ``input`` and, when the task has them, ``instructions`` and
            ``text`` (structured output format).
        decoding_size_wh: Upload size the decoder needs to map coordinates
            back, or ``None`` when the answer is resolution independent.
    """

    payload: Dict[str, Any]
    decoding_size_wh: Optional[SizeWH]


def default_model(model_version: Optional[str], *, task_type: str) -> str:
    """Return the model a step uses: its own, or V1's default for the task.

    Args:
        model_version: The step's ``model_version``; may be a selector.
        task_type: The configured task.

    Returns:
        ``model_version``, or ``gpt-6-astra`` for segmentation and ``gpt-5.1``
        otherwise.
    """
    if model_version is not None:
        return model_version

    model = (
        SEGMENTATION_DEFAULT_MODEL if task_type == SEGMENTATION_TASK else DEFAULT_MODEL
    )

    return model


def detection_style(model: str) -> str:
    """Return the detection prompt style of a model; unknown ones are structured.

    Args:
        model: OpenAI model id.

    Returns:
        ``structured-absolute``, ``plain-absolute`` or ``normalized-legacy``.
    """
    known = OPENAI_MODELS.get(model)
    style = STRUCTURED_ABSOLUTE_STYLE if known is None else known.detection_style

    return style


def box_format_for_model(model: str) -> str:
    """Return the box format a model's detection prompt asks for.

    Args:
        model: OpenAI model id.

    Returns:
        ``xyxy_absolute`` or ``named_normalized``.
    """
    box_format = BOX_FORMAT_BY_STYLE[detection_style(model)]

    return box_format


def resolved_labels(
    classes: Optional[List[str]], output_classes: Optional[List[str]]
) -> Optional[List[str]]:
    """Return the labels answers use: ``output_classes`` if given, else ``classes``.

    Args:
        classes: Class names or visual descriptions.
        output_classes: Labels paired with ``classes``, or ``None``.

    Returns:
        The labels.

    Raises:
        ValueError: With V1's message on a length mismatch or empty label.
    """
    if output_classes is None:
        return classes
    if classes is None or len(classes) != len(output_classes):
        raise ValueError("`output_classes` must have the same length as `classes`")
    if any(not isinstance(label, str) or not label.strip() for label in output_classes):
        raise ValueError("`output_classes` must contain non-empty strings")

    return output_classes


def openai_task_request(
    task_type: str,
    *,
    image: ImageData,
    model: str,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    output_classes: Optional[List[str]],
    image_detail: str,
) -> OpenAITaskRequest:
    """Build the Responses API ``input``, ``instructions`` and ``text`` of a task.

    Detection uploads a lossless PNG at most 2048 px on its longest edge and
    sends no ``detail``; every other task uploads the image as JPEG, as V1.

    Args:
        task_type: One of ``OPENAI_TASKS``.
        image: The image to ask about.
        model: Model id; selects the detection prompt.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names or descriptions.
        output_classes: Labels paired with ``classes``.
        image_detail: ``detail`` of the image part.

    Returns:
        The request fields and the decoding upload size.

    Raises:
        ValueError: On an unknown task or invalid classes.
    """
    if task_type == DETECTION_TASK:
        request = _detection_request(
            image, model=model, classes=classes, output_classes=output_classes
        )
        return request
    if task_type == SEGMENTATION_TASK:
        request = _segmentation_request(
            image,
            classes=classes,
            output_classes=output_classes,
            image_detail=image_detail,
        )
        return request

    chat_prompt = _text_task_prompt(
        task_type, prompt=prompt, output_structure=output_structure, classes=classes
    )
    payload = responses_fields(chat_prompt, encode_image(image), detail=image_detail)
    request = OpenAITaskRequest(payload=payload, decoding_size_wh=None)

    return request


def _text_task_prompt(
    task_type: str,
    *,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> ChatPrompt:
    class_list = ", ".join(classes or [])
    prompts = {
        "unconstrained": lambda: ChatPrompt(None, prompt),
        "ocr": lambda: ChatPrompt(OCR_INSTRUCTIONS, None),
        "visual-question-answering": lambda: ChatPrompt(
            VQA_INSTRUCTIONS, f"Question: {prompt}"
        ),
        "caption": lambda: ChatPrompt(
            CAPTION_INSTRUCTIONS.format(length="short"), None
        ),
        "detailed-caption": lambda: ChatPrompt(
            CAPTION_INSTRUCTIONS.format(length="extensive"), None
        ),
        "classification": lambda: ChatPrompt(
            CLASSIFICATION_INSTRUCTIONS, CLASS_LIST_TEXT.format(class_list=class_list)
        ),
        "multi-label-classification": lambda: ChatPrompt(
            MULTI_LABEL_INSTRUCTIONS, CLASS_LIST_TEXT.format(class_list=class_list)
        ),
        "structured-answering": lambda: ChatPrompt(
            STRUCTURED_ANSWERING_INSTRUCTIONS,
            "Specification of requirements regarding output fields: \n"
            + json.dumps(output_structure, indent=4),
        ),
    }
    if task_type not in prompts:
        raise ValueError(f"Task type: {task_type} not supported.")

    chat_prompt = prompts[task_type]()

    return chat_prompt


def _label_descriptions(classes: List[str], output_classes: Optional[List[str]]) -> str:
    if output_classes is None:
        return ""

    descriptions = [
        {"label": label, "description": description}
        for description, label in zip(classes, output_classes)
    ]
    text = (
        ". Use these visual descriptions to identify each label: "
        f"{json.dumps(descriptions, ensure_ascii=False)}. "
        "Return the label, never its description."
    )

    return text


def _constrained_labels(
    text_format: Dict[str, Any], labels: List[str]
) -> Dict[str, Any]:
    constrained = deepcopy(text_format)
    name = constrained["format"]["name"]
    entry = constrained["format"]["schema"]["properties"][name]["items"]
    entry["properties"]["label"]["enum"] = list(dict.fromkeys(labels))

    return constrained


def _detection_request(
    image: ImageData,
    *,
    model: str,
    classes: Optional[List[str]],
    output_classes: Optional[List[str]],
) -> OpenAITaskRequest:
    labels = resolved_labels(classes, output_classes)
    if labels is None:
        raise ValueError("`classes` must be provided for detection tasks")

    upload_size_wh = max_edge_upload_size(
        (image.width, image.height), max_edge=DETECTION_MAX_EDGE_PIXELS
    )
    upload = encode_image(image, image_format="png", size_wh=upload_size_wh)
    descriptions = _label_descriptions(classes, output_classes)
    style = detection_style(model)
    if style == NORMALIZED_LEGACY_STYLE:
        class_text = CLASS_LIST_TEXT.format(class_list=", ".join(labels))
        chat_prompt = ChatPrompt(
            NORMALIZED_DETECTION_INSTRUCTIONS, class_text + descriptions
        )
        payload = responses_fields(chat_prompt, upload, detail=None)
        request = OpenAITaskRequest(payload=payload, decoding_size_wh=None)
        return request

    width, height = upload.size_wh
    if style == STRUCTURED_ABSOLUTE_STYLE:
        text = STRUCTURED_DETECTION_PROMPT.format(
            width=width, height=height, class_list=", ".join(labels)
        )
    else:
        text = build_object_detection_prompt(
            ABSOLUTE_BOX_FORMAT, classes=labels, upload_size_wh=upload.size_wh
        )
    chat_prompt = ChatPrompt(None, text + descriptions, image_first=True)
    payload = responses_fields(chat_prompt, upload, detail=None)
    if style == STRUCTURED_ABSOLUTE_STYLE:
        payload["text"] = (
            DETECTION_OUTPUT_FORMAT
            if output_classes is None
            else _constrained_labels(DETECTION_OUTPUT_FORMAT, output_classes)
        )
    request = OpenAITaskRequest(payload=payload, decoding_size_wh=upload.size_wh)

    return request


def _segmentation_request(
    image: ImageData,
    *,
    classes: Optional[List[str]],
    output_classes: Optional[List[str]],
    image_detail: str,
) -> OpenAITaskRequest:
    labels = resolved_labels(classes, output_classes)
    if labels is None:
        raise ValueError("`classes` must be provided for instance-segmentation")

    upload = encode_image(image)
    text = build_instance_segmentation_prompt(
        classes=labels, upload_size_wh=upload.size_wh
    ) + _label_descriptions(classes, output_classes)
    payload = responses_fields(
        ChatPrompt(None, text, image_first=True), upload, detail=image_detail
    )
    payload["text"] = (
        SEGMENTATION_OUTPUT_FORMAT
        if output_classes is None
        else _constrained_labels(SEGMENTATION_OUTPUT_FORMAT, output_classes)
    )
    request = OpenAITaskRequest(payload=payload, decoding_size_wh=upload.size_wh)

    return request


def responses_fields(
    chat_prompt: ChatPrompt, image: EncodedImage, *, detail: Optional[str]
) -> Dict[str, Any]:
    """Format a prompt as Responses API ``input`` and ``instructions``.

    Args:
        chat_prompt: System text, user text and their order.
        image: The uploaded image.
        detail: ``detail`` of the image part; ``None`` sends none.

    Returns:
        ``{"input": [...]}``, with ``"instructions"`` when the prompt has a
        system text.
    """
    image_part = {"type": "input_image", "image_url": image.data_url}
    if detail is not None:
        image_part["detail"] = detail
    parts = [image_part]
    if chat_prompt.text is not None:
        text_part = {"type": "input_text", "text": chat_prompt.text}
        parts = (
            [image_part, text_part]
            if chat_prompt.image_first
            else [text_part, image_part]
        )

    fields: Dict[str, Any] = {"input": [{"role": "user", "content": parts}]}
    if chat_prompt.system is not None:
        fields["instructions"] = chat_prompt.system

    return fields


def responses_options(
    model: str,
    *,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> Dict[str, Any]:
    """Return ``model`` and the set generation options of a Responses request.

    Reasoning is sent only to models known to support it, as in V1.

    Args:
        model: OpenAI model id.
        reasoning_effort: Reasoning level or ``None``.
        max_tokens: Output token limit or ``None``.
        temperature: Sampling temperature or ``None``.

    Returns:
        The request options.

    Raises:
        ValueError: When the model does not support ``reasoning_effort``.
    """
    check_reasoning_effort(model, reasoning_effort, levels_by_model=REASONING_LEVELS)
    options: Dict[str, Any] = {"model": model}
    if max_tokens is not None:
        options["max_output_tokens"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if reasoning_effort is not None and REASONING_LEVELS.get(model):
        options["reasoning"] = {"effort": reasoning_effort}

    return options


def read_responses_answer(data: Dict[str, Any]) -> ChatAnswer:
    """Read a Responses API result, from the SDK or the Roboflow proxy.

    Args:
        data: The response as a JSON dictionary.

    Returns:
        The concatenated ``output_text`` parts and token usage.

    Raises:
        ChatRequestError: With V1's message on a failed, cancelled, incomplete
            or unexpected status, or an answer without text.
    """
    status = data.get("status")
    if status == "failed":
        error = data.get("error") or {}
        details = (
            f"{error.get('code', 'Unknown')}: {error.get('message', 'Unknown error')}"
        )
        raise ChatRequestError(f"OpenAI API request failed: {details}")
    if status == "cancelled":
        raise ChatRequestError("OpenAI API request was cancelled.")
    if status == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason", "Unknown reason")
        if reason == "max_output_tokens":
            raise ChatRequestError(
                "OpenAI API stopped generation because the max_tokens limit was "
                "reached. Please increase the max_tokens parameter to allow for a "
                "complete response."
            )
        raise ChatRequestError(
            f"OpenAI API returned an incomplete response. Reason: {reason}"
        )
    if status not in ("completed", "in_progress", "queued", None):
        raise ChatRequestError(f"OpenAI API returned unexpected status: {status}")

    texts = [
        content.get("text", "")
        for item in data.get("output") or []
        if item.get("type") == "message"
        for content in item.get("content") or []
        if content.get("type") == "output_text"
    ]
    text = "".join(texts)
    if not text:
        raise ChatRequestError("OpenAI API returned no text content in response.")

    usage = data.get("usage") or {}
    answer = ChatAnswer(
        text=text,
        input_tokens=token_count(usage.get("input_tokens")),
        output_tokens=token_count(usage.get("output_tokens")),
    )

    return answer


class OpenAICompatibleBlock(Block):
    """Send a prompt, with optional images, to any OpenAI-compatible server.

    Example: ``base_url="http://localhost:8000/v1"``, ``model_name="qwen3-vl"``,
    ``prompt="Describe {{ $parameters.photo }}"`` with
    ``prompt_parameters={"photo": "$inputs.image"}`` sends the text
    "Describe" and the image as a vision part. Failures do not raise: the
    step returns ``output=""`` and the message in ``error_status``.
    """

    type = "v2/openai_compatible@v1"
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


def compile_parameter_operations(
    operations: Dict[str, List[Any]], *, keys: Tuple[str, ...]
) -> Dict[str, CompiledOperations]:
    """Compile ``prompt_parameters_operations``; each must name a parameter.

    Args:
        operations: UQL operations by parameter name.
        keys: Keys of ``prompt_parameters``.

    Returns:
        Compiled operations by parameter name.

    Raises:
        ValueError: When an operations key is not a ``prompt_parameters`` key.
    """
    compiled = {}
    for name, chain in operations.items():
        if name not in keys:
            raise ValueError(
                f"prompt_parameters_operations names `{name}`, which is not a key "
                f"of prompt_parameters (keys: {list(keys)})"
            )
        compiled[name] = compile_operations(
            chain,
            path=f"$.prompt_parameters_operations.{name}",
            declared_names=frozenset(),
        )

    return compiled


def compatible_prompt(prompt: str, values: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Split prompt parameters into text substitutions and image parts, as V1.

    Images (``ImageData``, JPEG ``bytes`` and lists holding them) become
    ``data:`` URLs: referenced ones first, in first-use order, then the
    others. Their placeholders are removed; other referenced values are
    inserted with ``str``. The text is stripped.

    Args:
        prompt: Template with ``{{ $parameters.NAME }}`` placeholders.
        values: Parameter values after operations.

    Returns:
        ``(text, image_urls)``.
    """
    referenced = list(
        dict.fromkeys(
            name for _, name in PARAMETER_PATTERN.findall(prompt) if name in values
        )
    )
    ordered = referenced + [name for name in values if name not in referenced]
    image_urls: List[str] = []
    image_names = set()
    for name in ordered:
        urls = _image_urls(values[name])
        if urls:
            image_urls.extend(urls)
            image_names.add(name)

    def substitute(match: "re.Match[str]") -> str:
        name = match.group(2)
        if name not in values:
            return match.group(0)
        text = "" if name in image_names else str(values[name])
        return text

    text = PARAMETER_PATTERN.sub(substitute, prompt).strip()

    return text, image_urls


def _image_urls(value: Any) -> List[str]:
    if isinstance(value, list):
        items = [item for item in value if _is_image(item)]
    else:
        items = [value] if _is_image(value) else []

    urls = [_image_url(item) for item in items]

    return urls


def _is_image(value: Any) -> bool:
    image = isinstance(value, (ImageData, bytes))

    return image


def _image_url(value: Any) -> str:
    if isinstance(value, ImageData):
        url = encode_image(value).data_url
        return url

    url = f"data:image/jpeg;base64,{base64.b64encode(value).decode('ascii')}"

    return url


BLOCKS = (OpenAIBlock, OpenAICompatibleBlock)

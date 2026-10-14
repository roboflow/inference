"""Request and answer helpers of the OpenAI GPT and OpenAI-compatible blocks.

Used by ``open_ai/v7.py`` and ``openai_compatible/v1.py``, the ports of V1
``roboflow_core/open_ai@v7`` and ``roboflow_core/openai_compatible@v1``; each
block owns its Params, outputs and ``run``. They speak different OpenAI APIs::

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
from copy import deepcopy
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from openai import DefaultHttpxClient, OpenAI
from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr
from roboflow_workflows.v2.blocks.language.chat import (
    ChatAnswer,
    ChatPrompt,
    ChatRequestError,
    EncodedImage,
    SizeWH,
    check_reasoning_effort,
    encode_image,
    token_count,
)
from roboflow_workflows.v2.blocks.language.chat_completions import (  # noqa: F401
    chat_completion_usage,
    chat_messages,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    DETECTION_MAX_EDGE_PIXELS,
    build_instance_segmentation_prompt,
    build_object_detection_prompt,
    max_edge_upload_size,
)
from roboflow_workflows.v2.blocks.query import CompiledOperations, compile_operations
from roboflow_workflows.v2.data.image import ImageData

PARAMETER_PATTERN = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")

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

REASONING_LEVELS = {
    model_id: model.reasoning_levels for model_id, model in OPENAI_MODELS.items()
}

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

SEGMENTATION_OUTPUT_FORMAT = _json_schema_format(
    "segmentations",
    {
        "label": {"type": "string"},
        "polygon": {"type": "array", "items": {"type": "integer"}},
    },
)


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
        task_type: One of ``OPENAI_TASKS`` (open_ai/v7.py).
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

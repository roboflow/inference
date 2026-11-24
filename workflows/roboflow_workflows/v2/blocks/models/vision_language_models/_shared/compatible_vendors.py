"""Request helpers used by the Gemma, SpaceXAI and Qwen OpenRouter blocks.

Gemma and SpaceXAI are active. Qwen 3.5 and 3.6 OpenRouter are retired and
excluded from BLOCKS; current providers return native predictions. Each block
owns its model table, Params and ``run``; the requests reuse the helpers
below and the shared chat helpers::

    google_gemma@v4 (OpenRouterBlockBase, the request loop of openrouter@v3)
      model label ──▶ OpenRouter slug; openrouter@v3 task messages
      api_key rf_key:* ─▶ platform proxy;  sk-or-.. ─▶ OpenRouter directly
      answer ──▶ decode_vlm_output(named_normalized) ──▶ predictions

    qwen3_5_openrouter@v1, qwen3_6_openrouter@v1 (own OpenRouter key only)
      model label ──▶ OpenRouter slug; V1 Qwen messages
      ──▶ chat.completions.create(model, messages, max_tokens, temperature)
      raw answer only; no provider routing, quantization filter or reasoning

    spacexai@v3 (xAI Responses API)
      task ──▶ JPEG, detail "auto"
      object-detection ──▶ lossless PNG at full size, detail "high", image first
      api_key rf_key:* ─▶ platform_client.post("apiproxy/xai", payload)
      xAI key          ─▶ OpenAI(base_url=xai_base_url).responses.create(store=False)
      a direct error naming the reasoning config ─▶ retried once without it, as V1
      answer ──▶ decode_vlm_output(xyxy_percent) ──▶ predictions

Differences from V1:

* A failed request raises ``ChatRequestError`` with the keys scrubbed and no
  chained exception. V1 raised the vendor's exception, whose text may contain
  the key.
* ``max_concurrent_requests`` must be at least 1; unset means 8 instead of an
  environment variable. After a failure, requests not yet started are skipped.
* ``inference_id`` is a hex UUID, as in every V2 block; V1 used the dashed form.
* Predictions are native V2 carriers; see ``decoding``.
* Qwen: a Roboflow-managed ``rf_key:`` key fails before any request (V1 sent
  it to OpenRouter as a bearer token). An answer without ``message.content``
  fails with the OpenRouter message instead of returning ``None`` as output.
"""

import base64
import json
import logging
from typing import Any, Dict, List, Mapping, Optional
import cv2
import numpy as np

# spacexai/v3.py calls compatible_vendors.OpenAI at call time; tests patch it here.
from openai import OpenAI  # noqa: F401
from roboflow_workflows.v2.blocks.models.vision_language_models.chat import (
    ChatAnswer,
    ChatPrompt,
    ChatRequestError,
    EncodedImage,
    encode_image,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.decoding import (
    build_object_detection_prompt,
)
from roboflow_workflows.v2.blocks.models.vision_language_models._shared.openai import (
    CAPTION_INSTRUCTIONS,
    CLASS_LIST_TEXT,
    CLASSIFICATION_INSTRUCTIONS,
    MULTI_LABEL_INSTRUCTIONS,
    OCR_INSTRUCTIONS,
    STRUCTURED_ANSWERING_INSTRUCTIONS,
    VQA_INSTRUCTIONS,
    chat_messages,
    read_responses_answer,
    responses_fields,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.openrouter.v3 import (
    DETECTION_TASK,
    LEGACY_DETECTION_BOX_FORMAT,
    openrouter_task_messages,
)
from roboflow_workflows.v2.data.image import ImageData

logger = logging.getLogger(__name__)

QWEN_VQA_SYSTEM_PROMPT = (
    "You act as Visual Question Answering model. Your task is to provide answer to "
    "questionsubmitted by user. If this is open-question - answer with few "
    "sentences, for ABCD question, return only the indicator of the answer."
)

XAI_DETECTION_BOX_FORMAT = "xyxy_percent"

XAI_PNG_COMPRESSION = 9

_XAI_REASONING_REJECTION_MARKERS = (
    "unsupported",
    "not supported",
    "unknown",
    "invalid",
)


def model_slug(label: str, *, models: Mapping[str, str]) -> str:
    """Return the provider slug of a model label, as V1 maps it.

    Args:
        label: The label chosen in ``model_version``.
        models: Label to slug table of the block.

    Returns:
        The slug.

    Raises:
        ValueError: With V1 Qwen's message when the label is unknown.
    """
    slug = models.get(label)
    if slug is None:
        raise ValueError(
            f"Invalid model name: '{label}'. Please use one of {list(models)}."
        )

    return slug


def qwen_task_messages(
    task_type: str,
    *,
    image: ImageData,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Build V1 Qwen's chat messages of one task; the image goes as JPEG, full size.

    V1 Qwen's prompts are the OpenRouter ones except the VQA system prompt.

    Args:
        task_type: One of ``OPENROUTER_TASKS``.
        image: The image to ask about.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        The ``messages`` list.

    Raises:
        ValueError: On an unknown task.
    """
    if task_type == "visual-question-answering":
        upload = encode_image(image)
        chat_prompt = ChatPrompt(QWEN_VQA_SYSTEM_PROMPT, f"Question: {prompt}")
        messages = chat_messages(chat_prompt, [upload.data_url])
        return messages

    messages = openrouter_task_messages(
        task_type,
        image=image,
        prompt=prompt,
        output_structure=output_structure,
        classes=classes,
        detection_format=LEGACY_DETECTION_BOX_FORMAT,
    )

    return messages


def xai_task_request(
    task_type: str,
    *,
    image: ImageData,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> Dict[str, Any]:
    """Build V1 spacexai@v3's Responses API ``input`` and ``instructions``.

    Object detection uploads a lossless PNG at full size with ``detail``
    ``"high"`` and the image before the percent-box prompt. Every other task
    uploads a JPEG with ``detail`` ``"auto"`` and V1's OpenAI instructions.

    Args:
        task_type: One of ``OPENROUTER_TASKS`` (the same nine tasks).
        image: The image to ask about.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        ``{"input": [...]}``, with ``"instructions"`` when the task has them.

    Raises:
        ValueError: On an unknown task.
    """
    if task_type == DETECTION_TASK:
        text = build_object_detection_prompt(XAI_DETECTION_BOX_FORMAT, classes=classes)
        chat_prompt = ChatPrompt(None, text, image_first=True)
        fields = responses_fields(chat_prompt, lossless_png(image), detail="high")
        return fields

    class_text = CLASS_LIST_TEXT.format(class_list=", ".join(classes or []))
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
        "classification": lambda: ChatPrompt(CLASSIFICATION_INSTRUCTIONS, class_text),
        "multi-label-classification": lambda: ChatPrompt(
            MULTI_LABEL_INSTRUCTIONS, class_text
        ),
        "structured-answering": lambda: ChatPrompt(
            STRUCTURED_ANSWERING_INSTRUCTIONS,
            "Specification of requirements regarding output fields: \n"
            + json.dumps(output_structure, indent=4),
        ),
    }
    if task_type not in prompts:
        raise ValueError(f"Task type: {task_type} not supported.")

    fields = responses_fields(prompts[task_type](), encode_image(image), detail="auto")

    return fields


def lossless_png(image: ImageData) -> EncodedImage:
    """Encode an image as V1 spacexai@v3 uploads detections: PNG, level 9.

    ``chat.encode_image`` writes PNG at OpenCV's default level, which V1
    spacexai@v3 avoided; the pixels are the same, the file is smaller.

    Args:
        image: RGB or single-channel image.

    Returns:
        The encoded image at its own size.

    Raises:
        ValueError: When OpenCV cannot encode the image.
    """
    pixels = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    if pixels.shape[2] == 3:
        pixels = cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_RGB2BGR)
    else:
        pixels = np.ascontiguousarray(pixels[:, :, 0])
    encoded, data = cv2.imencode(
        ".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, XAI_PNG_COMPRESSION]
    )
    if not encoded:
        raise ValueError(f"OpenCV could not encode an image of shape {pixels.shape}")

    height, width = pixels.shape[:2]
    upload = EncodedImage(
        base64=base64.b64encode(data.tobytes()).decode("ascii"),
        media_type="image/png",
        size_wh=(width, height),
    )

    return upload


def read_xai_answer(data: Dict[str, Any]) -> ChatAnswer:
    """Read an xAI Responses API result with V1 spacexai@v3's messages.

    V1's xAI reader is V1's OpenAI reader with "xAI API" in its messages, so
    this reuses ``read_responses_answer`` and renames the API.

    Args:
        data: The response as a JSON dictionary.

    Returns:
        The concatenated ``output_text`` parts and token usage.

    Raises:
        ChatRequestError: On a failed, cancelled, incomplete or unexpected
            status, or an answer without text.
    """
    try:
        answer = read_responses_answer(data)
    except ChatRequestError as error:
        message = str(error).replace("OpenAI API", "xAI API", 1)
    else:
        return answer

    raise ChatRequestError(message)


def rejects_reasoning(error: Exception) -> bool:
    """Whether an xAI error rejected the request's reasoning config (V1 rule).

    Args:
        error: The direct request's exception.

    Returns:
        ``True`` when the message names ``reasoning`` and a rejection marker.
    """
    message = str(error).lower()
    rejected = "reasoning" in message and any(
        marker in message for marker in _XAI_REASONING_REJECTION_MARKERS
    )

    return rejected

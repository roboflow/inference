"""OpenRouter vendor helpers: model records, capped image upload, image-first prompts.

The Kimi, Meta Muse, Z.ai GLM and Llama blocks own their model tables,
Params, outputs and requests. They send OpenRouter chat completions through
``OpenRouterBlockBase``, which owns keys, privacy routing, quantization
filters, the reasoning retry and the Roboflow proxy::

    model_version label ──▶ VendorModel (slug, box format, reasoning rules)
    per image:
      kimi, llama ─▶ openrouter_task_messages      (V1 system + text + image)
      meta, zai   ─▶ encode_capped_jpeg + image_first_task_messages
                     (one user message: image, text; JPEG shrunk 10% at a
                     time over 9.5 MB base64)
    OpenRouterBlockBase.ask_all ─▶ answers in image order
    kimi, meta, zai: answer ─▶ decode_vlm_output(model box format) ─▶ predictions
    llama: raw answer only (V1 v2 decodes nothing)

``decode_answers`` and ``DECODED_PARSERS`` are used by the OpenRouter path of
``roboflow_core/qwen_vlm@v4`` (models/qwen_vlm/v4.py).

Differences from V1:

* A failed request raises ``ChatRequestError`` with the keys scrubbed.
* ``max_concurrent_requests`` must be at least 1; unset means 8 instead of an
  environment variable.
* Predictions are native V2 carriers; see ``decoding``.
* Every check runs again on the resolved values before any request: a key
  that resolved to ``None`` (V1 failed on it too; the shared OpenRouter base
  would send the ``OPENAI_API_KEY`` environment variable instead), an
  unknown model label (V1 Kimi and Llama raised ``KeyError``), a reasoning
  level the model rejects, a task field that resolved to ``None``, and
  invalid or duplicate ``classes`` of a decoding task (V1 sent the request
  first).
* A ``model_version`` selector is reported to workload introspection as the
  selector; V1 attached a label-to-slug resolver.
"""

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import cv2
import torch
from roboflow_workflows.v2.blocks.language.chat import (
    ChatAnswer,
    EncodedImage,
    encode_image,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    CLASSIFICATION_TASKS,
    DETECTION_TASKS,
    build_object_detection_prompt,
    decode_vlm_output,
    detection_box_format,
    max_edge_upload_size,
)
from roboflow_workflows.v2.blocks.language.openrouter.v3 import (
    CAPTION_SYSTEM_PROMPT,
    CLASS_LIST_TEXT,
    CLASSIFICATION_SYSTEM_PROMPT,
    LEGACY_DETECTION_BOX_FORMAT,
    MULTI_LABEL_SYSTEM_PROMPT,
    OCR_SYSTEM_PROMPT,
    STRUCTURED_ANSWERING_SYSTEM_PROMPT,
    VQA_SYSTEM_PROMPT,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData

OPENROUTER_MAX_BASE64_BYTES = 9_500_000

SHRINK_FACTOR = 0.9

DECODING_TASKS = CLASSIFICATION_TASKS + DETECTION_TASKS


@dataclass(frozen=True)
class VendorModel:
    """One selectable model of a vendor block.

    Args:
        slug: OpenRouter model slug.
        box_format: Box format detection answers are decoded with; must be
            normalized (no upload size).
        reasoning_levels: Accepted ``reasoning_effort`` values; empty when the
            block has no reasoning setting.
        detection_prompt: Template overriding the box format's own detection
            prompt; formatted with ``{class_list}``.
        reasoning_required: Whether the model rejects disabled reasoning, so
            ``none`` is sent as ``low``.

    Raises:
        ValueError: On an unknown box format or one needing the upload size.
    """

    slug: str
    box_format: str = LEGACY_DETECTION_BOX_FORMAT
    reasoning_levels: Tuple[str, ...] = ()
    detection_prompt: Optional[str] = None
    reasoning_required: bool = False

    def __post_init__(self) -> None:
        if detection_box_format(self.box_format).requires_upload_size:
            raise ValueError(
                f"Box format {self.box_format!r} needs the upload size; vendor "
                "blocks decode normalized formats only"
            )

    def detection_text(self, classes: Sequence[str]) -> str:
        """Render this model's object-detection prompt.

        Args:
            classes: Class names the model may answer with.

        Returns:
            The prompt text.
        """
        if self.detection_prompt is not None:
            text = self.detection_prompt.format(class_list=", ".join(classes))
            return text

        text = build_object_detection_prompt(self.box_format, classes=classes)

        return text


ZAI_REQUIRED_REASONING_FALLBACK = "low"

DECODED_PARSERS = {"structured-answering": "roboflow_core/json_parser@v1"}


def resolve_model(
    label: str, *, models: Mapping[str, VendorModel], family: str
) -> VendorModel:
    """Look up a model by its V1 label.

    Args:
        label: The resolved ``model_version``.
        models: The block's model table.
        family: Name in the error message, e.g. ``"Muse variant"``.

    Returns:
        The model.

    Raises:
        ValueError: When the label is not in ``models``.
    """
    model = models.get(label)
    if model is None:
        raise ValueError(f"Unknown {family} '{label}'. Pick one of: {list(models)}")

    return model


def encode_capped_jpeg(image: ImageData) -> EncodedImage:
    """Encode an image as JPEG, shrinking it until its base64 fits OpenRouter.

    V1 Meta and Z.ai rule: while the base64 text exceeds
    ``OPENROUTER_MAX_BASE64_BYTES``, the previous step's pixels are resized
    with ``INTER_AREA`` so the longest edge is 90% of what it was. Their box
    formats are normalized, so the upload size does not affect decoding.

    Args:
        image: The image to upload.

    Returns:
        The encoded upload with its pixel size.
    """
    upload = encode_image(image)
    if len(upload.base64) <= OPENROUTER_MAX_BASE64_BYTES:
        return upload

    pixels = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    while len(upload.base64) > OPENROUTER_MAX_BASE64_BYTES and max(upload.size_wh) > 1:
        longest_edge = max(int(max(upload.size_wh) * SHRINK_FACTOR), 1)
        size_wh = max_edge_upload_size(upload.size_wh, max_edge=longest_edge)
        pixels = cv2.resize(pixels, size_wh, interpolation=cv2.INTER_AREA)
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        shrunk = ImageData.from_tensor(torch.from_numpy(pixels).permute(2, 0, 1))
        upload = encode_image(shrunk)

    return upload


def image_first_task_messages(
    task_type: str,
    *,
    upload: EncodedImage,
    model: VendorModel,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Build V1 Meta/Z.ai messages: one user message, image first, no system.

    Args:
        task_type: One of ``OPENROUTER_TASKS``.
        upload: The encoded image.
        model: The selected model; gives the detection prompt.
        prompt: Prompt or question.
        output_structure: Fields of a structured answer.
        classes: Class names.

    Returns:
        The ``messages`` list.

    Raises:
        ValueError: On an unknown task.
    """
    class_text = CLASS_LIST_TEXT.format(class_list=", ".join(classes or []))
    texts = {
        "unconstrained": lambda: prompt,
        "ocr": lambda: OCR_SYSTEM_PROMPT,
        "visual-question-answering": lambda: (
            f"{VQA_SYSTEM_PROMPT}\n\nQuestion: {prompt}"
        ),
        "caption": lambda: CAPTION_SYSTEM_PROMPT.format(length="short"),
        "detailed-caption": lambda: CAPTION_SYSTEM_PROMPT.format(length="extensive"),
        "classification": lambda: f"{CLASSIFICATION_SYSTEM_PROMPT}\n\n{class_text}",
        "multi-label-classification": lambda: (
            f"{MULTI_LABEL_SYSTEM_PROMPT}\n\n{class_text}"
        ),
        "structured-answering": lambda: (
            f"{STRUCTURED_ANSWERING_SYSTEM_PROMPT}\n\n"
            "Specification of requirements regarding output fields: \n"
            f"{json.dumps(output_structure, indent=4)}"
        ),
        "object-detection": lambda: model.detection_text(classes),
    }
    if task_type not in texts:
        raise ValueError(f"Task type: {task_type} not supported.")

    content = [
        {"type": "image_url", "image_url": {"url": upload.data_url}},
        {"type": "text", "text": texts[task_type]()},
    ]
    messages = [{"role": "user", "content": content}]

    return messages


def decode_answers(
    images: Batch,
    answers: List[ChatAnswer],
    *,
    model: VendorModel,
    task_type: str,
    classes: Optional[List[str]],
    with_reasoning: bool,
) -> List[Dict[str, Any]]:
    """Decode each answer with the model's box format.

    Args:
        images: The images, in request order.
        answers: One answer per image.
        model: The selected model.
        task_type: What was asked.
        classes: Class names.
        with_reasoning: Whether to add ``thinking`` and token counts.

    Returns:
        One ``{"output", "classes", ["thinking", "input_tokens",
        "output_tokens",] "predictions", "error_status", "inference_id"}``
        per image.
    """
    results = []
    for image, answer in zip(images, answers):
        inference_id = str(uuid.uuid4())
        error_status, predictions = decode_vlm_output(
            task_type,
            answer.text,
            image=image,
            classes=classes,
            inference_id=inference_id,
            box_format=model.box_format,
        )
        result: Dict[str, Any] = {"output": answer.text, "classes": classes}
        if with_reasoning:
            result["thinking"] = answer.thinking
            result["input_tokens"] = answer.input_tokens
            result["output_tokens"] = answer.output_tokens
        result["predictions"] = predictions
        result["error_status"] = error_status
        result["inference_id"] = inference_id
        results.append(result)

    return results

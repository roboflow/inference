"""Florence2 request/response translation for ModelManagerAdapter.

Legacy Florence2 feeds the raw request prompt to the model and returns the HF
processor's ``post_process_generation`` dict keyed by the task token taken
from the prompt. The MMP wire exposes the raw decoded generation through the
vlm ``prompt`` action, so this module re-applies the same post-processing on
the legacy side: a pure-Python mirror of ``Florence2Processor``
(``transformers.models.florence2.processing_florence2``) parameterized with
the post-processor config shipped in the served Florence2 model packages.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.exceptions import ModelArtefactError
from inference.core.utils.image_utils import load_image_rgb

FLORENCE2_MODEL_CLASS = "Florence2HF"

# Legacy TransformerModel.predict generation cap; the legacy stack ignores
# per-request max_new_tokens for Florence2.
_LEGACY_MAX_NEW_TOKENS = 1000

_TASK_POST_PROCESSING_TYPE = {
    "<OCR>": "pure_text",
    "<OCR_WITH_REGION>": "ocr",
    "<CAPTION>": "pure_text",
    "<DETAILED_CAPTION>": "pure_text",
    "<MORE_DETAILED_CAPTION>": "pure_text",
    "<OD>": "description_with_bboxes",
    "<DENSE_REGION_CAPTION>": "description_with_bboxes",
    "<CAPTION_TO_PHRASE_GROUNDING>": "phrase_grounding",
    "<REFERRING_EXPRESSION_SEGMENTATION>": "polygons",
    "<REGION_TO_SEGMENTATION>": "polygons",
    "<OPEN_VOCABULARY_DETECTION>": "description_with_bboxes_or_polygons",
    "<REGION_TO_CATEGORY>": "pure_text",
    "<REGION_TO_DESCRIPTION>": "pure_text",
    "<REGION_TO_OCR>": "pure_text",
    "<REGION_PROPOSAL>": "bboxes",
}

_OCR_PATTERN = (
    r"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
)
_BOX_PATTERN = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
_PHRASE_WITH_BOXES_PATTERN = r"([^<]+(?:<loc_\d+>){4,})"
_EMPTY_PHRASE_WITH_BOXES_PATTERN = r"(?:(?:<loc_\d+>){4,})"
_PHRASE_TEXT_PATTERN = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
_POLY_PHRASE_PATTERN = r"([^<]+(?:<loc_\d+>|<sep>|<poly>|</poly>){4,})"
_POLY_EMPTY_PHRASE_PATTERN = r"(?:(?:<loc_\d+>|<sep>|<poly>|</poly>){4,})"
_POLY_PHRASE_TEXT_PATTERN = (
    r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_|<poly>)"
)
_POLY_INSTANCE_PATTERN = r"<poly>(.*?)</poly>"
_POLY_BOX_PATTERN = r"((?:<loc_\d+>)+)(?:<sep>|$)"

# phrase_grounding.banned_grounding_tokens from the Florence2 model package
# processor_config.json.
_BANNED_GROUNDING_TOKENS = frozenset(
    [
        "it",
        "I",
        "me",
        "mine",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "one",
        "oneself",
        "we",
        "us",
        "our",
        "ours",
        "its",
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "yourselves",
        "themselves",
        "this",
        "that",
        "these",
        "those",
        "who",
        "whom",
        "whose",
        "which",
        "what",
        "all",
        "another",
        "any",
        "anybody",
        "anyone",
        "anything",
        "each",
        "everybody",
        "everyone",
        "everything",
        "few",
        "many",
        "nobody",
        "none",
        "several",
        "some",
        "somebody",
        "someone",
        "something",
        "each other",
        "one another",
        "the image",
        "image",
        "images",
        "the",
        "a",
        "an",
        "a group",
        "other objects",
        "lots",
        "a set",
    ]
)


def is_florence2_route(route: dict) -> bool:
    if FLORENCE2_MODEL_CLASS in (route.get("model_mro_names") or []):
        return True
    return route.get("model_class_name") == FLORENCE2_MODEL_CLASS


def ensure_image_input_supported(request: Any) -> None:
    """Legacy Florence2 hands the raw request image straight to
    load_image_rgb, which rejects image lists; reproduce that rejection."""
    image = getattr(request, "image", None)
    if isinstance(image, list):
        load_image_rgb(image)


def build_prompt_params(request: Any) -> dict:
    prompt = getattr(request, "prompt", None)
    derive_task_token(prompt)
    return {
        "prompt": prompt,
        "max_new_tokens": _LEGACY_MAX_NEW_TOKENS,
        "do_sample": False,
        "skip_special_tokens": False,
    }


def derive_task_token(prompt: str) -> str:
    return prompt.split(">")[0] + ">"


def repack_response(
    prediction: Any, request: Any, dims: Tuple[int, int]
) -> LMMInferenceResponse:
    if isinstance(prediction, list) and len(prediction) == 1:
        prediction = prediction[0]
    if not isinstance(prediction, str):
        raise ModelArtefactError(
            "Unexpected Florence2 prediction shape from the inference backend."
        )
    task = derive_task_token(request.prompt)
    response = post_process_generation(text=prediction, task=task, image_size=dims)
    width, height = dims
    return LMMInferenceResponse(
        response=response,
        image=InferenceResponseImage(width=width, height=height),
    )


def post_process_generation(text: str, task: str, image_size: Tuple[int, int]) -> dict:
    post_proc_type = _TASK_POST_PROCESSING_TYPE.get(task, "pure_text")
    if post_proc_type == "pure_text":
        final_answer: Any = text.replace("<s>", "").replace("</s>", "").strip()
    elif post_proc_type in ("description_with_bboxes", "bboxes"):
        parsed = _parse_description_with_bboxes(
            text, image_size, allow_empty_phrase=post_proc_type == "bboxes"
        )
        final_answer = {
            "bboxes": [instance["bbox"] for instance in parsed],
            "labels": [instance["cat_name"] for instance in parsed],
        }
    elif post_proc_type == "ocr":
        parsed = _parse_ocr(text, image_size)
        final_answer = {
            "quad_boxes": [instance["quad_box"] for instance in parsed],
            "labels": [instance["text"] for instance in parsed],
        }
    elif post_proc_type == "phrase_grounding":
        bboxes = []
        labels = []
        for instance in _parse_phrase_grounding(text, image_size):
            for bbox in instance["bbox"]:
                bboxes.append(bbox)
                labels.append(instance["cat_name"])
        final_answer = {"bboxes": bboxes, "labels": labels}
    elif post_proc_type == "polygons":
        parsed = _parse_description_with_polygons(
            text, image_size, allow_empty_phrase=True
        )
        final_answer = {
            "polygons": [instance["polygons"] for instance in parsed],
            "labels": [instance["cat_name"] for instance in parsed],
        }
    else:
        if "<poly>" in text:
            parsed = _parse_description_with_polygons(text, image_size)
        else:
            parsed = _parse_description_with_bboxes(text, image_size)
        bboxes = []
        bboxes_labels = []
        polygons = []
        polygons_labels = []
        for instance in parsed:
            if "polygons" in instance:
                polygons.append(instance["polygons"])
                polygons_labels.append(instance["cat_name"])
            else:
                bboxes.append(instance["bbox"])
                bboxes_labels.append(instance["cat_name"])
        final_answer = {
            "bboxes": bboxes,
            "bboxes_labels": bboxes_labels,
            "polygons": polygons,
            "polygons_labels": polygons_labels,
        }
    return {task: final_answer}


def _dequantize(bins: List[int], image_size: Tuple[int, int]) -> List[int]:
    """(x, y) bin pairs -> pixel coordinates, mirroring the float32 arithmetic
    and int truncation of the HF processor's dequantize."""
    width, height = image_size
    pairs = np.asarray(bins, dtype=np.float32).reshape(-1, 2) + np.float32(0.5)
    pairs[:, 0] *= np.float32(width / 1000)
    pairs[:, 1] *= np.float32(height / 1000)
    return pairs.astype(np.int32).flatten().tolist()


def _strip_special_tokens(text: str) -> str:
    return text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")


def _parse_ocr(text: str, image_size: Tuple[int, int]) -> List[dict]:
    text = _strip_special_tokens(text)
    instances = []
    for content, *quad_str in re.findall(_OCR_PATTERN, text):
        quad_bins = [int(value) for value in quad_str]
        instances.append(
            {
                "quad_box": _dequantize(quad_bins, image_size),
                "text": content.strip(),
            }
        )
    return instances


def _parse_phrase_grounding(text: str, image_size: Tuple[int, int]) -> List[dict]:
    text = _strip_special_tokens(text)
    instances = []
    for phrase_text in re.findall(_PHRASE_WITH_BOXES_PATTERN, text):
        phrase_text = phrase_text.replace("<ground>", "", 1).replace("<obj>", "", 1)
        if not phrase_text:
            continue
        match = re.search(_PHRASE_TEXT_PATTERN, phrase_text)
        if not match:
            continue
        phrase = match.group().strip()
        if phrase in _BANNED_GROUNDING_TOKENS:
            continue
        boxes_matches = list(re.finditer(_BOX_PATTERN, phrase_text))
        if not boxes_matches:
            continue
        bboxes = [
            _dequantize([int(m.group(j)) for j in range(1, 5)], image_size)
            for m in boxes_matches
        ]
        phrase = phrase.encode("ascii", "ignore").decode("ascii")
        instances.append({"bbox": bboxes, "cat_name": phrase})
    return instances


def _parse_description_with_bboxes(
    text: str, image_size: Tuple[int, int], allow_empty_phrase: bool = False
) -> List[dict]:
    text = _strip_special_tokens(text)
    pattern = (
        _EMPTY_PHRASE_WITH_BOXES_PATTERN
        if allow_empty_phrase
        else _PHRASE_WITH_BOXES_PATTERN
    )
    instances = []
    for phrase_text in re.findall(pattern, text):
        phrase_text = phrase_text.replace("<ground>", "", 1).replace("<obj>", "", 1)
        if not phrase_text and not allow_empty_phrase:
            continue
        match = re.search(_PHRASE_TEXT_PATTERN, phrase_text)
        if not match:
            continue
        phrase = match.group().strip()
        boxes_matches = list(re.finditer(_BOX_PATTERN, phrase_text))
        if not boxes_matches:
            continue
        phrase = phrase.encode("ascii", "ignore").decode("ascii")
        for boxes_match in boxes_matches:
            bbox = _dequantize(
                [int(boxes_match.group(j)) for j in range(1, 5)], image_size
            )
            instances.append({"bbox": bbox, "cat_name": phrase})
    return instances


def _parse_description_with_polygons(
    text: str, image_size: Tuple[int, int], allow_empty_phrase: bool = False
) -> List[dict]:
    text = _strip_special_tokens(text)
    pattern = _POLY_EMPTY_PHRASE_PATTERN if allow_empty_phrase else _POLY_PHRASE_PATTERN
    instances = []
    for phrase_text in re.findall(pattern, text):
        phrase_text_strip = re.sub(r"^<loc_\d+>", "", phrase_text, count=1)
        if not phrase_text_strip and not allow_empty_phrase:
            continue
        match = re.search(_POLY_PHRASE_TEXT_PATTERN, phrase_text_strip)
        if not match:
            continue
        phrase = match.group().strip()
        if "<poly>" in phrase_text and "</poly>" in phrase_text:
            poly_instances = [
                m.group(1) for m in re.finditer(_POLY_INSTANCE_PATTERN, phrase_text)
            ]
        else:
            poly_instances = [phrase_text]
        for poly_instance in poly_instances:
            poly_matches = list(re.finditer(_POLY_BOX_PATTERN, poly_instance))
            if not poly_matches:
                continue
            polygons = []
            for poly_match in poly_matches:
                poly_bins = [
                    int(m.group(1))
                    for m in re.finditer(r"<loc_(\d+)>", poly_match.group(1))
                ]
                if len(poly_bins) % 2 == 1:
                    poly_bins = poly_bins[:-1]
                polygons.append(_dequantize(poly_bins, image_size))
            instances.append({"cat_name": phrase, "polygons": polygons})
    return instances

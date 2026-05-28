from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.vlm.handler import handle_vlm
from inference_server.handlers.vlm.input_parser import parse_vlm_input
from inference_server.handlers.vlm.introspection import (
    get_vlm_detections_image_only_interface,
    get_vlm_detections_prompt_interface,
    get_vlm_text_image_only_interface,
    get_vlm_text_prompt_interface,
)
from inference_server.handlers.vlm.output_serializer import (
    serialize_vlm_detections,
    serialize_vlm_text,
)


_TEXT_PROMPT = ModelHandlerDescription(
    input_parser=parse_vlm_input,
    handler=handle_vlm,
    output_serializer=serialize_vlm_text,
    interface_provider=get_vlm_text_prompt_interface,
)

_TEXT_IMAGE_ONLY = ModelHandlerDescription(
    input_parser=parse_vlm_input,
    handler=handle_vlm,
    output_serializer=serialize_vlm_text,
    interface_provider=get_vlm_text_image_only_interface,
)

_DETECTIONS_PROMPT = ModelHandlerDescription(
    input_parser=parse_vlm_input,
    handler=handle_vlm,
    output_serializer=serialize_vlm_detections,
    interface_provider=get_vlm_detections_prompt_interface,
)

_DETECTIONS_IMAGE_ONLY = ModelHandlerDescription(
    input_parser=parse_vlm_input,
    handler=handle_vlm,
    output_serializer=serialize_vlm_detections,
    interface_provider=get_vlm_detections_image_only_interface,
)


_DESCRIPTION = _TEXT_PROMPT


_register("vlm", "prompt", _TEXT_PROMPT)
_register("vlm", "query", _TEXT_PROMPT)
_register("vlm", "segment_phrase", _TEXT_PROMPT)
_register("vlm", "ground_phrase", _TEXT_PROMPT)
_register("vlm", "classify_region", _TEXT_PROMPT)
_register("vlm", "caption_region", _TEXT_PROMPT)
_register("vlm", "caption_image_region", _TEXT_PROMPT)

_register("vlm", "caption", _TEXT_IMAGE_ONLY)
_register("vlm", "caption_image", _TEXT_IMAGE_ONLY)
_register("vlm", "ocr", _TEXT_IMAGE_ONLY)
_register("vlm", "ocr_image", _TEXT_IMAGE_ONLY)
_register("vlm", "ocr_region", _TEXT_IMAGE_ONLY)
_register("vlm", "parse_document", _TEXT_IMAGE_ONLY)
_register("vlm", "recognize_text", _TEXT_IMAGE_ONLY)
_register("vlm", "recognize_table", _TEXT_IMAGE_ONLY)
_register("vlm", "recognize_formula", _TEXT_IMAGE_ONLY)

_register("vlm", "detect", _DETECTIONS_IMAGE_ONLY)
_register("vlm", "detect_objects", _DETECTIONS_IMAGE_ONLY)

_register("vlm", "point", _DETECTIONS_PROMPT)

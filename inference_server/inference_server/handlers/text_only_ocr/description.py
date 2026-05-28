from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.text_only_ocr.handler import handle_text_only_ocr
from inference_server.handlers.text_only_ocr.input_parser import (
    parse_text_only_ocr_input,
)
from inference_server.handlers.text_only_ocr.introspection import (
    get_text_only_ocr_interface,
)
from inference_server.handlers.text_only_ocr.output_serializer import (
    serialize_text_only_ocr,
)


_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_text_only_ocr_input,
    handler=handle_text_only_ocr,
    output_serializer=serialize_text_only_ocr,
    interface_provider=get_text_only_ocr_interface,
)


_register("text-only-ocr", "infer", _DESCRIPTION)

from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.structured_ocr.handler import handle_structured_ocr
from inference_server.handlers.structured_ocr.input_parser import (
    parse_structured_ocr_input,
)
from inference_server.handlers.structured_ocr.introspection import (
    get_structured_ocr_interface,
)
from inference_server.handlers.structured_ocr.output_serializer import (
    serialize_structured_ocr,
)

_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_structured_ocr_input,
    handler=handle_structured_ocr,
    output_serializer=serialize_structured_ocr,
    interface_provider=get_structured_ocr_interface,
)


_register("structured-ocr", "infer", _DESCRIPTION)

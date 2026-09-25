from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.object_detection.handler import handle_object_detection
from inference_server.handlers.object_detection.input_parser import (
    parse_object_detection_input,
)
from inference_server.handlers.object_detection.introspection import (
    get_object_detection_interface,
)
from inference_server.handlers.object_detection.output_serializer import (
    serialize_object_detection,
)

_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_object_detection_input,
    handler=handle_object_detection,
    output_serializer=serialize_object_detection,
    interface_provider=get_object_detection_interface,
)


_register("object-detection", "infer", _DESCRIPTION)

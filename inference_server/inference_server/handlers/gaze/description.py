from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.gaze.handler import handle_gaze
from inference_server.handlers.gaze.input_parser import parse_gaze_input
from inference_server.handlers.gaze.introspection import get_gaze_interface
from inference_server.handlers.gaze.output_serializer import serialize_gaze

_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_gaze_input,
    handler=handle_gaze,
    output_serializer=serialize_gaze,
    interface_provider=get_gaze_interface,
)


_register("gaze-detection", "infer", _DESCRIPTION)

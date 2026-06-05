from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.depth.handler import handle_depth
from inference_server.handlers.depth.input_parser import parse_depth_input
from inference_server.handlers.depth.introspection import get_depth_interface
from inference_server.handlers.depth.output_serializer import serialize_depth

_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_depth_input,
    handler=handle_depth,
    output_serializer=serialize_depth,
    interface_provider=get_depth_interface,
)


_register("depth-estimation", "infer", _DESCRIPTION)

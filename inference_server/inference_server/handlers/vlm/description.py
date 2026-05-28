from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.vlm.handler import handle_vlm
from inference_server.handlers.vlm.input_parser import parse_vlm_input
from inference_server.handlers.vlm.introspection import get_vlm_interface
from inference_server.handlers.vlm.output_serializer import serialize_vlm


_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_vlm_input,
    handler=handle_vlm,
    output_serializer=serialize_vlm,
    interface_provider=get_vlm_interface,
)


_register("vlm", "prompt", _DESCRIPTION)

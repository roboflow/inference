from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.passthrough.handler import handle_passthrough
from inference_server.handlers.passthrough.input_parser import parse_passthrough_input
from inference_server.handlers.passthrough.introspection import (
    get_passthrough_interface,
)
from inference_server.handlers.passthrough.output_serializer import (
    serialize_passthrough,
)

_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_passthrough_input,
    handler=handle_passthrough,
    output_serializer=serialize_passthrough,
    interface_provider=get_passthrough_interface,
)


_register("passthrough", "infer", _DESCRIPTION)

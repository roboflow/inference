from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.classification.handler import (
    handle_classification,
)
from inference_server.handlers.classification.input_parser import (
    parse_classification_input,
)
from inference_server.handlers.classification.introspection import (
    get_classification_interface,
)
from inference_server.handlers.classification.output_serializer import (
    serialize_classification,
)


_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_classification_input,
    handler=handle_classification,
    output_serializer=serialize_classification,
    interface_provider=get_classification_interface,
)


_register("classification", "infer", _DESCRIPTION)

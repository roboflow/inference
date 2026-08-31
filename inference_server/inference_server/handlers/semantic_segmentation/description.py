from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.semantic_segmentation.handler import (
    handle_semantic_segmentation,
)
from inference_server.handlers.semantic_segmentation.input_parser import (
    parse_semantic_segmentation_input,
)
from inference_server.handlers.semantic_segmentation.introspection import (
    get_semantic_segmentation_interface,
)
from inference_server.handlers.semantic_segmentation.output_serializer import (
    serialize_semantic_segmentation,
)

_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_semantic_segmentation_input,
    handler=handle_semantic_segmentation,
    output_serializer=serialize_semantic_segmentation,
    interface_provider=get_semantic_segmentation_interface,
)


_register("semantic-segmentation", "infer", _DESCRIPTION)

from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.instance_segmentation.handler import (
    handle_instance_segmentation,
)
from inference_server.handlers.instance_segmentation.input_parser import (
    parse_instance_segmentation_input,
)
from inference_server.handlers.instance_segmentation.introspection import (
    get_instance_segmentation_interface,
)
from inference_server.handlers.instance_segmentation.output_serializer import (
    serialize_instance_segmentation,
)


_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_instance_segmentation_input,
    handler=handle_instance_segmentation,
    output_serializer=serialize_instance_segmentation,
    interface_provider=get_instance_segmentation_interface,
)


_register("instance-segmentation", "infer", _DESCRIPTION)

from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.keypoints.handler import handle_keypoints
from inference_server.handlers.keypoints.input_parser import (
    parse_keypoints_input,
)
from inference_server.handlers.keypoints.introspection import (
    get_keypoints_interface,
)
from inference_server.handlers.keypoints.output_serializer import (
    serialize_keypoints,
)


_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_keypoints_input,
    handler=handle_keypoints,
    output_serializer=serialize_keypoints,
    interface_provider=get_keypoints_interface,
)


_register("keypoint-detection", "infer", _DESCRIPTION)

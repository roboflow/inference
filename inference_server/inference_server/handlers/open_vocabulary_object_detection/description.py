from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.open_vocabulary_object_detection.handler import (
    handle_open_vocabulary_object_detection,
)
from inference_server.handlers.open_vocabulary_object_detection.input_parser import (
    parse_open_vocabulary_object_detection_input,
)
from inference_server.handlers.open_vocabulary_object_detection.introspection import (
    get_open_vocabulary_object_detection_interface,
)
from inference_server.handlers.open_vocabulary_object_detection.output_serializer import (
    serialize_open_vocabulary_object_detection,
)


_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_open_vocabulary_object_detection_input,
    handler=handle_open_vocabulary_object_detection,
    output_serializer=serialize_open_vocabulary_object_detection,
    interface_provider=get_open_vocabulary_object_detection_interface,
)


_register("open-vocabulary-object-detection", "infer", _DESCRIPTION)

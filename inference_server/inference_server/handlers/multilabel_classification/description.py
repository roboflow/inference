from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.multilabel_classification.handler import (
    handle_multilabel_classification,
)
from inference_server.handlers.multilabel_classification.input_parser import (
    parse_multilabel_classification_input,
)
from inference_server.handlers.multilabel_classification.introspection import (
    get_multilabel_classification_interface,
)
from inference_server.handlers.multilabel_classification.output_serializer import (
    serialize_multilabel_classification,
)


_DESCRIPTION = ModelHandlerDescription(
    input_parser=parse_multilabel_classification_input,
    handler=handle_multilabel_classification,
    output_serializer=serialize_multilabel_classification,
    interface_provider=get_multilabel_classification_interface,
)


_register("multi-label-classification", "infer", _DESCRIPTION)

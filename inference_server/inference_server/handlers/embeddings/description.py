from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.embeddings.handler import handle_embeddings
from inference_server.handlers.embeddings.input_parser import parse_embeddings_input
from inference_server.handlers.embeddings.introspection import (
    get_compare_interface,
    get_embed_images_interface,
    get_embed_text_interface,
)
from inference_server.handlers.embeddings.output_serializer import (
    serialize_embeddings_response,
)

_EMBED_IMAGES = ModelHandlerDescription(
    input_parser=parse_embeddings_input,
    handler=handle_embeddings,
    output_serializer=serialize_embeddings_response,
    interface_provider=get_embed_images_interface,
)

_EMBED_TEXT = ModelHandlerDescription(
    input_parser=parse_embeddings_input,
    handler=handle_embeddings,
    output_serializer=serialize_embeddings_response,
    interface_provider=get_embed_text_interface,
)

_COMPARE = ModelHandlerDescription(
    input_parser=parse_embeddings_input,
    handler=handle_embeddings,
    output_serializer=serialize_embeddings_response,
    interface_provider=get_compare_interface,
)


_register("embedding", "embed_images", _EMBED_IMAGES)
_register("embedding", "embed_text", _EMBED_TEXT)
_register("embedding", "compare", _COMPARE)

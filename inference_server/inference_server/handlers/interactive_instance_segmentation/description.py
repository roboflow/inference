from __future__ import annotations

from inference_server.framework.entities import ModelHandlerDescription
from inference_server.framework.registry import _register
from inference_server.handlers.interactive_instance_segmentation.handler import (
    handle_interactive_instance_segmentation,
)
from inference_server.handlers.interactive_instance_segmentation.input_parser import (
    parse_interactive_instance_segmentation_input,
)
from inference_server.handlers.interactive_instance_segmentation.introspection import (
    get_sam3_text_prompts_interface,
    get_sam3_visual_prompts_interface,
    get_sam_embeddings_image_only_interface,
    get_sam_text_image_only_interface,
    get_sam_text_prompt_interface,
)
from inference_server.handlers.interactive_instance_segmentation.output_serializer import (
    serialize_sam_embeddings,
    serialize_sam_segmentation,
    serialize_sam_text,
)

_EMBEDDINGS_IMAGE_ONLY = ModelHandlerDescription(
    input_parser=parse_interactive_instance_segmentation_input,
    handler=handle_interactive_instance_segmentation,
    output_serializer=serialize_sam_embeddings,
    interface_provider=get_sam_embeddings_image_only_interface,
)

_TEXT_IMAGE_ONLY = ModelHandlerDescription(
    input_parser=parse_interactive_instance_segmentation_input,
    handler=handle_interactive_instance_segmentation,
    output_serializer=serialize_sam_text,
    interface_provider=get_sam_text_image_only_interface,
)

_TEXT_PROMPT = ModelHandlerDescription(
    input_parser=parse_interactive_instance_segmentation_input,
    handler=handle_interactive_instance_segmentation,
    output_serializer=serialize_sam_text,
    interface_provider=get_sam_text_prompt_interface,
)


_SAM3_VISUAL = ModelHandlerDescription(
    input_parser=parse_interactive_instance_segmentation_input,
    handler=handle_interactive_instance_segmentation,
    output_serializer=serialize_sam_segmentation,
    interface_provider=get_sam3_visual_prompts_interface,
)

_SAM3_TEXT = ModelHandlerDescription(
    input_parser=parse_interactive_instance_segmentation_input,
    handler=handle_interactive_instance_segmentation,
    output_serializer=serialize_sam_segmentation,
    interface_provider=get_sam3_text_prompts_interface,
)


_register("interactive-instance-segmentation", "embed", _EMBEDDINGS_IMAGE_ONLY)
_register("interactive-instance-segmentation", "embed_images", _EMBEDDINGS_IMAGE_ONLY)
_register("interactive-instance-segmentation", "prompt", _TEXT_PROMPT)
_register("interactive-instance-segmentation", "track", _TEXT_IMAGE_ONLY)
_register(
    "interactive-instance-segmentation",
    "segment_with_visual_prompts",
    _SAM3_VISUAL,
)
_register(
    "interactive-instance-segmentation",
    "segment_with_text_prompts",
    _SAM3_TEXT,
)

from __future__ import annotations

from inference_model_manager.registry_defaults import (
    _P_IMAGES,
    _P_SAM3_VISUAL_PROMPTS,
    _P_SAM_SEGMENT_COMMON,
    _p,
)
from inference_server.framework.entities import ModelInterfaceDescription


def get_sam_embeddings_image_only_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="interactive-instance-segmentation",
        params=_p(_P_IMAGES),
        output_schema={
            "type": "roboflow-embeddings-compact-v1",
        },
    )


def get_sam3_visual_prompts_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="interactive-instance-segmentation",
        params=_p(_P_SAM3_VISUAL_PROMPTS),
        output_schema={
            "type": "roboflow-sam3-segmentation-v1",
        },
    )


def get_sam3_text_prompts_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="interactive-instance-segmentation",
        params={
            "images": {"type": "image", "required": True},
            "prompts": {"type": "list", "required": True},
            "output_prob_thresh": {"type": "float", "required": False, "default": 0.5},
            "mask_format": {"type": "str", "required": False, "default": "rle"},
        },
        output_schema={
            "type": "roboflow-sam3-segmentation-v1",
        },
    )


def get_sam_segment_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="interactive-instance-segmentation",
        params=_p(_P_SAM_SEGMENT_COMMON),
        output_schema={
            "type": "roboflow-sam-segmentation-compact-v1",
        },
    )

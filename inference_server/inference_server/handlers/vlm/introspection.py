from __future__ import annotations

from inference_model_manager.registry_defaults import (
    _P_IMAGES,
    _P_IMAGES_PROMPT,
    _p,
)

from inference_server.framework.entities import ModelInterfaceDescription


def get_vlm_text_prompt_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="vlm",
        params=_p(_P_IMAGES_PROMPT),
        output_schema={
            "type": "roboflow-text-v1",
        },
    )


def get_vlm_text_image_only_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="vlm",
        params=_p(_P_IMAGES),
        output_schema={
            "type": "roboflow-text-v1",
        },
    )


def get_vlm_detections_prompt_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="vlm",
        params=_p(_P_IMAGES_PROMPT),
        output_schema={
            "type": "roboflow-object-detection-compact-v1",
        },
    )


def get_vlm_detections_image_only_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="vlm",
        params=_p(_P_IMAGES),
        output_schema={
            "type": "roboflow-object-detection-compact-v1",
        },
    )


def get_vlm_embeddings_image_only_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="vlm",
        params=_p(_P_IMAGES),
        output_schema={
            "type": "roboflow-embeddings-compact-v1",
        },
    )


get_vlm_interface = get_vlm_text_prompt_interface

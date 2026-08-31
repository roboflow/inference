from __future__ import annotations

from inference_model_manager.registry_defaults import _P_IMAGES, _P_TEXTS, _p
from inference_server.framework.entities import ModelInterfaceDescription


def get_embed_images_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="embedding",
        params=_p(_P_IMAGES),
        output_schema={"type": "roboflow-embeddings-compact-v1"},
    )


def get_embed_text_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="embedding",
        params=_p(_P_TEXTS),
        output_schema={"type": "roboflow-embeddings-compact-v1"},
    )


def get_compare_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="embedding",
        params={
            "images": {"type": "image", "required": False},
            "subject_text": {"type": "str", "required": False},
            "prompt_texts": {"type": "list[str]", "required": False},
        },
        output_schema={"type": "roboflow-comparison-v1"},
    )

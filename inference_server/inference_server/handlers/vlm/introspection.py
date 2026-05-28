from __future__ import annotations

from inference_model_manager.registry_defaults import _P_IMAGES_PROMPT, _p

from inference_server.framework.entities import ModelInterfaceDescription


def get_vlm_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="vlm",
        params=_p(_P_IMAGES_PROMPT),
        output_schema={
            "type": "roboflow-text-v1",
        },
    )

from __future__ import annotations

from inference_model_manager.registry_defaults import _P_IMAGES, _p

from inference_server.framework.entities import ModelInterfaceDescription


def get_depth_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="depth-estimation",
        params=_p(_P_IMAGES),
        output_schema={
            "type": "roboflow-depth-compact-v1",
        },
    )

from __future__ import annotations

from inference_model_manager.registry_defaults import _K_KP, _P_IMAGES, _p
from inference_server.framework.entities import ModelInterfaceDescription


def get_keypoints_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="keypoint-detection",
        params=_p(_P_IMAGES, _K_KP),
        output_schema={
            "type": "roboflow-keypoints-compact-v1",
        },
    )

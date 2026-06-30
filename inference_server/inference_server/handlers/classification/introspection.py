from __future__ import annotations

from inference_model_manager.registry_defaults import _P_IMAGES, _p
from inference_server.framework.entities import ModelInterfaceDescription


def get_classification_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="classification",
        params=_p(_P_IMAGES),
        output_schema={
            "type": "roboflow-classification-compact-v1",
        },
    )

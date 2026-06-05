from __future__ import annotations

from inference_model_manager.registry_defaults import _K_OD, _P_IMAGES_CLASSES, _p
from inference_server.framework.entities import ModelInterfaceDescription


def get_open_vocabulary_object_detection_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="open-vocabulary-object-detection",
        params=_p(_P_IMAGES_CLASSES, _K_OD),
        output_schema={
            "type": "roboflow-object-detection-compact-v1",
        },
    )

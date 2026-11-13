from __future__ import annotations

from inference_model_manager.registry_defaults import (
    _K_PREPROC,
    _P_CONFIDENCE,
    _P_IMAGES,
    _p,
)
from inference_server.framework.entities import ModelInterfaceDescription


def get_multilabel_classification_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="multi-label-classification",
        params=_p(_P_IMAGES, _P_CONFIDENCE, _K_PREPROC),
        output_schema={
            "type": "roboflow-classification-compact-v1",
        },
    )

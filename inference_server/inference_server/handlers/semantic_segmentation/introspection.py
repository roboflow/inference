from __future__ import annotations

from inference_model_manager.registry_defaults import (
    _K_PREPROC,
    _P_CONFIDENCE,
    _P_IMAGES,
    _p,
)
from inference_server.framework.entities import ModelInterfaceDescription


def get_semantic_segmentation_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="semantic-segmentation",
        params=_p(_P_IMAGES, _P_CONFIDENCE, _K_PREPROC),
        output_schema={
            "type": "roboflow-semantic-segmentation-compact-v1",
        },
    )

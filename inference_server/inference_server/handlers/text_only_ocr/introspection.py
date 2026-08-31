from __future__ import annotations

from inference_model_manager.registry_defaults import _P_IMAGES, _p
from inference_server.framework.entities import ModelInterfaceDescription


def get_text_only_ocr_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="text-only-ocr",
        params=_p(_P_IMAGES),
        output_schema={
            "type": "roboflow-text-v1",
        },
    )

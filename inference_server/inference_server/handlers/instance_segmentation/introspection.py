from __future__ import annotations

from inference_model_manager.registry_defaults import _K_ISEG, _P_IMAGES, _p

from inference_server.framework.entities import ModelInterfaceDescription


def get_instance_segmentation_interface() -> ModelInterfaceDescription:
    return ModelInterfaceDescription(
        task="instance-segmentation",
        params=_p(_P_IMAGES, _K_ISEG),
        output_schema={
            "type": "roboflow-instance-segmentation-compact-v1",
        },
    )

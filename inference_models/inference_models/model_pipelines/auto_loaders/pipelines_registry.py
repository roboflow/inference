from typing import Dict, List, Optional, Union

from inference_models.errors import ModelPipelineNotFound
from inference_models.utils.imports import LazyClass

REGISTERED_PIPELINES: Dict[str, LazyClass] = {
    "face-and-gaze-detection": LazyClass(
        module_name="inference_models.model_pipelines.face_and_gaze_detection.mediapipe_l2cs",
        class_name="FaceAndGazeDetectionMPAndL2CS",
    )
}

DEFAULT_PIPELINES_PARAMETERS: Dict[str, List[Union[str, dict]]] = {
    "face-and-gaze-detection": [
        "mediapipe/face-detector",
        "l2cs-net/rn50",
    ]
}


def resolve_pipeline_class(pipline_id: str) -> type:
    if pipline_id not in REGISTERED_PIPELINES:
        raise ModelPipelineNotFound(
            message=f"Could not find model pipeline with id: `{pipline_id}`. "
            f"Registered pipelines: {list(REGISTERED_PIPELINES.keys())}. This error ma be caused by typo "
            f"in the identifier, or pipeline is not registered / not supported in the environment you try to "
            f"run it.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#modelpipelinenotfound",
        )
    return REGISTERED_PIPELINES[pipline_id].resolve()


def get_default_pipeline_parameters(
    pipline_id: str,
) -> Optional[List[Union[str, dict]]]:
    return DEFAULT_PIPELINES_PARAMETERS.get(pipline_id)

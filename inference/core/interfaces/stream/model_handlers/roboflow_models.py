from typing import List

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import ModelConfig
from inference.core.models.roboflow import OnnxRoboflowInferenceModel


def default_process_frame(
    video_frame: VideoFrame,
    model: OnnxRoboflowInferenceModel,
    inference_config: ModelConfig,
) -> dict:
    postprocessing_args = inference_config.to_postprocessing_params()
    predictions = model.infer(
        video_frame.image,
        **postprocessing_args,
    )
    if issubclass(type(predictions), list):
        predictions = predictions[0].dict(
            by_alias=True,
            exclude_none=True,
        )
    return predictions

from functools import partial
from typing import List

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import ModelConfig
from inference.core.interfaces.stream.watchdog import PipelineWatchDog
from inference.core.models.roboflow import OnnxRoboflowInferenceModel


def default_process_frame(
    video_frame: VideoFrame,
    model: OnnxRoboflowInferenceModel,
    inference_config: ModelConfig,
) -> List[dict]:
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


try:
    from inference.models import YOLOWorld

    def get_process_frame_func_yolo_world(model_id: str, api_key: str, **kwargs):
        model = YOLOWorld(model_id=model_id, api_key=api_key)
        model = init_yolo_world_model(model, **kwargs)
        process_frame = partial(process_frame_yolo_world, model=model)
        return model, process_frame

    def init_yolo_world_model(
        model: OnnxRoboflowInferenceModel, classes: List[str], **kwargs
    ) -> OnnxRoboflowInferenceModel:
        model.set_classes(classes)
        return model

    def process_frame_yolo_world(
        video_frame: VideoFrame,
        model: OnnxRoboflowInferenceModel,
        inference_config: ModelConfig,
    ) -> List[dict]:
        postprocessing_args = inference_config.to_postprocessing_params()
        predictions = model.infer(
            video_frame.image,
            **postprocessing_args,
        )

        predictions = predictions.dict(
            by_alias=True,
            exclude_none=True,
        )
        return predictions

except ImportError:
    pass

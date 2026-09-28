from functools import partial
from typing import Any, Callable, List

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import ModelConfig
from inference.models import YOLOWorld


def build_yolo_world_inference_function(
    model_id: str,
    classes: List[str],
    inference_config: ModelConfig,
) -> Callable[[List[VideoFrame]], List[Any]]:
    model = YOLOWorld(model_id=model_id)
    model = init_yolo_world_model(model=model, classes=classes)
    return partial(
        process_frame_yolo_world, model=model, inference_config=inference_config
    )


def init_yolo_world_model(model: YOLOWorld, classes: List[str]) -> YOLOWorld:
    model.set_classes(classes)
    return model


def process_frame_yolo_world(
    video_frames: List[VideoFrame],
    model: YOLOWorld,
    inference_config: ModelConfig,
) -> List[dict]:
    postprocessing_args = inference_config.to_postprocessing_params()
    result = []
    for video_frame in video_frames:
        predictions = model.infer(
            video_frame.image,
            **postprocessing_args,
        ).dict(
            by_alias=True,
            exclude_none=True,
        )
        result.append(predictions)
    return result

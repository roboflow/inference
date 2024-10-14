import os
from threading import Thread
from typing import List, Optional, Union

import cv2
import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import BufferFillingStrategy, BufferConsumptionStrategy
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog, PipelineWatchDog
from inference.core.utils.drawing import create_tiles

WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "vehicle-count-in-drone-video/6",
            "confidence": 0.05,
        },
        {
            "type": "roboflow_core/byte_tracker@v2",
            "name": "byte_tracker",
            "image": "$inputs.image",
            "detections": "$steps.model.predictions",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bbox_visualiser",
            "predictions": "$steps.byte_tracker.tracked_detections",
            "image": "$inputs.image"
        },
        {
            "type": "roboflow_core/trace_visualization@v1",
            "name": "trace_visualization",
            "image": "$steps.bbox_visualiser.image",
            "predictions": "$steps.byte_tracker.tracked_detections",
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "predictions", "selector": "$steps.model.predictions"},
        {"type": "JsonField", "name": "visualization", "selector": "$steps.trace_visualization.image"}
    ],
}

STOP = False
ANNOTATOR = sv.BoundingBoxAnnotator()
fps_monitor = sv.FPSMonitor()


def main() -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=[os.environ["VIDEO_REFERENCE"]],
        workflow_specification=WORKFLOW_DEFINITION,
        watchdog=watchdog,
        on_prediction=workflows_sink,
        source_buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
        source_buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
    )
    control_thread = Thread(target=command_thread, args=(pipeline, watchdog))
    control_thread.start()
    pipeline.start()
    STOP = True
    pipeline.join()


def command_thread(pipeline: InferencePipeline, watchdog: PipelineWatchDog) -> None:
    global STOP
    while not STOP:
        key = input()
        if key == "i":
            print(watchdog.get_report())
        if key == "t":
            pipeline.terminate()
            STOP = True
        elif key == "p":
            pipeline.pause_stream()
        elif key == "m":
            pipeline.mute_stream()
        elif key == "r":
            pipeline.resume_stream()


def workflows_sink(
    predictions: Union[Optional[dict], List[Optional[dict]]],
    video_frames: Union[Optional[VideoFrame], List[Optional[VideoFrame]]],
) -> None:
    fps_monitor.tick()
    if not isinstance(predictions, list):
        predictions = [predictions]
    images_to_show = []
    for prediction in predictions:
        if prediction is None:
            continue
        images_to_show.append(prediction["visualization"].numpy_image)
    tiles = create_tiles(images=images_to_show)
    cv2.imshow(f"Predictions", tiles)
    cv2.waitKey(1)
    if hasattr(fps_monitor, "fps"):
        fps_value = fps_monitor.fps
    else:
        fps_value = fps_monitor()
    print(f"FPS: {fps_value}")


if __name__ == '__main__':
    main()
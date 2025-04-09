import os
from threading import Thread
from typing import List, Optional, Union

import cv2
import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)
from inference.core.utils.drawing import create_tiles

STOP = False

TIME_IN_ZONE_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowVideoMetadata", "name": "video_metadata"},
        {"type": "WorkflowParameter", "name": "zone"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "people_detector",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "confidence": 0.6,
        },
        {
            "type": "roboflow_core/byte_tracker@v1",
            "name": "byte_tracker",
            "detections": "$steps.people_detector.predictions",
            "metadata": "$inputs.video_metadata"
        },
        {
            "type": "roboflow_core/time_in_zone@v1",
            "name": "time_in_zone",
            "detections": f"$steps.byte_tracker.tracked_detections",
            "metadata": "$inputs.video_metadata",
            "zone": "$inputs.zone",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/label_visualization@v1",
            "name": "label_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.time_in_zone.timed_detections",
            "text": "Time In Zone",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bbox_visualization",
            "image": "$steps.label_visualization.image",
            "predictions": "$steps.time_in_zone.timed_detections",
        },
        {
            "type": "roboflow_core/polygon_zone_visualization@v1",
            "name": "zone_visualization",
            "image": "$steps.bbox_visualization.image",
            "zone": "$inputs.zone",
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "label_visualization", "selector": "$steps.zone_visualization.image"},
    ],
}


def main() -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=os.environ["VIDEO_REFERENCE"],
        workflow_specification=TIME_IN_ZONE_WORKFLOW,
        watchdog=watchdog,
        on_prediction=workflows_sink,
        workflows_parameters={
            "zone": [(0, 0), (512, 0), (512, 2000), (0, 2000)],
        }
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
    images_to_show = []
    if not isinstance(predictions, list):
        predictions = [predictions]
        video_frames = [video_frames]
    for prediction, frame in zip(predictions, video_frames):
        if prediction is None or frame is None:
            continue
        images_to_show.append(prediction["label_visualization"].numpy_image)
    tiles = create_tiles(images=images_to_show)
    cv2.imshow(f"Predictions", tiles)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()

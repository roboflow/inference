import os
from threading import Thread
from typing import List, Optional

import cv2
import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.watchdog import PipelineWatchDog, BasePipelineWatchDog
from inference.core.utils.drawing import create_tiles

STOP = False
ANNOTATOR = sv.BoundingBoxAnnotator()
TARGET_PROJECT = os.environ["TARGET_PROJECT"]


def main() -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    workflow_specification = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "step_1",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "confidence": 0.5,
            },
            {
                "type": "RoboflowDatasetUpload",
                "name": "roboflow_dataset_upload",
                "images": "$inputs.image",
                "predictions": "$steps.step_1.predictions",
                "target_project": TARGET_PROJECT,
                "usage_quota_name": "upload_quota_XXX",
                "fire_and_forget": True,
            },
            {
                "type": "RoboflowCustomMetadata",
                "name": "metadata_upload",
                "predictions": "$steps.step_1.predictions",
                "field_name": "dummy",
                "field_value": "dummy",
                "fire_and_forget": True,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "predictions", "selector": "$steps.step_1.predictions"},
            {"type": "JsonField", "name": "upload_error", "selector": "$steps.roboflow_dataset_upload.error_status"},
            {"type": "JsonField", "name": "upload_message", "selector": "$steps.roboflow_dataset_upload.message"},
            {"type": "JsonField", "name": "metadata_error", "selector": "$steps.metadata_upload.error_status"},
            {"type": "JsonField", "name": "metadata_message", "selector": "$steps.metadata_upload.message"},
        ],
    }
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=[os.environ["VIDEO_REFERENCE"]] * 2,
        workflow_specification=workflow_specification,
        watchdog=watchdog,
        on_prediction=workflows_sink,
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
    predictions: List[Optional[dict]],
    video_frames: List[Optional[VideoFrame]],
) -> None:
    images_to_show = []
    for prediction, frame in zip(predictions, video_frames):
        if prediction is None or frame is None:
            continue
        detections: sv.Detections = prediction["predictions"]
        visualised = ANNOTATOR.annotate(frame.image.copy(), detections)
        images_to_show.append(visualised)
        print(prediction["upload_message"], prediction["metadata_message"])
    tiles = create_tiles(images=images_to_show)
    cv2.imshow(f"Predictions", tiles)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()

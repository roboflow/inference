import os
from threading import Thread
from typing import List, Optional

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
        {"type": "WorkflowVideoMetadata", "name": "video_metadata"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "confidence": 0.5,
        },
        {
            "type": "roboflow_core/byte_tracker@v1",
            "name": "byte_tracker",
            "detections": "$steps.model.predictions",
            "metadata": "$inputs.video_metadata"
        },
        {
            "type": "roboflow_core/on_object_appeared@v1",
            "name": "on_instance_appeared",
            "predictions": "$steps.byte_tracker.tracked_detections",
            "video_metadata": "$inputs.video_metadata"
        },
        {
            "type": "roboflow_core/data_aggregator@v1",
            "name": "aggregation",
            "data": {
                "predictions": "$steps.on_instance_appeared.predictions"
            },
            "data_operations": {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            },
            "aggregation_mode": {"predictions": ["values_counts"]},
            "rolling_window": 60,
            "interval": 30,
        },
        {
            "type": "roboflow_core/csv_formatter@v1",
            "name": "csv_formatter",
            "columns_data": {
                "unique_classes": "$steps.aggregation.predictions_values_counts"
            },
            "interval": 2,
        },
        {
            "type": "roboflow_core/local_file_sink@v1",
            "name": "file_sink",
            "content": "$steps.csv_formatter.csv_content",
            "target_directory": ".",
            "file_name_prefix": "unique_classes",
            "file_extension": "csv"
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "predictions", "selector": "$steps.model.predictions"},
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
    predictions: List[Optional[dict]],
    video_frames: List[Optional[VideoFrame]],
) -> None:
    fps_monitor.tick()
    if not isinstance(predictions, list):
        predictions = [predictions]
        video_frames = [video_frames]
    images_to_show = []
    for prediction, frame in zip(predictions, video_frames):
        if prediction is None or frame is None:
            continue
        detections: sv.Detections = prediction["predictions"]
        visualised = ANNOTATOR.annotate(frame.image.copy(), detections)
        images_to_show.append(visualised)
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
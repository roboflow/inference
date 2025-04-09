import os
from typing import Any, Dict, Union

import cv2 as cv
import supervision as sv

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
from inference.core.workflows.core_steps.analytics.path_deviation.v1 import (
    OUTPUT_KEY as PATH_DEVIATION_OUTPUT,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v1 import (
    OUTPUT_KEY as BYTE_TRACKER_OUTPUT_KEY,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = ModelManager(model_registry=model_registry)


WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image", "video_metadata_input_name": "test"},
        {"type": "WorkflowVideoMetadata", "name": "video_metadata"},
        {"type": "InferenceParameter", "name": "reference_path"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "people_detector",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/byte_tracker@v1",
            "name": "byte_tracker",
            "detections": "$steps.people_detector.predictions",
            "metadata": "$inputs.video_metadata"
        },
        {
            "type": "roboflow_core/path_deviation_analytics@v1",
            "name": "path_deviation",
            "detections": f"$steps.byte_tracker.{BYTE_TRACKER_OUTPUT_KEY}",
            "metadata": "$inputs.video_metadata",
            "reference_path": "$inputs.reference_path",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": f"$steps.path_deviation.{PATH_DEVIATION_OUTPUT}",
        }
    ],
}


def custom_sink(prediction: Dict[str, Union[Any, sv.Detections]], video_frame: VideoFrame) -> None:
    cv.imshow("", video_frame)
    cv.waitKey(1)


pipeline = InferencePipeline.init_with_workflow(
    video_reference="/Users/grzegorzklimaszewski/Downloads/ball-track.mp4",
    workflow_specification=WORKFLOW,
    on_prediction=custom_sink,
    workflows_parameters={
        "reference_path": [[1, 2], [3, 4], [5, 6]]
    },
    video_metadata_input_name="video_metadata",
)
pipeline.start()
pipeline.join()

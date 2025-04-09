import os
from threading import Thread
from typing import List, Optional, Union

import cv2
import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)
from inference.core.utils.drawing import create_tiles

WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "line"},
        {"type": "WorkflowParameter", "name": "query_parameter"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "confidence": 0.2,
            "class_filter": ["person"]
        },
        {
            "type": "roboflow_core/webhook_sink@v1",
            "name": "predictions_sink",
            "url": "http://127.0.0.1:9999/data-sink/active-learning",
            "method": "POST",
            "json_payload": {
                "image": "$inputs.image",
                "predictions": "$steps.model.predictions",
            },
            "json_payload_operations": {
                "image": [{"type": "ConvertImageToBase64"}],
                "predictions": [{"type": "DetectionsToDictionary"}]
            },
            "fire_and_forget": True,
        },
        {
            "type": "roboflow_core/webhook_sink@v1",
            "name": "multipart_image_sink_get",
            "url": "http://127.0.0.1:9999/data-sink/multi-part-data",
            "method": "GET",
            "multi_part_encoded_files": {
                "image": "$inputs.image",
            },
            "multi_part_encoded_files_operations": {
                "image": [{"type": "ConvertImageToJPEG"}],
            },
            "form_data": {
                "form_field": "$inputs.query_parameter",
            },
            "fire_and_forget": True,
        },
        {
            "type": "roboflow_core/webhook_sink@v1",
            "name": "multipart_image_sink_post",
            "url": "http://127.0.0.1:9999/data-sink/multi-part-data",
            "method": "POST",
            "multi_part_encoded_files": {
                "image": "$inputs.image",
            },
            "multi_part_encoded_files_operations": {
                "image": [{"type": "ConvertImageToJPEG"}],
            },
            "form_data": {
                "form_field": "$inputs.query_parameter",
            },
            "fire_and_forget": True,
        },
        {
            "type": "roboflow_core/webhook_sink@v1",
            "name": "multipart_image_sink_put",
            "url": "http://127.0.0.1:9999/data-sink/multi-part-data",
            "method": "PUT",
            "multi_part_encoded_files": {
                "image": "$inputs.image",
            },
            "multi_part_encoded_files_operations": {
                "image": [{"type": "ConvertImageToJPEG"}],
            },
            "form_data": {
                "form_field": "$inputs.query_parameter",
            },
            "fire_and_forget": True,
        },
        {
            "type": "roboflow_core/byte_tracker@v3",
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
        },
        {
            "type": "roboflow_core/line_counter@v2",
            "name": "line_counter",
            "image": "$inputs.image",
            "detections": "$steps.byte_tracker.tracked_detections",
            "line_segment": "$inputs.line",
        },
        {
            "type": "roboflow_core/line_counter_visualization@v1",
            "name": "line_counter_visualization",
            "image": "$steps.trace_visualization.image",
            "zone": "$inputs.line",
            "count_in": "$steps.line_counter.count_in",
            "count_out": "$steps.line_counter.count_out",
        },
        {
            "type": "roboflow_core/data_aggregator@v1",
            "name": "data_aggregator",
            "data": {
                "people_passed": "$steps.line_counter.count_out"
            },
            "aggregation_mode": {
                "people_passed": ["values_difference"]
            },
            "interval": 30,
            "interval_unit": "seconds"
        },
        {
            "type": "roboflow_core/webhook_sink@v1",
            "name": "report_webhook_post",
            "url": "http://127.0.0.1:9999/data-sink/json-payload",
            "method": "POST",
            "query_parameters": {
                "static": "static-query-param",
                "list-param": ["a", "b", "c"],
                "bool-param": True,
                "numeric-param": 21.37,
                "dynamic-input": "$inputs.query_parameter",
                "count-out": "$steps.line_counter.count_out"
            },
            "headers": {
                "x-data-origin": "workflows",
                "x-dynamic-header": "$inputs.query_parameter",
            },
            "json_payload": {
                "aggregated_value": "$steps.data_aggregator.people_passed_values_difference",
            },
            "fire_and_forget": True,
        },
        {
            "type": "roboflow_core/webhook_sink@v1",
            "name": "report_webhook_get",
            "url": "http://127.0.0.1:9999/data-sink/json-payload",
            "method": "GET",
            "query_parameters": {
                "static": "static-query-param",
                "list-param": ["a", "b", "c"],
                "bool-param": True,
                "numeric-param": 21.37,
                "dynamic-input": "$inputs.query_parameter",
                "count-out": "$steps.line_counter.count_out"
            },
            "headers": {
                "x-data-origin": "workflows",
                "x-dynamic-header": "$inputs.query_parameter",
            },
            "json_payload": {
                "aggregated_value": "$steps.data_aggregator.people_passed_values_difference",
            },
            "fire_and_forget": True,
        },
        {
            "type": "roboflow_core/webhook_sink@v1",
            "name": "report_webhook_put",
            "url": "http://127.0.0.1:9999/data-sink/json-payload",
            "method": "PUT",
            "query_parameters": {
                "static": "static-query-param",
                "list-param": ["a", "b", "c"],
                "bool-param": True,
                "numeric-param": 21.37,
                "dynamic-input": "$inputs.query_parameter",
                "count-out": "$steps.line_counter.count_out"
            },
            "headers": {
                "x-data-origin": "workflows",
                "x-dynamic-header": "$inputs.query_parameter",
            },
            "json_payload": {
                "aggregated_value": "$steps.data_aggregator.people_passed_values_difference",
            },
            "fire_and_forget": True,
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "visualization", "selector": "$steps.line_counter_visualization.image"}
    ],
}

STOP = False
ANNOTATOR = sv.BoxAnnotator()
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
        workflows_parameters={
            "line": [[100, 900], [1900, 900]],
            "query_parameter": "my-dummy-parameter"
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
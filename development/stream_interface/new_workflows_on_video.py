import argparse

import sys
import os

import cv2

sys.path.append("/Users/ppeczek/Documents/repositories/workflows_blocks/")
os.environ["WORKFLOWS_PLUGINS"] = "pplugin"

from inference import InferencePipeline


def custom_sink(prediction, video_frame) -> None:
    image = prediction["result"][0]
    cv2.imshow("visualisation", image)
    cv2.waitKey(1)


def main(video_file: str) -> None:
    workflow_specification = {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "step_1",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "confidence": 0.4,
                "iou_threshold": 0.3,
            },
            {
                "type": "ObjectsCounting",
                "name": "step_2",
                "predictions": "$steps.step_1.predictions"
            },
            {
                "type": "SimpleVisualisation",
                "name": "step_3",
                "image": "$inputs.image",
                "predictions": "$steps.step_1.predictions",
                "image_metadata": "$steps.step_1.image",
                "total_objects": "$steps.step_2.total_objects",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.step_3.visualised"},
        ]
    }
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=video_file,
        workflow_specification=workflow_specification,
        on_prediction=custom_sink,
    )
    pipeline.start()
    pipeline.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Inference pipeline demo")
    parser.add_argument(
        "--video_file",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    main(video_file=args.video_file)

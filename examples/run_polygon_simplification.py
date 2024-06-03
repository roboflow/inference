import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import supervision as sv

# should be handled by upper layer
os.environ["WORKFLOWS_PLUGINS"] = "workflows_enterprise_blocks"

from workflows_enterprise_blocks.contour_reducer import (
    CONTOUR_REDUCER_OUTPUT_KEY,
    CONTOUR_REDUCER_TYPE,
)

from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.models.utils import ROBOFLOW_MODEL_TYPES

YOUR_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "InferenceParameter", "name": "reduce_class_name"},
        {"type": "InferenceParameter", "name": "required_number_of_vertices"},
        {"type": "InferenceParameter", "name": "model_id"},
        {"type": "InferenceParameter", "name": "confidence"},
        {"type": "InferenceParameter", "name": "iou_threshold"},
        {"type": "InferenceParameter", "name": "class_filter"},
    ],
    "steps": [
        {
            "type": "InstanceSegmentationModel",
            "name": "instance_segmentation",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
            "iou_threshold": "$inputs.iou_threshold",
            "class_filter": "$inputs.class_filter",
        },
        {
            "type": CONTOUR_REDUCER_TYPE,
            "name": "contour_reducer",
            "predictions": "$steps.instance_segmentation.predictions",
            "reduce_class_name": "$inputs.reduce_class_name",
            "required_number_of_vertices": "$inputs.required_number_of_vertices",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": CONTOUR_REDUCER_OUTPUT_KEY,
            "selector": f"$steps.contour_reducer.{CONTOUR_REDUCER_OUTPUT_KEY}",
        },
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.instance_segmentation.predictions",
        },
    ],
}


model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = ModelManager(model_registry=model_registry)

execution_engine = ExecutionEngine.init(
    workflow_definition=YOUR_WORKFLOW,
    init_parameters={
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": os.getenv("ROBOFLOW_API_KEY"),
    },
)


class FileMustExist(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values.strip() or not Path(values.strip()).exists():
            raise argparse.ArgumentError(argument=self, message="Incorrect path")
        setattr(namespace, self.dest, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contour Reducer demo")
    parser.add_argument("--source-path", required=True, type=str, action=FileMustExist)
    parser.add_argument(
        "--reduce-class-name", required=False, type=str, default="person"
    )
    parser.add_argument(
        "--required-number-of-vertices", required=False, type=int, default=4
    )
    parser.add_argument(
        "--model-id", required=False, type=str, default="yolov8n-seg-640"
    )
    parser.add_argument("--confidence", required=False, type=float, default=0.6)
    parser.add_argument("--iou-threshold", required=False, type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = execution_engine.run(
        runtime_parameters={
            "image": {
                "type": "file",
                "value": args.source_path,
            },
            "reduce_class_name": args.reduce_class_name,
            "required_number_of_vertices": args.required_number_of_vertices,
            "model_id": args.model_id,
            "confidence": args.confidence,
            "iou_threshold": args.iou_threshold,
            "class_filter": [args.reduce_class_name],
        }
    )

    if not result.get(CONTOUR_REDUCER_OUTPUT_KEY):
        print("Nothing detected, script terminated.")
        exit(0)

    img = cv.imread(args.source_path)

    contour_annotator = sv.PolygonAnnotator(thickness=10)
    original_contours = contour_annotator.annotate(
        scene=img.copy(), detections=result.get("predictions")[0]
    )
    cv.imshow("Original contour", original_contours)
    cv.waitKey(delay=0)

    cv.drawContours(
        img,
        [np.array(result.get(CONTOUR_REDUCER_OUTPUT_KEY)[0][0], dtype=int)],
        -1,
        (0, 255, 0),
        10,
    )
    cv.imshow("Reduced contour", img)
    cv.waitKey(delay=0)
    cv.destroyAllWindows()

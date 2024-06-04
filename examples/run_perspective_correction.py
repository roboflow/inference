import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import supervision as sv

from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.models.utils import ROBOFLOW_MODEL_TYPES

from inference.core.workflows.core_steps.transformations.perspective_correction import (
    OUTPUT_KEY as PERSPECTIVE_CORRECTION_OUTPUT_KEY,
    TYPE as PERSPECTIVE_CORRECTION_TYPE,
)
from inference.core.workflows.core_steps.transformations.polygon_simplification import (
    OUTPUT_KEY as POLYGON_SIMPLIFICATION_OUTPUT_KEY,
    TYPE as POLYGON_SIMPLIFICATION_TYPE,
)

YOUR_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "InferenceParameter", "name": "simplify_class_name"},
        {"type": "InferenceParameter", "name": "required_number_of_vertices"},
        {"type": "InferenceParameter", "name": "model_id"},
        {"type": "InferenceParameter", "name": "confidence"},
        {"type": "InferenceParameter", "name": "iou_threshold"},
        {"type": "InferenceParameter", "name": "transformed_rect_width"},
        {"type": "InferenceParameter", "name": "transformed_rect_height"},
        {
            "type": "InferenceParameter",
            "name": "extend_perspective_polygon_by_detections_anchor",
        },
    ],
    "steps": [
        {
            "type": "InstanceSegmentationModel",
            "name": "instance_segmentation",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
            "iou_threshold": "$inputs.iou_threshold",
        },
        {
            "type": POLYGON_SIMPLIFICATION_TYPE,
            "name": "contour_reducer",
            "predictions": "$steps.instance_segmentation.predictions",
            "simplify_class_name": "$inputs.simplify_class_name",
            "required_number_of_vertices": "$inputs.required_number_of_vertices",
        },
        {
            "type": PERSPECTIVE_CORRECTION_TYPE,
            "name": "coordinates_transformer",
            "predictions": "$steps.instance_segmentation.predictions",
            "perspective_polygons": f"$steps.contour_reducer.{POLYGON_SIMPLIFICATION_OUTPUT_KEY}",
            "transformed_rect_width": "$inputs.transformed_rect_width",
            "transformed_rect_height": "$inputs.transformed_rect_height",
            "extend_perspective_polygon_by_detections_anchor": "$inputs.extend_perspective_polygon_by_detections_anchor",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": POLYGON_SIMPLIFICATION_OUTPUT_KEY,
            "selector": f"$steps.contour_reducer.{POLYGON_SIMPLIFICATION_OUTPUT_KEY}",
        },
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.instance_segmentation.predictions",
        },
        {
            "type": "JsonField",
            "name": PERSPECTIVE_CORRECTION_OUTPUT_KEY,
            "selector": f"$steps.coordinates_transformer.{PERSPECTIVE_CORRECTION_OUTPUT_KEY}",
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


class MustBeValidSVPosition(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values and values not in set(sv.Position.list()):
            raise argparse.ArgumentError(
                argument=self,
                message=f"When set, must be one of: {', '.join(sv.Position.list())}",
            )
        setattr(namespace, self.dest, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contour Reducer demo")
    parser.add_argument("--source-path", required=True, type=str, action=FileMustExist)
    parser.add_argument(
        "--simplify-class-name", required=False, type=str, default="person"
    )
    parser.add_argument(
        "--required-number-of-vertices", required=False, type=int, default=4
    )
    parser.add_argument(
        "--model-id", required=False, type=str, default="yolov8n-seg-640"
    )
    parser.add_argument("--confidence", required=False, type=float, default=0.6)
    parser.add_argument("--iou-threshold", required=False, type=float, default=0.3)
    parser.add_argument(
        "--transformed-rect-width", required=False, type=int, default=1000
    )
    parser.add_argument(
        "--transformed-rect-height", required=False, type=int, default=1000
    )
    parser.add_argument(
        "--extend-perspective-polygon-by-detections-anchor",
        required=False,
        type=str,
        default="",
        action=MustBeValidSVPosition,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = execution_engine.run(
        runtime_parameters={
            "image": {
                "type": "file",
                "value": args.source_path,
            },
            "simplify_class_name": args.simplify_class_name,
            "required_number_of_vertices": args.required_number_of_vertices,
            "model_id": args.model_id,
            "confidence": args.confidence,
            "iou_threshold": args.iou_threshold,
            "transformed_rect_width": args.transformed_rect_width,
            "transformed_rect_height": args.transformed_rect_height,
            "extend_perspective_polygon_by_detections_anchor": args.extend_perspective_polygon_by_detections_anchor,
        }
    )

    if not result.get(POLYGON_SIMPLIFICATION_OUTPUT_KEY):
        print("Nothing detected, script terminated.")
        exit(0)

    img = cv.imread(args.source_path)

    polygon_annotator = sv.PolygonAnnotator(thickness=10)
    detections = result.get("predictions")[0]
    original_polygons = polygon_annotator.annotate(
        scene=img.copy(),
        detections=detections[detections["class_name"] != args.simplify_class_name],
    )
    cv.drawContours(
        original_polygons,
        [np.array(result.get(POLYGON_SIMPLIFICATION_OUTPUT_KEY)[0][0], dtype=int)],
        -1,
        (0, 255, 0),
        10,
    )
    cv.imshow("Scene before perspective correction", original_polygons)
    cv.waitKey(delay=0)

    blank_image = np.ones(
        (args.transformed_rect_width, args.transformed_rect_height, 3), np.uint8
    )
    h, w, _ = blank_image.shape
    corrected_detections = result.get(PERSPECTIVE_CORRECTION_OUTPUT_KEY)[0]
    annotated_frame = polygon_annotator.annotate(
        blank_image,
        detections=corrected_detections[corrected_detections["class_name"] != args.simplify_class_name],
    )
    cv.imshow("Scene after perspective correction", annotated_frame)
    cv.waitKey(0)

    cv.destroyAllWindows()

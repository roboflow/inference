"""Perspective-correction demo, and the reference for what a direct Python
caller has to bind.

The Execution Engine takes the server's capabilities as explicit
`workflows_core.*` init parameters instead of reaching for them itself, and its
standalone defaults refuse or no-op whatever is missing. A script that calls
`ExecutionEngine.init` therefore installs the same services the HTTP, stream and
CLI roots install:

* `ModelManagerModelsProvider(model_manager)` - blocks call the provider port
  (`run_object_detection`, `run_instance_segmentation`, ...), which a raw
  `ModelManager` does not implement. Passing the manager itself now fails.
* `install_workflows_platform_bindings` - Roboflow-managed VLM and notification
  proxy calls, workflows referenced by ID, workspace identity for authenticated
  Modal execution, and the shared cache behind sink cooldown/dedup. Unbound, the
  offline platform client raises and the cache is process-local.
* `bind_image_codec` - `{"type": "file"}` / `{"type": "url"}` and serialized
  numpy inputs, plus the later re-load of a stored image reference; the default
  codec refuses them. Call it AFTER merging overrides, so input deserialization
  and reference loading share one codec and one SSRF/local-file policy.
* `UsageTrackingExecutionObserver()` - workflow and custom-block usage plus the
  run's tracing spans; the default observer records nothing.
* `resolve_step_error_handler()` - the server's error classification; the
  engine's own default is the mapping-free legacy handler.
* `server_workflows_configuration()` - the object the rest of the process uses,
  so a mis-wire is reported instead of silently diverging.

`workflows_core.api_key` still authenticates the calls a block makes, but an API
key alone no longer supplies any of the above.

The usage categories are separate scopes, not substitutes: `request` (one HTTP
handler call), `workflows` (one observed engine run, with workflow identity, FPS
and preview), `workflow_block` (one custom-Python block execution) and `model`
(the existing model-level accounting). Server entry points keep their existing
layers; a direct caller gets the workflow and block scopes only from the
observer above. Leaving the observer out on purpose stays supported - execution
still runs, without workflow/custom-block collection.

An offline platform client is not "no network": blocks calling a third party
with the user's own provider key, and the REMOTE branches that go through
`inference_sdk`, keep working on their own terms.
"""

import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import supervision as sv

from inference.core.interfaces.roboflow_platform_client import (
    install_workflows_platform_bindings,
)
from inference.core.interfaces.workflows_configuration import (
    server_workflows_configuration,
)
from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.core.interfaces.workflows_image_codec import bind_image_codec
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.interfaces.workflows_step_error_handlers import (
    resolve_step_error_handler,
)
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.transformations.dynamic_zones.v1 import (
    OUTPUT_KEY as DYNAMIC_ZONES_OUTPUT_KEY,
)
from inference.core.workflows.core_steps.transformations.dynamic_zones.v1 import (
    TYPE as DYNAMIC_ZONES_TYPE,
)
from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    OUTPUT_DETECTIONS_KEY as PERSPECTIVE_CORRECTION_OUTPUT_DETECTIONS_KEY,
)
from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    OUTPUT_IMAGE_KEY as PERSPECTIVE_CORRECTION_OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    TYPE as PERSPECTIVE_CORRECTION_TYPE,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.models.utils import ROBOFLOW_MODEL_TYPES


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
        "--zones-from-class-name", required=False, type=str, default="person"
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
    parser.add_argument("--warp-image", required=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    WORKFLOW_DEFINITION = {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {"type": "InferenceParameter", "name": "zones_from_class_name"},
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
            {"type": "InferenceParameter", "name": "warp_image"},
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
                "type": "DetectionsFilter",
                "name": "zone_class_filter",
                "predictions": "$steps.instance_segmentation.predictions",
                "operations": [
                    {
                        "type": "DetectionsFilter",
                        "filter_operation": {
                            "type": "StatementGroup",
                            "operator": "and",
                            "statements": [
                                {
                                    "type": "BinaryStatement",
                                    "negate": False,
                                    "left_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "_",
                                        "operations": [
                                            {
                                                "type": "ExtractDetectionProperty",
                                                "property_name": "class_name",
                                            }
                                        ],
                                    },
                                    "comparator": {"type": "in (Sequence)"},
                                    "right_operand": {
                                        "type": "StaticOperand",
                                        "value": [
                                            args.zones_from_class_name  # in order to pass this as input selector we need to refactor block to pass-through the inputs
                                        ],
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "type": DYNAMIC_ZONES_TYPE,
                "name": "dynamic_zones",
                "predictions": "$steps.zone_class_filter.predictions",
                "required_number_of_vertices": "$inputs.required_number_of_vertices",
            },
            {
                "type": "DetectionsFilter",
                "name": "not_zone_class_filter",
                "predictions": "$steps.instance_segmentation.predictions",
                "operations": [
                    {
                        "type": "DetectionsFilter",
                        "filter_operation": {
                            "type": "StatementGroup",
                            "operator": "and",
                            "statements": [
                                {
                                    "type": "BinaryStatement",
                                    "negate": True,
                                    "left_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "_",
                                        "operations": [
                                            {
                                                "type": "ExtractDetectionProperty",
                                                "property_name": "class_name",
                                            }
                                        ],
                                    },
                                    "comparator": {"type": "in (Sequence)"},
                                    "right_operand": {
                                        "type": "StaticOperand",
                                        "value": [
                                            args.zones_from_class_name  # in order to pass this as input selector we need to refactor block to pass-through the inputs
                                        ],
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "type": PERSPECTIVE_CORRECTION_TYPE,
                "name": "coordinates_transformer",
                "predictions": "$steps.not_zone_class_filter.predictions",
                "images": "$inputs.image",
                "perspective_polygons": f"$steps.dynamic_zones.{DYNAMIC_ZONES_OUTPUT_KEY}",
                "transformed_rect_width": "$inputs.transformed_rect_width",
                "transformed_rect_height": "$inputs.transformed_rect_height",
                "extend_perspective_polygon_by_detections_anchor": "$inputs.extend_perspective_polygon_by_detections_anchor",
                "warp_image": "$inputs.warp_image",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": DYNAMIC_ZONES_OUTPUT_KEY,
                "selector": f"$steps.dynamic_zones.{DYNAMIC_ZONES_OUTPUT_KEY}",
            },
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.instance_segmentation.predictions",
            },
            {
                "type": "JsonField",
                "name": PERSPECTIVE_CORRECTION_OUTPUT_DETECTIONS_KEY,
                "selector": f"$steps.coordinates_transformer.{PERSPECTIVE_CORRECTION_OUTPUT_DETECTIONS_KEY}",
            },
            {
                "type": "JsonField",
                "name": PERSPECTIVE_CORRECTION_OUTPUT_IMAGE_KEY,
                "selector": f"$steps.coordinates_transformer.{PERSPECTIVE_CORRECTION_OUTPUT_IMAGE_KEY}",
            },
        ],
    }

    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)

    init_parameters = {
        "workflows_core.model_manager": ModelManagerModelsProvider(model_manager),
        "workflows_core.api_key": os.getenv("ROBOFLOW_API_KEY"),
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        "workflows_core.execution_observer": UsageTrackingExecutionObserver(),
    }
    install_workflows_platform_bindings(init_parameters)
    bind_image_codec(init_parameters)
    init_parameters.setdefault(
        "workflows_core.configuration", server_workflows_configuration()
    )

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION,
        init_parameters=init_parameters,
        step_error_handler=resolve_step_error_handler(),
    )

    result = execution_engine.run(
        runtime_parameters={
            "image": {
                "type": "file",
                "value": args.source_path,
            },
            "zones_from_class_name": args.zones_from_class_name,
            "required_number_of_vertices": args.required_number_of_vertices,
            "model_id": args.model_id,
            "confidence": args.confidence,
            "iou_threshold": args.iou_threshold,
            "transformed_rect_width": args.transformed_rect_width,
            "transformed_rect_height": args.transformed_rect_height,
            "extend_perspective_polygon_by_detections_anchor": args.extend_perspective_polygon_by_detections_anchor,
            "warp_image": args.warp_image,
        }
    )

    if not result[0].get(DYNAMIC_ZONES_OUTPUT_KEY):
        print("Nothing detected, script terminated.")
        exit(0)

    img = cv.imread(args.source_path)

    polygon_annotator = sv.PolygonAnnotator(thickness=10)
    detections = result[0].get("predictions")
    original_polygons = polygon_annotator.annotate(
        scene=img.copy(),
        detections=detections[detections["class_name"] != args.zones_from_class_name],
    )
    cv.drawContours(
        original_polygons,
        [np.array(result[0].get(DYNAMIC_ZONES_OUTPUT_KEY), dtype=int)],
        -1,
        (0, 0, 255),
        10,
    )
    cv.imshow("Scene before perspective correction", original_polygons)
    cv.waitKey(delay=0)

    blank_image = np.ones(
        (args.transformed_rect_width, args.transformed_rect_height, 3), np.uint8
    )
    corrected_detections = result[0].get(PERSPECTIVE_CORRECTION_OUTPUT_DETECTIONS_KEY)
    if args.warp_image:
        blank_image = result[0].get(PERSPECTIVE_CORRECTION_OUTPUT_IMAGE_KEY)
    annotated_frame = polygon_annotator.annotate(
        blank_image,
        detections=corrected_detections[
            corrected_detections["class_name"] != args.zones_from_class_name
        ],
    )
    cv.imshow("Scene after perspective correction", annotated_frame)
    cv.waitKey(0)

    cv.destroyAllWindows()

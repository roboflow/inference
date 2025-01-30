import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.transformations.dynamic_zones.v1 import (
    OUTPUT_KEY as DYNAMIC_ZONES_OUTPUT_KEY,
)
from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    OUTPUT_DETECTIONS_KEY as PERSPECTIVE_CORRECTION_OUTPUT_DETECTIONS_KEY,
)
from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    OUTPUT_IMAGE_KEY as PERSPECTIVE_CORRECTION_OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_DYNAMIC_ZONE_AND_PERSPECTIVE_CONVERTER = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_instance_segmentation_model@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "yolov8n-seg-640",
        },
        {
            "type": "roboflow_core/detections_filter@v1",
            "name": "detections_filter",
            "predictions": "$steps.model.predictions",
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
                                    "value": ["banana"],
                                },
                            }
                        ],
                    },
                }
            ],
            "operations_parameters": {},
        },
        {
            "type": "roboflow_core/dynamic_zone@v1",
            "name": "dynamic_zone",
            "predictions": "$steps.detections_filter.predictions",
            "required_number_of_vertices": 4,
        },
        {
            "type": "roboflow_core/perspective_correction@v1",
            "name": "perspective_correction",
            "images": "$inputs.image",
            "perspective_polygons": f"$steps.dynamic_zone.{DYNAMIC_ZONES_OUTPUT_KEY}",
            "predictions": "$steps.model.predictions",
            "warp_image": True,
            "extend_perspective_polygon_by_detections_anchor": "BOTTOM_CENTER",
        },
        {
            "type": "roboflow_core/polygon_visualization@v1",
            "name": "perspective_visualization",
            "image": f"$steps.perspective_correction.{PERSPECTIVE_CORRECTION_OUTPUT_IMAGE_KEY}",
            "predictions": f"$steps.perspective_correction.{PERSPECTIVE_CORRECTION_OUTPUT_DETECTIONS_KEY}",
        },
        {
            "type": "roboflow_core/polygon_visualization@v1",
            "name": "polygon_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "polygons_visualization",
            "coordinates_system": "own",
            "selector": "$steps.polygon_visualization.image",
        },
        {
            "type": "JsonField",
            "name": "perspective_visualization",
            "coordinates_system": "own",
            "selector": "$steps.perspective_visualization.image",
        },
        {
            "type": "JsonField",
            "name": "perspective_correction_outputs",
            "coordinates_system": "own",
            "selector": "$steps.perspective_correction.*",
        },
        {
            "type": "JsonField",
            "name": "dynamic_zones",
            "coordinates_system": "own",
            "selector": "$steps.dynamic_zone.zones",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow with dynamic zone and perspective converter",
    use_case_description="""
In this example dynamic zone with 4 vertices is calculated from detected segmentations.
Perspective correction is applied to the input image as well as to detected segmentations based on this zone.
    """,
    workflow_definition=WORKFLOW_DYNAMIC_ZONE_AND_PERSPECTIVE_CONVERTER,
    workflow_name_in_app="dynamic_zone_and_perspective_converter",
)
def test_workflow_with_classical_pattern_matching(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DYNAMIC_ZONE_AND_PERSPECTIVE_CONVERTER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One set of images provided, so one output expected"
    assert set(result[0].keys()) == {
        "dynamic_zones",
        "polygons_visualization",
        "perspective_correction_outputs",
        "perspective_visualization",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["perspective_correction_outputs"]["corrected_coordinates"]) == 6
    ), "Expected 6 detections in corrected coordinates"

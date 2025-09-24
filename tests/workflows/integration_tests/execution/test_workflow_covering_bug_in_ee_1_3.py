import numpy as np
from matplotlib import pyplot as plt

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_DEFINITION_DETECTIONS_FILTER = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
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
                                "left_operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "_",
                                    "operations": [
                                        {
                                            "type": "ExtractDetectionProperty",
                                            "property_name": "size",
                                        }
                                    ],
                                },
                                "comparator": {"type": "(Number) >="},
                                "right_operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "image",
                                    "operations": [
                                        {
                                            "type": "ExtractImageProperty",
                                            "property_name": "size",
                                        },
                                        {"type": "Multiply", "other": 0.025},
                                    ],
                                },
                            }
                        ],
                    },
                }
            ],
            "operations_parameters": {"image": "$inputs.image"},
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.predictions",
        },
        {
            "type": "JsonField",
            "name": "detections_filter",
            "coordinates_system": "own",
            "selector": "$steps.detections_filter.predictions",
        },
    ],
}


def test_workflow_with_detections_filter_referencing_image(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION_DETECTIONS_FILTER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
        "detections_filter",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["model_predictions"]) == 12
    assert len(result[0]["detections_filter"]) == 1


WORKFLOW_DEFINITION_PERSPECTIVE_CORRECTION = {
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
                                    "value": ["car"],
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
            "perspective_polygons": "$steps.dynamic_zone.zones",
            "predictions": "$steps.model.predictions",
            "warp_image": True,
        },
        {
            "type": "roboflow_core/label_visualization@v1",
            "name": "label_visualization",
            "image": "$steps.perspective_correction.warped_image",
            "predictions": "$steps.perspective_correction.corrected_coordinates",
            "copy_image": False,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "label_visualization",
            "coordinates_system": "own",
            "selector": "$steps.label_visualization.image",
        }
    ],
}


def test_workflow_with_perspective_correction(
    model_manager: ModelManager,
    car_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION_PERSPECTIVE_CORRECTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [car_image, car_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "label_visualization",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "label_visualization",
    }, "Expected all declared outputs to be delivered"
    assert isinstance(result[0]["label_visualization"].numpy_image, np.ndarray)
    assert isinstance(result[1]["label_visualization"].numpy_image, np.ndarray)

from unittest import mock
from uuid import UUID

import numpy as np
import pytest

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    USE_INFERENCE_MODELS,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import (
    DynamicBlockCodeError,
    DynamicBlockError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.v1.dynamic_blocks import block_assembler
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

FUNCTION_TO_GET_OVERLAP_OF_BBOXES = """
def run(self, predictions: sv.Detections, class_x: str, class_y: str) -> BlockResult:
    bboxes_class_x = predictions[predictions.data["class_name"] == class_x]
    bboxes_class_y = predictions[predictions.data["class_name"] == class_y]
    overlap = []
    for bbox_x in bboxes_class_x:
        bbox_x_coords = bbox_x[0]
        bbox_overlaps = []
        for bbox_y in bboxes_class_y:
            if bbox_y[-1]["detection_id"] == bbox_x[-1]["detection_id"]:
                continue
            bbox_y_coords = bbox_y[0]
            x_min = max(bbox_x_coords[0], bbox_y_coords[0])
            y_min = max(bbox_x_coords[1], bbox_y_coords[1])
            x_max = min(bbox_x_coords[2], bbox_y_coords[2])
            y_max = min(bbox_x_coords[3], bbox_y_coords[3])
            # compute the area of intersection rectangle
            intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
            box_x_area = (bbox_x_coords[2] - bbox_x_coords[0] + 1) * (bbox_x_coords[3] - bbox_x_coords[1] + 1)
            local_overlap = intersection_area / (box_x_area + 1e-5)
            bbox_overlaps.append(local_overlap)
        overlap.append(bbox_overlaps)
    return  {"overlap": overlap}
"""


FUNCTION_TO_GET_MAXIMUM_OVERLAP = """
def run(self, overlaps: List[List[float]]) -> BlockResult:
    max_value = -1
    for overlap in overlaps:
        for overlap_value in overlap:
            if not max_value:
                max_value = overlap_value
            else:
                max_value = max(max_value, overlap_value)
    return {"max_value": max_value}
"""

WORKFLOW_WITH_OVERLAP_MEASUREMENT = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "OverlapMeasurement",
                "inputs": {
                    "predictions": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                    },
                    "class_x": {
                        "type": "DynamicInputDefinition",
                        "value_types": ["string"],
                    },
                    "class_y": {
                        "type": "DynamicInputDefinition",
                        "value_types": ["string"],
                    },
                },
                "outputs": {"overlap": {"type": "DynamicOutputDefinition", "kind": []}},
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_TO_GET_OVERLAP_OF_BBOXES,
            },
        },
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "MaximumOverlap",
                "inputs": {
                    "overlaps": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                    },
                },
                "outputs": {
                    "max_value": {"type": "DynamicOutputDefinition", "kind": []}
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_TO_GET_MAXIMUM_OVERLAP,
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "OverlapMeasurement",
            "name": "overlap_measurement",
            "predictions": "$steps.model.predictions",
            "class_x": "dog",
            "class_y": "dog",
        },
        {
            "type": "ContinueIf",
            "name": "continue_if",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "overlaps",
                            "operations": [{"type": "SequenceLength"}],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": 1,
                        },
                    }
                ],
            },
            "evaluation_parameters": {"overlaps": "$steps.overlap_measurement.overlap"},
            "next_steps": ["$steps.maximum_overlap"],
        },
        {
            "type": "MaximumOverlap",
            "name": "maximum_overlap",
            "overlaps": "$steps.overlap_measurement.overlap",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "overlaps",
            "selector": "$steps.overlap_measurement.overlap",
        },
        {
            "type": "JsonField",
            "name": "max_overlap",
            "selector": "$steps.maximum_overlap.max_value",
        },
    ],
}


FUNCTION_TO_RETURN_WORKFLOW_CONTEXT = """
def run(self, value: int) -> BlockResult:
    context = self.get_workflow_context()
    context["value"] = value
    return context
"""


WORKFLOW_WITH_CUSTOM_BLOCK_WORKFLOW_CONTEXT = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "value"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "ContextProbe",
                "inputs": {
                    "value": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_parameter", "step_output"],
                    },
                },
                "outputs": {
                    "step_name": {"type": "DynamicOutputDefinition", "kind": []},
                    "step_selector": {"type": "DynamicOutputDefinition", "kind": []},
                    "workflow_execution_id": {
                        "type": "DynamicOutputDefinition",
                        "kind": [],
                    },
                    "block_type": {"type": "DynamicOutputDefinition", "kind": []},
                    "value": {"type": "DynamicOutputDefinition", "kind": []},
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_TO_RETURN_WORKFLOW_CONTEXT,
            },
        },
    ],
    "steps": [
        {
            "type": "ContextProbe",
            "name": "context_probe",
            "value": "$inputs.value",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "step_name",
            "selector": "$steps.context_probe.step_name",
        },
        {
            "type": "JsonField",
            "name": "step_selector",
            "selector": "$steps.context_probe.step_selector",
        },
        {
            "type": "JsonField",
            "name": "workflow_execution_id",
            "selector": "$steps.context_probe.workflow_execution_id",
        },
        {
            "type": "JsonField",
            "name": "block_type",
            "selector": "$steps.context_probe.block_type",
        },
        {
            "type": "JsonField",
            "name": "value",
            "selector": "$steps.context_probe.value",
        },
    ],
}


def test_workflow_with_custom_python_block_exposes_workflow_context() -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CUSTOM_BLOCK_WORKFLOW_CONTEXT,
        init_parameters={"workflows_core.api_key": None},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"value": 37})

    # then
    assert len(result) == 1
    assert UUID(result[0].pop("workflow_execution_id"))
    assert result == [
        {
            "step_name": "context_probe",
            "step_selector": "$steps.context_probe",
            "block_type": "ContextProbe",
            "value": 37,
        }
    ]


@add_to_workflows_gallery(
    category="Workflows with dynamic Python Blocks",
    use_case_title="Workflow measuring bounding boxes overlap",
    use_case_description="""
In real world use-cases you may not be able to find all pieces of functionalities required to complete 
your workflow within existing blocks. 

In such cases you may create piece of python code and put it in workflow as a dynamic block. Specifically 
here, we define two dynamic blocks:

- `OverlapMeasurement` which will accept object detection predictions and provide for boxes 
of specific class matrix of overlap with all boxes of another class.

- `MaximumOverlap` that will take overlap matrix produced by `OverlapMeasurement` and calculate maximum overlap.

Dynamic block may be used to create steps, exactly as if those blocks were standard Workflow blocks 
existing in ecosystem. The workflow presented in the example predicts from object detection model and 
calculates overlap matrix. Later, only if more than one object is detected, maximum overlap is calculated.
    """,
    workflow_definition=WORKFLOW_WITH_OVERLAP_MEASUREMENT,
)
def test_workflow_with_custom_python_blocks_measuring_overlap(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_OVERLAP_MEASUREMENT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "overlaps",
        "max_overlap",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "overlaps",
        "max_overlap",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["overlaps"]) == 2
    ), "Expected 2 instances of dogs found, each overlap with another for first image"
    if not USE_INFERENCE_MODELS:
        assert (
            abs(result[0]["max_overlap"] - 0.177946) < 1e-3
        ), "Expected max overlap to be calculated properly"
    else:
        assert (
            abs(result[0]["max_overlap"] - 0.16448630730758626) < 1e-3
        ), "Expected max overlap to be calculated properly"
    assert (
        len(result[1]["overlaps"]) == 0
    ), "Expected no instances of dogs found for second image"
    assert (
        result[1]["max_overlap"] is None
    ), "Expected `max_overlap` not to be calculated for second image due to conditional execution"


FUNCTION_TO_GET_MAXIMUM_CONFIDENCE_FROM_BATCH_OF_DETECTIONS = """
def run(self, predictions: Batch[sv.Detections]) -> BlockResult:
    result = []
    for prediction in predictions:
        result.append({"max_confidence": np.max(prediction.confidence).item()})
    return result
"""

WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_ON_BATCH = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "MaxConfidence",
                "inputs": {
                    "predictions": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                    },
                },
                "outputs": {
                    "max_confidence": {
                        "type": "DynamicOutputDefinition",
                        "kind": ["float_zero_to_one"],
                    }
                },
                "accepts_batch_input": True,
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_TO_GET_MAXIMUM_CONFIDENCE_FROM_BATCH_OF_DETECTIONS,
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "MaxConfidence",
            "name": "confidence_aggregation",
            "predictions": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "max_confidence",
            "selector": "$steps.confidence_aggregation.max_confidence",
        },
    ],
}


def test_workflow_with_custom_python_block_operating_on_batch(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_ON_BATCH,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "max_confidence",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "max_confidence",
    }, "Expected all declared outputs to be delivered"
    assert (
        abs(result[0]["max_confidence"] - 0.85599) < 1.5e-2
    ), "Expected max confidence to be extracted"
    assert (
        abs(result[1]["max_confidence"] - 0.84284) < 1.5e-2
    ), "Expected max confidence to be extracted"


FUNCTION_TO_ASSOCIATE_DETECTIONS_FOR_CROPS = """
def my_function(self, prediction: sv.Detections, crops: Batch[WorkflowImageData]) -> BlockResult:
    detection_id2bbox = {
        detection_id.item(): i for i, detection_id in enumerate(prediction.data["detection_id"])
    }
    results = []
    for crop in crops:
        parent_id = crop.parent_metadata.parent_id
        results.append({"associated_detections": prediction[detection_id2bbox[parent_id]]})
    return results
"""


WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_CROSS_DIMENSIONS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "DetectionsToCropsAssociation",
                "inputs": {
                    "prediction": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                        "selector_data_kind": {
                            "step_output": [
                                "object_detection_prediction",
                                "instance_segmentation_prediction",
                                "keypoint_detection_prediction",
                            ]
                        },
                    },
                    "crops": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output_image"],
                        "is_dimensionality_reference": True,
                        "dimensionality_offset": 1,
                    },
                },
                "outputs": {
                    "associated_detections": {
                        "type": "DynamicOutputDefinition",
                        "kind": [
                            "object_detection_prediction",
                            "instance_segmentation_prediction",
                            "keypoint_detection_prediction",
                        ],
                    }
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_TO_ASSOCIATE_DETECTIONS_FOR_CROPS,
                "run_function_name": "my_function",
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "Crop",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.model.predictions",
        },
        {
            "type": "DetectionsToCropsAssociation",
            "name": "detections_associations",
            "prediction": "$steps.model.predictions",
            "crops": "$steps.crop.crops",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "associated_detections",
            "selector": "$steps.detections_associations.associated_detections",
        },
    ],
}


def _class_names_of_prediction(prediction) -> list:
    """Representation-agnostic class-name extraction: raw engine outputs are
    sv.Detections flag-off and native inference_models Detections flag-on (the
    declared-kind OUT boundary converts them by design)."""
    import supervision as sv

    if isinstance(prediction, sv.Detections):
        return prediction["class_name"].tolist()
    per_box = prediction.bboxes_metadata or [{} for _ in range(len(prediction))]
    mapping = (prediction.image_metadata or {}).get("class_names", {})
    return [
        str(box.get("class", mapping.get(int(class_id))))
        for box, class_id in zip(per_box, prediction.class_id.detach().cpu().tolist())
    ]


def test_workflow_with_custom_python_block_operating_cross_dimensions(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_CROSS_DIMENSIONS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "associated_detections",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "associated_detections",
    }, "Expected all declared outputs to be delivered"
    assert len(result[1]["associated_detections"]) == 12
    class_names_first_image_crops = [
        _class_names_of_prediction(e) for e in result[0]["associated_detections"]
    ]
    for class_names in class_names_first_image_crops:
        assert len(class_names) == 1, "Expected single bbox to be associated"
    assert len(class_names_first_image_crops) == 2, "Expected 2 crops for first image"
    class_names_second_image_crops = [
        _class_names_of_prediction(e) for e in result[1]["associated_detections"]
    ]
    for class_names in class_names_second_image_crops:
        assert len(class_names) == 1, "Expected single bbox to be associated"
    assert (
        len(class_names_second_image_crops) == 12
    ), "Expected 12 crops for second image"


@mock.patch.object(block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False)
def test_workflow_with_custom_python_block_when_custom_python_execution_forbidden(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_CROSS_DIMENSIONS,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


FUNCTION_TO_MERGE_CROPS_INTO_TILES = """
def run(self, crops: Optional[Batch[Optional[WorkflowImageData]]]) -> BlockResult:
    from inference.core.utils.drawing import create_tiles
    if crops is None:
        return {"tiles": None}
    black_image = np.zeros((192, 168, 3), dtype=np.uint8)
    images = [crop.numpy_image if crop is not None else black_image for crop in crops]
    return {"tiles": create_tiles(images)}
"""


WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_DIMENSIONALITY_REDUCTION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "DimensionalityReduction",
                "inputs": {
                    "crops": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output_image"],
                    },
                },
                "outputs": {"tiles": {"type": "DynamicOutputDefinition", "kind": []}},
                "output_dimensionality_offset": -1,
                "accepts_empty_values": True,
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_TO_MERGE_CROPS_INTO_TILES,
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["person"],
        },
        {
            "type": "Crop",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.model.predictions",
        },
        {
            "type": "DimensionalityReduction",
            "name": "tile_creation",
            "crops": "$steps.crop.crops",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "tiles",
            "selector": "$steps.tile_creation.tiles",
        },
    ],
}


def test_workflow_with_custom_python_block_reducing_dimensionality(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_DIMENSIONALITY_REDUCTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "tiles",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "tiles",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["tiles"] is None, "Expected no crops - hence empty output"
    assert isinstance(result[1]["tiles"], np.ndarray), "Expected np array with tile"


MODEL_INIT_FUNCTION = """
def init_model() -> Dict[str, Any]:
    model = YOLOv8ObjectDetection(model_id="yolov8n-640")
    return {"model": model}
"""

MODEL_INFER_FUNCTION = """
def infer(self, image: WorkflowImageData) -> BlockResult:
    predictions = self._init_results["model"].infer(image.numpy_image)
    return {"predictions": sv.Detections.from_inference(predictions[0].model_dump(by_alias=True, exclude_none=True))}
"""

WORKFLOW_WITH_PYTHON_BLOCK_HOSTING_MODEL = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "CustomModel",
                "inputs": {
                    "image": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_image"],
                    },
                },
                "outputs": {
                    "predictions": {
                        "type": "DynamicOutputDefinition",
                        "kind": [
                            "object_detection_prediction",
                        ],
                    }
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": MODEL_INFER_FUNCTION,
                "run_function_name": "infer",
                "init_function_code": MODEL_INIT_FUNCTION,
                "init_function_name": "init_model",
                "imports": [
                    "from inference.models.yolov8 import YOLOv8ObjectDetection",
                ],
            },
        },
    ],
    "steps": [
        {
            "type": "CustomModel",
            "name": "model",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.model.predictions",
        },
    ],
}


def test_workflow_with_custom_python_block_running_custom_model(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PYTHON_BLOCK_HOSTING_MODEL,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert np.allclose(
        result[0]["predictions"].confidence,
        [0.85599, 0.50392],
        atol=1e-3,
    ), "Expected reproducible predictions for first image"
    assert np.allclose(
        result[1]["predictions"].confidence,
        [
            0.84284,
            0.83957,
            0.81555,
            0.80455,
            0.75804,
            0.75794,
            0.71715,
            0.71408,
            0.71003,
            0.56938,
            0.54092,
            0.43511,
        ],
        atol=1e-3,
    ), "Expected reproducible predictions for second image"


BROKEN_RUN_FUNCTION = """
def run(some: InvalidType):
    pass
"""


WORKFLOW_WITH_CODE_THAT_DOES_NOT_COMPILE = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "CustomModel",
                "inputs": {
                    "image": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_image"],
                    },
                },
                "outputs": {
                    "predictions": {
                        "type": "DynamicOutputDefinition",
                        "kind": [],
                    }
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": BROKEN_RUN_FUNCTION,
            },
        },
    ],
    "steps": [
        {
            "type": "CustomModel",
            "name": "model",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.model.predictions",
        },
    ],
}


def test_workflow_with_custom_python_block_when_code_cannot_be_compiled(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(DynamicBlockCodeError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_CODE_THAT_DOES_NOT_COMPILE,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


WORKFLOW_WITHOUT_RUN_FUNCTION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "CustomModel",
                "inputs": {
                    "image": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_image"],
                    },
                },
                "outputs": {
                    "predictions": {
                        "type": "DynamicOutputDefinition",
                        "kind": [],
                    }
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": "",
            },
        },
    ],
    "steps": [
        {
            "type": "CustomModel",
            "name": "model",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.model.predictions",
        },
    ],
}


def test_workflow_with_custom_python_block_when_code_does_not_define_declared_run_function(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(DynamicBlockError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITHOUT_RUN_FUNCTION,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


WORKFLOW_WITHOUT_DECLARED_INIT_FUNCTION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "CustomModel",
                "inputs": {
                    "image": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_image"],
                    },
                },
                "outputs": {
                    "predictions": {
                        "type": "DynamicOutputDefinition",
                        "kind": [],
                    }
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": MODEL_INFER_FUNCTION,
                "run_function_name": "infer",
                "init_function_code": "",
                "init_function_name": "init_model",
                "imports": [
                    "from inference.models.yolov8 import YOLOv8ObjectDetection",
                ],
            },
        },
    ],
    "steps": [
        {
            "type": "CustomModel",
            "name": "model",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.model.predictions",
        },
    ],
}


def test_workflow_with_custom_python_block_when_code_does_not_define_declared_init_function(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(DynamicBlockError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITHOUT_DECLARED_INIT_FUNCTION,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


FUNCTION_TO_FILTER_AND_PROBE = """
def run(self, predictions: sv.Detections) -> BlockResult:
    got_sv = isinstance(predictions, sv.Detections)
    filtered = predictions[predictions.confidence > 0.6]
    _ = filtered.xyxy.copy()  # numpy-style access, the documented legacy contract
    return {"filtered": filtered, "got_sv": got_sv}
"""

WORKFLOW_WITH_BOTH_BOUNDARIES_AND_TENSOR_CONSUMER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "FilterProbe",
                "inputs": {
                    "predictions": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                        "selector_data_kind": {
                            "step_output": ["object_detection_prediction"]
                        },
                    },
                },
                "outputs": {
                    "filtered": {
                        "type": "DynamicOutputDefinition",
                        "kind": ["object_detection_prediction"],
                    },
                    "got_sv": {"type": "DynamicOutputDefinition", "kind": []},
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_TO_FILTER_AND_PROBE,
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "FilterProbe",
            "name": "filter_probe",
            "predictions": "$steps.model.predictions",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "count",
            "data": "$steps.filter_probe.filtered",
            "operations": [{"type": "SequenceLength"}],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "filtered",
            "selector": "$steps.filter_probe.filtered",
        },
        {"type": "JsonField", "name": "count", "selector": "$steps.count.output"},
        {
            "type": "JsonField",
            "name": "got_sv",
            "selector": "$steps.filter_probe.got_sv",
        },
    ],
}


def test_workflow_with_custom_python_block_between_model_and_tensor_consumer(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Both boundaries in one workflow: model producer -> legacy custom block
    (default knob; must see sv.Detections and use the numpy contract) ->
    downstream UQL consumer (native ops under the flag) -> serialized output.
    The hardcoded expectations hold in BOTH flag directions, which is the
    integration-level parity statement (byte-parity is proven at the unit
    layer's round-trip tests)."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BOTH_BOUNDARIES_AND_TENSOR_CONSUMER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [dogs_image]},
        serialize_results=True,
    )

    # then
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["got_sv"] is True, "User code must receive sv.Detections"
    assert result[0]["count"] == 1, "Expected exactly one detection above 0.6"
    serialized = result[0]["filtered"]["predictions"]
    assert len(serialized) == 1
    assert serialized[0]["class"] == "dog"
    assert abs(serialized[0]["confidence"] - 0.856) < 0.1
    assert {
        "x",
        "y",
        "width",
        "height",
        "confidence",
        "class",
        "class_id",
        "detection_id",
    } <= set(serialized[0].keys())


FUNCTION_RETURNING_BARE_TENSOR = """
def run(self, predictions) -> BlockResult:
    import torch
    return {"weird": torch.zeros(3)}
"""

WORKFLOW_WITH_BARE_TENSOR_WILDCARD_OUTPUT = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "BareTensorBlock",
                "inputs": {
                    "predictions": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                    },
                },
                "outputs": {"weird": {"type": "DynamicOutputDefinition", "kind": []}},
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_RETURNING_BARE_TENSOR,
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "BareTensorBlock",
            "name": "bad_block",
            "predictions": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "weird", "selector": "$steps.bad_block.weird"},
    ],
}


def test_workflow_with_custom_python_block_returning_bare_tensor_via_wildcard(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """OUT-direction wildcard semantics: a bare torch.Tensor returned through a
    wildcard output PASSES THROUGH in both flag directions — on the OUT side a
    tensor is a native-representation value (the `tensor` kind's runtime type),
    so the boundary forwards it untouched. The loud-error contract for bare
    tensors applies to the IN direction (legacy user code cannot operate on
    them) and is covered by the representation-boundary unit tests."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BARE_TENSOR_WILDCARD_OUTPUT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": [dogs_image]})

    # then
    import torch

    assert isinstance(result[0]["weird"], torch.Tensor)
    assert result[0]["weird"].shape == (3,)


_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason=(
        "tensor_native dynamic blocks compile only under "
        "ENABLE_TENSOR_DATA_REPRESENTATION (the flag-off compile-time error is "
        "unit-tested in test_block_assembler.py)"
    ),
)

FUNCTION_MINTING_NATIVE_DETECTIONS = """
def run(self, image) -> BlockResult:
    detections = Detections(
        xyxy=torch.tensor(
            [[10.0, 10.0, 60.0, 60.0]], device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        class_id=torch.tensor([0], device=WORKFLOWS_IMAGE_TENSOR_DEVICE),
        confidence=torch.tensor([0.9], device=WORKFLOWS_IMAGE_TENSOR_DEVICE),
    )
    detections = attach_native_detection_metadata(
        detections=detections,
        image=image,
        class_names={0: "widget"},
        prediction_type="object-detection",
    )
    return {"predictions": detections}
"""

WORKFLOW_WITH_TENSOR_NATIVE_MINTING_BLOCK = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "NativeMinter",
                "tensor_compatibility": "tensor_native",
                "inputs": {
                    "image": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_image"],
                    },
                },
                "outputs": {
                    "predictions": {
                        "type": "DynamicOutputDefinition",
                        "kind": ["object_detection_prediction"],
                    },
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_MINTING_NATIVE_DETECTIONS,
            },
        },
    ],
    "steps": [
        {
            "type": "NativeMinter",
            "name": "minter",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.minter.predictions",
        },
    ],
}


@_TENSOR_ONLY
def test_workflow_with_tensor_native_custom_block_minting_native_detections(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """tensor_native authoring surface: the extended IMPORTS_LINES must resolve in
    the generated module namespace (torch, Detections,
    WORKFLOWS_IMAGE_TENSOR_DEVICE, attach_native_detection_metadata), and the
    minted output must satisfy the tensor serializer's hard requirements
    (class_names map + per-box detection_id) end-to-end."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_TENSOR_NATIVE_MINTING_BLOCK,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [dogs_image]},
        serialize_results=True,
    )

    # then
    assert isinstance(result, list) and len(result) == 1
    serialized = result[0]["predictions"]["predictions"]
    assert len(serialized) == 1
    assert serialized[0]["class"] == "widget"
    assert serialized[0]["class_id"] == 0
    assert abs(serialized[0]["confidence"] - 0.9) < 1e-6
    assert serialized[0]["x"] == 35.0 and serialized[0]["y"] == 35.0
    assert serialized[0]["width"] == 50.0 and serialized[0]["height"] == 50.0
    assert len(serialized[0]["detection_id"]) > 0


FUNCTION_PROBING_NATIVE_PASSTHROUGH = """
def run(self, predictions) -> BlockResult:
    got_native = isinstance(predictions, Detections)
    xyxy_is_tensor = isinstance(predictions.xyxy, torch.Tensor)
    return {
        "got_native": got_native,
        "xyxy_is_tensor": xyxy_is_tensor,
        "passthrough": predictions,
    }
"""

WORKFLOW_WITH_TENSOR_NATIVE_PROBING_BLOCK = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "NativeProbe",
                "tensor_compatibility": "tensor_native",
                "inputs": {
                    "predictions": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                        "selector_data_kind": {
                            "step_output": ["object_detection_prediction"]
                        },
                    },
                },
                "outputs": {
                    "got_native": {"type": "DynamicOutputDefinition", "kind": []},
                    "xyxy_is_tensor": {"type": "DynamicOutputDefinition", "kind": []},
                    "passthrough": {
                        "type": "DynamicOutputDefinition",
                        "kind": ["object_detection_prediction"],
                    },
                },
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_PROBING_NATIVE_PASSTHROUGH,
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "NativeProbe",
            "name": "probe",
            "predictions": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "got_native",
            "selector": "$steps.probe.got_native",
        },
        {
            "type": "JsonField",
            "name": "xyxy_is_tensor",
            "selector": "$steps.probe.xyxy_is_tensor",
        },
        {
            "type": "JsonField",
            "name": "passthrough",
            "selector": "$steps.probe.passthrough",
        },
    ],
}


@_TENSOR_ONLY
def test_workflow_with_tensor_native_custom_block_receiving_native_predictions(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """tensor_native pass-through: user code receives the NATIVE prediction from a
    model step (no boundary conversion in either direction) and can return it
    for downstream serialization."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_TENSOR_NATIVE_PROBING_BLOCK,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [dogs_image]},
        serialize_results=True,
    )

    # then
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["got_native"] is True, "User code must receive native Detections"
    assert result[0]["xyxy_is_tensor"] is True, "Native xyxy must stay a torch.Tensor"
    serialized = result[0]["passthrough"]["predictions"]
    assert len(serialized) >= 1
    assert {"x", "y", "width", "height", "confidence", "class", "detection_id"} <= set(
        serialized[0].keys()
    )

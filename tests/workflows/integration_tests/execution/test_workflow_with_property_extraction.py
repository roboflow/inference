import datetime
import os

import cv2 as cv
import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.interfaces.camera.video_source import VideoSource
from inference.core.interfaces.stream.entities import VideoFrame
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import StepOutputLineageError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_EXTRACTION_OF_CLASSES_FOR_DETECTIONS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "PropertyDefinition",
            "name": "property_extraction",
            "data": "$steps.general_detection.predictions",
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
            ],
        },
        {
            "type": "PropertyDefinition",
            "name": "instances_counter",
            "data": "$steps.general_detection.predictions",
            "operations": [{"type": "SequenceLength"}],
        },
        {
            "type": "Expression",
            "name": "expression",
            "data": {
                "class_names": "$steps.property_extraction.output",
                "reference": "$inputs.reference",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [
                    {
                        "type": "CaseDefinition",
                        "condition": {
                            "type": "StatementGroup",
                            "statements": [
                                {
                                    "type": "BinaryStatement",
                                    "left_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "class_names",
                                    },
                                    "comparator": {"type": "=="},
                                    "right_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "reference",
                                    },
                                }
                            ],
                        },
                        "result": {"type": "StaticCaseResult", "value": "PASS"},
                    }
                ],
                "default": {"type": "StaticCaseResult", "value": "FAIL"},
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detected_classes",
            "selector": "$steps.property_extraction.output",
        },
        {
            "type": "JsonField",
            "name": "number_of_detected_boxes",
            "selector": "$steps.instances_counter.output",
        },
        {
            "type": "JsonField",
            "name": "verdict",
            "selector": "$steps.expression.output",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with business logic",
    use_case_title="Workflow with extraction of classes for detections (1)",
    use_case_description="""
In practical use-cases you may find the need to inject pieces of business logic inside 
your Workflow, such that it is easier to integrate with app created in Workflows ecosystem.

Translation of model predictions into domain-specific language of your business is possible 
with specialised blocks that let you parametrise such programming constructs 
as switch-case statements.

In this example, our goal is to:

- tell how many objects are detected

- verify that the picture presents exactly two dogs

To achieve that goal, we run generic object detection model as first step, then we use special
block called Property Definition that is capable of executing various operations to
transform input data into desired output. We have two such blocks:

- `instances_counter` which takes object detection predictions and apply operation to extract sequence length - 
effectively calculating number of instances of objects that were predicted

- `property_extraction` which extracts class names from all detected bounding boxes

`instances_counter` basically completes first goal of the workflow, but to satisfy the second one we need to 
build evaluation logic that will tell "PASS" / "FAIL" based on comparison of extracted class names with 
reference parameter (provided via Workflow input `$inputs.reference`). We can use Expression block to achieve 
that goal - building custom case statements (checking if class names being list of classes 
extracted from object detection prediction matches reference passed in the input). 
    """,
    workflow_definition=WORKFLOW_WITH_EXTRACTION_OF_CLASSES_FOR_DETECTIONS,
    workflow_name_in_app="business-logic-1",
)
def test_workflow_with_extraction_of_classes_for_detections(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
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
        workflow_definition=WORKFLOW_WITH_EXTRACTION_OF_CLASSES_FOR_DETECTIONS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
            "reference": ["dog", "dog"],
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "detected_classes",
        "verdict",
        "number_of_detected_boxes",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "detected_classes",
        "verdict",
        "number_of_detected_boxes",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["detected_classes"] == [
        "dog",
        "dog",
    ], "Expected two instances of dogs found in first image"
    assert (
        result[0]["verdict"] == "PASS"
    ), "Expected first image to match expected classes"
    assert (
        result[0]["number_of_detected_boxes"] == 2
    ), "Expected 2 dogs detected in first image"
    assert (
        result[1]["detected_classes"] == ["person"] * 12
    ), "Expected 12 instances of person found in second image"
    assert (
        result[1]["verdict"] == "FAIL"
    ), "Expected second image not to match expected classes"
    assert (
        result[1]["number_of_detected_boxes"] == 12
    ), "Expected 12 people detected in second image"


WORKFLOW_WITH_EXTRACTION_OF_CLASS_NAME_FROM_CROPS_AND_CONCATENATION_OF_RESULTS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$steps.cropping.crops",
            "model_id": "dog-breed-xpaq6/1",
            "confidence": 0.09,
        },
        {
            "type": "PropertyDefinition",
            "name": "property_extraction",
            "data": "$steps.breds_classification.predictions",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
        },
        {
            "type": "DimensionCollapse",
            "name": "outputs_concatenation",
            "data": "$steps.property_extraction.output",
        },
        {
            "type": "FirstNonEmptyOrDefault",
            "name": "empty_values_replacement",
            "data": ["$steps.outputs_concatenation.output"],
            "default": [],
        },
        {
            "type": "Expression",
            "name": "expression",
            "data": {
                "detected_classes": "$steps.empty_values_replacement.output",
                "reference": "$inputs.reference",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [
                    {
                        "type": "CaseDefinition",
                        "condition": {
                            "type": "StatementGroup",
                            "statements": [
                                {
                                    "type": "BinaryStatement",
                                    "left_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "detected_classes",
                                    },
                                    "comparator": {"type": "=="},
                                    "right_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "reference",
                                    },
                                }
                            ],
                        },
                        "result": {"type": "StaticCaseResult", "value": "PASS"},
                    }
                ],
                "default": {"type": "StaticCaseResult", "value": "FAIL"},
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detected_classes",
            "selector": "$steps.property_extraction.output",
        },
        {
            "type": "JsonField",
            "name": "wrapped_classes",
            "selector": "$steps.empty_values_replacement.output",
        },
        {
            "type": "JsonField",
            "name": "verdict",
            "selector": "$steps.expression.output",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with business logic",
    use_case_title="Workflow with extraction of classes for detections (2)",
    use_case_description="""
In practical use-cases you may find the need to inject pieces of business logic inside 
your Workflow, such that it is easier to integrate with app created in Workflows ecosystem.

Translation of model predictions into domain-specific language of your business is possible 
with specialised blocks that let you parametrise such programming constructs 
as switch-case statements.

In this example, our goal is to:

- run generic object detection model to find instances of dogs

- crop dogs detection

- run specialised dogs breed classifier to assign granular label for each dog

- compare predicted dogs breeds to verify if detected labels matches exactly reverence value passed in input.

This example is quite complex as it requires quite deep understanding of Workflows ecosystem. Let's start from
the beginning - we run object detection model, crop its detections according to dogs class to perform 
classification. This is quite typical for workflows (you may find such pattern in remaining examples). 

The complexity increases when we try to handle classification output. We need to have a list of classes
for each input image, but for now we have complex objects with all classification predictions details
provided by `breds_classification` step - what is more - we have batch of such predictions for
each input image (as we created dogs crops based on object detection model predictions). To solve the 
problem, at first we apply Property Definition step taking classifier predictions and turning them into
strings representing predicted classes. We still have batch of class names at dimensionality level 2, 
which needs to be brought into dimensionality level 1 to make a single comparison against reference 
value for each input image. To achieve that effect we use Dimension Collapse block which does nothing
else but grabs the batch of classes and turns it into list of classes at dimensionality level 1 - one 
list for each input image.

That would solve our problems, apart from one nuance that must be taken into account. First-stage model
is not guaranteed to detect any dogs - and if that happens we do not execute cropping and further 
processing for that image, leaving all outputs derived from downstream computations `None` which is
suboptimal. To compensate for that, we may use First Non Empty Or Default block which will take 
`outputs_concatenation` step output and replace missing values with empty list (as effectively this is 
equivalent of not detecting any dog).

Such prepared output of `empty_values_replacement` step may be now plugged into Expression block, 
performing switch-case like logic to deduce if breeds of detected dogs match with reference value 
passed to workflow execution. 
    """,
    workflow_definition=WORKFLOW_WITH_EXTRACTION_OF_CLASS_NAME_FROM_CROPS_AND_CONCATENATION_OF_RESULTS,
    workflow_name_in_app="business-logic-2",
)
def test_workflow_with_extraction_of_classes_for_classification_on_crops(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
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
        workflow_definition=WORKFLOW_WITH_EXTRACTION_OF_CLASS_NAME_FROM_CROPS_AND_CONCATENATION_OF_RESULTS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
            "reference": [
                "116.Parson_russell_terrier",
                "131.Wirehaired_pointing_griffon",
            ],
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "detected_classes",
        "wrapped_classes",
        "verdict",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "detected_classes",
        "wrapped_classes",
        "verdict",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["detected_classes"] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected two instances of dogs found in first image"
    assert result[0]["wrapped_classes"] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected two instances of dogs found in first image"
    assert (
        result[0]["verdict"] == "PASS"
    ), "Expected first image to match expected classes"
    assert (
        result[1]["detected_classes"] == []
    ), "Expected no instances of dogs found in second image"
    assert (
        result[1]["wrapped_classes"] == []
    ), "Expected no instances of dogs found in second image"
    assert (
        result[1]["verdict"] == "FAIL"
    ), "Expected second image not to match expected classes"


WORKFLOW_PERFORMING_OCR_AND_AGGREGATION_TO_PERFORM_PASS_FAIL_FOR_ALL_PLATES_FOUND_IN_IMAGE_AT_ONCE = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "plates_detection",
            "image": "$inputs.image",
            "model_id": "vehicle-registration-plates-trudk/2",
        },
        {
            "type": "DetectionOffset",
            "name": "plates_offset",
            "predictions": "$steps.plates_detection.predictions",
            "offset_width": 50,
            "offset_height": 50,
        },
        {
            "type": "Crop",
            "name": "plates_crops",
            "image": "$inputs.image",
            "predictions": "$steps.plates_offset.predictions",
        },
        {
            "type": "OCRModel",
            "name": "ocr",
            "image": "$steps.plates_crops.crops",
        },
        {
            "type": "DimensionCollapse",
            "name": "outputs_concatenation",
            "data": "$steps.ocr.result",
        },
        {
            "type": "FirstNonEmptyOrDefault",
            "name": "empty_values_replacement",
            "data": ["$steps.outputs_concatenation.output"],
            "default": [],
        },
        {
            "type": "Expression",
            "name": "expression",
            "data": {
                "outputs_concatenation": "$steps.empty_values_replacement.output",
                "reference": "$inputs.reference",
            },
            "data_operations": {
                "outputs_concatenation": [{"type": "SequenceLength"}],
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [
                    {
                        "type": "CaseDefinition",
                        "condition": {
                            "type": "StatementGroup",
                            "statements": [
                                {
                                    "type": "BinaryStatement",
                                    "left_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "outputs_concatenation",
                                    },
                                    "comparator": {"type": "=="},
                                    "right_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "reference",
                                    },
                                }
                            ],
                        },
                        "result": {"type": "StaticCaseResult", "value": "PASS"},
                    }
                ],
                "default": {"type": "StaticCaseResult", "value": "FAIL"},
            },
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "plates_ocr", "selector": "$steps.ocr.result"},
        {
            "type": "JsonField",
            "name": "concatenated_ocr",
            "selector": "$steps.empty_values_replacement.output",
        },
        {
            "type": "JsonField",
            "name": "verdict",
            "selector": "$steps.expression.output",
        },
    ],
}


def test_workflow_with_aggregation_of_ocr_results_globally_for_image(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PERFORMING_OCR_AND_AGGREGATION_TO_PERFORM_PASS_FAIL_FOR_ALL_PLATES_FOUND_IN_IMAGE_AT_ONCE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": license_plate_image,
            "reference": 2,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 elements in the output for one input image"
    assert set(result[0].keys()) == {
        "plates_ocr",
        "concatenated_ocr",
        "verdict",
    }, "Expected all declared outputs to be delivered"
    assert (
        isinstance(result[0]["plates_ocr"], list) and len(result[0]["plates_ocr"]) == 2
    ), "Expected 2 plates to be found"
    # In this case, output does not reveal the concatenation result, but we could not make expression without concat
    assert (
        isinstance(result[0]["concatenated_ocr"], list)
        and len(result[0]["concatenated_ocr"]) == 2
    ), "Expected 2 plates to be found"
    assert result[0]["verdict"] == "PASS", "Expected to meet the condition"


WORKFLOW_PERFORMING_OCR_AND_AGGREGATION_TO_PERFORM_PASS_FAIL_FOR_EACH_PLATE_SEPARATELY = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "plates_detection",
            "image": "$inputs.image",
            "model_id": "vehicle-registration-plates-trudk/2",
        },
        {
            "type": "DetectionOffset",
            "name": "plates_offset",
            "predictions": "$steps.plates_detection.predictions",
            "offset_width": 50,
            "offset_height": 50,
        },
        {
            "type": "Crop",
            "name": "plates_crops",
            "image": "$inputs.image",
            "predictions": "$steps.plates_offset.predictions",
        },
        {
            "type": "OCRModel",
            "name": "ocr",
            "image": "$steps.plates_crops.crops",
        },
        {
            "type": "Expression",
            "name": "expression",
            "data": {
                "outputs_concatenation": "$steps.ocr.result",
                "reference": "$inputs.reference",
            },
            "data_operations": {
                "outputs_concatenation": [{"type": "SequenceLength"}],
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [
                    {
                        "type": "CaseDefinition",
                        "condition": {
                            "type": "StatementGroup",
                            "statements": [
                                {
                                    "type": "BinaryStatement",
                                    "left_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "outputs_concatenation",
                                    },
                                    "comparator": {"type": "(Number) >"},
                                    "right_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "reference",
                                    },
                                }
                            ],
                        },
                        "result": {"type": "StaticCaseResult", "value": "PASS"},
                    }
                ],
                "default": {"type": "StaticCaseResult", "value": "FAIL"},
            },
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "plates_ocr", "selector": "$steps.ocr.result"},
        {
            "type": "JsonField",
            "name": "verdict",
            "selector": "$steps.expression.output",
        },
    ],
}


def test_workflow_with_pass_fail_applied_for_each_ocr_result(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PERFORMING_OCR_AND_AGGREGATION_TO_PERFORM_PASS_FAIL_FOR_EACH_PLATE_SEPARATELY,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, dogs_image],
            "reference": 0,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "plates_ocr",
        "verdict",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "plates_ocr",
        "verdict",
    }, "Expected all declared outputs to be delivered"
    assert (
        isinstance(result[0]["plates_ocr"], list) and len(result[0]["plates_ocr"]) == 2
    ), "Expected 2 plates to be found in first image"
    assert (
        isinstance(result[0]["verdict"], list) and len(result[0]["verdict"]) == 2
    ), "Expected verdict for each plate recognised in first image"
    assert result[0]["verdict"][0] in {"PASS", "FAIL"}, "Expected valid verdict"
    assert result[0]["verdict"][1] in {"PASS", "FAIL"}, "Expected valid verdict"
    assert (
        isinstance(result[1]["plates_ocr"], list) and len(result[1]["plates_ocr"]) == 0
    ), "Expected 0 plates to be found in second image"
    assert (
        isinstance(result[1]["verdict"], list) and len(result[1]["verdict"]) == 0
    ), "Expected 0 verdicts to be given in second image"


WORKFLOW_WITH_INVALID_AGGREGATION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "plates_detection",
            "image": "$inputs.image",
            "model_id": "vehicle-registration-plates-trudk/2",
        },
        {
            "type": "DimensionCollapse",
            "name": "invalid_concatenation",
            "data": "$steps.plates_detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.invalid_concatenation.output",
        },
    ],
}


def test_workflow_when_there_is_faulty_application_of_aggregation_step_at_batch_with_dimension_1(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(StepOutputLineageError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_INVALID_AGGREGATION,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


WORKFLOW_WITH_ASPECT_RATIO_EXTRACTION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "PropertyDefinition",
            "name": "property_extraction",
            "data": "$inputs.image",
            "operations": [
                {"type": "ExtractImageProperty", "property_name": "aspect_ratio"}
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "aspect_ratio",
            "selector": "$steps.property_extraction.output",
        },
    ],
}


def test_workflow_with_aspect_ratio_extraction_with_valid_input(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_ASPECT_RATIO_EXTRACTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": license_plate_image})

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "aspect_ratio",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["aspect_ratio"] == 1.5, "Expected aspect ratio to be 1.5"


WORKFLOW_WITH_FRAME_NUMBER_EXTRACTION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "PropertyDefinition",
            "name": "property_extraction",
            "data": "$inputs.image",
            "operations": [
                {"type": "ExtractFrameMetadata", "property_name": "frame_number"}
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "frame_number",
            "selector": "$steps.property_extraction.output",
        },
    ],
}


def test_workflow_with_frame_number_extraction_from_photo(
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_FRAME_NUMBER_EXTRACTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": license_plate_image})

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "frame_number",
    }, "Expected all declared outputs to be delivered"
    assert (
        result[0]["frame_number"] == 0
    ), "Expected frame number to be 0 (for photos there is always only one frame)"


WORKFLOW_WITH_FRAME_TIMESTAMP_EXTRACTION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "PropertyDefinition",
            "name": "frame_timestamp",
            "data": "$inputs.image",
            "operations": [
                {"type": "ExtractFrameMetadata", "property_name": "frame_timestamp"}
            ],
        },
        {
            "type": "PropertyDefinition",
            "name": "frame_number",
            "data": "$inputs.image",
            "operations": [
                {"type": "ExtractFrameMetadata", "property_name": "frame_number"}
            ],
        },
        {
            "type": "PropertyDefinition",
            "name": "seconds_since_start",
            "data": "$inputs.image",
            "operations": [
                {"type": "ExtractFrameMetadata", "property_name": "seconds_since_start"}
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "frame_timestamp",
            "selector": "$steps.frame_timestamp.output",
        },
        {
            "type": "JsonField",
            "name": "frame_number",
            "selector": "$steps.frame_number.output",
        },
        {
            "type": "JsonField",
            "name": "seconds_since_start",
            "selector": "$steps.seconds_since_start.output",
        },
    ],
}


def test_workflow_with_timestamp_extraction_from_photo(
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_FRAME_TIMESTAMP_EXTRACTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": license_plate_image})

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "frame_timestamp",
        "frame_number",
        "seconds_since_start",
    }, "Expected all declared outputs to be delivered"


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_workflow_with_timestamp_extraction_from_video(
    local_video_path: str,
) -> None:
    # given
    video_capture = cv.VideoCapture(local_video_path)
    frames_count = int(round(video_capture.get(cv.CAP_PROP_FRAME_COUNT)))
    fps = video_capture.get(cv.CAP_PROP_FPS)
    video_capture.release()
    last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(local_video_path))
    timestamp_created = last_modified - datetime.timedelta(seconds=frames_count / fps)

    video_source = VideoSource.init(video_reference=local_video_path)
    watchdog = BasePipelineWatchDog()
    watchdog.register_video_sources(video_sources=[video_source])
    predictions = []

    def on_prediction(prediction: dict, video_frame: VideoFrame) -> None:
        predictions.append(prediction)

    workflow_init_parameters = {
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    inference_pipeline = InferencePipeline.init_with_workflow(
        workflow_specification=WORKFLOW_WITH_FRAME_TIMESTAMP_EXTRACTION,
        workflow_init_parameters=workflow_init_parameters,
        video_reference=local_video_path,
        on_prediction=on_prediction,
        max_fps=200,
        watchdog=watchdog,
    )

    # when
    inference_pipeline.start()
    inference_pipeline.join()

    # then
    assert len(predictions) == frames_count
    assert set(predictions[0].keys()) == {
        "frame_timestamp",
        "frame_number",
        "seconds_since_start",
    }, "Expected all declared outputs to be delivered"

    for i in range(1, frames_count + 1):
        assert predictions[i - 1]["frame_number"] == i
        assert predictions[i - 1][
            "frame_timestamp"
        ] == timestamp_created + datetime.timedelta(seconds=(i - 1) / fps)
        assert predictions[i - 1]["seconds_since_start"] == (i - 1) / fps

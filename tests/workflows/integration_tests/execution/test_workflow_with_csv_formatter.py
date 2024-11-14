import io

import numpy as np
import pandas as pd

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_CSV_FORMATTER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "additional_column_value"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/csv_formatter@v1",
            "name": "csv_formatter",
            "columns_data": {
                "predicted_classes": "$steps.model.predictions",
                "number_of_bounding_boxes": "$steps.model.predictions",
                "additional_column": "$inputs.additional_column_value",
            },
            "columns_operations": {
                "predicted_classes": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ],
                "number_of_bounding_boxes": [{"type": "SequenceLength"}],
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "csv",
            "coordinates_system": "own",
            "selector": "$steps.csv_formatter.csv_content",
        }
    ],
}


@add_to_workflows_gallery(
    category="Data analytics in Workflows",
    use_case_title="Workflow producing CSV",
    use_case_description="""
This example showcases how to export CSV file out of Workflow. Object detection results are 
processed with **CSV Formatter** block to produce aggregated results. 
    """,
    workflow_definition=WORKFLOW_WITH_CSV_FORMATTER,
    workflow_name_in_app="csv-formatter",
)
def test_workflow_with_csv_formatter_against_batch_of_images(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CSV_FORMATTER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, license_plate_image],
            "additional_column_value": 2137,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 element in the output for two input images"
    assert set(result[0].keys()) == {
        "csv",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "csv",
    }, "Expected all declared outputs to be delivered"
    assert (
        result[0]["csv"] is None
    ), "Expected the output for batch to be provided for the last batch element"
    parsed_data = pd.read_csv(io.StringIO(result[1]["csv"]))
    assert set(parsed_data.columns.tolist()) == {
        "number_of_bounding_boxes",
        "predicted_classes",
        "additional_column",
        "timestamp",
    }, "Expected to see specific columns in output CSV"
    assert len(parsed_data) == 2, "Expected 2 rows in dataframe"
    assert parsed_data["number_of_bounding_boxes"].tolist() == [
        2,
        3,
    ], "Expected 2 dogs and 3 cars"
    assert (
        parsed_data["predicted_classes"].iloc[0] == "['dog', 'dog']"
    ), "Expected 2 dogs in first image"
    assert (
        parsed_data["predicted_classes"].iloc[1] == "['car', 'car', 'car']"
    ), "Expected 3 cars in second image"
    assert parsed_data["additional_column"].tolist() == [
        2137,
        2137,
    ], "Expected input data to propagate to output CSV"


def test_workflow_with_csv_formatter_against_single_image(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CSV_FORMATTER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "additional_column_value": 2137,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "csv",
    }, "Expected all declared outputs to be delivered"
    parsed_data = pd.read_csv(io.StringIO(result[0]["csv"]))
    assert set(parsed_data.columns.tolist()) == {
        "number_of_bounding_boxes",
        "predicted_classes",
        "additional_column",
        "timestamp",
    }, "Expected to see specific columns in output CSV"
    assert len(parsed_data) == 1, "Expected 1 rows i dataframe"
    assert parsed_data["number_of_bounding_boxes"].tolist() == [2], "Expected 2"
    assert (
        parsed_data["predicted_classes"].iloc[0] == "['dog', 'dog']"
    ), "Expected 2 dogs in first image"
    assert parsed_data["additional_column"].tolist() == [
        2137
    ], "Expected input data to propagate to output CSV"


WORKFLOW_WITH_NON_BATCH_ORIENTED_CSV_FORMATTER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "a"},
        {"type": "WorkflowParameter", "name": "b"},
        {"type": "WorkflowParameter", "name": "c"},
    ],
    "steps": [
        {
            "type": "roboflow_core/csv_formatter@v1",
            "name": "csv_formatter",
            "columns_data": {
                "a": "$inputs.a",
                "b": "$inputs.b",
                "c": "$inputs.c",
            },
            "columns_operations": {
                "a": [{"type": "StringToUpperCase"}],
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "csv",
            "coordinates_system": "own",
            "selector": "$steps.csv_formatter.csv_content",
        }
    ],
}


def test_workflow_with_csv_formatter_against_non_batch_inputs(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_CSV_FORMATTER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "a": "some",
            "b": "other",
            "c": 4,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "csv",
    }, "Expected all declared outputs to be delivered"
    parsed_data = pd.read_csv(io.StringIO(result[0]["csv"]))
    assert set(parsed_data.columns.tolist()) == {
        "a",
        "b",
        "c",
        "timestamp",
    }, "Expected to see specific columns in output CSV"
    assert len(parsed_data) == 1, "Expected 1 rows i dataframe"
    assert parsed_data["a"].tolist() == ["SOME"]
    assert parsed_data["b"].tolist() == ["other"]
    assert parsed_data["c"].tolist() == [4]

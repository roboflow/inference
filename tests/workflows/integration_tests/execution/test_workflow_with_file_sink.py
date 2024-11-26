import json
import os.path
from glob import glob

import numpy as np
import pandas as pd

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_FILE_SINK_AND_AGGREGATION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "target_directory"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "$inputs.model_id",
        },
        {
            "type": "roboflow_core/expression@v1",
            "name": "json_formatter",
            "data": {
                "predictions": "$steps.model.predictions",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {
                    "type": "DynamicCaseResult",
                    "parameter_name": "predictions",
                    "operations": [
                        {"type": "DetectionsToDictionary"},
                        {"type": "ConvertDictionaryToJSON"},
                    ],
                },
            },
        },
        {
            "type": "roboflow_core/local_file_sink@v1",
            "name": "predictions_sink",
            "content": "$steps.json_formatter.output",
            "file_type": "json",
            "output_mode": "separate_files",
            "target_directory": "$inputs.target_directory",
            "file_name_prefix": "prediction",
        },
        {
            "type": "roboflow_core/data_aggregator@v1",
            "name": "data_aggregation",
            "data": {
                "predicted_classes": "$steps.model.predictions",
                "number_of_predictions": "$steps.model.predictions",
            },
            "data_operations": {
                "predicted_classes": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ],
                "number_of_predictions": [{"type": "SequenceLength"}],
            },
            "aggregation_mode": {
                "predicted_classes": ["count_distinct"],
                "number_of_predictions": ["min", "max", "sum"],
            },
            "interval": 6,
            "interval_unit": "runs",
        },
        {
            "type": "roboflow_core/csv_formatter@v1",
            "name": "csv_formatter",
            "columns_data": {
                "number_of_distinct_classes": "$steps.data_aggregation.predicted_classes_count_distinct",
                "min_number_of_bounding_boxes": "$steps.data_aggregation.number_of_predictions_min",
                "max_number_of_bounding_boxes": "$steps.data_aggregation.number_of_predictions_max",
                "total_number_of_bounding_boxes": "$steps.data_aggregation.number_of_predictions_sum",
            },
        },
        {
            "type": "roboflow_core/local_file_sink@v1",
            "name": "reports_sink",
            "content": "$steps.csv_formatter.csv_content",
            "file_type": "csv",
            "output_mode": "append_log",
            "target_directory": "$inputs.target_directory",
            "file_name_prefix": "aggregation_report",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.model.predictions",
        }
    ],
}


@add_to_workflows_gallery(
    category="Data analytics in Workflows",
    use_case_title="Saving Workflow results into file",
    use_case_description="""
This Workflow was created to achieve few ends:

* getting predictions from object detection model and returning them to the caller

* persisting the predictions - each one in separate JSON file

* aggregating the predictions data - producing report on each 6th input image

* saving the results in CSV file, appending rows until file size is exceeded 

!!! warning "Run on video to produce *meaningful* results"

    This workflow will not work using the docs preview. You must run it on video file.
    Copy the template into your Roboflow app, start `inference` server and use video preview 
    to get the results. 
""",
    workflow_definition=WORKFLOW_WITH_FILE_SINK_AND_AGGREGATION,
    workflow_name_in_app="file-sink-for-data-aggregation",
)
def test_workflow_with_data_aggregation(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
    license_plate_image: np.ndarray,
    empty_directory: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_FILE_SINK_AND_AGGREGATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    recorded_predictions = []
    for _ in range(4):
        results = execution_engine.run(
            runtime_parameters={
                "image": [dogs_image, crowd_image, license_plate_image],
                "target_directory": empty_directory,
            }
        )
        for result in results:
            recorded_predictions.append(result["predictions"])
    # trigger aggregated file flush
    del execution_engine

    # then
    assert len(recorded_predictions) == 12, "Expected 12 predictions"
    persisted_predictions = glob(os.path.join(empty_directory, "prediction_*.json"))
    assert len(persisted_predictions) == 12, "Expected all predictions to be persisted"
    recovered_predictions = []
    for path in sorted(persisted_predictions):
        recovered_predictions.append(_load_json_file(path=path))
    for recorded_prediction, recovered_prediction in zip(
        recorded_predictions, recovered_predictions
    ):
        assert len(recorded_prediction) == len(
            recovered_prediction["predictions"]
        ), "Expected persisted prediction ot be the same as the one recorded in memory"
    reports_files = glob(os.path.join(empty_directory, "aggregation_report_*.csv"))
    assert len(reports_files) == 1, "Expected one report file"
    report = pd.read_csv(reports_files[0])
    assert len(report) == 2, "Expected 2 rows in report"
    assert report["number_of_distinct_classes"].tolist() == [
        3,
        3,
    ], "dog, person, car unique classes expected"
    assert report["min_number_of_bounding_boxes"].tolist() == [2, 2]
    assert report["max_number_of_bounding_boxes"].tolist() == [12, 12]
    assert report["total_number_of_bounding_boxes"].tolist() == [34, 34]


def _load_json_file(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

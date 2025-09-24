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

WORKFLOW_WITH_DELTA_FILTER = {
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
                                            "property_name": "center",
                                        }
                                    ],
                                },
                                "comparator": {"type": "(Detection) in zone"},
                                "right_operand": {
                                    "type": "StaticOperand",
                                    "value": [
                                        [0, 0],
                                        [0, 1000],
                                        [1000, 1000],
                                        [1000, 0],
                                    ],
                                },
                                "negate": False,
                            }
                        ],
                    },
                }
            ],
            "operations_parameters": {},
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "property_definition",
            "data": "$steps.detections_filter.predictions",
            "operations": [{"type": "SequenceLength"}],
        },
        {
            "type": "roboflow_core/delta_filter@v1",
            "name": "delta_filter",
            "value": "$steps.property_definition.output",
            "image": "$inputs.image",
            "next_steps": [
                "$steps.csv_formatter",
            ],
        },
        {
            "type": "roboflow_core/csv_formatter@v1",
            "name": "csv_formatter",
            "columns_data": {"Class Name": "$steps.detections_filter.predictions"},
            "columns_operations": {
                "Class Name": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            },
        },
        {
            "type": "roboflow_core/local_file_sink@v1",
            "name": "reports_sink",
            "content": "$steps.csv_formatter.csv_content",
            "file_type": "csv",
            "output_mode": "append_log",
            "target_directory": "$inputs.target_directory",
            "file_name_prefix": "csv_containing_changes",
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
    category="Filtering resulting data based on value delta change",
    use_case_title="Saving Workflow results into file, but only if value changes between frames",
    use_case_description="""
This Workflow was created to achieve few ends:

* getting predictions from object detection model

* filtering out predictions found outside of zone

* counting detections in zone

* if count of detection in zone changes save results to csv file

!!! warning "Run on video to produce *meaningful* results"

    This workflow will not work using the docs preview. You must run it on video file.
    Copy the template into your Roboflow app, start `inference` server and use video preview 
    to get the results. 
""",
    workflow_definition=WORKFLOW_WITH_DELTA_FILTER,
    workflow_name_in_app="file-sink-for-data-aggregation",
)
def test_workflow_with_delta_filter(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
    empty_directory: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_DELTA_FILTER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    execution_engine.run(
        runtime_parameters={
            "image": [crowd_image],
            "target_directory": empty_directory,
        }
    )
    execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "target_directory": empty_directory,
        }
    )
    execution_engine.run(
        runtime_parameters={
            "image": [crowd_image],
            "target_directory": empty_directory,
        }
    )
    execution_engine.run(
        runtime_parameters={
            "image": [crowd_image],
            "target_directory": empty_directory,
        }
    )
    # trigger aggregated file flush
    del execution_engine

    # then
    reports_files = glob(os.path.join(empty_directory, "csv_containing_changes_*.csv"))
    assert len(reports_files) == 1, "Expected one report file"
    report = pd.read_csv(reports_files[0])
    assert len(report) == 3, "Expected 3 rows in report"

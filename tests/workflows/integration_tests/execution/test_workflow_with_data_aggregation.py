import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_DATA_AGGREGATION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
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
                "predicted_classes": ["distinct", "count_distinct", "values_counts"],
                "number_of_predictions": ["min", "max", "sum"],
            },
            "interval": 6,
            "interval_unit": "runs",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "aggregation_results",
            "selector": "$steps.data_aggregation.*",
        }
    ],
}


@add_to_workflows_gallery(
    category="Data analytics in Workflows",
    use_case_title="Aggregation of results over time",
    use_case_description="""
This example shows how to aggregate and analyse predictions using Workflows.

The key for data analytics in this example is **Data Aggregator** block which is fed with model 
predictions and perform the following aggregations **on each 6 consecutive predictions:**

* taking **classes names** from  bounding boxes, it outputs **unique classes names, number of unique classes and
number of bounding boxes for each class**

* taking the **number of detected bounding boxes** in each prediction, it outputs **minimum, maximum and total number** 
of bounding boxes per prediction in aggregated time window 

!!! warning "Run on video to produce *meaningful* results"

    This workflow will not work using the docs preview. You must run it on video file.
    Copy the template into your Roboflow app, start `inference` server and use video preview 
    to get the results. 
""",
    workflow_definition=WORKFLOW_WITH_DATA_AGGREGATION,
    workflow_name_in_app="data-aggregation",
)
def test_workflow_with_data_aggregation(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_DATA_AGGREGATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = []
    for _ in range(4):
        result = execution_engine.run(
            runtime_parameters={
                "image": [dogs_image, crowd_image, license_plate_image],
            }
        )
        results.append(result)

    for result in results:
        assert (
            len(result) == 3
        ), "Each result must hold output values for 3 input images"

    empty_aggregation = {
        "predicted_classes_distinct": None,
        "predicted_classes_count_distinct": None,
        "predicted_classes_values_counts": None,
        "number_of_predictions_min": None,
        "number_of_predictions_max": None,
        "number_of_predictions_sum": None,
    }
    assert (
        results[0] == [{"aggregation_results": empty_aggregation}] * 3
    ), "Aggregation results should be empty for first round of images"
    assert (
        results[1][:2] == [{"aggregation_results": empty_aggregation}] * 2
    ), "Aggregation results should be empty for first two elements of second round of images"
    assert (
        results[2] == [{"aggregation_results": empty_aggregation}] * 3
    ), "Aggregation results should be empty for third round of images"
    assert (
        results[3][:2] == [{"aggregation_results": empty_aggregation}] * 2
    ), "Aggregation results should be empty for first two elements of fourth round of images"
    assert set(results[1][2]["aggregation_results"]["predicted_classes_distinct"]) == {
        "car",
        "person",
        "dog",
    }
    assert results[1][2]["aggregation_results"]["predicted_classes_count_distinct"] == 3
    assert results[1][2]["aggregation_results"]["predicted_classes_values_counts"] == {
        "dog": 4,
        "person": 24,
        "car": 6,
    }
    assert results[1][2]["aggregation_results"]["number_of_predictions_min"] == 2
    assert results[1][2]["aggregation_results"]["number_of_predictions_max"] == 12
    assert results[1][2]["aggregation_results"]["number_of_predictions_sum"] == 34
    assert set(results[3][2]["aggregation_results"]["predicted_classes_distinct"]) == {
        "car",
        "person",
        "dog",
    }
    assert results[3][2]["aggregation_results"]["predicted_classes_count_distinct"] == 3
    assert results[3][2]["aggregation_results"]["predicted_classes_values_counts"] == {
        "dog": 4,
        "person": 24,
        "car": 6,
    }
    assert results[3][2]["aggregation_results"]["number_of_predictions_min"] == 2
    assert results[3][2]["aggregation_results"]["number_of_predictions_max"] == 12
    assert results[3][2]["aggregation_results"]["number_of_predictions_sum"] == 34

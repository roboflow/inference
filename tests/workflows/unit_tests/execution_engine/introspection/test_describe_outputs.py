from inference.core.interfaces.http.handlers.workflows import (
    handle_describe_workflows_output,
)
from inference.core.entities.requests.workflows import DescribeOutputRequest

workflow_definition = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.detection.predictions",
        }
    ],
}


def test_handle_describe_workflows_output_when_valid_request_provided() -> None:
    # when
    result = handle_describe_workflows_output(
        workflow_request=DescribeOutputRequest(
            api_key="test",
            workspace_name="test",
            workflow_id="test",
        ),
        workflow_specification=workflow_definition,
    )
    print("result", result)
    assert result == {
        "$steps.detection.predictions": {
            "inference_id": ["string"],
            "predictions": ["Batch[object_detection_prediction]"],
        }
    }

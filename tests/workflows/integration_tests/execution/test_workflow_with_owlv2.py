import numpy as np
from roboflow_workflows.core_steps.models.foundation.owlv2 import v1

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine


def test_remote_workflow_sends_runtime_prompts_and_returns_labeled_boxes(
    requests_mock, monkeypatch, model_manager
):
    api_url = "http://localhost:9001"
    monkeypatch.setattr(v1, "WORKFLOWS_REMOTE_API_TARGET", "local")
    monkeypatch.setattr(v1, "LOCAL_INFERENCE_API_URL", api_url)
    monkeypatch.setattr(v1, "WORKFLOWS_REMOTE_API_KEY_TRANSPORT", "header")
    box = {
        "x": 4,
        "y": 5,
        "width": 2,
        "height": 4,
        "class": "bicycle",
        "class_id": 1,
        "confidence": 0.9,
    }
    endpoint = requests_mock.post(
        f"{api_url}/owlv2/infer",
        json={"image": {"width": 8, "height": 8}, "predictions": [box]},
    )
    workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "prompts"},
        ],
        "steps": [
            {
                "type": "roboflow_core/owlv2_model@v1",
                "name": "detect",
                "images": "$inputs.image",
                "model_id": "ds/1",
                "text_prompts": "$inputs.prompts",
                "confidence": 0.4,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "boxes",
                "selector": "$steps.detect.predictions",
            }
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={
            "workflows_core.step_execution_mode": StepExecutionMode.REMOTE,
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "test-key",
        },
    )
    prompts = ["bus", "bicycle"]
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        outputs = engine.run(
            runtime_parameters={"image": [pixels, pixels], "prompts": prompts},
            serialize_results=True,
        )
    finally:
        engine.close()

    assert len(outputs) == endpoint.call_count == 2
    for request in endpoint.request_history:
        body = request.json()
        assert request.headers["Authorization"] == "Bearer test-key"
        assert body["model_id"] == "ds/1" and body["text"] == prompts
        assert body["confidence"] == 0.4
        assert body["image"]["type"] == "base64" and body["image"]["value"]
    for output in outputs:
        (prediction,) = output["boxes"]["predictions"]
        assert prediction["class"] == "bicycle" and prediction["class_id"] == 1
        assert [prediction[k] for k in ("x", "y", "width", "height")] == [4, 5, 2, 4]

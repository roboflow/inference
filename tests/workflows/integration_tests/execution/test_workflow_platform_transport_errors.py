"""Platform transport errors must be safe after workflow error serialization."""

import traceback

import numpy as np
import pytest
import requests

from inference.core import roboflow_api
from inference.core.interfaces.roboflow_platform_client import (
    ServerRoboflowPlatformClient,
)
from inference.core.workflows.errors import StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine

API_KEY = "synthetic-workflow-transport-api-key"


@pytest.mark.parametrize(
    "block_type", ["roboflow_core/open_ai@v7", "roboflow_core/open_ai_decisions@v1"]
)
@pytest.mark.parametrize("failure", [requests.Timeout, requests.ConnectionError])
def test_managed_transport_errors_redact_workflow_diagnostics(
    block_type, failure, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    proxy_url = f"{roboflow_api.API_PROXY_BASE_URL}/apiproxy/"
    calls = []

    def fail_request(url, **kwargs):
        # requests.post is patched for the whole process, so the usage
        # collector's background flush can land here too. Count only the
        # requests the workflow sends to the API proxy.
        if url.startswith(proxy_url):
            calls.append(url)

        raise failure("Request failed for " + url)

    monkeypatch.setattr(roboflow_api.requests, "post", fail_request)
    step = {
        "type": block_type,
        "name": "model",
        "images": "$inputs.image",
        "api_key": "rf_key:account",
    }
    if block_type == "roboflow_core/open_ai_decisions@v1":
        step.update(
            task_type="questions",
            questions=[{"type": "predicate", "instructions": "Is there a cat?"}],
        )
    else:
        step.update(task_type="unconstrained", prompt="Describe the image.")

    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [step],
        "outputs": [
            {"type": "JsonField", "name": "id", "selector": "$steps.model.inference_id"}
        ],
    }
    engine = ExecutionEngine.init(
        definition,
        init_parameters={
            "workflows_core.api_key": API_KEY,
            "workflows_core.platform_client": ServerRoboflowPlatformClient(),
        },
    )
    try:
        with pytest.raises(StepExecutionError) as caught:
            engine.run({"image": np.zeros((37, 83, 3), dtype=np.uint8)})
    finally:
        engine.close()

    assert len(calls) == 1
    assert API_KEY in calls[0], "The failure must originate from a credentialed URL."
    assert caught.value.block_type == block_type
    assert caught.value.block_traceback is not None
    formatted_traceback = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    assert API_KEY not in str(caught.value)
    assert API_KEY not in caught.value.block_traceback.model_dump_json()
    assert API_KEY not in formatted_traceback
    assert API_KEY not in caplog.text

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np

from inference.core.workflows.core_steps.flow_control.inner_workflow.v1 import (
    BlockManifest,
    InnerWorkflowBlockV1,
    prepare_workflow_dispatch_request,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_dispatch_manifest_has_no_outputs() -> None:
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "dispatch",
            "execution_mode": "dispatch_to_serverless",
            "workflow_workspace_id": "workspace",
            "workflow_id": "slow-workflow",
            "parameter_bindings": {"image": "$inputs.image"},
        }
    )

    assert manifest.get_actual_outputs() == []


def test_embedded_manifest_retains_wildcard_compile_time_output() -> None:
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "embedded",
            "workflow_workspace_id": "workspace",
            "workflow_id": "child",
            "parameter_bindings": {"message": "$inputs.message"},
        }
    )

    assert [output.name for output in manifest.get_actual_outputs()] == ["*"]


def test_prepare_named_workflow_dispatch_serializes_inputs_and_uses_override_url() -> (
    None
):
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((2, 3, 3), dtype=np.uint8),
    )
    video_metadata = VideoMetadata(
        video_identifier="camera-1",
        frame_number=7,
        frame_timestamp=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    url, payload = prepare_workflow_dispatch_request(
        dispatch_target_url="http://127.0.0.1:9001/",
        api_key="secret",
        parameter_bindings={
            "image": image,
            "video_metadata": video_metadata,
            "threshold": np.float32(0.5),
        },
        workflow_definition=None,
        workflow_workspace_id="workspace",
        workflow_id="slow-workflow",
        workflow_version_id="3",
    )

    assert url == "http://127.0.0.1:9001/workspace/workflows/slow-workflow"
    assert payload["api_key"] == "secret"
    assert payload["inputs"]["image"]["type"] == "base64"
    assert isinstance(payload["inputs"]["image"]["value"], str)
    assert payload["inputs"]["threshold"] == 0.5
    assert payload["inputs"]["video_metadata"]["video_identifier"] == "camera-1"
    assert payload["inputs"]["video_metadata"]["frame_timestamp"] == (
        "2026-09-02T00:00:00Z"
    )
    assert payload["use_cache"] is True
    assert payload["workflow_version_id"] == "3"


def test_prepare_inline_workflow_dispatch_uses_specification_endpoint() -> None:
    specification = {
        "version": "1.0",
        "inputs": [],
        "steps": [],
        "outputs": [],
    }

    url, payload = prepare_workflow_dispatch_request(
        dispatch_target_url="https://serverless.roboflow.com",
        api_key=None,
        parameter_bindings={},
        workflow_definition=specification,
        workflow_workspace_id=None,
        workflow_id=None,
        workflow_version_id=None,
    )

    assert url == "https://serverless.roboflow.com/workflows/run"
    assert payload["specification"] is specification


def test_dispatch_is_submitted_to_background_executor() -> None:
    executor = MagicMock()
    block = InnerWorkflowBlockV1(
        api_key="secret",
        background_tasks=None,
        thread_pool_executor=executor,
        inner_workflow_dispatch_target_url="https://dedicated.example.com",
    )

    result = block.run(
        execution_mode="dispatch_to_serverless",
        parameter_bindings={"message": "hello"},
        workflow_definition=None,
        workflow_workspace_id="workspace",
        workflow_id="slow-workflow",
        workflow_version_id=None,
    )

    assert result == {}
    executor.submit.assert_called_once()
    submitted_request = executor.submit.call_args.args[0]
    assert submitted_request.keywords["url"] == (
        "https://dedicated.example.com/workspace/workflows/slow-workflow"
    )
    assert submitted_request.keywords["payload"]["inputs"] == {"message": "hello"}

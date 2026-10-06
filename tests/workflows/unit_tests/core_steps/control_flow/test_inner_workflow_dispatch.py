import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from inference.core.entities.requests.workflows import (
    PredefinedWorkflowInferenceRequest,
)
from inference.core.env import WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH
from inference.core.workflows.core_steps.flow_control.inner_workflow.v1 import (
    BlockManifest,
    InnerWorkflowBlockV1,
    execute_workflow_dispatch_request,
    logger,
    normalize_workflow_remote_target,
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
            "execution_mode": "remote_dispatch",
            "remote_target": "https://dedicated.example.com",
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
    video_metadata = VideoMetadata(
        video_identifier="camera-1",
        frame_number=7,
        frame_timestamp=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((2, 3, 3), dtype=np.uint8),
        video_metadata=video_metadata,
    )

    url, payload = prepare_workflow_dispatch_request(
        remote_target="http://127.0.0.1:9001/",
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
    encoded_payload = json.loads(json.dumps(payload))
    assert (
        datetime.fromisoformat(
            encoded_payload["inputs"]["image"]["video_metadata"]["frame_timestamp"]
        )
        == video_metadata.frame_timestamp
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
        remote_target="https://serverless.roboflow.com",
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
        inner_workflow_remote_target="https://serverless.roboflow.com",
    )

    result = block.run(
        execution_mode="remote_dispatch",
        remote_target="https://dedicated.example.com",
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


def test_dispatch_forwards_credentials_only_to_runtime_target() -> None:
    for configured_target, override, expected_key in [
        ("https://serverless.roboflow.com", None, "secret"),
        ("https://dedicated.example.com/", " https://dedicated.example.com ", "secret"),
        ("https://serverless.roboflow.com", "https://receiver.example.com", None),
        ("https://serverless.roboflow.com", "http://serverless.roboflow.com", None),
        (
            "https://serverless.roboflow.com",
            "https://serverless.roboflow.com.evil.example",
            None,
        ),
    ]:
        executor = MagicMock()
        block = InnerWorkflowBlockV1(
            api_key="secret",
            background_tasks=None,
            thread_pool_executor=executor,
            inner_workflow_remote_target=configured_target,
        )
        block.run(
            execution_mode="remote_dispatch",
            remote_target=override,
            parameter_bindings={},
            workflow_definition=None,
            workflow_workspace_id="workspace",
            workflow_id="child",
            workflow_version_id=None,
        )
        payload = executor.submit.call_args.args[0].keywords["payload"]
        assert payload["api_key"] == expected_key


def test_dispatch_rejects_ambiguous_target_urls() -> None:
    for target in [
        "file:///tmp/workflow",
        "https:///missing-host",
        "https://user:password@example.com",
        "https://example.com?redirect=elsewhere",
        "https://example.com#fragment",
    ]:
        with pytest.raises(ValueError, match="HTTP"):
            normalize_workflow_remote_target(target)


def test_dispatch_does_not_follow_redirects() -> None:
    response = MagicMock(status_code=307)
    payload = {"api_key": "secret", "inputs": {}}
    with patch.object(requests, "post", return_value=response) as post:
        with patch.object(logger, "warning") as warning:
            execute_workflow_dispatch_request(
                "https://trusted.example/workflows/run", payload
            )
    assert post.call_args.kwargs["allow_redirects"] is False
    warning.assert_called_once()
    response.raise_for_status.assert_not_called()


def test_self_dispatch_stops_at_depth_limit() -> None:
    request = PredefinedWorkflowInferenceRequest(inputs={})
    for depth in range(WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH + 1):
        executor = MagicMock()
        block = InnerWorkflowBlockV1(
            api_key="secret",
            background_tasks=None,
            thread_pool_executor=executor,
            inner_workflow_remote_target="https://serverless.roboflow.com",
            inner_workflow_dispatch_depth=request.inner_workflow_dispatch_depth,
        )
        arguments = dict(
            execution_mode="remote_dispatch",
            remote_target=None,
            parameter_bindings={},
            workflow_definition=None,
            workflow_workspace_id="workspace",
            workflow_id="self",
            workflow_version_id=None,
        )
        if depth == WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH:
            with pytest.raises(ValueError, match="dispatch depth"):
                block.run(**arguments)
            executor.submit.assert_not_called()
        else:
            block.run(**arguments)
            payload = executor.submit.call_args.args[0].keywords["payload"]
            request = PredefinedWorkflowInferenceRequest.model_validate(payload)
            assert request.inner_workflow_dispatch_depth == depth + 1


def test_disabled_dispatch_does_not_check_depth_or_submit() -> None:
    executor = MagicMock()
    block = InnerWorkflowBlockV1(
        api_key="secret",
        background_tasks=None,
        thread_pool_executor=executor,
        inner_workflow_remote_target="https://serverless.roboflow.com",
        inner_workflow_dispatch_depth=WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH,
        disable_sinks=True,
    )
    assert (
        block.run(
            execution_mode="remote_dispatch",
            remote_target=None,
            parameter_bindings={},
            workflow_definition=None,
            workflow_workspace_id="workspace",
            workflow_id="self",
            workflow_version_id=None,
        )
        == {}
    )
    executor.submit.assert_not_called()


def test_empty_inline_definition_dispatches_saved_workflow_reference() -> None:
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "dispatch",
            "execution_mode": "remote_dispatch",
            "workflow_definition": {},
            "workflow_workspace_id": "workspace",
            "workflow_id": "child",
            "workflow_version_id": "3",
            "parameter_bindings": {},
        }
    )
    executor = MagicMock()
    block = InnerWorkflowBlockV1(
        api_key="secret",
        background_tasks=None,
        thread_pool_executor=executor,
        inner_workflow_remote_target="https://serverless.roboflow.com",
    )

    block.run(**manifest.model_dump(exclude={"name", "type"}))

    request = executor.submit.call_args.args[0].keywords
    assert request["url"] == "https://serverless.roboflow.com/workspace/workflows/child"
    assert "specification" not in request["payload"]
    assert request["payload"]["workflow_version_id"] == "3"

from unittest import mock
from unittest.mock import MagicMock, PropertyMock

import pytest
from pydantic import ValidationError
from requests import Response

from inference.core.workflows.core_steps.composition.workflow_caller import v1
from inference.core.workflows.core_steps.composition.workflow_caller.v1 import (
    BlockManifest,
    WorkflowCallerBlockV1,
    _RESOLVED_WORKFLOW_INPUTS,
    _RESOLVED_WORKFLOW_OUTPUTS,
    _build_kind_name_map,
    _build_kinds_deserializers_map,
    _convert_output_descriptions_to_kinds,
    _deserialize_output_value,
    _extract_output_names_with_wildcard,
    _make_cache_key,
    _resolve_and_cache_workflow_inputs,
    _resolve_and_cache_workflow_outputs,
    _resolve_output_kinds_for_run,
    _extract_workflow_caller_steps,
    _validate_required_inputs,
    build_workflow_inputs,
    build_workflow_url,
    call_workflow,
    execute_workflow_request,
    validate_workflow_caller_no_circular_references,
    MAX_WORKFLOW_CALL_DEPTH,
    WORKFLOW_CALL_CHAIN_HEADER,
    WORKFLOW_CALLER_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    IMAGE_KIND,
    STRING_KIND,
    WILDCARD_KIND,
)
from inference.core.workflows.errors import ExecutionGraphStructureError


def _make_mock_image(base64_value: str = "dGVzdA==") -> MagicMock:
    """Create a mock WorkflowImageData for testing."""
    mock_img = MagicMock()
    mock_img.base64_image = base64_value
    return mock_img


@pytest.fixture(autouse=True)
def _clear_resolved_caches():
    """Ensure the module-level caches are clean before and after each test."""
    _RESOLVED_WORKFLOW_OUTPUTS.clear()
    _RESOLVED_WORKFLOW_INPUTS.clear()
    _build_kind_name_map.cache_clear()
    _build_kinds_deserializers_map.cache_clear()
    yield
    _RESOLVED_WORKFLOW_OUTPUTS.clear()
    _RESOLVED_WORKFLOW_INPUTS.clear()
    _build_kind_name_map.cache_clear()
    _build_kinds_deserializers_map.cache_clear()


# --- Manifest parsing tests ---


def test_manifest_parsing_when_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "my_workflow_caller",
        "workflow_id": "my-detection-pipeline",
        "inputs": {
            "image": "$inputs.image",
            "confidence_threshold": 0.5,
        },
        "request_timeout": 30,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.type == "roboflow_core/workflow_caller@v1"
    assert result.workflow_id == "my-detection-pipeline"
    assert result.inputs == {"image": "$inputs.image", "confidence_threshold": 0.5}
    assert result.request_timeout == 30


def test_manifest_parsing_with_minimal_config() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "my_workflow_caller",
        "workflow_id": "my-workflow",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.workflow_id == "my-workflow"
    assert result.inputs == {}
    assert result.request_timeout == 30


def test_manifest_parsing_with_selector_references() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "my_workflow_caller",
        "workflow_id": "$inputs.workflow_id",
        "inputs": {
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "threshold": "$steps.config.threshold",
        },
        "request_timeout": "$inputs.timeout",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.workflow_id == "$inputs.workflow_id"
    assert result.inputs == {
        "image": "$inputs.image",
        "model_id": "$inputs.model_id",
        "threshold": "$steps.config.threshold",
    }


def test_manifest_parsing_rejects_zero_timeout() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "my_workflow_caller",
        "workflow_id": "my-workflow",
        "request_timeout": 0,
    }

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_parsing_rejects_negative_timeout() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "my_workflow_caller",
        "workflow_id": "my-workflow",
        "request_timeout": -5,
    }

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_rejects_workflow_id_with_spaces() -> None:
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "caller",
        "workflow_id": "my workflow",
    }
    with pytest.raises(ValidationError, match="valid slug"):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_rejects_workflow_id_with_slashes() -> None:
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "caller",
        "workflow_id": "../../etc/passwd",
    }
    with pytest.raises(ValidationError, match="valid slug"):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_rejects_empty_workflow_id() -> None:
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "caller",
        "workflow_id": "",
    }
    with pytest.raises(ValidationError, match="valid slug"):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_accepts_workflow_id_with_underscores() -> None:
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "caller",
        "workflow_id": "my_workflow",
    }
    result = BlockManifest.model_validate(raw_manifest)
    assert result.workflow_id == "my_workflow"


def test_manifest_accepts_single_char_workflow_id() -> None:
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "caller",
        "workflow_id": "a",
    }
    result = BlockManifest.model_validate(raw_manifest)
    assert result.workflow_id == "a"


def test_manifest_skips_slug_validation_for_selector_workflow_id() -> None:
    raw_manifest = {
        "type": "roboflow_core/workflow_caller@v1",
        "name": "caller",
        "workflow_id": "$inputs.workflow_id",
    }
    result = BlockManifest.model_validate(raw_manifest)
    assert result.workflow_id == "$inputs.workflow_id"


def test_manifest_describe_outputs() -> None:
    # when
    outputs = BlockManifest.describe_outputs()

    # then
    output_names = {o.name for o in outputs}
    assert output_names == {"*"}


# --- get_actual_outputs tests ---


def test_get_actual_outputs_fallback_when_not_resolved() -> None:
    # given - no cache entry for this workflow_id
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "some-workflow",
    })

    # when
    outputs = manifest.get_actual_outputs()

    # then - should fall back to generic "result" output
    output_names = {o.name for o in outputs}
    assert output_names == {"result"}
    result_output = next(o for o in outputs if o.name == "result")
    assert result_output.kind == [DICTIONARY_KIND]


def test_get_actual_outputs_fallback_for_selector_workflow_id() -> None:
    # given - workflow_id is a selector
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "$inputs.workflow_id",
    })

    # when
    outputs = manifest.get_actual_outputs()

    # then - should fall back to generic "result" output
    output_names = {o.name for o in outputs}
    assert output_names == {"result"}


def test_get_actual_outputs_with_resolved_outputs() -> None:
    # given - populate cache with resolved outputs
    _RESOLVED_WORKFLOW_OUTPUTS[("detection-workflow", None)] = {
        "predictions": [OBJECT_DETECTION_PREDICTION_KIND],
        "visualization": [IMAGE_KIND],
    }
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "detection-workflow",
    })

    # when
    outputs = manifest.get_actual_outputs()

    # then - should return resolved outputs with proper kinds
    output_names = {o.name for o in outputs}
    assert output_names == {"predictions", "visualization"}
    predictions_output = next(o for o in outputs if o.name == "predictions")
    assert predictions_output.kind == [OBJECT_DETECTION_PREDICTION_KIND]
    viz_output = next(o for o in outputs if o.name == "visualization")
    assert viz_output.kind == [IMAGE_KIND]


# --- build_workflow_inputs tests ---


def test_build_workflow_inputs_with_image_and_inputs() -> None:
    # given - image is passed as a WorkflowImageData via inputs
    from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
    image = MagicMock(spec=WorkflowImageData)
    image.base64_image = "base64_encoded_data"

    # when
    result = build_workflow_inputs(
        inputs={"image": image, "confidence": 0.5, "model_id": "my_model"},
    )

    # then
    assert result == {
        "image": {"type": "base64", "value": "base64_encoded_data"},
        "confidence": 0.5,
        "model_id": "my_model",
    }


def test_build_workflow_inputs_with_only_image() -> None:
    # given - image is passed via inputs dict
    from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
    mock_img = MagicMock(spec=WorkflowImageData)
    mock_img.base64_image = "dGVzdA=="

    # when
    result = build_workflow_inputs(
        inputs={"image": mock_img, "param": "value"},
    )

    # then - image is serialized
    assert result == {
        "image": {"type": "base64", "value": "dGVzdA=="},
        "param": "value",
    }


def test_build_workflow_inputs_with_custom_image_input_name() -> None:
    # given - image is passed under a custom key in inputs
    from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
    image = MagicMock(spec=WorkflowImageData)
    image.base64_image = "data"

    # when
    result = build_workflow_inputs(
        inputs={"input_image": image},
    )

    # then
    assert "input_image" in result
    assert result["input_image"] == {"type": "base64", "value": "data"}


def test_build_workflow_inputs_serializes_workflow_image_data_in_inputs() -> None:
    # given - inputs contains multiple WorkflowImageData objects
    from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
    primary_image = MagicMock(spec=WorkflowImageData)
    primary_image.base64_image = "primary_data"
    mask_image_real = MagicMock(spec=WorkflowImageData)
    mask_image_real.base64_image = "mask_data"

    # when
    result = build_workflow_inputs(
        inputs={"image": primary_image, "mask": mask_image_real, "threshold": 0.5},
    )

    # then - both images should be serialized as base64 dict, threshold stays as-is
    assert result["image"] == {"type": "base64", "value": "primary_data"}
    assert result["mask"] == {"type": "base64", "value": "mask_data"}
    assert result["threshold"] == 0.5


def test_build_workflow_inputs_does_not_mutate_inputs() -> None:
    # given
    original_inputs = {"key": "value"}

    # when
    result = build_workflow_inputs(
        inputs=original_inputs,
    )

    # then - original dict is not mutated
    assert original_inputs == {"key": "value"}
    assert result is not original_inputs
    assert result["key"] == "value"


# --- build_workflow_url tests ---


def test_build_workflow_url() -> None:
    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"):
        result = build_workflow_url(
            workspace_name="my-workspace",
            workflow_id="my-workflow",
        )

    # then
    assert result == "http://127.0.0.1:9001/my-workspace/workflows/my-workflow"


def test_build_workflow_url_strips_trailing_slash() -> None:
    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001/"):
        result = build_workflow_url(
            workspace_name="my-workspace",
            workflow_id="my-workflow",
        )

    # then
    assert result == "http://127.0.0.1:9001/my-workspace/workflows/my-workflow"


# --- Circular dependency detection tests ---


def test_call_workflow_detects_direct_circular_reference() -> None:
    # when
    error_status, message, result = call_workflow(
        api_key="test_key",
        workspace_name="my-workspace",
        workflow_id="workflow-a",
        inputs={},
        request_timeout=30,
        call_chain="workflow-a",
    )

    # then
    assert error_status is True
    assert "Circular workflow call detected" in message
    assert "workflow-a" in message
    assert result == {}


def test_call_workflow_detects_indirect_circular_reference() -> None:
    # when
    error_status, message, result = call_workflow(
        api_key="test_key",
        workspace_name="my-workspace",
        workflow_id="workflow-a",
        inputs={},
        request_timeout=30,
        call_chain="workflow-a,workflow-b",
    )

    # then
    assert error_status is True
    assert "Circular workflow call detected" in message
    assert result == {}


def test_call_workflow_allows_non_circular_chain() -> None:
    # given
    response = Response()
    response.status_code = 200
    response._content = b'{"outputs": [{"detection": "result"}]}'

    # when
    with mock.patch("requests.post", return_value=response):
        error_status, message, result = call_workflow(
            api_key="test_key",
            workspace_name="my-workspace",
            workflow_id="workflow-c",
            inputs={},
            request_timeout=30,
            call_chain="workflow-a,workflow-b",
        )

    # then
    assert error_status is False
    assert result == {"detection": "result"}


def test_call_workflow_enforces_max_depth() -> None:
    # given
    chain = ",".join([f"workflow-{i}" for i in range(MAX_WORKFLOW_CALL_DEPTH)])

    # when
    error_status, message, result = call_workflow(
        api_key="test_key",
        workspace_name="my-workspace",
        workflow_id="one-more-workflow",
        inputs={},
        request_timeout=30,
        call_chain=chain,
    )

    # then
    assert error_status is True
    assert "depth limit" in message
    assert result == {}


def test_call_workflow_allows_chain_within_depth_limit() -> None:
    # given
    chain = ",".join([f"workflow-{i}" for i in range(MAX_WORKFLOW_CALL_DEPTH - 1)])
    response = Response()
    response.status_code = 200
    response._content = b'{"outputs": [{"key": "value"}]}'

    # when
    with mock.patch("requests.post", return_value=response):
        error_status, message, result = call_workflow(
            api_key="test_key",
            workspace_name="my-workspace",
            workflow_id="last-workflow",
            inputs={},
            request_timeout=30,
            call_chain=chain,
        )

    # then
    assert error_status is False


# --- execute_workflow_request tests ---


@mock.patch("requests.post")
def test_execute_workflow_request_when_successful(
    post_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 200
    response._content = b'{"outputs": [{"predictions": [1, 2, 3]}]}'
    post_mock.return_value = response

    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"):
        error_status, message, result = execute_workflow_request(
            api_key="test_key",
            workspace_name="my-workspace",
            workflow_id="my-workflow",
            inputs={"image": {"type": "base64", "value": "data"}},
            request_timeout=30,
            call_chain="parent-workflow,my-workflow",
        )

    # then
    assert error_status is False
    assert message == "Workflow executed successfully"
    assert result == {"predictions": [1, 2, 3]}
    post_mock.assert_called_once_with(
        "http://127.0.0.1:9001/my-workspace/workflows/my-workflow",
        json={
            "inputs": {"image": {"type": "base64", "value": "data"}},
            "api_key": "test_key",
        },
        headers={WORKFLOW_CALL_CHAIN_HEADER: "parent-workflow,my-workflow"},
        timeout=30,
    )


@mock.patch("requests.post")
def test_execute_workflow_request_when_server_returns_error(
    post_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 500
    post_mock.return_value = response

    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"):
        error_status, message, result = execute_workflow_request(
            api_key="test_key",
            workspace_name="my-workspace",
            workflow_id="my-workflow",
            inputs={},
            request_timeout=30,
            call_chain="my-workflow",
        )

    # then
    assert error_status is True
    assert "Failed to execute workflow" in message
    assert result == {}


@mock.patch("requests.post")
def test_execute_workflow_request_when_connection_error(
    post_mock: MagicMock,
) -> None:
    # given
    post_mock.side_effect = ConnectionError("Connection refused")

    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"):
        error_status, message, result = execute_workflow_request(
            api_key="test_key",
            workspace_name="my-workspace",
            workflow_id="my-workflow",
            inputs={},
            request_timeout=30,
            call_chain="my-workflow",
        )

    # then
    assert error_status is True
    assert "Failed to execute workflow" in message
    assert result == {}


@mock.patch("requests.post")
def test_execute_workflow_request_with_empty_outputs(
    post_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 200
    response._content = b'{"outputs": []}'
    post_mock.return_value = response

    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"):
        error_status, message, result = execute_workflow_request(
            api_key="test_key",
            workspace_name="my-workspace",
            workflow_id="my-workflow",
            inputs={},
            request_timeout=30,
            call_chain="my-workflow",
        )

    # then
    assert error_status is False
    assert result == {}


@mock.patch("requests.post")
def test_execute_workflow_request_without_api_key(
    post_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 200
    response._content = b'{"outputs": [{"key": "value"}]}'
    post_mock.return_value = response

    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"):
        error_status, message, result = execute_workflow_request(
            api_key=None,
            workspace_name="my-workspace",
            workflow_id="my-workflow",
            inputs={"param": "value"},
            request_timeout=10,
            call_chain="my-workflow",
        )

    # then
    assert error_status is False
    # Verify api_key is not in the payload when None
    call_args = post_mock.call_args
    assert "api_key" not in call_args.kwargs["json"]


@mock.patch("requests.post")
def test_execute_workflow_request_passes_call_chain_header(
    post_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 200
    response._content = b'{"outputs": [{}]}'
    post_mock.return_value = response

    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"):
        execute_workflow_request(
            api_key="key",
            workspace_name="ws",
            workflow_id="wf",
            inputs={},
            request_timeout=30,
            call_chain="wf-a,wf-b,wf",
        )

    # then
    call_args = post_mock.call_args
    assert call_args.kwargs["headers"] == {
        WORKFLOW_CALL_CHAIN_HEADER: "wf-a,wf-b,wf",
    }


# --- Block run() tests ---


@mock.patch.object(v1, "call_workflow")
def test_block_run_successful_execution_without_resolution(
    call_workflow_mock: MagicMock,
) -> None:
    # given - no cache entry, falls back to "result" wrapper
    call_workflow_mock.return_value = (
        False,
        "Workflow executed successfully",
        {"predictions": [1, 2]},
    )
    block = WorkflowCallerBlockV1(api_key="test_key")
    block._workspace_name = "my-workspace"

    # when
    result = block.run(
        workflow_id="my-workflow",
        inputs={"param": "value"},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
    )

    # then
    assert result == {
        "result": {"predictions": [1, 2]},
    }
    call_workflow_mock.assert_called_once_with(
        api_key="test_key",
        workspace_name="my-workspace",
        workflow_id="my-workflow",
        inputs={"param": "value"},
        request_timeout=30,
        call_chain="",
        workflow_version_id=None,
    )


@mock.patch.object(v1, "_build_kinds_deserializers_map")
@mock.patch.object(v1, "call_workflow")
def test_block_run_returns_individual_outputs_when_resolved(
    call_workflow_mock: MagicMock,
    deserializers_mock: MagicMock,
) -> None:
    # given - cache has resolved outputs for this workflow_id
    _RESOLVED_WORKFLOW_OUTPUTS[("detection-wf", None)] = {
        "predictions": [STRING_KIND],
        "count": [STRING_KIND],
    }
    call_workflow_mock.return_value = (
        False,
        "Workflow executed successfully",
        {"predictions": "some_value", "count": "42"},
    )
    # Mock deserializers to verify they are called
    mock_deserializer = MagicMock(side_effect=lambda name, val: f"deserialized_{val}")
    deserializers_mock.return_value = {
        "string": mock_deserializer,
    }
    block = WorkflowCallerBlockV1(api_key="test_key")
    block._workspace_name = "my-workspace"

    # when
    result = block.run(
        workflow_id="detection-wf",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
    )

    # then - individual outputs with deserialization applied
    assert result["predictions"] == "deserialized_some_value"
    assert result["count"] == "deserialized_42"
    assert "result" not in result


@mock.patch.object(v1, "call_workflow")
def test_block_run_raises_on_workflow_error(
    call_workflow_mock: MagicMock,
) -> None:
    # given - cache has resolved outputs but the call fails
    _RESOLVED_WORKFLOW_OUTPUTS[("failing-wf", None)] = {
        "predictions": [OBJECT_DETECTION_PREDICTION_KIND],
    }
    call_workflow_mock.return_value = (
        True,
        "Timeout error",
        {},
    )
    block = WorkflowCallerBlockV1(api_key="test_key")
    block._workspace_name = "my-workspace"

    # when / then
    with pytest.raises(RuntimeError, match="Timeout error"):
        block.run(
            workflow_id="failing-wf",
            inputs={},
            request_timeout=5,
            input_definitions={},
            output_definitions={},
        )


def test_block_run_when_workspace_cannot_be_resolved() -> None:
    # given
    block = WorkflowCallerBlockV1(api_key=None)

    # when / then
    with pytest.raises(RuntimeError, match="Could not resolve workspace"):
        block.run(
            workflow_id="my-workflow",
            inputs={},
            request_timeout=30,
            input_definitions={},
            output_definitions={},
        )


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "call_workflow")
def test_block_run_resolves_workspace_from_api_key(
    call_workflow_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given
    get_workspace_mock.return_value = "resolved-workspace"
    call_workflow_mock.return_value = (False, "ok", {"key": "value"})
    block = WorkflowCallerBlockV1(api_key="test_api_key")

    # when
    block.run(
        workflow_id="my-workflow",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
    )

    # then
    get_workspace_mock.assert_called_once_with(api_key="test_api_key")
    call_workflow_mock.assert_called_once_with(
        api_key="test_api_key",
        workspace_name="resolved-workspace",
        workflow_id="my-workflow",
        inputs={},
        request_timeout=30,
        call_chain="",
        workflow_version_id=None,
    )


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "call_workflow")
def test_block_run_caches_workspace_resolution(
    call_workflow_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given
    get_workspace_mock.return_value = "resolved-workspace"
    call_workflow_mock.return_value = (False, "ok", {})
    block = WorkflowCallerBlockV1(api_key="test_api_key")

    # when - call run() twice
    block.run(
        workflow_id="workflow-1",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
    )
    block.run(
        workflow_id="workflow-2",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
    )

    # then - workspace is resolved only once
    get_workspace_mock.assert_called_once()


@mock.patch.object(v1, "get_roboflow_workspace")
def test_block_run_when_workspace_resolution_fails(
    get_workspace_mock: MagicMock,
) -> None:
    # given
    get_workspace_mock.side_effect = Exception("API error")
    block = WorkflowCallerBlockV1(api_key="test_api_key")

    # when / then
    with pytest.raises(RuntimeError, match="Could not resolve workspace"):
        block.run(
            workflow_id="my-workflow",
            inputs={},
            request_timeout=30,
            input_definitions={},
            output_definitions={},
        )


def test_block_get_init_parameters() -> None:
    assert WorkflowCallerBlockV1.get_init_parameters() == ["api_key", "workflow_call_chain"]


def test_block_get_manifest() -> None:
    assert WorkflowCallerBlockV1.get_manifest() is BlockManifest


# --- Output resolution tests ---


def test_extract_output_names_with_wildcard() -> None:
    # given
    spec = {
        "outputs": [
            {"type": "JsonField", "name": "predictions", "selector": "$steps.model.predictions"},
            {"type": "JsonField", "name": "image", "selector": "$steps.viz.image"},
        ]
    }

    # when
    result = _extract_output_names_with_wildcard(spec=spec)

    # then
    assert set(result.keys()) == {"predictions", "image"}
    assert result["predictions"] == [WILDCARD_KIND]
    assert result["image"] == [WILDCARD_KIND]


def test_extract_output_names_with_wildcard_empty_outputs() -> None:
    # given
    spec = {"outputs": []}

    # when
    result = _extract_output_names_with_wildcard(spec=spec)

    # then
    assert result == {}


def test_convert_output_descriptions_to_kinds_simple_kinds() -> None:
    # given
    descriptions = {
        "predictions": ["object_detection_prediction"],
        "status": ["boolean"],
    }

    # when
    result = _convert_output_descriptions_to_kinds(
        outputs_description=descriptions,
    )

    # then
    assert result["predictions"] == [OBJECT_DETECTION_PREDICTION_KIND]
    assert result["status"] == [BOOLEAN_KIND]


def test_convert_output_descriptions_to_kinds_wildcard_selector() -> None:
    # given - wildcard output produces a dict of property -> kinds
    descriptions = {
        "all_data": {
            "predictions": ["object_detection_prediction"],
            "image": ["image"],
        },
    }

    # when
    result = _convert_output_descriptions_to_kinds(
        outputs_description=descriptions,
    )

    # then - wildcard outputs become DICTIONARY_KIND
    assert result["all_data"] == [DICTIONARY_KIND]


def test_convert_output_descriptions_to_kinds_unknown_kind() -> None:
    # given - kind name that doesn't exist in the registry
    descriptions = {
        "custom_output": ["nonexistent_kind_xyz"],
    }

    # when
    result = _convert_output_descriptions_to_kinds(
        outputs_description=descriptions,
    )

    # then - falls back to WILDCARD_KIND
    assert result["custom_output"] == [WILDCARD_KIND]


def test_build_kind_name_map() -> None:
    # when
    kind_map = _build_kind_name_map()

    # then - should contain standard kinds
    assert "image" in kind_map
    assert kind_map["image"] == IMAGE_KIND
    assert "object_detection_prediction" in kind_map
    assert kind_map["object_detection_prediction"] == OBJECT_DETECTION_PREDICTION_KIND
    assert "boolean" in kind_map
    assert kind_map["boolean"] == BOOLEAN_KIND


@mock.patch.object(v1, "_describe_outputs_from_spec")
def test_resolve_and_cache_workflow_outputs_with_successful_resolution(
    describe_mock: MagicMock,
) -> None:
    # given
    describe_mock.return_value = {
        "predictions": ["object_detection_prediction"],
        "visualization": ["image"],
    }

    # when
    _resolve_and_cache_workflow_outputs(
        cache_key=("test-wf", None),
        spec={"outputs": [], "steps": []},
    )

    # then
    assert ("test-wf", None) in _RESOLVED_WORKFLOW_OUTPUTS
    resolved = _RESOLVED_WORKFLOW_OUTPUTS[("test-wf", None)]
    assert resolved["predictions"] == [OBJECT_DETECTION_PREDICTION_KIND]
    assert resolved["visualization"] == [IMAGE_KIND]


def test_resolve_and_cache_workflow_outputs_fallback_on_error() -> None:
    # given - spec with outputs but describe_workflow_outputs will fail
    # because the spec doesn't have valid steps/block types
    spec = {
        "outputs": [
            {"type": "JsonField", "name": "result_a", "selector": "$steps.fake_step.output"},
        ],
        "steps": [
            {"type": "nonexistent_block_type@v99", "name": "fake_step"},
        ],
    }

    # when - should fall back to extracting output names with WILDCARD_KIND
    _resolve_and_cache_workflow_outputs(
        cache_key=("fallback-wf", None),
        spec=spec,
    )

    # then
    assert ("fallback-wf", None) in _RESOLVED_WORKFLOW_OUTPUTS
    resolved = _RESOLVED_WORKFLOW_OUTPUTS[("fallback-wf", None)]
    assert resolved["result_a"] == [WILDCARD_KIND]


def test_resolve_and_cache_skips_empty_outputs() -> None:
    # given - spec with no outputs
    spec = {"outputs": [], "steps": []}

    # when
    _resolve_and_cache_workflow_outputs(
        cache_key=("empty-wf", None),
        spec=spec,
    )

    # then - nothing cached (empty dict is falsy)
    assert ("empty-wf", None) not in _RESOLVED_WORKFLOW_OUTPUTS


# --- Compile-time circular dependency validation tests ---


def _make_manifest(
    workflow_id: str,
    name: str = "step_1",
    inputs: dict = None,
    workflow_version_id: str = None,
) -> BlockManifest:
    data = {
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": name,
        "workflow_id": workflow_id,
    }
    if inputs is not None:
        data["inputs"] = inputs
    if workflow_version_id is not None:
        data["workflow_version_id"] = workflow_version_id
    return BlockManifest.model_validate(data)


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "get_workflow_specification")
def test_compile_time_validation_detects_direct_cycle(
    get_spec_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given - workflow-b calls itself (self-reference)
    get_workspace_mock.return_value = "my-workspace"
    get_spec_mock.return_value = {
        "inputs": [],
        "steps": [
            {
                "type": WORKFLOW_CALLER_BLOCK_TYPE,
                "name": "call_self",
                "workflow_id": "workflow-b",
            }
        ],
        "outputs": [],
    }
    steps = [_make_manifest(workflow_id="workflow-b")]

    # when / then
    with pytest.raises(ExecutionGraphStructureError, match="Circular workflow call"):
        validate_workflow_caller_no_circular_references(
            steps=steps,
            api_key="test_key",
        )


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "get_workflow_specification")
def test_compile_time_validation_detects_indirect_cycle(
    get_spec_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given - chain: parent calls A, A calls B, B calls A (cycle)
    get_workspace_mock.return_value = "my-workspace"

    def spec_side_effect(api_key, workspace_id, workflow_id, **kwargs):
        if workflow_id == "workflow-a":
            return {
                "inputs": [],
                "steps": [
                    {
                        "type": WORKFLOW_CALLER_BLOCK_TYPE,
                        "name": "call_b",
                        "workflow_id": "workflow-b",
                    }
                ],
                "outputs": [],
            }
        if workflow_id == "workflow-b":
            return {
                "inputs": [],
                "steps": [
                    {
                        "type": WORKFLOW_CALLER_BLOCK_TYPE,
                        "name": "call_a",
                        "workflow_id": "workflow-a",
                    }
                ],
                "outputs": [],
            }
        return {"inputs": [], "steps": [], "outputs": []}

    get_spec_mock.side_effect = spec_side_effect
    steps = [_make_manifest(workflow_id="workflow-a")]

    # when / then
    with pytest.raises(ExecutionGraphStructureError, match="Circular workflow call"):
        validate_workflow_caller_no_circular_references(
            steps=steps,
            api_key="test_key",
        )


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "get_workflow_specification")
def test_compile_time_validation_detects_indirect_cycle_with_version_ids(
    get_spec_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given - A@v1 calls B@v2, B@v2 calls A (cycle); version_ids must be
    # propagated through recursive calls so the correct specs are fetched.
    get_workspace_mock.return_value = "my-workspace"

    def spec_side_effect(api_key, workspace_id, workflow_id, **kwargs):
        version = kwargs.get("workflow_version_id")
        if workflow_id == "workflow-a" and version == "v1":
            return {
                "inputs": [],
                "steps": [
                    {
                        "type": WORKFLOW_CALLER_BLOCK_TYPE,
                        "name": "call_b",
                        "workflow_id": "workflow-b",
                        "workflow_version_id": "v2",
                    }
                ],
                "outputs": [],
            }
        if workflow_id == "workflow-b" and version == "v2":
            return {
                "inputs": [],
                "steps": [
                    {
                        "type": WORKFLOW_CALLER_BLOCK_TYPE,
                        "name": "call_a",
                        "workflow_id": "workflow-a",
                    }
                ],
                "outputs": [],
            }
        return {"inputs": [], "steps": [], "outputs": []}

    get_spec_mock.side_effect = spec_side_effect
    steps = [_make_manifest(workflow_id="workflow-a", workflow_version_id="v1")]

    # when / then
    with pytest.raises(ExecutionGraphStructureError, match="Circular workflow call"):
        validate_workflow_caller_no_circular_references(
            steps=steps,
            api_key="test_key",
        )

    # Verify version_ids were passed through recursive calls
    calls = get_spec_mock.call_args_list
    # First call: workflow-a@v1 (prefetched, so this is the initial fetch)
    assert calls[0] == mock.call(
        api_key="test_key",
        workspace_id="my-workspace",
        workflow_id="workflow-a",
        workflow_version_id="v1",
    )
    # Second call: workflow-b@v2 (recursive, version_id propagated)
    assert calls[1] == mock.call(
        api_key="test_key",
        workspace_id="my-workspace",
        workflow_id="workflow-b",
        workflow_version_id="v2",
    )


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "get_workflow_specification")
def test_compile_time_validation_allows_non_circular_chain(
    get_spec_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given - parent calls A, A calls B, B has no workflow_caller steps
    get_workspace_mock.return_value = "my-workspace"

    def spec_side_effect(api_key, workspace_id, workflow_id, **kwargs):
        if workflow_id == "workflow-a":
            return {
                "inputs": [],
                "steps": [
                    {
                        "type": WORKFLOW_CALLER_BLOCK_TYPE,
                        "name": "call_b",
                        "workflow_id": "workflow-b",
                    }
                ],
                "outputs": [],
            }
        if workflow_id == "workflow-b":
            return {
                "inputs": [],
                "steps": [
                    {"type": "roboflow_core/some_other_block@v1", "name": "some_step"}
                ],
                "outputs": [],
            }
        return {"inputs": [], "steps": [], "outputs": []}

    get_spec_mock.side_effect = spec_side_effect
    steps = [_make_manifest(workflow_id="workflow-a")]

    # when / then - should not raise
    validate_workflow_caller_no_circular_references(
        steps=steps,
        api_key="test_key",
    )


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "get_workflow_specification")
def test_compile_time_validation_resolves_outputs(
    get_spec_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given - workflow with outputs defined in its spec
    get_workspace_mock.return_value = "my-workspace"
    get_spec_mock.return_value = {
        "inputs": [],
        "steps": [
            {"type": "roboflow_core/some_block@v1", "name": "detector"}
        ],
        "outputs": [
            {"type": "JsonField", "name": "detections", "selector": "$steps.detector.predictions"},
        ],
    }
    steps = [_make_manifest(workflow_id="target-wf")]

    # when
    validate_workflow_caller_no_circular_references(
        steps=steps,
        api_key="test_key",
    )

    # then - outputs should be cached (at minimum with WILDCARD_KIND fallback)
    assert ("target-wf", None) in _RESOLVED_WORKFLOW_OUTPUTS
    assert "detections" in _RESOLVED_WORKFLOW_OUTPUTS[("target-wf", None)]
    # inputs should also be cached
    assert ("target-wf", None) in _RESOLVED_WORKFLOW_INPUTS


def test_compile_time_validation_skips_selector_workflow_ids() -> None:
    # given - workflow_id is a selector, not a static string
    steps = [_make_manifest(workflow_id="$inputs.workflow_id")]

    # when / then - should not raise (can't validate dynamic IDs)
    validate_workflow_caller_no_circular_references(
        steps=steps,
        api_key="test_key",
    )


def test_compile_time_validation_skips_when_no_api_key() -> None:
    # given
    steps = [_make_manifest(workflow_id="my-workflow")]

    # when / then - should not raise (can't resolve workspace without API key)
    validate_workflow_caller_no_circular_references(
        steps=steps,
        api_key=None,
    )


@mock.patch.object(v1, "get_roboflow_workspace")
def test_compile_time_validation_skips_when_workspace_resolution_fails(
    get_workspace_mock: MagicMock,
) -> None:
    # given
    get_workspace_mock.side_effect = Exception("API error")
    steps = [_make_manifest(workflow_id="my-workflow")]

    # when / then - should not raise
    validate_workflow_caller_no_circular_references(
        steps=steps,
        api_key="test_key",
    )


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "get_workflow_specification")
def test_compile_time_validation_skips_when_spec_fetch_fails(
    get_spec_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given - spec fetch fails for target workflow
    get_workspace_mock.return_value = "my-workspace"
    get_spec_mock.side_effect = Exception("Not found")
    steps = [_make_manifest(workflow_id="missing-workflow")]

    # when / then - should not raise (falls back to runtime check)
    validate_workflow_caller_no_circular_references(
        steps=steps,
        api_key="test_key",
    )


def test_compile_time_validation_skips_non_workflow_caller_steps() -> None:
    # given - step with a different type
    step = MagicMock()
    step.type = "roboflow_core/some_other_block@v1"

    # when / then - should not raise
    validate_workflow_caller_no_circular_references(
        steps=[step],
        api_key="test_key",
    )


# --- Output deserialization tests ---


def test_deserialize_output_value_returns_none_for_none() -> None:
    # when
    result = _deserialize_output_value(
        output_name="test",
        raw_value=None,
        output_kinds=[STRING_KIND],
        deserializers={"string": lambda name, val: val},
    )

    # then
    assert result is None


def test_deserialize_output_value_applies_matching_deserializer() -> None:
    # given
    deserializers = {
        "string": lambda name, val: f"deserialized:{val}",
    }

    # when
    result = _deserialize_output_value(
        output_name="my_output",
        raw_value="hello",
        output_kinds=[STRING_KIND],
        deserializers=deserializers,
    )

    # then
    assert result == "deserialized:hello"


def test_deserialize_output_value_returns_raw_for_wildcard_kind() -> None:
    # given - WILDCARD_KIND outputs are passed through as-is
    deserializers = {"string": lambda name, val: f"deserialized:{val}"}

    # when
    result = _deserialize_output_value(
        output_name="my_output",
        raw_value={"key": "value"},
        output_kinds=[WILDCARD_KIND],
        deserializers=deserializers,
    )

    # then - returns raw value unchanged
    assert result == {"key": "value"}


def test_deserialize_output_value_raises_on_failure_for_known_kinds() -> None:
    # given - deserializer raises an error for a known kind
    def failing_deserializer(name, val):
        raise ValueError("Cannot deserialize")

    deserializers = {
        "object_detection_prediction": failing_deserializer,
    }

    # when / then - should raise, not silently return raw value
    with pytest.raises(RuntimeError, match="Failed to deserialize output"):
        _deserialize_output_value(
            output_name="predictions",
            raw_value={"raw": "data"},
            output_kinds=[OBJECT_DETECTION_PREDICTION_KIND],
            deserializers=deserializers,
        )


def test_deserialize_output_value_tries_kinds_in_order() -> None:
    # given - first kind's deserializer fails, second succeeds
    def failing_deserializer(name, val):
        raise ValueError("Nope")

    def working_deserializer(name, val):
        return f"worked:{val}"

    deserializers = {
        "object_detection_prediction": failing_deserializer,
        "string": working_deserializer,
    }

    # when
    result = _deserialize_output_value(
        output_name="output",
        raw_value="test",
        output_kinds=[OBJECT_DETECTION_PREDICTION_KIND, STRING_KIND],
        deserializers=deserializers,
    )

    # then - second deserializer used
    assert result == "worked:test"


def test_deserialize_output_value_returns_raw_when_no_deserializer_exists() -> None:
    # given - kind exists but has no registered deserializer
    deserializers = {}

    # when
    result = _deserialize_output_value(
        output_name="my_output",
        raw_value="hello",
        output_kinds=[STRING_KIND],
        deserializers=deserializers,
    )

    # then - no deserializer found, returns raw (no error since none was attempted)
    assert result == "hello"


def test_build_kinds_deserializers_map() -> None:
    # when
    deserializers = _build_kinds_deserializers_map()

    # then - should contain standard deserializers
    assert "image" in deserializers
    assert "object_detection_prediction" in deserializers
    assert "string" in deserializers
    assert callable(deserializers["image"])


# --- Input resolution tests ---


def test_resolve_and_cache_workflow_inputs_with_image() -> None:
    # given
    spec = {
        "inputs": [
            {"type": "WorkflowImage", "name": "my_photo"},
            {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.5},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "outputs": [],
    }

    # when
    _resolve_and_cache_workflow_inputs(cache_key=("test-wf", None), spec=spec)

    # then
    assert ("test-wf", None) in _RESOLVED_WORKFLOW_INPUTS
    inputs = _RESOLVED_WORKFLOW_INPUTS[("test-wf", None)]
    assert len(inputs) == 3

    image_input = inputs[0]
    assert image_input["name"] == "my_photo"
    assert image_input["has_default"] is False

    conf_input = inputs[1]
    assert conf_input["name"] == "confidence"
    assert conf_input["has_default"] is True

    model_input = inputs[2]
    assert model_input["name"] == "model_id"
    assert model_input["has_default"] is False


def test_resolve_and_cache_workflow_inputs_empty() -> None:
    # given
    spec = {"inputs": [], "outputs": []}

    # when
    _resolve_and_cache_workflow_inputs(cache_key=("empty-wf", None), spec=spec)

    # then
    assert ("empty-wf", None) in _RESOLVED_WORKFLOW_INPUTS
    assert _RESOLVED_WORKFLOW_INPUTS[("empty-wf", None)] == []


def test_resolve_and_cache_workflow_inputs_inference_image_type() -> None:
    # given - legacy InferenceImage type
    spec = {
        "inputs": [
            {"type": "InferenceImage", "name": "frame"},
        ],
        "outputs": [],
    }

    # when
    _resolve_and_cache_workflow_inputs(cache_key=("legacy-wf", None), spec=spec)

    # then
    inputs = _RESOLVED_WORKFLOW_INPUTS[("legacy-wf", None)]
    assert inputs[0]["name"] == "frame"


# --- Input validation tests ---


def test_validate_required_inputs_fails_when_required_param_missing() -> None:
    # given - inner workflow needs "model_id" with no default
    _RESOLVED_WORKFLOW_INPUTS[("wf-with-params", None)] = [
        {"name": "model_id", "type": "WorkflowParameter", "has_default": False},
    ]
    manifest = _make_manifest(workflow_id="wf-with-params")

    # when / then
    with pytest.raises(ExecutionGraphStructureError, match="requires input"):
        _validate_required_inputs(workflow_id="wf-with-params", step_manifest=manifest)


def test_validate_required_inputs_passes_when_param_provided() -> None:
    # given
    _RESOLVED_WORKFLOW_INPUTS[("wf-with-params", None)] = [
        {"name": "model_id", "type": "WorkflowParameter", "has_default": False},
    ]
    manifest = _make_manifest(
        workflow_id="wf-with-params",
        inputs={"model_id": "my-model/1"},
    )

    # when / then - should not raise
    _validate_required_inputs(workflow_id="wf-with-params", step_manifest=manifest)


def test_validate_required_inputs_skips_params_with_defaults() -> None:
    # given - param has default, not provided in inputs
    _RESOLVED_WORKFLOW_INPUTS[("wf-defaults", None)] = [
        {"name": "confidence", "type": "WorkflowParameter", "has_default": True},
    ]
    manifest = _make_manifest(workflow_id="wf-defaults")

    # when / then - should not raise (has default)
    _validate_required_inputs(workflow_id="wf-defaults", step_manifest=manifest)


def test_validate_required_inputs_skips_when_not_resolved() -> None:
    # given - no resolved inputs (dynamic workflow_id)
    manifest = _make_manifest(workflow_id="unknown-wf")

    # when / then - should not raise
    _validate_required_inputs(workflow_id="unknown-wf", step_manifest=manifest)


def test_validate_required_inputs_fails_when_image_input_missing() -> None:
    # given - inner workflow has an image input that is not provided in inputs
    _RESOLVED_WORKFLOW_INPUTS[("img-wf", None)] = [
        {"name": "image", "type": "WorkflowImage", "has_default": False},
    ]
    manifest = _make_manifest(workflow_id="img-wf")

    # when / then - should error about missing "image"
    with pytest.raises(ExecutionGraphStructureError, match="image"):
        _validate_required_inputs(workflow_id="img-wf", step_manifest=manifest)


def test_validate_required_inputs_passes_when_image_input_provided() -> None:
    # given - inner workflow has an image input, and it is provided in inputs
    _RESOLVED_WORKFLOW_INPUTS[("img-wf", None)] = [
        {"name": "image", "type": "WorkflowImage", "has_default": False},
    ]
    manifest = _make_manifest(
        workflow_id="img-wf",
        inputs={"image": "$inputs.image"},
    )

    # when / then - should not raise
    _validate_required_inputs(workflow_id="img-wf", step_manifest=manifest)


def test_validate_required_inputs_fails_when_multiple_images_not_all_covered() -> None:
    # given - inner workflow has 2 image inputs, only one is provided
    _RESOLVED_WORKFLOW_INPUTS[("multi-img-wf", None)] = [
        {"name": "image", "type": "WorkflowImage", "has_default": False},
        {"name": "mask", "type": "WorkflowImage", "has_default": False},
    ]
    manifest = _make_manifest(
        workflow_id="multi-img-wf",
        inputs={"image": "$inputs.image"},
    )

    # when / then - should error about missing "mask"
    with pytest.raises(ExecutionGraphStructureError, match="mask"):
        _validate_required_inputs(workflow_id="multi-img-wf", step_manifest=manifest)


def test_validate_required_inputs_passes_when_multiple_images_all_covered() -> None:
    # given - inner workflow has 2 image inputs, both provided via inputs
    _RESOLVED_WORKFLOW_INPUTS[("multi-img-wf", None)] = [
        {"name": "image", "type": "WorkflowImage", "has_default": False},
        {"name": "mask", "type": "WorkflowImage", "has_default": False},
    ]
    manifest = _make_manifest(
        workflow_id="multi-img-wf",
        inputs={"image": "$inputs.image", "mask": "$inputs.mask"},
    )

    # when / then - should not raise
    _validate_required_inputs(workflow_id="multi-img-wf", step_manifest=manifest)


# --- Compile-time validation: input integration tests ---


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "get_workflow_specification")
def test_compile_time_validation_fails_when_required_param_missing(
    get_spec_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given - inner workflow requires "model_id" with no default
    get_workspace_mock.return_value = "my-workspace"
    get_spec_mock.return_value = {
        "inputs": [
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [],
        "outputs": [],
    }
    steps = [_make_manifest(workflow_id="param-wf")]  # no inputs

    # when / then
    with pytest.raises(ExecutionGraphStructureError, match="requires input"):
        validate_workflow_caller_no_circular_references(
            steps=steps,
            api_key="test_key",
        )


@mock.patch.object(v1, "get_roboflow_workspace")
@mock.patch.object(v1, "get_workflow_specification")
def test_compile_time_validation_caches_resolved_inputs(
    get_spec_mock: MagicMock,
    get_workspace_mock: MagicMock,
) -> None:
    # given
    get_workspace_mock.return_value = "my-workspace"
    get_spec_mock.return_value = {
        "inputs": [
            {"type": "WorkflowImage", "name": "photo"},
            {"type": "WorkflowParameter", "name": "conf", "default_value": 0.5},
        ],
        "steps": [],
        "outputs": [],
    }
    steps = [_make_manifest(workflow_id="cached-wf", inputs={"photo": "$inputs.image"})]

    # when
    validate_workflow_caller_no_circular_references(
        steps=steps,
        api_key="test_key",
    )

    # then
    assert ("cached-wf", None) in _RESOLVED_WORKFLOW_INPUTS
    inputs = _RESOLVED_WORKFLOW_INPUTS[("cached-wf", None)]
    assert len(inputs) == 2
    assert inputs[0]["name"] == "photo"
    assert inputs[1]["name"] == "conf"
    assert inputs[1]["has_default"] is True


# --- User-defined output_definitions tests ---


def test_manifest_parsing_with_output_definitions() -> None:
    # given
    raw_manifest = {
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
        "output_definitions": {
            "predictions": ["object_detection_prediction"],
            "result_image": ["image"],
        },
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.output_definitions == {
        "predictions": ["object_detection_prediction"],
        "result_image": ["image"],
    }


@mock.patch.object(v1, "_build_kind_name_map")
def test_get_actual_outputs_with_user_defined_output_definitions(
    build_kind_name_map_mock: MagicMock,
) -> None:
    # given
    build_kind_name_map_mock.return_value = {
        "object_detection_prediction": OBJECT_DETECTION_PREDICTION_KIND,
        "image": IMAGE_KIND,
        "boolean": BOOLEAN_KIND,
        "string": STRING_KIND,
    }
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "some-workflow",
        "output_definitions": {
            "predictions": ["object_detection_prediction"],
            "viz": ["image"],
        },
    })

    # when
    outputs = manifest.get_actual_outputs()

    # then - should use output_definitions, not cache or fallback
    output_names = {o.name for o in outputs}
    assert output_names == {"predictions", "viz"}
    predictions_output = next(o for o in outputs if o.name == "predictions")
    assert predictions_output.kind == [OBJECT_DETECTION_PREDICTION_KIND]
    viz_output = next(o for o in outputs if o.name == "viz")
    assert viz_output.kind == [IMAGE_KIND]
    assert "result" not in output_names


@mock.patch.object(v1, "_build_kind_name_map")
def test_get_actual_outputs_user_defined_takes_precedence_over_cache(
    build_kind_name_map_mock: MagicMock,
) -> None:
    # given - cache has different outputs than what the user defined
    build_kind_name_map_mock.return_value = {
        "object_detection_prediction": OBJECT_DETECTION_PREDICTION_KIND,
        "image": IMAGE_KIND,
        "boolean": BOOLEAN_KIND,
        "string": STRING_KIND,
    }
    _RESOLVED_WORKFLOW_OUTPUTS[("cached-wf", None)] = {
        "cached_output": [STRING_KIND],
    }
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "cached-wf",
        "output_definitions": {
            "predictions": ["object_detection_prediction"],
        },
    })

    # when
    outputs = manifest.get_actual_outputs()

    # then - user-defined output_definitions take precedence over cache
    output_names = {o.name for o in outputs}
    assert "predictions" in output_names
    assert "cached_output" not in output_names


@mock.patch.object(v1, "_build_kind_name_map")
def test_get_actual_outputs_user_defined_unknown_kind_falls_back_to_wildcard(
    build_kind_name_map_mock: MagicMock,
) -> None:
    # given - kind map is empty, so all kinds are unknown
    build_kind_name_map_mock.return_value = {}
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "some-workflow",
        "output_definitions": {
            "predictions": ["nonexistent_kind"],
        },
    })

    # when
    outputs = manifest.get_actual_outputs()

    # then - unknown kind falls back to WILDCARD_KIND
    predictions_output = next(o for o in outputs if o.name == "predictions")
    assert predictions_output.kind == [WILDCARD_KIND]


@mock.patch.object(v1, "_build_kind_name_map")
def test_resolve_output_kinds_for_run_with_user_definitions(
    build_kind_name_map_mock: MagicMock,
) -> None:
    # given
    build_kind_name_map_mock.return_value = {
        "object_detection_prediction": OBJECT_DETECTION_PREDICTION_KIND,
        "image": IMAGE_KIND,
        "boolean": BOOLEAN_KIND,
        "string": STRING_KIND,
    }

    # when
    result = _resolve_output_kinds_for_run(
        output_definitions={
            "predictions": ["object_detection_prediction"],
            "viz": ["image"],
        },
        workflow_id="any-workflow",
    )

    # then
    assert result is not None
    assert result["predictions"] == [OBJECT_DETECTION_PREDICTION_KIND]
    assert result["viz"] == [IMAGE_KIND]


@mock.patch.object(v1, "_build_kind_name_map")
def test_resolve_output_kinds_for_run_falls_back_to_cache(
    build_kind_name_map_mock: MagicMock,
) -> None:
    # given - output_definitions is empty, but cache has data
    build_kind_name_map_mock.return_value = {
        "object_detection_prediction": OBJECT_DETECTION_PREDICTION_KIND,
        "image": IMAGE_KIND,
        "boolean": BOOLEAN_KIND,
        "string": STRING_KIND,
    }
    _RESOLVED_WORKFLOW_OUTPUTS[("cached-wf", None)] = {
        "predictions": [OBJECT_DETECTION_PREDICTION_KIND],
    }

    # when
    result = _resolve_output_kinds_for_run(
        output_definitions={},
        workflow_id="cached-wf",
    )

    # then - falls back to cache
    assert result is not None
    assert result["predictions"] == [OBJECT_DETECTION_PREDICTION_KIND]
    # _build_kind_name_map should not have been called since output_definitions is empty
    build_kind_name_map_mock.assert_not_called()


def test_resolve_output_kinds_for_run_returns_none_when_nothing() -> None:
    # given - both output_definitions and cache are empty

    # when
    result = _resolve_output_kinds_for_run(
        output_definitions={},
        workflow_id="unknown-wf",
    )

    # then
    assert result is None


@mock.patch.object(v1, "_build_kind_name_map")
@mock.patch.object(v1, "_build_kinds_deserializers_map")
@mock.patch.object(v1, "call_workflow")
def test_block_run_with_user_defined_output_definitions(
    call_workflow_mock: MagicMock,
    deserializers_mock: MagicMock,
    build_kind_name_map_mock: MagicMock,
) -> None:
    # given
    build_kind_name_map_mock.return_value = {
        "object_detection_prediction": OBJECT_DETECTION_PREDICTION_KIND,
        "image": IMAGE_KIND,
        "boolean": BOOLEAN_KIND,
        "string": STRING_KIND,
    }
    call_workflow_mock.return_value = (
        False,
        "Workflow executed successfully",
        {"predictions": "raw_pred", "viz": "raw_viz"},
    )
    mock_deserializer = MagicMock(side_effect=lambda name, val: f"deserialized_{val}")
    deserializers_mock.return_value = {
        "object_detection_prediction": mock_deserializer,
        "image": mock_deserializer,
    }
    block = WorkflowCallerBlockV1(api_key="test_key")
    block._workspace_name = "my-workspace"

    # when
    result = block.run(
        workflow_id="my-workflow",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={
            "predictions": ["object_detection_prediction"],
            "viz": ["image"],
        },
    )

    # then - output_definitions used for deserialization
    assert result["predictions"] == "deserialized_raw_pred"
    assert result["viz"] == "deserialized_raw_viz"
    assert "result" not in result


# --- input_definitions tests ---


def test_manifest_parsing_with_input_definitions() -> None:
    # given
    raw_manifest = {
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
        "input_definitions": {
            "confidence_threshold": ["float"],
            "model_id": ["roboflow_model_id"],
        },
        "inputs": {
            "confidence_threshold": 0.5,
            "model_id": "my-model/1",
        },
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.input_definitions == {
        "confidence_threshold": ["float"],
        "model_id": ["roboflow_model_id"],
    }


def test_manifest_parsing_with_empty_input_definitions() -> None:
    # given
    raw_manifest = {
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then - defaults to empty dict
    assert result.input_definitions == {}


def test_manifest_parsing_with_both_input_and_output_definitions() -> None:
    # given
    raw_manifest = {
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
        "input_definitions": {
            "threshold": ["float"],
        },
        "output_definitions": {
            "predictions": ["object_detection_prediction"],
        },
        "inputs": {
            "threshold": 0.5,
        },
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.input_definitions == {"threshold": ["float"]}
    assert result.output_definitions == {"predictions": ["object_detection_prediction"]}


def test_validate_required_inputs_passes_when_input_definitions_covered() -> None:
    # given - input_definitions declares inputs, all present in inputs
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-wf",
        "input_definitions": {
            "threshold": ["float"],
            "model_id": ["string"],
        },
        "inputs": {
            "threshold": 0.5,
            "model_id": "my-model/1",
        },
    })

    # when / then - should not raise
    _validate_required_inputs(workflow_id="my-wf", step_manifest=manifest)


def test_validate_required_inputs_fails_when_input_definitions_not_covered() -> None:
    # given - input_definitions declares "threshold" but it's not in inputs
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-wf",
        "input_definitions": {
            "threshold": ["float"],
            "model_id": ["string"],
        },
        "inputs": {
            "model_id": "my-model/1",
        },
    })

    # when / then
    with pytest.raises(ExecutionGraphStructureError, match="threshold"):
        _validate_required_inputs(workflow_id="my-wf", step_manifest=manifest)


def test_validate_required_inputs_input_definitions_checked_even_without_spec() -> None:
    # given - no resolved inputs (spec fetch failed), but input_definitions is set
    # with a missing key in inputs
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "unknown-wf",
        "input_definitions": {
            "threshold": ["float"],
        },
        "inputs": {},
    })

    # when / then - should still catch the missing input from input_definitions
    with pytest.raises(ExecutionGraphStructureError, match="threshold"):
        _validate_required_inputs(workflow_id="unknown-wf", step_manifest=manifest)


def test_validate_required_inputs_empty_input_definitions_skips_check() -> None:
    # given - empty input_definitions should skip the check
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "unknown-wf",
        "inputs": {},
    })

    # when / then - should not raise (no input_definitions, no resolved inputs)
    _validate_required_inputs(workflow_id="unknown-wf", step_manifest=manifest)


@mock.patch.object(v1, "_build_kind_name_map")
@mock.patch.object(v1, "_build_kinds_deserializers_map")
@mock.patch.object(v1, "call_workflow")
def test_block_run_with_input_definitions(
    call_workflow_mock: MagicMock,
    deserializers_mock: MagicMock,
    build_kind_name_map_mock: MagicMock,
) -> None:
    # given - input_definitions is passed through to run() but doesn't affect
    # runtime behavior (it's UI/validation metadata only)
    build_kind_name_map_mock.return_value = {
        "object_detection_prediction": OBJECT_DETECTION_PREDICTION_KIND,
        "image": IMAGE_KIND,
        "boolean": BOOLEAN_KIND,
        "string": STRING_KIND,
    }
    call_workflow_mock.return_value = (
        False,
        "Workflow executed successfully",
        {"predictions": "raw_pred"},
    )
    mock_deserializer = MagicMock(side_effect=lambda name, val: f"deserialized_{val}")
    deserializers_mock.return_value = {
        "object_detection_prediction": mock_deserializer,
    }
    block = WorkflowCallerBlockV1(api_key="test_key")
    block._workspace_name = "my-workspace"

    # when
    result = block.run(
        workflow_id="my-workflow",
        inputs={"threshold": 0.5},
        request_timeout=30,
        input_definitions={"threshold": ["float"]},
        output_definitions={"predictions": ["object_detection_prediction"]},
    )

    # then - run() works with both input_definitions and output_definitions
    assert result["predictions"] == "deserialized_raw_pred"


# --- workflow_version_id tests ---


def test_manifest_parsing_with_workflow_version_id() -> None:
    # given
    raw_manifest = {
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
        "workflow_version_id": "1709234567890",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.workflow_version_id == "1709234567890"


def test_manifest_parsing_with_workflow_version_id_as_selector() -> None:
    # given
    raw_manifest = {
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
        "workflow_version_id": "$inputs.version_id",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.workflow_version_id == "$inputs.version_id"


def test_manifest_parsing_without_workflow_version_id_defaults_to_none() -> None:
    # given
    raw_manifest = {
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.workflow_version_id is None


@mock.patch.object(v1, "call_workflow")
def test_block_run_passes_workflow_version_id(
    call_workflow_mock: MagicMock,
) -> None:
    # given
    call_workflow_mock.return_value = (
        False,
        "Workflow executed successfully",
        {"predictions": [1, 2]},
    )
    block = WorkflowCallerBlockV1(api_key="test_key")
    block._workspace_name = "my-workspace"

    # when
    result = block.run(
        workflow_id="my-workflow",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
        workflow_version_id="1709234567890",
    )

    # then - version_id is passed through to call_workflow
    call_workflow_mock.assert_called_once_with(
        api_key="test_key",
        workspace_name="my-workspace",
        workflow_id="my-workflow",
        inputs={},
        request_timeout=30,
        call_chain="",
        workflow_version_id="1709234567890",
    )
    assert result == {"result": {"predictions": [1, 2]}}


@mock.patch.object(v1, "call_workflow")
def test_block_run_passes_none_version_when_not_specified(
    call_workflow_mock: MagicMock,
) -> None:
    # given
    call_workflow_mock.return_value = (False, "ok", {"data": 1})
    block = WorkflowCallerBlockV1(api_key="test_key")
    block._workspace_name = "my-workspace"

    # when
    block.run(
        workflow_id="my-workflow",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
    )

    # then - version_id defaults to None
    call_workflow_mock.assert_called_once_with(
        api_key="test_key",
        workspace_name="my-workspace",
        workflow_id="my-workflow",
        inputs={},
        request_timeout=30,
        call_chain="",
        workflow_version_id=None,
    )


@mock.patch("requests.post")
def test_call_workflow_passes_version_id_to_execute(
    mock_post: MagicMock,
) -> None:
    # given
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "outputs": [{"predictions": [1]}],
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    # when
    error_status, message, result = call_workflow(
        api_key="test_key",
        workspace_name="ws",
        workflow_id="wf",
        inputs={},
        request_timeout=30,
        workflow_version_id="1709234567890",
    )

    # then - version_id included in POST payload
    assert error_status is False
    call_args = mock_post.call_args
    payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
    assert payload["workflow_version_id"] == "1709234567890"


@mock.patch("requests.post")
def test_call_workflow_omits_version_id_when_none(
    mock_post: MagicMock,
) -> None:
    # given
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "outputs": [{"predictions": [1]}],
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    # when
    call_workflow(
        api_key="test_key",
        workspace_name="ws",
        workflow_id="wf",
        inputs={},
        request_timeout=30,
    )

    # then - no workflow_version_id in payload
    call_args = mock_post.call_args
    payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
    assert "workflow_version_id" not in payload


@mock.patch("requests.post")
def test_execute_workflow_request_includes_version_in_payload(
    mock_post: MagicMock,
) -> None:
    # given
    mock_response = MagicMock()
    mock_response.json.return_value = {"outputs": [{"data": "ok"}]}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    # when
    execute_workflow_request(
        api_key="key",
        workspace_name="ws",
        workflow_id="wf",
        inputs={"image": {"type": "base64", "value": "abc"}},
        request_timeout=30,
        call_chain="parent",
        workflow_version_id="1709234567890",
    )

    # then
    call_args = mock_post.call_args
    payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
    assert payload["workflow_version_id"] == "1709234567890"
    assert payload["api_key"] == "key"


@mock.patch("requests.post")
def test_execute_workflow_request_omits_version_when_none(
    mock_post: MagicMock,
) -> None:
    # given
    mock_response = MagicMock()
    mock_response.json.return_value = {"outputs": [{"data": "ok"}]}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    # when
    execute_workflow_request(
        api_key="key",
        workspace_name="ws",
        workflow_id="wf",
        inputs={"image": {"type": "base64", "value": "abc"}},
        request_timeout=30,
        call_chain="parent",
    )

    # then
    call_args = mock_post.call_args
    payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
    assert "workflow_version_id" not in payload


def test_extract_workflow_caller_steps_includes_version_id() -> None:
    # given
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
        "workflow_version_id": "1709234567890",
    })

    # when
    result = _extract_workflow_caller_steps(steps=[manifest])

    # then - returns 3-tuple with version_id
    assert len(result) == 1
    workflow_id, version_id, step_manifest = result[0]
    assert workflow_id == "my-workflow"
    assert version_id == "1709234567890"
    assert step_manifest is manifest


def test_extract_workflow_caller_steps_version_none_when_not_set() -> None:
    # given
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
    })

    # when
    result = _extract_workflow_caller_steps(steps=[manifest])

    # then
    assert len(result) == 1
    workflow_id, version_id, step_manifest = result[0]
    assert workflow_id == "my-workflow"
    assert version_id is None


def test_extract_workflow_caller_steps_version_none_when_selector() -> None:
    # given - version_id is a selector, can't use at compile time
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
        "workflow_version_id": "$inputs.version",
    })

    # when
    result = _extract_workflow_caller_steps(steps=[manifest])

    # then - version_id should be None since it's a selector
    assert len(result) == 1
    _, version_id, _ = result[0]
    assert version_id is None


@mock.patch.object(v1, "get_workflow_specification")
@mock.patch.object(v1, "get_roboflow_workspace")
def test_compile_time_validation_passes_version_to_spec_fetch(
    get_workspace_mock: MagicMock,
    get_spec_mock: MagicMock,
) -> None:
    # given
    get_workspace_mock.return_value = "my-workspace"
    get_spec_mock.return_value = {
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [],
        "outputs": [{"type": "JsonField", "name": "predictions", "selector": "$steps.m.predictions"}],
    }
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-workflow",
        "workflow_version_id": "1709234567890",
        "inputs": {"image": "$inputs.image"},
    })

    # when
    validate_workflow_caller_no_circular_references(
        steps=[manifest],
        api_key="test_key",
    )

    # then - get_workflow_specification called with the version_id
    get_spec_mock.assert_any_call(
        api_key="test_key",
        workspace_id="my-workspace",
        workflow_id="my-workflow",
        workflow_version_id="1709234567890",
    )


# --- _make_cache_key tests ---


def test_make_cache_key_with_version() -> None:
    assert _make_cache_key("wf-id", "v1") == ("wf-id", "v1")


def test_make_cache_key_without_version() -> None:
    assert _make_cache_key("wf-id") == ("wf-id", None)


def test_make_cache_key_different_versions_produce_different_keys() -> None:
    key_a = _make_cache_key("my-wf", "version-a")
    key_b = _make_cache_key("my-wf", "version-b")
    key_none = _make_cache_key("my-wf", None)
    assert key_a != key_b
    assert key_a != key_none
    assert key_b != key_none


# --- Call chain forwarding tests ---


@mock.patch.object(v1, "call_workflow")
def test_block_run_forwards_call_chain_from_init(
    call_workflow_mock: MagicMock,
) -> None:
    # given - block was initialised with a call chain from an incoming request
    call_workflow_mock.return_value = (False, "ok", {"key": "value"})
    block = WorkflowCallerBlockV1(
        api_key="test_key",
        workflow_call_chain="parent-wf,grandparent-wf",
    )
    block._workspace_name = "my-workspace"

    # when
    block.run(
        workflow_id="child-wf",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
    )

    # then - the existing call chain is forwarded
    call_workflow_mock.assert_called_once_with(
        api_key="test_key",
        workspace_name="my-workspace",
        workflow_id="child-wf",
        inputs={},
        request_timeout=30,
        call_chain="parent-wf,grandparent-wf",
        workflow_version_id=None,
    )


@mock.patch.object(v1, "call_workflow")
def test_block_run_forwards_empty_chain_when_none(
    call_workflow_mock: MagicMock,
) -> None:
    # given - block initialised without call chain
    call_workflow_mock.return_value = (False, "ok", {})
    block = WorkflowCallerBlockV1(api_key="key", workflow_call_chain=None)
    block._workspace_name = "ws"

    # when
    block.run(
        workflow_id="wf",
        inputs={},
        request_timeout=30,
        input_definitions={},
        output_definitions={},
    )

    # then - empty string forwarded
    call_args = call_workflow_mock.call_args
    assert call_args.kwargs["call_chain"] == ""


# --- Cache key versioning in get_actual_outputs tests ---


def test_get_actual_outputs_with_versioned_cache() -> None:
    # given - cache has entry for specific version
    _RESOLVED_WORKFLOW_OUTPUTS[("my-wf", "v42")] = {
        "predictions": [OBJECT_DETECTION_PREDICTION_KIND],
    }
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-wf",
        "workflow_version_id": "v42",
    })

    # when
    outputs = manifest.get_actual_outputs()

    # then - should use the versioned cache entry
    output_names = {o.name for o in outputs}
    assert "predictions" in output_names
    assert "result" not in output_names


def test_get_actual_outputs_version_mismatch_falls_back() -> None:
    # given - cache has entry for a different version
    _RESOLVED_WORKFLOW_OUTPUTS[("my-wf", "v42")] = {
        "predictions": [OBJECT_DETECTION_PREDICTION_KIND],
    }
    manifest = BlockManifest.model_validate({
        "type": WORKFLOW_CALLER_BLOCK_TYPE,
        "name": "caller",
        "workflow_id": "my-wf",
        "workflow_version_id": "v99",
    })

    # when
    outputs = manifest.get_actual_outputs()

    # then - no cache match, should fall back to "result"
    output_names = {o.name for o in outputs}
    assert output_names == {"result"}


# --- _check_workflow_for_circular_references with version_id ---


@mock.patch.object(v1, "get_workflow_specification")
def test_check_circular_references_passes_version_id(
    get_spec_mock: MagicMock,
) -> None:
    # given - no nested workflow callers
    get_spec_mock.return_value = {
        "inputs": [],
        "steps": [],
        "outputs": [],
    }

    # when
    from inference.core.workflows.core_steps.composition.workflow_caller.v1 import (
        _check_workflow_for_circular_references,
    )
    _check_workflow_for_circular_references(
        api_key="key",
        workspace_name="ws",
        target_workflow_id="my-wf",
        target_workflow_version_id="v42",
        visited=set(),
    )

    # then - spec fetched with version_id
    get_spec_mock.assert_called_once_with(
        api_key="key",
        workspace_id="ws",
        workflow_id="my-wf",
        workflow_version_id="v42",
    )


@mock.patch.object(v1, "get_workflow_specification")
def test_check_circular_references_uses_prefetched_spec(
    get_spec_mock: MagicMock,
) -> None:
    # given - a prefetched spec with no nested callers
    prefetched = {
        "inputs": [],
        "steps": [],
        "outputs": [],
    }

    # when - pass prefetched_spec so no API call is needed
    from inference.core.workflows.core_steps.composition.workflow_caller.v1 import (
        _check_workflow_for_circular_references,
    )
    _check_workflow_for_circular_references(
        api_key="key",
        workspace_name="ws",
        target_workflow_id="my-wf",
        visited=set(),
        prefetched_spec=prefetched,
    )

    # then - get_workflow_specification should NOT be called
    get_spec_mock.assert_not_called()


@mock.patch.object(v1, "get_workflow_specification")
def test_check_circular_references_fetches_for_nested_without_prefetched(
    get_spec_mock: MagicMock,
) -> None:
    # given - prefetched spec has a nested workflow caller with a version_id
    prefetched = {
        "inputs": [],
        "steps": [
            {
                "type": WORKFLOW_CALLER_BLOCK_TYPE,
                "workflow_id": "nested-wf",
                "workflow_version_id": "v99",
            },
        ],
        "outputs": [],
    }
    # The nested workflow has no further callers
    get_spec_mock.return_value = {
        "inputs": [],
        "steps": [],
        "outputs": [],
    }

    # when
    from inference.core.workflows.core_steps.composition.workflow_caller.v1 import (
        _check_workflow_for_circular_references,
    )
    _check_workflow_for_circular_references(
        api_key="key",
        workspace_name="ws",
        target_workflow_id="my-wf",
        visited=set(),
        prefetched_spec=prefetched,
    )

    # then - fetch called for the nested workflow with its version_id
    get_spec_mock.assert_called_once_with(
        api_key="key",
        workspace_id="ws",
        workflow_id="nested-wf",
        workflow_version_id="v99",
    )


# --- Sanitized error message test ---


@mock.patch("requests.post")
def test_execute_workflow_request_sanitizes_error_message(
    post_mock: MagicMock,
) -> None:
    # given - requests raises an exception with potentially sensitive details
    post_mock.side_effect = ConnectionError(
        "Connection refused: http://internal:9001/secret-path?api_key=sk-1234"
    )

    # when
    with mock.patch.object(v1, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"):
        error_status, message, result = execute_workflow_request(
            api_key="test_key",
            workspace_name="my-workspace",
            workflow_id="my-workflow",
            inputs={},
            request_timeout=30,
            call_chain="my-workflow",
        )

    # then - error message contains only exception type, not the full details
    assert error_status is True
    assert "ConnectionError" in message
    assert "sk-1234" not in message
    assert "secret-path" not in message

"""Tests for the Modal wire serializer's tensor-pivot defense-in-depth
(Step 7 of the tensor_compatibility plan): native ``inference_models`` objects
must never be silently stringified on the wire — flag-on they raise, flag-off
the pre-existing generic-object contract stays byte-identical."""

import json
from unittest import mock

import numpy as np
import pytest
import requests
import supervision as sv
import torch
from roboflow_workflows.errors import DynamicBlockError
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
    modal_executor,
    representation_boundary,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.entities import PythonCode
from roboflow_workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    CODE_VALIDATION_CACHE_TTL_SECONDS,
    serialize_for_modal_remote_execution,
    validate_code_in_modal,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
    RepresentationBoundaryError,
)

from inference_models.models.base.object_detection import Detections

_flag_on = mock.patch.object(
    representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", True
)
_flag_off = mock.patch.object(
    representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", False
)


def _native_detections() -> Detections:
    return Detections(
        xyxy=torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
    )


class _ArbitraryObject:
    def __init__(self):
        self.field = "value"

    def __str__(self):
        return "arbitrary-object"


def test_modal_serializer_raises_on_unconverted_native_detections_when_flag_on() -> (
    None
):
    # when
    with _flag_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = serialize_for_modal_remote_execution(
            inputs={"predictions": _native_detections()}
        )

    # then
    assert "Detections" in str(error.value)
    assert "reached the Modal wire serializer unconverted" in str(error.value)


def test_modal_serializer_raises_on_bare_tensor_when_flag_on() -> None:
    # when
    with _flag_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = serialize_for_modal_remote_execution(
            inputs={"value": torch.tensor([1.0, 2.0])}
        )

    # then
    assert "Tensor" in str(error.value)


def test_modal_serializer_raises_on_native_nested_in_containers_when_flag_on() -> None:
    # given - natives hide inside dicts/lists; the JSON encoder is the choke
    # point, so nesting must not smuggle them past the guard
    inputs = {"payload": {"items": [_native_detections()]}}

    # when
    with _flag_on, pytest.raises(RepresentationBoundaryError):
        _ = serialize_for_modal_remote_execution(inputs=inputs)


def test_modal_serializer_keeps_generic_fallback_for_arbitrary_objects_when_flag_on() -> (
    None
):
    # when
    with _flag_on:
        payload = json.loads(
            serialize_for_modal_remote_execution(inputs={"value": _ArbitraryObject()})
        )

    # then - pre-existing generic contract untouched for non-native objects
    assert payload["value"] == {
        "_type": "object",
        "class": "_ArbitraryObject",
        "value": "arbitrary-object",
    }


def test_modal_serializer_flag_off_behavior_is_byte_identical_to_legacy() -> None:
    # given - flag-off natives cannot exist in real flows, but the guard must be
    # provably inert: everything with __dict__ stringifies exactly as before
    inputs = {
        "native": _native_detections(),
        "tensor": torch.tensor([1.0]),
        "object": _ArbitraryObject(),
    }

    # when
    with _flag_off:
        payload = json.loads(serialize_for_modal_remote_execution(inputs=inputs))

    # then
    assert payload["native"]["_type"] == "object"
    assert payload["native"]["class"] == "Detections"
    assert payload["tensor"]["_type"] == "object"
    assert payload["tensor"]["class"] == "Tensor"
    assert payload["object"] == {
        "_type": "object",
        "class": "_ArbitraryObject",
        "value": "arbitrary-object",
    }


def test_modal_serializer_sv_detections_ride_dedicated_arm_when_flag_on() -> None:
    # given - the Option-A input leg: boundary-converted sv rides the dedicated
    # wire arm, never the generic fallback
    detections = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float32),
        data={
            "class_name": np.array(["widget"]),
            "detection_id": np.array(["det-1"]),
            "image_dimensions": np.array([[480, 640]]),
        },
    )

    # when
    with _flag_on:
        payload = json.loads(
            serialize_for_modal_remote_execution(inputs={"predictions": detections})
        )

    # then
    assert payload["predictions"]["_type"] == "sv_detections"
    assert payload["predictions"]["predictions"][0]["class"] == "widget"


def _workflow_image(seed: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"crop_{seed}"),
        numpy_image=np.full((8, 8, 3), seed, dtype=np.uint8),
    )


def test_modal_serializer_preserves_batch_of_images() -> None:
    """Batch must ride a dedicated arm.

    ``Batch`` subclasses neither ``list`` nor ``dict``, so without an explicit
    case it reaches the generic encoder, which stringifies it via ``str(obj)``.
    Blocks declaring ``batch_oriented_parameters`` then receive the repr instead
    of their crops.
    """
    batch = Batch(
        content=[_workflow_image(1), _workflow_image(2)], indices=[(0,), (1,)]
    )

    payload = json.loads(serialize_for_modal_remote_execution(inputs={"image": batch}))

    entry = payload["image"]
    assert entry["_type"] == "batch"
    assert entry["indices"] == [[0], [1]]
    assert len(entry["value"]) == 2
    for item in entry["value"]:
        assert item["_type"] == "workflow_image"
        assert item["value"], "image payload must survive, not be stringified"
    assert "Batch object at" not in json.dumps(payload)


def test_modal_serializer_preserves_batch_of_scalars() -> None:
    """Non-image batch parameters regress the same way (e.g. per-crop floats)."""
    batch = Batch(content=[0.357, 0.642], indices=[(0,), (1,)])

    payload = json.loads(serialize_for_modal_remote_execution(inputs={"frac": batch}))

    assert payload["frac"] == {
        "_type": "batch",
        "value": [0.357, 0.642],
        "indices": [[0], [1]],
    }


def test_modal_serializer_handles_batch_without_indices() -> None:
    batch = Batch(content=[1, 2, 3], indices=None)

    payload = json.loads(serialize_for_modal_remote_execution(inputs={"v": batch}))

    assert payload["v"]["value"] == [1, 2, 3]
    assert payload["v"]["indices"] is None


@pytest.fixture
def clock() -> mock.Mock:
    return mock.Mock(return_value=0.0)


@pytest.fixture
def execute_remote(clock: mock.Mock):
    executor = mock.MagicMock()
    validation_cache = modal_executor._create_code_validation_cache(clock=clock)
    with mock.patch.object(modal_executor, "MODAL_AVAILABLE", True), mock.patch.object(
        modal_executor, "ModalExecutor", return_value=executor
    ), mock.patch.object(modal_executor, "CODE_VALIDATION_CACHE", validation_cache):
        yield executor.execute_remote


def _block_code() -> PythonCode:
    return PythonCode(
        type="PythonCode",
        run_function_code="def run(self, image):\n    return {'x': 1}\n",
    )


def _checked_workspaces(execute_remote: mock.MagicMock) -> list:
    return [call.kwargs["workspace_id"] for call in execute_remote.call_args_list]


def test_validate_code_in_modal_reuses_success_within_workspace(
    execute_remote,
) -> None:
    execute_remote.return_value = {"valid": True}

    first = validate_code_in_modal(_block_code(), "workspace-a")
    second = validate_code_in_modal(_block_code(), "workspace-a")

    assert first is True
    assert second is True
    execute_remote.assert_called_once()


def test_validate_code_in_modal_checks_each_workspace_independently(
    execute_remote,
) -> None:
    execute_remote.side_effect = [
        {"valid": True},
        {"valid": False, "error": "warning treated as error", "line": 1},
    ]

    first = validate_code_in_modal(_block_code(), "workspace-a")
    with pytest.raises(DynamicBlockError) as error:
        validate_code_in_modal(_block_code(), "workspace-b")

    assert first is True
    assert (
        error.value.public_message
        == "Code validation failed: Line 1: warning treated as error"
    )
    assert _checked_workspaces(execute_remote) == ["workspace-a", "workspace-b"]


def test_validate_code_in_modal_revalidates_after_expiry_even_when_read_often(
    execute_remote, clock: mock.Mock
) -> None:
    execute_remote.return_value = {"valid": True}

    validate_code_in_modal(_block_code(), "workspace-a")
    for elapsed in range(60, CODE_VALIDATION_CACHE_TTL_SECONDS, 60):
        clock.return_value = elapsed
        validate_code_in_modal(_block_code(), "workspace-a")
    checks_before_expiry = execute_remote.call_count

    clock.return_value = CODE_VALIDATION_CACHE_TTL_SECONDS
    validate_code_in_modal(_block_code(), "workspace-a")
    clock.return_value = 2 * CODE_VALIDATION_CACHE_TTL_SECONDS - 1
    validate_code_in_modal(_block_code(), "workspace-a")

    assert checks_before_expiry == 1
    assert execute_remote.call_count == 2


def test_validate_code_in_modal_does_not_cache_failed_check(execute_remote) -> None:
    execute_remote.side_effect = [
        {"valid": False, "error": "invalid syntax", "line": 2},
        {"valid": True},
    ]

    with pytest.raises(DynamicBlockError) as error:
        validate_code_in_modal(_block_code(), "workspace-a")
    retried = validate_code_in_modal(_block_code(), "workspace-a")

    assert (
        error.value.public_message == "Code validation failed: Line 2: invalid syntax"
    )
    assert error.value.context == "modal_executor | code_validation"
    assert retried is True
    assert execute_remote.call_count == 2


def test_validate_code_in_modal_does_not_cache_reply_without_verdict(
    execute_remote,
) -> None:
    execute_remote.return_value = {}

    first = validate_code_in_modal(_block_code(), "workspace-a")
    second = validate_code_in_modal(_block_code(), "workspace-a")

    assert first is True
    assert second is True
    assert execute_remote.call_count == 2


@pytest.mark.parametrize(
    "anonymous_workspace", [modal_executor.MODAL_ANONYMOUS_WORKSPACE_NAME, None]
)
def test_validate_code_in_modal_always_checks_anonymous_workspace(
    execute_remote, anonymous_workspace
) -> None:
    execute_remote.return_value = {"valid": True}

    validate_code_in_modal(_block_code(), anonymous_workspace)
    result = validate_code_in_modal(_block_code(), anonymous_workspace)

    assert result is True
    assert execute_remote.call_count == 2


@pytest.mark.parametrize(
    "failure",
    [
        DynamicBlockError(
            public_message="Modal endpoint returned status 408: timeout",
            context="modal_executor | http_request",
        ),
        requests.exceptions.ReadTimeout("read timed out"),
    ],
)
def test_validate_code_in_modal_does_not_cache_failure_to_reach_modal(
    execute_remote, failure: Exception
) -> None:
    execute_remote.side_effect = [failure, {"valid": True}]

    with pytest.raises(DynamicBlockError):
        validate_code_in_modal(_block_code(), "workspace-a")
    result = validate_code_in_modal(_block_code(), "workspace-a")

    assert result is True
    assert execute_remote.call_count == 2

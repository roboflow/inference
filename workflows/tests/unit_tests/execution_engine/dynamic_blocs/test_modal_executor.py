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
    CODE_VALIDATION_CACHE,
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


def _clear_code_validation_cache() -> None:
    with CODE_VALIDATION_CACHE._cache_lock:
        CODE_VALIDATION_CACHE._cache.clear()
        CODE_VALIDATION_CACHE._keys_buffer.clear()


@pytest.fixture
def execute_remote():
    executor = mock.MagicMock()
    _clear_code_validation_cache()
    with mock.patch.object(modal_executor, "MODAL_AVAILABLE", True), mock.patch.object(
        modal_executor, "ModalExecutor", return_value=executor
    ):
        yield executor.execute_remote
    _clear_code_validation_cache()


def _block_code() -> PythonCode:
    return PythonCode(
        type="PythonCode",
        run_function_code="def run(self, image):\n    return {'x': 1}\n",
    )


def test_validate_code_in_modal_reuses_valid_verdict_across_workspaces(
    execute_remote,
) -> None:
    execute_remote.return_value = {"valid": True}

    first = validate_code_in_modal(_block_code(), "workspace-a")
    second = validate_code_in_modal(_block_code(), "workspace-b")

    assert first is True
    assert second is True
    execute_remote.assert_called_once()


def test_validate_code_in_modal_reuses_syntax_error(execute_remote) -> None:
    execute_remote.return_value = {
        "valid": False,
        "error": "invalid syntax",
        "line": 2,
    }
    errors = []

    for _ in range(2):
        with pytest.raises(DynamicBlockError) as error:
            validate_code_in_modal(_block_code(), "workspace-a")
        errors.append((error.value.public_message, error.value.context))

    expected = (
        "Code validation failed: Line 2: invalid syntax",
        "modal_executor | code_validation",
    )
    assert errors == [expected, expected]
    execute_remote.assert_called_once()


@pytest.mark.parametrize(
    "anonymous_workspace", [modal_executor.MODAL_ANONYMOUS_WORKSPACE_NAME, None]
)
def test_validate_code_in_modal_does_not_serve_cache_to_anonymous_workspace(
    execute_remote, anonymous_workspace
) -> None:
    execute_remote.return_value = {"valid": True}
    validate_code_in_modal(_block_code(), "workspace-a")

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

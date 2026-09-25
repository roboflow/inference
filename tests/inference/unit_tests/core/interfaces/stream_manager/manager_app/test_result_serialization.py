"""WP-A04 characterization of the stream manager's result serialisation.

The manager used to serialise buffered workflow results with the deprecated
HTTP helper `inference.core.interfaces.http.orjson_utils`. These tests pin its
concrete output (nested containers, images, detections, excluded fields, the
call-time choice between the numpy and tensor wildcard serialisers) and then
hold the manager's own helper to the same values. The HTTP helper itself
stays callable and keeps warning.
"""

import base64
import pickle
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import supervision as sv

import inference.core.env as core_env
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.http import orjson_utils
from inference.core.interfaces.stream import environment as stream_environment
from inference.core.interfaces.stream.sinks import InMemoryBufferSink
from inference.core.interfaces.stream_manager.manager_app import (
    inference_pipeline_manager,
    result_serialization,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    CommandType,
    OperationStatus,
)
from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (
    InferencePipelineManager,
)
from inference.core.warnings import InferenceDeprecationWarning
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

torch = pytest.importorskip("torch")

DETECTIONS_SERIALISED = {
    "image": {"width": None, "height": None},
    "predictions": [
        {
            "width": 2.0,
            "height": 3.0,
            "x": 2.0,
            "y": 3.5,
            "confidence": 0.5,
            "class_id": 1,
            "class": "cat",
            "detection_id": "d1",
        }
    ],
}


def _detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[1.0, 2.0, 3.0, 5.0]]),
        confidence=np.array([0.5]),
        class_id=np.array([1]),
        data={"class_name": np.array(["cat"]), "detection_id": np.array(["d1"])},
    )


def _image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.full((2, 3, 3), 128, dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier="camera-1",
            frame_number=3,
            frame_timestamp=datetime(2024, 1, 2, 3, 4, 5),
            fps=15.0,
        ),
    )


def _result_element() -> Dict[str, Any]:
    return {
        "image": _image(),
        "detections": _detections(),
        "nested": {
            "list": [1, (2, 3), {"skip": "kept when nested"}],
            "timestamp": datetime(2024, 1, 2, 3, 4, 5),
        },
        "tuple": (1, [_detections()]),
        "array": np.array([1, 2]),
        "tensor": torch.tensor([1.0, 2.0]),
        "skip": "excluded at top level",
    }


def _legacy_serialiser(
    result_element: Dict[str, Any],
    excluded_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InferenceDeprecationWarning)
        serialised = orjson_utils.serialise_single_workflow_result_element(
            result_element=result_element,
            excluded_fields=excluded_fields,
        )

    return serialised


def _set_legacy_tensor_mode(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", enabled)


def _set_stream_tensor_mode(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    # The manager's helper reads the installed streams configuration, never
    # `inference.core.env`: pin the legacy flag to the opposite value to prove
    # it is not consulted.
    monkeypatch.setattr(
        stream_environment, "ENABLE_TENSOR_DATA_REPRESENTATION", enabled
    )
    monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", not enabled)


# (serialiser, function setting its tensor mode for the call)
SERIALISERS = [
    pytest.param(_legacy_serialiser, _set_legacy_tensor_mode, id="legacy-http"),
    pytest.param(
        result_serialization.serialise_single_workflow_result_element,
        _set_stream_tensor_mode,
        id="manager",
    ),
]


def _assert_image_serialised(value: Any) -> None:
    assert {key: item for key, item in value.items() if key != "value"} == {
        "type": "base64",
        "video_metadata": {
            "video_identifier": "camera-1",
            "frame_number": 3,
            "frame_timestamp": datetime(2024, 1, 2, 3, 4, 5),
            "fps": 15.0,
            "measured_fps": None,
            "comes_from_video_file": None,
        },
    }
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(value["value"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == (2, 3, 3)
    assert np.abs(decoded.astype(int) - 128).max() <= 2


@pytest.mark.parametrize("serialise, set_tensor_mode", SERIALISERS)
def test_numpy_mode_serialises_nested_values(
    serialise: Callable,
    set_tensor_mode: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_tensor_mode(monkeypatch, False)
    element = _result_element()

    result = serialise(result_element=element, excluded_fields=["skip"])

    assert list(result) == [
        "image",
        "detections",
        "nested",
        "tuple",
        "array",
        "tensor",
    ]
    _assert_image_serialised(result["image"])
    assert result["detections"] == DETECTIONS_SERIALISED
    assert result["nested"] == {
        "list": [1, (2, 3), {"skip": "kept when nested"}],
        "timestamp": "2024-01-02T03:04:05",
    }
    # Tuples, arrays and tensors pass through the numpy serialiser untouched.
    assert result["tuple"][0] == 1
    assert result["tuple"][1][0] is element["tuple"][1][0]
    assert result["array"] is element["array"]
    assert result["tensor"] is element["tensor"]


@pytest.mark.parametrize("serialise, set_tensor_mode", SERIALISERS)
def test_tensor_mode_converts_every_nested_tensor(
    serialise: Callable,
    set_tensor_mode: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_tensor_mode(monkeypatch, True)

    result = serialise(result_element=_result_element(), excluded_fields=["skip"])

    _assert_image_serialised(result["image"])
    assert result["detections"] == DETECTIONS_SERIALISED
    assert result["nested"] == {
        "list": [1, (2, 3), {"skip": "kept when nested"}],
        "timestamp": "2024-01-02T03:04:05",
    }
    assert result["tuple"] == (1, [DETECTIONS_SERIALISED])
    assert result["array"].tolist() == [1, 2]
    assert result["tensor"] == [1.0, 2.0]
    assert type(result["tensor"]) is list


@pytest.mark.parametrize("serialise, set_tensor_mode", SERIALISERS)
def test_tensor_serialiser_is_selected_at_call_time(
    serialise: Callable,
    set_tensor_mode: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.tensor([[3.0]])

    set_tensor_mode(monkeypatch, True)
    tensor_mode_result = serialise(result_element={"value": (tensor, "context")})
    set_tensor_mode(monkeypatch, False)
    numpy_mode_result = serialise(result_element={"value": (tensor, "context")})

    assert tensor_mode_result == {"value": ([[3.0]], "context")}
    assert numpy_mode_result["value"][0] is tensor


@pytest.mark.parametrize("serialise, set_tensor_mode", SERIALISERS)
@pytest.mark.parametrize(
    "excluded_fields, expected_keys",
    [
        (None, ["a", "b", "c"]),
        ([], ["a", "b", "c"]),
        (["b"], ["a", "c"]),
        (["a", "c", "missing"], ["b"]),
        (("a", "a"), ["b", "c"]),
    ],
)
def test_excluded_fields_filter_only_top_level_keys(
    serialise: Callable,
    set_tensor_mode: Callable,
    excluded_fields: Optional[List[str]],
    expected_keys: List[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_tensor_mode(monkeypatch, False)
    element = {"a": 1, "b": {"a": 2}, "c": [{"b": 3}]}

    result = serialise(result_element=element, excluded_fields=excluded_fields)

    assert list(result) == expected_keys
    for key in expected_keys:
        assert result[key] == element[key]


def test_legacy_http_helpers_stay_callable_and_deprecated() -> None:
    with pytest.warns(InferenceDeprecationWarning) as record:
        single = orjson_utils.serialise_single_workflow_result_element(
            result_element={"a": 1, "b": 2}, excluded_fields=["b"]
        )
        batch = orjson_utils.serialise_workflow_result(
            result=[{"a": 1, "b": 2}], excluded_fields=["b"]
        )

    assert single == {"a": 1}
    assert batch == [{"a": 1}]
    messages = [str(warning.message) for warning in record]
    assert messages[0].startswith(
        "serialise_single_workflow_result_element is deprecated: "
    )
    assert any(
        message.startswith("serialise_workflow_result is deprecated: ")
        for message in messages
    )


# --------------------------------------------------------------------------
# The manager's CONSUME_RESULT command
# --------------------------------------------------------------------------


def _find_tensors(value: Any) -> List[Any]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _find_tensors(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _find_tensors(item)]
    return []


def _consume(
    predictions: List[Optional[dict]],
    payload_extra: Optional[dict] = None,
) -> dict:
    responses_queue = MagicMock()
    manager = InferencePipelineManager(
        pipeline_id="my_pipeline",
        command_queue=MagicMock(),
        responses_queue=responses_queue,
    )
    buffer_sink = InMemoryBufferSink(queue_size=4)
    frames = [
        (
            None
            if prediction is None
            else VideoFrame(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                frame_id=index,
                frame_timestamp=datetime(2024, 1, 2, 3, 4, 5),
                source_id=7,
            )
        )
        for index, prediction in enumerate(predictions)
    ]
    buffer_sink.on_prediction(predictions=predictions, video_frame=frames)
    manager._buffer_sink = buffer_sink

    manager._handle_command(
        request_id="request",
        payload={"type": CommandType.CONSUME_RESULT, **(payload_extra or {})},
    )

    responses_queue.put.assert_called_once()
    request_id, response = responses_queue.put.call_args[0][0]
    assert request_id == "request"

    return response


def _set_manager_tensor_mode(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    _set_stream_tensor_mode(monkeypatch, enabled)


def test_manager_uses_its_own_helper_and_it_does_not_warn() -> None:
    helper = result_serialization.serialise_single_workflow_result_element

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        serialised = helper(result_element={"a": 1, "b": 2}, excluded_fields=["b"])

    assert inference_pipeline_manager.serialise_single_workflow_result_element is helper
    assert helper is not orjson_utils.serialise_single_workflow_result_element
    assert serialised == {"a": 1}


@pytest.mark.parametrize("tensor_mode", [False, True])
@pytest.mark.parametrize("excluded_fields", [None, ["skip"], ["image", "tuple"]])
def test_manager_helper_matches_legacy_helper_output(
    tensor_mode: bool,
    excluded_fields: Optional[List[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor_mode)
    monkeypatch.setattr(
        stream_environment, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor_mode
    )
    legacy_element = _result_element()
    element = _result_element()

    expected = _legacy_serialiser(
        result_element=legacy_element, excluded_fields=excluded_fields
    )
    actual = result_serialization.serialise_single_workflow_result_element(
        result_element=element, excluded_fields=excluded_fields
    )

    assert list(actual) == list(expected)
    for key in expected:
        if key in {"array", "tensor"} or (key == "tuple" and not tensor_mode):
            # Passed through: compare the values, not the fresh objects.
            continue
        assert actual[key] == expected[key], key
    if "array" in expected:
        assert actual["array"].tolist() == expected["array"].tolist()
    if tensor_mode:
        assert actual["tensor"] == expected["tensor"] == [1.0, 2.0]
    else:
        assert actual["tensor"] is element["tensor"]


def test_serialisers_import_no_http_modules() -> None:
    import ast

    source = open(result_serialization.__file__, encoding="utf-8").read()
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }

    assert imported == {
        "typing",
        "inference.core.interfaces.stream",
        "roboflow_workflows.core_steps.common.serializers",
        "roboflow_workflows.core_steps.common.serializers_tensor",
    }


def test_manager_consumption_serialises_results_and_frames_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_manager_tensor_mode(monkeypatch, False)

    response = _consume(
        predictions=[None, {"detections": _detections(), "skip": 1}],
        payload_extra={"excluded_fields": ["skip"]},
    )

    assert response == {
        "status": OperationStatus.SUCCESS,
        "outputs": [None, {"detections": DETECTIONS_SERIALISED}],
        "frames_metadata": [
            None,
            {
                "frame_timestamp": "2024-01-02T03:04:05",
                "frame_id": 1,
                "source_id": 7,
            },
        ],
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_manager_consumption_never_hands_a_live_tensor_to_the_ipc_queue(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Jetson/Tegra: pickling a live CUDA tensor into the responses queue uses
    # CUDA IPC, which fails there. The serialiser's `.detach().cpu()` is the
    # same call regardless of the source device, so both are exercised here.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    _set_manager_tensor_mode(monkeypatch, True)

    response = _consume(
        predictions=[
            {
                "wildcard": (torch.tensor([1.0, 2.0], device=device), "context"),
                "nested": [{"pair": (torch.tensor([[3.0]], device=device), 4)}],
                "plain": torch.tensor(5, device=device),
            }
        ],
    )

    assert _find_tensors(response) == []
    assert response["outputs"] == [
        {
            "wildcard": ([1.0, 2.0], "context"),
            "nested": [{"pair": ([[3.0]], 4)}],
            "plain": 5,
        }
    ]
    assert _find_tensors(pickle.loads(pickle.dumps(response))) == []

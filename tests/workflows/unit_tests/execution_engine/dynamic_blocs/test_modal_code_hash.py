import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    _compute_code_hash,
    _deserialize_msgpack_result,
    deserialize_for_modal_remote_execution,
    serialize_for_modal_remote_execution,
    serialize_inputs_for_msgpack,
)


class _FakeModalImage:
    @classmethod
    def debian_slim(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def apt_install(self, *args, **kwargs):
        return self

    def pip_install(self, *args, **kwargs):
        return self

    def entrypoint(self, *args, **kwargs):
        return self


class _FakeModalApp:
    def __init__(self, name: str):
        self.name = name

    def cls(self, *args, **kwargs):
        return lambda cls: cls


def _identity_decorator(*args, **kwargs):
    return lambda obj: obj


@pytest.fixture()
def modal_app_with_fake_modal(monkeypatch):
    fake_modal = ModuleType("modal")
    fake_modal.App = _FakeModalApp
    fake_modal.Image = _FakeModalImage
    fake_modal.parameter = lambda *args, **kwargs: None
    fake_modal.enter = _identity_decorator
    fake_modal.fastapi_endpoint = _identity_decorator
    fake_modal.asgi_app = _identity_decorator
    fake_modal.concurrent = _identity_decorator
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    modal_app_path = Path(__file__).resolve().parents[5] / "modal" / "modal_app.py"
    spec = importlib.util.spec_from_file_location(
        "modal_app_code_hash_parity_test", modal_app_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "code,imports",
    [
        ("def run():\n    return {}", []),
        (
            "def run(x):\n    return {'sqrt': math.sqrt(x)}",
            ["import math", "from typing import Any"],
        ),
        ("def run():\n    return 'ok'", None),
    ],
)
def test_client_and_server_code_hashes_stay_in_sync(
    modal_app_with_fake_modal,
    code,
    imports,
) -> None:
    executor = modal_app_with_fake_modal.Executor.__new__(
        modal_app_with_fake_modal.Executor
    )

    assert _compute_code_hash(code, imports) == executor._get_code_hash(code, imports)


def _make_video_metadata() -> VideoMetadata:
    return VideoMetadata(
        video_identifier="video-1",
        frame_number=7,
        frame_timestamp=datetime(2026, 1, 2, 3, 4, 5),
        fps=29.97,
        measured_fps=28.5,
        comes_from_video_file=True,
    )


def _make_workflow_image(video_metadata: VideoMetadata) -> WorkflowImageData:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="crop-1",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=2,
                left_top_y=3,
                origin_width=16,
                origin_height=16,
            ),
        ),
        workflow_root_ancestor_metadata=ImageParentMetadata(
            parent_id="root-1",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=10,
                left_top_y=20,
                origin_width=64,
                origin_height=64,
            ),
        ),
        numpy_image=image,
        video_metadata=video_metadata,
    )


def _make_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[1.0, 2.0, 5.0, 6.0]], dtype=float),
        confidence=np.array([0.91]),
        class_id=np.array([3]),
        data={
            "class_name": np.array(["widget"]),
            DETECTION_ID_KEY: np.array(["det-1"]),
            IMAGE_DIMENSIONS_KEY: np.array([[20, 30]]),
            PARENT_ID_KEY: np.array(["frame-1"]),
            ROOT_PARENT_ID_KEY: np.array(["root-frame"]),
        },
    )


def _make_modal_payload() -> dict:
    video_metadata = _make_video_metadata()
    return {
        "image": _make_workflow_image(video_metadata=video_metadata),
        "detections": _make_detections(),
        "video_metadata": video_metadata,
    }


def _assert_video_metadata_equal(
    actual: VideoMetadata, expected: VideoMetadata
) -> None:
    assert actual.model_dump() == expected.model_dump()


def _assert_workflow_images_equal(
    actual: WorkflowImageData,
    expected: WorkflowImageData,
) -> None:
    assert actual.parent_metadata.parent_id == expected.parent_metadata.parent_id
    assert (
        actual.parent_metadata.origin_coordinates
        == expected.parent_metadata.origin_coordinates
    )
    assert (
        actual.workflow_root_ancestor_metadata.parent_id
        == expected.workflow_root_ancestor_metadata.parent_id
    )
    assert (
        actual.workflow_root_ancestor_metadata.origin_coordinates
        == expected.workflow_root_ancestor_metadata.origin_coordinates
    )
    _assert_video_metadata_equal(actual.video_metadata, expected.video_metadata)
    np.testing.assert_array_equal(actual.numpy_image, expected.numpy_image)


def _assert_detections_equal(
    actual: sv.Detections,
    expected: sv.Detections,
) -> None:
    np.testing.assert_allclose(actual.xyxy, expected.xyxy)
    np.testing.assert_allclose(actual.confidence, expected.confidence)
    np.testing.assert_array_equal(actual.class_id, expected.class_id)
    assert actual.data.keys() == expected.data.keys()
    for key, expected_value in expected.data.items():
        actual_value = actual.data[key]
        if isinstance(actual_value, torch.Tensor):
            actual_value = actual_value.detach().cpu().numpy()
        if isinstance(expected_value, torch.Tensor):
            expected_value = expected_value.detach().cpu().numpy()
        if isinstance(actual_value, np.ndarray) or isinstance(
            expected_value, np.ndarray
        ):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _assert_payloads_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    _assert_workflow_images_equal(actual["image"], expected["image"])
    _assert_detections_equal(actual["detections"], expected["detections"])
    _assert_video_metadata_equal(
        actual["video_metadata"],
        expected["video_metadata"],
    )


def _assert_msgpack_compatible(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_msgpack_compatible(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_msgpack_compatible(item)
        return
    assert value is None or isinstance(value, (bool, int, float, str, bytes))


def test_msgpack_input_round_trip_matches_http_json_path(
    modal_app_with_fake_modal,
) -> None:
    payload = _make_modal_payload()
    msgpack_wire_payload = serialize_inputs_for_msgpack(payload)

    _assert_msgpack_compatible(msgpack_wire_payload)
    msgpack_inputs = modal_app_with_fake_modal.Executor._deserialize_msgpack_inputs(
        msgpack_wire_payload
    )
    http_inputs = deserialize_for_modal_remote_execution(
        serialize_for_modal_remote_execution(payload)
    )

    _assert_payloads_equal(msgpack_inputs, http_inputs)


def test_msgpack_result_round_trip_matches_http_json_path(
    modal_app_with_fake_modal,
) -> None:
    payload = _make_modal_payload()
    msgpack_wire_payload = modal_app_with_fake_modal.Executor._serialize_msgpack_result(
        payload
    )

    _assert_msgpack_compatible(msgpack_wire_payload)
    msgpack_result = _deserialize_msgpack_result(msgpack_wire_payload)
    http_result = deserialize_for_modal_remote_execution(
        serialize_for_modal_remote_execution(payload)
    )

    _assert_payloads_equal(msgpack_result, http_result)

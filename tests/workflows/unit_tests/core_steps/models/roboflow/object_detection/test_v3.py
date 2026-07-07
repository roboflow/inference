from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.remote_stream_pipeline import (
    make_prediction_future,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection import (
    v3 as object_detection_v3,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    BlockManifest,
    RoboflowObjectDetectionModelBlockV3,
)

_CANNED_PREDICTION = {
    "predictions": [],
    "image": {"width": 10, "height": 10},
    "time": 0.1,
}


class _FakeRemoteImage:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    @property
    def base64_image(self) -> str:
        return self.tag


def _make_remote_block() -> RoboflowObjectDetectionModelBlockV3:
    return RoboflowObjectDetectionModelBlockV3(
        model_manager=MagicMock(),
        api_key="key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )


def _patch_remote_http_client(monkeypatch) -> MagicMock:
    mock_client = MagicMock()
    mock_client.infer.return_value = _CANNED_PREDICTION
    monkeypatch.setattr(
        object_detection_v3,
        "InferenceHTTPClient",
        MagicMock(return_value=mock_client),
    )
    monkeypatch.setattr(object_detection_v3, "InferenceConfiguration", MagicMock())
    return mock_client


def _stub_post_process(block: RoboflowObjectDetectionModelBlockV3) -> None:
    block._post_process_result = lambda images, predictions, class_filter, model_id: [
        {
            "inference_id": None,
            "predictions": f"post-processed-{images[0].base64_image}",
            "model_id": model_id,
        }
    ]


def _run_remotely(
    block: RoboflowObjectDetectionModelBlockV3,
    images,
    model_id: str = "workspace/model/1",
):
    return block.run_remotely(
        images=images,
        model_id=model_id,
        class_agnostic_nms=None,
        class_filter=None,
        confidence=None,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
    )


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_object_detection_model_validation_when_minimalistic_config_is_provided(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_object_detection_model@v3",
        "name": "some",
        images_field_alias: "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/roboflow_object_detection_model@v3",
        name="some",
        images="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "images", "model_id"])
def test_object_detection_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_object_detection_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_object_detection_model_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_object_detection_model_validation_when_model_id_has_invalid_type() -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_object_detection_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_object_detection_model_validation_when_active_learning_flag_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/roboflow_object_detection_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_object_detection_model_validation_when_custom_mode_missing_custom_confidence() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/roboflow_object_detection_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "confidence_mode": "custom",
        "custom_confidence": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("custom_confidence", 1.1),
        ("images", "some"),
        ("disable_active_learning", "some"),
        ("class_agnostic_nms", "some"),
        ("class_filter", "some"),
        ("confidence_mode", "invalid-mode"),
        ("custom_confidence", "some"),
        ("iou_threshold", "some"),
        ("iou_threshold", 1.1),
        ("max_detections", 0),
        ("max_candidates", 0),
    ],
)
def test_object_detection_model_when_parameters_have_invalid_type(
    parameter: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_object_detection_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        parameter: value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize(
    "depth, execution_mode, expected_pipelined, expected_depth",
    [
        (1, StepExecutionMode.REMOTE, False, 0),
        (4, StepExecutionMode.REMOTE, True, 3),
        (4, StepExecutionMode.LOCAL, False, 0),
        (1, StepExecutionMode.LOCAL, False, 0),
    ],
)
def test_object_detection_stream_pipeline_protocol_gating(
    monkeypatch,
    depth: int,
    execution_mode: StepExecutionMode,
    expected_pipelined: bool,
    expected_depth: int,
) -> None:
    # given
    monkeypatch.setattr(
        object_detection_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", depth
    )
    block = RoboflowObjectDetectionModelBlockV3(
        model_manager=MagicMock(),
        api_key="key",
        step_execution_mode=execution_mode,
    )

    # when / then
    assert block.is_stream_pipelined() is expected_pipelined
    assert block.can_activate_stream_pipeline() is expected_pipelined
    assert block.stream_pipeline_depth() == expected_depth
    # The defer marker is unconditional for this block.
    assert block.defers_downstream_execution() is True


def test_object_detection_pipelined_run_remotely_returns_prediction_futures(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(
        object_detection_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", 4
    )
    _patch_remote_http_client(monkeypatch)
    block = _make_remote_block()
    _stub_post_process(block)

    # when
    result = _run_remotely(block, images=[_FakeRemoteImage("a")])

    # then
    assert len(result) == 1
    assert result[0]["inference_id"] is None
    assert result[0]["model_id"] == "workspace/model/1"
    assert isinstance(result[0]["predictions"], Future)
    assert result[0]["predictions"].result(timeout=5) == "post-processed-a"
    assert block._remote_pipeline.pending_requests == 1

    flushed = block.flush_stream_pipeline_outputs()
    assert flushed == [
        (
            [(0,)],
            [
                {
                    "inference_id": None,
                    "predictions": "post-processed-a",
                    "model_id": "workspace/model/1",
                }
            ],
        )
    ]
    # A single context is drained per call; the emptied deque yields nothing.
    assert block.flush_stream_pipeline_outputs() == []

    block.close_stream_pipeline()
    assert block._remote_pipeline is None


def test_object_detection_pipelined_run_remotely_flushes_contexts_in_fifo_order(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(
        object_detection_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", 4
    )
    _patch_remote_http_client(monkeypatch)
    block = _make_remote_block()
    _stub_post_process(block)

    # when
    first = _run_remotely(block, images=[_FakeRemoteImage("a")])
    second = _run_remotely(block, images=[_FakeRemoteImage("b")])

    # then
    assert first[0]["predictions"].result(timeout=5) == "post-processed-a"
    assert second[0]["predictions"].result(timeout=5) == "post-processed-b"
    assert block._remote_pipeline.pending_requests == 2

    first_flush = block.flush_stream_pipeline_outputs()
    second_flush = block.flush_stream_pipeline_outputs()

    assert first_flush[0][1][0]["predictions"] == "post-processed-a"
    assert second_flush[0][1][0]["predictions"] == "post-processed-b"
    assert block.flush_stream_pipeline_outputs() == []

    block.close_stream_pipeline()


def test_object_detection_non_pipelined_run_remotely_executes_synchronously(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(
        object_detection_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", 1
    )
    mock_client = _patch_remote_http_client(monkeypatch)
    block = _make_remote_block()
    _stub_post_process(block)

    # when
    result = _run_remotely(block, images=[_FakeRemoteImage("a")])

    # then
    assert result == [
        {
            "inference_id": None,
            "predictions": "post-processed-a",
            "model_id": "workspace/model/1",
        }
    ]
    mock_client.infer.assert_called_once()
    assert block._remote_pipeline is None


def test_make_prediction_future_resolves_to_indexed_predictions() -> None:
    # given
    result_future: Future = Future()
    prediction_future = make_prediction_future(
        result_future=result_future, image_index=1
    )

    # when
    result_future.set_result([{"predictions": "first"}, {"predictions": "second"}])

    # then
    assert prediction_future.result(timeout=5) == "second"


def test_make_prediction_future_propagates_exception() -> None:
    # given
    result_future: Future = Future()
    prediction_future = make_prediction_future(
        result_future=result_future, image_index=0
    )

    # when
    result_future.set_exception(RuntimeError("remote failed"))

    # then
    with pytest.raises(RuntimeError, match="remote failed"):
        prediction_future.result(timeout=5)

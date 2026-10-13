"""Unit tests of ``v2/roboflow_custom_metadata@v1`` against fake ports.

Fakes prove wiring and V1 outcome rules only; the host adapter and its HTTP
requests are tested in the inference package. Every credential is invented.
"""

import logging
from typing import Any, Dict, List, Optional

import pytest
import torch
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.custom_metadata import (
    NO_INFERENCE_ID_MESSAGE,
    RoboflowCustomMetadataBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

API_KEY = "rf-test-KEY-r1-custom-0001"


class FakeMetadataPort:
    """Records calls; may fail either operation. Never reveals the key."""

    def __init__(
        self,
        *,
        workspace_error: Optional[Exception] = None,
        add_error: Optional[Exception] = None,
    ) -> None:
        self.workspace_error = workspace_error
        self.add_error = add_error
        self.calls: List[Dict[str, Any]] = []

    def get_workspace(self, *, api_key: Secret) -> str:
        self.calls.append({"operation": "get_workspace", "api_key": api_key})
        if self.workspace_error is not None:
            raise self.workspace_error
        return "fake-workspace"

    def add_custom_metadata(self, **arguments: Any) -> None:
        self.calls.append({"operation": "add_custom_metadata", **arguments})
        if self.add_error is not None:
            raise self.add_error


def _detections(metadata: Optional[dict]) -> Detections:
    detections = Detections(
        xyxy=torch.zeros((1, 4)),
        class_id=torch.zeros((1,), dtype=torch.long),
        confidence=torch.ones((1,)),
        image_metadata=metadata,
    )

    return detections


def _key_points(metadata: Optional[dict]) -> KeyPoints:
    key_points = KeyPoints(
        xy=torch.zeros((1, 2, 2)),
        class_id=torch.zeros((1,), dtype=torch.long),
        confidence=torch.ones((1, 2)),
        image_metadata=metadata,
    )

    return key_points


def _run(block: RoboflowCustomMetadataBlock, predictions: Any, **params: Any) -> dict:
    arguments = {
        "predictions": predictions,
        "field_name": "location",
        "field_value": "toronto",
        "inference_id": None,
        "fire_and_forget": True,
        **params,
    }
    result = block.run(**arguments)

    return result


def test_explicit_inference_id_is_tagged_with_the_workspace_of_the_key() -> None:
    # given
    port = FakeMetadataPort()
    block = RoboflowCustomMetadataBlock(custom_metadata_port=port, api_key=API_KEY)

    # when
    result = _run(block, _detections({}), inference_id="inf-1")

    # then
    assert result == {
        "error_status": False,
        "message": "Custom metadata upload was successful",
    }
    assert [call["operation"] for call in port.calls] == [
        "get_workspace",
        "add_custom_metadata",
    ]
    added = port.calls[1]
    assert added["workspace_id"] == "fake-workspace"
    assert added["inference_ids"] == ["inf-1"]
    assert (added["field_name"], added["field_value"]) == ("location", "toronto")
    assert isinstance(added["api_key"], Secret)
    assert added["api_key"].name == "api_key"


@pytest.mark.parametrize(
    "predictions",
    [
        _detections({"inference_id": "inf-meta"}),
        ClassificationPrediction(
            class_id=torch.tensor([0]),
            confidence=torch.tensor([[1.0]]),
            images_metadata=[{"inference_id": "inf-meta"}],
        ),
        MultiLabelClassificationPrediction(
            class_ids=torch.tensor([0]),
            confidence=torch.tensor([1.0]),
            image_metadata={"inference_id": "inf-meta"},
        ),
        (_key_points({"inference_id": "inf-meta"}), None),
        (_key_points({}), _detections({"inference_id": "inf-meta"})),
    ],
    ids=[
        "detections",
        "classification",
        "multi_label",
        "key_points",
        "key_points+boxes",
    ],
)
def test_inference_id_is_read_from_prediction_metadata_as_v1(predictions: Any) -> None:
    # given
    port = FakeMetadataPort()
    block = RoboflowCustomMetadataBlock(custom_metadata_port=port, api_key=API_KEY)

    # when
    result = _run(block, predictions)

    # then
    assert result["error_status"] is False
    assert port.calls[1]["inference_ids"] == ["inf-meta"]


def test_explicit_inference_id_replaces_the_one_in_metadata() -> None:
    # given
    port = FakeMetadataPort()
    block = RoboflowCustomMetadataBlock(custom_metadata_port=port, api_key=API_KEY)
    predictions = _detections({"inference_id": "inf-meta"})

    # when
    _run(block, predictions, inference_id="inf-explicit")

    # then
    assert port.calls[1]["inference_ids"] == ["inf-explicit"]
    assert predictions.image_metadata == {"inference_id": "inf-meta"}


@pytest.mark.parametrize("metadata", [None, {}, {"other": 1}])
def test_missing_inference_id_reports_v1_error_without_requests(
    metadata: Optional[dict],
) -> None:
    # given
    port = FakeMetadataPort()
    block = RoboflowCustomMetadataBlock(custom_metadata_port=port, api_key=API_KEY)

    # when
    result = _run(block, _detections(metadata))

    # then
    assert result == {
        "error_status": True,
        "message": (
            "Custom metadata upload failed because no inference_ids were received. "
            "For native V2 model predictions, bind the inference_id parameter to "
            "$steps.<model>.inference_id. If inference_id is omitted, the fallback "
            "requires an existing inference_id in prediction image metadata."
        ),
    }
    assert port.calls == []


def test_disabled_sinks_skip_before_any_other_check() -> None:
    # given
    port = FakeMetadataPort()
    block = RoboflowCustomMetadataBlock(custom_metadata_port=port, disable_sinks=True)

    # when
    result = _run(block, _detections({}))

    # then
    assert result == {
        "error_status": False,
        "message": "Sink was disabled by workflow execution policy",
    }
    assert port.calls == []


def test_missing_api_key_fails_the_step_with_v1_message() -> None:
    # given
    port = FakeMetadataPort()
    block = RoboflowCustomMetadataBlock(custom_metadata_port=port)

    # when
    with pytest.raises(ValueError, match="cannot run without a Roboflow API key"):
        _run(block, _detections({}), inference_id="inf-1")

    # then
    assert port.calls == []


def test_replayed_api_key_fails_before_dispatch() -> None:
    # given
    port = FakeMetadataPort()
    block = RoboflowCustomMetadataBlock(
        custom_metadata_port=port, api_key=Secret.unavailable("ROBOFLOW_API_KEY")
    )

    # when
    with pytest.raises(SecretUnavailableError, match="ROBOFLOW_API_KEY"):
        _run(block, _detections({}), inference_id="inf-1")

    # then
    assert port.calls == []


def test_default_port_refuses_and_names_the_resource() -> None:
    # given
    block = RoboflowCustomMetadataBlock(api_key=API_KEY)

    # when
    with pytest.raises(PlatformUnavailableError) as raised:
        _run(block, _detections({}), inference_id="inf-1")

    # then
    assert "'custom_metadata_port'" in str(raised.value)
    assert API_KEY not in str(raised.value)


def test_failed_upload_is_reported_in_outputs_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    error = PlatformRequestError(
        "Unsuccessful request to Roboflow API with response code: 500",
        error_type="inference.core.exceptions.RoboflowAPIUnsuccessfulRequestError",
    )
    port = FakeMetadataPort(add_error=error)
    block = RoboflowCustomMetadataBlock(custom_metadata_port=port, api_key=API_KEY)

    # when
    with caplog.at_level(logging.WARNING):
        result = _run(block, _detections({}), inference_id="inf-1")

    # then
    assert result == {
        "error_status": True,
        "message": (
            "Error while custom metadata registration. Error type: "
            "<class 'inference.core.exceptions.RoboflowAPIUnsuccessfulRequestError'>. "
            "Details: Unsuccessful request to Roboflow API with response code: 500"
        ),
    }
    assert "Could not add custom metadata for inference IDs: ['inf-1']" in caplog.text
    assert API_KEY not in caplog.text + repr(result)


def test_a_failed_workspace_lookup_fails_the_step_as_in_v1() -> None:
    # given
    error = PlatformRequestError(
        "Could not connect", error_type="builtins.ConnectionError"
    )
    port = FakeMetadataPort(workspace_error=error)
    block = RoboflowCustomMetadataBlock(custom_metadata_port=port, api_key=API_KEY)

    # when
    with pytest.raises(PlatformRequestError, match="Could not connect"):
        _run(block, _detections({}), inference_id="inf-1")

    # then
    assert [call["operation"] for call in port.calls] == ["get_workspace"]


def test_params_follow_v1_shape_and_reject_loose_values() -> None:
    # given
    spec = spec_of(RoboflowCustomMetadataBlock)
    step = {
        "predictions": "$steps.model.predictions",
        "field_name": "location",
        "field_value": "$inputs.location",
    }

    # when
    params = spec.validate_params(step)

    # then
    assert params.fire_and_forget is True
    assert params.inference_id is None
    assert spec.identities == (
        "v2/roboflow_custom_metadata@v1",
        "RoboflowCustomMetadata",
    )
    with pytest.raises(ParamsValidationError):
        spec.validate_params({**step, "fire_and_forget": "yes"})
    with pytest.raises(ParamsValidationError):
        spec.validate_params({key: step[key] for key in ("predictions", "field_value")})

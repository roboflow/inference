import base64
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.roboflow.anomaly_detection import v1
from roboflow_workflows.core_steps.models.roboflow.anomaly_detection.v1 import (
    AnomalyDetectionModelBlockV1,
    BlockManifest,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)

TYPE = "roboflow_core/anomaly_detection_model@v1"
MAP = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)


def _map_payload(values: np.ndarray = MAP) -> dict:
    array = np.ascontiguousarray(values, dtype="<f4")
    return {
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "dtype": "float32",
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _response(
    *, score: float = 3.0, threshold: float = 2.0, anomaly_map: Any = None
) -> Dict[str, Any]:
    is_anomalous = score >= threshold
    anomalous_confidence = 0.75 if is_anomalous else 0.25
    classes = [
        {"class": "anomalous", "class_id": 1, "confidence": anomalous_confidence},
        {"class": "normal", "class_id": 0, "confidence": 1 - anomalous_confidence},
    ]
    if not is_anomalous:
        classes.reverse()
    response = {
        "inference_id": "iid-1",
        "time": 0.01,
        "image": {"width": 4, "height": 3},
        "predictions": classes,
        "top": classes[0]["class"],
        "confidence": classes[0]["confidence"],
        "anomaly_score": score,
        "anomaly_threshold": threshold,
        "is_anomalous": is_anomalous,
    }
    if anomaly_map is not None:
        response["anomaly_map"] = anomaly_map
    return response


def _image(parent_id: str = "img") -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=parent_id),
        numpy_image=np.zeros((3, 4, 3), dtype=np.uint8),
    )


class _StubProvider:
    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = responses
        self.add_model = MagicMock()
        self.calls: List[Dict[str, Any]] = []

    def run_classification(self, **kwargs: Any) -> List[dict]:
        self.calls.append(kwargs)
        return [dict(response) for response in self.responses]


def _block(provider: Any, mode: StepExecutionMode) -> AnomalyDetectionModelBlockV1:
    return AnomalyDetectionModelBlockV1(
        model_manager=provider, api_key="key", step_execution_mode=mode
    )


def _run(
    block: AnomalyDetectionModelBlockV1, images: List[WorkflowImageData], **overrides
):
    kwargs = dict(
        model_id="ws/anomaly/1",
        include_anomaly_map=True,
        anomaly_threshold=None,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )
    kwargs.update(overrides)
    return block.run(
        images=Batch.init(content=images, indices=[(i,) for i in range(len(images))]),
        **kwargs
    )


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_manifest_validation_when_minimal_config_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": TYPE,
        "name": "anomaly",
        images_field_alias: "$inputs.image",
        "model_id": "ws/anomaly/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=TYPE, name="anomaly", images="$inputs.image", model_id="ws/anomaly/1"
    )
    assert result.include_anomaly_map is True
    assert result.anomaly_threshold is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("include_anomaly_map", "$inputs.include_map"),
        ("include_anomaly_map", False),
        ("anomaly_threshold", "$inputs.threshold"),
        ("anomaly_threshold", 1.25),
        ("anomaly_threshold", None),
    ],
)
def test_manifest_accepts_selectors_and_literals(field: str, value: Any) -> None:
    # given
    data = {
        "type": TYPE,
        "name": "anomaly",
        "images": "$inputs.image",
        "model_id": "a/1",
    }
    data[field] = value

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert getattr(result, field) == value


@pytest.mark.parametrize("field", ["type", "name", "images", "model_id"])
def test_manifest_validation_when_required_field_is_missing(field: str) -> None:
    # given
    data = {
        "type": TYPE,
        "name": "anomaly",
        "images": "$inputs.image",
        "model_id": "a/1",
    }
    del data[field]

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


@pytest.mark.parametrize(
    "field, value",
    [
        ("anomaly_threshold", "high"),
        ("include_anomaly_map", "maybe"),
        ("model_id", None),
    ],
)
def test_manifest_validation_when_field_has_invalid_type(
    field: str, value: Any
) -> None:
    # given
    data = {
        "type": TYPE,
        "name": "anomaly",
        "images": "$inputs.image",
        "model_id": "a/1",
    }
    data[field] = value

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_manifest_outputs_are_declared_in_contract_order() -> None:
    names = [output.name for output in BlockManifest.describe_outputs()]
    assert names == [
        "predictions",
        "anomaly_score",
        "anomaly_threshold",
        "is_anomalous",
        "anomaly_map",
        "inference_id",
        "model_id",
    ]


def test_run_locally_forwards_the_map_flag_and_decodes_the_payload() -> None:
    # given
    provider = _StubProvider([_response(anomaly_map=_map_payload())])
    block = _block(provider, StepExecutionMode.LOCAL)

    # when
    result = _run(block, [_image()])

    # then
    provider.add_model.assert_called_once_with(model_id="ws/anomaly/1", api_key="key")
    assert provider.calls[0]["include_anomaly_map"] is True
    assert provider.calls[0]["images"] == [
        {"type": "numpy_object", "value": provider.calls[0]["images"][0]["value"]}
    ]
    assert len(result) == 1
    assert result[0]["anomaly_score"] == 3.0
    assert result[0]["anomaly_threshold"] == 2.0
    assert result[0]["is_anomalous"] is True
    assert result[0]["inference_id"] == "iid-1"
    assert result[0]["model_id"] == "ws/anomaly/1"
    np.testing.assert_array_equal(result[0]["anomaly_map"], MAP)
    assert result[0]["anomaly_map"].dtype == np.float32
    predictions = result[0]["predictions"]
    assert "anomaly_map" not in predictions
    assert predictions["top"] == "anomalous"
    assert predictions["prediction_type"] == "classification"
    assert predictions["parent_id"] == "img"
    assert predictions["root_parent_id"] == "img"


def test_run_locally_without_map_outputs_none() -> None:
    # given
    provider = _StubProvider([_response()])
    block = _block(provider, StepExecutionMode.LOCAL)

    # when
    result = _run(block, [_image()], include_anomaly_map=False)

    # then
    assert provider.calls[0]["include_anomaly_map"] is False
    assert result[0]["anomaly_map"] is None


def test_run_locally_applies_the_threshold_override() -> None:
    # given: the model says normal (1.5 < 2.0)
    provider = _StubProvider([_response(score=1.5, threshold=2.0)])
    block = _block(provider, StepExecutionMode.LOCAL)

    # when
    result = _run(block, [_image()], anomaly_threshold=1.0)

    # then
    assert result[0]["is_anomalous"] is True
    assert result[0]["anomaly_threshold"] == 1.0
    assert result[0]["anomaly_score"] == 1.5
    assert result[0]["predictions"]["top"] == "anomalous"
    assert result[0]["predictions"]["anomaly_threshold"] == 1.0
    assert result[0]["predictions"]["is_anomalous"] is True


def test_run_rejects_a_plain_classifier_response() -> None:
    # given
    plain = {
        "predictions": [{"class": "cat", "class_id": 0, "confidence": 0.9}],
        "top": "cat",
        "confidence": 0.9,
    }
    block = _block(_StubProvider([plain]), StepExecutionMode.LOCAL)

    # when / then
    with pytest.raises(ValueError, match="ws/anomaly/1"):
        _run(block, [_image()])


@pytest.mark.parametrize("invalid", [float("nan"), "0.5"])
def test_run_rejects_an_invalid_threshold_override(invalid: Any) -> None:
    block = _block(_StubProvider([_response()]), StepExecutionMode.LOCAL)
    with pytest.raises(ValueError):
        _run(block, [_image()], anomaly_threshold=invalid)


def test_run_locally_handles_a_batch_in_order() -> None:
    # given
    provider = _StubProvider([_response(score=5.0), _response(score=0.5)])
    block = _block(provider, StepExecutionMode.LOCAL)

    # when
    result = _run(block, [_image("a"), _image("b")])

    # then
    assert [r["is_anomalous"] for r in result] == [True, False]
    assert [r["predictions"]["parent_id"] for r in result] == ["a", "b"]


@pytest.mark.parametrize(
    "served_map", [_map_payload(), MAP.copy()], ids=["payload", "ndarray"]
)
def test_run_remotely_configures_the_client_and_accepts_both_map_forms(
    served_map,
) -> None:
    # given
    block = _block(MagicMock(), StepExecutionMode.REMOTE)
    with patch.object(v1, "InferenceHTTPClient") as client_cls:
        client = client_cls.return_value
        client.infer.return_value = [_response(anomaly_map=served_map)]

        # when
        result = _run(block, [_image()], anomaly_threshold=2.5)

    # then
    configuration = client.configure.call_args.kwargs["inference_configuration"]
    assert configuration.include_anomaly_map is True
    assert configuration.disable_active_learning is True
    assert configuration.source == "workflow-execution"
    client.infer.assert_called_once()
    assert client.infer.call_args.kwargs["model_id"] == "ws/anomaly/1"
    np.testing.assert_array_equal(result[0]["anomaly_map"], MAP)
    assert result[0]["anomaly_threshold"] == 2.5
    assert result[0]["is_anomalous"] is True
    assert "anomaly_map" not in result[0]["predictions"]


def test_run_remotely_wraps_a_single_response() -> None:
    # given
    block = _block(MagicMock(), StepExecutionMode.REMOTE)
    with patch.object(v1, "InferenceHTTPClient") as client_cls:
        client_cls.return_value.infer.return_value = _response()

        # when
        result = _run(block, [_image()], include_anomaly_map=False)

    # then
    assert len(result) == 1
    assert result[0]["anomaly_map"] is None
    configuration = client_cls.return_value.configure.call_args.kwargs[
        "inference_configuration"
    ]
    assert configuration.include_anomaly_map is False

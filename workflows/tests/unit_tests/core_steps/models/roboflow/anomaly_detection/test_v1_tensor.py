from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.roboflow.anomaly_detection import v1_tensor
from roboflow_workflows.core_steps.models.roboflow.anomaly_detection.v1 import (
    BlockManifest as NumpyManifest,
)
from roboflow_workflows.core_steps.models.roboflow.anomaly_detection.v1_tensor import (
    AnomalyDetectionModelBlockV1,
    BlockManifest,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)

from inference_models.models.base.anomaly_detection import AnomalyDetectionPrediction
from inference_models.models.base.classification import ClassificationPrediction

MAP = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)


def _image(parent_id: str = "img") -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=parent_id),
        numpy_image=np.zeros((3, 4, 3), dtype=np.uint8),
    )


def _batched(
    scores: List[float], threshold: float = 2.0, with_map: bool = True
) -> AnomalyDetectionPrediction:
    scores_tensor = torch.tensor(scores, dtype=torch.float64)
    return AnomalyDetectionPrediction(
        anomaly_scores=scores_tensor,
        anomaly_threshold=threshold,
        is_anomalous=scores_tensor >= threshold,
        anomaly_maps=(
            torch.from_numpy(np.stack([MAP] * len(scores))) if with_map else None
        ),
    )


class _StubProvider:
    def __init__(self, prediction: Any):
        self.prediction = prediction
        self.add_model = MagicMock()
        self.calls: List[Dict[str, Any]] = []

    def run_tensor_native_inference(self, model_id: str, **kwargs: Any):
        self.calls.append({"model_id": model_id, **kwargs})
        return self.prediction


def _block(provider: Any, mode: StepExecutionMode) -> AnomalyDetectionModelBlockV1:
    return AnomalyDetectionModelBlockV1(
        model_manager=provider, api_key="key", step_execution_mode=mode
    )


def _run(block, images: List[WorkflowImageData], **overrides):
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
        **kwargs,
    )


def _remote_response(score: float = 3.0, threshold: float = 2.0) -> dict:
    return {
        "inference_id": "iid-1",
        "time": 0.01,
        "image": {"width": 4, "height": 3},
        "anomaly_score": score,
        "anomaly_threshold": threshold,
        "is_anomalous": score >= threshold,
        "anomaly_map": MAP.copy(),
    }


def test_sibling_manifests_declare_the_same_surface() -> None:
    assert set(BlockManifest.model_fields) == set(NumpyManifest.model_fields)
    numpy_outputs = [
        (o.name, [k.name for k in o.kind]) for o in NumpyManifest.describe_outputs()
    ]
    tensor_outputs = [
        (o.name, [k.name for k in o.kind]) for o in BlockManifest.describe_outputs()
    ]
    assert numpy_outputs == tensor_outputs
    assert "predictions" not in [name for name, _ in tensor_outputs]


def test_run_locally_fans_out_the_batched_prediction() -> None:
    # given
    provider = _StubProvider(_batched([3.0, 0.5]))
    block = _block(provider, StepExecutionMode.LOCAL)

    # when
    result = _run(block, [_image("a"), _image("b")])

    # then
    provider.add_model.assert_called_once_with(model_id="ws/anomaly/1", api_key="key")
    assert provider.calls[0]["include_anomaly_map"] is True
    assert provider.calls[0]["anomaly_threshold"] is None
    assert provider.calls[0]["input_color_format"] == "bgr"
    assert [r["is_anomalous"] for r in result] == [True, False]
    assert [r["anomaly_score"] for r in result] == [3.0, 0.5]
    assert [r["anomaly_threshold"] for r in result] == [2.0, 2.0]
    assert [r["model_id"] for r in result] == ["ws/anomaly/1"] * 2
    assert len({r["inference_id"] for r in result}) == 2
    for entry in result:
        np.testing.assert_array_equal(entry["anomaly_map"], MAP)
        assert entry["anomaly_map"].dtype == np.float32
        assert isinstance(entry["anomaly_map"], np.ndarray)
        assert set(entry) == {
            "anomaly_score",
            "anomaly_threshold",
            "is_anomalous",
            "anomaly_map",
            "inference_id",
            "model_id",
        }


def test_run_locally_without_map_forwards_the_threshold() -> None:
    # given: the model decided against the caller's threshold of 1.0
    provider = _StubProvider(_batched([1.5], threshold=1.0, with_map=False))
    block = _block(provider, StepExecutionMode.LOCAL)

    # when
    result = _run(block, [_image()], include_anomaly_map=False, anomaly_threshold=1.0)

    # then
    assert provider.calls[0]["include_anomaly_map"] is False
    assert provider.calls[0]["anomaly_threshold"] == 1.0
    assert result[0]["anomaly_map"] is None
    assert result[0]["is_anomalous"] is True
    assert result[0]["anomaly_threshold"] == 1.0
    assert result[0]["anomaly_score"] == 1.5


def test_run_locally_rejects_a_plain_classifier() -> None:
    # given: a classifier returns a classification prediction, not a verdict
    plain = ClassificationPrediction(
        class_id=torch.tensor([0]), confidence=torch.tensor([[0.9, 0.1]])
    )
    block = _block(_StubProvider(plain), StepExecutionMode.LOCAL)

    # when / then
    with pytest.raises(ValueError, match="ws/anomaly/1"):
        _run(block, [_image()])


def test_run_locally_fails_when_the_requested_maps_are_missing() -> None:
    # given
    block = _block(
        _StubProvider(_batched([1.5], with_map=False)), StepExecutionMode.LOCAL
    )

    # when / then
    with pytest.raises(ValueError, match="anomaly maps"):
        _run(block, [_image()], include_anomaly_map=True)


def test_run_locally_rejects_a_verdict_count_mismatch() -> None:
    # given
    block = _block(_StubProvider(_batched([1.5])), StepExecutionMode.LOCAL)

    # when / then
    with pytest.raises(ValueError, match="1 verdict"):
        _run(block, [_image("a"), _image("b")])


def test_run_remotely_reads_the_server_response() -> None:
    # given
    block = _block(MagicMock(), StepExecutionMode.REMOTE)
    with patch.object(v1_tensor, "InferenceHTTPClient") as client_cls:
        client = client_cls.return_value
        client.infer.return_value = [_remote_response(score=3.0, threshold=2.5)]

        # when
        result = _run(block, [_image()], anomaly_threshold=2.5)

    # then
    configuration = client.configure.call_args.kwargs["inference_configuration"]
    assert configuration.include_anomaly_map is True
    assert configuration.anomaly_threshold == 2.5
    assert result[0]["anomaly_score"] == 3.0
    assert result[0]["anomaly_threshold"] == 2.5
    assert result[0]["is_anomalous"] is True
    assert result[0]["inference_id"] == "iid-1"
    np.testing.assert_array_equal(result[0]["anomaly_map"], MAP)

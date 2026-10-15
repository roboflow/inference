from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.serializers_tensor import (
    serialise_native_classification,
)
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

from inference_models.models.base.classification import ClassificationPrediction

MAP = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)


def _image(parent_id: str = "img") -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=parent_id),
        numpy_image=np.zeros((3, 4, 3), dtype=np.uint8),
    )


def _batched(scores: List[float], threshold: float = 2.0, with_map: bool = True):
    metadata = []
    for score in scores:
        entry: Dict[str, Any] = {
            "anomaly_score": score,
            "anomaly_threshold": threshold,
            "is_anomalous": score >= threshold,
        }
        if with_map:
            entry["anomaly_map"] = MAP.copy()
        metadata.append(entry)
    anomalous = torch.tensor([[0.75 if s >= threshold else 0.25] for s in scores])
    return ClassificationPrediction(
        class_id=torch.tensor([int(s >= threshold) for s in scores]),
        confidence=torch.cat([1 - anomalous, anomalous], dim=1).float(),
        images_metadata=metadata,
    )


class _StubProvider:
    def __init__(self, prediction: ClassificationPrediction):
        self.prediction = prediction
        self.add_model = MagicMock()
        self.calls: List[Dict[str, Any]] = []

    def run_tensor_native_inference(self, model_id: str, **kwargs: Any):
        self.calls.append({"model_id": model_id, **kwargs})
        return self.prediction

    def get_class_names(self, model_id: str) -> List[str]:
        return ["normal", "anomalous"]


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
    is_anomalous = score >= threshold
    anomalous_confidence = 0.75 if is_anomalous else 0.25
    classes = [
        {"class": "anomalous", "class_id": 1, "confidence": anomalous_confidence},
        {"class": "normal", "class_id": 0, "confidence": 1 - anomalous_confidence},
    ]
    if not is_anomalous:
        classes.reverse()
    return {
        "inference_id": "iid-1",
        "time": 0.01,
        "image": {"width": 4, "height": 3},
        "predictions": classes,
        "top": classes[0]["class"],
        "confidence": classes[0]["confidence"],
        "anomaly_score": score,
        "anomaly_threshold": threshold,
        "is_anomalous": is_anomalous,
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


def test_run_locally_fans_out_the_batched_prediction() -> None:
    # given
    provider = _StubProvider(_batched([3.0, 0.5]))
    block = _block(provider, StepExecutionMode.LOCAL)

    # when
    result = _run(block, [_image("a"), _image("b")])

    # then
    provider.add_model.assert_called_once_with(model_id="ws/anomaly/1", api_key="key")
    assert provider.calls[0]["include_anomaly_map"] is True
    assert provider.calls[0]["input_color_format"] == "bgr"
    assert [r["is_anomalous"] for r in result] == [True, False]
    assert [r["anomaly_score"] for r in result] == [3.0, 0.5]
    assert [r["anomaly_threshold"] for r in result] == [2.0, 2.0]
    for entry in result:
        np.testing.assert_array_equal(entry["anomaly_map"], MAP)
        prediction = entry["predictions"]
        assert isinstance(prediction, ClassificationPrediction)
        assert prediction.class_id.shape == (1,)
        assert prediction.confidence.shape == (1, 2)
        metadata = prediction.images_metadata[0]
        assert metadata["class_names"] == {0: "normal", 1: "anomalous"}
        assert "anomaly_map" not in metadata
        assert metadata["inference_id"] == entry["inference_id"]
    assert result[0]["predictions"].class_id.tolist() == [1]
    assert result[1]["predictions"].class_id.tolist() == [0]
    assert result[0]["predictions"].images_metadata[0]["parent_id"] == "a"
    assert result[1]["predictions"].images_metadata[0]["parent_id"] == "b"


def test_run_locally_without_map_and_with_override() -> None:
    # given: the model says normal at 2.0; the override at 1.0 flips it
    provider = _StubProvider(_batched([1.5], with_map=False))
    block = _block(provider, StepExecutionMode.LOCAL)

    # when
    result = _run(block, [_image()], include_anomaly_map=False, anomaly_threshold=1.0)

    # then
    assert provider.calls[0]["include_anomaly_map"] is False
    assert result[0]["anomaly_map"] is None
    assert result[0]["is_anomalous"] is True
    assert result[0]["anomaly_threshold"] == 1.0
    assert result[0]["predictions"].class_id.tolist() == [1]
    assert result[0]["predictions"].images_metadata[0]["is_anomalous"] is True


def test_run_locally_rejects_a_plain_classifier() -> None:
    # given: a classifier returns no per-image anomaly metadata
    plain = ClassificationPrediction(
        class_id=torch.tensor([0]), confidence=torch.tensor([[0.9, 0.1]])
    )
    block = _block(_StubProvider(plain), StepExecutionMode.LOCAL)

    # when / then
    with pytest.raises(ValueError, match="ws/anomaly/1"):
        _run(block, [_image()])


def test_serialised_local_prediction_matches_the_numpy_dict_shape() -> None:
    # given
    provider = _StubProvider(_batched([1.5]))
    block = _block(provider, StepExecutionMode.LOCAL)
    result = _run(block, [_image()], anomaly_threshold=1.0)

    # when
    serialised = serialise_native_classification(result[0]["predictions"])

    # then: the override decided "anomalous" although its confidence is lower
    assert list(serialised.keys()) == [
        "inference_id",
        "time",
        "image",
        "predictions",
        "top",
        "confidence",
        "anomaly_score",
        "anomaly_threshold",
        "is_anomalous",
        "prediction_type",
        "parent_id",
        "root_parent_id",
    ]
    assert serialised["image"] == {"width": 4, "height": 3}
    assert [p["class"] for p in serialised["predictions"]] == ["anomalous", "normal"]
    assert serialised["top"] == "anomalous"
    assert serialised["confidence"] == 0.25
    assert serialised["anomaly_score"] == 1.5
    assert serialised["anomaly_threshold"] == 1.0
    assert serialised["is_anomalous"] is True
    assert "anomaly_map" not in serialised


def test_run_remotely_rebuilds_a_native_prediction() -> None:
    # given
    block = _block(MagicMock(), StepExecutionMode.REMOTE)
    with patch.object(v1_tensor, "InferenceHTTPClient") as client_cls:
        client = client_cls.return_value
        client.infer.return_value = [_remote_response()]

        # when
        result = _run(block, [_image()])

    # then
    configuration = client.configure.call_args.kwargs["inference_configuration"]
    assert configuration.include_anomaly_map is True
    prediction = result[0]["predictions"]
    assert isinstance(prediction, ClassificationPrediction)
    assert prediction.class_id.tolist() == [1]
    assert prediction.confidence.tolist() == [[0.25, 0.75]]
    assert prediction.images_metadata[0]["class_names"] == {0: "normal", 1: "anomalous"}
    assert prediction.images_metadata[0]["inference_id"] == "iid-1"
    np.testing.assert_array_equal(result[0]["anomaly_map"], MAP)
    serialised = serialise_native_classification(prediction)
    assert serialised["top"] == "anomalous"
    assert serialised["anomaly_score"] == 3.0

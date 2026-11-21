from typing import Any

import numpy as np
import pytest
import supervision as sv
import torch
from pydantic import ValidationError
from roboflow_workflows.core_steps.transformations.per_class_confidence_filter.v1 import (
    BlockManifest,
    PerClassConfidenceFilterBlockV1,
    filter_detections_by_class_confidence,
)
from roboflow_workflows.core_steps.transformations.per_class_confidence_filter.v1_tensor import (
    filter_detections_by_class_confidence as filter_native_detections,
)

from inference_models.models.base.object_detection import Detections


def test_manifest_parsing_when_valid_data_provided() -> None:
    # given
    data = {
        "type": "roboflow_core/per_class_confidence_filter@v1",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "class_thresholds": {"person": 0.98, "car": 0.5},
        "default_threshold": 0.25,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/per_class_confidence_filter@v1",
        name="some",
        predictions="$steps.some.predictions",
        class_thresholds={"person": 0.98, "car": 0.5},
        default_threshold=0.25,
    )


def test_manifest_parsing_uses_default_threshold() -> None:
    # given
    data = {
        "type": "roboflow_core/per_class_confidence_filter@v1",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "class_thresholds": {"person": 0.98},
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.default_threshold == 0.3


def test_manifest_parsing_accepts_selector_for_class_thresholds() -> None:
    # given
    data = {
        "type": "roboflow_core/per_class_confidence_filter@v1",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "class_thresholds": "$inputs.class_thresholds",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.class_thresholds == "$inputs.class_thresholds"


def test_manifest_parsing_when_invalid_predictions_selector_provided() -> None:
    # given
    data = {
        "type": "roboflow_core/per_class_confidence_filter@v1",
        "name": "some",
        "predictions": "invalid",
        "class_thresholds": {"person": 0.98},
        "default_threshold": 0.3,
    }

    # when / then
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def _make_detections(class_names: list[str], confidences: list[float]) -> sv.Detections:
    n = len(class_names)
    return sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]] * n, dtype=np.float64),
        class_id=np.arange(n),
        confidence=np.array(confidences, dtype=np.float64),
        data={
            "class_name": np.array(class_names),
            "detection_id": np.array([f"d{i}" for i in range(n)]),
        },
    )


def test_filter_when_all_detections_pass() -> None:
    # given
    detections = _make_detections(
        class_names=["person", "car"], confidences=[0.99, 0.6]
    )

    # when
    result = filter_detections_by_class_confidence(
        detections=detections,
        class_thresholds={"person": 0.98, "car": 0.5},
        default_threshold=0.3,
    )

    # then
    assert len(result) == 2
    assert list(result.data["class_name"]) == ["person", "car"]


def test_filter_drops_below_per_class_threshold() -> None:
    # given
    detections = _make_detections(
        class_names=["person", "person", "car"],
        confidences=[0.99, 0.7, 0.6],
    )

    # when
    result = filter_detections_by_class_confidence(
        detections=detections,
        class_thresholds={"person": 0.98, "car": 0.5},
        default_threshold=0.3,
    )

    # then
    assert len(result) == 2
    assert list(result.data["class_name"]) == ["person", "car"]
    assert list(result.confidence) == pytest.approx([0.99, 0.6])


def test_filter_boundary_is_inclusive() -> None:
    # given
    detections = _make_detections(class_names=["car"], confidences=[0.5])

    # when
    result = filter_detections_by_class_confidence(
        detections=detections,
        class_thresholds={"car": 0.5},
        default_threshold=0.3,
    )

    # then
    assert len(result) == 1


def test_filter_unknown_class_falls_back_to_default_threshold() -> None:
    # given
    detections = _make_detections(class_names=["dog", "dog"], confidences=[0.25, 0.4])

    # when
    result = filter_detections_by_class_confidence(
        detections=detections,
        class_thresholds={"person": 0.98},
        default_threshold=0.3,
    )

    # then
    assert len(result) == 1
    assert list(result.data["class_name"]) == ["dog"]
    assert list(result.confidence) == pytest.approx([0.4])


def test_filter_when_all_detections_fail() -> None:
    # given
    detections = _make_detections(class_names=["person", "car"], confidences=[0.5, 0.1])

    # when
    result = filter_detections_by_class_confidence(
        detections=detections,
        class_thresholds={"person": 0.98, "car": 0.5},
        default_threshold=0.3,
    )

    # then
    assert len(result) == 0


def test_filter_when_predictions_are_empty() -> None:
    # given
    detections = sv.Detections.empty()

    # when
    result = filter_detections_by_class_confidence(
        detections=detections,
        class_thresholds={"person": 0.98},
        default_threshold=0.3,
    )

    # then
    assert len(result) == 0


def test_filter_when_class_thresholds_is_empty_uses_default() -> None:
    # given
    detections = _make_detections(class_names=["person", "car"], confidences=[0.5, 0.2])

    # when
    result = filter_detections_by_class_confidence(
        detections=detections,
        class_thresholds={},
        default_threshold=0.3,
    )

    # then
    assert len(result) == 1
    assert list(result.data["class_name"]) == ["person"]


def test_block_run_processes_a_batch() -> None:
    # given
    block = PerClassConfidenceFilterBlockV1()
    batch = [
        _make_detections(class_names=["person", "car"], confidences=[0.99, 0.4]),
        _make_detections(class_names=["car"], confidences=[0.55]),
    ]

    # when
    result = block.run(
        predictions=batch,
        class_thresholds={"person": 0.98, "car": 0.5},
        default_threshold=0.3,
    )

    # then
    assert len(result) == 2
    assert len(result[0]["predictions"]) == 1
    assert result[0]["predictions"].data["class_name"][0] == "person"
    assert len(result[1]["predictions"]) == 1
    assert result[1]["predictions"].data["class_name"][0] == "car"


@pytest.mark.parametrize(
    ("dtype", "threshold", "kept_rows"),
    [
        (torch.float16, 0.9, [2]),
        (torch.bfloat16, 0.7, [2]),
        (torch.float32, 0.7, [2]),
        (torch.float32, 0.3, [1, 2]),
        (torch.float32, 0.5, [1, 2]),
        (torch.float64, 0.7, [1, 2]),
    ],
)
@pytest.mark.parametrize("use_class_threshold", [False, True])
def test_native_filter_compares_to_configured_threshold(
    dtype: torch.dtype,
    threshold: float,
    kept_rows: list[int],
    use_class_threshold: bool,
) -> None:
    center = torch.tensor(threshold, dtype=dtype)
    confidence = torch.stack(
        [
            torch.nextafter(center, torch.zeros_like(center)),
            center,
            torch.nextafter(center, torch.ones_like(center)),
        ]
    )
    prediction = Detections(
        xyxy=torch.zeros((3, 4)),
        confidence=confidence,
        class_id=torch.tensor([0, 0, 0]),
        image_metadata={"class_names": {0: "person"}},
        bboxes_metadata=[{"detection_id": i} for i in range(3)],
    )

    filtered = filter_native_detections(
        prediction=prediction,
        class_thresholds={"person": threshold} if use_class_threshold else {},
        default_threshold=0.0 if use_class_threshold else threshold,
    )

    assert [row["detection_id"] for row in filtered.bboxes_metadata] == kept_rows
    assert torch.equal(filtered.confidence, confidence[kept_rows])
    assert filtered.confidence.dtype == dtype
    assert filtered.image_metadata == prediction.image_metadata
    assert len(prediction) == 3


def test_native_filter_compares_integral_scores_to_fractional_threshold() -> None:
    prediction = Detections(
        xyxy=torch.zeros((2, 4)),
        confidence=torch.tensor([0, 1]),
        class_id=torch.tensor([0, 0]),
    )

    filtered = filter_native_detections(
        prediction=prediction, class_thresholds={}, default_threshold=0.5
    )

    assert filtered.confidence.tolist() == [1]
    assert filtered.confidence.dtype == prediction.confidence.dtype


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS")
def test_native_filter_keeps_subnormal_mps_threshold_positive() -> None:
    smallest_positive = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    device = torch.device("mps")
    prediction = Detections(
        xyxy=torch.zeros((3, 4), device=device),
        confidence=torch.tensor([0.0, smallest_positive.item(), 1.0]).to(device),
        class_id=torch.zeros((3,), dtype=torch.long, device=device),
        bboxes_metadata=[{"detection_id": row} for row in range(3)],
    )

    filtered = filter_native_detections(
        prediction=prediction, class_thresholds={}, default_threshold=1e-46
    )

    assert [row["detection_id"] for row in filtered.bboxes_metadata] == [1, 2]
    assert filtered.confidence.device.type == device.type


@pytest.mark.parametrize("threshold", [float("nan"), -1.0, 300.0])
def test_native_filter_rejects_unrepresentable_integral_threshold(
    threshold: float,
) -> None:
    prediction = Detections(
        xyxy=torch.zeros((1, 4)),
        confidence=torch.tensor([0], dtype=torch.uint8),
        class_id=torch.zeros((1,), dtype=torch.long),
    )

    with pytest.raises((ValueError, RuntimeError), match="cannot .*convert"):
        filter_native_detections(
            prediction=prediction, class_thresholds={}, default_threshold=threshold
        )

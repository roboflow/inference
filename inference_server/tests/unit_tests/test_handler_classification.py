from __future__ import annotations

import json

import pytest
import torch

from inference_models import ClassificationPrediction
from inference_models.models.common.anomaly_detection import (
    AnomalyCalibration,
    AnomalyRawPrediction,
    post_process_anomaly_scores,
)
from inference_server.framework.entities import CommonRequestParams
from inference_server.handlers.classification.output_serializer import (
    serialize_classification,
)


def _anomaly_prediction(include_anomaly_map: bool) -> ClassificationPrediction:
    raw = AnomalyRawPrediction(
        scores=torch.tensor([3.0]),
        maps=torch.tensor([[[0.5, 1.0], [2.0, 3.0]]]),
        original_sizes_hw=[(8, 8)],
    )
    prediction = post_process_anomaly_scores(
        raw,
        calibration=AnomalyCalibration(threshold=2.0, scale=1.0),
        include_anomaly_map=include_anomaly_map,
    )

    return prediction


def _common(style: str) -> CommonRequestParams:
    return CommonRequestParams(model_id="m/1", api_key="", response_style=style)


@pytest.mark.parametrize("style", ["compact", "rich"])
def test_anomaly_prediction_keeps_score_threshold_and_decision(style):
    response = serialize_classification(
        _anomaly_prediction(include_anomaly_map=False), _common(style)
    )

    item = json.loads(response.body)["predictions"][0]
    assert item["anomaly_score"] == 3.0
    assert item["anomaly_threshold"] == 2.0
    assert item["is_anomalous"] is True
    assert "anomaly_map" not in item


@pytest.mark.parametrize("style", ["compact", "rich"])
def test_anomaly_map_is_serialized_only_when_requested(style):
    response = serialize_classification(
        _anomaly_prediction(include_anomaly_map=True), _common(style)
    )

    item = json.loads(response.body)["predictions"][0]
    assert item["anomaly_map"] == [[0.5, 1.0], [2.0, 3.0]]


@pytest.mark.parametrize(
    "style, expected",
    [
        (
            "compact",
            b'{"type":"roboflow-classification-compact-v1","class_names":null,'
            b'"confidences":[[0.25,0.75]],"top_classes_ids":[1]}',
        ),
        (
            "rich",
            b'{"type":"roboflow-classification-rich-v1","candidates":'
            b'[{"class_id":1,"confidence":0.75},{"class_id":0,"confidence":0.25}],'
            b'"top":[{"class_id":1,"confidence":0.75}]}',
        ),
    ],
)
def test_plain_classification_response_bytes_are_unchanged(style, expected):
    prediction = ClassificationPrediction(
        class_id=torch.tensor([1]), confidence=torch.tensor([[0.25, 0.75]])
    )

    response = serialize_classification(prediction, _common(style))

    assert response.body == (
        b'{"type":"roboflow-inference-server-response-v1","model_info":'
        b'{"model_id":"m/1","task":"classification"},"usage":{},"predictions":['
        + expected
        + b"]}"
    )

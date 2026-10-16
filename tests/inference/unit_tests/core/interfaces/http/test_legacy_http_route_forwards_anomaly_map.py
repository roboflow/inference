"""The legacy ``POST /{dataset_id}/{version_id}`` route builds its request from
named query params, so ``include_anomaly_map`` and ``anomaly_threshold`` must be
declared there for the platform's inference preview to receive anomaly heatmaps
and to decide against a caller threshold."""

import pytest
from starlette.testclient import TestClient

from tests.inference.unit_tests.core.interfaces.http.test_legacy_http_route_accepts_confidence_modes import (
    _build_interface,
)


@pytest.mark.parametrize("include_anomaly_map", [True, False])
def test_legacy_classification_route_forwards_include_anomaly_map(
    monkeypatch, include_anomaly_map: bool
) -> None:
    interface, model_manager = _build_interface(monkeypatch, task_type="classification")
    params = {"api_key": "query-api-key", "image": "https://example.com/test.jpg"}
    if include_anomaly_map:
        params["include_anomaly_map"] = "true"

    with TestClient(interface.app) as client:
        response = client.post("/dummy-dataset/1", params=params)

    assert response.status_code == 200, response.text
    inference_request = model_manager.infer_from_request_sync.call_args.args[1]
    assert inference_request.include_anomaly_map is include_anomaly_map


@pytest.mark.parametrize("anomaly_threshold", [None, 0.0, 1.25])
def test_legacy_classification_route_forwards_anomaly_threshold(
    monkeypatch, anomaly_threshold
) -> None:
    interface, model_manager = _build_interface(monkeypatch, task_type="classification")
    params = {"api_key": "query-api-key", "image": "https://example.com/test.jpg"}
    if anomaly_threshold is not None:
        params["anomaly_threshold"] = str(anomaly_threshold)

    with TestClient(interface.app) as client:
        response = client.post("/dummy-dataset/1", params=params)

    assert response.status_code == 200, response.text
    inference_request = model_manager.infer_from_request_sync.call_args.args[1]
    assert inference_request.anomaly_threshold == anomaly_threshold


@pytest.mark.parametrize("anomaly_threshold", ["nan", "inf", "high"])
def test_legacy_classification_route_rejects_non_finite_anomaly_threshold(
    monkeypatch, anomaly_threshold: str
) -> None:
    interface, model_manager = _build_interface(monkeypatch, task_type="classification")
    params = {
        "api_key": "query-api-key",
        "image": "https://example.com/test.jpg",
        "anomaly_threshold": anomaly_threshold,
    }

    with TestClient(interface.app) as client:
        response = client.post("/dummy-dataset/1", params=params)

    assert response.status_code == 422, response.text
    model_manager.infer_from_request_sync.assert_not_called()


def test_legacy_route_maps_anomaly_threshold_contradiction_to_400(monkeypatch) -> None:
    """A plain classifier given `anomaly_threshold` is a bad request, not a 500."""
    from inference.core.exceptions import RequestDataContradiction

    interface, model_manager = _build_interface(monkeypatch, task_type="classification")
    model_manager.infer_from_request_sync.side_effect = RequestDataContradiction(
        "`anomaly_threshold` was sent for model `dummy-dataset/1`, which is not an "
        "anomaly detection model."
    )
    params = {
        "api_key": "query-api-key",
        "image": "https://example.com/test.jpg",
        "anomaly_threshold": "1.25",
    }

    with TestClient(interface.app) as client:
        response = client.post("/dummy-dataset/1", params=params)

    assert response.status_code == 400, response.text
    assert "anomaly_threshold" in response.json()["message"]

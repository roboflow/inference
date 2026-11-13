"""The legacy ``POST /{dataset_id}/{version_id}`` route builds its request from
named query params, so ``include_anomaly_map`` must be declared there for the
platform's inference preview to receive anomaly heatmaps."""

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


def test_legacy_classification_route_leaves_omitted_confidence_unset(
    monkeypatch,
) -> None:
    interface, model_manager = _build_interface(monkeypatch, task_type="classification")

    with TestClient(interface.app) as client:
        omitted = client.post(
            "/dummy-dataset/1",
            params={"api_key": "query-api-key", "image": "https://example.com/a.jpg"},
        )
        omitted_request = model_manager.infer_from_request_sync.call_args.args[1]
        explicit = client.post(
            "/dummy-dataset/1",
            params={
                "api_key": "query-api-key",
                "image": "https://example.com/a.jpg",
                "confidence": "0.4",
            },
        )
        explicit_request = model_manager.infer_from_request_sync.call_args.args[1]

    assert omitted.status_code == 200, omitted.text
    assert explicit.status_code == 200, explicit.text
    assert "confidence" not in omitted_request.model_fields_set
    assert "confidence" in explicit_request.model_fields_set
    assert explicit_request.confidence == 0.4

from unittest import mock

import pytest
import requests
import requests_mock

from inference_sdk.cloud_streaming import api as api_module
from inference_sdk.cloud_streaming.api import CloudStreamingAPI, CloudStreamingError

API_URL = "https://api.example.test"


@pytest.fixture(autouse=True)
def no_sleep():
    with mock.patch.object(api_module.time, "sleep"):
        yield


def test_resolves_the_workspace_from_the_key_once() -> None:
    api = CloudStreamingAPI("key", api_url=API_URL)
    with requests_mock.Mocker() as http:
        root = http.get(f"{API_URL}/", json={"workspace": "acme"})
        http.get(f"{API_URL}/acme/video-sources/v1", json={"sources": []})

        api.list_sources()
        api.list_sources()

    assert root.call_count == 1
    assert http.request_history[0].headers["Authorization"] == "Bearer key"


@pytest.mark.parametrize(
    "body, message",
    [
        ({"error": "resource not found"}, "resource not found"),
        (
            {"error": {"message": "invalid request", "type": "OAuthException"}},
            "invalid request",
        ),
    ],
)
def test_reports_the_api_error_message_in_either_shape(body, message) -> None:
    api = CloudStreamingAPI("key", workspace="acme", api_url=API_URL)
    with requests_mock.Mocker() as http:
        http.get(f"{API_URL}/acme/video-jobs/v1/job-1", status_code=404, json=body)

        with pytest.raises(CloudStreamingError) as error:
            api.get_job("job-1")

    assert error.value.status == 404
    assert str(error.value).endswith(message)


def test_resends_an_idempotent_request_with_the_same_key_until_it_lands() -> None:
    api = CloudStreamingAPI("key", workspace="acme", api_url=API_URL)
    with requests_mock.Mocker() as http:
        start = http.post(
            f"{API_URL}/acme/video-sources/v1/src-1/jobs",
            [
                {"exc": requests.ConnectionError},
                {"status_code": 503, "json": {"error": "Unknown error"}},
                {"status_code": 201, "json": {"job": {"id": "job-1"}}},
            ],
        )

        job = api.start_job("src-1", {"workflowUrl": "w"}, idempotency_key="run-1")

    assert job == {"id": "job-1"}
    assert [r.headers["Idempotency-Key"] for r in start.request_history] == [
        "run-1"
    ] * 3


def test_does_not_resend_a_request_without_an_idempotency_key() -> None:
    api = CloudStreamingAPI("key", workspace="acme", api_url=API_URL)
    with requests_mock.Mocker() as http:
        cancel = http.post(
            f"{API_URL}/acme/video-jobs/v1/job-1/cancel", status_code=503, json={}
        )

        with pytest.raises(CloudStreamingError) as error:
            api.cancel_job("job-1")

    assert error.value.status == 503
    assert cancel.call_count == 1

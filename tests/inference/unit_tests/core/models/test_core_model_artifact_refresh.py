from pathlib import Path

import pytest
from requests.exceptions import ChunkedEncodingError

from inference.core import roboflow_api
from inference.core.cache import MemoryCache, model_artifacts
from inference.core.exceptions import (
    RoboflowAPIForbiddenError,
    RoboflowAPINotNotFoundError,
)
from inference.core.models import roboflow
from inference.core.utils.url_utils import wrap_url

OLD_URL = "https://weights.example/model.bin?signature=old"
NEW_URL = "https://weights.example/model.bin?signature=new"
CONFIG_URL = "https://weights.example/config.json?signature=old"
NEW_CONFIG_URL = "https://weights.example/config.json?signature=new"
METADATA_URL = "https://api.roboflow.com/serverless/core_model/sam3/test"


@pytest.fixture
def model(monkeypatch, tmp_path):
    monkeypatch.setattr(roboflow, "MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(model_artifacts, "MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(roboflow_api, "cache", MemoryCache())
    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", False)
    monkeypatch.setattr(roboflow_api, "MD5_VERIFICATION_ENABLED", False)
    monkeypatch.setattr(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", False)
    monkeypatch.setattr(roboflow_api, "API_BASE_URL", "https://api.roboflow.com/")
    monkeypatch.setattr(roboflow_api, "INTERNAL_WEIGHTS_URL_SUFFIX", "serverless")
    monkeypatch.setattr(roboflow_api, "service_secret_is_valid", lambda secret: True)

    instance = object.__new__(roboflow.RoboflowCoreModel)
    instance.api_key = "test-key"
    instance.endpoint = "sam3/test"
    instance.device_id = "test-device"
    instance.countinference = False
    instance.service_secret = "test-service-secret"
    instance.cache_dir = str(tmp_path / "model")

    return instance


def _metadata_response(url, config_url=CONFIG_URL):
    response = {"weights": {"config": config_url, "model": url}}

    return response


def _artifact_path(model, filename="model.bin"):
    path = Path(model_artifacts.get_cache_file_path(filename, model.endpoint))

    return path


def _assert_metadata_auth(requests):
    for request in requests:
        assert request.qs["api_key"] == ["test-key"]
        assert request.qs["device"] == ["test-device"]
        assert request.qs["countinference"] == ["false"]
        assert request.qs["service_secret"] == ["test-service-secret"]


@pytest.mark.parametrize(
    "failure", ["forbidden", "interrupted_then_forbidden", "interrupted"]
)
def test_download_refreshes_failed_artifact_url(model, requests_mock, failure):
    metadata = requests_mock.get(
        wrap_url(METADATA_URL),
        response_list=[
            {"json": _metadata_response(OLD_URL)},
            {"json": _metadata_response(NEW_URL, NEW_CONFIG_URL)},
        ],
    )
    config = requests_mock.get(wrap_url(CONFIG_URL), content=b"{}", complete_qs=True)
    failure_responses = {
        "forbidden": [{"status_code": 403}],
        "interrupted_then_forbidden": [
            {"exc": ChunkedEncodingError("incomplete body")},
            {"status_code": 403},
        ],
        "interrupted": [{"exc": ChunkedEncodingError("incomplete body")}],
    }
    old = requests_mock.get(
        wrap_url(OLD_URL), response_list=failure_responses[failure], complete_qs=True
    )
    fresh = requests_mock.get(
        wrap_url(NEW_URL), content=b"complete weights", complete_qs=True
    )

    model.download_model_from_roboflow_api()

    assert _artifact_path(model).read_bytes() == b"complete weights"
    assert _artifact_path(model, "config.json").read_bytes() == b"{}"
    assert config.call_count == 1
    assert metadata.call_count == 2
    assert fresh.call_count == 1
    expected_attempts = {
        "forbidden": 1,
        "interrupted_then_forbidden": 2,
        "interrupted": roboflow_api.TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES,
    }
    assert old.call_count == expected_attempts[failure]
    _assert_metadata_auth(metadata.request_history)


@pytest.mark.parametrize("failure", ["forbidden", "interrupted"])
def test_download_stops_after_one_url_refresh(model, requests_mock, failure):
    metadata = requests_mock.get(
        wrap_url(METADATA_URL),
        response_list=[
            {"json": _metadata_response(OLD_URL)},
            {"json": _metadata_response(NEW_URL)},
        ],
    )
    requests_mock.get(wrap_url(CONFIG_URL), content=b"{}", complete_qs=True)
    response = (
        {"status_code": 403}
        if failure == "forbidden"
        else {"exc": ChunkedEncodingError("incomplete body")}
    )
    old = requests_mock.get(wrap_url(OLD_URL), complete_qs=True, **response)
    fresh = requests_mock.get(wrap_url(NEW_URL), complete_qs=True, **response)
    error = (
        RoboflowAPIForbiddenError if failure == "forbidden" else ChunkedEncodingError
    )

    with pytest.raises(error):
        model.download_model_from_roboflow_api()

    attempts = (
        1
        if failure == "forbidden"
        else roboflow_api.TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES
    )
    assert metadata.call_count == 2
    assert old.call_count == fresh.call_count == attempts
    assert not _artifact_path(model).exists()


@pytest.mark.parametrize("refresh", [False, True])
def test_metadata_denial_is_not_retried(model, requests_mock, refresh):
    responses = [{"json": _metadata_response(OLD_URL)}] if refresh else []
    responses.append({"status_code": 403})
    metadata = requests_mock.get(wrap_url(METADATA_URL), response_list=responses)
    requests_mock.get(wrap_url(CONFIG_URL), content=b"{}", complete_qs=True)
    old = requests_mock.get(wrap_url(OLD_URL), status_code=403, complete_qs=True)

    with pytest.raises(RoboflowAPIForbiddenError):
        model.download_model_from_roboflow_api()

    assert metadata.call_count == (2 if refresh else 1)
    assert old.call_count == (1 if refresh else 0)
    assert not _artifact_path(model).exists()


def test_missing_artifact_is_not_retried(model, requests_mock):
    metadata = requests_mock.get(
        wrap_url(METADATA_URL), json=_metadata_response(OLD_URL)
    )
    requests_mock.get(wrap_url(CONFIG_URL), content=b"{}", complete_qs=True)
    old = requests_mock.get(wrap_url(OLD_URL), status_code=404, complete_qs=True)

    with pytest.raises(RoboflowAPINotNotFoundError):
        model.download_model_from_roboflow_api()

    assert metadata.call_count == old.call_count == 1
    assert not _artifact_path(model).exists()


def test_slow_download_refreshes_remaining_urls_and_preserves_auth(
    model, requests_mock, monkeypatch
):
    times = iter([0, 121, 122, 123])
    monkeypatch.setattr(roboflow, "perf_counter", lambda: next(times))
    metadata = requests_mock.get(
        wrap_url(METADATA_URL),
        response_list=[
            {"json": _metadata_response(OLD_URL)},
            {"json": _metadata_response(NEW_URL, NEW_CONFIG_URL)},
        ],
    )
    requests_mock.get(wrap_url(CONFIG_URL), content=b"{}", complete_qs=True)
    fresh = requests_mock.get(wrap_url(NEW_URL), content=b"weights", complete_qs=True)

    model.download_model_from_roboflow_api()

    assert metadata.call_count == 2
    assert fresh.call_count == 1
    assert _artifact_path(model).read_bytes() == b"weights"
    _assert_metadata_auth(metadata.request_history)

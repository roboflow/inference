import json

import pytest
import requests
import requests_mock as rm

import inference_server.workflows.host as host
from inference_server import configuration
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.workflows import definition_cache

_URL = "https://api.roboflow.com/ws/workflows/wf?api_key=k"
_PAYLOAD = {
    "workflow": {
        "id": "wf-internal",
        "config": json.dumps({"specification": {"version": "1.0"}}),
    }
}
_SPECIFICATION = {"version": "1.0", "id": "wf-internal"}


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(configuration, "SINGLE_TENANT_WORKFLOW_CACHE", False)
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", False)
    monkeypatch.setattr(configuration, "USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS", True)
    return tmp_path / "cache" / "workflow"


def _write_path(api_key="k"):
    return definition_cache.cache_file_path(
        "ws", "wf", api_key=api_key, workflow_version_id=None
    )


def _fetch(use_cache=False):
    return host.get_workflow_specification("k", "ws", "wf", use_cache=use_cache)


def _raising(error):
    def _get(*args, **kwargs):
        raise error

    return _get


def test_successful_fetch_writes_the_definition_at_the_legacy_path(cache_root):
    with rm.Mocker() as m:
        m.get(_URL, json=_PAYLOAD)
        assert _fetch() == _SPECIFICATION

    assert json.loads(_write_path().read_text()) == _PAYLOAD


def test_unreachable_platform_serves_the_cached_definition(cache_root, monkeypatch):
    with rm.Mocker() as m:
        m.get(_URL, json=_PAYLOAD)
        live = _fetch()
    monkeypatch.setattr(
        requests, "get", _raising(requests.exceptions.ConnectionError("down"))
    )

    assert _fetch() == live


def test_timeout_serves_the_cached_definition(cache_root, monkeypatch):
    with rm.Mocker() as m:
        m.get(_URL, json=_PAYLOAD)
        live = _fetch()
    monkeypatch.setattr(requests, "get", _raising(requests.exceptions.Timeout("slow")))

    assert _fetch() == live


@pytest.mark.parametrize(
    "error,status_code,message",
    [
        (
            requests.exceptions.ConnectionError("down"),
            503,
            "Internal error. Could not connect to Roboflow API.",
        ),
        (
            requests.exceptions.Timeout("slow"),
            504,
            "Timeout when attempting to connect to Roboflow API.",
        ),
    ],
)
def test_cache_miss_re_raises_the_transport_error(
    cache_root, monkeypatch, error, status_code, message
):
    monkeypatch.setattr(requests, "get", _raising(error))

    with pytest.raises(LegacyHTTPError) as exc:
        _fetch()

    assert (exc.value.status_code, exc.value.message) == (status_code, message)


def test_platform_refusal_is_not_served_from_the_cache(cache_root):
    _write_path().parent.mkdir(parents=True)
    _write_path().write_text(json.dumps(_PAYLOAD))

    with rm.Mocker() as m:
        m.get(_URL, status_code=404, json={})
        with pytest.raises(LegacyHTTPError) as exc:
            _fetch()

    assert exc.value.status_code == 404


def test_setting_off_neither_writes_nor_reads(cache_root, monkeypatch):
    monkeypatch.setattr(
        configuration, "USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS", False
    )
    with rm.Mocker() as m:
        m.get(_URL, json=_PAYLOAD)
        _fetch()
    assert not cache_root.exists()

    _write_path().parent.mkdir(parents=True)
    _write_path().write_text(json.dumps(_PAYLOAD))
    monkeypatch.setattr(
        requests, "get", _raising(requests.exceptions.ConnectionError("down"))
    )
    with pytest.raises(LegacyHTTPError) as exc:
        _fetch()
    assert exc.value.status_code == 503


def test_offline_mode_serves_only_the_cache_without_calling_the_platform(
    cache_root, monkeypatch
):
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", True)
    monkeypatch.setattr(configuration, "SINGLE_TENANT_WORKFLOW_CACHE", True)
    monkeypatch.setattr(requests, "get", _raising(AssertionError("network call")))

    with pytest.raises(LegacyHTTPError) as exc:
        _fetch()
    assert (exc.value.status_code, exc.value.message) == (
        503,
        "Internal error. Could not connect to Roboflow API.",
    )

    canonical = cache_root / "ws" / ".canonical-v2" / "wf.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(json.dumps(_PAYLOAD))
    assert _fetch() == _SPECIFICATION


def test_write_failure_does_not_fail_the_request(cache_root, caplog):
    cache_root.parent.mkdir(parents=True)
    cache_root.write_text("not a directory")

    with rm.Mocker() as m:
        m.get(_URL, json=_PAYLOAD)
        assert _fetch() == _SPECIFICATION

    assert "Could not write the Workflow definition cache file" in caplog.text


class _BrokenCache:
    def __init__(self, fail_get=False, fail_set=False):
        self.fail_get = fail_get
        self.fail_set = fail_set

    def get(self, key):
        if self.fail_get:
            raise ConnectionError("redis down")
        return None

    def set(self, key, value, expire=None):
        if self.fail_set:
            raise ConnectionError("redis down")


def test_cache_read_failure_falls_through_to_the_platform(
    cache_root, monkeypatch, caplog
):
    monkeypatch.setattr(host, "WORKFLOWS_CACHE", _BrokenCache(fail_get=True))

    with rm.Mocker() as m:
        m.get(_URL, json=_PAYLOAD)
        assert _fetch(use_cache=True) == _SPECIFICATION

    warnings = [
        r.getMessage() for r in caplog.records if "ConnectionError" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "redis down" not in warnings[0]
    assert "api_key" not in warnings[0]
    assert "workflow_definition" not in warnings[0]


def test_cache_write_failure_still_returns_the_fetched_definition(
    cache_root, monkeypatch, caplog
):
    monkeypatch.setattr(host, "WORKFLOWS_CACHE", _BrokenCache(fail_set=True))

    with rm.Mocker() as m:
        m.get(_URL, json=_PAYLOAD)
        assert _fetch(use_cache=True) == _SPECIFICATION

    assert "ConnectionError" in caplog.text


def test_file_fallback_serves_when_the_cache_read_fails_and_the_platform_is_down(
    cache_root, monkeypatch
):
    with rm.Mocker() as m:
        m.get(_URL, json=_PAYLOAD)
        live = _fetch()
    monkeypatch.setattr(host, "WORKFLOWS_CACHE", _BrokenCache(fail_get=True))
    monkeypatch.setattr(
        requests, "get", _raising(requests.exceptions.ConnectionError("down"))
    )

    assert _fetch(use_cache=True) == live

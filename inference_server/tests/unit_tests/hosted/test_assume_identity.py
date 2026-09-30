import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from inference_server import configuration
from inference_server.framework.entities import CommonRequestParams
from inference_server.framework.model_stat import (
    _reset_cache_for_tests,
    stat_model_while_checking_auth,
)
from inference_server.hosted.assume_identity import (
    add_assume_identity_headers,
    assume_identity_authorised_workspace_db_id,
    enforce_credits_verification,
)
from inference_server.workflows import host


@pytest.fixture(autouse=True)
def _reset_stat_cache():
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setattr(
        configuration, "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", "tok"
    )


def _with_workspace_db_id(value):
    reset = assume_identity_authorised_workspace_db_id.set(value)
    return lambda: assume_identity_authorised_workspace_db_id.reset(reset)


def test_no_headers_without_token():
    undo = _with_workspace_db_id("db-1")
    try:
        headers = {"a": "b"}
        add_assume_identity_headers(headers)
        assert headers == {"a": "b"}
    finally:
        undo()


def test_no_headers_without_authorised_workspace(token):
    headers = {}
    add_assume_identity_headers(headers)
    assert headers == {}


def test_no_headers_for_invalid_workspace_db_id(token):
    undo = _with_workspace_db_id("has space")
    try:
        headers = {}
        add_assume_identity_headers(headers)
        assert headers == {}
    finally:
        undo()


def test_headers_added_when_token_and_workspace_present(token):
    undo = _with_workspace_db_id("db-1")
    try:
        headers = {}
        add_assume_identity_headers(headers)
        assert headers == {
            "x-assume-identity-access-token": "tok",
            "x-assume-identity-authorised-workspace": "db-1",
        }
    finally:
        undo()


def test_platform_request_carries_assume_identity_headers(token, monkeypatch):
    seen = {}

    def _get(url, **kwargs):
        seen.update(kwargs)
        return MagicMock(status_code=200)

    monkeypatch.setattr(requests, "get", _get)
    undo = _with_workspace_db_id("db-1")
    try:
        host._platform_request(
            "get", "https://x", headers={"h": "v"}, timeout=1, assume_identity=True
        )
    finally:
        undo()

    assert seen["headers"] == {
        "h": "v",
        "x-assume-identity-access-token": "tok",
        "x-assume-identity-authorised-workspace": "db-1",
    }
    assert seen["timeout"] == 1


def test_platform_request_omits_assume_identity_by_default(token, monkeypatch):
    seen = {}

    def _get(url, **kwargs):
        seen.update(kwargs)
        return MagicMock(status_code=200)

    monkeypatch.setattr(requests, "get", _get)
    undo = _with_workspace_db_id("db-1")
    try:
        host._platform_request("get", "https://x", headers={"h": "v"})
    finally:
        undo()

    assert seen["headers"] == {"h": "v"}


def test_workflow_definition_fetch_omits_assume_identity(token, monkeypatch):
    seen = {}

    def _get(url, **kwargs):
        seen.update(kwargs)
        return MagicMock(status_code=200, json=lambda: {"workflow": {}})

    monkeypatch.setattr(requests, "get", _get)
    undo = _with_workspace_db_id("db-1")
    try:
        host._fetch_workflow_response(
            api_key="k", workspace_id="ws", workflow_id="wf", workflow_version_id=None
        )
    finally:
        undo()

    assert "x-assume-identity-access-token" not in seen["headers"]
    assert "x-assume-identity-authorised-workspace" not in seen["headers"]


def test_platform_request_unchanged_without_token(monkeypatch):
    seen = {}

    def _post(url, **kwargs):
        seen.update(kwargs)
        return MagicMock(status_code=200)

    monkeypatch.setattr(requests, "post", _post)
    host._platform_request("post", "https://x", headers={"h": "v"}, json={"a": 1})

    assert seen == {"headers": {"h": "v"}, "json": {"a": 1}}


@pytest.mark.asyncio
async def test_model_stat_has_no_extra_headers_by_default():
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        return_value=MagicMock(task_type="object-detection"),
    ) as fetch:
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="acme/1", api_key="k")
        )

    fetch.assert_called_once_with(model_id="acme/1", api_key="k")


@pytest.mark.asyncio
async def test_model_stat_sends_credits_header_when_enforced(monkeypatch):
    monkeypatch.setattr(configuration, "ENFORCE_CREDITS_VERIFICATION", True)
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        return_value=MagicMock(task_type="object-detection"),
    ) as fetch:
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="acme/1", api_key="k")
        )

    fetch.assert_called_once_with(
        model_id="acme/1",
        api_key="k",
        extra_headers={"x-enforce-credits-verification": "true"},
    )


@pytest.mark.asyncio
async def test_model_stat_skips_credits_header_for_non_billable_request(
    monkeypatch,
):
    monkeypatch.setattr(configuration, "ENFORCE_CREDITS_VERIFICATION", True)
    reset = enforce_credits_verification.set(False)
    try:
        with patch(
            "inference_server.framework.model_stat.get_one_page_of_model_metadata",
            return_value=MagicMock(task_type="object-detection"),
        ) as fetch:
            await stat_model_while_checking_auth(
                CommonRequestParams(model_id="acme/1", api_key="k")
            )
    finally:
        enforce_credits_verification.reset(reset)

    fetch.assert_called_once_with(model_id="acme/1", api_key="k")


@pytest.mark.asyncio
async def test_model_stat_sends_assume_identity_headers(token):
    undo = _with_workspace_db_id("db-1")
    try:
        with patch(
            "inference_server.framework.model_stat.get_one_page_of_model_metadata",
            return_value=MagicMock(task_type="object-detection"),
        ) as fetch:
            await stat_model_while_checking_auth(
                CommonRequestParams(model_id="acme/1", api_key="k")
            )
    finally:
        undo()

    fetch.assert_called_once_with(
        model_id="acme/1",
        api_key="k",
        extra_headers={
            "x-assume-identity-access-token": "tok",
            "x-assume-identity-authorised-workspace": "db-1",
        },
    )


def _recording_fetch(calls, delay=0.0):
    def _fetch(model_id, api_key=None, extra_headers=None):
        calls.append(extra_headers)
        if delay:
            time.sleep(delay)
        return MagicMock(task_type="object-detection")

    return _fetch


async def _stat_with_enforcement(flag: bool):
    reset = enforce_credits_verification.set(flag)
    try:
        return await stat_model_while_checking_auth(
            CommonRequestParams(model_id="acme/1", api_key="k")
        )
    finally:
        enforce_credits_verification.reset(reset)


@pytest.mark.asyncio
async def test_model_stat_cache_is_keyed_by_enforcement_context(monkeypatch):
    monkeypatch.setattr(configuration, "ENFORCE_CREDITS_VERIFICATION", True)
    calls = []
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=_recording_fetch(calls),
    ):
        await _stat_with_enforcement(False)
        await _stat_with_enforcement(True)
        await _stat_with_enforcement(False)
        await _stat_with_enforcement(True)

    assert calls == [None, {"x-enforce-credits-verification": "true"}]


@pytest.mark.asyncio
async def test_model_stat_cache_is_keyed_by_assume_identity_context(token):
    calls = []
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=_recording_fetch(calls),
    ):
        undo = _with_workspace_db_id("db-1")
        try:
            await stat_model_while_checking_auth(
                CommonRequestParams(model_id="acme/1", api_key="k")
            )
        finally:
            undo()
        undo = _with_workspace_db_id("db-2")
        try:
            await stat_model_while_checking_auth(
                CommonRequestParams(model_id="acme/1", api_key="k")
            )
        finally:
            undo()

    assert [headers["x-assume-identity-authorised-workspace"] for headers in calls] == [
        "db-1",
        "db-2",
    ]


@pytest.mark.asyncio
async def test_model_stat_inflight_is_not_shared_across_contexts(monkeypatch):
    monkeypatch.setattr(configuration, "ENFORCE_CREDITS_VERIFICATION", True)
    calls = []
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=_recording_fetch(calls, delay=0.05),
    ):
        await asyncio.gather(
            _stat_with_enforcement(True), _stat_with_enforcement(False)
        )

    assert sorted(calls, key=str) == [
        None,
        {"x-enforce-credits-verification": "true"},
    ]


@pytest.mark.asyncio
async def test_model_stat_inflight_is_shared_within_one_context(monkeypatch):
    monkeypatch.setattr(configuration, "ENFORCE_CREDITS_VERIFICATION", True)
    calls = []
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=_recording_fetch(calls, delay=0.05),
    ):
        await asyncio.gather(_stat_with_enforcement(True), _stat_with_enforcement(True))

    assert calls == [{"x-enforce-credits-verification": "true"}]

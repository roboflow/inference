import socket
import time
from unittest import mock
from unittest.mock import MagicMock

import pytest
import requests
from roboflow_workflows.core_steps.common.query_language.entities.operations import (
    StringToUpperCase,
)
from roboflow_workflows.core_steps.sinks.webhook import v1
from roboflow_workflows.core_steps.sinks.webhook.v1 import (
    BlockManifest,
    WebhookSinkBlockV1,
    execute_operations_on_parameters,
    execute_request,
)
from roboflow_workflows.utils import url_input
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool


def _fake_getaddrinfo(*ips):
    def _inner(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
            for ip in ips
        ]

    return _inner


class _FakeResponse:
    """Minimal stand-in for `requests.Response` recording close() calls."""

    def __init__(self, status_code: int = 200, headers=None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"status={self.status_code}")


def _install_capturing_adapter(monkeypatch, response: _FakeResponse):
    """Swap `SSRFProtectedHTTPAdapter` inside v1 with a factory that captures
    the (request, kwargs) it receives and returns ``response``. Records both
    adapter and response close() calls on the returned holder.
    """
    holder = {
        "request": None,
        "kwargs": None,
        "adapter_closed": False,
        "allow_non_global_addresses": None,
    }

    class _FakeAdapter:
        def __init__(self, *, allow_non_global_addresses: bool) -> None:
            holder["allow_non_global_addresses"] = allow_non_global_addresses

        def send(self, request, **kwargs):
            holder["request"] = request
            holder["kwargs"] = kwargs
            return response

        def close(self) -> None:
            holder["adapter_closed"] = True

    monkeypatch.setattr(v1, "SSRFProtectedHTTPAdapter", _FakeAdapter)
    # Adapter path only exists in hardened mode.
    monkeypatch.setattr(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", False
    )
    return holder


def test_manifest_parsing_when_the_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/webhook_sink@v1",
        "name": "multipart_image_sink_put",
        "url": "http://127.0.0.1:9999/data-sink/multi-part-data",
        "method": "PUT",
        "multi_part_encoded_files": {
            "image": "$inputs.image",
        },
        "multi_part_encoded_files_operations": {
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        "form_data": {
            "form_field": "$inputs.query_parameter",
        },
        "fire_and_forget": True,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/webhook_sink@v1",
        name="multipart_image_sink_put",
        url="http://127.0.0.1:9999/data-sink/multi-part-data",
        method="PUT",
        multi_part_encoded_files={
            "image": "$inputs.image",
        },
        multi_part_encoded_files_operations={
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        form_data={
            "form_field": "$inputs.query_parameter",
        },
        fire_and_forget=True,
    )


def test_execute_request_forwards_payload_to_adapter(monkeypatch) -> None:
    response = _FakeResponse(status_code=200)
    holder = _install_capturing_adapter(monkeypatch, response)

    result = execute_request(
        url="https://public.example/webhook",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        form_data={"field": "value"},
        multi_part_encoded_files={"file": b"data"},
        timeout=3,
    )

    assert result == (False, "Notification sent successfully")
    sent = holder["request"]
    assert sent.method == "POST"
    assert sent.url.startswith("https://public.example/webhook?")
    assert "a=b" in sent.url
    assert sent.headers.get("c") == "d"
    assert holder["kwargs"]["timeout"] == 3
    # Proxies must be pinned to an empty dict so env HTTP(S)_PROXY cannot
    # steer resolution outside the validating adapter.
    assert holder["kwargs"]["proxies"] == {}
    assert holder["kwargs"]["stream"] is True
    assert holder["allow_non_global_addresses"] is False
    assert response.closed is True
    assert holder["adapter_closed"] is True


def test_execute_request_reports_http_failure_and_closes(monkeypatch) -> None:
    response = _FakeResponse(status_code=500)
    holder = _install_capturing_adapter(monkeypatch, response)

    ok, message = execute_request(
        url="https://public.example/webhook",
        method="POST",
        query_parameters={},
        headers={},
        json_payload={},
        form_data={},
        multi_part_encoded_files={},
        timeout=3,
    )

    assert ok is True
    assert "500" in message
    assert response.closed is True
    assert holder["adapter_closed"] is True


def test_execute_request_rejects_unknown_method() -> None:
    ok, message = execute_request(
        url="https://public.example/webhook",
        method="DELETE",  # type: ignore[arg-type]
        query_parameters={},
        headers={},
        json_payload={},
        form_data={},
        multi_part_encoded_files={},
        timeout=1,
    )
    assert ok is True
    assert "DELETE" in message


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://10.0.0.1/x",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/x",
        "http://[::ffff:127.0.0.1]/x",
    ],
)
def test_execute_request_rejects_non_global_ip_literals(url, monkeypatch) -> None:
    monkeypatch.setattr(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", False
    )
    ok, message = execute_request(
        url=url,
        method="POST",
        query_parameters={},
        headers={},
        json_payload={},
        form_data={},
        multi_part_encoded_files={},
        timeout=1,
    )
    assert ok is True
    assert (
        "non-global" in message.lower()
        or "not allowed" in message.lower()
        or "'" in message
    )


def test_execute_request_rejects_public_hostname_that_resolves_privately(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", False
    )
    monkeypatch.setattr(
        url_input.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")
    )

    # If pool.urlopen ever runs, a socket has been opened for a metadata IP.
    def _explode(*a, **k):
        raise AssertionError("must not open socket")

    monkeypatch.setattr(HTTPConnectionPool, "urlopen", _explode)
    monkeypatch.setattr(HTTPSConnectionPool, "urlopen", _explode)

    ok, message = execute_request(
        url="https://metadata.example/latest/meta-data",
        method="GET",
        query_parameters={},
        headers={},
        json_payload={},
        form_data={},
        multi_part_encoded_files={},
        timeout=1,
    )
    assert ok is True
    assert "non-global" in message.lower()


def test_execute_request_pins_first_dns_result_against_rebinding(monkeypatch) -> None:
    monkeypatch.setattr(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", False
    )
    calls = {"n": 0}
    responses = [
        [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("8.8.8.8", 443),
            )
        ],
        [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 443),
            )
        ],
    ]

    def _resolve(host, port, *args, **kwargs):
        calls["n"] += 1
        return responses.pop(0) if responses else responses[-1]

    monkeypatch.setattr(url_input.socket, "getaddrinfo", _resolve)

    captured = {}

    def _pool_urlopen(self, method, url, **kwargs):
        captured["pool_host"] = self.host
        response = requests.Response()
        response.status_code = 200
        response.raw = mock.MagicMock()
        response.raw.release_conn = lambda: None
        return _RawResponse()

    class _RawResponse:
        # urllib3 response surface that requests' HTTPAdapter.build_response uses.
        status = 200
        headers = {}
        reason = "OK"
        version = 11
        msg = None
        release_conn = staticmethod(lambda: None)

        def read(self, *a, **k):
            return b""

        def stream(self, *a, **k):
            return iter(())

        def close(self):
            pass

        def get_redirect_location(self):
            return False

    monkeypatch.setattr(HTTPSConnectionPool, "urlopen", _pool_urlopen)

    ok, message = execute_request(
        url="https://example.com/webhook",
        method="GET",
        query_parameters={},
        headers={},
        json_payload={},
        form_data={},
        multi_part_encoded_files={},
        timeout=1,
    )
    assert ok is False, message
    assert calls["n"] == 1  # only the first DNS answer is used
    assert captured["pool_host"] == "8.8.8.8"


def test_execute_request_treats_redirect_as_failure(monkeypatch) -> None:
    response = _FakeResponse(
        status_code=302, headers={"Location": "http://127.0.0.1/x"}
    )
    holder = _install_capturing_adapter(monkeypatch, response)

    ok, message = execute_request(
        url="https://public.example/webhook",
        method="GET",
        query_parameters={},
        headers={},
        json_payload={},
        form_data={},
        multi_part_encoded_files={},
        timeout=1,
    )

    assert ok is True
    assert "redirect" in message.lower()
    assert response.closed is True
    assert holder["adapter_closed"] is True
    # A second .send() would mean the Location was followed.
    assert holder["request"].url.startswith("https://public.example/webhook")


def test_execute_request_ignores_environment_proxies(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("NO_PROXY", "")
    response = _FakeResponse(status_code=200)
    holder = _install_capturing_adapter(monkeypatch, response)

    ok, _ = execute_request(
        url="https://public.example/webhook",
        method="POST",
        query_parameters={},
        headers={},
        json_payload={"a": 1},
        form_data={},
        multi_part_encoded_files={},
        timeout=1,
    )
    assert ok is False
    assert holder["kwargs"]["proxies"] == {}


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http:///path",
        "http://example.com:not-a-number/",
        "http://example.com\\@evil.com/",
    ],
)
def test_execute_request_rejects_malformed_or_non_http_urls(url, monkeypatch) -> None:
    monkeypatch.setattr(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", False
    )
    # None of these may reach a socket; if getaddrinfo or urlopen runs, fail loudly.
    monkeypatch.setattr(
        url_input.socket,
        "getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )
    ok, message = execute_request(
        url=url,
        method="GET",
        query_parameters={},
        headers={},
        json_payload={},
        form_data={},
        multi_part_encoded_files={},
        timeout=1,
    )
    assert ok is True
    assert message  # failure surfaces via error status + message


def test_execute_operations_on_parameters() -> None:
    # given
    parameters = {"a": "some", "b": "other"}
    operations = {"a": [StringToUpperCase(type="StringToUpperCase")]}

    # when
    result = execute_operations_on_parameters(
        parameters=parameters,
        operations=operations,
    )

    # then
    assert result == {
        "a": "SOME",
        "b": "other",
    }


def test_cooldown_in_webhook_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            url="http://some.com",
            method="POST",
            query_parameters={"a": "b"},
            headers={"c": "d"},
            json_payload={"e": "f"},
            json_payload_operations={},
            form_data={"field": "value"},
            form_data_operations={},
            multi_part_encoded_files={"file": b"data"},
            multi_part_encoded_files_operations={},
            request_timeout=3,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=100,
        )
        results.append(result)

    # then
    assert results[0] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    assert results[1] == {
        "error_status": False,
        "throttling_status": True,
        "message": "Sink cooldown applies",
    }


def test_disabling_cooldown_in_webhook_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            url="http://some.com",
            method="POST",
            query_parameters={"a": "b"},
            headers={"c": "d"},
            json_payload={"e": "f"},
            json_payload_operations={},
            form_data={"field": "value"},
            form_data_operations={},
            multi_part_encoded_files={"file": b"data"},
            multi_part_encoded_files_operations={},
            request_timeout=3,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=0,
        )
        results.append(result)

    # then
    assert results[0] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    assert results[1] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }


def test_cooldown_recovery_in_webhook_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            url="http://some.com",
            method="POST",
            query_parameters={"a": "b"},
            headers={"c": "d"},
            json_payload={"e": "f"},
            json_payload_operations={},
            form_data={"field": "value"},
            form_data_operations={},
            multi_part_encoded_files={"file": b"data"},
            multi_part_encoded_files_operations={},
            request_timeout=3,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=1,
        )
        results.append(result)
        time.sleep(1.5)

    # then
    assert results[0] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    assert results[1] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }


@mock.patch.object(v1, "execute_request")
def test_sending_webhook_notification_synchronously(
    execute_request_mock: MagicMock,
) -> None:
    # given
    execute_request_mock.return_value = (False, "ok")
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        json_payload_operations={"e": [StringToUpperCase(type="StringToUpperCase")]},
        form_data={"field": "value"},
        form_data_operations={},
        multi_part_encoded_files={"file": b"data"},
        multi_part_encoded_files_operations={},
        request_timeout=3,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "ok",
    }
    execute_request_mock.assert_called_once_with(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "F"},
        multi_part_encoded_files={"file": b"data"},
        form_data={"field": "value"},
        timeout=3,
    )


def test_disabling_webhook_notification() -> None:
    # given
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        json_payload_operations={"e": [StringToUpperCase(type="StringToUpperCase")]},
        form_data={"field": "value"},
        form_data_operations={},
        multi_part_encoded_files={"file": b"data"},
        multi_part_encoded_files_operations={},
        request_timeout=3,
        fire_and_forget=False,
        disable_sink=True,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Sink was disabled by parameter `disable_sink`",
    }


def test_sending_webhook_notification_asynchronously_in_background_tasks() -> None:
    # given
    background_tasks = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        json_payload_operations={"e": [StringToUpperCase(type="StringToUpperCase")]},
        form_data={"field": "value"},
        form_data_operations={},
        multi_part_encoded_files={"file": b"data"},
        multi_part_encoded_files_operations={},
        request_timeout=3,
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    background_tasks.add_task.assert_called_once()


def test_sending_webhook_notification_asynchronously_in_thread_pool_executor() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    result = block.run(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        json_payload_operations={"e": [StringToUpperCase(type="StringToUpperCase")]},
        form_data={"field": "value"},
        form_data_operations={},
        multi_part_encoded_files={"file": b"data"},
        multi_part_encoded_files_operations={},
        request_timeout=3,
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    thread_pool_executor.submit.assert_called_once()


def test_execute_request_uses_plain_requests_when_non_global_allowed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", True
    )
    monkeypatch.setattr(
        v1,
        "SSRFProtectedHTTPAdapter",
        lambda **_: (_ for _ in ()).throw(AssertionError("adapter must not run")),
    )
    calls = {}

    def _fake_post(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return _FakeResponse(status_code=200)

    monkeypatch.setitem(v1.METHOD_TO_HANDLER, "POST", _fake_post)

    result = execute_request(
        url="http://127.0.0.1/webhook",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        form_data={"field": "value"},
        multi_part_encoded_files={"file": b"data"},
        timeout=3,
    )

    assert result == (False, "Notification sent successfully")
    assert calls["url"] == "http://127.0.0.1/webhook"
    assert calls["kwargs"] == {
        "params": {"a": "b"},
        "headers": {"c": "d"},
        "json": {"e": "f"},
        "files": {"file": b"data"},
        "data": {"field": "value"},
        "timeout": 3,
    }


def test_execute_request_forwards_deny_non_global_when_flag_disabled(
    monkeypatch,
) -> None:
    response = _FakeResponse(status_code=200)
    holder = _install_capturing_adapter(monkeypatch, response)
    monkeypatch.setattr(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", False
    )

    execute_request(
        url="https://public.example/webhook",
        method="POST",
        query_parameters={},
        headers={},
        json_payload={},
        form_data={},
        multi_part_encoded_files={},
        timeout=1,
    )

    assert holder["allow_non_global_addresses"] is False

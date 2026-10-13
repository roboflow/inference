"""Unit tests of ``v2/slack_notification@v1`` with real ``slack_sdk`` on loopback.

``slack_api_url`` points ``WebClient`` at an in-process fake of the Slack Web
API on ``127.0.0.1``; the token is an invented sentinel.
"""

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.query import QueryRuntimeError
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    BackgroundEffects,
)
from roboflow_workflows.v2.blocks.sinks.slack import SlackNotificationBlock
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

from tests.v2.support.sinks.loopback import ManualClock

TOKEN = "xoxb-test-SENTINEL-3b8e"
SENT = {
    "error_status": False,
    "throttling_status": False,
    "message": "Notification sent successfully",
}


class FakeSlackApi:
    """Slack Web API on ``127.0.0.1`` covering what the block calls.

    Records ``(method, fields, authorization)`` per API call and uploaded
    bytes by file ID. Use as a context manager.

    Args:
        error: Slack error code returned by every API method, when set.
    """

    def __init__(self, *, error: Optional[str] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.uploads: Dict[str, bytes] = {}
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                path = urllib.parse.urlsplit(self.path).path
                if path.startswith("/upload/"):
                    api.uploads[path.rsplit("/", 1)[1]] = body
                    self._reply("OK", content_type="text/plain")
                    return
                method = path.rsplit("/", 1)[1]
                api.calls.append(
                    {
                        "method": method,
                        "fields": api.fields(self.headers, body),
                        "authorization": self.headers.get("Authorization"),
                    }
                )
                self._reply(json.dumps(api.response(method)))

            def _reply(
                self, text: str, *, content_type: str = "application/json"
            ) -> None:
                data = text.encode()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                return None

        self._error = error
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )

    @property
    def url(self) -> str:
        """Base URL to pass as ``slack_api_url``."""
        host, port = self._httpd.server_address[:2]

        return f"http://{host}:{port}/api/"

    def methods(self) -> List[str]:
        """Return the called API method names in order."""
        return [call["method"] for call in self.calls]

    @staticmethod
    def fields(headers: Any, body: bytes) -> Dict[str, Any]:
        if "json" in (headers.get("Content-Type") or ""):
            return json.loads(body)
        return {
            key: values[0]
            for key, values in urllib.parse.parse_qs(body.decode()).items()
        }

    def response(self, method: str) -> Dict[str, Any]:
        if self._error is not None:
            return {"ok": False, "error": self._error}
        if method == "files.getUploadURLExternal":
            file_id = f"F{len(self.calls)}"
            host, port = self._httpd.server_address[:2]
            return {
                "ok": True,
                "file_id": file_id,
                "upload_url": f"http://{host}:{port}/upload/{file_id}",
            }
        if method == "files.completeUploadExternal":
            return {"ok": True, "files": [{"id": "F"}]}
        return {"ok": True, "ts": "1.0"}

    def __enter__(self) -> "FakeSlackApi":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _send(block: SlackNotificationBlock, **params: Any) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "slack_token": Secret(TOKEN, name="SLACK_TOKEN"),
        "channel": "C123",
        "message": "{{ $parameters.count }} people at the door",
        "message_parameters": {"count": 2},
        "message_parameters_operations": {},
        "attachments": {},
        "fire_and_forget": False,
        "disable_sink": False,
        "cooldown_seconds": 0,
    }
    arguments.update(params)
    result = block.run(**arguments)

    return result


def test_message_is_posted_with_the_token_to_the_configured_api() -> None:
    with FakeSlackApi() as api:
        block = SlackNotificationBlock(slack_api_url=api.url)

        result = _send(block, slack_token=TOKEN)

    assert result == SENT
    [call] = api.calls
    assert call["method"] == "chat.postMessage"
    assert call["authorization"] == f"Bearer {TOKEN}"
    assert call["fields"]["channel"] == "C123"
    assert call["fields"]["text"] == "2 people at the door"


def test_attachments_are_uploaded_with_their_names_and_the_message_as_comment() -> None:
    with FakeSlackApi() as api:
        block = SlackNotificationBlock(slack_api_url=api.url)

        result = _send(
            block, attachments={"report.csv": "a,b\n1,2", "raw.bin": b"\x00\x01"}
        )

    assert result == SENT
    assert api.methods() == [
        "files.getUploadURLExternal",
        "files.getUploadURLExternal",
        "files.completeUploadExternal",
    ]
    names = [call["fields"]["filename"] for call in api.calls[:2]]
    assert names == ["report.csv", "raw.bin"]
    assert sorted(api.uploads.values()) == [b"\x00\x01", b"a,b\n1,2"]
    completion = api.calls[2]["fields"]
    assert completion["channel_id"] == "C123"
    assert completion["initial_comment"] == "2 people at the door"
    files = completion["files"]
    if isinstance(files, str):
        files = json.loads(files)
    assert [item["title"] for item in files] == ["report.csv", "raw.bin"]


def test_operations_run_only_for_parameters_the_message_uses() -> None:
    with FakeSlackApi() as api:
        block = SlackNotificationBlock(slack_api_url=api.url)

        _send(
            block,
            message="Classes: {{ $parameters.labels }}",
            message_parameters={"labels": ["car", "bus"], "unused": 1},
            message_parameters_operations={
                "labels": [{"type": "SequenceLength"}],
                "unused": [{"type": "SequenceLength"}],
            },
        )

    assert api.calls[0]["fields"]["text"] == "Classes: 2"


def test_slack_error_code_is_reported_like_v1() -> None:
    with FakeSlackApi(error="channel_not_found") as api:
        block = SlackNotificationBlock(slack_api_url=api.url)

        result = _send(block)

    assert result == {
        "error_status": True,
        "throttling_status": False,
        "message": (
            "Failed to send Slack notification. Internal error details: "
            "channel_not_found"
        ),
    }


def test_unreachable_api_is_a_failed_outcome() -> None:
    with FakeSlackApi() as api:
        closed_url = api.url
    block = SlackNotificationBlock(slack_api_url=closed_url)

    result = _send(block)

    assert result["error_status"] is True
    assert result["message"].startswith(
        "Failed to send Slack notification. Internal error details: <urlopen error"
    )


def test_token_echoed_by_the_api_is_replaced_by_its_secret_name() -> None:
    with FakeSlackApi(error=f"token_revoked:{TOKEN}") as api:
        block = SlackNotificationBlock(slack_api_url=api.url)

        result = _send(block)

    assert result["message"].endswith("token_revoked:<secret SLACK_TOKEN>")
    assert TOKEN not in result["message"]


def test_replayed_token_fails_the_step_and_sends_nothing() -> None:
    with FakeSlackApi() as api:
        block = SlackNotificationBlock(slack_api_url=api.url)

        with pytest.raises(SecretUnavailableError, match="SLACK_TOKEN"):
            _send(block, slack_token=Secret.unavailable("SLACK_TOKEN"))

    assert api.calls == []


def test_failed_operation_raises_and_does_not_start_the_cooldown() -> None:
    clock = ManualClock()
    extract = {"labels": [{"type": "SequenceLength"}]}
    with FakeSlackApi() as api:
        block = SlackNotificationBlock(slack_api_url=api.url, cooldown_clock=clock)

        with pytest.raises(QueryRuntimeError, match="SequenceLength"):
            _send(
                block,
                message="{{ $parameters.labels }}",
                message_parameters={"labels": 3},
                message_parameters_operations=extract,
                cooldown_seconds=5,
            )
        clock.advance(1)
        second = _send(block, cooldown_seconds=5)

    assert second == SENT


def test_cooldown_is_one_window_per_step_across_tokens_and_channels() -> None:
    # given: V1 kept one window per token; V2 keeps one per step
    clock = ManualClock()
    with FakeSlackApi() as api:
        block = SlackNotificationBlock(slack_api_url=api.url, cooldown_clock=clock)

        first = _send(block, cooldown_seconds=5)
        clock.advance(2)
        other = _send(block, cooldown_seconds=5, slack_token="xoxb-other", channel="C9")
        clock.advance(3)
        third = _send(block, cooldown_seconds=5)

    assert first == third == SENT
    assert other == {
        "error_status": False,
        "throttling_status": True,
        "message": "Sink cooldown applies",
    }
    assert len(api.calls) == 2


@pytest.mark.parametrize(
    "by_policy, message",
    [
        (True, "Sink was disabled by workflow execution policy"),
        (False, "Sink was disabled by parameter `disable_sink`"),
    ],
)
def test_disabled_sink_sends_nothing(by_policy: bool, message: str) -> None:
    with FakeSlackApi() as api:
        block = SlackNotificationBlock(slack_api_url=api.url, disable_sinks=by_policy)

        result = _send(
            block,
            disable_sink=not by_policy,
            slack_token=Secret.unavailable("SLACK_TOKEN"),
        )

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": message,
    }
    assert api.calls == []


def test_fire_and_forget_posts_in_the_background() -> None:
    with FakeSlackApi() as api:
        with BackgroundEffects() as effects:
            block = SlackNotificationBlock(slack_api_url=api.url, sink_effects=effects)

            result = _send(block, fire_and_forget=True)
            assert effects.drain(timeout=10)

        stats = effects.stats()

    assert result["message"] == BACKGROUND_MESSAGE
    assert stats.completed == 1
    assert api.methods() == ["chat.postMessage"]


def test_background_failure_is_counted_on_the_resource() -> None:
    with FakeSlackApi(error="invalid_auth") as api:
        with BackgroundEffects() as effects:
            block = SlackNotificationBlock(slack_api_url=api.url, sink_effects=effects)

            _send(block, fire_and_forget=True)
            assert effects.drain(timeout=10)

    [failure] = effects.failures()
    assert failure.label == "v2/slack_notification@v1"
    assert failure.message.endswith("invalid_auth")


def test_params_reject_unknown_operation_keys_and_negative_cooldown() -> None:
    base = {
        "slack_token": "$inputs.token",
        "channel": "C1",
        "message": "m",
    }

    with pytest.raises(ValidationError, match="`labels`, which is not a key"):
        SlackNotificationBlock.Params.model_validate(
            {
                **base,
                "message_parameters_operations": {
                    "labels": [{"type": "SequenceLength"}]
                },
            }
        )
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SlackNotificationBlock.Params.model_validate({**base, "cooldown_seconds": -1})

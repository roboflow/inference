"""Unit tests of ``roboflow_core/twilio_sms_notification@v2``.

Custom Twilio runs the real ``twilio`` client over a recording ``HttpClient``;
the managed provider is a recording ``platform_client``. Credentials are
invented sentinels and nothing is sent anywhere.
"""

import base64
import json
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from roboflow_workflows.v2.blocks.query import QueryRuntimeError
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    BackgroundEffects,
)
from roboflow_workflows.v2.blocks.sinks.twilio_sms_notification.v2 import CREDITS_MESSAGE, NO_UPLOADER_MESSAGE, TwilioSmsNotificationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError
from twilio.http import HttpClient
from twilio.http.response import Response

from tests.v2.support.sinks.loopback import ManualClock

ACCOUNT_SID = "AC-test-SENTINEL-0d4f"
AUTH_TOKEN = "tw-test-SENTINEL-77aa"
API_KEY = "rf-test-SENTINEL-9e07"
SENT = {
    "error_status": False,
    "throttling_status": False,
    "message": "Notification sent successfully",
}
PROXY_SENT = {
    "error_status": False,
    "throttling_status": False,
    "message": "Notification sent successfully via Roboflow proxy",
}


class RecordingTwilioHttp(HttpClient):
    """``twilio`` HTTP layer that records requests and answers locally.

    Args:
        status: Response status; 201 creates the message.
    """

    def __init__(self, *, status: int = 201) -> None:
        super().__init__(logger=None, is_async=False)
        self.requests: List[Dict[str, Any]] = []
        self._status = status

    def request(
        self,
        method: str,
        url: str,
        params: Any = None,
        data: Any = None,
        headers: Any = None,
        auth: Optional[Tuple[str, str]] = None,
        timeout: Any = None,
        allow_redirects: bool = False,
    ) -> Response:
        self.requests.append({"method": method, "url": url, "data": data, "auth": auth})
        if self._status < 400:
            body = {"sid": "SM1", "status": "queued"}
        else:
            body = {"code": 20003, "message": f"Authenticate {auth[1]}", "status": 401}

        return Response(self._status, json.dumps(body))


class RecordingPlatform:
    """Fake ``platform_client``: records posts; may call one error handler."""

    def __init__(self, *, status: Optional[int] = None, body: Any = None) -> None:
        self.posts: List[Dict[str, Any]] = []
        self._status = status
        self._body = body

    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Any = None,
        http_errors_handlers: Any = None,
    ) -> dict:
        self.posts.append(
            {
                "endpoint": endpoint,
                "api_key": api_key,
                "payload": payload,
                "handled": sorted(http_errors_handlers or {}),
            }
        )
        if self._status is None:
            return {}

        error = RuntimeError(f"HTTP {self._status}")
        error.response = type("R", (), {"json": lambda _: self._body})()
        (http_errors_handlers or {}).get(self._status, _raise)(error)
        raise error


def _raise(error: Exception) -> None:
    raise error


def _image(value: int = 90) -> ImageData:
    return ImageData.from_tensor(torch.full((3, 12, 12), value, dtype=torch.uint8))


def _send(block: TwilioSmsNotificationBlock, **params: Any) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "sms_provider": "Custom Twilio",
        "receiver_number": "+15550001111",
        "message": "Gate: {{ $parameters.count }} trucks",
        "message_parameters": {"count": 4},
        "message_parameters_operations": {},
        "media_url": None,
        "twilio_account_sid": Secret(ACCOUNT_SID, name="TWILIO_SID"),
        "twilio_auth_token": Secret(AUTH_TOKEN, name="TWILIO_TOKEN"),
        "sender_number": "+15552223333",
        "fire_and_forget": False,
        "disable_sink": False,
        "cooldown_seconds": 0,
    }
    arguments.update(params)
    result = block.run(**arguments)

    return result


def test_custom_twilio_creates_a_message_with_the_account_credentials() -> None:
    http = RecordingTwilioHttp()
    block = TwilioSmsNotificationBlock(twilio_http_client=http)

    result = _send(block)

    assert result == SENT
    [request] = http.requests
    assert request["method"] == "POST"
    assert request["url"] == (
        f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT_SID}/Messages.json"
    )
    assert request["auth"] == (ACCOUNT_SID, AUTH_TOKEN)
    assert request["data"] == {
        "To": "+15550001111",
        "From": "+15552223333",
        "Body": "Gate: 4 trucks",
    }


def test_media_urls_pass_through_and_images_go_through_the_host_uploader() -> None:
    uploaded: List[Tuple[bytes, str]] = []

    def uploader(content: bytes, name: str) -> str:
        uploaded.append((content, name))
        return f"https://media.example.test/{len(uploaded)}.jpg"

    http = RecordingTwilioHttp()
    block = TwilioSmsNotificationBlock(twilio_http_client=http, media_uploader=uploader)

    result = _send(block, media_url=["https://cdn.example.test/a.jpg", _image()])

    assert result == SENT
    assert http.requests[0]["data"]["MediaUrl"] == [
        "https://cdn.example.test/a.jpg",
        "https://media.example.test/1.jpg",
    ]
    [(content, name)] = uploaded
    assert content[:3] == b"\xff\xd8\xff"
    assert name == "image.jpg"


def test_image_without_uploader_fails_before_anything_is_sent() -> None:
    # given: V1 uploaded images to a public temporary host on its own
    http = RecordingTwilioHttp()
    block = TwilioSmsNotificationBlock(twilio_http_client=http)

    result = _send(block, media_url=_image())

    assert result == {
        "error_status": True,
        "throttling_status": False,
        "message": NO_UPLOADER_MESSAGE,
    }
    assert http.requests == []


def test_upload_failure_fails_the_message_instead_of_dropping_the_image() -> None:
    def failing_uploader(content: bytes, name: str) -> str:
        raise ConnectionError("media host down")

    http = RecordingTwilioHttp()
    block = TwilioSmsNotificationBlock(
        twilio_http_client=http, media_uploader=failing_uploader
    )

    result = _send(block, media_url=[_image()])

    assert result["error_status"] is True
    assert result["message"].endswith("media host down")
    assert http.requests == []


def test_long_text_is_one_concatenated_sms_and_is_truncated_at_1600() -> None:
    http = RecordingTwilioHttp()
    block = TwilioSmsNotificationBlock(twilio_http_client=http)

    _send(block, message="x" * 2000, message_parameters={})

    body = http.requests[0]["data"]["Body"]
    assert len(body) == 1600
    assert body.endswith("x [...]")
    assert "MediaUrl" not in http.requests[0]["data"]


def test_twilio_error_is_reported_without_the_credentials() -> None:
    http = RecordingTwilioHttp(status=401)
    block = TwilioSmsNotificationBlock(twilio_http_client=http)

    result = _send(block)

    assert result["error_status"] is True
    assert result["message"].startswith(
        "Failed to send Twilio SMS notification. Internal error details: HTTP 401"
    )
    assert "Authenticate <secret TWILIO_TOKEN>" in result["message"]
    assert AUTH_TOKEN not in result["message"]


@pytest.mark.parametrize(
    "missing", ["twilio_account_sid", "twilio_auth_token", "sender_number"]
)
def test_missing_twilio_settings_fail_without_starting_the_cooldown(
    missing: str,
) -> None:
    http = RecordingTwilioHttp()
    block = TwilioSmsNotificationBlock(
        twilio_http_client=http, cooldown_clock=ManualClock()
    )

    first = _send(block, cooldown_seconds=60, **{missing: None})
    second = _send(block, cooldown_seconds=60)

    assert first == {
        "error_status": True,
        "throttling_status": False,
        "message": (
            "Custom Twilio requires twilio_account_sid, twilio_auth_token, and "
            "sender_number"
        ),
    }
    assert second == SENT


def test_replayed_credentials_fail_the_step_and_send_nothing() -> None:
    http = RecordingTwilioHttp()
    block = TwilioSmsNotificationBlock(twilio_http_client=http)

    with pytest.raises(SecretUnavailableError, match="TWILIO_TOKEN"):
        _send(block, twilio_auth_token=Secret.unavailable("TWILIO_TOKEN"))

    assert http.requests == []


def test_managed_provider_posts_rendered_text_urls_and_base64_images() -> None:
    platform = RecordingPlatform()
    block = TwilioSmsNotificationBlock(platform_client=platform, api_key=API_KEY)

    result = _send(
        block,
        sms_provider="Roboflow Managed API Key",
        twilio_account_sid=None,
        twilio_auth_token=None,
        media_url=["https://cdn.example.test/a.jpg", _image()],
    )

    assert result == PROXY_SENT
    [post] = platform.posts
    assert post["endpoint"] == "apiproxy/twilio"
    assert post["api_key"] == API_KEY
    assert post["handled"] == [403, 429]
    payload = post["payload"]
    assert payload["receiver_number"] == "+15550001111"
    assert payload["message"] == "Gate: 4 trucks"
    assert payload["media_urls"] == ["https://cdn.example.test/a.jpg"]
    [image] = payload["media_base64"]
    assert image["mimeType"] == "image/jpeg"
    assert base64.b64decode(image["base64"])[:3] == b"\xff\xd8\xff"
    assert "force_mms" not in payload


def test_managed_long_text_without_media_asks_for_mms() -> None:
    platform = RecordingPlatform()
    block = TwilioSmsNotificationBlock(platform_client=platform)

    _send(block, sms_provider="Roboflow Managed API Key", message="y" * 161)

    assert platform.posts[0]["payload"]["force_mms"] is True


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (
            403,
            {"details": "Number not allowed"},
            "Failed to send SMS: access forbidden. Number not allowed",
        ),
        (429, {"error": "Credits exceeded for workspace"}, CREDITS_MESSAGE),
        (500, {}, "Failed to send SMS via proxy. Internal error details: HTTP 500"),
    ],
)
def test_managed_rejections_keep_v1_messages(
    status: int, body: Dict[str, str], expected: str
) -> None:
    block = TwilioSmsNotificationBlock(
        platform_client=RecordingPlatform(status=status, body=body)
    )

    result = _send(block, sms_provider="Roboflow Managed API Key")

    assert result == {
        "error_status": True,
        "throttling_status": False,
        "message": expected,
    }


def test_failed_operation_raises_and_does_not_start_the_cooldown() -> None:
    http = RecordingTwilioHttp()
    clock = ManualClock()
    block = TwilioSmsNotificationBlock(twilio_http_client=http, cooldown_clock=clock)
    extract = {"count": [{"type": "SequenceLength"}]}

    with pytest.raises(QueryRuntimeError, match="SequenceLength"):
        _send(
            block,
            message_parameters={"count": 3},
            message_parameters_operations=extract,
            cooldown_seconds=5,
        )
    clock.advance(1)
    second = _send(
        block,
        message_parameters={"count": ["a", "b"]},
        message_parameters_operations=extract,
        cooldown_seconds=5,
    )

    assert second == SENT
    assert http.requests[0]["data"]["Body"] == "Gate: 2 trucks"


def test_cooldown_throttles_every_call_of_the_step() -> None:
    http = RecordingTwilioHttp()
    clock = ManualClock()
    block = TwilioSmsNotificationBlock(twilio_http_client=http, cooldown_clock=clock)

    first = _send(block, cooldown_seconds=5)
    clock.advance(4)
    throttled = _send(block, cooldown_seconds=5, receiver_number="+15559999999")

    assert first == SENT
    assert throttled["throttling_status"] is True
    assert len(http.requests) == 1


@pytest.mark.parametrize(
    "by_policy, message",
    [
        (True, "Sink was disabled by workflow execution policy"),
        (False, "Sink was disabled by parameter `disable_sink`"),
    ],
)
def test_disabled_sink_sends_nothing(by_policy: bool, message: str) -> None:
    http = RecordingTwilioHttp()
    block = TwilioSmsNotificationBlock(twilio_http_client=http, disable_sinks=by_policy)

    result = _send(
        block,
        disable_sink=not by_policy,
        twilio_auth_token=Secret.unavailable("TWILIO_TOKEN"),
    )

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": message,
    }
    assert http.requests == []


def test_fire_and_forget_sends_in_the_background() -> None:
    http = RecordingTwilioHttp()
    with BackgroundEffects() as effects:
        block = TwilioSmsNotificationBlock(
            twilio_http_client=http, sink_effects=effects
        )

        result = _send(block, fire_and_forget=True)
        assert effects.drain(timeout=10)

    assert result["message"] == BACKGROUND_MESSAGE
    assert len(http.requests) == 1

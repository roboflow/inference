"""Unit tests of ``v2/email_notification@v2``: loopback SMTP and a fake platform.

Every credential is an invented sentinel; SMTP goes to ``127.0.0.1`` only and
the managed provider is a recording fake, so nothing is sent anywhere.
"""

import base64
import smtplib
import threading
from typing import Any, Dict, List, Optional

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.query import QueryRuntimeError
from roboflow_workflows.v2.blocks.sinks.email import (
    NON_MEMBER_MESSAGE,
    RATE_LIMIT_MESSAGE,
    TOO_LARGE_MESSAGE,
    EmailNotificationBlock,
    SmtpSslConnector,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    BackgroundEffects,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

from tests.v2.support.sinks.loopback import ManualClock
from tests.v2.support.sinks.smtp import LoopbackSmtpServer, tls_contexts

PASSWORD = "smtp-test-SENTINEL-41c2"
API_KEY = "rf-test-SENTINEL-9e07"
SMTP_SENT = {
    "error_status": False,
    "throttling_status": False,
    "message": "Notification sent successfully",
}
PROXY_SENT = {
    "error_status": False,
    "throttling_status": False,
    "message": "Notification sent successfully via Roboflow proxy",
}


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

        error = FakeHttpError(self._status, self._body)
        handler = (http_errors_handlers or {}).get(self._status)
        if handler is not None:
            handler(error)
        raise error


class FakeHttpError(Exception):
    """Stands in for ``requests.HTTPError`` with a JSON response."""

    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"HTTP {status} for key {API_KEY}")
        self.response = self
        self._body = body

    def json(self) -> Any:
        return self._body


def _image(value: int = 128) -> ImageData:
    pixels = torch.full((3, 16, 24), value, dtype=torch.uint8)

    return ImageData.from_tensor(pixels)


def _plain_smtp(host: str, port: int) -> smtplib.SMTP:
    return smtplib.SMTP(host, port, timeout=5)


def _send(block: EmailNotificationBlock, **params: Any) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "email_provider": "Custom SMTP",
        "subject": "Gate alert",
        "receiver_email": "to@example.test",
        "message": "Seen: {{ $parameters.count }}",
        "message_parameters": {"count": 2},
        "message_parameters_operations": {},
        "sender_email": "from@example.test",
        "smtp_server": "127.0.0.1",
        "sender_email_password": PASSWORD,
        "cc_receiver_email": None,
        "bcc_receiver_email": None,
        "smtp_port": 465,
        "attachments": {},
        "fire_and_forget": False,
        "disable_sink": False,
        "cooldown_seconds": 0,
    }
    arguments.update(params)
    result = block.run(**arguments)

    return result


def test_default_tls_connector_delivers_to_to_cc_and_bcc_without_a_bcc_header() -> None:
    # given: the production connector class, trusting only a throwaway CA
    pytest.importorskip("cryptography")
    server_context, client_context = tls_contexts()
    with LoopbackSmtpServer(tls_context=server_context) as server:
        block = EmailNotificationBlock(
            smtp_connect=SmtpSslConnector(context=client_context, timeout=5)
        )

        # when
        result = _send(
            block,
            smtp_server=server.host,
            smtp_port=server.port,
            cc_receiver_email=["cc@example.test"],
            bcc_receiver_email="hidden@example.test",
        )

    # then: V1 sent only to the To list and showed Bcc to every recipient
    assert result == SMTP_SENT
    assert server.logins == [("from@example.test", PASSWORD)]
    [mail] = server.received
    assert mail.sender == "from@example.test"
    assert mail.recipients == (
        "to@example.test",
        "cc@example.test",
        "hidden@example.test",
    )
    parsed = mail.parsed()
    assert parsed["To"] == "to@example.test"
    assert parsed["Cc"] == "cc@example.test"
    assert parsed["Bcc"] is None
    assert "hidden@example.test" not in mail.data


def test_untrusted_server_certificate_is_a_failed_outcome() -> None:
    # given: the default context trusts the system store, not this test CA
    pytest.importorskip("cryptography")
    server_context, _ = tls_contexts()
    with LoopbackSmtpServer(tls_context=server_context) as server:
        block = EmailNotificationBlock(smtp_connect=SmtpSslConnector(timeout=5))

        # when
        result = _send(block, smtp_server=server.host, smtp_port=server.port)

    # then
    assert result["error_status"] is True
    assert "CERTIFICATE_VERIFY_FAILED" in result["message"]
    assert server.logins == []


def test_body_is_html_with_escaped_values_inline_images_and_named_attachments() -> None:
    with LoopbackSmtpServer() as server:
        block = EmailNotificationBlock(smtp_connect=_plain_smtp)

        result = _send(
            block,
            smtp_server=server.host,
            smtp_port=server.port,
            message=(
                "Label: {{ $parameters.label }}\n"
                "Frame: {{$parameters.frame}} {{ $parameters.unknown }}"
            ),
            message_parameters={"label": "<b>car</b>", "frame": _image()},
            attachments={
                "report.csv": "a,b\n1,2",
                "raw.bin": b"\x00\x01",
                "snapshot": _image(10),
                "my photo.jpeg": _image(20),
            },
        )

    assert result == SMTP_SENT
    parts = list(server.received[0].parsed().walk())
    html_part = next(part for part in parts if part.get_content_type() == "text/html")
    html_body = html_part.get_payload(decode=True).decode()
    assert "Label: &lt;b&gt;car&lt;/b&gt;<br>\n" in html_body
    assert '<img src="cid:image_frame" alt="frame"' in html_body
    assert "{{ $parameters.unknown }}" in html_body
    [inline] = [part for part in parts if part.get("Content-ID") == "<image_frame>"]
    assert inline.get_payload(decode=True)[:3] == b"\xff\xd8\xff"
    attached = {
        part.get_filename(): part.get_payload(decode=True)
        for part in parts
        if part.get_content_disposition() == "attachment"
    }
    assert set(attached) == {"report.csv", "raw.bin", "snapshot.jpg", "my photo.jpeg"}
    assert attached["report.csv"] == b"a,b\n1,2"
    assert attached["raw.bin"] == b"\x00\x01"
    assert attached["snapshot.jpg"][:3] == b"\xff\xd8\xff"


def test_placeholder_text_inside_a_value_is_not_expanded_again() -> None:
    with LoopbackSmtpServer() as server:
        block = EmailNotificationBlock(smtp_connect=_plain_smtp)

        _send(
            block,
            smtp_server=server.host,
            smtp_port=server.port,
            message="{{ $parameters.a }}|{{ $parameters.b }}",
            message_parameters={"a": "{{ $parameters.b }}", "b": "B"},
        )

    body = server.received[0].data
    assert "{{ $parameters.b }}|B" in body


def test_operations_run_only_for_parameters_the_message_uses() -> None:
    # given: V1 evaluates only referenced parameters on the SMTP path
    with LoopbackSmtpServer() as server:
        block = EmailNotificationBlock(smtp_connect=_plain_smtp)

        result = _send(
            block,
            smtp_server=server.host,
            smtp_port=server.port,
            message="Count: {{ $parameters.labels }}",
            message_parameters={"labels": ["car", "bus"], "unused": 3},
            message_parameters_operations={
                "labels": [{"type": "SequenceLength"}],
                "unused": [{"type": "SequenceLength"}],
            },
        )

    assert result == SMTP_SENT
    assert "Count: 2" in server.received[0].data


def test_missing_smtp_settings_fail_without_connecting_or_starting_the_cooldown() -> (
    None
):
    with LoopbackSmtpServer() as server:
        block = EmailNotificationBlock(
            smtp_connect=_plain_smtp, cooldown_clock=ManualClock()
        )
        settings = {
            "smtp_server": server.host,
            "smtp_port": server.port,
            "cooldown_seconds": 60,
        }

        missing = _send(block, sender_email_password="", **settings)
        sent = _send(block, **settings)

    assert missing == {
        "error_status": True,
        "throttling_status": False,
        "message": (
            "Custom SMTP requires sender_email, smtp_server, and sender_email_password"
        ),
    }
    assert sent == SMTP_SENT
    assert len(server.received) == 1


def test_smtp_failure_message_is_scrubbed_of_the_password() -> None:
    class LeakyServer:
        def __enter__(self) -> "LeakyServer":
            return self

        def __exit__(self, *exc_info: Any) -> None:
            return None

        def login(self, user: str, password: str) -> None:
            raise smtplib.SMTPAuthenticationError(535, f"bad password {password}")

    block = EmailNotificationBlock(smtp_connect=lambda host, port: LeakyServer())

    result = _send(block, sender_email_password=Secret(PASSWORD, name="SMTP_PASS"))

    assert result["error_status"] is True
    assert result["message"].startswith(
        "Failed to send e-mail. Internal error details:"
    )
    assert "bad password <secret SMTP_PASS>" in result["message"]
    assert PASSWORD not in result["message"]


def test_wrong_password_rejected_by_server_is_a_failed_outcome() -> None:
    with LoopbackSmtpServer(accept_password="other") as server:
        block = EmailNotificationBlock(smtp_connect=_plain_smtp)

        result = _send(block, smtp_server=server.host, smtp_port=server.port)

    assert result["error_status"] is True
    assert "535" in result["message"]
    assert server.received == []


def test_replayed_password_fails_the_step_and_sends_nothing() -> None:
    connections: List[Any] = []
    block = EmailNotificationBlock(
        smtp_connect=lambda host, port: connections.append(1)
    )

    with pytest.raises(SecretUnavailableError, match="SMTP_PASS"):
        _send(block, sender_email_password=Secret.unavailable("SMTP_PASS"))

    assert connections == []


def test_managed_provider_posts_unrendered_message_with_processed_parameters() -> None:
    # given
    platform = RecordingPlatform()
    block = EmailNotificationBlock(
        platform_client=platform, api_key=Secret(API_KEY, name="ROBOFLOW_API_KEY")
    )
    image = _image()

    # when
    result = _send(
        block,
        email_provider="Roboflow Managed API Key",
        receiver_email=["a@example.test", "b@example.test"],
        cc_receiver_email="cc@example.test",
        message="Seen {{ $parameters.labels }}",
        message_parameters={"labels": ["car", "bus"], "frame": image, "site": "north"},
        message_parameters_operations={"labels": [{"type": "SequenceLength"}]},
        attachments={"report.csv": "x,y", "snap": image},
    )

    # then
    assert result == PROXY_SENT
    [post] = platform.posts
    assert post["endpoint"] == "apiproxy/email"
    assert post["api_key"] == API_KEY
    assert post["handled"] == [403, 413, 429]
    payload = post["payload"]
    assert payload["receiver_email"] == ["a@example.test", "b@example.test"]
    assert payload["cc_receiver_email"] == ["cc@example.test"]
    assert "bcc_receiver_email" not in payload
    assert payload["message"] == "Seen {{ $parameters.labels }}"
    assert payload["message_parameters"]["labels"] == 2
    assert payload["message_parameters"]["site"] == "north"
    assert (
        base64.b64decode(payload["message_parameters"]["frame"])[:3] == b"\xff\xd8\xff"
    )
    assert base64.b64decode(payload["attachments"]["report.csv"]) == b"x,y"
    assert base64.b64decode(payload["attachments"]["snap.jpg"])[:3] == b"\xff\xd8\xff"


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (403, {"details": "Recipients are non-workspace members"}, NON_MEMBER_MESSAGE),
        (
            403,
            {"error": "Workspace blocked"},
            "Failed to send email: access forbidden. Workspace blocked",
        ),
        (413, {"error": "Request Entity"}, TOO_LARGE_MESSAGE),
        (429, {"details": "Rate limit reached"}, RATE_LIMIT_MESSAGE),
        (429, {}, "Failed to send email via proxy. HTTP 429 for key <secret api_key>"),
    ],
)
def test_managed_rejections_keep_v1_messages(
    status: int, body: Dict[str, str], expected: str
) -> None:
    platform = RecordingPlatform(status=status, body=body)
    block = EmailNotificationBlock(platform_client=platform, api_key=API_KEY)

    result = _send(block, email_provider="Roboflow Managed API Key")

    assert result == {
        "error_status": True,
        "throttling_status": False,
        "message": expected,
    }


def test_unhandled_managed_error_is_generic_and_scrubbed() -> None:
    platform = RecordingPlatform(status=500, body={})
    block = EmailNotificationBlock(platform_client=platform, api_key=API_KEY)

    result = _send(block, email_provider="Roboflow Managed API Key")

    assert result["message"] == (
        "Failed to send e-mail via proxy. Internal error details: "
        "HTTP 500 for key <secret api_key>"
    )


def test_offline_default_platform_reports_how_to_configure_the_host() -> None:
    result = _send(EmailNotificationBlock(), email_provider="Roboflow Managed API Key")

    assert result["error_status"] is True
    assert "provided no platform client" in result["message"]


def test_managed_parameter_that_is_not_json_fails_before_posting() -> None:
    platform = RecordingPlatform()
    block = EmailNotificationBlock(platform_client=platform)

    result = _send(
        block,
        email_provider="Roboflow Managed API Key",
        message_parameters={"value": object()},
    )

    assert result["error_status"] is True
    assert "not JSON serializable" in result["message"]
    assert platform.posts == []


def test_failed_operation_raises_and_does_not_start_the_cooldown() -> None:
    platform = RecordingPlatform()
    clock = ManualClock()
    block = EmailNotificationBlock(platform_client=platform, cooldown_clock=clock)
    extract = {"labels": [{"type": "SequenceLength"}]}
    managed = {"email_provider": "Roboflow Managed API Key", "cooldown_seconds": 5}

    with pytest.raises(QueryRuntimeError, match="SequenceLength"):
        _send(
            block,
            message_parameters={"labels": 3},
            message_parameters_operations=extract,
            **managed,
        )
    clock.advance(1)
    second = _send(
        block,
        message_parameters={"labels": ["car"]},
        message_parameters_operations=extract,
        **managed,
    )

    assert second == PROXY_SENT


def test_cooldown_throttles_every_call_of_the_step() -> None:
    platform = RecordingPlatform()
    clock = ManualClock()
    block = EmailNotificationBlock(platform_client=platform, cooldown_clock=clock)
    managed = {"email_provider": "Roboflow Managed API Key", "cooldown_seconds": 5}

    first = _send(block, **managed)
    clock.advance(4.9)
    throttled = _send(block, receiver_email="other@example.test", **managed)
    clock.advance(0.1)
    third = _send(block, **managed)

    assert first == third == PROXY_SENT
    assert throttled == {
        "error_status": False,
        "throttling_status": True,
        "message": "Sink cooldown applies",
    }
    assert len(platform.posts) == 2


@pytest.mark.parametrize(
    "by_policy, message",
    [
        (True, "Sink was disabled by workflow execution policy"),
        (False, "Sink was disabled by parameter `disable_sink`"),
    ],
)
def test_disabled_sink_sends_nothing(by_policy: bool, message: str) -> None:
    platform = RecordingPlatform()
    block = EmailNotificationBlock(platform_client=platform, disable_sinks=by_policy)

    result = _send(
        block,
        email_provider="Roboflow Managed API Key",
        disable_sink=not by_policy,
        sender_email_password=Secret.unavailable("SMTP_PASS"),
    )

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": message,
    }
    assert platform.posts == []


def test_fire_and_forget_hands_sending_to_background_effects() -> None:
    gate = threading.Event()
    sent: List[str] = []

    class SlowPlatform(RecordingPlatform):
        def post(self, endpoint: str, *args: Any, **kwargs: Any) -> dict:
            gate.wait(timeout=10)
            sent.append(endpoint)
            return {}

    with BackgroundEffects() as effects:
        block = EmailNotificationBlock(
            platform_client=SlowPlatform(), sink_effects=effects
        )
        result = _send(
            block, email_provider="Roboflow Managed API Key", fire_and_forget=True
        )
        assert sent == []
        gate.set()
        assert effects.drain(timeout=10)

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": BACKGROUND_MESSAGE,
    }
    assert sent == ["apiproxy/email"]


def test_params_reject_empty_receivers_and_unknown_operation_keys() -> None:
    base = {"subject": "s", "message": "m"}

    with pytest.raises(ValidationError, match="at least one receiver"):
        EmailNotificationBlock.Params.model_validate({**base, "receiver_email": []})
    with pytest.raises(ValidationError, match="`labels`, which is not a key"):
        EmailNotificationBlock.Params.model_validate(
            {
                **base,
                "receiver_email": "a@example.test",
                "message_parameters": {"count": 1},
                "message_parameters_operations": {
                    "labels": [{"type": "SequenceLength"}]
                },
            }
        )

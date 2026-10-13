"""Unit tests of ``v2/webhook_sink@v1`` with in-process and loopback transports."""

import threading
from typing import Any, Dict

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.query import QueryRuntimeError
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    BackgroundEffects,
)
from roboflow_workflows.v2.blocks.sinks.webhook import (
    HttpWebhookClient,
    WebhookRequest,
    WebhookSinkBlock,
)

from tests.v2.support.sinks.loopback import (
    LoopbackHttpServer,
    ManualClock,
    RecordingWebhookClient,
)

URL = "http://hooks.test/notify"
SENT = {
    "error_status": False,
    "throttling_status": False,
    "message": "Notification sent successfully",
}


def _send(block: WebhookSinkBlock, **params: Any) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "url": URL,
        "method": "POST",
        "query_parameters": {},
        "headers": {},
        "json_payload": {},
        "json_payload_operations": {},
        "multi_part_encoded_files": {},
        "multi_part_encoded_files_operations": {},
        "form_data": {},
        "form_data_operations": {},
        "request_timeout": 2,
        "fire_and_forget": False,
        "disable_sink": False,
        "cooldown_seconds": 0,
    }
    arguments.update(params)
    result = block.run(**arguments)

    return result


def test_request_carries_every_configured_part() -> None:
    client = RecordingWebhookClient()
    block = WebhookSinkBlock(webhook_client=client)

    result = _send(
        block,
        method="PUT",
        query_parameters={"site": "north", "ids": [1, 2]},
        headers={"X-Camera": "gate"},
        json_payload={"count": 3, "labels": ["car"]},
        form_data={"note": "ok"},
        multi_part_encoded_files={"report": "text"},
        request_timeout=1.5,
    )

    assert result == SENT
    assert client.requests == [
        WebhookRequest(
            method="PUT",
            url=URL,
            query_parameters={"site": "north", "ids": [1, 2]},
            headers={"X-Camera": "gate"},
            json_payload={"count": 3, "labels": ["car"]},
            form_data={"note": "ok"},
            files={"report": "text"},
            timeout=1.5,
        )
    ]


def test_request_holds_a_copy_not_the_workflow_value() -> None:
    client = RecordingWebhookClient()
    labels = ["car"]

    _send(WebhookSinkBlock(webhook_client=client), json_payload={"labels": labels})
    labels.append("bus")

    assert client.requests[0].json_payload == {"labels": ["car"]}


def test_operations_transform_payload_values_first() -> None:
    client = RecordingWebhookClient()

    _send(
        WebhookSinkBlock(webhook_client=client),
        json_payload={"labels": ["car", "bus"], "site": "north"},
        json_payload_operations={"labels": [{"type": "SequenceLength"}]},
        form_data={"site": "north"},
        form_data_operations={"site": [{"type": "StringToUpperCase"}]},
    )

    assert client.requests[0].json_payload == {"labels": 2, "site": "north"}
    assert client.requests[0].form_data == {"site": "NORTH"}


def test_unserializable_payload_fails_before_sending() -> None:
    client = RecordingWebhookClient()

    result = _send(
        WebhookSinkBlock(webhook_client=client),
        json_payload={"boxes": torch.zeros(2, 4)},
        fire_and_forget=True,
    )

    assert result["error_status"] is True
    assert "json_payload is not JSON data" in result["message"]
    assert "Tensor" in result["message"]
    assert client.requests == []


def test_transport_error_is_reported_like_v1() -> None:
    client = RecordingWebhookClient(error=ConnectionError("refused"))

    result = _send(WebhookSinkBlock(webhook_client=client))

    assert result == {
        "error_status": True,
        "throttling_status": False,
        "message": (
            "Failed to send webhook notification. Internal error details: refused"
        ),
    }


def test_failed_operation_does_not_start_the_cooldown_window() -> None:
    # given: V1 starts the window only after the payload operations succeed
    client = RecordingWebhookClient()
    clock = ManualClock()
    block = WebhookSinkBlock(webhook_client=client, cooldown_clock=clock)
    extract = {"labels": [{"type": "SequenceLength"}]}

    # when: the first call's operation fails, the next one is valid
    with pytest.raises(QueryRuntimeError, match="SequenceLength"):
        _send(
            block,
            cooldown_seconds=5,
            json_payload={"labels": 3},
            json_payload_operations=extract,
        )
    clock.advance(1)
    second = _send(
        block,
        cooldown_seconds=5,
        json_payload={"labels": ["car"]},
        json_payload_operations=extract,
    )

    # then: the valid call is sent, not throttled
    assert second == SENT
    assert client.requests[0].json_payload == {"labels": 1}


@pytest.mark.parametrize("field", ["query_parameters", "headers"])
def test_query_or_header_value_that_is_not_json_is_an_error_outcome(
    field: str,
) -> None:
    client = RecordingWebhookClient()

    result = _send(
        WebhookSinkBlock(webhook_client=client), **{field: {"n": np.int64(3)}}
    )

    assert result["error_status"] is True
    assert f"{field} is not JSON data" in result["message"]
    assert client.requests == []


def test_cooldown_throttles_every_call_of_the_step() -> None:
    client = RecordingWebhookClient()
    clock = ManualClock()
    block = WebhookSinkBlock(webhook_client=client, cooldown_clock=clock)

    first = _send(block, cooldown_seconds=5)
    clock.advance(1)
    second = _send(block, cooldown_seconds=5, url="http://other.test/")
    clock.advance(4)
    third = _send(block, cooldown_seconds=5)

    assert [first, third] == [SENT, SENT]
    assert second == {
        "error_status": False,
        "throttling_status": True,
        "message": "Sink cooldown applies",
    }
    assert len(client.requests) == 2


@pytest.mark.parametrize(
    "by_policy, message",
    [
        (True, "Sink was disabled by workflow execution policy"),
        (False, "Sink was disabled by parameter `disable_sink`"),
    ],
)
def test_disabled_sink_sends_nothing(by_policy: bool, message: str) -> None:
    client = RecordingWebhookClient()
    block = WebhookSinkBlock(webhook_client=client, disable_sinks=by_policy)

    result = _send(block, disable_sink=not by_policy)

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": message,
    }
    assert client.requests == []


def test_fire_and_forget_hands_the_request_to_background_effects() -> None:
    gate = threading.Event()
    client = RecordingWebhookClient(gate=gate)
    with BackgroundEffects() as effects:
        block = WebhookSinkBlock(webhook_client=client, sink_effects=effects)

        result = _send(block, fire_and_forget=True)
        sent_before_gate = len(client.requests)
        gate.set()
        effects.drain(timeout=5)

    assert result["message"] == BACKGROUND_MESSAGE
    assert sent_before_gate == 0
    assert len(client.requests) == 1
    assert effects.stats().completed == 1


def test_background_failure_is_observable_on_the_resource() -> None:
    client = RecordingWebhookClient(error=TimeoutError("slow"))
    with BackgroundEffects() as effects:
        block = WebhookSinkBlock(webhook_client=client, sink_effects=effects)

        result = _send(block, fire_and_forget=True)
        effects.drain(timeout=5)

    assert result["error_status"] is False
    assert effects.stats().failed == 1
    assert effects.failures()[0].message.endswith("Internal error details: slow")


def test_refused_background_request_is_an_error_outcome() -> None:
    gate = threading.Event()
    client = RecordingWebhookClient(gate=gate)
    with BackgroundEffects(max_pending=1, when_full="reject") as effects:
        block = WebhookSinkBlock(webhook_client=client, sink_effects=effects)

        results = [_send(block, fire_and_forget=True) for _ in range(2)]
        gate.set()

    assert [result["error_status"] for result in results] == [False, True]
    assert "not run" in results[1]["message"]
    assert len(client.requests) == 1


def test_http_client_posts_json_to_a_loopback_server() -> None:
    with LoopbackHttpServer() as server:
        block = WebhookSinkBlock(
            webhook_client=HttpWebhookClient(allow_non_global_addresses=True)
        )

        result = _send(
            block,
            url=server.url(),
            query_parameters={"site": "north"},
            headers={"X-Camera": "gate"},
            json_payload={"count": 3},
        )

    assert result == SENT
    [received] = server.received
    assert (received.method, received.path) == ("POST", "/hook")
    assert received.query == {"site": ["north"]}
    assert received.headers["x-camera"] == "gate"
    assert received.json() == {"count": 3}


@pytest.mark.parametrize("status, expected", [(500, "HTTP 500"), (302, "redirect 302")])
def test_http_client_status_errors_leave_out_url_credentials_and_query(
    status: int, expected: str
) -> None:
    with LoopbackHttpServer(status=status) as server:
        block = WebhookSinkBlock(
            webhook_client=HttpWebhookClient(allow_non_global_addresses=True)
        )
        url = server.url().replace("http://", "http://user:pw-secret@")

        result = _send(block, url=f"{url}?token=inurl-secret")

    assert result["error_status"] is True
    assert expected in result["message"]
    assert server.url() in result["message"]
    assert "pw-secret" not in result["message"]
    assert "inurl-secret" not in result["message"]
    assert server.received[0].query == {"token": ["inurl-secret"]}


def test_http_client_refuses_loopback_by_default() -> None:
    with LoopbackHttpServer() as server:
        block = WebhookSinkBlock(webhook_client=HttpWebhookClient())

        result = _send(block, url=server.url())

    assert result["error_status"] is True
    assert server.received == []

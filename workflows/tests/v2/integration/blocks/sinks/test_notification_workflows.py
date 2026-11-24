"""Notification sinks in compiled workflows: real compiler, session and executor.

Credentials are invented sentinels from ``StaticSecrets`` or secret inputs.
SMTP goes only to a ``127.0.0.1`` server; managed providers are a recording
``platform_client``::

    secrets resource ─▶ roboflow_core/environment_secrets_store@v1 ─▶ roboflow_core/email_notification@v2
    image, count (batch) ──────────────────────────────────▶   └▶ loopback SMTP
    mute, count ─▶ roboflow_core/twilio_sms_notification@v2 ─▶ platform_client (+ sink_effects)
"""

import smtplib
from typing import Any, Dict, List, Optional

import pytest
import torch
from roboflow_workflows.v2.blocks.secrets import (
    EnvironmentSecretsStoreBlock,
    StaticSecrets,
)
from roboflow_workflows.v2.blocks.sinks.email_notification.v2 import EmailNotificationBlock
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.blocks.sinks.slack_notification.v1 import SlackNotificationBlock
from roboflow_workflows.v2.blocks.sinks.twilio_sms_notification.v2 import TwilioSmsNotificationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.secrets import SECRET_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.support.sinks.smtp import LoopbackSmtpServer

PASSWORD = "smtp-test-SENTINEL-41c2"
API_KEY = "rf-test-SENTINEL-9e07"
CATALOGUE = Catalogue(
    [
        EnvironmentSecretsStoreBlock,
        EmailNotificationBlock,
        SlackNotificationBlock,
        TwilioSmsNotificationBlock,
    ],
    kinds=[SECRET_KIND, IMAGE_KIND],
)
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
COUNT_INPUT = {"type": "WorkflowBatchInput", "name": "count", "kind": ["integer"]}


class RecordingPlatform:
    """Fake ``platform_client``: records posts and returns an empty response."""

    def __init__(self) -> None:
        self.posts: List[Dict[str, Any]] = []

    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Any = None,
        http_errors_handlers: Any = None,
    ) -> dict:
        self.posts.append(
            {"endpoint": endpoint, "api_key": api_key, "payload": payload}
        )
        return {}


def _plain_smtp(host: str, port: int) -> smtplib.SMTP:
    return smtplib.SMTP(host, port, timeout=5)


def _image(value: int) -> ImageData:
    return ImageData.from_tensor(torch.full((3, 10, 14), value, dtype=torch.uint8))


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _status_outputs(step: str) -> List[Dict[str, str]]:
    outputs = [
        _output(name, f"$steps.{step}.{name}")
        for name in ("error_status", "throttling_status", "message")
    ]

    return outputs


def _compile(steps: List[Dict[str, Any]], outputs: List[Dict[str, str]], inputs: List):
    definition = {
        "version": "2.0",
        "inputs": inputs,
        "steps": steps,
        "outputs": outputs,
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def test_stored_password_sends_one_email_per_cooldown_with_image_attachment() -> None:
    # given: a password from the host's secrets and a batch of two frames
    steps = [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "secrets",
            "variables_storing_secrets": ["SMTP_PASSWORD"],
        },
        {
            "type": "roboflow_core/email_notification@v2",
            "name": "email",
            "email_provider": "Custom SMTP",
            "subject": "Gate alert",
            "receiver_email": "ops@example.test",
            "message": "{{ $parameters.count }} trucks",
            "message_parameters": {"count": "$inputs.count"},
            "sender_email": "camera@example.test",
            "smtp_server": "127.0.0.1",
            "sender_email_password": "$steps.secrets.smtp_password",
            "attachments": {"frame": "$inputs.image"},
            "fire_and_forget": False,
            "cooldown_seconds": 60,
        },
    ]
    with LoopbackSmtpServer() as server:
        steps[1]["smtp_port"] = server.port  # known only once the server runs
        plan = _compile(steps, _status_outputs("email"), [IMAGE_INPUT, COUNT_INPUT])
        session = plan.create_session(
            resources={
                "secrets": StaticSecrets({"SMTP_PASSWORD": PASSWORD}),
                "smtp_connect": _plain_smtp,
            }
        )

        # when
        result = session.run({"image": [_image(40), _image(200)], "count": [3, 5]})

    # then: the second row falls inside the step's cooldown window
    rows = result.rows(serialize=True)
    assert rows == [
        {
            "error_status": False,
            "throttling_status": False,
            "message": "Notification sent successfully",
        },
        {
            "error_status": False,
            "throttling_status": True,
            "message": "Sink cooldown applies",
        },
    ]
    assert server.logins == [("camera@example.test", PASSWORD)]
    [mail] = server.received
    assert "3 trucks" in mail.data
    attachments = [
        part.get_filename()
        for part in mail.parsed().walk()
        if part.get_content_disposition() == "attachment"
    ]
    assert attachments == ["frame.jpg"]
    assert PASSWORD not in repr(result.rows()) + repr(rows)


def test_managed_sms_follows_a_runtime_mute_input_and_runs_in_the_background() -> None:
    # given
    steps = [
        {
            "type": "roboflow_core/twilio_sms_notification@v2",
            "name": "sms",
            "receiver_number": "+15550001111",
            "message": "Gate: {{ $parameters.count }} trucks",
            "message_parameters": {"count": "$inputs.count"},
            "disable_sink": "$inputs.mute",
            "cooldown_seconds": 0,
        }
    ]
    inputs = [
        COUNT_INPUT,
        {"type": "WorkflowParameter", "name": "mute", "kind": ["boolean"]},
    ]
    plan = _compile(steps, _status_outputs("sms"), inputs)
    platform = RecordingPlatform()

    # when
    with BackgroundEffects() as effects:
        session = plan.create_session(
            resources={
                "platform_client": platform,
                "api_key": API_KEY,
                "sink_effects": effects,
            }
        )
        muted = session.run({"count": [1], "mute": True}).rows()
        sent = session.run({"count": [2, 7], "mute": False}).rows()
        assert effects.drain(timeout=10)
    stats = effects.stats()

    # then
    assert muted[0]["message"] == "Sink was disabled by parameter `disable_sink`"
    assert [row["message"] for row in sent] == [
        "Notification sent in the background task"
    ] * 2
    assert stats.completed == 2
    bodies = sorted(post["payload"]["message"] for post in platform.posts)
    assert bodies == ["Gate: 2 trucks", "Gate: 7 trucks"]
    assert {post["api_key"] for post in platform.posts} == {API_KEY}


def test_operation_key_that_is_not_a_parameter_fails_compilation() -> None:
    steps = [
        {
            "type": "roboflow_core/slack_notification@v1",
            "name": "slack",
            "slack_token": "$inputs.token",
            "channel": "C1",
            "message": "{{ $parameters.count }}",
            "message_parameters": {"count": "$inputs.count"},
            "message_parameters_operations": {"labels": [{"type": "SequenceLength"}]},
        }
    ]
    inputs = [
        COUNT_INPUT,
        {"type": "WorkflowParameter", "name": "token", "kind": ["secret"]},
    ]

    with pytest.raises(
        ParamsValidationError,
        match="`labels`, which is not a key of message_parameters",
    ):
        _compile(steps, _status_outputs("slack"), inputs)

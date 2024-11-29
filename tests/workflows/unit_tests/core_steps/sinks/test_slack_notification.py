import time
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from slack_sdk.errors import SlackApiError
from slack_sdk.web import SlackResponse

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StringToUpperCase,
)
from inference.core.workflows.core_steps.sinks.slack.notification import v1
from inference.core.workflows.core_steps.sinks.slack.notification.v1 import (
    BlockManifest,
    SlackNotificationBlockV1,
    format_message,
    send_slack_notification,
)


def test_manifest_parsing_when_the_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/slack_notification@v1",
        "name": "slack_notification",
        "slack_token": "$inputs.slack_token",
        "message": "My message",
        "channel": "$inputs.slack_channel",
        "message_parameters": {
            "image": "$inputs.image",
        },
        "message_parameters_operations": {
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        "attachments": {
            "form_field": "$inputs.query_parameter",
        },
        "fire_and_forget": True,
        "cooldown_seconds": "$inputs.cooldown",
        "cooldown_session_key": "unique-session-key",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/slack_notification@v1",
        name="slack_notification",
        slack_token="$inputs.slack_token",
        channel="$inputs.slack_channel",
        message="My message",
        message_parameters={
            "image": "$inputs.image",
        },
        message_parameters_operations={
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        attachments={
            "form_field": "$inputs.query_parameter",
        },
        fire_and_forget=True,
        cooldown_seconds="$inputs.cooldown",
        cooldown_session_key="unique-session-key",
    )


def test_manifest_parsing_when_cooldown_seconds_is_invalid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/slack_notification@v1",
        "name": "slack_notification",
        "slack_token": "$inputs.slack_token",
        "message": "My message",
        "channel": "$inputs.slack_channel",
        "message_parameters": {
            "image": "$inputs.image",
        },
        "message_parameters_operations": {
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        "attachments": {
            "form_field": "$inputs.query_parameter",
        },
        "fire_and_forget": True,
        "cooldown_seconds": -1,
        "cooldown_session_key": "unique-session-key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_format_message_when_multiple_occurrences_of_the_same_parameter_exist() -> None:
    # given
    message = "This is example param: {{{ $parameters.param }}} - and this is aloso param: `{{ $parameters.param }}`"

    # when
    result = format_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={},
    )

    # then
    assert result == "This is example param: {some} - and this is aloso param: `some`"


def test_format_message_when_multiple_parameters_exist() -> None:
    # given
    message = "This is example param: {{ $parameters.param }} - and this is aloso param: `{{ $parameters.other }}`"

    # when
    result = format_message(
        message=message,
        message_parameters={"param": "some", "other": 42},
        message_parameters_operations={},
    )

    # then
    assert result == "This is example param: some - and this is aloso param: `42`"


def test_format_message_when_different_combinations_of_whitespaces_exist_in_template_parameter_anchor() -> (
    None
):
    # given
    message = "{{{ $parameters.param }}} - {{$parameters.param }} - {{ $parameters.param}} - {{     $parameters.param     }}"

    # when
    result = format_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={},
    )

    # then
    assert result == "{some} - some - some - some"


def test_format_message_when_operation_to_apply_on_parameter() -> None:
    # given
    message = "This is example param: {{{ $parameters.param }}} - and this is aloso param: `{{ $parameters.param }}`"

    # when
    result = format_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={
            "param": [StringToUpperCase(type="StringToUpperCase")]
        },
    )

    # then
    assert result == "This is example param: {SOME} - and this is aloso param: `SOME`"


def test_send_slack_notification_when_slack_api_error_raised() -> None:
    # given
    client = MagicMock()
    client.chat_postMessage.side_effect = SlackApiError(
        message="some",
        response=SlackResponse(
            client=client,
            http_verb="",
            api_url="",
            req_args={},
            data={},
            headers={},
            status_code=500,
        ),
    )

    # when
    result = send_slack_notification(
        client=client,
        channel="some",
        message="msg",
        attachments={},
    )

    # then
    assert result[0] is True
    assert result[1].startswith("Failed to send Slack notification")


def test_send_slack_notification_when_generic_error_raised() -> None:
    # given
    client = MagicMock()
    client.chat_postMessage.side_effect = Exception()

    # when
    result = send_slack_notification(
        client=client,
        channel="some",
        message="msg",
        attachments={},
    )

    # then
    assert result[0] is True
    assert result[1].startswith("Failed to send Slack notification")


def test_send_slack_notification_when_operation_succeeds_without_attachments() -> None:
    # given
    client = MagicMock()

    # when
    result = send_slack_notification(
        client=client,
        channel="some",
        message="msg",
        attachments={},
    )

    # then
    client.chat_postMessage.assert_called_once_with(
        channel="some",
        text="msg",
    )
    assert result[0] is False


def test_send_slack_notification_when_operation_succeeds_with_attachments() -> None:
    # given
    client = MagicMock()

    # when
    result = send_slack_notification(
        client=client,
        channel="some",
        message="msg",
        attachments={"some": b"data"},
    )

    # then
    client.chat_postMessage.files_upload_v2(
        channel="some",
        initial_comment="msg",
        file_uploads=[{"title": "some", "content": b"data"}],
    )
    assert result[0] is False


def test_cooldown_in_slack_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = SlackNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            slack_token="token",
            message="some",
            channel="channel",
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=100,
            cooldown_session_key="unique",
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


def test_cooldown_in_slack_notification_bloc_for_separate_sessionsk() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = SlackNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for i in range(2):
        result = block.run(
            slack_token="token",
            message="some",
            channel="channel",
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=100,
            cooldown_session_key=f"unique-{i}",
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


def test_disabling_cooldown_in_slack_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = SlackNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            slack_token="token",
            message="some",
            channel="channel",
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=0,
            cooldown_session_key="unique",
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


def test_cooldown_recovery_in_slack_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = SlackNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            slack_token="token",
            message="some",
            channel="channel",
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=1,
            cooldown_session_key="unique",
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


@mock.patch.object(v1, "send_slack_notification")
def test_sending_slack_notification_synchronously(
    send_slack_notification_mock: MagicMock,
) -> None:
    # given
    send_slack_notification_mock.return_value = (False, "ok")
    block = SlackNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=MagicMock(),
    )

    # when
    result = block.run(
        slack_token="token",
        message="some",
        channel="channel",
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=1,
        cooldown_session_key="unique",
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "ok",
    }


def test_disabling_slack_notification() -> None:
    # given
    block = SlackNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=MagicMock(),
    )

    # when
    result = block.run(
        slack_token="token",
        message="some",
        channel="channel",
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        fire_and_forget=False,
        disable_sink=True,
        cooldown_seconds=1,
        cooldown_session_key="unique",
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Sink was disabled by parameter `disable_sink`",
    }


def test_sending_slack_notification_asynchronously_in_background_tasks() -> None:
    # given
    background_tasks = MagicMock()
    block = SlackNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        slack_token="token",
        message="some",
        channel="channel",
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
        cooldown_session_key="unique",
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    background_tasks.add_task.assert_called_once()


def test_sending_slack_notification_asynchronously_in_thread_pool_executor() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = SlackNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    result = block.run(
        slack_token="token",
        message="some",
        channel="channel",
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
        cooldown_session_key="unique",
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    thread_pool_executor.submit.assert_called_once()

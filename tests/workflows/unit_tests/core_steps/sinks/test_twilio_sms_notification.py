import time
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StringToUpperCase,
)
from inference.core.workflows.core_steps.sinks.twilio.sms import v1
from inference.core.workflows.core_steps.sinks.twilio.sms.v1 import (
    BlockManifest,
    TwilioSMSNotificationBlockV1,
    format_message,
    send_sms_notification,
)


def test_manifest_parsing_when_the_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/twilio_sms_notification@v1",
        "name": "twilio_sms_notification",
        "twilio_account_sid": "$inputs.sid",
        "twilio_auth_token": "$inputs.auth_token",
        "message": "My message",
        "sender_number": "some",
        "receiver_number": "other",
        "message_parameters": {
            "image": "$inputs.image",
        },
        "message_parameters_operations": {
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        "fire_and_forget": True,
        "cooldown_seconds": "$inputs.cooldown",
        "cooldown_session_key": "unique-session-key",
        "length_limit": 160,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/twilio_sms_notification@v1",
        name="twilio_sms_notification",
        twilio_account_sid="$inputs.sid",
        twilio_auth_token="$inputs.auth_token",
        message="My message",
        sender_number="some",
        receiver_number="other",
        message_parameters={
            "image": "$inputs.image",
        },
        message_parameters_operations={
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        fire_and_forget=True,
        cooldown_seconds="$inputs.cooldown",
        cooldown_session_key="unique-session-key",
        length_limit=160,
    )


def test_manifest_parsing_when_cooldown_seconds_is_invalid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/twilio_sms_notification@v1",
        "name": "twilio_sms_notification",
        "twilio_account_sid": "$inputs.sid",
        "twilio_auth_token": "$inputs.auth_token",
        "message": "My message",
        "sender_number": "some",
        "receiver_number": "other",
        "message_parameters": {
            "image": "$inputs.image",
        },
        "message_parameters_operations": {
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        "fire_and_forget": True,
        "cooldown_seconds": -1,
        "cooldown_session_key": "unique-session-key",
        "length_limit": 160,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_manifest_parsing_when_length_limit_is_invalid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/twilio_sms_notification@v1",
        "name": "twilio_sms_notification",
        "twilio_account_sid": "$inputs.sid",
        "twilio_auth_token": "$inputs.auth_token",
        "message": "My message",
        "sender_number": "some",
        "receiver_number": "other",
        "message_parameters": {
            "image": "$inputs.image",
        },
        "message_parameters_operations": {
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        "fire_and_forget": True,
        "cooldown_seconds": 5,
        "cooldown_session_key": "unique-session-key",
        "length_limit": 0,
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
        length_limit=1024,
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
        length_limit=1024,
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
        length_limit=1024,
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
        length_limit=1024,
    )

    # then
    assert result == "This is example param: {SOME} - and this is aloso param: `SOME`"


def test_format_message_when_output_needs_to_be_truncated() -> None:
    # given
    message = "This is example param: {{{ $parameters.param }}} - and this is aloso param: `{{ $parameters.param }}`"

    # when
    result = format_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={
            "param": [StringToUpperCase(type="StringToUpperCase")]
        },
        length_limit=32,
    )

    # then
    assert result == "This is example param: {SO [...]"
    assert len(result) == 32


def test_send_twilio_sms_notification_when_error_raised() -> None:
    # given
    client = MagicMock()
    client.messages.create = Exception()

    # when
    result = send_sms_notification(
        client=client,
        message="msg",
        sender_number="some",
        receiver_number="other",
    )

    # then
    assert result[0] is True
    assert result[1].startswith("Failed to send Twilio SMS notification")


def test_send_twilio_sms_notification_when_operation_succeeded() -> None:
    # given
    client = MagicMock()

    # when
    result = send_sms_notification(
        client=client,
        message="msg",
        sender_number="some",
        receiver_number="other",
    )

    # then
    assert result[0] is False
    client.messages.create.assert_called_once_with(
        body="msg",
        from_="some",
        to="other",
    )


def test_cooldown_in_twilio_sms_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = TwilioSMSNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            twilio_account_sid="token",
            twilio_auth_token="token",
            message="some",
            sender_number="from",
            receiver_number="to",
            message_parameters={},
            message_parameters_operations={},
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=100,
            cooldown_session_key="unique",
            length_limit=128,
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


def test_cooldown_in_twilio_sms_notification_block_for_separate_sessions() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = TwilioSMSNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for i in range(2):
        result = block.run(
            twilio_account_sid="token",
            twilio_auth_token="token",
            message="some",
            sender_number="from",
            receiver_number="to",
            message_parameters={},
            message_parameters_operations={},
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=100,
            cooldown_session_key=f"unique-{i}",
            length_limit=128,
        )
        results.append(result)

    # then
    assert results[0] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    assert results[0] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }


def test_disabling_cooldown_in_twilio_sms_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = TwilioSMSNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            twilio_account_sid="token",
            twilio_auth_token="token",
            message="some",
            sender_number="from",
            receiver_number="to",
            message_parameters={},
            message_parameters_operations={},
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=0,
            cooldown_session_key=f"unique",
            length_limit=128,
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


def test_cooldown_recovery_in_twilio_sms_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = TwilioSMSNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            twilio_account_sid="token",
            twilio_auth_token="token",
            message="some",
            sender_number="from",
            receiver_number="to",
            message_parameters={},
            message_parameters_operations={},
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=1,
            cooldown_session_key=f"unique",
            length_limit=128,
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


@mock.patch.object(v1, "send_sms_notification")
def test_sending_twilio_sms_notification_synchronously(
    send_sms_notification_mock: MagicMock,
) -> None:
    # given
    send_sms_notification_mock.return_value = (False, "ok")
    block = TwilioSMSNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=MagicMock(),
    )

    # when
    result = block.run(
        twilio_account_sid="token",
        twilio_auth_token="token",
        message="some",
        sender_number="from",
        receiver_number="to",
        message_parameters={},
        message_parameters_operations={},
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=1,
        cooldown_session_key=f"unique",
        length_limit=128,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "ok",
    }


def test_disabling_twilio_sms_notification() -> None:
    # given
    block = TwilioSMSNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=MagicMock(),
    )

    # when
    result = block.run(
        twilio_account_sid="token",
        twilio_auth_token="token",
        message="some",
        sender_number="from",
        receiver_number="to",
        message_parameters={},
        message_parameters_operations={},
        fire_and_forget=False,
        disable_sink=True,
        cooldown_seconds=1,
        cooldown_session_key=f"unique",
        length_limit=128,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Sink was disabled by parameter `disable_sink`",
    }


def test_sending_twilio_sms_notification_asynchronously_in_background_tasks() -> None:
    # given
    background_tasks = MagicMock()
    block = TwilioSMSNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        twilio_account_sid="token",
        twilio_auth_token="token",
        message="some",
        sender_number="from",
        receiver_number="to",
        message_parameters={},
        message_parameters_operations={},
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
        cooldown_session_key=f"unique",
        length_limit=128,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    background_tasks.add_task.assert_called_once()


def test_sending_twilio_sms_notification_asynchronously_in_thread_pool_executor() -> (
    None
):
    # given
    thread_pool_executor = MagicMock()
    block = TwilioSMSNotificationBlockV1(
        cache=MemoryCache(),
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    result = block.run(
        twilio_account_sid="token",
        twilio_auth_token="token",
        message="some",
        sender_number="from",
        receiver_number="to",
        message_parameters={},
        message_parameters_operations={},
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
        cooldown_session_key=f"unique",
        length_limit=128,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    thread_pool_executor.submit.assert_called_once()

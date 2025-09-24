import time
from typing import List, Optional, Union
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StringToUpperCase,
)
from inference.core.workflows.core_steps.sinks.email_notification import v1
from inference.core.workflows.core_steps.sinks.email_notification.v1 import (
    BlockManifest,
    EmailNotificationBlockV1,
    format_email_message,
    send_email_using_smtp_server,
)


@pytest.mark.parametrize(
    "receiver_email,cc_receiver_email,bcc_receiver_email",
    [
        ("receiver@gmail.com", None, None),
        ("receiver@gmail.com", "cc_receiver@gmail.com", "bcc_receiver@gmail.com"),
        (["receiver@gmail.com"], ["cc_receiver@gmail.com"], ["bcc_receiver@gmail.com"]),
        ("$inputs.a", "$inputs.b", "$inputs.c"),
    ],
)
def test_manifest_parsing_when_input_is_valid(
    receiver_email: Union[str, List[str]],
    cc_receiver_email: Optional[Union[str, List[str]]],
    bcc_receiver_email: Optional[Union[str, List[str]]],
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/email_notification@v1",
        "name": "email_notifier",
        "email_service_provider": "custom",
        "subject": "Workflow alert",
        "message": "In last aggregation window we found {{ $parameters.people_passed }} people passed through the line",
        "message_parameters": {
            "people_passed": "$steps.data_aggregator.people_passed_values_difference"
        },
        "attachments": {
            "report.csv": "$steps.csv_formatter.csv_content",
        },
        "receiver_email": receiver_email,
        "cc_receiver_email": cc_receiver_email,
        "bcc_receiver_email": bcc_receiver_email,
        "smtp_server": "smtp.gmail.com",
        "sender_email": "$inputs.email",
        "sender_email_password": "$inputs.email_password",
        "fire_and_forget": True,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/email_notification@v1",
        name="email_notifier",
        email_service_provider="custom",
        subject="Workflow alert",
        message="In last aggregation window we found {{ $parameters.people_passed }} people passed through the line",
        message_parameters={
            "people_passed": "$steps.data_aggregator.people_passed_values_difference"
        },
        attachments={
            "report.csv": "$steps.csv_formatter.csv_content",
        },
        receiver_email=receiver_email,
        cc_receiver_email=cc_receiver_email,
        bcc_receiver_email=bcc_receiver_email,
        smtp_server="smtp.gmail.com",
        sender_email="$inputs.email",
        sender_email_password="$inputs.email_password",
        fire_and_forget=True,
    )


def test_format_email_message_when_multiple_occurrences_of_the_same_parameter_exist() -> (
    None
):
    # given
    message = "This is example param: {{{ $parameters.param }}} - and this is aloso param: `{{ $parameters.param }}`"

    # when
    result = format_email_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={},
    )

    # then
    assert result == "This is example param: {some} - and this is aloso param: `some`"


def test_format_email_message_when_multiple_parameters_exist() -> None:
    # given
    message = "This is example param: {{ $parameters.param }} - and this is aloso param: `{{ $parameters.other }}`"

    # when
    result = format_email_message(
        message=message,
        message_parameters={"param": "some", "other": 42},
        message_parameters_operations={},
    )

    # then
    assert result == "This is example param: some - and this is aloso param: `42`"


def test_format_email_message_when_different_combinations_of_whitespaces_exist_in_template_parameter_anchor() -> (
    None
):
    # given
    message = "{{{ $parameters.param }}} - {{$parameters.param }} - {{ $parameters.param}} - {{     $parameters.param     }}"

    # when
    result = format_email_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={},
    )

    # then
    assert result == "{some} - some - some - some"


def test_format_email_message_when_operation_to_apply_on_parameter() -> None:
    # given
    message = "This is example param: {{{ $parameters.param }}} - and this is aloso param: `{{ $parameters.param }}`"

    # when
    result = format_email_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={
            "param": [StringToUpperCase(type="StringToUpperCase")]
        },
    )

    # then
    assert result == "This is example param: {SOME} - and this is aloso param: `SOME`"


@mock.patch.object(v1, "establish_smtp_connection")
def test_send_email_using_smtp_server_when_send_succeeds(
    establish_smtp_connection_mock: MagicMock,
) -> None:
    # given
    smtp_server_mock = MagicMock()
    establish_smtp_connection_mock.return_value.__enter__.return_value = (
        smtp_server_mock
    )

    # when
    result = send_email_using_smtp_server(
        sender_email="sender@gmail.com",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=["bcc_receiver@gmail.com"],
        subject="some-subject",
        message="my-message",
        attachments={},
        smtp_server="smtp.gmail.com",
        smtp_port=465,
        sender_email_password="xxx",
    )

    # then
    assert result == (False, "Notification sent successfully")
    smtp_server_mock.login.assert_called_once_with("sender@gmail.com", "xxx")
    smtp_server_mock.sendmail.assert_called_once()


@mock.patch.object(v1, "establish_smtp_connection")
def test_send_email_using_smtp_server_when_send_fails(
    establish_smtp_connection_mock: MagicMock,
) -> None:
    # given
    establish_smtp_connection_mock.side_effect = Exception()

    # when
    result = send_email_using_smtp_server(
        sender_email="sender@gmail.com",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=["bcc_receiver@gmail.com"],
        subject="some-subject",
        message="my-message",
        attachments={},
        smtp_server="smtp.gmail.com",
        smtp_port=465,
        sender_email_password="xxx",
    )

    # then
    assert result[0] is True


def test_cooldown_in_email_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            subject="some",
            message="other",
            sender_email="sender@gmail.com",
            receiver_email="receiver@gmail.com",
            cc_receiver_email=None,
            bcc_receiver_email=None,
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            smtp_server="server.smtp.com",
            sender_email_password="xxx",
            smtp_port=465,
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


def test_cooldown_recovery_in_email_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            subject="some",
            message="other",
            sender_email="sender@gmail.com",
            receiver_email="receiver@gmail.com",
            cc_receiver_email=None,
            bcc_receiver_email=None,
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            smtp_server="server.smtp.com",
            sender_email_password="xxx",
            smtp_port=465,
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


def test_email_notification_without_cooldown_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            subject="some",
            message="other",
            sender_email="sender@gmail.com",
            receiver_email="receiver@gmail.com",
            cc_receiver_email=None,
            bcc_receiver_email=None,
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            smtp_server="server.smtp.com",
            sender_email_password="xxx",
            smtp_port=465,
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


@mock.patch.object(v1, "send_email_using_smtp_server")
def test_sending_notification_synchronously(
    send_email_using_smtp_server_mock: MagicMock,
) -> None:
    # given
    send_email_using_smtp_server_mock.return_value = False, "success"
    block = EmailNotificationBlockV1(
        background_tasks=None,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        subject="some",
        message="other",
        sender_email="sender@gmail.com",
        receiver_email="receiver@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=["my_bcc@receiver.com"],
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        smtp_server="server.smtp.com",
        sender_email_password="xxx",
        smtp_port=465,
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "success",
    }


def test_sending_notification_asynchronously_with_background_tasks() -> None:
    # given
    background_tasks = MagicMock()
    block = EmailNotificationBlockV1(
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        subject="some",
        message="other",
        sender_email="sender@gmail.com",
        receiver_email="receiver@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=["my_bcc@receiver.com"],
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        smtp_server="server.smtp.com",
        sender_email_password="xxx",
        smtp_port=465,
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


def test_sending_notification_asynchronously_with_thread_pool() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    result = block.run(
        subject="some",
        message="other",
        sender_email="sender@gmail.com",
        receiver_email="receiver@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=["my_bcc@receiver.com"],
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        smtp_server="server.smtp.com",
        sender_email_password="xxx",
        smtp_port=465,
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


@mock.patch.object(v1, "send_email_using_smtp_server")
def test_sink_disabling(send_email_using_smtp_server_mock: MagicMock) -> None:
    # given
    send_email_using_smtp_server_mock.return_value = False, "success"
    block = EmailNotificationBlockV1(
        background_tasks=None,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        subject="some",
        message="other",
        sender_email="sender@gmail.com",
        receiver_email="receiver@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=["my_bcc@receiver.com"],
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        smtp_server="server.smtp.com",
        sender_email_password="xxx",
        smtp_port=465,
        fire_and_forget=True,
        disable_sink=True,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Sink was disabled by parameter `disable_sink`",
    }

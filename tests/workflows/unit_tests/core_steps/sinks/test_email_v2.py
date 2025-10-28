import time
from typing import List, Optional, Union
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StringToUpperCase,
)
from inference.core.workflows.core_steps.sinks.email_notification import v2
from inference.core.workflows.core_steps.sinks.email_notification.v2 import (
    BlockManifest,
    EmailNotificationBlockV2,
    format_email_message,
    send_email_via_roboflow_proxy,
)


@pytest.mark.parametrize(
    "email_provider,receiver_email,cc_receiver_email,bcc_receiver_email",
    [
        ("Roboflow Managed API Key", "receiver@gmail.com", None, None),
        ("Custom SMTP", "receiver@gmail.com", "cc@gmail.com", "bcc@gmail.com"),
        ("Roboflow Managed API Key", ["receiver@gmail.com"], None, None),
        ("Custom SMTP", "$inputs.a", "$inputs.b", "$inputs.c"),
    ],
)
def test_v2_manifest_parsing_when_input_is_valid(
    email_provider: str,
    receiver_email: Union[str, List[str]],
    cc_receiver_email: Optional[Union[str, List[str]]],
    bcc_receiver_email: Optional[Union[str, List[str]]],
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/email_notification@v2",
        "name": "email_notifier",
        "email_provider": email_provider,
        "subject": "Workflow alert",
        "message": "In last aggregation window we found {{ $parameters.people_passed }} people passed through the line",
        "message_parameters": {
            "people_passed": "$steps.data_aggregator.people_passed_values_difference"
        },
        "attachments": {
            "report.csv": "$steps.csv_formatter.csv_content",
        },
        "receiver_email": receiver_email,
        "fire_and_forget": True,
    }
    
    # Add SMTP fields if Custom SMTP
    if email_provider == "Custom SMTP":
        raw_manifest.update({
            "smtp_server": "smtp.gmail.com",
            "sender_email": "$inputs.email",
            "sender_email_password": "$inputs.email_password",
            "cc_receiver_email": cc_receiver_email,
            "bcc_receiver_email": bcc_receiver_email,
        })

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.type == "roboflow_core/email_notification@v2"
    assert result.email_provider == email_provider
    assert result.subject == "Workflow alert"
    assert result.receiver_email == receiver_email


def test_v2_manifest_validates_roboflow_managed_without_smtp_fields() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/email_notification@v2",
        "name": "email_notifier",
        "email_provider": "Roboflow Managed API Key",
        "subject": "Test",
        "message": "Test message",
        "receiver_email": "test@example.com",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.email_provider == "Roboflow Managed API Key"
    assert result.sender_email is None
    assert result.smtp_server is None
    assert result.sender_email_password is None


def test_v2_format_email_message_with_multiple_occurrences() -> None:
    # given
    message = "This is example param: {{{ $parameters.param }}} - and this is also param: `{{ $parameters.param }}`"

    # when
    result = format_email_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={},
    )

    # then
    assert result == "This is example param: {some} - and this is also param: `some`"


def test_v2_format_email_message_with_multiple_parameters() -> None:
    # given
    message = "This is example param: {{ $parameters.param }} - and this is also param: `{{ $parameters.other }}`"

    # when
    result = format_email_message(
        message=message,
        message_parameters={"param": "some", "other": 42},
        message_parameters_operations={},
    )

    # then
    assert result == "This is example param: some - and this is also param: `42`"


def test_v2_format_email_message_with_operation() -> None:
    # given
    message = "This is example param: {{{ $parameters.param }}} - and this is also param: `{{ $parameters.param }}`"

    # when
    result = format_email_message(
        message=message,
        message_parameters={"param": "some"},
        message_parameters_operations={
            "param": [StringToUpperCase(type="StringToUpperCase")]
        },
    )

    # then
    assert result == "This is example param: {SOME} - and this is also param: `SOME`"


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_via_roboflow_proxy_success(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test Subject",
        message="Test message with {{ $parameters.var }}",
        message_parameters={"var": "value"},
        message_parameters_operations={},
        attachments={},
    )

    # then
    assert result == (False, "Notification sent successfully via Roboflow proxy")
    post_to_roboflow_api_mock.assert_called_once()
    call_args = post_to_roboflow_api_mock.call_args
    assert call_args[1]["endpoint"] == "apiproxy/email"
    assert call_args[1]["api_key"] == "test_api_key"
    payload = call_args[1]["payload"]
    assert payload["receiver_email"] == ["receiver@gmail.com"]
    assert payload["subject"] == "Test Subject"
    assert payload["message"] == "Test message with {{ $parameters.var }}"
    assert payload["message_parameters"] == {"var": "value"}


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_via_roboflow_proxy_with_cc_bcc(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=["cc@gmail.com"],
        bcc_receiver_email=["bcc@gmail.com"],
        subject="Test Subject",
        message="Test message",
        message_parameters={},
        message_parameters_operations={},
        attachments={"report.csv": "csv_content"},
    )

    # then
    assert result == (False, "Notification sent successfully via Roboflow proxy")
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    assert payload["cc_receiver_email"] == ["cc@gmail.com"]
    assert payload["bcc_receiver_email"] == ["bcc@gmail.com"]
    assert payload["attachments"] == {"report.csv": "csv_content"}


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_via_roboflow_proxy_failure(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.side_effect = Exception("API Error")

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test Subject",
        message="Test message",
        message_parameters={},
        message_parameters_operations={},
        attachments={},
    )

    # then
    assert result[0] is True
    assert "API Error" in result[1]


def test_v2_roboflow_managed_mode_sends_via_proxy() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
        api_key="test_roboflow_key",
    )

    with mock.patch.object(v2, "send_email_via_roboflow_proxy") as proxy_mock:
        proxy_mock.return_value = (False, "success")
        
        # when
        result = block.run(
            subject="Test",
            message="Test {{ $parameters.count }}",
            receiver_email="receiver@gmail.com",
            email_provider="Roboflow Managed API Key",
            sender_email=None,
            cc_receiver_email=None,
            bcc_receiver_email=None,
            message_parameters={"count": 5},
            message_parameters_operations={},
            attachments={},
            smtp_server=None,
            sender_email_password=None,
            smtp_port=465,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=0,
        )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    thread_pool_executor.submit.assert_called_once()


@mock.patch.object(v2, "send_email_using_smtp_server")
def test_v2_custom_smtp_mode_sends_via_smtp(
    send_email_using_smtp_server_mock: MagicMock,
) -> None:
    # given
    send_email_using_smtp_server_mock.return_value = (False, "success")
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test",
        message="Test {{ $parameters.count }}",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={"count": 5},
        message_parameters_operations={},
        attachments={},
        smtp_server="smtp.gmail.com",
        sender_email_password="password",
        smtp_port=465,
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    thread_pool_executor.submit.assert_called_once()


def test_v2_custom_smtp_validates_required_fields() -> None:
    # given
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when - missing sender_email
    result = block.run(
        subject="Test",
        message="Test message",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email=None,
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        smtp_server="smtp.gmail.com",
        sender_email_password="password",
        smtp_port=465,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result == {
        "error_status": True,
        "throttling_status": False,
        "message": "Custom SMTP requires sender_email, smtp_server, and sender_email_password",
    }


def test_v2_cooldown_functionality() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
        api_key="test_roboflow_key",
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            subject="Test",
            message="Test message",
            receiver_email="receiver@gmail.com",
            email_provider="Roboflow Managed API Key",
            sender_email=None,
            cc_receiver_email=None,
            bcc_receiver_email=None,
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            smtp_server=None,
            sender_email_password=None,
            smtp_port=465,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=100,
        )
        results.append(result)

    # then
    assert results[0]["throttling_status"] is False
    assert results[1]["throttling_status"] is True


def test_v2_cooldown_recovery() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
        api_key="test_roboflow_key",
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            subject="Test",
            message="Test message",
            receiver_email="receiver@gmail.com",
            email_provider="Roboflow Managed API Key",
            sender_email=None,
            cc_receiver_email=None,
            bcc_receiver_email=None,
            message_parameters={},
            message_parameters_operations={},
            attachments={},
            smtp_server=None,
            sender_email_password=None,
            smtp_port=465,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=1,
        )
        results.append(result)
        time.sleep(1.5)

    # then
    assert results[0]["throttling_status"] is False
    assert results[1]["throttling_status"] is False


def test_v2_disable_sink() -> None:
    # given
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test",
        message="Test message",
        receiver_email="receiver@gmail.com",
        email_provider="Roboflow Managed API Key",
        sender_email=None,
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        smtp_server=None,
        sender_email_password=None,
        smtp_port=465,
        fire_and_forget=True,
        disable_sink=True,
        cooldown_seconds=0,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Sink was disabled by parameter `disable_sink`",
    }


@mock.patch.object(v2, "send_email_via_roboflow_proxy")
def test_v2_synchronous_execution_with_roboflow_managed(
    send_email_via_roboflow_proxy_mock: MagicMock,
) -> None:
    # given
    send_email_via_roboflow_proxy_mock.return_value = (False, "success")
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test",
        message="Test message",
        receiver_email="receiver@gmail.com",
        email_provider="Roboflow Managed API Key",
        sender_email=None,
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        smtp_server=None,
        sender_email_password=None,
        smtp_port=465,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "success",
    }
    send_email_via_roboflow_proxy_mock.assert_called_once()


def test_v2_asynchronous_execution_with_background_tasks() -> None:
    # given
    background_tasks = MagicMock()
    block = EmailNotificationBlockV2(
        background_tasks=background_tasks,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test",
        message="Test message",
        receiver_email="receiver@gmail.com",
        email_provider="Roboflow Managed API Key",
        sender_email=None,
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={},
        message_parameters_operations={},
        attachments={},
        smtp_server=None,
        sender_email_password=None,
        smtp_port=465,
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    background_tasks.add_task.assert_called_once()


def test_v2_message_parameters_not_flattened_in_roboflow_mode() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
        api_key="test_roboflow_key",
    )

    with mock.patch.object(v2, "send_email_via_roboflow_proxy") as proxy_mock:
        proxy_mock.return_value = (False, "success")
        
        # when
        result = block.run(
            subject="Test",
            message="Detected {{ $parameters.count }} objects",
            receiver_email="receiver@gmail.com",
            email_provider="Roboflow Managed API Key",
            sender_email=None,
            cc_receiver_email=None,
            bcc_receiver_email=None,
            message_parameters={"count": "$steps.model.predictions"},
            message_parameters_operations={"count": [{"type": "SequenceLength"}]},
            attachments={},
            smtp_server=None,
            sender_email_password=None,
            smtp_port=465,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=0,
        )

    # then
    # Verify that the handler was called with unflattened message
    call_args = thread_pool_executor.submit.call_args[0][0]
    # The partial function should have the raw message template, not flattened
    assert "{{ $parameters.count }}" in call_args.keywords["message"]

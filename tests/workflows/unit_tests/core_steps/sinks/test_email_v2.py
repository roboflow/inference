import time
from typing import List, Optional, Union
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
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
    serialize_image_data,
    serialize_message_parameters,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
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
    # Attachment should be base64 encoded
    import base64
    assert "report.csv" in payload["attachments"]
    decoded_csv = base64.b64decode(payload["attachments"]["report.csv"]).decode('utf-8')
    assert decoded_csv == "csv_content"


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


def test_v2_serialize_image_data_with_base64_image() -> None:
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/4AAQSkZJRgABAQAASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAAA//EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AH//Z",
    )

    # when
    result = serialize_image_data(image_data)

    # then
    assert isinstance(result, str)
    assert result.startswith("/9j/")  # JPEG signature


def test_v2_serialize_image_data_with_numpy_array() -> None:
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    numpy_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_array,
    )

    # when
    result = serialize_image_data(image_data)

    # then
    assert isinstance(result, str)
    assert len(result) > 0
    # Should be valid base64
    import base64
    try:
        base64.b64decode(result)
        valid_base64 = True
    except Exception:
        valid_base64 = False
    assert valid_base64


def test_v2_serialize_image_data_with_non_image_value() -> None:
    # given
    value = "plain string"

    # when
    result = serialize_image_data(value)

    # then
    assert result == "plain string"


def test_v2_serialize_image_data_with_dict() -> None:
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/test",
    )
    value = {
        "image": image_data,
        "text": "some text",
        "number": 42,
    }

    # when
    result = serialize_image_data(value)

    # then
    assert isinstance(result, dict)
    assert result["image"] == "/9j/test"
    assert result["text"] == "some text"
    assert result["number"] == 42


def test_v2_serialize_image_data_with_list() -> None:
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image_data1 = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/first",
    )
    image_data2 = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/second",
    )
    value = [image_data1, "text", image_data2]

    # when
    result = serialize_image_data(value)

    # then
    assert isinstance(result, list)
    assert result[0] == "/9j/first"
    assert result[1] == "text"
    assert result[2] == "/9j/second"


def test_v2_serialize_message_parameters() -> None:
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/image_content",
    )
    message_parameters = {
        "image": image_data,
        "count": 5,
        "text": "detection",
    }

    # when
    result = serialize_message_parameters(message_parameters)

    # then
    assert result["image"] == "/9j/image_content"
    assert result["count"] == 5
    assert result["text"] == "detection"


def test_v2_serialize_message_parameters_with_nested_structures() -> None:
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/nested",
    )
    message_parameters = {
        "data": {
            "image": image_data,
            "metadata": {"count": 10},
        },
        "images": [image_data, image_data],
    }

    # when
    result = serialize_message_parameters(message_parameters)

    # then
    assert result["data"]["image"] == "/9j/nested"
    assert result["data"]["metadata"]["count"] == 10
    assert result["images"][0] == "/9j/nested"
    assert result["images"][1] == "/9j/nested"


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_via_roboflow_proxy_serializes_images(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    parent_metadata = ImageParentMetadata(parent_id="test")
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/4AAQSkZJRgABAQAASABIAAD/test",
    )

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test with Image",
        message="Image: {{ $parameters.image }}",
        message_parameters={"image": image_data},
        message_parameters_operations={},
        attachments={},
    )

    # then
    assert result == (False, "Notification sent successfully via Roboflow proxy")
    post_to_roboflow_api_mock.assert_called_once()
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    # Verify that WorkflowImageData was serialized to base64 string
    assert payload["message_parameters"]["image"] == "/9j/4AAQSkZJRgABAQAASABIAAD/test"
    assert isinstance(payload["message_parameters"]["image"], str)


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_via_roboflow_proxy_with_multiple_images(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    parent_metadata = ImageParentMetadata(parent_id="test")
    image1 = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/first_image",
    )
    image2 = WorkflowImageData(
        parent_metadata=parent_metadata,
        base64_image="/9j/second_image",
    )

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test with Multiple Images",
        message="Images: {{ $parameters.images }}",
        message_parameters={"images": [image1, image2]},
        message_parameters_operations={},
        attachments={},
    )

    # then
    assert result == (False, "Notification sent successfully via Roboflow proxy")
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    assert payload["message_parameters"]["images"] == ["/9j/first_image", "/9j/second_image"]
    assert all(isinstance(img, str) for img in payload["message_parameters"]["images"])


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_with_image_attachment(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    parent_metadata = ImageParentMetadata(parent_id="test")
    numpy_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_array,
    )

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test with Image Attachment",
        message="Please find the detection image attached.",
        message_parameters={},
        message_parameters_operations={},
        attachments={"detection": image_data},
    )

    # then
    assert result == (False, "Notification sent successfully via Roboflow proxy")
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    # Verify attachment was processed and sent
    assert "attachments" in payload
    assert "detection.jpg" in payload["attachments"]
    # Should be base64 encoded
    import base64
    attachment_data = payload["attachments"]["detection.jpg"]
    assert isinstance(attachment_data, str)
    # Verify it's valid base64
    try:
        decoded = base64.b64decode(attachment_data)
        valid_base64 = True
    except Exception:
        valid_base64 = False
    assert valid_base64


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_with_image_attachment_existing_jpg_extension(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    parent_metadata = ImageParentMetadata(parent_id="test")
    numpy_array = np.zeros((50, 50, 3), dtype=np.uint8)
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_array,
    )

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test",
        message="Test",
        message_parameters={},
        message_parameters_operations={},
        attachments={"image.jpg": image_data},
    )

    # then
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    assert "image.jpg" in payload["attachments"]
    # Should not add another .jpg extension
    assert "image.jpg.jpg" not in payload["attachments"]


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_with_mixed_attachments(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    parent_metadata = ImageParentMetadata(parent_id="test")
    numpy_array = np.zeros((50, 50, 3), dtype=np.uint8)
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_array,
    )

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test",
        message="Test",
        message_parameters={},
        message_parameters_operations={},
        attachments={
            "detection_image": image_data,
            "report.csv": "name,count\ndog,5\ncat,3",
        },
    )

    # then
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    assert "detection_image.jpg" in payload["attachments"]
    assert "report.csv" in payload["attachments"]
    # CSV should be base64 encoded
    import base64
    csv_decoded = base64.b64decode(payload["attachments"]["report.csv"]).decode('utf-8')
    assert "name,count" in csv_decoded


@mock.patch.object(v2, "send_email_using_smtp_server")
def test_v2_smtp_mode_with_image_attachment(
    send_email_using_smtp_server_mock: MagicMock,
) -> None:
    # given
    send_email_using_smtp_server_mock.return_value = (False, "success")
    parent_metadata = ImageParentMetadata(parent_id="test")
    numpy_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_array,
    )
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
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={},
        message_parameters_operations={},
        attachments={"detection": image_data},
        smtp_server="smtp.gmail.com",
        sender_email_password="password",
        smtp_port=465,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result["error_status"] is False
    send_email_using_smtp_server_mock.assert_called_once()
    call_kwargs = send_email_using_smtp_server_mock.call_args[1]
    # Verify image was converted to bytes
    assert "detection.jpg" in call_kwargs["attachments"]
    attachment_data = call_kwargs["attachments"]["detection.jpg"]
    assert isinstance(attachment_data, bytes)
    # Should be JPEG signature
    assert attachment_data[:2] == b'\xff\xd8'


@mock.patch.object(v2, "send_email_using_smtp_server")
def test_v2_smtp_mode_with_mixed_attachments(
    send_email_using_smtp_server_mock: MagicMock,
) -> None:
    # given
    send_email_using_smtp_server_mock.return_value = (False, "success")
    parent_metadata = ImageParentMetadata(parent_id="test")
    numpy_array = np.zeros((50, 50, 3), dtype=np.uint8)
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_array,
    )
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
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={},
        message_parameters_operations={},
        attachments={
            "detection.jpg": image_data,
            "report.csv": "name,count\ndog,5",
        },
        smtp_server="smtp.gmail.com",
        sender_email_password="password",
        smtp_port=465,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result["error_status"] is False
    call_kwargs = send_email_using_smtp_server_mock.call_args[1]
    # Both attachments should be present
    assert "detection.jpg" in call_kwargs["attachments"]
    assert "report.csv" in call_kwargs["attachments"]
    # Image should be bytes (JPEG)
    assert isinstance(call_kwargs["attachments"]["detection.jpg"], bytes)
    # CSV should be bytes (UTF-8 encoded string)
    assert isinstance(call_kwargs["attachments"]["report.csv"], bytes)
    csv_content = call_kwargs["attachments"]["report.csv"].decode('utf-8')
    assert "name,count" in csv_content



@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_with_multiple_image_attachments(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    parent_metadata = ImageParentMetadata(parent_id="test")
    
    # Create two different images
    image1 = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    image2 = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.ones((50, 50, 3), dtype=np.uint8) * 255,
    )

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test with Multiple Images",
        message="Multiple images attached",
        message_parameters={},
        message_parameters_operations={},
        attachments={
            "detection1": image1,
            "detection2.jpg": image2,
        },
    )

    # then
    assert result == (False, "Notification sent successfully via Roboflow proxy")
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    assert "detection1.jpg" in payload["attachments"]
    assert "detection2.jpg" in payload["attachments"]
    # Both should be base64 encoded
    import base64
    assert isinstance(payload["attachments"]["detection1.jpg"], str)
    assert isinstance(payload["attachments"]["detection2.jpg"], str)
    # Verify both are valid base64
    for key in ["detection1.jpg", "detection2.jpg"]:
        try:
            base64.b64decode(payload["attachments"][key])
        except Exception:
            pytest.fail(f"Attachment {key} is not valid base64")


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_with_image_attachment_jpeg_extension(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    parent_metadata = ImageParentMetadata(parent_id="test")
    numpy_array = np.zeros((50, 50, 3), dtype=np.uint8)
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_array,
    )

    # when - filename already has .jpeg extension
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test",
        message="Test",
        message_parameters={},
        message_parameters_operations={},
        attachments={"image.jpeg": image_data},
    )

    # then
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    assert "image.jpeg" in payload["attachments"]
    # Should not add another extension
    assert "image.jpeg.jpg" not in payload["attachments"]


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_with_bytes_attachment_via_proxy(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    binary_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00'

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test with Binary",
        message="Binary attachment",
        message_parameters={},
        message_parameters_operations={},
        attachments={"data.bin": binary_data},
    )

    # then
    assert result == (False, "Notification sent successfully via Roboflow proxy")
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    assert "data.bin" in payload["attachments"]
    # Should be base64 encoded
    import base64
    decoded = base64.b64decode(payload["attachments"]["data.bin"])
    assert decoded == binary_data


@mock.patch.object(v2, "send_email_using_smtp_server")
def test_v2_smtp_mode_with_bytes_attachment(
    send_email_using_smtp_server_mock: MagicMock,
) -> None:
    # given
    send_email_using_smtp_server_mock.return_value = (False, "success")
    binary_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00'
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test",
        message="Binary attachment",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={},
        message_parameters_operations={},
        attachments={"data.bin": binary_data},
        smtp_server="smtp.gmail.com",
        sender_email_password="password",
        smtp_port=465,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result["error_status"] is False
    call_kwargs = send_email_using_smtp_server_mock.call_args[1]
    assert "data.bin" in call_kwargs["attachments"]
    assert call_kwargs["attachments"]["data.bin"] == binary_data
    assert isinstance(call_kwargs["attachments"]["data.bin"], bytes)


@mock.patch.object(v2, "send_email_using_smtp_server")
def test_v2_smtp_mode_with_multiple_image_attachments(
    send_email_using_smtp_server_mock: MagicMock,
) -> None:
    # given
    send_email_using_smtp_server_mock.return_value = (False, "success")
    parent_metadata = ImageParentMetadata(parent_id="test")
    
    image1 = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    image2 = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.ones((50, 50, 3), dtype=np.uint8) * 255,
    )
    
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test",
        message="Multiple images",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={},
        message_parameters_operations={},
        attachments={
            "detection1": image1,
            "detection2.jpg": image2,
        },
        smtp_server="smtp.gmail.com",
        sender_email_password="password",
        smtp_port=465,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result["error_status"] is False
    call_kwargs = send_email_using_smtp_server_mock.call_args[1]
    assert "detection1.jpg" in call_kwargs["attachments"]
    assert "detection2.jpg" in call_kwargs["attachments"]
    # Both should be JPEG bytes
    assert call_kwargs["attachments"]["detection1.jpg"][:2] == b'\xff\xd8'
    assert call_kwargs["attachments"]["detection2.jpg"][:2] == b'\xff\xd8'


@mock.patch.object(v2, "post_to_roboflow_api")
def test_v2_send_email_with_all_attachment_types(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    # given
    post_to_roboflow_api_mock.return_value = {"status": "success"}
    parent_metadata = ImageParentMetadata(parent_id="test")
    
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    binary_data = b'\x00\x01\x02\x03'
    text_data = "This is CSV content"

    # when
    result = send_email_via_roboflow_proxy(
        roboflow_api_key="test_api_key",
        receiver_email=["receiver@gmail.com"],
        cc_receiver_email=None,
        bcc_receiver_email=None,
        subject="Test All Types",
        message="All attachment types",
        message_parameters={},
        message_parameters_operations={},
        attachments={
            "image": image_data,
            "binary.bin": binary_data,
            "text.csv": text_data,
        },
    )

    # then
    assert result == (False, "Notification sent successfully via Roboflow proxy")
    payload = post_to_roboflow_api_mock.call_args[1]["payload"]
    assert "image.jpg" in payload["attachments"]
    assert "binary.bin" in payload["attachments"]
    assert "text.csv" in payload["attachments"]
    
    # All should be base64 encoded strings
    import base64
    assert isinstance(payload["attachments"]["image.jpg"], str)
    assert isinstance(payload["attachments"]["binary.bin"], str)
    assert isinstance(payload["attachments"]["text.csv"], str)
    
    # Verify content can be decoded
    assert base64.b64decode(payload["attachments"]["binary.bin"]) == binary_data
    assert base64.b64decode(payload["attachments"]["text.csv"]).decode('utf-8') == text_data

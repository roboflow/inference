import time
from typing import List, Optional, Union
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StringToUpperCase,
)
from inference.core.workflows.core_steps.sinks.twilio.sms import v2
from inference.core.workflows.core_steps.sinks.twilio.sms.v2 import (
    BlockManifest,
    TwilioSMSNotificationBlockV2,
    format_message,
    send_sms_via_roboflow_proxy,
    serialize_media_for_api,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "sms_provider,receiver_number,media_url",
    [
        ("Roboflow Managed API Key", "+15551234567", None),
        ("Custom Twilio", "+15551234567", "https://example.com/image.jpg"),
        ("Roboflow Managed API Key", "$inputs.receiver", None),
        ("Custom Twilio", "$inputs.receiver", "$steps.visualization.image"),
    ],
)
def test_v2_manifest_parsing_when_input_is_valid(
    sms_provider: str,
    receiver_number: str,
    media_url: Optional[str],
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/twilio_sms_notification@v2",
        "name": "sms_notifier",
        "sms_provider": sms_provider,
        "receiver_number": receiver_number,
        "message": "Alert! Detected {{ $parameters.num_detections }} objects",
        "message_parameters": {"num_detections": "$steps.model.predictions"},
        "fire_and_forget": True,
    }

    if media_url:
        raw_manifest["media_url"] = media_url

    # Add Twilio fields if Custom Twilio
    if sms_provider == "Custom Twilio":
        raw_manifest.update(
            {
                "twilio_account_sid": "$inputs.twilio_account_sid",
                "twilio_auth_token": "$inputs.twilio_auth_token",
                "sender_number": "$inputs.sender_number",
            }
        )

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.type == "roboflow_core/twilio_sms_notification@v2"
    assert result.sms_provider == sms_provider
    assert result.receiver_number == receiver_number


def test_v2_manifest_validates_roboflow_managed_without_twilio_fields() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/twilio_sms_notification@v2",
        "name": "sms_notifier",
        "sms_provider": "Roboflow Managed API Key",
        "receiver_number": "+15551234567",
        "message": "Test message",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.sms_provider == "Roboflow Managed API Key"
    assert result.twilio_account_sid is None
    assert result.twilio_auth_token is None
    assert result.sender_number is None


def test_format_message_simple_parameters() -> None:
    # given
    message = "Detected {{ $parameters.count }} objects of type {{ $parameters.type }}"
    message_parameters = {
        "count": 5,
        "type": "person",
    }
    message_parameters_operations = {}

    # when
    result, needs_mms = format_message(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    assert result == "Detected 5 objects of type person"
    assert needs_mms is False  # Message is under 160 chars


def test_format_message_with_operations() -> None:
    # given
    message = "Detected {{ $parameters.classes }}"
    message_parameters = {
        "classes": "PERSON",
    }
    message_parameters_operations = {
        "classes": [StringToUpperCase(type="StringToUpperCase")]
    }

    # when
    result, needs_mms = format_message(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    assert "PERSON" in result
    assert needs_mms is False


def test_format_message_long_message_needs_mms() -> None:
    # given - message over 160 chars
    message = "A" * 200
    message_parameters = {}
    message_parameters_operations = {}

    # when
    result, needs_mms = format_message(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    assert len(result) == 200  # No truncation yet (under MMS limit)
    assert needs_mms is True  # Message exceeds SMS limit


def test_format_message_truncates_at_mms_limit() -> None:
    # given - message over 1600 chars (MMS limit)
    message = "A" * 2000
    message_parameters = {}
    message_parameters_operations = {}

    # when
    result, needs_mms = format_message(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    assert len(result) == 1600  # Truncated to MMS limit
    assert result.endswith("[...]")
    assert needs_mms is True


def test_serialize_media_for_api_with_string_url() -> None:
    # given
    media_url = "https://example.com/image.jpg"

    # when
    media_urls, media_base64 = serialize_media_for_api(media_url)

    # then
    assert media_urls == ["https://example.com/image.jpg"]
    assert media_base64 is None


def test_serialize_media_for_api_with_list() -> None:
    # given
    media_urls = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
    ]

    # when
    result_urls, result_base64 = serialize_media_for_api(media_urls)

    # then
    assert result_urls == media_urls
    assert result_base64 is None


def test_serialize_media_for_api_with_workflow_image() -> None:
    # given
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    workflow_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=image,
    )

    # when
    media_urls, media_base64 = serialize_media_for_api(workflow_image)

    # then
    assert media_urls is None
    assert media_base64 is not None
    assert len(media_base64) == 1
    assert "base64" in media_base64[0]
    assert "mimeType" in media_base64[0]
    assert media_base64[0]["mimeType"] == "image/jpeg"


@mock.patch.object(v2, "post_to_roboflow_api")
def test_send_sms_via_roboflow_proxy_success(mock_post: MagicMock) -> None:
    # given
    mock_post.return_value = {"success": True, "message_sid": "SM123"}

    # when
    error, message = send_sms_via_roboflow_proxy(
        roboflow_api_key="test_key",
        receiver_number="+15551234567",
        message="Test {{ $parameters.count }}",
        message_parameters={"count": 5},
        message_parameters_operations={},
        media_url=None,
    )

    # then
    assert not error
    assert "successfully" in message.lower()
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["endpoint"] == "apiproxy/twilio"
    assert call_args[1]["api_key"] == "test_key"
    payload = call_args[1]["payload"]
    assert payload["receiver_number"] == "+15551234567"
    assert "Test" in payload["message"]


@mock.patch.object(v2, "post_to_roboflow_api")
def test_send_sms_via_roboflow_proxy_with_media(mock_post: MagicMock) -> None:
    # given
    mock_post.return_value = {"success": True, "message_sid": "SM123"}

    # when
    error, message = send_sms_via_roboflow_proxy(
        roboflow_api_key="test_key",
        receiver_number="+15551234567",
        message="Check this out",
        message_parameters={},
        message_parameters_operations={},
        media_url="https://example.com/image.jpg",
    )

    # then
    assert not error
    payload = mock_post.call_args[1]["payload"]
    assert "media_urls" in payload
    assert payload["media_urls"] == ["https://example.com/image.jpg"]


@mock.patch.object(v2, "post_to_roboflow_api")
def test_send_sms_via_roboflow_proxy_rate_limit_error(mock_post: MagicMock) -> None:
    # given
    from inference.core.exceptions import RoboflowAPIUnsuccessfulRequestError

    def raise_rate_limit(endpoint, api_key, payload, http_errors_handlers):
        handler = http_errors_handlers[429]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": "Rate limit exceeded",
            "details": "Rate limit exceeded. Please wait before sending more messages.",
        }
        mock_error = Exception()
        mock_error.response = mock_response
        handler(mock_error)

    mock_post.side_effect = raise_rate_limit

    # when
    error, message = send_sms_via_roboflow_proxy(
        roboflow_api_key="test_key",
        receiver_number="+15551234567",
        message="Test",
        message_parameters={},
        message_parameters_operations={},
        media_url=None,
    )

    # then
    assert error
    assert "rate limit" in message.lower()


@mock.patch.object(v2, "post_to_roboflow_api")
def test_send_sms_via_roboflow_proxy_credits_exceeded(mock_post: MagicMock) -> None:
    # given
    from inference.core.exceptions import RoboflowAPIUnsuccessfulRequestError

    def raise_credits_error(endpoint, api_key, payload, http_errors_handlers):
        handler = http_errors_handlers.get(429, http_errors_handlers.get(403))
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": "Credits exceeded",
            "details": "Workspace credits exceeded.",
        }
        mock_error = Exception()
        mock_error.response = mock_response
        handler(mock_error)

    mock_post.side_effect = raise_credits_error

    # when
    error, message = send_sms_via_roboflow_proxy(
        roboflow_api_key="test_key",
        receiver_number="+15551234567",
        message="Test",
        message_parameters={},
        message_parameters_operations={},
        media_url=None,
    )

    # then
    assert error
    assert "credits" in message.lower()


def test_twilio_block_v2_roboflow_managed_success() -> None:
    # given
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_key",
    )

    with mock.patch.object(v2, "post_to_roboflow_api") as mock_post:
        mock_post.return_value = {"success": True}

        # when
        result = block.run(
            sms_provider="Roboflow Managed API Key",
            receiver_number="+15551234567",
            message="Test message",
            message_parameters={},
            message_parameters_operations={},
            media_url=None,
            twilio_account_sid=None,
            twilio_auth_token=None,
            sender_number=None,
            fire_and_forget=False,
            disable_sink=False,
            cooldown_seconds=5,
        )

        # then
        assert result["error_status"] is False
        assert result["throttling_status"] is False
        assert "successfully" in result["message"].lower()


def test_twilio_block_v2_custom_twilio_success() -> None:
    # given
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_key",
    )

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_client.messages = mock_messages

    with mock.patch(
        "inference.core.workflows.core_steps.sinks.twilio.sms.v2.Client"
    ) as mock_client_class:
        mock_client_class.return_value = mock_client

        # when
        result = block.run(
            sms_provider="Custom Twilio",
            receiver_number="+15551234567",
            message="Test message",
            message_parameters={},
            message_parameters_operations={},
            media_url=None,
            twilio_account_sid="AC123",
            twilio_auth_token="auth_token",
            sender_number="+15559876543",
            fire_and_forget=False,
            disable_sink=False,
            cooldown_seconds=5,
        )

        # then
        assert result["error_status"] is False
        mock_messages.create.assert_called_once()


def test_twilio_block_v2_custom_twilio_missing_credentials() -> None:
    # given
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_key",
    )

    # when
    result = block.run(
        sms_provider="Custom Twilio",
        receiver_number="+15551234567",
        message="Test message",
        message_parameters={},
        message_parameters_operations={},
        media_url=None,
        twilio_account_sid=None,  # Missing!
        twilio_auth_token="auth_token",
        sender_number="+15559876543",
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=5,
    )

    # then
    assert result["error_status"] is True
    assert "requires" in result["message"].lower()


def test_twilio_block_v2_disable_sink() -> None:
    # given
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_key",
    )

    # when
    result = block.run(
        sms_provider="Roboflow Managed API Key",
        receiver_number="+15551234567",
        message="Test message",
        message_parameters={},
        message_parameters_operations={},
        media_url=None,
        twilio_account_sid=None,
        twilio_auth_token=None,
        sender_number=None,
        fire_and_forget=False,
        disable_sink=True,  # Sink disabled
        cooldown_seconds=5,
    )

    # then
    assert result["error_status"] is False
    assert result["throttling_status"] is False


def test_twilio_block_v2_custom_twilio_with_mms_list() -> None:
    # given
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_key",
    )

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_client.messages = mock_messages

    with mock.patch(
        "inference.core.workflows.core_steps.sinks.twilio.sms.v2.Client"
    ) as mock_client_class:
        mock_client_class.return_value = mock_client

        # when
        result = block.run(
            sms_provider="Custom Twilio",
            receiver_number="+15551234567",
            message="Check out these images",
            message_parameters={},
            message_parameters_operations={},
            media_url=[
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
            ],
            twilio_account_sid="AC123",
            twilio_auth_token="auth_token",
            sender_number="+15559876543",
            fire_and_forget=False,
            disable_sink=False,
            cooldown_seconds=5,
        )

        # then
        assert result["error_status"] is False
        mock_messages.create.assert_called_once()
        call_kwargs = mock_messages.create.call_args[1]
        assert "media_url" in call_kwargs
        assert len(call_kwargs["media_url"]) == 2


def test_twilio_block_v2_custom_twilio_with_workflow_image() -> None:
    # given
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_key",
    )

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    workflow_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=image,
    )

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_client.messages = mock_messages

    with mock.patch(
        "inference.core.workflows.core_steps.sinks.twilio.sms.v2.Client"
    ) as mock_client_class, mock.patch(
        "inference.core.workflows.core_steps.sinks.twilio.sms.v2._upload_image_to_ephemeral_host"
    ) as mock_upload:

        mock_client_class.return_value = mock_client
        mock_upload.return_value = "https://example.org/dl/12345/image.jpg"

        # when
        result = block.run(
            sms_provider="Custom Twilio",
            receiver_number="+15551234567",
            message="Check out this image",
            message_parameters={},
            message_parameters_operations={},
            media_url=workflow_image,
            twilio_account_sid="AC123",
            twilio_auth_token="auth_token",
            sender_number="+15559876543",
            fire_and_forget=False,
            disable_sink=False,
            cooldown_seconds=5,
        )

        # then
        assert result["error_status"] is False
        mock_upload.assert_called_once()
        mock_messages.create.assert_called_once()
        call_kwargs = mock_messages.create.call_args[1]
        assert "media_url" in call_kwargs
        assert call_kwargs["media_url"] == ["https://example.org/dl/12345/image.jpg"]


def test_twilio_block_v2_cooldown_behavior() -> None:
    # given
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_key",
    )

    with mock.patch.object(v2, "post_to_roboflow_api") as mock_post:
        mock_post.return_value = {"success": True}

        # First call - should succeed
        result1 = block.run(
            sms_provider="Roboflow Managed API Key",
            receiver_number="+15551234567",
            message="Test message 1",
            message_parameters={},
            message_parameters_operations={},
            media_url=None,
            twilio_account_sid=None,
            twilio_auth_token=None,
            sender_number=None,
            fire_and_forget=False,
            disable_sink=False,
            cooldown_seconds=5,
        )

        # then
        assert result1["error_status"] is False
        assert result1["throttling_status"] is False

        # Second call immediately - should be throttled
        result2 = block.run(
            sms_provider="Roboflow Managed API Key",
            receiver_number="+15551234567",
            message="Test message 2",
            message_parameters={},
            message_parameters_operations={},
            media_url=None,
            twilio_account_sid=None,
            twilio_auth_token=None,
            sender_number=None,
            fire_and_forget=False,
            disable_sink=False,
            cooldown_seconds=5,
        )

        # then
        assert result2["error_status"] is False
        assert result2["throttling_status"] is True
        assert "cooldown" in result2["message"].lower()


def test_twilio_block_v2_cooldown_expires() -> None:
    # given
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_key",
    )

    with mock.patch.object(v2, "post_to_roboflow_api") as mock_post:
        mock_post.return_value = {"success": True}

        # First call
        result1 = block.run(
            sms_provider="Roboflow Managed API Key",
            receiver_number="+15551234567",
            message="Test message 1",
            message_parameters={},
            message_parameters_operations={},
            media_url=None,
            twilio_account_sid=None,
            twilio_auth_token=None,
            sender_number=None,
            fire_and_forget=False,
            disable_sink=False,
            cooldown_seconds=1,  # 1 second cooldown
        )

        assert result1["throttling_status"] is False

        # Wait for cooldown to expire
        time.sleep(1.1)

        # Second call after cooldown - should succeed
        result2 = block.run(
            sms_provider="Roboflow Managed API Key",
            receiver_number="+15551234567",
            message="Test message 2",
            message_parameters={},
            message_parameters_operations={},
            media_url=None,
            twilio_account_sid=None,
            twilio_auth_token=None,
            sender_number=None,
            fire_and_forget=False,
            disable_sink=False,
            cooldown_seconds=1,
        )

        # then
        assert result2["error_status"] is False
        assert result2["throttling_status"] is False


def test_twilio_block_v2_fire_and_forget_with_thread_pool() -> None:
    # given
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=2)
    block = TwilioSMSNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=executor,
        api_key="test_key",
    )

    with mock.patch.object(v2, "post_to_roboflow_api") as mock_post:
        mock_post.return_value = {"success": True}

        # when
        result = block.run(
            sms_provider="Roboflow Managed API Key",
            receiver_number="+15551234567",
            message="Test message",
            message_parameters={},
            message_parameters_operations={},
            media_url=None,
            twilio_account_sid=None,
            twilio_auth_token=None,
            sender_number=None,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=5,
        )

        # then - should return immediately without error
        assert result["error_status"] is False
        assert result["throttling_status"] is False
        assert "background" in result["message"].lower()

    executor.shutdown(wait=True)


def test_twilio_block_v2_fire_and_forget_with_background_tasks() -> None:
    # given
    from fastapi import BackgroundTasks

    background_tasks = BackgroundTasks()
    block = TwilioSMSNotificationBlockV2(
        background_tasks=background_tasks,
        thread_pool_executor=None,
        api_key="test_key",
    )

    with mock.patch.object(v2, "post_to_roboflow_api") as mock_post:
        mock_post.return_value = {"success": True}

        # when
        result = block.run(
            sms_provider="Roboflow Managed API Key",
            receiver_number="+15551234567",
            message="Test message",
            message_parameters={},
            message_parameters_operations={},
            media_url=None,
            twilio_account_sid=None,
            twilio_auth_token=None,
            sender_number=None,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=5,
        )

        # then - should return immediately without error
        assert result["error_status"] is False
        assert result["throttling_status"] is False
        assert "background" in result["message"].lower()

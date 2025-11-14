"""Tests for inline image support in email notification v2."""
import numpy as np
import pytest
from unittest import mock
from unittest.mock import MagicMock

from inference.core.workflows.core_steps.sinks.email_notification import v2
from inference.core.workflows.core_steps.sinks.email_notification.v2 import (
    EmailNotificationBlockV2,
    format_email_message_html_with_images,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_format_email_message_html_with_single_inline_image() -> None:
    """Test HTML formatting with a single inline image."""
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    message = "Check this image: {{ $parameters.detection }}"
    message_parameters = {"detection": image}
    message_parameters_operations = {}

    # when
    html_message, inline_images = format_email_message_html_with_images(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    assert '<img src="cid:image_detection"' in html_message
    assert 'alt="detection"' in html_message
    assert 'style="max-width: 600px; height: auto;"' in html_message
    assert "image_detection" in inline_images
    assert isinstance(inline_images["image_detection"], bytes)
    # Verify it's JPEG format (starts with JPEG magic bytes)
    assert inline_images["image_detection"][:2] == b'\xff\xd8'


def test_format_email_message_html_with_multiple_inline_images() -> None:
    """Test HTML formatting with multiple inline images."""
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image1 = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    image2 = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.ones((60, 60, 3), dtype=np.uint8) * 128,
    )
    message = "First: {{ $parameters.img1 }}\nSecond: {{ $parameters.img2 }}"
    message_parameters = {"img1": image1, "img2": image2}
    message_parameters_operations = {}

    # when
    html_message, inline_images = format_email_message_html_with_images(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    assert '<img src="cid:image_img1"' in html_message
    assert '<img src="cid:image_img2"' in html_message
    assert "image_img1" in inline_images
    assert "image_img2" in inline_images
    assert len(inline_images) == 2
    # Verify both are JPEG format
    assert inline_images["image_img1"][:2] == b'\xff\xd8'
    assert inline_images["image_img2"][:2] == b'\xff\xd8'


def test_format_email_message_html_with_mixed_text_and_images() -> None:
    """Test HTML formatting with both text and image parameters."""
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    message = "Count: {{ $parameters.count }}\nImage: {{ $parameters.photo }}"
    message_parameters = {"count": 42, "photo": image}
    message_parameters_operations = {}

    # when
    html_message, inline_images = format_email_message_html_with_images(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    assert "Count: 42" in html_message
    assert '<img src="cid:image_photo"' in html_message
    assert "image_photo" in inline_images
    assert len(inline_images) == 1


def test_format_email_message_html_escapes_special_characters() -> None:
    """Test that HTML special characters in text parameters are escaped."""
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    message = "XSS: {{ $parameters.user_input }}\nImage: {{ $parameters.img }}"
    message_parameters = {
        "user_input": '<script>alert("xss")</script>',
        "img": image,
    }
    message_parameters_operations = {}

    # when
    html_message, inline_images = format_email_message_html_with_images(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    # HTML entities should be escaped
    assert "&lt;script&gt;" in html_message
    assert "&quot;xss&quot;" in html_message
    assert '<script>alert("xss")</script>' not in html_message
    # Image should still work
    assert '<img src="cid:image_img"' in html_message


def test_format_email_message_html_converts_newlines() -> None:
    """Test that newlines are converted to <br> tags."""
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    message = "Line 1\nLine 2\nImage: {{ $parameters.img }}"
    message_parameters = {"img": image}
    message_parameters_operations = {}

    # when
    html_message, inline_images = format_email_message_html_with_images(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    assert "<br>" in html_message
    # Should have 2 <br> tags (for 2 newlines)
    assert html_message.count("<br>") == 2


def test_format_email_message_html_with_same_image_multiple_times() -> None:
    """Test that the same parameter appearing multiple times reuses the same CID."""
    # given
    parent_metadata = ImageParentMetadata(parent_id="test")
    image = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    message = "First: {{ $parameters.img }}\nSecond: {{ $parameters.img }}"
    message_parameters = {"img": image}
    message_parameters_operations = {}

    # when
    html_message, inline_images = format_email_message_html_with_images(
        message=message,
        message_parameters=message_parameters,
        message_parameters_operations=message_parameters_operations,
    )

    # then
    # Should only have one image in attachments
    assert len(inline_images) == 1
    assert "image_img" in inline_images
    # Both occurrences should reference the same CID
    assert html_message.count('src="cid:image_img"') == 2


@mock.patch.object(v2, "send_email_using_smtp_server_v2")
def test_v2_smtp_mode_with_inline_image(
    send_email_using_smtp_server_v2_mock: MagicMock,
) -> None:
    """Test SMTP mode with inline image in message_parameters."""
    # given
    send_email_using_smtp_server_v2_mock.return_value = (False, "success")
    parent_metadata = ImageParentMetadata(parent_id="test")
    
    image = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test Inline Image",
        message="Here's the detection: {{ $parameters.detection }}",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={"detection": image},
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
    assert result["error_status"] is False
    call_kwargs = send_email_using_smtp_server_v2_mock.call_args[1]
    
    # Should be HTML
    assert call_kwargs["is_html"] is True
    
    # Message should contain HTML img tag
    assert '<img src="cid:image_detection"' in call_kwargs["message"]
    
    # Should have inline_images dict
    assert call_kwargs["inline_images"] is not None
    assert "image_detection" in call_kwargs["inline_images"]
    assert isinstance(call_kwargs["inline_images"]["image_detection"], bytes)


@mock.patch.object(v2, "send_email_using_smtp_server_v2")
def test_v2_smtp_mode_with_inline_and_attachment_images(
    send_email_using_smtp_server_v2_mock: MagicMock,
) -> None:
    """Test SMTP mode with same image as both inline and attachment."""
    # given
    send_email_using_smtp_server_v2_mock.return_value = (False, "success")
    parent_metadata = ImageParentMetadata(parent_id="test")
    
    image = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test",
        message="Preview: {{ $parameters.preview }}",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={"preview": image},
        message_parameters_operations={},
        attachments={"full_resolution.jpg": image},
        smtp_server="smtp.gmail.com",
        sender_email_password="password",
        smtp_port=465,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
    )

    # then
    assert result["error_status"] is False
    call_kwargs = send_email_using_smtp_server_v2_mock.call_args[1]
    
    # Should be HTML with inline image
    assert call_kwargs["is_html"] is True
    assert "image_preview" in call_kwargs["inline_images"]
    
    # Should also have regular attachment
    assert "full_resolution.jpg" in call_kwargs["attachments"]
    
    # Both should be JPEG bytes
    assert call_kwargs["inline_images"]["image_preview"][:2] == b'\xff\xd8'
    assert call_kwargs["attachments"]["full_resolution.jpg"][:2] == b'\xff\xd8'


@mock.patch.object(v2, "send_email_using_smtp_server_v2")
def test_v2_smtp_html_support_without_images(
    send_email_using_smtp_server_v2_mock: MagicMock,
) -> None:
    """Test that SMTP pathway uses HTML even without images (feature parity with Resend)."""
    # given
    send_email_using_smtp_server_v2_mock.return_value = (False, "success")
    
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Test",
        message="Count: {{ $parameters.count }}",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={"count": 42},
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
    assert result["error_status"] is False
    call_kwargs = send_email_using_smtp_server_v2_mock.call_args[1]
    
    # Should be HTML for feature parity with Resend pathway
    assert call_kwargs["is_html"] is True
    
    # Message should be plain text wrapped in HTML formatting
    assert "Count: 42" in call_kwargs["message"]
    
    # inline_images should be empty dict (no images)
    assert call_kwargs["inline_images"] == {}


@mock.patch.object(v2, "send_email_using_smtp_server_v2")
def test_v2_smtp_mode_with_multiple_inline_images(
    send_email_using_smtp_server_v2_mock: MagicMock,
) -> None:
    """Test SMTP mode with multiple inline images."""
    # given
    send_email_using_smtp_server_v2_mock.return_value = (False, "success")
    parent_metadata = ImageParentMetadata(parent_id="test")
    
    image1 = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    image2 = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=np.ones((60, 60, 3), dtype=np.uint8) * 255,
    )
    
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="Multiple Images",
        message="First: {{ $parameters.img1 }}\nSecond: {{ $parameters.img2 }}",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={"img1": image1, "img2": image2},
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
    assert result["error_status"] is False
    call_kwargs = send_email_using_smtp_server_v2_mock.call_args[1]
    
    # Should have two inline images
    assert len(call_kwargs["inline_images"]) == 2
    assert "image_img1" in call_kwargs["inline_images"]
    assert "image_img2" in call_kwargs["inline_images"]
    
    # Message should contain both img tags
    assert '<img src="cid:image_img1"' in call_kwargs["message"]
    assert '<img src="cid:image_img2"' in call_kwargs["message"]


@mock.patch.object(v2, "send_email_using_smtp_server_v2")
def test_v2_smtp_mode_preserves_html_formatting(
    send_email_using_smtp_server_v2_mock: MagicMock,
) -> None:
    """Test that HTML formatting like <b>bold</b> is preserved in SMTP emails."""
    # given
    send_email_using_smtp_server_v2_mock.return_value = (False, "success")
    
    block = EmailNotificationBlockV2(
        background_tasks=None,
        thread_pool_executor=None,
        api_key="test_roboflow_key",
    )

    # when
    result = block.run(
        subject="HTML Formatting Test",
        message="With <b>bold</b> and {{ $parameters.value }}.",
        receiver_email="receiver@gmail.com",
        email_provider="Custom SMTP",
        sender_email="sender@gmail.com",
        cc_receiver_email=None,
        bcc_receiver_email=None,
        message_parameters={"value": "attachment"},
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
    assert result["error_status"] is False
    call_kwargs = send_email_using_smtp_server_v2_mock.call_args[1]
    
    # Should be HTML
    assert call_kwargs["is_html"] is True
    
    # HTML tags should be preserved (not escaped)
    assert "<b>bold</b>" in call_kwargs["message"]
    assert "With <b>bold</b> and attachment." in call_kwargs["message"]
